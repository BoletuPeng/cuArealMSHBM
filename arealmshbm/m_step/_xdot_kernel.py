"""_xdot_kernel.py

The one ``x_dot_sl_bits`` CUDA kernel text, shared by the step-3
``gpu`` M-step (:mod:`arealmshbm.m_step.m_step_gpu`) and the
step-2 ``gpu`` session (:mod:`arealmshbm.step2_em_iter_master._kernels_gpu`).
Each module concatenates it into its own ``RawModule`` source and
substitutes ``XDOT_MAX_NB_VALUE`` / ``XDOT_BS_VALUE`` with its own launch
geometry (step 3: 128 threads x 2 bytes; step 2: 256 x 1). The text is
flag-insensitive: every fp32 multiply-add is an explicit ``fmaf`` and the
only other fp32 ops are lone multiplies, so ``-fmad`` does not change it.

The text must stay ASCII: cupy writes a ``RawModule`` source to a ``.cu``
file in the locale codec before NVRTC sees it.
``m_step/tests/test_m_step_gpu.py`` pins that here, and holds the
shared-memory-staged form as the bit-exact oracle, instantiated at both
consumers' geometries.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

XDOT_SL_BITS_SRC = r"""
// ---------------------------------------------------------------------
// X_dot_sl[t, l, d] = sum_{n in members(l)} sl[n, l] * X[n, t, d]
//                   = sum_{n : bit(t,n,d)} w_n  -  sum_n w_n * row_mean[t, n]
//     with w_n = sl[n, l] * row_inv[t, n], over a bit-packed BOLD.
//
// One block per (t, l); threads own BYTES of the packed row (XDOT_BS
// threads x XDOT_MAX_NB bytes, so D_bytes <= XDOT_BS * XDOT_MAX_NB).
// Member BYTES are read straight from global memory with ``__ldg``: every
// thread of the block walks the SAME member row at the same time, so the
// per-member read is already fully coalesced; staging the bytes in
// shared memory would only add a serial per-member copy loop and
// ``chunk * D_bytes`` bytes of shared memory. Member weights are staged
// per chunk next to the row ids the ``__ldg`` loop indexes (``chunk * 8``
// bytes of dynamic shared memory; two __syncthreads per chunk). The
// 4-deep member prefetch issues four loads before any dependent ``fmaf``,
// which hides the 1-byte load latency.
//
// Op order: each ``(d)`` accumulator folds the members of a chunk in
// ascending order with ``fmaf(w, bit, a)`` (bit is exactly 0.0f or 1.0f,
// so this equals the predicated ``a += w``), and the fp32 partial is
// folded into the fp64 running sum every ``chunk`` members; none of this
// depends on the launch geometry. The B term is an fp64 sum whose thread
// partition and ``block_reduce_f64`` tree span ``blockDim.x`` lanes, so it
// depends on XDOT_BS. Hence the result is bit-exact across XDOT_MAX_NB,
// and against the shared-memory-staged oracle in
// m_step/tests/test_m_step_gpu.py, at the SAME
// XDOT_BS and ``chunk``; changing XDOT_BS changes only B, which leaves
// the fp32 output alone unless A_d - B cancels -- where it does, B's
// last-bit fp64 difference is amplified by |B| / |out| and moves the
// output by many ulps (every cell of the test's cancellation fixture), so
// the step-3 and step-2 instances are not bit-comparable to each other.
// ``XDOT_BS``
// must be a power of two (``block_reduce_f64``'s fixed tree), and the
// enclosing module must define ``block_reduce_f64``.
//
// Launch contract, written out at each call site (the step-3 wrapper,
// step 2's ``run_iter``, the test oracle): grid ``(T * L,)``,
// ``blockDim.x == XDOT_BS`` exactly (a smaller block never stores the
// ``out`` cells of the bytes it does not own -- callers pass ``cp.empty``,
// so that is silent garbage -- and a larger one overruns ``sh_red``),
// dynamic shared memory ``chunk * 8`` bytes (``chunk`` fp32 weights, then
// ``chunk`` int32 row ids), ``D_bytes <= XDOT_BS * XDOT_MAX_NB`` (guarded
// by the step-3 wrapper and by step 2's ``check_dims``). Any ``chunk``
// whose ``chunk * 8`` bytes fit next to ``sh_red`` in the per-block
// shared-memory limit runs, but it sets the fp32->fp64 fold points and so
// the output bits; both callers pass 64.
// ---------------------------------------------------------------------
#define XDOT_MAX_NB XDOT_MAX_NB_VALUE
#define XDOT_BS     XDOT_BS_VALUE

extern "C" __global__
void x_dot_sl_bits(const unsigned char* __restrict__ packed,  // (T, N, Db)
                   const float* __restrict__ row_mean,        // (T, N)
                   const float* __restrict__ row_inv,         // (T, N)
                   const int* __restrict__ col_ptr,           // (L+1,)
                   const int* __restrict__ csc_row,           // (P,)
                   const int* __restrict__ csc_pidx,          // (P,)
                   const float* __restrict__ s_lambda_P,      // (P,)
                   float* __restrict__ out,                   // (T, L, D)
                   int N, int L, int D, int D_bytes, int chunk)
{
    extern __shared__ float sh_w[];          // (chunk,) w, then (chunk,) row id
    int* sh_n = (int*)(sh_w + chunk);
    __shared__ double sh_red[XDOT_BS];
    __shared__ double sh_B;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int blk = blockIdx.x;
    const int t = blk / L;
    const int l = blk - t * L;

    const int start = col_ptr[l];
    const int end = col_ptr[l + 1];

    const unsigned char* base_t =
        packed + (size_t)t * (size_t)N * (size_t)D_bytes;
    const float* rm_t = row_mean + (size_t)t * (size_t)N;
    const float* ri_t = row_inv + (size_t)t * (size_t)N;
    float* out_tl = out + ((size_t)t * (size_t)L + (size_t)l) * (size_t)D;

    double bl = 0.0;
    for (int i = start + tid; i < end; i += bs) {
        const int n = csc_row[i];
        const float w = s_lambda_P[csc_pidx[i]] * ri_t[n];
        bl += (double)(w * rm_t[n]);
    }
    bl = block_reduce_f64(bl, sh_red);
    if (tid == 0) sh_B = bl;
    __syncthreads();
    const double B = sh_B;

    float a32[XDOT_MAX_NB * 8];
    double acc[XDOT_MAX_NB * 8];
    #pragma unroll
    for (int k = 0; k < XDOT_MAX_NB * 8; ++k) acc[k] = 0.0;

    for (int cs = start; cs < end; cs += chunk) {
        const int cnt = (end - cs) < chunk ? (end - cs) : chunk;
        __syncthreads();
        for (int i = tid; i < cnt; i += bs) {
            const int n = csc_row[cs + i];
            sh_w[i] = s_lambda_P[csc_pidx[cs + i]] * ri_t[n];
            sh_n[i] = n;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < XDOT_MAX_NB * 8; ++k) a32[k] = 0.0f;

        // 4-deep member prefetch: the four ``__ldg`` are issued before any of
        // the dependent ``fmaf``, which is what hides the 1-byte load latency.
        // The per-``d`` accumulator still folds members in ascending order,
        // so this is bit-exact. (Per-launch timings are workload-specific:
        // docs/step2_sparse_design.md section 9, docs/step3_sparse_design.md
        // section 4.)
        #pragma unroll
        for (int j = 0; j < XDOT_MAX_NB; ++j) {
            const int b = tid + j * XDOT_BS;
            if (b < D_bytes) {
                int i = 0;
                for (; i + 4 <= cnt; i += 4) {
                    const unsigned int v0 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i] * (size_t)D_bytes + b);
                    const unsigned int v1 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i + 1] * (size_t)D_bytes + b);
                    const unsigned int v2 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i + 2] * (size_t)D_bytes + b);
                    const unsigned int v3 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i + 3] * (size_t)D_bytes + b);
                    const float w0 = sh_w[i], w1 = sh_w[i + 1];
                    const float w2 = sh_w[i + 2], w3 = sh_w[i + 3];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float a = a32[j * 8 + k];
                        a = fmaf(w0, (float)((v0 >> k) & 1u), a);
                        a = fmaf(w1, (float)((v1 >> k) & 1u), a);
                        a = fmaf(w2, (float)((v2 >> k) & 1u), a);
                        a = fmaf(w3, (float)((v3 >> k) & 1u), a);
                        a32[j * 8 + k] = a;
                    }
                }
                for (; i < cnt; ++i) {
                    const float w = sh_w[i];
                    const unsigned int bv = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i] * (size_t)D_bytes + b);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const float bit = (float)((bv >> k) & 1u);
                        a32[j * 8 + k] = fmaf(w, bit, a32[j * 8 + k]);
                    }
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < XDOT_MAX_NB * 8; ++k) acc[k] += (double)a32[k];
    }

    #pragma unroll
    for (int j = 0; j < XDOT_MAX_NB; ++j) {
        const int b = tid + j * XDOT_BS;
        if (b < D_bytes) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = b * 8 + k;
                if (d < D) out_tl[d] = (float)(acc[j * 8 + k] - B);
            }
        }
    }
}
"""
