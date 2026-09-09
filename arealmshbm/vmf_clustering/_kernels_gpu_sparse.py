"""_kernels_gpu_sparse.py

CUDA kernels (CuPy ``RawModule``) for the ``gpu_sparse`` step-3 backend.
Everything operates on the candidate-set layout described in
:mod:`arealmshbm.vmf_clustering.sparse_layout` and in
``docs/step3_sparse_design.md`` (§2 is the contract each kernel below
implements; the reference is the CPU numba kernel named in each
docstring).

Design rules
------------
* No floating-point atomics. Every reduction is a fixed tree (block
  partials → single-block final pass), so a run is bit-reproducible.
* The module is compiled with ``-fmad=false``: the fp32 op sequences that
  mirror the CPU kernels (E-step assembly, softmax, intra_em algebra)
  must not be contracted into FMAs, or they would round differently from
  numba's ``fastmath=False`` code.
* ``exp`` is evaluated in fp64 on the fp32 difference and cast, exactly
  like numba's ``float32(math.exp(x))``.
* CuPy appends ``-ftz=true`` to every nvrtc compile, so fp32 denormals
  DO flush to zero here — that cannot be turned off from ``options=``.
  It is safe for this backend because the one quantity with denormal
  inputs, ``log(theta)``, is computed on the host in fp64 (see
  :class:`VmfClusteringSessionSparseCUDA`); every other fp32 value the
  kernels touch is far from the 1e-38 boundary.
* Kernels never allocate; the Session owns every buffer.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import cupy as cp
import numpy as np

from arealmshbm.em_stop_criterion._kernels import LOG_EPS_POW20
from arealmshbm.vmf_clustering.sparse_layout import MAX_D_BYTES


_SRC = r"""

#define WARP 32

// ---------------------------------------------------------------------
// Block-level fixed-tree reduction of doubles (blockDim.x must be a
// power of two <= 1024). Result valid in thread 0's return value.
// ---------------------------------------------------------------------
__device__ __forceinline__ double block_reduce_sum_d(double v, double* sh) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    double r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ double warp_reduce_sum_d(double v) {
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_sum_f(float v) {
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Final fixed-order sum of per-block partials -> out[0].
extern "C" __global__
void reduce_partials_d(const double* __restrict__ part, int n, double* __restrict__ out) {
    __shared__ double sh[1024];
    double v = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += part[i];
    v = block_reduce_sum_d(v, sh);
    if (threadIdx.x == 0) out[0] = v;
}

// ---------------------------------------------------------------------
// Packed-BOLD helpers
// ---------------------------------------------------------------------
// Zero the MW rows of the packed BOLD for every session t.
extern "C" __global__
void zero_rows_packed(unsigned char* __restrict__ packed, int T, int N, int Db,
                      const int* __restrict__ rows, int n_rows) {
    const long long total = (long long)T * n_rows * Db;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int b = (int)(i % Db);
        long long r = i / Db;
        int ri = (int)(r % n_rows);
        int t = (int)(r / n_rows);
        packed[((long long)t * N + rows[ri]) * Db + b] = 0;
    }
}

// ---------------------------------------------------------------------
// s_t_nu column statistics: S[t,l] = Sum_d s_t_nu[t,l,d] (fp64),
// allzero[t,l], anynan[t,l]. One block per (t, l).
// ---------------------------------------------------------------------
extern "C" __global__
void stnu_col_stats(const float* __restrict__ stnu, int D,
                    double* __restrict__ S, unsigned char* __restrict__ allzero,
                    unsigned char* __restrict__ anynan) {
    __shared__ double sh[1024];
    __shared__ int flags[2];
    const int tl = blockIdx.x;
    const float* row = stnu + (size_t)tl * D;
    if (threadIdx.x == 0) { flags[0] = 1; flags[1] = 0; }
    __syncthreads();
    double s = 0.0;
    int nz = 0, nan = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = row[d];
        s += (double)v;
        if (v != 0.0f) nz = 1;
        if (v != v) nan = 1;
    }
    if (nz) atomicAnd(&flags[0], 0);
    if (nan) atomicOr(&flags[1], 1);
    s = block_reduce_sum_d(s, sh);
    if (threadIdx.x == 0) {
        S[tl] = s;
        allzero[tl] = (unsigned char)flags[0];
        anynan[tl] = (unsigned char)flags[1];
    }
}

// Combine the per-(t,l) flags over t -> per-l col_zero / col_nan.
extern "C" __global__
void combine_col_flags(const unsigned char* __restrict__ allzero,
                       const unsigned char* __restrict__ anynan,
                       int T, int L, unsigned char* __restrict__ col_zero,
                       unsigned char* __restrict__ col_nan) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= L) return;
    unsigned char z = 1, n = 0;
    for (int t = 0; t < T; ++t) {
        z &= allzero[t * L + l];
        n |= anynan[t * L + l];
    }
    col_zero[l] = z;
    col_nan[l] = n;
}

// ---------------------------------------------------------------------
// acc[p] = Sum_t Sum_d X[n,t,d] * s_t_nu[t,l,d]   (design ?2.1)
//   X[n,t,d] = (bit - mean[t,n]) * inv[t,n]
//   => acc = Sum_t inv * (Sum_{d: bit} s_t_nu[t,l,d] - mean * S[t,l])
// One warp per active row m. The warp first expands the set bits of
// row (t, n) into a shared-memory list, then for each candidate l the
// lanes stride over that list gathering s_t_nu[t,l,*] (a 4.7 KB,
// L1-resident segment). fp64 accumulation throughout, fp32 store.
// Candidates are processed in chunks of ACC_CHUNK so the per-candidate
// fp64 accumulators live in shared memory regardless of row degree.
// ---------------------------------------------------------------------
#define ACC_WARPS 4
#define ACC_MAXB 8          // bytes per lane; 32 lanes => Db <= MAX_D_BYTES

extern "C" __global__
void acc_bits(int M, const int* __restrict__ row_idx_active,
              const int* __restrict__ row_ptr, const int* __restrict__ col,
              int T, int N, int Db, int D, int L,
              const unsigned char* __restrict__ packed,
              const float* __restrict__ row_mean, const float* __restrict__ row_inv,
              const float* __restrict__ stnu,      // (T, L, D)
              const double* __restrict__ S,        // (T, L)
              float* __restrict__ acc_out) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int m = blockIdx.x * ACC_WARPS + wid;
    if (m >= M) return;
    const int n = row_idx_active[m];
    const int p0 = row_ptr[m], p1 = row_ptr[m + 1];
    const int nB = (Db + WARP - 1) / WARP;      // bytes per lane

    for (int p = p0; p < p1; ++p) {
        const int l = col[p];
        float part = 0.0f;                       // lane partial of Sum_t inv_t * (bits_t . s_t_nu[t,l,:])
        for (int t = 0; t < T; ++t) {
            const float inv = row_inv[t * N + n];
            if (inv == 0.0f) continue;           // zero row (has_zero gate)
            const unsigned char* prow = packed + ((size_t)t * N + n) * Db;
            const float* srow = stnu + ((size_t)t * L + l) * D;
            float pt = 0.0f;
            #pragma unroll
            for (int i = 0; i < ACC_MAXB; ++i) {
                if (i >= nB) break;
                const int b = lane + i * WARP;
                unsigned int byte = (b < Db) ? (unsigned int)__ldg(prow + b) : 0u;
                const float* sb = srow + b * 8;
                // Fixed 8-step predicated form: no data-dependent loop,
                // the 8 loads are independent and issue together.
                float v0 = (byte & 1u)   ? __ldg(sb + 0) : 0.0f;
                float v1 = (byte & 2u)   ? __ldg(sb + 1) : 0.0f;
                float v2 = (byte & 4u)   ? __ldg(sb + 2) : 0.0f;
                float v3 = (byte & 8u)   ? __ldg(sb + 3) : 0.0f;
                float v4 = (byte & 16u)  ? __ldg(sb + 4) : 0.0f;
                float v5 = (byte & 32u)  ? __ldg(sb + 5) : 0.0f;
                float v6 = (byte & 64u)  ? __ldg(sb + 6) : 0.0f;
                float v7 = (byte & 128u) ? __ldg(sb + 7) : 0.0f;
                pt += ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7));
            }
            part += inv * pt;
        }
        const float tot = warp_reduce_sum_f(part);
        if (lane == 0) {
            double cval = 0.0;                   // Sum_t inv_t * mean_t * S[t,l]
            for (int t = 0; t < T; ++t) {
                const float inv = row_inv[t * N + n];
                if (inv == 0.0f) continue;
                cval += (double)inv * (double)row_mean[t * N + n] * S[t * L + l];
            }
            acc_out[p] = (float)((double)tot - cval);
        }
    }
}

// ---------------------------------------------------------------------
// E-step lambda-iteration (design ?2.2) ? mirrors
// V_lambda/_kernels.v_lambda_potts_closeform_fused_full_lam_f32 +
// vmf_clustering/_kernels.fused_v_lambda_assemble_softmax_drift_f32.
// ---------------------------------------------------------------------
extern "C" __global__
void row_sum_k(int M, const int* __restrict__ row_ptr, const float* __restrict__ lam,
               float* __restrict__ row_sum) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float s = 0.0f;
    for (int p = row_ptr[m]; p < row_ptr[m + 1]; ++p) s += lam[p];
    row_sum[m] = s;
}

extern "C" __global__
void estep_k(int M, const int* __restrict__ row_ptr, const int* __restrict__ col,
             const int* __restrict__ nbh, int M1,
             const float* __restrict__ lam, const float* __restrict__ row_sum,
             const float* __restrict__ acc, const float* __restrict__ kappa,
             const float* __restrict__ cdln_T, const unsigned char* __restrict__ col_zero,
             const float* __restrict__ log_theta, float w, float two_c,
             const float* __restrict__ beta, const float* __restrict__ scv,
             const float* __restrict__ sxv, const float* __restrict__ bm,
             const unsigned char* __restrict__ row_poison,
             float* __restrict__ V_temp, float* __restrict__ out,
             double* __restrict__ block_drift) {
    __shared__ double sh[1024];
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    double drift = 0.0;
    if (m < M) {
        const int p0 = row_ptr[m], p1 = row_ptr[m + 1];
        // Phase 2: neighbour sum of row sums (ascending n).
        float nbr = 0.0f;
        for (int nn = 0; nn < M1; ++nn) {
            int j = nbh[(size_t)m * M1 + nn];
            if (j != 0) nbr += row_sum[j - 1];
        }
        // Pass A: V_lambda close-form + log_vmf assembly, rmax.
        float rmax = (-__int_as_float(0x7f800000));
        for (int p = p0; p < p1; ++p) {
            const int k = col[p];
            float acc_lam = 0.0f;
            for (int nn = 0; nn < M1; ++nn) {
                int j = nbh[(size_t)m * M1 + nn];
                if (j == 0) continue;
                float val = 0.0f;
                const int q0 = row_ptr[j - 1], q1 = row_ptr[j];
                for (int q = q0; q < q1; ++q) {
                    int ck = col[q];
                    if (ck == k) { val = lam[q]; break; }
                    if (ck > k) break;
                }
                acc_lam += val;
            }
            const float vt = nbr - acc_lam;
            V_temp[p] = vt;
            float v = acc[p] * kappa[k] + (col_zero[k] ? 0.0f : cdln_T[k]);
            v += w * log_theta[p];
            v -= two_c * vt;
            v += beta[k] * scv[p];
            v += sxv[p];
            out[p] = v;
            if (v > rmax) rmax = v;
        }
        // Pass B: exp + boundary mask.
        float rsum = 0.0f;
        for (int p = p0; p < p1; ++p) {
            const float bmv = bm[p];
            if (bmv == 0.0f) { out[p] = 0.0f; continue; }
            float ev = (float)exp((double)(out[p] - rmax));
            ev *= bmv;
            out[p] = ev;
            rsum += ev;
        }
        // Pass C: normalise + drift.
        if (rsum > 0.0f && rsum == rsum && row_poison[m] == 0) {
            const float inv = 1.0f / rsum;
            for (int p = p0; p < p1; ++p) {
                const int k = col[p];
                float nv = col_zero[k] ? 0.0f : out[p] * inv;
                out[p] = nv;
                double d = (double)nv - (double)lam[p];
                drift += (d < 0.0) ? -d : d;
            }
        } else {
            for (int p = p0; p < p1; ++p) {
                drift += (double)lam[p];
                out[p] = 0.0f;
            }
        }
    }
    drift = block_reduce_sum_d(drift, sh);
    if (threadIdx.x == 0) block_drift[blockIdx.x] = drift;
}

// row_poison[m] = any(col_nan[l] for l in bm support of row m)
extern "C" __global__
void row_poison_k(int M, const int* __restrict__ bm_row_ptr, const int* __restrict__ bm_col,
                  const unsigned char* __restrict__ col_nan, unsigned char* __restrict__ out) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    unsigned char r = 0;
    for (int p = bm_row_ptr[m]; p < bm_row_ptr[m + 1]; ++p) r |= col_nan[bm_col[p]];
    out[m] = r;
}

// ---------------------------------------------------------------------
// argmax labels: labels[n] = 1 + first argmax over P(n); 0 if the row
// sums to zero or is inactive. (data_io.derive_labels semantics)
// ---------------------------------------------------------------------
extern "C" __global__
void argmax_labels_k(int N, const int* __restrict__ inv_active,
                     const int* __restrict__ row_ptr, const int* __restrict__ col,
                     const float* __restrict__ lam, int* __restrict__ labels) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int m = inv_active[n];
    int lab = 0;
    if (m >= 0) {
        float s = 0.0f, best = (-__int_as_float(0x7f800000)); int bk = -1;
        for (int p = row_ptr[m]; p < row_ptr[m + 1]; ++p) {
            float v = lam[p];
            s += v;
            if (v > best) { best = v; bk = col[p]; }
        }
        if (s != 0.0f && bk >= 0) lab = bk + 1;
    }
    labels[n] = lab;
}

// ---------------------------------------------------------------------
// spatial_connect (design ?2.4) ? mirrors spatial_priors.ConnectSession
// ---------------------------------------------------------------------
// u[l, :] and u_sq[l]: one block per parcel, thread per gradient dim.
extern "C" __global__
void connect_u_k(int L, int Dg, const int* __restrict__ col_ptr,
                 const int* __restrict__ csc_row, const int* __restrict__ csc_pidx,
                 const float* __restrict__ lam, const float* __restrict__ grad,
                 float* __restrict__ u, float* __restrict__ u_sq) {
    __shared__ float s_u[1024];
    const int l = blockIdx.x;
    const int i0 = col_ptr[l], i1 = col_ptr[l + 1];
    double slam = 0.0;
    for (int i = i0; i < i1; ++i) slam += (double)lam[csc_pidx[i]];
    const float slam_f = (float)slam;
    for (int d = threadIdx.x; d < Dg; d += blockDim.x) {
        double s = 0.0;
        for (int i = i0; i < i1; ++i) {
            float sl = lam[csc_pidx[i]];
            s += (double)(sl * grad[(size_t)csc_row[i] * Dg + d]);
        }
        float uu = (float)s / slam_f;          // NaN at empty parcels (0/0)
        u[(size_t)l * Dg + d] = uu;
        if (d < 1024) s_u[d] = uu;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sq = 0.0f;
        for (int d = 0; d < Dg; ++d) {
            float uu = (d < 1024) ? s_u[d] : u[(size_t)l * Dg + d];
            sq += uu * uu;
        }
        u_sq[l] = sq;
    }
}

// scv[p] = 2*(g_n*u_l) - ||g_n||^2 - ||u_l||^2, NaN -> -Inf. One thread per p.
extern "C" __global__
void connect_scv_k(int P, const int* __restrict__ p_row_m, const int* __restrict__ row_idx_active,
                   const int* __restrict__ col, int Dg,
                   const float* __restrict__ grad, const float* __restrict__ grad_sq,
                   const float* __restrict__ u, const float* __restrict__ u_sq,
                   float* __restrict__ scv) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int n = row_idx_active[p_row_m[p]];
    const int l = col[p];
    const float* g = grad + (size_t)n * Dg;
    const float* uu = u + (size_t)l * Dg;
    double cross = 0.0;
    for (int d = 0; d < Dg; ++d) cross += (double)(g[d] * uu[d]);
    float v = 2.0f * (float)cross - grad_sq[n] - u_sq[l];
    scv[p] = (v != v) ? (-__int_as_float(0x7f800000)) : v;
}

// ---------------------------------------------------------------------
// spatial_xyz (design ?2.5) ? mirrors spatial_priors.XyzSession
// ---------------------------------------------------------------------
#define LOG_2PI 1.8378770664093454835606594728112

extern "C" __global__
void xyz_muc_k(int L, const int* __restrict__ col_ptr, const int* __restrict__ csc_row,
               const int* __restrict__ csc_pidx, const float* __restrict__ lam,
               const float* __restrict__ sphere,        // (N, 3)
               const double* __restrict__ gamma,        // (L,) fp64
               float* __restrict__ s_muc,               // (3, L)
               float* __restrict__ cdln3, float* __restrict__ gamma_f32) {
    // One block per parcel; threads stride over the members, fixed-tree
    // fp64 block reductions (deterministic).
    __shared__ double sh[1024];
    const int l = blockIdx.x;
    double x = 0.0, y = 0.0, z = 0.0;
    for (int i = col_ptr[l] + threadIdx.x; i < col_ptr[l + 1]; i += blockDim.x) {
        const float sl = lam[csc_pidx[i]];
        const float* sp = sphere + (size_t)csc_row[i] * 3;
        x += (double)(sp[0] * sl); y += (double)(sp[1] * sl); z += (double)(sp[2] * sl);
    }
    x = block_reduce_sum_d(x, sh);
    y = block_reduce_sum_d(y, sh);
    z = block_reduce_sum_d(z, sh);
    if (threadIdx.x != 0) return;
    float xf = (float)x, yf = (float)y, zf = (float)z;
    float cn = sqrtf(xf * xf + yf * yf + zf * zf);
    s_muc[l] = xf / cn; s_muc[L + l] = yf / cn; s_muc[2 * L + l] = zf / cn;
    // Cdln(k, 3) closed form (spatial_priors/_cdln.cdln_d3_to_f32).
    double k = gamma[l];
    float c;
    if (k <= 0.0) c = __int_as_float(0x7fc00000);
    else {
        double lom = (k < 30.0) ? log1p(-exp(-2.0 * k)) : 0.0;
        c = (float)(log(k) - k + LOG_2PI * 0.5 - lom);
    }
    cdln3[l] = c;
    gamma_f32[l] = (float)k;
}

extern "C" __global__
void xyz_sxv_k(int P, int L, const int* __restrict__ p_row_m, const int* __restrict__ row_idx_active,
               const int* __restrict__ col, const float* __restrict__ sphere,
               const float* __restrict__ s_muc, const float* __restrict__ cdln3,
               const float* __restrict__ gamma_f32, float* __restrict__ sxv) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int n = row_idx_active[p_row_m[p]];
    const int l = col[p];
    const float* s = sphere + (size_t)n * 3;
    float cosv = s[0] * s_muc[l] + s[1] * s_muc[L + l] + s[2] * s_muc[2 * L + l];
    float v = cdln3[l] + gamma_f32[l] * cosv;
    sxv[p] = (v != v) ? 0.0f : v;
}

// ---------------------------------------------------------------------
// EM stop cost (design ?2.7) ? mirrors
// em_stop_criterion/_kernels.fused_em_stop_assemble_cleanup_cost_f32
// ---------------------------------------------------------------------
extern "C" __global__
void em_stop_k(int P, const int* __restrict__ col, const float* __restrict__ acc,
               const float* __restrict__ kappa, const float* __restrict__ cdln,
               float Tf, const float* __restrict__ lam, const float* __restrict__ log_theta_cost,
               const float* __restrict__ V_temp, float* __restrict__ scv,
               double w, double c, const float* __restrict__ beta, float floor_f32,
               double floor_f64, double* __restrict__ block_out) {
    __shared__ double sh[1024];
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    double term = 0.0;
    if (p < P) {
        const int l = col[p];
        float llp = Tf * cdln[l] + kappa[l] * acc[p];
        float scv_v = scv[p];
        if (!isfinite(scv_v)) scv_v = floor_f32;
        scv[p] = scv_v;
        float sl_f = lam[p];
        double sl = (double)sl_f;
        double log_sl = (sl_f <= 0.0f) ? floor_f64 : log(sl);
        double t = (double)llp + (double)log_theta_cost[p] - w * log_sl
                   - c * (double)V_temp[p] + (double)beta[l] * (double)scv_v;
        term = sl * t;
    }
    term = block_reduce_sum_d(term, sh);
    if (threadIdx.x == 0) block_out[blockIdx.x] = term;
}

// ---------------------------------------------------------------------
// intra_em outer-loop algebra (design ?2.8) ? mirrors
// intra_em.intra_subject_var + intra_em.intra_em_cost. One thread per l.
//   s_t_nu (T, L, D), mu / s_psi (L, D)
// ---------------------------------------------------------------------
extern "C" __global__
void intra_psi_k(int T, int L, int D, const float* __restrict__ stnu,
                 const float* __restrict__ sigma, const float* __restrict__ epsil,
                 const float* __restrict__ mu, float* __restrict__ s_psi_new,
                 float* __restrict__ summed_LD,
                 double* __restrict__ per_col1, double* __restrict__ per_col2) {
    // One block per parcel l. Pass 1/3 are parallel over d; the fp32
    // column norm (pass 2) is a serial ascending-d sum on thread 0 so it
    // rounds exactly like numpy's axis-0 reduction in intra_subject_var.
    // The T sum below is serial too, which matches numpy bit-for-bit
    // only for T < 8 (above that numpy's pairwise sum starts blocking).
    __shared__ double sh[1024];
    __shared__ float s_cn;
    const int l = blockIdx.x;
    const float sig = sigma[l], eps = epsil[l];
    float* upd_row = s_psi_new + (size_t)l * D;
    float* sum_row = summed_LD + (size_t)l * D;
    const float* mu_row = mu + (size_t)l * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float summed = 0.0f;
        for (int t = 0; t < T; ++t) summed += stnu[((size_t)t * L + l) * D + d];
        sum_row[d] = summed;
        upd_row[d] = summed * sig + mu_row[d] * eps;      // un-normalised
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float norm_acc = 0.0f;
        for (int d = 0; d < D; ++d) { float u = upd_row[d]; norm_acc += u * u; }
        s_cn = sqrtf(norm_acc);
    }
    __syncthreads();
    const float cn = s_cn;
    const float safe = (cn > 0.0f) ? cn : 1.0f;
    double pc1 = 0.0, pc2 = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float psi = (cn == 0.0f) ? 0.0f : upd_row[d] / safe;
        upd_row[d] = psi;
        pc1 += (double)psi * (double)sum_row[d];
        pc2 += (double)mu_row[d] * (double)psi;
    }
    pc1 = block_reduce_sum_d(pc1, sh);
    pc2 = block_reduce_sum_d(pc2, sh);
    if (threadIdx.x == 0) { per_col1[l] = pc1; per_col2[l] = pc2; }
}

// Reset s_t_nu[t, l, :] = mu[l, :] for every t.
extern "C" __global__
void broadcast_mu_k(int T, long long LD, const float* __restrict__ mu, float* __restrict__ stnu) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < LD; i += (long long)gridDim.x * blockDim.x) {
        float v = mu[i];
        for (int t = 0; t < T; ++t) stnu[(size_t)t * LD + i] = v;
    }
}

// Scatter a P-vector into a dense (N, L) array (zero-filled by caller).
extern "C" __global__
void scatter_dense_k(int P, const int* __restrict__ p_row_m, const int* __restrict__ row_idx_active,
                     const int* __restrict__ col, int L, const float* __restrict__ x,
                     float* __restrict__ dense) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    dense[(size_t)row_idx_active[p_row_m[p]] * L + col[p]] = x[p];
}

"""

_MODULE = None


def module() -> cp.RawModule:
    global _MODULE
    if _MODULE is None:
        _MODULE = cp.RawModule(code=_SRC, options=("-std=c++17", "-fmad=false"))
    return _MODULE


def _k(name: str):
    return module().get_function(name)


def _grid(n: int, block: int) -> int:
    return (int(n) + block - 1) // block


# ─────────────────────────────────────────────────────────────────────
# Python wrappers. All arguments are device arrays unless noted.
# ─────────────────────────────────────────────────────────────────────
def zero_rows_packed(packed_TND: cp.ndarray, rows: cp.ndarray) -> None:
    T, N, Db = packed_TND.shape
    n_rows = int(rows.size)
    if n_rows == 0:
        return
    total = T * n_rows * Db
    _k("zero_rows_packed")((min(_grid(total, 256), 4096),), (256,),
                           (packed_TND, np.int32(T), np.int32(N), np.int32(Db),
                            rows, np.int32(n_rows)))


def stnu_col_stats(stnu_TLD: cp.ndarray, S_TL: cp.ndarray,
                   allzero_TL: cp.ndarray, anynan_TL: cp.ndarray,
                   col_zero_L: cp.ndarray, col_nan_L: cp.ndarray) -> None:
    T, L, D = stnu_TLD.shape
    _k("stnu_col_stats")((T * L,), (256,),
                         (stnu_TLD, np.int32(D), S_TL, allzero_TL, anynan_TL))
    _k("combine_col_flags")((_grid(L, 128),), (128,),
                            (allzero_TL, anynan_TL, np.int32(T), np.int32(L),
                             col_zero_L, col_nan_L))


def acc_bits(lay: dict, packed_TND: cp.ndarray, row_mean: cp.ndarray,
             row_inv: cp.ndarray, stnu_TLD: cp.ndarray, S_TL: cp.ndarray,
             D: int, acc_out: cp.ndarray) -> None:
    T, N, Db = packed_TND.shape
    L = stnu_TLD.shape[1]
    if Db > MAX_D_BYTES:
        raise ValueError(
            f"acc_bits: D_bytes > {MAX_D_BYTES} (D > {MAX_D_BYTES * 8}) "
            f"not supported")
    M = lay["M_active"]
    _k("acc_bits")((_grid(M, 4),), (128,),
                   (np.int32(M), lay["row_idx_active"], lay["row_ptr"], lay["col"],
                    np.int32(T), np.int32(N), np.int32(Db), np.int32(D), np.int32(L),
                    packed_TND, row_mean, row_inv, stnu_TLD, S_TL, acc_out))


def row_poison(lay: dict, col_nan_L: cp.ndarray, out_M: cp.ndarray) -> None:
    M = lay["M_active"]
    _k("row_poison_k")((_grid(M, 256),), (256,),
                       (np.int32(M), lay["bm_row_ptr"], lay["bm_col"], col_nan_L, out_M))


class EStepWorkspace:
    """Scratch for the λ-iteration: row sums + fp64 block partials."""

    BLOCK = 256

    def __init__(self, M: int):
        self.M = int(M)
        self.nblocks = _grid(self.M, self.BLOCK)
        self.row_sum = cp.empty(self.M, dtype=cp.float32)
        self.partials = cp.empty(self.nblocks, dtype=cp.float64)
        self.scalar = cp.empty(1, dtype=cp.float64)
        self.host = cp.cuda.alloc_pinned_memory(8)
        self.host_np = np.frombuffer(self.host, dtype=np.float64, count=1)


def estep_iteration(lay: dict, ws: EStepWorkspace,
                    lam: cp.ndarray, acc: cp.ndarray, kappa_f32: cp.ndarray,
                    cdln_T: cp.ndarray, col_zero: cp.ndarray, log_theta: cp.ndarray,
                    w: float, c: float, beta: cp.ndarray, scv: cp.ndarray,
                    sxv: cp.ndarray, bm: cp.ndarray, row_poison_M: cp.ndarray,
                    V_temp: cp.ndarray, out: cp.ndarray) -> float:
    """One λ-iteration; returns ``mean(|out - lam|)`` over N·L (fp64).

    Performs one device→host copy (the drift scalar)."""
    M = lay["M_active"]
    M1 = int(lay["neighborhood"].shape[1])
    _k("row_sum_k")((ws.nblocks,), (ws.BLOCK,), (np.int32(M), lay["row_ptr"], lam, ws.row_sum))
    _k("estep_k")((ws.nblocks,), (ws.BLOCK,),
                  (np.int32(M), lay["row_ptr"], lay["col"], lay["neighborhood"], np.int32(M1),
                   lam, ws.row_sum, acc, kappa_f32, cdln_T, col_zero, log_theta,
                   np.float32(w), np.float32(np.float32(2.0) * np.float32(c)),
                   beta, scv, sxv, bm, row_poison_M, V_temp, out, ws.partials))
    _k("reduce_partials_d")((1,), (1024,), (ws.partials, np.int32(ws.nblocks), ws.scalar))
    ws.scalar.get(out=ws.host_np)
    return float(ws.host_np[0]) / float(lay["N"] * lay["L"])


def argmax_labels(lay: dict, lam: cp.ndarray, labels_N: cp.ndarray) -> None:
    N = lay["N"]
    _k("argmax_labels_k")((_grid(N, 256),), (256,),
                          (np.int32(N), lay["inv_active"], lay["row_ptr"], lay["col"],
                           lam, labels_N))


def connect_prior(lay: dict, p_row_m: cp.ndarray, lam: cp.ndarray, grad: cp.ndarray,
                  grad_sq: cp.ndarray, u: cp.ndarray, u_sq: cp.ndarray,
                  scv: cp.ndarray) -> None:
    L = lay["L"]; P = lay["P"]
    Dg = int(grad.shape[1])
    _k("connect_u_k")((L,), (128,),
                      (np.int32(L), np.int32(Dg), lay["col_ptr"], lay["csc_row"],
                       lay["csc_pidx"], lam, grad, u, u_sq))
    _k("connect_scv_k")((_grid(P, 256),), (256,),
                        (np.int32(P), p_row_m, lay["row_idx_active"], lay["col"], np.int32(Dg),
                         grad, grad_sq, u, u_sq, scv))


def xyz_prior(lay: dict, p_row_m: cp.ndarray, lam: cp.ndarray, sphere: cp.ndarray,
              gamma_f64: cp.ndarray, s_muc: cp.ndarray, cdln3: cp.ndarray,
              gamma_f32: cp.ndarray, sxv: cp.ndarray) -> None:
    L = lay["L"]; P = lay["P"]
    _k("xyz_muc_k")((L,), (128,),
                    (np.int32(L), lay["col_ptr"], lay["csc_row"], lay["csc_pidx"], lam,
                     sphere, gamma_f64, s_muc, cdln3, gamma_f32))
    _k("xyz_sxv_k")((_grid(P, 256),), (256,),
                    (np.int32(P), np.int32(L), p_row_m, lay["row_idx_active"], lay["col"],
                     sphere, s_muc, cdln3, gamma_f32, sxv))


class EMStopWorkspace:
    BLOCK = 256

    def __init__(self, P: int):
        self.nblocks = _grid(int(P), self.BLOCK)
        self.partials = cp.empty(self.nblocks, dtype=cp.float64)
        self.scalar = cp.empty(1, dtype=cp.float64)
        self.host = cp.cuda.alloc_pinned_memory(8)
        self.host_np = np.frombuffer(self.host, dtype=np.float64, count=1)


def em_stop_cost(lay: dict, ws: EMStopWorkspace, acc: cp.ndarray, kappa_f32: cp.ndarray,
                 cdln: cp.ndarray, T: int, lam: cp.ndarray, log_theta_cost: cp.ndarray,
                 V_temp: cp.ndarray, scv: cp.ndarray, w: float, c: float,
                 beta: cp.ndarray) -> float:
    """fp64 cost over P; also rewrites non-finite ``scv`` to the floor."""
    P = lay["P"]
    _k("em_stop_k")((ws.nblocks,), (ws.BLOCK,),
                    (np.int32(P), lay["col"], acc, kappa_f32, cdln, np.float32(T), lam,
                     log_theta_cost, V_temp, scv, np.float64(w), np.float64(c), beta,
                     np.float32(LOG_EPS_POW20), np.float64(LOG_EPS_POW20), ws.partials))
    _k("reduce_partials_d")((1,), (1024,), (ws.partials, np.int32(ws.nblocks), ws.scalar))
    ws.scalar.get(out=ws.host_np)
    return float(ws.host_np[0])


def intra_psi(stnu_TLD: cp.ndarray, sigma: cp.ndarray, epsil: cp.ndarray, mu_LD: cp.ndarray,
              s_psi_new_LD: cp.ndarray, summed_LD: cp.ndarray,
              per_col1: cp.ndarray, per_col2: cp.ndarray) -> None:
    T, L, D = stnu_TLD.shape
    _k("intra_psi_k")((L,), (256,),
                      (np.int32(T), np.int32(L), np.int32(D), stnu_TLD, sigma, epsil, mu_LD,
                       s_psi_new_LD, summed_LD, per_col1, per_col2))


def broadcast_mu(mu_LD: cp.ndarray, stnu_TLD: cp.ndarray) -> None:
    T, L, D = stnu_TLD.shape
    LD = L * D
    _k("broadcast_mu_k")((min(_grid(LD, 256), 4096),), (256,),
                         (np.int32(T), np.int64(LD), mu_LD, stnu_TLD))


def scatter_dense(lay: dict, p_row_m: cp.ndarray, x_P: cp.ndarray, dense_NL: cp.ndarray) -> None:
    P = lay["P"]
    _k("scatter_dense_k")((_grid(P, 256),), (256,),
                          (np.int32(P), p_row_m, lay["row_idx_active"], lay["col"],
                           np.int32(lay["L"]), x_P, dense_NL))


def p_row_m_of(lay: dict) -> cp.ndarray:
    """(P,) int32 — active-row index m of each CSR entry."""
    row_ptr = cp.asnumpy(lay["row_ptr"])
    counts = np.diff(row_ptr)
    return cp.asarray(np.repeat(np.arange(lay["M_active"], dtype=np.int32), counts))


__all__ = [
    "module", "zero_rows_packed", "stnu_col_stats", "acc_bits", "row_poison",
    "EStepWorkspace", "estep_iteration", "argmax_labels", "connect_prior",
    "xyz_prior", "EMStopWorkspace", "em_stop_cost", "intra_psi", "broadcast_mu",
    "scatter_dense", "p_row_m_of",
]
