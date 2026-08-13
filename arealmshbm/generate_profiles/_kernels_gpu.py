"""_kernels_gpu.py

GPU kernels for the per-session FC-profile leaf — CuPy counterparts to
the three numba kernels in :mod:`._kernels`.

    zscore_unit_norm_columns_cupy(x_TxN_dev, out_TxN_dev)
        Fused per-column zero-mean + L2-unit-norm. One RawKernel pass:
        thread-block per column, shared-mem fp64 reduction for
        sum / sumsq, then a write-back pass that uses
        ``post_sumsq = sumsq - T * mean²`` to recover the centered L2
        norm without a separate centered-array pass. Matches the CPU
        ``_zscore_unit_norm_columns_kernel`` semantics: fp64 reductions,
        fp32 storage, ``inv_norm = +inf`` for the zero-variance column
        (the downstream NaN→0 kernel sweeps the resulting inf away).

    binarize_mwzero_pack_cupy(corr_KxV_dev, mw_v_dev, threshold,
                              packed_VxDb_dev)
        Fused binarize + MW-zero + transpose-pack-along-K. Reads
        ``(K, V) fp32``, writes ``(V, ⌈K/8⌉) uint8`` directly. One
        thread per output byte; warps of 32 threads cover 32 adjacent
        v's at the same byte-index, giving coalesced fp32 reads from
        the K-major input. MW vertices are zeroed at the byte level
        (entire byte = 0). Bit convention matches
        ``numpy.packbits(bitorder='little')``: cell k ↔ bit (k & 7) of
        byte (k >> 3); padding bits past K in the last byte are zero.

NaN-to-zero is inlined at the call site as ``cp.nan_to_num(corr, copy=False)``
(one elementwise pass; no RawKernel needed).

cupy is imported at module top — this file is only loaded via the
lazy import inside :mod:`.profiles_gpu`, so the CPU path never pays
the cupy import cost.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import cupy as cp


_ZSCORE_BLOCK = 256
_ZSCORE_BLOCK_MAX = 256

_zscore_kernel = cp.RawKernel(r"""
extern "C" __global__
void zscore_unit_norm_columns(const float* __restrict__ x,
                               float* __restrict__ out,
                               int T, int N) {
    // One thread block per column j in [0, N). Two passes over the
    // column of length T:
    //   pass 1: sum + sumsq  (fp64 accumulators; shared-mem reduction)
    //   pass 2: write (v - mean) * inv_norm
    //
    // We recover the centered L2 norm via the identity
    //   post_sumsq = sumsq - T * mean * mean
    // -- same trick as step3's normalize_bold kernel. This is the
    // catastrophic-cancellation-prone form: when |mean| is comparable
    // to sqrt(sumsq / T), the subtraction loses ULPs. Precondition:
    // inputs must arrive demeaned upstream (CBIG preprocessing emits
    // detrended BOLD; the YS dataset complies). A caller feeding raw
    // BOLD with a large DC offset would see the binary-agreement spec
    // bar slip below 99.99 %. The CPU kernel uses the well-conditioned
    // two-pass formulation and would not.

    const int j = blockIdx.x;
    if (j >= N) return;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (int i = tid; i < T; i += bs) {
        float v = x[i * N + j];   // x is (T, N) C-contig; stride is N.
        local_sum += (double)v;
        local_sumsq += (double)v * (double)v;
    }
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }
    __shared__ float s_mean;
    __shared__ float s_inv_norm;
    if (tid == 0) {
        double mean_d = s_sum[0] / (double)T;
        double post_sumsq = s_sumsq[0] - (double)T * mean_d * mean_d;
        s_mean = (float)mean_d;
        // post_sumsq <= 0 indicates a zero-variance column. CPU kernel
        // sets inv_norm = +inf in that case (subsequent (v - mean) is
        // 0, so 0 * inf = NaN, caught by the NaN-to-0 sweep). Match it
        // bit-for-bit so the threshold cut comes out identical.
        s_inv_norm = (post_sumsq > 0.0)
            ? (float)(1.0 / sqrt(post_sumsq))
            : __int_as_float(0x7f800000);   // +inf
    }
    __syncthreads();
    for (int i = tid; i < T; i += bs) {
        out[i * N + j] = (x[i * N + j] - s_mean) * s_inv_norm;
    }
}
""", "zscore_unit_norm_columns")


def zscore_unit_norm_columns_cupy(x_TxN: cp.ndarray,
                                   out_TxN: cp.ndarray) -> None:
    """Per-column zscore + L2-unit-norm, GPU port of the CPU kernel.

    Inputs / outputs are CuPy device arrays. ``x_TxN`` is read-only;
    ``out_TxN`` must be the same (T, N) fp32 C-contig shape.
    """
    if x_TxN.dtype != cp.float32 or out_TxN.dtype != cp.float32:
        raise ValueError(
            f"zscore: inputs must be fp32; got {x_TxN.dtype}, {out_TxN.dtype}"
        )
    if x_TxN.shape != out_TxN.shape or x_TxN.ndim != 2:
        raise ValueError(
            f"zscore: shape mismatch / not 2-D: x={x_TxN.shape}, out={out_TxN.shape}"
        )
    if not (x_TxN.flags["C_CONTIGUOUS"] and out_TxN.flags["C_CONTIGUOUS"]):
        raise ValueError("zscore: both arrays must be C-contiguous")
    T, N = x_TxN.shape
    block = _ZSCORE_BLOCK
    if block <= 0 or (block & (block - 1)) != 0 or block > _ZSCORE_BLOCK_MAX:
        # Power-of-2 and ≤ shared-mem cap on s_sum / s_sumsq.
        raise ValueError(
            f"zscore: block size must be a power-of-2 ≤ "
            f"{_ZSCORE_BLOCK_MAX}; got {block}"
        )
    _zscore_kernel((N,), (block,),
                    (x_TxN, out_TxN, cp.int32(T), cp.int32(N)))


# ─────────────────────────────────────────────────────────────────────
# Fused binarize + MW-zero + transpose-pack-along-K
# ─────────────────────────────────────────────────────────────────────
_BINPACK_BLOCK_V = 128


_binarize_mwzero_pack_kernel = cp.RawKernel(r"""
extern "C" __global__
void binarize_mwzero_pack_KV_to_VDb(
    const float*   __restrict__ corr_KxV,    // (K, V) fp32 C-contig in
    const unsigned char* __restrict__ mw_v,  // (V,) uint8: 1 = MW, 0 = cortex
    float          threshold,
    int            K,
    int            V,
    int            D_bytes,
    unsigned char* __restrict__ packed_VxDb  // (V, D_bytes) uint8 C-contig out
) {
    // One thread = one output byte at (v, kb).
    //
    // Grid:   gridDim.x = ceil(V / blockDim.x),  gridDim.y = D_bytes
    // Block:  blockDim.x = _BINPACK_BLOCK_V (typically 128)
    //
    // Memory access:
    //   * Reads corr_KxV[k * V + v]: within a warp (32 threads, varying
    //     v at fixed k) consecutive v's hit consecutive fp32 cells ->
    //     COALESCED (one 128 B transaction per warp per k row).
    //   * Writes packed_VxDb[v * D_bytes + kb]: stride is D_bytes bytes
    //     between adjacent v's; uint8 writes coalesce through L1/L2.
    //
    // Bit convention: cell k -> bit (k & 7) of byte (k >> 3), LSB-first.
    // Matches numpy.packbits(bitorder='little'). Padding bits past K in
    // the last byte are zero (the byte starts at 0 and only sets bits
    // for k < K).

    const int v  = blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = blockIdx.y;
    if (v >= V) return;

    unsigned int byte = 0u;

    // MW vertex → emit a zero byte. This enforces the writer-side
    // contract _apply_mw_zero used to enforce on host, but in the
    // packed-byte representation directly.
    if (mw_v[v] == 0u) {
        const int k_base = kb * 8;
        const int k_lim  = (k_base + 8 < K) ? (k_base + 8) : K;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int k = k_base + b;
            if (k < k_lim) {
                const float val = corr_KxV[(size_t)k * (size_t)V + (size_t)v];
                if (val >= threshold) {
                    byte |= (1u << b);
                }
            }
        }
    }
    packed_VxDb[(size_t)v * (size_t)D_bytes + (size_t)kb] =
        (unsigned char)byte;
}
""", "binarize_mwzero_pack_KV_to_VDb")


def binarize_mwzero_pack_cupy(corr_KxV: cp.ndarray,
                               mw_v: cp.ndarray,
                               threshold: float,
                               packed_VxDb: cp.ndarray) -> None:
    """Fused binarize + MW-zero + transpose-pack-along-K, GPU.

    Parameters
    ----------
    corr_KxV : cp.ndarray
        ``(K, V) fp32`` C-contig device array — the per-session
        correlation accumulator after ``*= 1/n_runs`` (i.e. the input
        the current ``(corr_sum_dev >= t).astype(fp32)`` consumes).
    mw_v : cp.ndarray
        ``(V,) uint8`` device array — 1 at medial-wall vertices,
        0 at cortex. Matches ``MARS_label == 1`` on the target mesh.
    threshold : float
        The fp32 threshold value (from ``cp.partition`` on the joint
        (lh + rh) correlation sum). Compared with ``>=``, matching the
        CPU path.
    packed_VxDb : cp.ndarray
        ``(V, ⌈K/8⌉) uint8`` C-contig device array — the output
        packed bytes. Bit convention matches
        ``numpy.packbits(bitorder='little')``.

    Outputs are written **in place** into ``packed_VxDb``. Padding bits
    past K in the last byte are guaranteed zero. MW rows in the output
    are entirely zero bytes.
    """
    if corr_KxV.dtype != cp.float32:
        raise ValueError(
            f"binarize_mwzero_pack: corr_KxV must be fp32; got {corr_KxV.dtype}"
        )
    if mw_v.dtype != cp.uint8:
        raise ValueError(
            f"binarize_mwzero_pack: mw_v must be uint8; got {mw_v.dtype}"
        )
    if packed_VxDb.dtype != cp.uint8:
        raise ValueError(
            f"binarize_mwzero_pack: packed_VxDb must be uint8; got {packed_VxDb.dtype}"
        )
    if corr_KxV.ndim != 2 or packed_VxDb.ndim != 2 or mw_v.ndim != 1:
        raise ValueError(
            f"binarize_mwzero_pack: shape ranks must be (2D, 1D, 2D); got "
            f"corr={corr_KxV.shape}, mw_v={mw_v.shape}, packed={packed_VxDb.shape}"
        )
    K, V = corr_KxV.shape
    V_out, D_bytes = packed_VxDb.shape
    if V_out != V:
        raise ValueError(
            f"binarize_mwzero_pack: V mismatch — corr has V={V}, packed has V={V_out}"
        )
    if mw_v.shape[0] != V:
        raise ValueError(
            f"binarize_mwzero_pack: mw_v.size ({mw_v.shape[0]}) != V ({V})"
        )
    expected_D_bytes = (int(K) + 7) // 8
    if D_bytes != expected_D_bytes:
        raise ValueError(
            f"binarize_mwzero_pack: D_bytes ({D_bytes}) != ⌈K/8⌉ "
            f"({expected_D_bytes}) for K={K}"
        )
    if not (corr_KxV.flags["C_CONTIGUOUS"] and
            packed_VxDb.flags["C_CONTIGUOUS"] and
            mw_v.flags["C_CONTIGUOUS"]):
        raise ValueError(
            "binarize_mwzero_pack: all arrays must be C-contiguous"
        )

    block_v = _BINPACK_BLOCK_V
    grid_x = (int(V) + block_v - 1) // block_v
    grid_y = int(D_bytes)
    _binarize_mwzero_pack_kernel(
        (grid_x, grid_y), (block_v,),
        (corr_KxV, mw_v, cp.float32(threshold),
         cp.int32(K), cp.int32(V), cp.int32(D_bytes),
         packed_VxDb),
    )
