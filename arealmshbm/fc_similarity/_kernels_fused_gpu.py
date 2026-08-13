"""_kernels_fused_gpu.py

Fused demean + L2-norm RawKernels for fc_similarity_gpu.

Replaces the 5-pass / fp64-buffer-materialising implementation of
``_demean_norm_columns`` / ``_demean_norm_rows`` (which together cost
~570 ms/subject on sub-A — the dominant non-GEMM work in fc_similarity)
with a single fused kernel per call doing:

    Pass 1 — accumulate fp64 mean over the reduction axis.
    Pass 2 — in-place demean of the fp32 buffer + accumulate fp64 SS;
             write back ``mag[k] = sqrtf(SS)``.

Both passes share L1 cache: pass-1 reads pull each cell once; pass-2
reads hit L1 (warm) and the demean write goes to L2/HBM. The fp64
accumulator lives entirely in registers / shared memory — no
materialisation of an intermediate fp64 array on global memory.

Memory traffic per call (B = buffer size in bytes):

    current  : ~16 B (5 passes + two fp64 materialisations)
    fused    : ~3 B  (1 cold read + 1 warm-read + 1 write)

Theoretical bandwidth speedup ≈ 5x; on small buffers small-kernel
launch overhead caps the realistic speedup.

Two kernels for two layouts because the coalescing patterns flip:

  * axis0 ((T, K) row-major, reduce over T):
      block (32, 1) — 32 threads, each owns one column j.
      Warp loads x[t, j..j+31] = 32 consecutive fp32 → coalesced.
      Each thread runs its own scalar reduction; no shared mem
      needed.

  * axis1 ((K, T) row-major, reduce over T):
      block (256, 1), one block per row k. Threads cooperate via
      shared-mem tree reduction over the T axis. Coalesced because
      consecutive threads read consecutive bytes of row k.

Bank conflicts: only the axis1 kernel uses shared mem. The
``smem[256]`` fp64 array (8 bytes/cell, 32 banks of 4 bytes each)
gives a stride-1 access pattern → no conflict in the tree-reduce
where threads access ``smem[threadIdx.x + s]``; the reduction step
also uses stride-1 accesses, also clean.

Precision contract vs current:

  * Pass-1 mean: fp64 register accumulation, sum order matches
    column-major traversal — identical to cupy.mean(x.astype(fp64),
    axis=...) modulo reduction grouping (warp-shuffle tree reduction
    introduces a different add-tree shape than cupy's. ULP-level
    drift expected; max-abs-diff ~ a few ULP of fp32(mean)).
  * Pass-2 demean: ``x -= float(mean64)`` matches the current
    ``x_d -= mean64.astype(cp.float32)`` exactly per cell.
  * Pass-2 SS: fp64 accumulation of (fp32 demeaned)^2, same as
    ``(x_d.astype(cp.float64) ** 2).sum``.

The reduction-tree shape difference is the only ULP-level source of
drift; mean / mag values match cupy.mean / cupy.sqrt(SS) to within
~1 ULP fp32. The post-demean buffer ``x_d`` is bit-identical to the
current path on any given (mean, x) pair.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import cupy as cp


# ─────────────────────────────────────────────────────────────────────
# Axis-0 reduction: (T, K) row-major, reduce over T (the slow axis,
# i.e., the "demean each column" operation).
#
# Each thread owns one column j. Loop over t = 0..T-1 (sequential in
# the register-side scalar accumulator).
#
# Warp coalescing: 32 threads with consecutive j read x[t, j..j+31]
# which are 32 consecutive fp32 bytes — 1 transaction per t-step.
#
# Block (32, 1) = exactly 1 warp. Grid (ceil(K/32), 1). No shared
# memory, no __syncthreads.
# ─────────────────────────────────────────────────────────────────────
_axis0_kernel_src = r"""
extern "C" __global__
void fused_demean_norm_axis0(
        float* __restrict__ x,     // (T, K) row-major; mutated in place
        float* __restrict__ mag,   // (K,) output
        int T, int K) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= K) return;

    // ── Pass 1: fp64 mean accumulation over column j.
    // Scalar reduction in register — no cross-thread coordination.
    double sum = 0.0;
    for (int t = 0; t < T; t++) {
        sum += (double)x[(size_t)t * (size_t)K + (size_t)j];
    }
    // Cast mean to fp32 BEFORE the demean subtract — matches the host
    // path's ``x_d -= mean64.astype(cp.float32)`` so the stored fp32
    // demeaned value is bit-identical per cell.
    const float mean = (float)(sum / (double)T);

    // ── Pass 2: in-place demean + fp64 SS accumulation.
    // Pass-2 reads find pass-1's loads in L1 (each column's T floats =
    // ~960 B at T=240 fits in L1 / texture).
    double ss = 0.0;
    for (int t = 0; t < T; t++) {
        const size_t idx = (size_t)t * (size_t)K + (size_t)j;
        const float v = x[idx];
        const float diff = v - mean;
        x[idx] = diff;
        ss += (double)diff * (double)diff;
    }
    mag[j] = sqrtf((float)ss);
}
"""


_axis0_kernel = cp.RawKernel(_axis0_kernel_src, "fused_demean_norm_axis0")
_AXIS0_BLOCK = 32


# ─────────────────────────────────────────────────────────────────────
# Axis-1 reduction: (K, T) row-major, reduce over T.
#
# One block per row k. Block (256, 1). Threads cooperate via a
# shared-mem tree reduction over the T axis (T ≤ ~256 in practice; the
# loop covers larger T but the canonical fc_similarity case is T=240).
#
# Warp coalescing: warp 0 reads x[k, 0..31] which are 32 consecutive
# fp32 bytes — coalesced.
# ─────────────────────────────────────────────────────────────────────
_axis1_kernel_src = r"""
extern "C" __global__
void fused_demean_norm_axis1(
        float* __restrict__ x,     // (K, T) row-major; mutated in place
        float* __restrict__ mag,   // (K,) output
        int K, int T) {
    extern __shared__ double smem[];

    const int k = blockIdx.x;
    if (k >= K) return;
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const size_t row_off = (size_t)k * (size_t)T;

    // ── Pass 1: parallel sum over row k, tree-reduce in shared mem.
    double local_sum = 0.0;
    for (int t = tid; t < T; t += B) {
        local_sum += (double)x[row_off + (size_t)t];
    }
    smem[tid] = local_sum;
    __syncthreads();

    // Tree reduce. Stride halves each step; bank-conflict-free because
    // threads access stride-1 slots of an 8-byte-per-cell smem buffer.
    for (int s = B / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    // After reduction, smem[0] holds the fp64 row sum.
    const float mean = (float)(smem[0] / (double)T);
    __syncthreads();  // ensure all threads see the same mean before reuse

    // ── Pass 2: in-place demean + parallel SS reduction.
    double local_ss = 0.0;
    for (int t = tid; t < T; t += B) {
        const size_t idx = row_off + (size_t)t;
        const float v = x[idx];
        const float diff = v - mean;
        x[idx] = diff;
        local_ss += (double)diff * (double)diff;
    }
    smem[tid] = local_ss;
    __syncthreads();

    for (int s = B / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) mag[k] = sqrtf((float)smem[0]);
}
"""


_axis1_kernel = cp.RawKernel(_axis1_kernel_src, "fused_demean_norm_axis1")
_AXIS1_BLOCK = 256


# ─────────────────────────────────────────────────────────────────────
# Axis-0 reduction WITH pre-divide: (T, K) row-major, reduce over T.
#
# Variant of fused_demean_norm_axis0 that absorbs two broadcast divides
# that the caller would otherwise run as two separate kernels before
# the demean+norm pass:
#
#     x[t, j] /= mr[t] * mc[j]   ← fused into pass 1
#     mean over t per column     ← fused into pass 1
#     demean + ss                ← pass 2 (unchanged)
#
# This collapses three kernel launches and three HBM passes through x
# (two divides + one demean+norm read+write) into a single launch and
# two HBM passes (one read+write per pass). At fc_similarity FC_A /
# FC_B shapes (T=N2=240, K=block_a_size or kb ≈ 800-2700) this saves
# ~10-15 ms per iter_b inner iteration.
#
# Block (32, 1) — 32 threads = 1 warp, each owns one column j. Same
# coalescing pattern as fused_demean_norm_axis0: warp loads x[t, j..j+31]
# as 32 consecutive fp32 → 1 transaction per t.
#
# Optimisation: mr[] (the row-axis divisor — typically mt[t] for
# fc_similarity, T=240 → ~960 B) is cooperatively loaded into shared
# memory once per block. Without this, every thread issues T loads
# of the SAME mr[t] value (broadcast-able but still T-bound) and the
# L1 hit rate degrades when T-loops from different blocks contend.
# With smem the second pass also benefits.
#
# Precision contract:
#   * Pass-1 divide is fp32 (matches the CPU reference path's
#     ``x[i, j] / (row_scale[i] * col_scale[j])`` in
#     :func:`fc_similarity._scale_demean_norm_inplace`).
#   * Mean accumulation is fp64 in register; cast back to fp32 before
#     subtract (matches CPU exactly per cell).
#   * SS accumulation is fp64 in register, fp32 sqrt.
# ─────────────────────────────────────────────────────────────────────
_axis0_div_kernel_src = r"""
extern "C" __global__
void fused_div_demean_norm_axis0(
        float* __restrict__ x,         // (T, K) row-major; mutated in place
        const float* __restrict__ mr,  // (T,) divisor for axis 0 (rows)
        const float* __restrict__ mc,  // (K,) divisor for axis 1 (cols)
        float* __restrict__ mag,       // (K,) output
        int T, int K) {
    extern __shared__ float smem_mr[];
    const int tid = threadIdx.x;
    const int j = blockIdx.x * blockDim.x + tid;

    // Cooperative load of mr[] into shared memory. Stride-1 reads
    // across warp threads → coalesced. For T <= blockDim.x the loop
    // is one iteration; for T > blockDim.x each thread loads multiple
    // entries (this never triggers in practice at fc_similarity T=240,
    // _AXIS0_BLOCK=32, but kept generic).
    for (int t = tid; t < T; t += blockDim.x) {
        smem_mr[t] = mr[t];
    }
    __syncthreads();

    if (j >= K) return;
    const float mc_j = mc[j];

    // Pass 1: fused divide + fp64 mean accumulation.
    double sum = 0.0;
    for (int t = 0; t < T; t++) {
        const size_t idx = (size_t)t * (size_t)K + (size_t)j;
        const float v = x[idx] / (smem_mr[t] * mc_j);
        x[idx] = v;
        sum += (double)v;
    }
    const float mean = (float)(sum / (double)T);

    // Pass 2: in-place demean + fp64 SS (identical to axis-0 baseline).
    double ss = 0.0;
    for (int t = 0; t < T; t++) {
        const size_t idx = (size_t)t * (size_t)K + (size_t)j;
        const float v = x[idx];
        const float diff = v - mean;
        x[idx] = diff;
        ss += (double)diff * (double)diff;
    }
    mag[j] = sqrtf((float)ss);
}
"""


_axis0_div_kernel = cp.RawKernel(_axis0_div_kernel_src,
                                 "fused_div_demean_norm_axis0")


# ─────────────────────────────────────────────────────────────────────
# Per-cell divide-by-two-broadcasts + NaN→0 clamp + slice write.
#
# Fuses the tail of compute_FC_simi_block_gpu's inner loop:
#
#     block_d /= mag_b[:, None]
#     block_d /= mag_a_row
#     FC_simi_block_d[b_start:b_end, :] = block_d
#     # and at the end: cp.nan_to_num(FC_simi_block_d, copy=False, nan=0)
#
# into a single grid-strided kernel: read block, divide by mr[i]*mc[j],
# clamp non-finite to 0, write to ``out``. Trades 4 kernel launches +
# 4 HBM passes for 1 launch + 1 read + 1 write.
#
# The clamp matches ``cp.nan_to_num(..., nan=0, posinf=0, neginf=0)``.
# ``isfinite`` returns false on NaN and ±inf; both map to 0. See the
# CPU path's analogous defensive comment in
# :func:`fc_similarity.compute_FC_simi_block` — only 0/0=NaN is
# mathematically reachable; the inf branches are kept for symmetry.
# ─────────────────────────────────────────────────────────────────────
_div_nanclamp_kernel_src = r"""
extern "C" __global__
void fused_div_nanclamp(
        const float* __restrict__ x,    // (M, K) row-major source
        const float* __restrict__ mr,   // (M,) row-axis divisor
        const float* __restrict__ mc,   // (K,) col-axis divisor
        float* __restrict__ y,          // (M, K) row-major dest
        int M, int K) {
    const long long total = (long long)M * (long long)K;
    long long idx = (long long)blockIdx.x * (long long)blockDim.x
                  + (long long)threadIdx.x;
    const long long stride = (long long)gridDim.x * (long long)blockDim.x;
    for (; idx < total; idx += stride) {
        const int i = (int)(idx / (long long)K);
        const int j = (int)(idx - (long long)i * (long long)K);
        const float v = x[idx] / (mr[i] * mc[j]);
        y[idx] = isfinite(v) ? v : 0.0f;
    }
}
"""


_div_nanclamp_kernel = cp.RawKernel(_div_nanclamp_kernel_src,
                                    "fused_div_nanclamp")
_DIV_NANCLAMP_BLOCK = 256


# ─────────────────────────────────────────────────────────────────────
# Python entry points — mirror the original ``_demean_norm_*`` API:
# mutate ``x`` in place; return ``mag`` as a fresh (K,) fp32 device
# array (1-D, not reshaped — caller handles broadcast shape).
# ─────────────────────────────────────────────────────────────────────
def fused_demean_norm_columns_cupy(x_d: cp.ndarray) -> cp.ndarray:
    """Fused replacement for ``_demean_norm_columns``.

    Parameters
    ----------
    x_d : (T, K) fp32 row-major cupy.ndarray
        Mutated in place to be column-wise demeaned.

    Returns
    -------
    mag : (K,) fp32 cupy.ndarray
        L2 norms of each demeaned column.
    """
    assert x_d.dtype == cp.float32, f"x_d must be fp32, got {x_d.dtype}"
    assert x_d.ndim == 2, f"x_d must be 2D, got {x_d.shape}"
    if not x_d.flags.c_contiguous:
        x_d = cp.ascontiguousarray(x_d)
    T, K = int(x_d.shape[0]), int(x_d.shape[1])
    mag = cp.empty((K,), dtype=cp.float32)
    grid = ((K + _AXIS0_BLOCK - 1) // _AXIS0_BLOCK,)
    block = (_AXIS0_BLOCK,)
    _axis0_kernel(grid, block, (x_d, mag, T, K))
    return mag


def fused_demean_norm_rows_cupy(x_d: cp.ndarray) -> cp.ndarray:
    """Fused replacement for ``_demean_norm_rows``.

    Parameters
    ----------
    x_d : (K, T) fp32 row-major cupy.ndarray
        Mutated in place to be row-wise demeaned.

    Returns
    -------
    mag : (K,) fp32 cupy.ndarray
        L2 norms of each demeaned row.
    """
    assert x_d.dtype == cp.float32, f"x_d must be fp32, got {x_d.dtype}"
    assert x_d.ndim == 2, f"x_d must be 2D, got {x_d.shape}"
    if not x_d.flags.c_contiguous:
        x_d = cp.ascontiguousarray(x_d)
    K, T = int(x_d.shape[0]), int(x_d.shape[1])
    mag = cp.empty((K,), dtype=cp.float32)
    grid = (K,)
    block = (_AXIS1_BLOCK,)
    # ``shared_mem`` arg = bytes of dynamic shared memory: B * sizeof(double).
    shared_bytes = _AXIS1_BLOCK * 8
    _axis1_kernel(grid, block, (x_d, mag, K, T), shared_mem=shared_bytes)
    return mag


def fused_div_demean_norm_columns_cupy(
        x_d: cp.ndarray,
        mr_d: cp.ndarray,
        mc_d: cp.ndarray) -> cp.ndarray:
    """Fused divide-by-(mr⊗mc) + column demean + L2 norm.

    Equivalent to (CPU ``_scale_demean_norm_inplace``):

        x_d /= mr_d[:, None]
        x_d /= mc_d[None, :]
        mag = fused_demean_norm_columns_cupy(x_d)

    but one kernel launch, one read + one write per cell across two
    passes. See the kernel docstring for the precision contract.

    Parameters
    ----------
    x_d : (T, K) fp32 row-major cupy.ndarray
        Mutated in place: divided by ``mr[:, None] * mc[None, :]``,
        then column-demeaned.
    mr_d : (T,) fp32 cupy.ndarray
        Row-axis divisor.
    mc_d : (K,) fp32 cupy.ndarray
        Column-axis divisor.

    Returns
    -------
    mag : (K,) fp32 cupy.ndarray
        L2 norms of each demeaned column.
    """
    assert x_d.dtype == cp.float32, f"x_d must be fp32, got {x_d.dtype}"
    assert mr_d.dtype == cp.float32, f"mr_d must be fp32, got {mr_d.dtype}"
    assert mc_d.dtype == cp.float32, f"mc_d must be fp32, got {mc_d.dtype}"
    assert x_d.ndim == 2, f"x_d must be 2D, got {x_d.shape}"
    if not x_d.flags.c_contiguous:
        x_d = cp.ascontiguousarray(x_d)
    T, K = int(x_d.shape[0]), int(x_d.shape[1])
    assert mr_d.size == T, f"mr_d shape mismatch: {mr_d.shape} vs T={T}"
    assert mc_d.size == K, f"mc_d shape mismatch: {mc_d.shape} vs K={K}"
    mr_flat = mr_d.reshape(-1)
    mc_flat = mc_d.reshape(-1)
    mag = cp.empty((K,), dtype=cp.float32)
    grid = ((K + _AXIS0_BLOCK - 1) // _AXIS0_BLOCK,)
    block = (_AXIS0_BLOCK,)
    # Shared mem holds mr[] — T floats = 4*T bytes.
    shared_bytes = T * 4
    _axis0_div_kernel(grid, block, (x_d, mr_flat, mc_flat, mag, T, K),
                      shared_mem=shared_bytes)
    return mag


def fused_div_nanclamp_cupy(
        x_d: cp.ndarray,
        mr_d: cp.ndarray,
        mc_d: cp.ndarray,
        out_d: cp.ndarray) -> None:
    """Fused per-cell ``out = nan_to_num(x / (mr[:, None] * mc[None, :]))``.

    Parameters
    ----------
    x_d : (M, K) fp32 row-major cupy.ndarray (source, unchanged).
    mr_d : (M,) fp32 row-axis divisor.
    mc_d : (K,) fp32 col-axis divisor.
    out_d : (M, K) fp32 row-major cupy.ndarray (destination, written).

    Notes
    -----
    ``out_d`` MUST be a contiguous fp32 view at least (M, K). The
    common caller pattern in fc_similarity_gpu writes into a
    row-major slice ``FC_simi_block_d[b_start:b_end, :]`` — that
    slice IS contiguous because the parent array's K axis is the
    innermost (row stride = K, slice has full K).
    """
    assert x_d.dtype == cp.float32, f"x_d must be fp32, got {x_d.dtype}"
    assert out_d.dtype == cp.float32, f"out_d must be fp32, got {out_d.dtype}"
    assert mr_d.dtype == cp.float32, f"mr_d must be fp32, got {mr_d.dtype}"
    assert mc_d.dtype == cp.float32, f"mc_d must be fp32, got {mc_d.dtype}"
    assert x_d.shape == out_d.shape, (
        f"shape mismatch: x={x_d.shape} out={out_d.shape}")
    if not x_d.flags.c_contiguous:
        x_d = cp.ascontiguousarray(x_d)
    assert out_d.flags.c_contiguous, "out_d must be contiguous"
    M, K = int(x_d.shape[0]), int(x_d.shape[1])
    assert mr_d.size == M, f"mr_d shape mismatch: {mr_d.shape} vs M={M}"
    assert mc_d.size == K, f"mc_d shape mismatch: {mc_d.shape} vs K={K}"
    total = M * K
    grid_size = min((total + _DIV_NANCLAMP_BLOCK - 1) // _DIV_NANCLAMP_BLOCK,
                    65535)
    grid = (grid_size,)
    block = (_DIV_NANCLAMP_BLOCK,)
    _div_nanclamp_kernel(grid, block,
                         (x_d, mr_d.reshape(-1), mc_d.reshape(-1),
                          out_d, M, K))
