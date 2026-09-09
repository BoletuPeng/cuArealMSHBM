"""_kernels_fused_gpu.py

Fused demean + L2-norm RawKernels for fc_similarity_gpu. Each entry
point is one launch: pass 1 accumulates the fp64 mean over the reduction
axis, pass 2 demeans the fp32 buffer in place and writes
``mag[k] = sqrtf(SS)`` from an fp64 sum of squares.

The mean is cast to fp32 *before* the subtract, so a stored cell is
bit-identical to ``x - float(mean64)``, and the pre-divide variants
divide in fp32 to match ``fc_similarity._scale_demean_norm_inplace``.
Every demean entry point mutates ``x_d`` in place — so it requires a
C-contiguous ``x_d`` and rejects anything else — and returns the
``(K,)`` fp32 magnitudes.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import cupy as cp


# Threads-per-block default for the axis-0 kernels; every value is
# bit-neutral, so this is a pure occupancy knob.
_AXIS0_BLOCK = 256


# ─────────────────────────────────────────────────────────────────────
# Axis-0 reduction: (T, K) row-major, reduce over T ("demean each
# column"). One thread per column j, so a warp of 32 consecutive j
# reads 32 consecutive fp32 per t-step. No shared memory.
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


# ─────────────────────────────────────────────────────────────────────
# Axis-1 reduction: (K, T) row-major, reduce over T. One block per row
# k; threads cooperate via a shared-mem tree reduction over the T axis.
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
# Axis-0 divide + demean + norm, one HBM write instead of two.
#
# Pass 1 absorbs the caller's two broadcast divides and does NOT store
# the divided value: fp32 division is deterministic, so pass 2's
# recompute reproduces the exact bits a store would have round-tripped.
#
# Both passes are hand-blocked into DIV1W_PF-element register batches so
# that a batch's loads all issue before any is consumed — the only way
# to keep several memory requests per thread in flight (``#pragma
# unroll`` alone does not). Consumption still runs in strict ascending
# t, so the fp64 add tree and every stored bit are unchanged.
# ─────────────────────────────────────────────────────────────────────
_axis0_div1w_kernel_src = r"""
#define DIV1W_PF 8
extern "C" __global__
void fused_div_demean_norm_axis0_1w(
        float* __restrict__ x,         // (T, K) row-major; mutated in place
        const float* __restrict__ mr,  // (T,) divisor for axis 0 (rows)
        const float* __restrict__ mc,  // (K,) divisor for axis 1 (cols)
        float* __restrict__ mag,       // (K,) output
        int T, int K) {
    extern __shared__ float smem_mr[];
    const int tid = threadIdx.x;
    const int j = blockIdx.x * blockDim.x + tid;

    // Cooperative, coalesced staging of mr[] — every thread of the
    // block walks the whole T axis twice, so the row divisor is read
    // 2*T times per thread out of shared memory instead of L1.
    for (int t = tid; t < T; t += blockDim.x) {
        smem_mr[t] = mr[t];
    }
    __syncthreads();

    if (j >= K) return;
    const float mc_j = mc[j];

    // Pass 1: divide + fp64 mean accumulation. NO store.
    double sum = 0.0;
    for (int t0 = 0; t0 < T; t0 += DIV1W_PF) {
        float buf[DIV1W_PF];
#pragma unroll
        for (int p = 0; p < DIV1W_PF; p++) {
            const int t = t0 + p;
            buf[p] = (t < T) ? x[(size_t)t * (size_t)K + (size_t)j] : 0.f;
        }
#pragma unroll
        for (int p = 0; p < DIV1W_PF; p++) {
            const int t = t0 + p;
            if (t < T) sum += (double)(buf[p] / (smem_mr[t] * mc_j));
        }
    }
    const float mean = (float)(sum / (double)T);

    // Pass 2: re-divide (same bits), demean, store, fp64 SS.
    double ss = 0.0;
    for (int t0 = 0; t0 < T; t0 += DIV1W_PF) {
        float buf[DIV1W_PF];
#pragma unroll
        for (int p = 0; p < DIV1W_PF; p++) {
            const int t = t0 + p;
            buf[p] = (t < T) ? x[(size_t)t * (size_t)K + (size_t)j] : 0.f;
        }
#pragma unroll
        for (int p = 0; p < DIV1W_PF; p++) {
            const int t = t0 + p;
            if (t < T) {
                const float diff = buf[p] / (smem_mr[t] * mc_j) - mean;
                x[(size_t)t * (size_t)K + (size_t)j] = diff;
                ss += (double)diff * (double)diff;
            }
        }
    }
    mag[j] = sqrtf((float)ss);
}
"""


_axis0_div1w_kernel = cp.RawKernel(_axis0_div1w_kernel_src,
                                   "fused_div_demean_norm_axis0_1w")

# ``_axis0_div1w_kernel`` stages ``mr`` (T floats) in dynamic shared
# memory; staying inside the 48 KB no-opt-in budget avoids ever needing
# ``max_dynamic_shared_size_bytes``. Over the cap the launcher raises
# rather than letting the driver report a bare CUDA_ERROR_INVALID_VALUE.
_AXIS0_DIV1W_SHMEM_LIMIT = 48 * 1024
_AXIS0_DIV1W_MAX_T = _AXIS0_DIV1W_SHMEM_LIMIT // 4


# ─────────────────────────────────────────────────────────────────────
# Per-cell divide-by-two-broadcasts + NaN→0 clamp + slice write, fused
# into one grid-strided kernel.
#
# The clamp matches ``cp.nan_to_num(..., nan=0, posinf=0, neginf=0)``:
# ``isfinite`` is false on NaN and ±inf and both map to 0. Only 0/0=NaN
# is mathematically reachable; the inf branches are kept for symmetry
# with the CPU path.
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
# Python entry points. ``mag`` comes back 1-D (K,); the caller handles
# the broadcast shape.
# ─────────────────────────────────────────────────────────────────────
def _require_c_contiguous(x_d: cp.ndarray) -> None:
    """The in-place entry points index ``x_d`` with a flat row-major
    offset and write through it, so a non-C-contiguous buffer cannot be
    demeaned in place. Rejected rather than copied: a copy would return
    correct magnitudes while silently leaving the caller's array
    untouched."""
    if not x_d.flags.c_contiguous:
        raise ValueError(
            "x_d must be C-contiguous; this kernel mutates it in place.")


def fused_demean_norm_columns_cupy(x_d: cp.ndarray,
                                   *,
                                   block: int = _AXIS0_BLOCK) -> cp.ndarray:
    """Column-wise demean + L2 norm of a (T, K) fp32 device array.

    ``x_d`` is demeaned in place; ``block`` is bit-neutral (one thread
    per column). Returns the (K,) fp32 norms.
    """
    assert x_d.dtype == cp.float32, f"x_d must be fp32, got {x_d.dtype}"
    assert x_d.ndim == 2, f"x_d must be 2D, got {x_d.shape}"
    _require_c_contiguous(x_d)
    T, K = int(x_d.shape[0]), int(x_d.shape[1])
    mag = cp.empty((K,), dtype=cp.float32)
    _axis0_kernel(((K + block - 1) // block,), (block,), (x_d, mag, T, K))
    return mag


def fused_demean_norm_rows_cupy(x_d: cp.ndarray) -> cp.ndarray:
    """Row-wise demean + L2 norm of a (K, T) fp32 device array.

    ``x_d`` is demeaned in place; returns the (K,) fp32 row norms.
    """
    assert x_d.dtype == cp.float32, f"x_d must be fp32, got {x_d.dtype}"
    assert x_d.ndim == 2, f"x_d must be 2D, got {x_d.shape}"
    _require_c_contiguous(x_d)
    K, T = int(x_d.shape[0]), int(x_d.shape[1])
    mag = cp.empty((K,), dtype=cp.float32)
    grid = (K,)
    block = (_AXIS1_BLOCK,)
    # Dynamic shared memory: one fp64 accumulator slot per thread.
    shared_bytes = _AXIS1_BLOCK * 8
    _axis1_kernel(grid, block, (x_d, mag, K, T), shared_mem=shared_bytes)
    return mag


def fused_div_demean_norm_columns_cupy(
        x_d: cp.ndarray,
        mr_d: cp.ndarray,
        mc_d: cp.ndarray,
        *,
        block: int = _AXIS0_BLOCK) -> cp.ndarray:
    """Fused divide-by-(mr⊗mc) + column demean + L2 norm.

    Equivalent to the CPU ``_scale_demean_norm_inplace``::

        x_d /= mr_d[:, None]
        x_d /= mc_d[None, :]
        mag = fused_demean_norm_columns_cupy(x_d)

    ``x_d`` (T, K) is mutated in place. ``mr_d`` (T,) is staged whole in
    dynamic shared memory, so ``T`` is capped at ``_AXIS0_DIV1W_MAX_T``.
    ``block`` is bit-neutral. Returns the (K,) fp32 column norms.
    """
    assert x_d.dtype == cp.float32, f"x_d must be fp32, got {x_d.dtype}"
    assert mr_d.dtype == cp.float32, f"mr_d must be fp32, got {mr_d.dtype}"
    assert mc_d.dtype == cp.float32, f"mc_d must be fp32, got {mc_d.dtype}"
    assert x_d.ndim == 2, f"x_d must be 2D, got {x_d.shape}"
    _require_c_contiguous(x_d)
    T, K = int(x_d.shape[0]), int(x_d.shape[1])
    assert mr_d.size == T, f"mr_d shape mismatch: {mr_d.shape} vs T={T}"
    assert mc_d.size == K, f"mc_d shape mismatch: {mc_d.shape} vs K={K}"
    if T > _AXIS0_DIV1W_MAX_T:
        raise ValueError(
            f"T={T} exceeds the {_AXIS0_DIV1W_MAX_T} rows this kernel can "
            f"stage in dynamic shared memory ({T * 4} B needed, "
            f"{_AXIS0_DIV1W_SHMEM_LIMIT} B available).")
    mag = cp.empty((K,), dtype=cp.float32)
    _axis0_div1w_kernel(((K + block - 1) // block,), (block,),
                        (x_d, mr_d.reshape(-1), mc_d.reshape(-1), mag, T, K),
                        shared_mem=T * 4)
    return mag


def fused_div_nanclamp_cupy(
        x_d: cp.ndarray,
        mr_d: cp.ndarray,
        mc_d: cp.ndarray,
        out_d: cp.ndarray) -> None:
    """Fused per-cell ``out = nan_to_num(x / (mr[:, None] * mc[None, :]))``.

    ``x_d`` (M, K) is read, ``out_d`` (M, K) written; ``out_d`` must be
    C-contiguous, which a row-major row-slice of a larger array is.
    Aliasing ``x_d`` and ``out_d`` is safe: every output cell is a pure
    function of the input cell at the SAME index plus two broadcast
    vectors, and the grid-stride loop touches each index once.
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
