"""_kernels_gpu.py

GPU kernels for the vMF init-params supercall — CuPy counterparts to
the four numba kernels in :mod:`._kernels`.

    zero_mw_and_detect_nonzero_cupy(profile, medial_mask) -> keep
        Zero MW rows in place + return the per-row any-nonzero mask.
        Pure CuPy primitives (one fancy-index write + one row-wise reduce).

    demean_l2norm_inplace_cupy(profile, keep)
        Per-row demean + L2-unit-norm, in place. RawKernel — one block
        per row, fp64 sum/sumsq reduction via shared mem, two passes
        over D. Skips rows where ``keep`` is False (matches CPU).

    groupsum_cupy(profile, labels, L) -> mtc_DxL
        ``mtc[d, l] = Σ_{i: labels[i]==l+1} profile[i, d]``. Implemented
        as ``profile.T @ one_hot(labels)`` — one cuBLAS GEMM at the
        profile's dtype (fp64 in production), no atomicAdd needed.

    epsil_input_cupy(inner, labels) -> host scalar
        ``Σ_{i: labels[i]>0} inner[i, labels[i]-1]`` via gather + sum.

cupy is imported at module top — this file is only loaded via the
lazy import inside :mod:`.ini_params_gpu`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import cupy as cp


_DEMEAN_BLOCK = 256
_DEMEAN_BLOCK_MAX = 256


_demean_l2norm_kernel = cp.RawKernel(r"""
extern "C" __global__
void demean_l2norm_rows_inplace(double* __restrict__ profile,
                                 const unsigned char* __restrict__ keep,
                                 int N, int D) {
    // One thread block per row i. Two passes over D:
    //   pass 1: sum -> mean (fp64 block-reduce via shared mem)
    //   pass 2: write (v - mean), accumulate sumsq -> norm
    //   pass 3: multiply by 1/norm (or 1.0 if zero-variance row)
    //
    // ``keep`` gates the entire row: rows whose pre-check rejected
    // them (MW row OR all-zero row) are left unchanged; matches the
    // CPU kernel's ``if not keep_mask_N[i]: continue``.

    const int i = blockIdx.x;
    if (i >= N) return;
    if (!keep[i]) return;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    double* row = profile + (size_t)i * (size_t)D;

    double local_sum = 0.0;
    for (int j = tid; j < D; j += bs) local_sum += row[j];
    __shared__ double s_red[256];
    s_red[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    __shared__ double s_mean;
    if (tid == 0) s_mean = s_red[0] / (double)D;
    __syncthreads();
    const double mean_d = s_mean;

    // pass 2: write centered + accumulate sumsq.
    double local_ss = 0.0;
    for (int j = tid; j < D; j += bs) {
        double v = row[j] - mean_d;
        row[j] = v;
        local_ss += v * v;
    }
    s_red[tid] = local_ss;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    __shared__ double s_inv_norm;
    if (tid == 0) {
        double ss = s_red[0];
        s_inv_norm = (ss > 0.0) ? (1.0 / sqrt(ss)) : 1.0;
    }
    __syncthreads();
    const double inv_n = s_inv_norm;

    // pass 3: scale.
    for (int j = tid; j < D; j += bs) row[j] *= inv_n;
}
""", "demean_l2norm_rows_inplace")


def zero_mw_and_detect_nonzero_cupy(profile_dev: cp.ndarray,
                                     medial_mask_dev: cp.ndarray) -> cp.ndarray:
    """Zero MW rows in place; return (N,) keep mask.

    ``profile_dev`` is (N, D), modified in place. ``medial_mask_dev``
    is (N,) bool. Returns a (N,) bool device array marking rows that
    are non-MW and have at least one non-zero entry.
    """
    profile_dev[medial_mask_dev, :] = 0
    # ``any`` is a single reduction kernel.
    return (profile_dev != 0).any(axis=1)


def demean_l2norm_inplace_cupy(profile_dev: cp.ndarray,
                                keep_dev: cp.ndarray) -> None:
    """Per-row demean + L2-unit-norm of ``profile_dev``, in place,
    skipping rows where ``keep_dev`` is False.
    """
    if profile_dev.dtype != cp.float64:
        raise ValueError(
            f"demean_l2norm: profile must be fp64; got {profile_dev.dtype}"
        )
    if not profile_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("demean_l2norm: profile must be C-contiguous")
    N, D = profile_dev.shape
    # ``keep`` arrives as bool; the kernel reads bytes, so contiguity matters.
    keep_u8 = keep_dev.astype(cp.uint8) if keep_dev.dtype != cp.uint8 else keep_dev
    if not keep_u8.flags["C_CONTIGUOUS"]:
        keep_u8 = cp.ascontiguousarray(keep_u8)
    block = _DEMEAN_BLOCK
    if block <= 0 or (block & (block - 1)) != 0 or block > _DEMEAN_BLOCK_MAX:
        raise ValueError(
            f"demean_l2norm: block size must be power-of-2 <= "
            f"{_DEMEAN_BLOCK_MAX}; got {block}"
        )
    _demean_l2norm_kernel((N,), (block,),
                           (profile_dev, keep_u8, cp.int32(N), cp.int32(D)))


def groupsum_cupy(profile_dev: cp.ndarray, labels_dev: cp.ndarray,
                  L: int, mtc_DxL_dev: cp.ndarray) -> None:
    """``mtc[d, l] = Σ_{i: labels[i]==l+1} profile[i, d]``.

    Materialises a (N, L) one-hot at the profile's dtype and uses a
    single cuBLAS GEMM. For (N=80k, D=1175, L=400) the one-hot is
    256 MB fp64 — within budget, and the GEMM (~37 GFLOP) finishes in
    a few tens of ms on cuBLAS dgemm.

    Writes the result into the caller-provided ``mtc_DxL_dev`` buffer
    so the supercall owns the lifetime of the (D, L) output.
    """
    N = labels_dev.shape[0]
    dtype = profile_dev.dtype
    one_hot = cp.zeros((N, L), dtype=dtype)
    active = labels_dev > 0
    # ``labels[active] - 1`` is a (k,) device int64 → fancy-index columns.
    rows_idx = cp.flatnonzero(active)
    cols_idx = labels_dev[active] - 1
    one_hot[rows_idx, cols_idx] = dtype.type(1.0)   # dtype is np.dtype here.
    # profile.T (D, N) @ one_hot (N, L) -> (D, L). cp.matmul writes into ``out=``.
    cp.matmul(profile_dev.T, one_hot, out=mtc_DxL_dev)


def epsil_input_cupy(inner_dev: cp.ndarray, labels_dev: cp.ndarray) -> float:
    """``Σ_{i: labels[i]>0} inner[i, labels[i]-1]`` -> host fp64 scalar.

    Single D2H sync at the end. The gather degenerates to an empty
    selection if every label is 0 (the all-MW edge case); summing an
    empty fp64 array on cupy returns 0.0, so no early-out is needed.
    """
    active = labels_dev > 0
    rows_idx = cp.flatnonzero(active)
    cols_idx = labels_dev[active] - 1
    gathered = inner_dev[rows_idx, cols_idx]
    return float(gathered.astype(cp.float64).sum().item())
