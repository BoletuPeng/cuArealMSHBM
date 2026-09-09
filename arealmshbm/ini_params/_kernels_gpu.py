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

    build_parcel_csr_cupy(labels, L) -> (offsets, rows)
        Rows grouped by parcel, ascending within each parcel.

    groupsum_csr_cupy(profile, offsets, rows, L, OUT mtc)
        ``mtc[d, l] = Σ_{i: labels[i]==l+1} profile[i, d]`` in the CPU
        kernel's exact fp64 summation order — no dense one-hot, no GEMM
        (see the section comment further down).

    colnorm_scale_cupy(mtc)
        In-place column L2-renorm in numpy's ``sum(axis=0)`` order.

    epsil_input_rowdot_cupy(profile, mtc, labels) -> host scalar
        Fused row-dot + deterministic two-stage reduction; replaces
        ``inner = profile @ mtc`` plus the gather-sum.

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


# ─────────────────────────────────────────────────────────────────────
# Ordered fp64 reductions — CPU-bit-exact replacements for the dense
# one-hot dgemm (`groupsum`) and the `profile @ mtc` dgemm (`epsil`).
#
# Both kernels fix the fp64 summation order explicitly, because the
# saved ``mtc`` is compared bit-for-bit against the numba CPU path:
#
#   * ``groupsum_csr``  sums each parcel's rows in ASCENDING row index,
#     exactly like ``_kernels._groupsum_kernel``'s ``for i in range(N)``.
#   * ``colnorm_scale`` sums ``mtc[d, l]**2`` over ASCENDING d, exactly
#     like numpy's ``(mtc * mtc).sum(axis=0)`` on a C-contig (D, L)
#     array — numpy reduces the outer axis by accumulating row-by-row
#     into the (L,) output buffer, i.e. sequentially in d, NOT pairwise
#     (verified empirically against a serial loop). That holds for
#     L >= 2 (checked bit-exact up to D = 100000); the degenerate
#     L == 1 case is contiguous along d, where numpy does go pairwise
#     and the two part ways by ~10 ulps at D = 4096.
#     Production L is the parcel count (300-400), so the kernel keeps
#     the CPU-matching order.
#
# FMA contraction is disabled by hand (``__dmul_rn`` / ``__dadd_rn``)
# wherever the CPU/numpy reference rounds the product before the add.
# ─────────────────────────────────────────────────────────────────────

_GROUPSUM_BLOCK = 256
_COLNORM_BLOCK = 64
_ROWDOT_BLOCK = 256
_EPSIL_CHUNKS = 256


_groupsum_csr_kernel = cp.RawKernel(r"""
extern "C" __global__
void groupsum_csr_fp64(const double* __restrict__ profile,
                       const int* __restrict__ offsets,
                       const int* __restrict__ rows,
                       double* __restrict__ mtc,
                       int N, int D, int L) {
    // mtc[d, l] = sum_{i in parcel l, ascending i} profile[i, d].
    //
    // blockIdx.y = parcel l; blockIdx.x * blockDim.x + threadIdx.x = d.
    // Threads in a warp vary d, so each iteration of the row loop is a
    // fully coalesced 8-byte-per-lane read of one profile row segment.
    // Each (d, l) output is owned by exactly one thread -> no atomics,
    // and the accumulation order is the CSR order, which the builder
    // guarantees is ascending row index within the parcel.
    const int l = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= L || d >= D) return;
    const int beg = offsets[l];
    const int end = offsets[l + 1];
    double acc = 0.0;
    for (int k = beg; k < end; ++k) {
        const long long i = (long long)rows[k];
        acc += profile[i * (long long)D + (long long)d];
    }
    mtc[(long long)d * (long long)L + (long long)l] = acc;
}
""", "groupsum_csr_fp64")


_colnorm_scale_kernel = cp.RawKernel(r"""
extern "C" __global__
void colnorm_scale_fp64(double* __restrict__ mtc, int D, int L) {
    // One thread per column l. Two sequential passes over d:
    //   acc = sum_d rn(mtc[d,l] * mtc[d,l])      (ascending d)
    //   n   = sqrt(acc); if n == 0 -> 1.0
    //   mtc[d,l] = mtc[d,l] / n                  (IEEE round-to-nearest)
    // Mirrors numpy's
    //   col = sqrt((mtc*mtc).sum(axis=0)); col = where(col==0, 1, col)
    //   mtc /= col
    // bit-for-bit for L >= 2 (at L == 1 numpy's reduction axis is
    // contiguous and it switches to pairwise summation instead):
    // __dmul_rn/__dadd_rn block FMA contraction (numpy
    // materialises the squared temp before summing), CUDA's double
    // sqrt and __ddiv_rn are IEEE correctly rounded like numpy's.
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= L) return;
    double acc = 0.0;
    for (int d = 0; d < D; ++d) {
        const double v = mtc[(long long)d * (long long)L + (long long)l];
        acc = __dadd_rn(acc, __dmul_rn(v, v));
    }
    double nrm = sqrt(acc);
    if (nrm == 0.0) nrm = 1.0;
    for (int d = 0; d < D; ++d) {
        const long long o = (long long)d * (long long)L + (long long)l;
        mtc[o] = __ddiv_rn(mtc[o], nrm);
    }
}
""", "colnorm_scale_fp64")


_rowdot_kernel = cp.RawKernel(r"""
extern "C" __global__
void rowdot_labelled_fp64(const double* __restrict__ profile,
                          const double* __restrict__ mtc_T,
                          const long long* __restrict__ labels,
                          double* __restrict__ rowout,
                          int N, int D, int L) {
    // rowout[i] = dot(profile[i, :], mtc[:, labels[i]-1]) for
    // labels[i] > 0, else 0. ``mtc_T`` is the (L, D) C-contig
    // transpose of mtc so both operands stream coalesced.
    //
    // One block per row; blockDim.x threads stride over D, then a
    // fixed-shape shared-memory tree reduction. Block size is a
    // compile-time-constant 256 at every call site, so the reduction
    // tree (and therefore the fp64 result) is identical run to run.
    const int i = blockIdx.x;
    if (i >= N) return;
    __shared__ double red[256];
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const long long lab = labels[i];
    if (lab <= 0) {
        if (tid == 0) rowout[i] = 0.0;
        return;
    }
    const long long col = lab - 1;
    const double* prow = profile + (long long)i * (long long)D;
    const double* mrow = mtc_T + col * (long long)D;
    double acc = 0.0;
    for (int d = tid; d < D; d += bs) acc += prow[d] * mrow[d];
    red[tid] = acc;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) rowout[i] = red[0];
}
""", "rowdot_labelled_fp64")


_chunk_sum_kernel = cp.RawKernel(r"""
extern "C" __global__
void chunk_sum_fp64(const double* __restrict__ x,
                    double* __restrict__ out,
                    int N, int nchunks) {
    // Stage-2 of the epsil reduction: chunk b sums x[beg:end)
    // sequentially in fp64. Chunk boundaries depend only on (N,
    // nchunks), both fixed, so the partial set is deterministic; the
    // host then folds the nchunks partials in ascending order.
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nchunks) return;
    const long long chunk = ((long long)N + nchunks - 1) / nchunks;
    long long beg = (long long)b * chunk;
    long long end = beg + chunk;
    if (end > (long long)N) end = (long long)N;
    double acc = 0.0;
    for (long long i = beg; i < end; ++i) acc += x[i];
    out[b] = acc;
}
""", "chunk_sum_fp64")


def build_parcel_csr_cupy(labels_dev: cp.ndarray, L: int):
    """Group row indices by parcel, ascending within each parcel.

    Returns ``(offsets, rows)`` where ``offsets`` is (L+1,) int32 and
    ``rows`` is (K,) int32 holding, for parcel ``l``, the row indices
    ``rows[offsets[l]:offsets[l+1]]`` in ASCENDING order — the exact
    order the numba CPU ``_groupsum_kernel`` accumulates them in.

    Ascending order is obtained by sorting the composite key
    ``(label - 1) * N + row`` rather than relying on a stable argsort,
    so the guarantee does not depend on cupy's sort stability.
    """
    if labels_dev.ndim != 1:
        raise ValueError(
            f"build_parcel_csr_cupy: labels must be 1-D; got shape "
            f"{labels_dev.shape}"
        )
    if labels_dev.dtype != cp.int64:
        raise ValueError(
            f"build_parcel_csr_cupy: labels must be int64; got "
            f"{labels_dev.dtype}"
        )
    L = int(L)
    if L <= 0:
        raise ValueError(f"build_parcel_csr_cupy: L must be > 0; got {L}")
    N = int(labels_dev.shape[0])
    active = labels_dev > 0
    idx = cp.flatnonzero(active)                    # int64, ascending
    lab0 = labels_dev[active] - 1                   # int64 in [0, L)
    key = lab0 * cp.int64(N) + idx
    order = cp.argsort(key)
    rows = idx[order].astype(cp.int32)
    counts = cp.bincount(lab0, minlength=L)
    offsets = cp.zeros(L + 1, dtype=cp.int32)
    offsets[1:] = cp.cumsum(counts).astype(cp.int32)
    return offsets, rows


def groupsum_csr_cupy(profile_dev: cp.ndarray, offsets_dev: cp.ndarray,
                      rows_dev: cp.ndarray, L: int,
                      mtc_DxL_dev: cp.ndarray) -> None:
    """``mtc[d, l] = Σ_{i ∈ parcel l, ascending i} profile[i, d]``.

    Bit-exact against :func:`._kernels._groupsum_kernel` for the same
    fp64 input: identical summation order, plain adds only (no FMA to
    contract). The backends' saved ``mtc`` still differ by ~1e-15 —
    that comes from ``demean_l2norm_inplace``, not from here.

    Replaces the dense one-hot dgemm — no (N, L) fp64 one-hot is
    materialised (256 MB at N=80k, L=400) and the arithmetic drops
    from O(N·L·D) to O(N·D).
    """
    if profile_dev.dtype != cp.float64 or mtc_DxL_dev.dtype != cp.float64:
        raise ValueError(
            f"groupsum_csr_cupy: profile and mtc must be fp64; got "
            f"{profile_dev.dtype} / {mtc_DxL_dev.dtype}"
        )
    if not profile_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("groupsum_csr_cupy: profile must be C-contiguous")
    if not mtc_DxL_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("groupsum_csr_cupy: mtc must be C-contiguous")
    if offsets_dev.dtype != cp.int32 or rows_dev.dtype != cp.int32:
        raise ValueError(
            f"groupsum_csr_cupy: CSR arrays must be int32; got "
            f"{offsets_dev.dtype} / {rows_dev.dtype}"
        )
    N, D = profile_dev.shape
    L = int(L)
    if mtc_DxL_dev.shape != (D, L):
        raise ValueError(
            f"groupsum_csr_cupy: mtc must be (D, L) = ({D}, {L}); got "
            f"{mtc_DxL_dev.shape}"
        )
    if offsets_dev.shape != (L + 1,):
        raise ValueError(
            f"groupsum_csr_cupy: offsets must be (L+1,) = ({L + 1},); "
            f"got {offsets_dev.shape}"
        )
    grid = ((D + _GROUPSUM_BLOCK - 1) // _GROUPSUM_BLOCK, L)
    _groupsum_csr_kernel(
        grid, (_GROUPSUM_BLOCK,),
        (profile_dev, offsets_dev, rows_dev, mtc_DxL_dev,
         cp.int32(N), cp.int32(D), cp.int32(L)),
    )


def colnorm_scale_cupy(mtc_DxL_dev: cp.ndarray) -> None:
    """In-place column L2-renormalisation of ``mtc``, numpy-ordered.

    Equivalent to::

        col = np.sqrt((mtc * mtc).sum(axis=0, keepdims=True))
        col = np.where(col == 0, 1.0, col)
        mtc /= col

    down to the bit **for L >= 2** — see the kernel comment for why the
    ordering and the ``__dmul_rn`` / ``__dadd_rn`` intrinsics are
    required. Verified bit-exact up to D = 100000.

    The one documented exception is ``L == 1``: a (D, 1) C-contig array
    is contiguous along the reduction axis, so numpy stops accumulating
    row-by-row into the (L,) output and switches to its pairwise
    summation, which this kernel deliberately does not reproduce (the
    production shape is L = number of parcels >> 1, and matching the
    CPU/numba order for L >= 2 is what the saved ``mtc`` is compared
    on). The two then differ by a handful of ulps of the column norm:
    measured max |got - expect| / |expect| of 1.3e-15 (10 ulps) at
    D = 4096 and 3.7e-15 (32 ulps) at D = 100000, on standard-normal
    input.
    """
    if mtc_DxL_dev.dtype != cp.float64:
        raise ValueError(
            f"colnorm_scale_cupy: mtc must be fp64; got {mtc_DxL_dev.dtype}"
        )
    if not mtc_DxL_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("colnorm_scale_cupy: mtc must be C-contiguous")
    D, L = mtc_DxL_dev.shape
    grid = ((L + _COLNORM_BLOCK - 1) // _COLNORM_BLOCK,)
    _colnorm_scale_kernel(grid, (_COLNORM_BLOCK,),
                          (mtc_DxL_dev, cp.int32(D), cp.int32(L)))


def epsil_input_rowdot_cupy(profile_dev: cp.ndarray, mtc_DxL_dev: cp.ndarray,
                            labels_dev: cp.ndarray) -> float:
    """``Σ_{i: labels[i]>0} <profile[i, :], mtc[:, labels[i]-1]>``.

    The fused replacement for ``inner = profile @ mtc`` (a 77 GFLOP
    fp64 GEMM producing a 196 MB (N, L) intermediate) followed by the
    gather-sum: only the L columns each row actually needs are touched,
    so the cost drops to one pass over ``profile`` (O(N·D)).

    Determinism: per-row dots use a fixed 256-thread shared-memory
    tree; the cross-row fold is a fixed 256-chunk sequential partial
    sum, folded on the host in ascending chunk order. No fp64 atomics,
    no library reduction — the result is bit-identical run to run.

    This is a genuine reformulation of the CPU path's arithmetic (BLAS
    dgemm + numba per-thread partial sums), so the scalar differs from
    the CPU value in the last few ulps; the caller feeds it to
    ``invAd``, which is insensitive at that level.
    """
    if profile_dev.dtype != cp.float64 or mtc_DxL_dev.dtype != cp.float64:
        raise ValueError(
            f"epsil_input_rowdot_cupy: profile and mtc must be fp64; got "
            f"{profile_dev.dtype} / {mtc_DxL_dev.dtype}"
        )
    if labels_dev.dtype != cp.int64:
        raise ValueError(
            f"epsil_input_rowdot_cupy: labels must be int64; got "
            f"{labels_dev.dtype}"
        )
    if not profile_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("epsil_input_rowdot_cupy: profile must be C-contiguous")
    N, D = profile_dev.shape
    Dm, L = mtc_DxL_dev.shape
    if Dm != D:
        raise ValueError(
            f"epsil_input_rowdot_cupy: mtc has D={Dm}, profile has D={D}"
        )
    if labels_dev.shape != (N,):
        raise ValueError(
            f"epsil_input_rowdot_cupy: labels must be ({N},); got "
            f"{labels_dev.shape}"
        )
    # The kernel indexes ``mtc_T[labels[i] - 1]`` with no clamp, so an
    # out-of-range label would read past the (L, D) buffer and return a
    # silently wrong scalar. One device max + sync (microseconds, once
    # per ini_params call) buys a named error instead.
    if N > 0:
        max_label = int(labels_dev.max().item())
        if max_label > L:
            raise ValueError(
                f"epsil_input_rowdot_cupy: labels.max() ({max_label}) > L "
                f"({L}); every label must be in [0, L] (0 = no parcel)."
            )
    mtc_T = cp.ascontiguousarray(mtc_DxL_dev.T)      # (L, D) fp64
    rowout = cp.empty(N, dtype=cp.float64)
    _rowdot_kernel(
        (N,), (_ROWDOT_BLOCK,),
        (profile_dev, mtc_T, labels_dev, rowout,
         cp.int32(N), cp.int32(D), cp.int32(L)),
    )
    nchunks = min(_EPSIL_CHUNKS, N)
    partials = cp.empty(nchunks, dtype=cp.float64)
    _chunk_sum_kernel(
        ((nchunks + 127) // 128,), (128,),
        (rowout, partials, cp.int32(N), cp.int32(nchunks)),
    )
    host = cp.asnumpy(partials)
    total = 0.0
    for v in host:
        total += float(v)
    return total
