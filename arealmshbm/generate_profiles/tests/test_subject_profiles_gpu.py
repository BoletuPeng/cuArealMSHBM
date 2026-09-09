"""test_subject_profiles_gpu.py

The fused whole-subject GPU leaf against the per-session path it
replaced.

The oracle is ``_reference_packed`` -- the whole of that retired
algorithm, transcribed here and now living nowhere else: the
``inv_norm = +inf`` zscore RawKernel below (moved verbatim out of
``_kernels_gpu`` when the per-session leaf was deleted), a per-hemi
seed gather, two sgemms, ``cp.nan_to_num``, accumulate, ``* 1/n_runs``,
``cp.concatenate`` for the threshold, ``cp.partition`` for the cut,
then the same binarize+MW-zero+pack kernel the production path uses.
Keeping it executable here is what makes the bit-exactness claim a test
rather than a memory -- and what makes the NaN cases below expressible
at all.

What is pinned
--------------
1. **Bit-exact packed output** vs that oracle, for single-run and
   multi-run sessions, with and without a censor vector.
2. **The ``cp.nan_to_num`` removal is a no-op on the result** --
   including the three cases it exists for: an exactly constant
   column, a near-constant column whose ``post_sumsq`` cancels to
   ``<= 0``, and a column carrying a genuine ``NaN`` / ``+-inf`` in the
   BOLD itself.
3. The seed gather may be taken from the normalised array (the fused
   path) or from the raw one (the legacy path) -- same bits.
4. Generator input, so ingest can overlap compute.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.generate_profiles._kernels_gpu import (  # noqa: E402
    binarize_mwzero_pack_cupy,
)
from arealmshbm.generate_profiles.profiles_subject_gpu import (  # noqa: E402
    compute_subject_profiles_gpu,
)


# ─────────────────────────────────────────────────────────────────────
# The retired zscore kernel, verbatim
# ─────────────────────────────────────────────────────────────────────
# The production kernel is ``zscore_unit_norm_columns_zerovar_cupy``,
# which writes an exact ``0.0f`` where this one wrote ``+inf``. The
# point of the comparison below is that the difference is not
# observable, so the ``+inf`` formulation has to stay executable
# somewhere -- here.
_ZSCORE_BLOCK = 256          # power of 2; the kernel's s_sum / s_sumsq size

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
    _zscore_kernel((N,), (_ZSCORE_BLOCK,),
                    (x_TxN, out_TxN, cp.int32(T), cp.int32(N)))


# ─────────────────────────────────────────────────────────────────────
# Oracle: the retired per-session leaf's algorithm, verbatim
# ─────────────────────────────────────────────────────────────────────
def _reference_packed(sessions, seed_idx, mw, n_lh, threshold):
    """``(n_sess, n_full, ceil(K/8)) uint8``, the legacy way."""
    n_full = int(mw.shape[0])
    K = int(seed_idx.shape[0])
    D_bytes = (K + 7) // 8
    lh_seed = seed_idx[seed_idx < n_lh]
    rh_seed = seed_idx[seed_idx >= n_lh] - n_lh
    lh_seed_d = cp.asarray(lh_seed)
    rh_seed_d = cp.asarray(rh_seed)
    mw_d = cp.asarray(mw.astype(np.uint8))
    out = cp.empty((len(sessions), n_full, D_bytes), dtype=cp.uint8)

    for si, runs in enumerate(sessions):
        n_runs = len(runs)
        lh_sum = rh_sum = None
        for buf in runs:
            lh = cp.ascontiguousarray(buf[:, :n_lh])
            rh = cp.ascontiguousarray(buf[:, n_lh:])
            T = int(lh.shape[0])
            s = cp.empty((T, K), dtype=cp.float32)
            s[:, :lh_seed_d.size] = lh[:, lh_seed_d]
            s[:, lh_seed_d.size:] = rh[:, rh_seed_d]
            sn = cp.empty_like(s)
            ln = cp.empty_like(lh)
            rn = cp.empty_like(rh)
            zscore_unit_norm_columns_cupy(s, sn)
            zscore_unit_norm_columns_cupy(lh, ln)
            zscore_unit_norm_columns_cupy(rh, rn)
            lc = sn.T @ ln
            rc = sn.T @ rn
            cp.nan_to_num(lc, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            cp.nan_to_num(rc, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            lh_sum = lc if lh_sum is None else lh_sum + lc
            rh_sum = rc if rh_sum is None else rh_sum + rc
        inv = cp.float32(1.0 / n_runs)
        lh_sum *= inv
        rh_sum *= inv
        combined = cp.concatenate([lh_sum, rh_sum], axis=1)
        flat = combined.reshape(-1)
        numel = int(flat.size)
        idx1 = max(1, min(int(np.floor(numel * float(threshold) + 0.5)), numel))
        kth = numel - idx1
        t = float(cp.partition(flat, kth)[kth].get())
        binarize_mwzero_pack_cupy(combined, mw_d, t, out[si])
    return cp.asnumpy(out)


def _fused(sessions, seed_idx, mw, threshold, as_generator=False,
           censor=None):
    n_full = int(mw.shape[0])
    items = [(i, (r[0] if len(r) == 1 else r))
             for i, r in enumerate(sessions)]
    src = (x for x in items) if as_generator else items
    packed, K = compute_subject_profiles_gpu(
        src, n_sess=len(sessions), n_full=n_full, seed_idx=seed_idx,
        mw_mask=mw, threshold=threshold, censor=censor)
    return np.array(packed, copy=True), K


def _make(n_lh, n_rh, T, n_sess, n_runs=1, seed=0, mutate=None):
    rng = np.random.default_rng(seed)
    n_full = n_lh + n_rh
    sessions = []
    for _ in range(n_sess):
        runs = []
        for _r in range(n_runs):
            a = rng.standard_normal((T, n_full)).astype(np.float32)
            if mutate is not None:
                mutate(a, rng)
            runs.append(cp.asarray(a))
        sessions.append(runs)
    # Seeds are the first 12 lh + first 9 rh vertices; medial wall is a
    # disjoint slice so the two never collide.
    seed_idx = np.concatenate([np.arange(12),
                               np.arange(9) + n_lh]).astype(np.int64)
    mw = np.zeros(n_full, dtype=np.uint8)
    mw[n_lh - 5:n_lh] = 1
    mw[n_full - 4:] = 1
    return sessions, seed_idx, mw


@pytest.mark.parametrize("as_generator", [False, True])
def test_matches_the_per_session_path(as_generator):
    sessions, seed_idx, mw = _make(64, 48, 20, 3, seed=1)
    ref = _reference_packed(sessions, seed_idx, mw, 64, 0.1)
    got, K = _fused(sessions, seed_idx, mw, 0.1, as_generator=as_generator)
    assert K == seed_idx.size
    assert np.array_equal(got, ref)


def test_whole_subject_sgemm_matches_the_per_hemi_pair():
    """The fused leaf issues ONE (K, T) x (T, n_lh + n_rh) sgemm where the
    per-session fallback issues two per-hemi ones. cuBLAS picks its
    tiling / split-K from the shape, so the two reduction orders are
    only equal as a fact about the production shape class — pin it
    there (fsaverage3 seed, fsaverage6 target)."""
    import cupy as cp
    K, T, n_lh = 1175, 242, 40962
    rng = cp.random.RandomState(0)
    a = rng.standard_normal((T, K), dtype=cp.float32)
    b = rng.standard_normal((T, 2 * n_lh), dtype=cp.float32)
    full = a.T @ b
    lh = a.T @ cp.ascontiguousarray(b[:, :n_lh])
    rh = a.T @ cp.ascontiguousarray(b[:, n_lh:])
    assert bool(cp.array_equal(full[:, :n_lh], lh))
    assert bool(cp.array_equal(full[:, n_lh:], rh))


def test_multi_run_sessions():
    sessions, seed_idx, mw = _make(48, 48, 17, 2, n_runs=3, seed=2)
    ref = _reference_packed(sessions, seed_idx, mw, 48, 0.1)
    got, _K = _fused(sessions, seed_idx, mw, 0.1)
    assert np.array_equal(got, ref)


def test_censor_drops_the_same_timepoints():
    """A censor vector must be equivalent to having been handed the
    already-compacted buffers."""
    sessions, seed_idx, mw = _make(40, 40, 24, 2, n_runs=2, seed=5)
    rng = np.random.default_rng(9)
    keep = [[(rng.integers(0, 2, size=24) | np.array(
        [1] * 24)).astype(np.int32) for _ in range(2)] for _ in range(2)]
    # Make some real drops (keep at least 8 frames).
    for s in range(2):
        for r in range(2):
            keep[s][r][:] = 1
            keep[s][r][3] = 0
            keep[s][r][11] = 0
    compacted = [[cp.ascontiguousarray(b[cp.asarray(k == 1)])
                  for b, k in zip(runs, ks)]
                 for runs, ks in zip(sessions, keep)]
    ref = _reference_packed(compacted, seed_idx, mw, 40, 0.1)
    got, _K = _fused(sessions, seed_idx, mw, 0.1,
                     censor={0: keep[0], 1: keep[1]})
    assert np.array_equal(got, ref)


# ─────────────────────────────────────────────────────────────────────
# The nan_to_num removal
# ─────────────────────────────────────────────────────────────────────
def _constant_columns(a, rng):
    a[:, 3] = 2.5                       # exactly constant
    a[:, 7] = 0.0                       # exactly zero
    a[:, 70] = -1.0


def _cancelling_columns(a, rng):
    # |mean| >> spread: sumsq - T*mean^2 cancels to <= 0 in fp64.
    a[:, 5] = np.float32(1e12)
    a[:, 5] += rng.standard_normal(a.shape[0]).astype(np.float32) * 1e-6
    a[:, 41] = np.float32(3e13)


def _nan_and_inf_in_the_bold(a, rng):
    a[4, 9] = np.nan
    a[0, 55] = np.inf
    a[7, 80] = -np.inf
    a[2, 2] = np.nan


def _fp32_overflow_columns(a, rng):
    """post_sumsq > 0, but ``(float)(1/sqrt(post_sumsq))`` overflows.

    A near-constant column of magnitude ~1e-36 with a 1-ULP spread has
    a perfectly ordinary positive fp64 ``post_sumsq`` of ~1e-90, whose
    reciprocal square root is ~1e45 -- past FLT_MAX. The first version
    of the zerovar kernel tested only ``post_sumsq > 0.0`` and so took
    the NON-degenerate branch here, stored ``inv_norm = +inf``, and
    leaked NaN into the corr and the selector with no ``nan_to_num``
    left to catch it.
    """
    for col, a0 in ((11, 1e-36), (57, 1e-37), (4, 1e-38)):
        base = np.float32(a0)
        a[:, col] = base
        a[-1, col] = np.nextafter(base, np.float32(np.inf))
    # ... and one that is also a SEED column (col < 12 is a seed).
    base = np.float32(1e-36)
    a[:, 6] = base
    a[-1, 6] = np.nextafter(base, np.float32(np.inf))


@pytest.mark.parametrize("mutate", [_constant_columns,
                                     _cancelling_columns,
                                     _nan_and_inf_in_the_bold,
                                     _fp32_overflow_columns])
def test_dropping_nan_to_num_changes_nothing(mutate):
    """The fused path sets ``inv_norm = 0`` (and stores a literal zero)
    where the legacy path sets ``+inf`` and sweeps the resulting
    non-finite corr rows/columns with ``cp.nan_to_num``. Same bits."""
    sessions, seed_idx, mw = _make(64, 48, 20, 2, seed=13, mutate=mutate)
    ref = _reference_packed(sessions, seed_idx, mw, 64, 0.1)
    got, _K = _fused(sessions, seed_idx, mw, 0.1)
    assert np.array_equal(got, ref)


def test_degenerate_seed_column_too():
    """A degenerate column that is also a SEED zeroes a whole corr ROW
    on both paths (the legacy one through NaN, this one through 0)."""
    def mutate(a, rng):
        a[:, 2] = 7.0                    # seed column (< 12) constant
        a[:, 64 + 1] = 0.0               # rh seed column all zero

    sessions, seed_idx, mw = _make(64, 48, 16, 2, seed=17, mutate=mutate)
    ref = _reference_packed(sessions, seed_idx, mw, 64, 0.1)
    got, _K = _fused(sessions, seed_idx, mw, 0.1)
    assert np.array_equal(got, ref)


# ─────────────────────────────────────────────────────────────────────
# Boundary checks
# ─────────────────────────────────────────────────────────────────────
def test_rejects_wrong_width_and_dtype():
    sessions, seed_idx, mw = _make(32, 32, 8, 1, seed=21)
    with pytest.raises(ValueError, match="expected \\(T, 64\\)"):
        compute_subject_profiles_gpu(
            [(0, cp.zeros((8, 63), dtype=cp.float32))], n_sess=1,
            n_full=64, seed_idx=seed_idx, mw_mask=mw, threshold=0.1)
    with pytest.raises(ValueError, match="expected fp32"):
        compute_subject_profiles_gpu(
            [(0, cp.zeros((8, 64), dtype=cp.float64))], n_sess=1,
            n_full=64, seed_idx=seed_idx, mw_mask=mw, threshold=0.1)


def test_rejects_a_short_session_stream():
    sessions, seed_idx, mw = _make(32, 32, 8, 2, seed=23)
    with pytest.raises(ValueError, match="consumed 1 sessions"):
        compute_subject_profiles_gpu(
            [(0, sessions[0][0])], n_sess=2, n_full=64,
            seed_idx=seed_idx, mw_mask=mw, threshold=0.1)


def test_rejects_mw_length_mismatch():
    sessions, seed_idx, mw = _make(32, 32, 8, 1, seed=27)
    with pytest.raises(ValueError, match="mw_mask length"):
        compute_subject_profiles_gpu(
            [(0, sessions[0][0])], n_sess=1, n_full=64,
            seed_idx=seed_idx, mw_mask=mw[:60], threshold=0.1)
