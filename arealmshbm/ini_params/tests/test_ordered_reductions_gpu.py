"""test_ordered_reductions_gpu.py — bit-exactness gate for the ordered
fp64 device reductions that replaced the two dgemms in ini_params.

The whole point of these kernels is that they reproduce a *specific*
fp64 summation order, so the tests compare with ``np.array_equal``,
not ``allclose``:

  * ``groupsum_csr_cupy`` vs the numba ``_groupsum_kernel`` — same
    ascending-row order per parcel, hence bit-identical ``mtc``.
  * ``colnorm_scale_cupy`` vs numpy's
    ``mtc / sqrt((mtc*mtc).sum(axis=0))`` — numpy reduces the outer
    axis of a C-contig (D, L) array sequentially in d, and the kernel
    matches that (with FMA contraction disabled). Bit-exact for L >= 2
    only: the degenerate (D, 1) case is contiguous along the reduction
    axis, where numpy switches to pairwise summation, so that shape is
    checked to a documented ~1e-15 relative bound instead.
  * ``build_parcel_csr_cupy`` — rows grouped by parcel, ascending
    within each group, empty parcels present as empty ranges.
  * ``epsil_input_rowdot_cupy`` vs the CPU ``inner = profile @ mtc``
    gather-sum — a genuine reformulation, so this one is checked to a
    tight relative tolerance plus exact run-to-run determinism.

Edge cases covered: empty parcels, all-MW (every label 0) rows, D not
a multiple of the 256-thread block, D == 1, single-row parcels.

Run::

    python -m pytest arealmshbm/ini_params/tests/test_ordered_reductions_gpu.py -v

Skips if cupy is unavailable.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.ini_params._kernels import (
    _epsil_input_kernel, _groupsum_kernel,
)
from arealmshbm.ini_params._kernels_gpu import (
    build_parcel_csr_cupy,
    colnorm_scale_cupy,
    epsil_input_rowdot_cupy,
    groupsum_csr_cupy,
)


def _case(rng, N, D, L, *, empty_parcel=None, all_mw=False):
    prof = rng.standard_normal((N, D))
    if all_mw:
        labels = np.zeros(N, dtype=np.int64)
        labels[0] = 1                       # keep L >= 1 well-defined
        labels[0] = 1
    else:
        labels = rng.integers(0, L + 1, size=N).astype(np.int64)
        labels[: min(5, N)] = 0             # some medial-wall rows
        if empty_parcel is not None and L >= empty_parcel:
            labels[labels == empty_parcel] = 0
        if not (labels > 0).any():
            labels[0] = 1
    return prof, labels


def _numpy_colnorm(mtc):
    col = np.sqrt((mtc * mtc).sum(axis=0, keepdims=True))
    col = np.where(col == 0, 1.0, col)
    return mtc / col


SHAPES = [
    (1000, 1175, 300),      # production-ish
    (513, 17, 5),           # tiny D
    (64, 1, 3),             # D == 1
    (2048, 257, 7),         # D not a multiple of the 256-thread block
    (37, 256, 40),          # D exactly one block; more parcels than rows
    (200, 4096, 1),         # L == 1: numpy goes pairwise, we do not
]

# Max |got - expect| / |expect| tolerated at L == 1, where numpy's
# pairwise summation and the kernel's sequential one legitimately part
# ways. Measured 1.3e-15 (10 ulps) at D = 4096 and 3.7e-15 (32 ulps) at
# D = 100000; L >= 2 stays bit-exact at both.
COLNORM_L1_RTOL = 1e-14


@pytest.mark.parametrize("N,D,L", SHAPES)
def test_groupsum_csr_bit_exact_vs_numba(N, D, L):
    rng = np.random.default_rng(N * 7 + D * 13 + L)
    prof, labels = _case(rng, N, D, L, empty_parcel=3)

    ref = np.empty((D, L), dtype=np.float64)
    _groupsum_kernel(prof, labels, L, ref)

    off, rows = build_parcel_csr_cupy(cp.asarray(labels), L)
    got = cp.empty((D, L), dtype=cp.float64)
    groupsum_csr_cupy(cp.asarray(prof), off, rows, L, got)

    assert np.array_equal(cp.asnumpy(got), ref)


@pytest.mark.parametrize("N,D,L", SHAPES)
def test_colnorm_bit_exact_vs_numpy(N, D, L):
    rng = np.random.default_rng(N + D * 3 + L * 5)
    prof, labels = _case(rng, N, D, L, empty_parcel=2)
    ref = np.empty((D, L), dtype=np.float64)
    _groupsum_kernel(prof, labels, L, ref)
    expect = _numpy_colnorm(ref)

    dev = cp.asarray(ref)
    colnorm_scale_cupy(dev)
    got = cp.asnumpy(dev)
    if L == 1:
        # (D, 1) is contiguous along the reduction axis -> numpy uses
        # pairwise summation there. The kernel keeps the CPU-matching
        # sequential order on purpose (production L is 300-400), so
        # only a tight bound is claimed. See colnorm_scale_cupy's
        # docstring.
        rel = np.abs(got - expect) / np.maximum(np.abs(expect), 1e-300)
        assert rel.max() <= COLNORM_L1_RTOL
    else:
        assert np.array_equal(got, expect)


def test_colnorm_handles_all_zero_column():
    # parcel 1 empty -> its mtc column is all-zero -> norm clamped to 1.
    mtc = np.zeros((9, 3), dtype=np.float64)
    mtc[:, 0] = 2.0
    mtc[:, 2] = -0.5
    expect = _numpy_colnorm(mtc)
    dev = cp.asarray(mtc)
    colnorm_scale_cupy(dev)
    got = cp.asnumpy(dev)
    assert np.array_equal(got, expect)
    assert np.array_equal(got[:, 1], np.zeros(9))


@pytest.mark.parametrize("N,D,L", SHAPES)
def test_epsil_rowdot_matches_cpu_and_is_deterministic(N, D, L):
    rng = np.random.default_rng(11 * N + D + L)
    prof, labels = _case(rng, N, D, L, empty_parcel=4)
    mtc = np.empty((D, L), dtype=np.float64)
    _groupsum_kernel(prof, labels, L, mtc)
    mtc = _numpy_colnorm(mtc)

    ref = _epsil_input_kernel(prof @ mtc, labels)
    pd, md, ld = cp.asarray(prof), cp.asarray(mtc), cp.asarray(labels)
    got = epsil_input_rowdot_cupy(pd, md, ld)
    got2 = epsil_input_rowdot_cupy(pd, md, ld)

    assert got == got2, "row-dot reduction must be run-to-run identical"
    denom = max(abs(ref), 1e-30)
    assert abs(got - ref) / denom < 1e-12


def test_all_medial_wall_rows_contribute_nothing():
    """Every label 0 except one row: mtc has a single populated column
    and epsil sees exactly that row's self-dot."""
    rng = np.random.default_rng(3)
    N, D, L = 128, 33, 4
    prof = rng.standard_normal((N, D))
    labels = np.zeros(N, dtype=np.int64)
    labels[17] = 2

    ref = np.empty((D, L), dtype=np.float64)
    _groupsum_kernel(prof, labels, L, ref)
    off, rows = build_parcel_csr_cupy(cp.asarray(labels), L)
    got = cp.empty((D, L), dtype=cp.float64)
    groupsum_csr_cupy(cp.asarray(prof), off, rows, L, got)
    assert np.array_equal(cp.asnumpy(got), ref)
    assert np.array_equal(cp.asnumpy(got)[:, 1], prof[17])
    assert not cp.asnumpy(got)[:, [0, 2, 3]].any()

    mtc = _numpy_colnorm(ref)
    e = epsil_input_rowdot_cupy(cp.asarray(prof), cp.asarray(mtc),
                                cp.asarray(labels))
    assert abs(e - float(prof[17] @ mtc[:, 1])) < 1e-12 * max(abs(e), 1.0)


def test_parcel_csr_is_ascending_and_complete():
    rng = np.random.default_rng(21)
    N, L = 5000, 40
    labels = rng.integers(0, L + 1, size=N).astype(np.int64)
    labels[labels == 7] = 0                     # force an empty parcel
    off, rows = build_parcel_csr_cupy(cp.asarray(labels), L)
    off_h, rows_h = cp.asnumpy(off), cp.asnumpy(rows)

    assert off_h.dtype == np.int32 and rows_h.dtype == np.int32
    assert off_h[0] == 0
    assert off_h[-1] == int((labels > 0).sum()) == rows_h.size
    assert off_h[7] == off_h[6 + 1]             # parcel 7 (1-based) empty
    for l in range(L):
        seg = rows_h[off_h[l]:off_h[l + 1]]
        assert np.array_equal(seg, np.sort(seg)), "rows must ascend"
        assert np.array_equal(labels[seg], np.full(seg.size, l + 1))


def test_kernel_boundary_validation():
    prof = cp.zeros((8, 4), dtype=cp.float64)
    labels = cp.ones(8, dtype=cp.int64)
    off, rows = build_parcel_csr_cupy(labels, 1)
    mtc = cp.zeros((4, 1), dtype=cp.float64)

    with pytest.raises(ValueError, match="must be fp64"):
        groupsum_csr_cupy(prof.astype(cp.float32), off, rows, 1, mtc)
    with pytest.raises(ValueError, match=r"mtc must be \(D, L\)"):
        groupsum_csr_cupy(prof, off, rows, 1, cp.zeros((3, 1), cp.float64))
    with pytest.raises(ValueError, match="CSR arrays must be int32"):
        groupsum_csr_cupy(prof, off.astype(cp.int64), rows, 1, mtc)
    with pytest.raises(ValueError, match="must be fp64"):
        colnorm_scale_cupy(mtc.astype(cp.float32))
    with pytest.raises(ValueError, match="labels must be int64"):
        build_parcel_csr_cupy(labels.astype(cp.int32), 1)
    with pytest.raises(ValueError, match="L must be > 0"):
        build_parcel_csr_cupy(labels, 0)
    with pytest.raises(ValueError, match="labels must be int64"):
        epsil_input_rowdot_cupy(prof, mtc, labels.astype(cp.int32))
    with pytest.raises(ValueError, match="mtc has D="):
        epsil_input_rowdot_cupy(prof, cp.zeros((3, 1), cp.float64), labels)


def test_epsil_rejects_out_of_range_labels():
    """``mtc_T[label - 1]`` is indexed unclamped inside the kernel, so a
    label above L must be refused rather than read out of bounds."""
    prof = cp.zeros((8, 4), dtype=cp.float64)
    mtc = cp.zeros((4, 3), dtype=cp.float64)        # L = 3
    ok = cp.asarray(np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.int64))
    epsil_input_rowdot_cupy(prof, mtc, ok)          # L itself is in range

    bad = ok.copy()
    bad[4] = 4
    with pytest.raises(ValueError, match=r"labels\.max\(\) \(4\) > L \(3\)"):
        epsil_input_rowdot_cupy(prof, mtc, bad)
