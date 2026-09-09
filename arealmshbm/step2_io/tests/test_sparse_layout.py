"""test_sparse_layout.py — the static P-layout of the step-2 ``gpu`` backend.

The layout is the whole backend's index contract
(``docs/step2_sparse_design.md`` §2.1), so the bar here is structural
equality with the dense reference builder plus every invariant the
builders promise to enforce.

The bench-backed test (``MSHBM_STEP2_BENCH_DIR``, default
``testdata/step2_bench``) is the one that matters in production: the
sparse builder fed the raw MATLAB masks must produce exactly the layout
the dense path's ``build_step2_boundary_mask`` would.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from arealmshbm.step2_io import (
    build_step2_layout,
    build_step2_layout_dense,
    layouts_equal,
)


BENCH_DIR = Path(os.environ.get("MSHBM_STEP2_BENCH_DIR", "testdata/step2_bench"))


def _block_masks(n_h: int, L_h: int, seed: int = 3, density: float = 0.3):
    rng = np.random.default_rng(seed)
    lh = (rng.random((n_h, L_h)) < density).astype(np.float64)
    rh = (rng.random((n_h, L_h)) < density).astype(np.float64)
    # Leave a couple of empty rows in (they are legal — fsaverage6 has
    # 1 695) but no empty column, matching the real mask.
    lh[0] = 0.0
    rh[1] = 0.0
    for m in (lh, rh):
        empty = np.flatnonzero(m.sum(axis=0) == 0)
        for c in empty:
            m[2 + int(c) % (n_h - 2), c] = 1.0
    return lh, rh


def _dense_from_blocks(lh, rh):
    n_h, L_h = lh.shape
    top = np.concatenate([lh, np.zeros((n_h, L_h))], axis=1)
    bot = np.concatenate([np.zeros((n_h, L_h)), rh], axis=1)
    return np.concatenate([top, bot], axis=0)


def test_sparse_and_dense_builders_agree() -> None:
    lh, rh = _block_masks(23, 7)
    a = build_step2_layout(sp.csc_matrix(lh), sp.csc_matrix(rh))
    b = build_step2_layout_dense(_dense_from_blocks(lh, rh))
    assert layouts_equal(a, b)
    assert a.P == int(lh.sum() + rh.sum())


def test_csr_order_and_csc_view() -> None:
    lh, rh = _block_masks(19, 5, seed=9)
    lay = build_step2_layout(lh, rh)
    dense = _dense_from_blocks(lh, rh)

    # CSR: p_row non-decreasing, col strictly ascending inside a row.
    assert np.all(np.diff(lay.p_row) >= 0)
    for n in range(lay.N):
        cols = lay.col[lay.row_ptr[n]:lay.row_ptr[n + 1]]
        assert np.all(np.diff(cols) > 0)
        assert np.array_equal(np.sort(cols), np.flatnonzero(dense[n]))

    # CSC: rows ascending inside a column, csc_pidx points back at CSR.
    for l in range(lay.L):
        rows = lay.csc_row[lay.col_ptr[l]:lay.col_ptr[l + 1]]
        assert np.all(np.diff(rows) > 0)
        assert np.array_equal(rows, np.flatnonzero(dense[:, l]))
    assert np.array_equal(lay.p_row[lay.csc_pidx], lay.csc_row)
    assert np.array_equal(lay.col[lay.csc_pidx],
                          np.repeat(np.arange(lay.L), np.diff(lay.col_ptr)))


def test_scatter_gather_round_trip() -> None:
    lh, rh = _block_masks(17, 4, seed=5)
    lay = build_step2_layout(lh, rh)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(lay.P).astype(np.float32)
    assert np.array_equal(lay.gather(lay.scatter(x)), x)
    # batched
    xb = rng.standard_normal((3, lay.P)).astype(np.float32)
    assert np.array_equal(lay.gather(lay.scatter(xb)), xb)
    # and the dense image is the mask's support
    dense = _dense_from_blocks(lh, rh)
    assert np.array_equal(lay.scatter(np.ones(lay.P, np.float64)), dense)


def test_as_csr_matrix_is_the_mask() -> None:
    lh, rh = _block_masks(13, 3, seed=2)
    lay = build_step2_layout(lh, rh)
    m = lay.as_csr_matrix()
    assert np.array_equal(m.toarray(), _dense_from_blocks(lh, rh))


def test_dtypes_and_int32() -> None:
    lh, rh = _block_masks(11, 3, seed=4)
    lay = build_step2_layout(lh, rh)
    for f in ("row_ptr", "col", "p_row", "col_ptr", "csc_row", "csc_pidx"):
        arr = getattr(lay, f)
        assert arr.dtype == np.int32, f
        assert arr.flags["C_CONTIGUOUS"], f
    assert lay.row_ptr.shape == (lay.N + 1,)
    assert lay.col_ptr.shape == (lay.L + 1,)
    assert lay.col.shape == lay.p_row.shape == (lay.P,)


def test_non_binary_mask_rejected() -> None:
    lh, rh = _block_masks(9, 3, seed=6)
    lh[lh != 0] = 2.0
    with pytest.raises(ValueError, match="0/1 indicator"):
        build_step2_layout(lh, rh)
    with pytest.raises(ValueError, match="0/1 indicator"):
        build_step2_layout_dense(_dense_from_blocks(lh, rh))


def test_cross_hemisphere_cell_rejected() -> None:
    lh, rh = _block_masks(9, 3, seed=7)
    dense = _dense_from_blocks(lh, rh)
    dense[0, dense.shape[1] - 1] = 1.0          # LH vertex, RH parcel
    with pytest.raises(ValueError, match="cross-hemisphere"):
        build_step2_layout_dense(dense)


def test_mismatched_hemispheres_rejected() -> None:
    lh, _ = _block_masks(9, 3, seed=8)
    rh, _ = _block_masks(8, 3, seed=8)
    with pytest.raises(ValueError, match="hemispheres must match"):
        build_step2_layout(lh, rh[:8])


def test_empty_mask_rejected() -> None:
    z = np.zeros((6, 2))
    with pytest.raises(ValueError, match="no nonzero cell"):
        build_step2_layout(z, z)


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.float64])
def test_a_sparse_caller_matrix_is_not_mutated(dtype) -> None:
    """``build_step2_layout`` must not touch the caller's CSR.

    ``eliminate_zeros``/``sort_indices`` are in-place, so a non-copying
    conversion rewrites the caller's arrays. With a dtype change scipy
    copies ``data`` but still shares ``indices``, which corrupts values,
    not just structure — hence the explicit stored zero below and the
    second call, which would otherwise see a different mask.
    """
    # Row 1 carries an explicit zero at column 1; the dense image is
    # [[1,0],[1,0],[0,1],[0,0]].
    data = np.array([1, 0, 1, 1], dtype=dtype)
    indices = np.array([0, 1, 0, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3, 4, 4], dtype=np.int32)
    m = sp.csr_matrix((data, indices, indptr), shape=(4, 2))
    img = m.toarray().astype(np.float64)

    lay1 = build_step2_layout(m, m)

    assert m.nnz == 4
    assert np.array_equal(m.indices, indices)
    assert np.array_equal(m.indptr, indptr)
    assert np.array_equal(m.toarray().astype(np.float64), img)

    lay2 = build_step2_layout(m, m)
    assert layouts_equal(lay1, lay2)
    assert layouts_equal(lay1, build_step2_layout_dense(
        _dense_from_blocks(img, img)))


def test_layouts_equal_detects_a_difference() -> None:
    lh, rh = _block_masks(15, 4, seed=10)
    a = build_step2_layout(lh, rh)
    lh2 = lh.copy()
    r, c = (int(v[0]) for v in np.nonzero(lh2))
    lh2[r, c] = 0.0
    b = build_step2_layout(lh2, rh)
    assert not layouts_equal(a, b)
    assert layouts_equal(a, build_step2_layout(lh, rh))


# ─────────────────────────────────────────────────────────────────────
# Bench-backed — the real fsaverage6 / Schaefer-300 mask
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (BENCH_DIR / "proj" / "spatial_mask"
         / "spatial_mask_fsaverage6.mat").exists(),
    reason="step-2 bench project not present")
def test_bench_layout_matches_the_dense_pipeline_path() -> None:
    from arealmshbm.data_io import mat5_stream
    from arealmshbm.data_io.load_spatial_mask import load_spatial_mask
    from arealmshbm.step2_init.boundary_mask import build_step2_boundary_mask

    p = BENCH_DIR / "proj" / "spatial_mask" / "spatial_mask_fsaverage6.mat"
    lay = build_step2_layout(mat5_stream.read_sparse(p, "lh_boundary"),
                             mat5_stream.read_sparse(p, "rh_boundary"))
    lh_b, rh_b = load_spatial_mask(p)
    bm = build_step2_boundary_mask(lh_boundary=lh_b, rh_boundary=rh_b,
                                   mode="gMSHBM")
    assert layouts_equal(lay, build_step2_layout_dense(bm))
    assert lay.N == 81924 and lay.L == 300 and lay.n_lh == 40962
    assert lay.P == int((bm != 0).sum())
