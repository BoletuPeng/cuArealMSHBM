"""test_step2_init

Synthetic self-tests for the step-2 init leaves: the boundary mask and
the fused ``compose_init_state`` super-call (the stock CBIG init), the
latter against a numpy transcription of the stock formulas in single
precision.

These use small hand-rolled inputs and are fully self-contained.

Run with::

    python -m pytest arealmshbm/step2_init/tests/ -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.step2_em_iter_master.tests._loaders import (
    InMemoryProfileLoader,
)
from arealmshbm.step2_init import (
    build_step2_boundary_mask,
    compose_init_state,
)

EPS = np.finfo(np.float64).eps


# =====================================================================
# Self-tests — boundary_mask and the fused ``compose_init_state``
# super-call (also covered end-to-end by the step2_pipeline harness).
# =====================================================================

class TestBoundaryMask:
    """L9 — synthetic 10x3 lh + 10x3 rh."""

    def _build_inputs(self):
        rng = np.random.default_rng(1)
        lh = rng.random((10, 3)).astype(np.float32)
        rh = rng.random((10, 3)).astype(np.float32)
        return lh, rh

    def test_gMSHBM_block_diagonal(self):
        lh, rh = self._build_inputs()
        bm = build_step2_boundary_mask(lh, rh, mode="gMSHBM")
        assert bm.shape == (20, 6)
        assert bm.dtype == np.float32
        # Top-left block matches lh.
        np.testing.assert_array_equal(bm[:10, :3], lh)
        # Top-right block is zero.
        np.testing.assert_array_equal(bm[:10, 3:], 0)
        # Bottom-left block is zero.
        np.testing.assert_array_equal(bm[10:, :3], 0)
        # Bottom-right block matches rh.
        np.testing.assert_array_equal(bm[10:, 3:], rh)

    def test_dMSHBM_matches_gMSHBM_on_bilateral(self):
        # On bilateral meshes the two arms produce identical output;
        # this is the documented behavior.
        lh, rh = self._build_inputs()
        bm_g = build_step2_boundary_mask(lh, rh, mode="gMSHBM")
        bm_d = build_step2_boundary_mask(lh, rh, mode="dMSHBM")
        np.testing.assert_array_equal(bm_g, bm_d)

    def test_invalid_mode(self):
        # cMSHBM was supported in earlier rounds; now rejected.
        lh, rh = self._build_inputs()
        with pytest.raises(ValueError):
            build_step2_boundary_mask(lh, rh, mode="MSHBM")
        with pytest.raises(ValueError):
            build_step2_boundary_mask(lh, rh, mode="cMSHBM")


class _PrefetchingLoader(InMemoryProfileLoader):
    """In-memory loader that advertises the host packed cache, so
    ``compose_init_state`` takes its prefetch path."""
    cache_mode = "eager_bitpacked"

    def prefetch_packed(self, s: int) -> None:
        pass


def _init_cohort(ini_val=3.0, mask_density=0.5, seed=7):
    rng = np.random.default_rng(seed)
    S, N, T, D, L = 6, 12, 2, 5, 4
    bold = rng.standard_normal((S, N, T, D)).astype(np.float32)
    bold[:, 0] = 0.0                                   # a medial vertex
    mtc = rng.standard_normal((D, L))
    bm = (rng.random((N, L)) < mask_density).astype(np.float32)
    bm[:, 0] = 1.0                                     # every row has an in-mask cell
    return bold, mtc, bm, T, L, ini_val


def _stock_reference(bold, mtc, bm, ini_val):
    """numpy transcription of the stock init (child ``ini_s_lambda`` +
    parent mask / normalise / theta_num), single precision throughout
    like MATLAB."""
    S, N, T, D = bold.shape
    L = mtc.shape[1]
    g = mtc.astype(np.float32)
    kappa_T = np.float32(np.float32(ini_val) * np.float32(T))
    sl = np.zeros((S, N, L), dtype=np.float32)
    for s in range(S):
        a = (bold[s].sum(axis=1, dtype=np.float32)
             * (np.float32(1) / np.float32(T))).astype(np.float32)
        medial = (a ** 2).sum(1) == 0
        v = ((a @ g) * kappa_T).astype(np.float32)
        e = np.exp((v - v.max(1, keepdims=True)).astype(np.float32)
                   ).astype(np.float32) * (bm != 0)
        rs = e.sum(1, keepdims=True, dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            n_ = (e / rs).astype(np.float32)
        n_[~np.isfinite(n_).all(1)] = 0
        n_[medial] = 0
        sl[s] = n_
    th = (sl.sum(0, dtype=np.float32) / np.float32(S)).astype(np.float32)
    cnt = (sl != 0).sum(0)
    nz = th != 0
    th[nz] = (th[nz] / cnt[nz].astype(np.float32)).astype(np.float32)
    th = (th + np.float32(EPS)).astype(np.float32)
    return sl, th


@pytest.mark.parametrize("loader_cls",
                         [InMemoryProfileLoader, _PrefetchingLoader])
def test_compose_init_state_matches_stock_formula(loader_cls):
    """s_lambda = masked, row-normalised exp(v - rowmax); theta = per-cell
    mean over the subjects nonzero there + eps. Only the BLAS summation
    order of the reference's ``a @ g`` differs from the kernel's serial
    fp32 sum, hence the (tight) tolerance; the zero pattern is exact."""
    bold, mtc, bm, T, L, ini_val = _init_cohort()
    s_lambda, theta = compose_init_state(
        loader_cls(bold, num_session=T), mtc, bm, L, ini_val)
    sl_ref, th_ref = _stock_reference(bold, mtc, bm, ini_val)
    assert s_lambda.dtype == np.float32 and theta.dtype == np.float32
    np.testing.assert_array_equal(s_lambda == 0, sl_ref == 0)
    np.testing.assert_allclose(s_lambda, sl_ref, rtol=1e-5, atol=0)
    np.testing.assert_allclose(theta, th_ref, rtol=1e-5, atol=0)
    # Every live row is a distribution over its in-mask cells; the
    # medial vertex is all-zero in every subject and theta = eps there.
    live = s_lambda.sum(2)
    assert np.allclose(live[:, 1:], 1.0, atol=1e-6)
    assert np.all(live[:, 0] == 0)
    assert np.all(theta[0] == np.float32(EPS))
    assert np.all((s_lambda != 0) <= (bm != 0)[None])


def test_theta_num_rule():
    """Stock's theta is ``mean(s_lambda, 3)`` (÷ S) divided AGAIN by
    ``theta_num``, the number of subjects nonzero at the cell — not the
    mean over those subjects. A cell no subject touches is eps."""
    bold, mtc, bm, T, L, ini_val = _init_cohort(mask_density=0.5)
    S = bold.shape[0]
    s_lambda, theta = compose_init_state(
        InMemoryProfileLoader(bold, num_session=T), mtc, bm, L, ini_val)
    nz = s_lambda[:, 3, :] != 0
    assert 0 < nz.sum() < S * L          # the row mixes touched and untouched cells
    for l in range(L):
        k = int(nz[:, l].sum())
        if k == 0:
            assert theta[3, l] == np.float32(EPS)
        else:
            mean_S = np.float32(s_lambda[:, 3, l].sum(dtype=np.float32)
                                / np.float32(S))
            expect = np.float32(mean_S / np.float32(k)) + np.float32(EPS)
            assert np.isclose(theta[3, l], expect, rtol=1e-6, atol=0)


def test_all_in_mask_underflow_row_is_zero():
    """A row whose in-mask cells all sit > ~104 nats below the row max
    underflows in fp32 and stays all-zero (stock's 0/0 -> NaN -> 0)."""
    S, N, T, D, L = 1, 4, 1, 3, 3
    bold = np.zeros((S, N, T, D), dtype=np.float32)
    bold[0, :, 0, 0] = 1.0                             # every row = e_0
    mtc = np.zeros((D, L))
    mtc[0] = [1.0, 0.0, 0.1]                           # logits ∝ (1, 0, 0.1)
    bm = np.ones((N, L), dtype=np.float32)
    bm[1, 0] = 0.0                                     # row 1: best parcel out of mask
    s_lambda, theta = compose_init_state(
        InMemoryProfileLoader(bold, num_session=T), mtc, bm, L, ini_val=120.0)
    # row 1: in-mask logits are 120 and 108 nats below the max -> both
    # underflow in fp32 (exp(-104) is the smallest subnormal)
    assert np.all(s_lambda[0, 1] == 0)
    assert np.all(theta[1] == np.float32(EPS))
    # rows with parcel 0 in mask: parcel 0 carries the mass (S = 1, so
    # theta = 1/1/1 + eps = 1)
    for n in (0, 2, 3):
        assert s_lambda[0, n, 0] == np.float32(1.0)
        assert np.isclose(theta[n, 0], 1.0, atol=1e-6)
    # with a milder kappa (logits 10, 0, 1) the same row is alive and
    # normalised over its two in-mask cells
    s2, _ = compose_init_state(
        InMemoryProfileLoader(bold, num_session=T), mtc, bm, L, ini_val=10.0)
    assert np.isclose(s2[0, 1].sum(), 1.0, atol=1e-6) and s2[0, 1, 0] == 0
    assert s2[0, 1, 2] > s2[0, 1, 1] > 0


@pytest.mark.parametrize("loader_cls",
                         [InMemoryProfileLoader, _PrefetchingLoader])
def test_compose_init_state_is_per_subject(loader_cls):
    """Each subject's ``s_lambda`` slice depends on that subject only: a
    cohort larger than the worker pool (slot recycling) gives the same
    slices as S=1 runs."""
    bold, mtc, bm, T, L, ini_val = _init_cohort()
    s_lambda, _theta = compose_init_state(
        loader_cls(bold, num_session=T), mtc, bm, L, ini_val)
    for s in range(bold.shape[0]):
        one, _ = compose_init_state(
            loader_cls(bold[s:s + 1].copy(), num_session=T), mtc, bm, L,
            ini_val)
        np.testing.assert_array_equal(s_lambda[s], one[0])


def test_compose_init_state_prefetch_and_decode_paths_agree():
    bold, mtc, bm, T, L, ini_val = _init_cohort()
    a = compose_init_state(InMemoryProfileLoader(bold, num_session=T),
                           mtc, bm, L, ini_val)
    b = compose_init_state(_PrefetchingLoader(bold, num_session=T),
                           mtc, bm, L, ini_val)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
