"""test_step2_init

Synthetic self-tests for the step-2 init leaves.

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


def _init_cohort():
    rng = np.random.default_rng(7)
    S, N, T, D, L = 6, 12, 2, 5, 4
    bold = rng.standard_normal((S, N, T, D)).astype(np.float32)
    bold[:, 0] = 0.0                                   # a medial vertex
    mtc = rng.standard_normal((D, L))
    bm = np.ones((N, L), dtype=np.float32)
    return bold, mtc, bm, T, L


@pytest.mark.parametrize("loader_cls",
                         [InMemoryProfileLoader, _PrefetchingLoader])
def test_compose_init_state_is_per_subject(loader_cls):
    """Each subject's one-hot ``s_lambda`` slice depends on that subject
    only: a cohort larger than the worker pool (slot recycling) gives the
    same slices as S=1 runs."""
    bold, mtc, bm, T, L = _init_cohort()
    s_lambda, _theta = compose_init_state(
        loader_cls(bold, num_session=T), mtc, bm, L)
    for s in range(bold.shape[0]):
        one, _ = compose_init_state(
            loader_cls(bold[s:s + 1].copy(), num_session=T), mtc, bm, L)
        np.testing.assert_array_equal(s_lambda[s], one[0])


def test_compose_init_state_prefetch_and_decode_paths_agree():
    bold, mtc, bm, T, L = _init_cohort()
    a = compose_init_state(InMemoryProfileLoader(bold, num_session=T),
                           mtc, bm, L)
    b = compose_init_state(_PrefetchingLoader(bold, num_session=T),
                           mtc, bm, L)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
