"""test_step2_init

Synthetic self-tests for the step-2 init leaves.

These use small hand-rolled inputs and are fully self-contained. A
MATLAB-GT validation class (``TestGTBoundaryMask``) used to live here
too; it was retired when the pipeline decoupled from MATLAB.

Run with::

    python -m pytest arealmshbm/step2_init/tests/ -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.step2_init import build_step2_boundary_mask


# =====================================================================
# Self-tests — boundary_mask is the only step2_init leaf with a
# standalone public API; the rest of the init block ships as the fused
# ``compose_init_state`` super-call and is covered end-to-end by the
# step2_pipeline harness.
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
