"""test_mw_clamp.py — synthetic unit test for the writer-side MW=0
clamp on the (K, V_h) binarized profile arrays.

Failure mode this guards against: if a future refactor silently
removes the ``_apply_mw_zero`` call from the production path, every
downstream step2 subject loader will raise on init because the
bitpacked ``.b2nd`` contract requires zero rows at medial-wall
vertices.

This is the unit-level guard — fully synthetic, fast (<1s), no disk
I/O. The pre-decoupling MATLAB-GT validate.py used to catch the same
regression via a read-back assertion after a full step1 run; that
script is gone, so this test is now the only in-tree gate.

Run::

    python -m pytest arealmshbm/generate_profiles/tests/ -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.generate_profiles.profiles import _apply_mw_zero


def _make_binary_profile(K: int, V: int, rng: np.random.Generator) -> np.ndarray:
    """Build a (K, V) fp32 binarized profile with non-trivial 1s."""
    arr = (rng.random((K, V), dtype=np.float32) > 0.5).astype(np.float32)
    # Plant a deliberately strong signal at every vertex so the test
    # below cannot pass by coincidence.
    arr[:, ::3] = 1.0
    return arr


def _make_mars(V: int, mw_idx) -> np.ndarray:
    """Build a (V,) int MARS_label array with `1` at mw_idx, `2` (cortex)
    elsewhere — matches the CBIG convention used by load_avg_mesh.
    """
    mars = np.full(V, 2, dtype=np.int32)
    mars[np.asarray(mw_idx, dtype=np.int64)] = 1
    return mars


def test_mw_columns_zeroed_cortex_preserved():
    """Columns at MW indices must be zero post-clamp; columns at cortex
    indices must equal the pre-clamp values bit-for-bit.
    """
    rng = np.random.default_rng(0)
    K, V_lh, V_rh = 8, 64, 48
    lh_mw = [2, 5, 17, 60]
    rh_mw = [0, 1, 47]

    lh_bin = _make_binary_profile(K, V_lh, rng)
    rh_bin = _make_binary_profile(K, V_rh, rng)
    lh_mars = _make_mars(V_lh, lh_mw)
    rh_mars = _make_mars(V_rh, rh_mw)

    lh_before = lh_bin.copy()
    rh_before = rh_bin.copy()

    _apply_mw_zero(lh_bin, rh_bin, lh_mars, rh_mars)

    # MW columns are zero.
    assert (lh_bin[:, lh_mw] == 0.0).all()
    assert (rh_bin[:, rh_mw] == 0.0).all()

    # Cortex columns unchanged.
    cortex_lh = np.setdiff1d(np.arange(V_lh), np.asarray(lh_mw, dtype=np.int64))
    cortex_rh = np.setdiff1d(np.arange(V_rh), np.asarray(rh_mw, dtype=np.int64))
    np.testing.assert_array_equal(lh_bin[:, cortex_lh], lh_before[:, cortex_lh])
    np.testing.assert_array_equal(rh_bin[:, cortex_rh], rh_before[:, cortex_rh])


def test_no_mw_is_identity():
    """When MARS_label has no MW entries (all cortex), the clamp is a
    no-op. Catches a future bug where the clamp accidentally zeros
    cortex columns under empty-MW.
    """
    rng = np.random.default_rng(1)
    K, V_lh, V_rh = 8, 32, 24
    lh_bin = _make_binary_profile(K, V_lh, rng)
    rh_bin = _make_binary_profile(K, V_rh, rng)
    lh_mars = _make_mars(V_lh, [])  # all cortex
    rh_mars = _make_mars(V_rh, [])

    lh_before = lh_bin.copy()
    rh_before = rh_bin.copy()
    _apply_mw_zero(lh_bin, rh_bin, lh_mars, rh_mars)

    np.testing.assert_array_equal(lh_bin, lh_before)
    np.testing.assert_array_equal(rh_bin, rh_before)


def test_oversized_mars_fires():
    """Strict shape contract: MARS_label.size MUST equal V_h. An
    oversized MARS_label could otherwise be silently truncated to the
    first V_h entries, landing MW indices on wrong vertices.
    """
    rng = np.random.default_rng(2)
    K, V_lh, V_rh = 4, 16, 12
    lh_bin = _make_binary_profile(K, V_lh, rng)
    rh_bin = _make_binary_profile(K, V_rh, rng)

    # Oversized lh MARS_label (V_lh + 8 entries).
    lh_mars = _make_mars(V_lh + 8, [3, 5])
    rh_mars = _make_mars(V_rh, [])

    with pytest.raises(AssertionError, match="lh MARS_label.size"):
        _apply_mw_zero(lh_bin, rh_bin, lh_mars, rh_mars)


def test_undersized_mars_fires():
    """Mirror of the oversized check on the rh side."""
    rng = np.random.default_rng(3)
    K, V_lh, V_rh = 4, 16, 12
    lh_bin = _make_binary_profile(K, V_lh, rng)
    rh_bin = _make_binary_profile(K, V_rh, rng)

    lh_mars = _make_mars(V_lh, [])
    rh_mars = _make_mars(V_rh - 2, [])  # short

    with pytest.raises(AssertionError, match="rh MARS_label.size"):
        _apply_mw_zero(lh_bin, rh_bin, lh_mars, rh_mars)
