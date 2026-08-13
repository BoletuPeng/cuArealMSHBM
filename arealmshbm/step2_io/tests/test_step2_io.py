"""test_step2_io.py — synthetic self-tests for ``load_group_mtc``.

These tests are fully self-contained (no external data dependency). A
MATLAB-GT bytewise-equality check used to live here too; it was retired
when the pipeline decoupled from MATLAB.

Run:
    python -m pytest arealmshbm/step2_io/tests/ -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from arealmshbm.step2_io import load_group_mtc


# -----------------------------------------------------------------------------
# load_group_mtc
# -----------------------------------------------------------------------------

def test_load_group_mtc_synthetic(tmp_path: Path) -> None:
    """Construct a minimal synthetic group.mat and verify shapes/values."""
    Dp1, L = 7, 5
    N_lh, N_rh = 11, 13
    rng = np.random.default_rng(0)
    mtc = rng.standard_normal((Dp1, L)).astype(np.float64)
    epsil = np.array([[0.12345]], dtype=np.float64)
    lh_labels = rng.integers(0, L + 1, size=N_lh, dtype=np.int64).reshape(-1, 1)
    rh_labels = rng.integers(0, L + 1, size=N_rh, dtype=np.int64).reshape(-1, 1)

    out_path = tmp_path / "group.mat"
    savemat(str(out_path), {
        "mtc": mtc,
        "epsil": epsil,
        "lh_labels": lh_labels,
        "rh_labels": rh_labels,
    })

    out = load_group_mtc(out_path)

    assert isinstance(out, dict)
    assert set(out.keys()) >= {"mtc", "epsil", "lh_labels", "rh_labels"}

    # mtc verbatim, fp64, exact bytewise.
    assert out["mtc"].shape == (Dp1, L)
    assert out["mtc"].dtype == np.float64
    np.testing.assert_array_equal(out["mtc"], mtc)

    # epsil collapses (1, 1) to a python float.
    assert isinstance(out["epsil"], float)
    assert out["epsil"] == pytest.approx(0.12345, rel=0, abs=0)

    # labels flatten to 1-D int.
    assert out["lh_labels"].shape == (N_lh,)
    assert out["rh_labels"].shape == (N_rh,)
    np.testing.assert_array_equal(out["lh_labels"], lh_labels.ravel())
    np.testing.assert_array_equal(out["rh_labels"], rh_labels.ravel())


def test_load_group_mtc_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_group_mtc(tmp_path / "does_not_exist.mat")


def test_load_group_mtc_missing_field(tmp_path: Path) -> None:
    """Missing-field errors surface as KeyError, not a silent partial result."""
    p = tmp_path / "no_mtc.mat"
    savemat(str(p), {
        # mtc deliberately absent.
        "epsil": np.array([[0.1]]),
        "lh_labels": np.zeros((3, 1), dtype=np.int64),
        "rh_labels": np.zeros((3, 1), dtype=np.int64),
    })
    with pytest.raises(KeyError):
        load_group_mtc(p)
