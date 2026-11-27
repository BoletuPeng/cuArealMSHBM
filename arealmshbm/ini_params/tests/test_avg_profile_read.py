"""test_avg_profile_read.py — the ``.npy`` read path of ``ini_params``.

Pins:

  * ``_read_profile_concat`` returns lh over rh as one fp64 ``(N, D)``
    matrix, equal to the fp32 files value for value;
  * lh / rh files that are not both ``(V_h, D)`` with one ``D`` are
    rejected — a ``(V_rh, 1)`` file must not broadcast across ``D``.

Run::

    python -m pytest arealmshbm/ini_params/tests/test_avg_profile_read.py -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.ini_params.ini_params import _read_profile_concat


def _stage(tmp_path, lh, rh):
    base = tmp_path / "profiles" / "avg_profile"
    base.mkdir(parents=True)
    np.save(base / "lh_avg_profile.npy", lh)
    np.save(base / "rh_avg_profile.npy", rh)
    return str(tmp_path)


def test_read_stacks_lh_over_rh_in_fp64(tmp_path):
    rng = np.random.default_rng(7)
    lh = rng.random((10, 5), dtype=np.float32)
    rh = rng.random((8, 5), dtype=np.float32)
    out = _read_profile_concat(_stage(tmp_path, lh, rh))
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    assert np.array_equal(out, np.vstack([lh, rh]).astype(np.float64))


@pytest.mark.parametrize("rh_shape", [(8, 1), (8, 4), (8,)])
def test_read_rejects_mismatched_lh_rh(tmp_path, rh_shape):
    lh = np.ones((10, 5), dtype=np.float32)
    rh = np.ones(rh_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="one D"):
        _read_profile_concat(_stage(tmp_path, lh, rh))
