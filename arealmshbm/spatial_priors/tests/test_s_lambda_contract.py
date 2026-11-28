"""test_s_lambda_contract.py — what ``s_lambda`` the spatial-prior
sessions take.

``ConnectSession.compute`` / ``XyzSession.compute`` run sgemm straight
on the caller's ``(N, L)`` array, so it must be float32 and contiguous;
anything else is refused with the reason, not converted.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.spatial_priors.spatial_connect import ConnectSession
from arealmshbm.spatial_priors.spatial_xyz import XyzSession

N, L, D_GRAD = 8, 4, 5


def _sessions():
    rng = np.random.default_rng(0)
    connect = ConnectSession(
        rng.standard_normal((N, D_GRAD)).astype(np.float32), N, L)
    xyz = XyzSession(rng.standard_normal((N, 3)).astype(np.float32), L)
    gamma = np.ones(L, dtype=np.float32)
    return [lambda s: connect.compute(s), lambda s: xyz.compute(s, gamma)]


@pytest.mark.parametrize("which", [0, 1], ids=["connect", "xyz"])
def test_refuses_float64(which):
    with pytest.raises(ValueError, match="got float64, contiguous=True"):
        _sessions()[which](np.zeros((N, L), dtype=np.float64))


@pytest.mark.parametrize("which", [0, 1], ids=["connect", "xyz"])
def test_refuses_non_contiguous_float32(which):
    strided = np.zeros((N, 2 * L), dtype=np.float32)[:, ::2]
    with pytest.raises(ValueError, match="got float32, contiguous=False"):
        _sessions()[which](strided)
