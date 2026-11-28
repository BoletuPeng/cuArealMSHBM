"""test_session_ctor.py — what ``Step2EmIterSession`` refuses up front.

gMSHBM needs at least one gradient component. The session keeps a
width-1 placeholder when ``D_grad`` is 0, which is dMSHBM's case only:
a gMSHBM call with ``D_grad < 1`` is refused before any buffer exists.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arealmshbm.step2_em_iter_master.session import (
    _validate_session_ctor_args,
)

S, T, N, D, L = 1, 2, 4, 3, 2


def _args(**over):
    args = dict(
        mode="gMSHBM",
        bold_loader=SimpleNamespace(num_sub=S, dims=lambda: (N, T, D)),
        grad_loader=SimpleNamespace(dims=lambda: (N, 5)),
        num_sub=S, N=N, T=T, D=D, D_grad=5, num_clusters=L, n_lh=N // 2,
        boundary_mask=np.zeros((N, L), dtype=np.float32),
        s_psi=np.zeros((S, L, D), dtype=np.float32),
        sigma=np.zeros(L, dtype=np.float32),
    )
    args.update(over)
    return args


def test_valid_gmshbm_args_pass():
    _validate_session_ctor_args(**_args())


def test_gmshbm_refuses_zero_width_gradient():
    with pytest.raises(ValueError, match="requires D_grad >= 1"):
        _validate_session_ctor_args(**_args(
            D_grad=0, grad_loader=SimpleNamespace(dims=lambda: (N, 0))))
