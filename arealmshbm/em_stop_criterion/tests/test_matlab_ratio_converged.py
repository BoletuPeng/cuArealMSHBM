"""Truth table for the shared MATLAB convergence rule.

Pins ``matlab_ratio_converged`` to IEEE fp64 semantics and records the
one cell where it differs from the pre-fix branch implementation.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from arealmshbm.em_stop_criterion import matlab_ratio_converged
from arealmshbm.em_stop_criterion.em_stop_criterion import convergence_test

NAN = float("nan")
INF = float("inf")
VALUES = [0.0, 1.0, -1.0, 1e-5, 1.0001, NAN, INF, -INF]
THRESHOLD = 1e-4


def _reference(update_cost: float, cost: float) -> bool:
    """MATLAB ``abs(abs(update_cost-cost)./cost) > 1e-4`` negated, in IEEE fp64."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs((np.float64(update_cost) - np.float64(cost))
                       / np.float64(cost))
    return not bool(ratio > THRESHOLD)


def _pre_fix(update_cost: float, cost: float) -> bool:
    """The branch implementation this replaced."""
    if cost == 0.0:
        ratio = NAN if update_cost == 0.0 else INF
    else:
        ratio = abs((update_cost - cost) / cost)
    return not (ratio > THRESHOLD)


def _pairs():
    for uc, co in itertools.product(VALUES, VALUES):
        yield uc, co
        # cost carried across an EM iteration is narrowed to fp32 first.
        yield uc, float(np.float32(co))


def test_matches_ieee_reference_everywhere():
    for uc, co in _pairs():
        assert matlab_ratio_converged(uc, co) == _reference(uc, co), (uc, co)


def test_only_nan_over_zero_moved_versus_branch_implementation():
    moved = [(uc, co) for uc, co in _pairs()
             if matlab_ratio_converged(uc, co) != _pre_fix(uc, co)]
    assert all(np.isnan(uc) and co == 0.0 for uc, co in moved)
    # (NaN, 0.0) reached twice — once raw, once fp32-narrowed.
    assert len(moved) == 2


@pytest.mark.parametrize("uc, co, expected", [
    (0.0, 0.0, True),        # 0/0 -> NaN -> converged
    (NAN, 0.0, True),        # NaN/0 -> NaN -> converged (was: not converged)
    (1.0, 0.0, False),       # nonzero/0 -> Inf
    (INF, 0.0, False),
    (-INF, 0.0, False),
    (NAN, 1.0, True),
    (1.0, 1.0, True),
    (1.001, 1.0, False),
    (1.00001, 1.0, True),
])
def test_documented_cells(uc, co, expected):
    assert matlab_ratio_converged(uc, co) is expected


def test_convergence_test_stops_on_nan_cost_zero():
    """(update_cost=NaN, cost=0) converges instead of running another
    EM iteration the CPU M-step cannot survive."""
    stop_em, cost_em = convergence_test(NAN, 0.0, iter_em=1)
    assert stop_em == 1
    assert float(np.asarray(cost_em).ravel()[0]) == 0.0
def test_no_warning_when_the_ratio_overflows():
    """A huge/tiny ratio is Inf (not converged), silently — the
    branch implementation this replaced used Python float arithmetic."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert matlab_ratio_converged(1e300, 1e-300) is False
