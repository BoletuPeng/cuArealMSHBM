# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Unit tests for ``Step3Config.__post_init__`` validators.

Only covers the cMSHBM_isolated_component_min_size validator added in
PR #52 — the other __post_init__ checks (epsilon > 0, max_iter_* > 0,
mesh in supported set, backend valid, pipeline_type valid, …) are
covered indirectly by every downstream test that constructs a
Step3Config, but the new field has no other coverage.
"""
from __future__ import annotations

import pytest

from arealmshbm.step3_pipeline.config import Step3Config


def _base_kwargs(**overrides):
    """Minimal Step3Config kwargs that pass __post_init__."""
    d = dict(
        project_dir="/tmp/nonexistent",
        num_session=6,
        num_clusters=300,
    )
    d.update(overrides)
    return d


def test_cMSHBM_min_size_zero_rejected() -> None:
    with pytest.raises(ValueError,
                       match=r"cMSHBM_isolated_component_min_size must be positive"):
        Step3Config(**_base_kwargs(cMSHBM_isolated_component_min_size=0))


def test_cMSHBM_min_size_negative_rejected() -> None:
    with pytest.raises(ValueError,
                       match=r"cMSHBM_isolated_component_min_size must be positive"):
        Step3Config(**_base_kwargs(cMSHBM_isolated_component_min_size=-1))


def test_cMSHBM_min_size_positive_accepted() -> None:
    # Default (5) and an explicit override (8) both pass.
    cfg5 = Step3Config(**_base_kwargs())
    assert cfg5.cMSHBM_isolated_component_min_size == 5
    cfg8 = Step3Config(**_base_kwargs(cMSHBM_isolated_component_min_size=8))
    assert cfg8.cMSHBM_isolated_component_min_size == 8


# ─────────────────────────────────────────────────────────────────────
# w > 0 on every backend
#
# At w=0 the E-step's w*log(theta) term is 0*log(0) = NaN outside
# supp(theta) and the parcellation degenerates (stock CBIG does the same).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_w_zero_rejected(backend: str) -> None:
    with pytest.raises(ValueError, match=r"w must be > 0 \(got 0\.0\)"):
        Step3Config(**_base_kwargs(w=0.0, backend=backend))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_w_positive_accepted(backend: str) -> None:
    cfg = Step3Config(**_base_kwargs(w=1.0, backend=backend))
    assert cfg.w == 1.0 and cfg.backend == backend


def test_cMSHBM_accepted_with_gpu() -> None:
    cfg = Step3Config(**_base_kwargs(pipeline_type="cMSHBM", backend="gpu"))
    assert cfg.backend == "gpu" and cfg.variant.pre_predicate_remove_isolated
    assert cfg.connect_th == 0.0
