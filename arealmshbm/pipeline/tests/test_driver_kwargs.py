# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""AST static checks: every keyword arg the driver passes to
``Step{0,2,3}Config(...)`` and the four step1 runners must match a
real field / parameter of the callee.

This is the static replacement for the byte-identity manual smoke item
in the PR test plan. The field-match test (test_knobs_field_match)
pins Knobs↔leaf field-name match; the threading test pins that every
Knobs field is referenced. Neither catches a *typo on the kwarg name
itself* like

    Step2Config(
        ...,
        intra_em_eps=k2.intra_em_convergence_eps,
                 # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ valid Knobs field,
                 # threading test passes,
                 # but `intra_em_eps` is not a Step2Config field —
                 # only the runtime TypeError at driver invocation
                 # would surface this.

These AST tests parse the driver source, find each Call to the
relevant callee by name, extract its keyword argument names, and
assert each one is a valid dataclass field / function parameter. The
typo above would fail
``test_driver_step2config_kwargs_valid`` at PR time.

Limitations
-----------
- ``ast`` matches by the callee's *name* (``ast.Name(id=...)``), so a
  call written as ``arealmshbm.step3_pipeline.config.Step3Config(...)``
  via attribute access would be invisible. Today the driver imports
  ``Step3Config`` by name; if a future refactor switches to attribute
  access, extend ``_kwargs_in_calls`` to also visit
  ``ast.Attribute`` calls.
- Splatted kwargs (``**dataclasses.asdict(k0)``) appear as
  ``ast.keyword(arg=None, ...)`` and are intentionally skipped — the
  threading + field-match tests cover them.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Iterable

from arealmshbm.pipeline import driver as driver_module
from arealmshbm.pipeline import _step0_stage_pipeline as step0_stage_module
from arealmshbm.pipeline import step1_runners
from arealmshbm.step0_pipeline.config import Step0Config
from arealmshbm.step2_pipeline.config import Step2Config
from arealmshbm.step3_pipeline.config import Step3Config


def _kwargs_in_calls(sources: Iterable[str], callee_name: str) -> set[str]:
    """Aggregate the keyword-arg names of every direct Call to
    ``callee_name(...)`` across the given source strings.

    ``**splat`` keyword args (``kw.arg is None``) are skipped — the
    threading + field-match tests cover them.
    """
    kwargs: set[str] = set()
    for src in sources:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == callee_name):
                continue
            for kw in node.keywords:
                if kw.arg is not None:
                    kwargs.add(kw.arg)
    return kwargs


def _src(*modules) -> list[str]:
    return [inspect.getsource(m) for m in modules]


# ─────────────────────────────────────────────────────────────────────
# Step{N}Config kwargs from driver
# ─────────────────────────────────────────────────────────────────────
def test_driver_step0config_kwargs_valid() -> None:
    # Step0Config is built at two sites: driver._run_step0_all_subjects
    # (the seed_cfg) and _step0_stage_pipeline._stage_A (the per-subject
    # cfg). Both must match.
    kwargs = _kwargs_in_calls(
        _src(driver_module, step0_stage_module), "Step0Config")
    assert kwargs, "expected at least one Step0Config(...) call"
    valid = set(Step0Config.__dataclass_fields__.keys())
    invalid = kwargs - valid
    assert not invalid, (
        f"Step0Config(...) called with kwargs not in its dataclass: "
        f"{invalid}. Either fix the kwarg name or add a Step0Config field."
    )


def test_driver_step2config_kwargs_valid() -> None:
    kwargs = _kwargs_in_calls(_src(driver_module), "Step2Config")
    assert kwargs, "expected at least one Step2Config(...) call"
    valid = set(Step2Config.__dataclass_fields__.keys())
    invalid = kwargs - valid
    assert not invalid, (
        f"Step2Config(...) called with kwargs not in its dataclass: "
        f"{invalid}. Either fix the kwarg name or add a Step2Config field."
    )


def test_driver_step3config_kwargs_valid() -> None:
    kwargs = _kwargs_in_calls(_src(driver_module), "Step3Config")
    assert kwargs, "expected at least one Step3Config(...) call"
    valid = set(Step3Config.__dataclass_fields__.keys())
    invalid = kwargs - valid
    assert not invalid, (
        f"Step3Config(...) called with kwargs not in its dataclass: "
        f"{invalid}. Either fix the kwarg name or add a Step3Config field."
    )


# ─────────────────────────────────────────────────────────────────────
# step1 runner kwargs from driver
# ─────────────────────────────────────────────────────────────────────
def _check_runner_kwargs(callee_name: str) -> None:
    """Assert every kwarg passed to ``callee_name(...)`` in the driver
    matches a parameter of the actual function."""
    func = getattr(step1_runners, callee_name)
    kwargs = _kwargs_in_calls(_src(driver_module), callee_name)
    assert kwargs, f"expected at least one {callee_name}(...) call"
    valid = set(inspect.signature(func).parameters.keys())
    invalid = kwargs - valid
    assert not invalid, (
        f"{callee_name}(...) called with kwargs not in its signature: "
        f"{invalid}. Either fix the kwarg name or add a {callee_name} "
        f"parameter."
    )


def test_driver_run_generate_profiles_kwargs_valid() -> None:
    _check_runner_kwargs("run_generate_profiles")


def test_driver_run_avg_profiles_kwargs_valid() -> None:
    _check_runner_kwargs("run_avg_profiles")


def test_driver_run_ini_params_kwargs_valid() -> None:
    _check_runner_kwargs("run_ini_params")


def test_driver_run_radius_mask_kwargs_valid() -> None:
    _check_runner_kwargs("run_radius_mask")


def test_driver_resolve_group_labels_kwargs_valid() -> None:
    _check_runner_kwargs("resolve_group_labels")
