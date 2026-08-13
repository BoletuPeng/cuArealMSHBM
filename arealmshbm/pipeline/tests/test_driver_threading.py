# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Tests that every Step{N}Knobs field actually reaches the corresponding
Step{N}Config inside the driver.

The field-match tests (test_knobs_field_match) pin name 1:1 between
Step{N}Knobs fields and Step{N}Config / step1_runners.* signatures —
so a ``**asdict(...)`` splat in the driver wouldn't TypeError at
runtime. But they don't catch the "added a knob to Step{N}Knobs, forgot
to thread it into driver.py" case: a Knobs field could exist and never
reach the leaf config, and the field-match test would still pass.

Approach: source-introspection. For each step{N}, the driver method
(or its split-stage equivalent for step0) must either:
  * splat the Knobs via ``**asdict(...)``, OR
  * reference every Knobs field by name (``k{N}.<field>``).

This catches the misspell-or-forget case at PR time, regardless of
which threading style is used. Adding a new Knobs field will require
either updating the splat (free) or adding the explicit reference
(visible in the diff).
"""
from __future__ import annotations

import inspect

from arealmshbm.pipeline.config import (
    Step0Knobs,
    Step1Knobs,
    Step2Knobs,
    Step3Knobs,
)
from arealmshbm.pipeline.driver import Pipeline
from arealmshbm.pipeline._step0_stage_pipeline import Step0StagePipeline


def _assert_knobs_threaded(src: str, knobs_cls, alias: str) -> None:
    """Assert ``src`` references every ``knobs_cls`` field by name
    (``<alias>.<field>``) OR splats the exact alias via ``asdict(...)``.

    ``alias`` is the local name the driver uses for the Knobs instance
    (e.g. ``k0``, ``self.config.step0``, ``self.step0_knobs``). Earlier
    drafts had a generic ``"asdict(self.config.step" in src`` disjunct
    that matched any of step0/1/2/3, which would silently pass a step3
    threading test if someone wrote ``asdict(self.config.step0)``.
    Splat-check is now scoped to the exact alias only.
    """
    has_splat = f"asdict({alias})" in src
    if has_splat:
        return

    missing = [
        name for name in knobs_cls.__dataclass_fields__
        if f"{alias}.{name}" not in src
    ]
    assert not missing, (
        f"{knobs_cls.__name__} fields not threaded in driver source via "
        f"`{alias}.<field>`: {missing}. Either add explicit "
        f"`{alias}.<field>` references or splat via "
        f"`**asdict({alias})`."
    )


def test_driver_threads_step0_knobs() -> None:
    """``_run_step0_all_subjects`` must thread every Step0Knobs field
    into Step0Config (the seed_cfg construction)."""
    src = inspect.getsource(Pipeline._run_step0_all_subjects)
    _assert_knobs_threaded(src, Step0Knobs, "self.config.step0")


def test_step0_stage_pipeline_threads_step0_knobs() -> None:
    """``Step0StagePipeline._stage_A`` is the second site that builds
    per-subject Step0Configs from Step0Knobs. Must stay in sync."""
    src = inspect.getsource(Step0StagePipeline._stage_A)
    _assert_knobs_threaded(src, Step0Knobs, "self.step0_knobs")


def test_driver_threads_step1_knobs() -> None:
    """``_run_step1_all_subjects`` distributes Step1Knobs across three
    runners (generate_profiles, ini_params, radius_mask)."""
    src = inspect.getsource(Pipeline._run_step1_all_subjects)
    _assert_knobs_threaded(src, Step1Knobs, "k1")


def test_driver_threads_step2_knobs() -> None:
    """``_run_step2_train_prior`` builds a single Step2Config."""
    src = inspect.getsource(Pipeline._run_step2_train_prior)
    _assert_knobs_threaded(src, Step2Knobs, "k2")


def test_driver_threads_step3_knobs() -> None:
    """``_run_step3_all_subjects`` builds one Step3Config per subject."""
    src = inspect.getsource(Pipeline._run_step3_all_subjects)
    _assert_knobs_threaded(src, Step3Knobs, "k3")
