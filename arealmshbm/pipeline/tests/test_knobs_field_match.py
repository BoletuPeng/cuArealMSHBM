# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Field-name 1:1 contract between Step{N}Knobs, the leaf
Step{N}Config / step1_runners.*, and lib/hyperparameters/step{N}.json.

Three layers must stay in lockstep:

  1. Step{N}Knobs (in arealmshbm.pipeline.config) — what the parser
     fills from pipeline_config.json.
  2. Leaf Step{N}Config (or step1_runners signatures) — what the
     driver passes the parsed Knobs into.
  3. lib/hyperparameters/step{N}.json — what the project creator UI
     reads to know which knobs to expose and with what range / default.

These tests pin all three pairs. The driver splats Knobs straight
through to the leaf via ``**asdict(k{N})``, so a name divergence
(layer 1 ↔ 2) TypeErrors at runtime; a divergence between catalog
JSON and Knobs (layer 1 ↔ 3) leaves the UI either missing knobs or
referencing phantom ones. We catch both at PR time.

The one explicit-exception name divergence is documented in
``test_step1_radius_mask_radius_mm_maps_to_run_radius_mask`` (catalog
uses the longer name; runner uses ``radius_mm``).

The ``enable_tf32`` knob in Step0Knobs is consumed by the driver's
tf32_scope wrapper (NOT by Step0Config), so it is excluded from the
Knobs↔leaf-Config comparison via ``_NON_LEAF_CONFIG_FIELDS`` below.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from arealmshbm.pipeline.config import (
    Step0Knobs,
    Step1Knobs,
    Step2Knobs,
    Step3Knobs,
)
from arealmshbm.pipeline import step1_runners
from arealmshbm.step0_pipeline.config import Step0Config
from arealmshbm.step2_pipeline.config import Step2Config
from arealmshbm.step3_pipeline.config import Step3Config


# Knobs fields that the driver consumes directly (e.g. wrapping the
# leaf pipeline in a context manager) rather than splatting through
# to the leaf Config. Excluded from the Knobs ↔ leaf-Config name match.
_NON_LEAF_CONFIG_FIELDS = frozenset({"enable_tf32"})


def _find_repo_root() -> Path:
    """Walk up from this test file looking for a marker."""
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / ".git").exists() or (ancestor / "pyproject.toml").exists():
            return ancestor
    raise RuntimeError(
        f"could not locate repo root above {here} "
        "(neither .git nor pyproject.toml found)"
    )


def _catalog_step_keys(section: str) -> set[str]:
    """Load lib/hyperparameters/<section>.json, extract `entries[*].key`
    that begin with f"{section}." (i.e. the in-block knobs), and return
    them stripped of the section prefix.

    Top-level cross-cutters appearing in the same JSON (e.g.
    ``backend_step0`` in step0.json) are filtered out — they don't
    live in Step{N}Knobs, they live on PipelineConfig directly.
    """
    repo = _find_repo_root()
    path = repo / "lib" / "hyperparameters" / f"{section}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: set[str] = set()
    prefix = f"{section}."
    for entry in data["entries"]:
        k = entry["key"]
        if k.startswith(prefix):
            out.add(k[len(prefix):])
    return out


# ─────────────────────────────────────────────────────────────────────
# Step 0 — Knobs ↔ leaf Config name match
# ─────────────────────────────────────────────────────────────────────
def test_step0_knobs_field_names_match_step0config() -> None:
    """Every Step0Knobs algorithm field (excluding driver-only fields
    like enable_tf32) must exist as a Step0Config field, so the
    driver's `**asdict(k0)` splat is safe after popping the driver-
    only fields."""
    knobs = set(Step0Knobs.__dataclass_fields__) - _NON_LEAF_CONFIG_FIELDS
    cfg = set(Step0Config.__dataclass_fields__)
    missing = knobs - cfg
    assert not missing, (
        f"Step0Knobs has algorithm field(s) {missing} not present on "
        f"Step0Config — driver's `**asdict(k0)` splat would TypeError "
        f"at runtime."
    )


# ─────────────────────────────────────────────────────────────────────
# Step 1 — runners (no Step1Config dataclass)
# ─────────────────────────────────────────────────────────────────────
def test_step1_threshold_param_exists_on_runner() -> None:
    assert "threshold" in inspect.signature(
        step1_runners.run_generate_profiles).parameters


def test_step1_split_flag_param_exists_on_runner() -> None:
    assert "split_flag" in inspect.signature(
        step1_runners.run_generate_profiles).parameters


def test_step1_profile_dtype_param_exists_on_runner() -> None:
    assert "profile_dtype" in inspect.signature(
        step1_runners.run_ini_params).parameters


def test_step1_reduction_dtype_param_exists_on_runner() -> None:
    assert "reduction_dtype" in inspect.signature(
        step1_runners.run_ini_params).parameters


def test_step1_radius_mask_radius_mm_maps_to_run_radius_mask() -> None:
    """The runner parameter is named ``radius_mm`` (not
    ``radius_mask_radius_mm``); the driver maps explicitly. This test
    pins the name on the runner side so a rename surfaces here rather
    than at first run. The runner-side comment in
    step1_runners.run_radius_mask documents the mapping."""
    assert "radius_mm" in inspect.signature(
        step1_runners.run_radius_mask).parameters


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Knobs ↔ leaf Config name match
# ─────────────────────────────────────────────────────────────────────
def test_step2_knobs_field_names_match_step2config() -> None:
    knobs = set(Step2Knobs.__dataclass_fields__) - _NON_LEAF_CONFIG_FIELDS
    cfg = set(Step2Config.__dataclass_fields__)
    missing = knobs - cfg
    assert not missing, (
        f"Step2Knobs has algorithm field(s) {missing} not present on "
        f"Step2Config — driver's `**asdict(k2)` splat would TypeError "
        f"at runtime."
    )


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Knobs ↔ leaf Config name match
# ─────────────────────────────────────────────────────────────────────
def test_step3_knobs_field_names_match_step3config() -> None:
    knobs = set(Step3Knobs.__dataclass_fields__) - _NON_LEAF_CONFIG_FIELDS
    cfg = set(Step3Config.__dataclass_fields__)
    missing = knobs - cfg
    assert not missing, (
        f"Step3Knobs has algorithm field(s) {missing} not present on "
        f"Step3Config — driver's `**asdict(k3)` splat would TypeError "
        f"at runtime."
    )


# ─────────────────────────────────────────────────────────────────────
# Catalog ↔ Knobs entry-key 1:1 — drift detection
#
# The catalog is docs-only (correct call: pipeline doesn't read it at
# runtime). But nothing was previously pinning catalog entry keys to
# Knobs field names. Reviewer flagged: rename `smooth_sigma` in
# Step0Knobs and every other test still passes while step0.json
# references a phantom field. These tests close the loop.
# ─────────────────────────────────────────────────────────────────────
def test_catalog_step0_keys_match_step0knobs() -> None:
    catalog = _catalog_step_keys("step0")
    knobs = set(Step0Knobs.__dataclass_fields__)
    extra = catalog - knobs
    missing = knobs - catalog
    assert not extra, (
        f"lib/hyperparameters/step0.json declares step0.* knob(s) "
        f"{extra} that don't exist on Step0Knobs (phantom field — "
        f"rename in Knobs and update the catalog)."
    )
    assert not missing, (
        f"Step0Knobs has field(s) {missing} not in "
        f"lib/hyperparameters/step0.json — the UI won't expose them. "
        f"Add an entry to the catalog."
    )


def test_catalog_step1_keys_match_step1knobs() -> None:
    catalog = _catalog_step_keys("step1")
    knobs = set(Step1Knobs.__dataclass_fields__)
    extra = catalog - knobs
    missing = knobs - catalog
    assert not extra, (
        f"lib/hyperparameters/step1.json declares step1.* knob(s) "
        f"{extra} that don't exist on Step1Knobs."
    )
    assert not missing, (
        f"Step1Knobs has field(s) {missing} not in step1.json catalog."
    )


def test_catalog_step2_keys_match_step2knobs() -> None:
    catalog = _catalog_step_keys("step2")
    knobs = set(Step2Knobs.__dataclass_fields__)
    extra = catalog - knobs
    missing = knobs - catalog
    assert not extra, (
        f"lib/hyperparameters/step2.json declares step2.* knob(s) "
        f"{extra} that don't exist on Step2Knobs."
    )
    assert not missing, (
        f"Step2Knobs has field(s) {missing} not in step2.json catalog."
    )


def test_catalog_step3_keys_match_step3knobs() -> None:
    catalog = _catalog_step_keys("step3")
    knobs = set(Step3Knobs.__dataclass_fields__)
    extra = catalog - knobs
    missing = knobs - catalog
    assert not extra, (
        f"lib/hyperparameters/step3.json declares step3.* knob(s) "
        f"{extra} that don't exist on Step3Knobs."
    )
    assert not missing, (
        f"Step3Knobs has field(s) {missing} not in step3.json catalog."
    )
