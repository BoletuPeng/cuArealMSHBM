# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Unit tests for the ``pipeline_config.json`` reader + validator.

The parser is the single entry point for every project run, so its
failure modes (what gets rejected, with what message) are the contract
the user sees first. We pin the surface here so silent regressions get
caught at PR time instead of during a real run.

Schema v2 is strict-mode: every algorithm-relevant knob is required,
every step block required by the mode is required, and unknown keys
are rejected. There is no default-fallback path.

Covered:
  - v1 schema_version rejection with migration hint
  - Mode-aware block presence (Mode A: step0/1/3 required, step2 forbidden;
    Mode B: all four required)
  - Missing required top-level fields → raises with field name
  - Missing required step block → raises with mode + block name
  - Missing required field within a step block → raises with block.key
  - num_clusters enum quantization (350 rejected, 100.0 rejected too)
  - step1.split_flag enum (only "0")
  - step3.connect_th == null survives parse (dMSHBM contract)
  - Unknown-key rejection at top-level + each step{N} block
  - Strict bool/int/float type guards
  - cMSHBM × modeB_train_prior → NotImplementedError
  - Per-step validator range / positivity checks
  - Non-default round-trip across all four step blocks
  - emb_output_format / bold_cache_mode enum rejection at parse time
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from arealmshbm.pipeline.config import (
    PipelineConfig,
    Step0Knobs,
    Step1Knobs,
    Step2Knobs,
    Step3Knobs,
    read_pipeline_config,
)


def _find_repo_root() -> Path:
    """Walk up from this test file looking for a marker (``.git`` or
    ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / ".git").exists() or (ancestor / "pyproject.toml").exists():
            return ancestor
    raise RuntimeError(
        f"could not locate repo root above {here} "
        "(neither .git nor pyproject.toml found)"
    )


# ─────────────────────────────────────────────────────────────────────
# Fixtures — fully-populated configs (the only kind v2 accepts)
# ─────────────────────────────────────────────────────────────────────
def _complete_step0() -> dict:
    """Canonical step0 block — every field present."""
    return {
        "sub_FC": 100, "sub_verts": 200,
        "block_a": 3, "block_b": 10,
        "smooth_sigma": 2.55, "K_hop": 3,
        "watershed_steps": 50, "watershed_frac": 1.0,
        "downsample": 3.2,
        "emb_output_format": "npy",
        "save_geodesic_distance": False,
        "save_edge_density": False,
        "enable_tf32": False,
    }


def _complete_step1() -> dict:
    return {
        "threshold": 0.1,
        "split_flag": "0",
        "profile_dtype": "float64",
        "reduction_dtype": "float64",
        "radius_mask_radius_mm": 30.0,
    }


def _complete_step2() -> dict:
    return {
        "max_iter_inter": 10,
        "max_iter_intra_em": 15,
        "max_iter_em": 100,
        "max_iter_m": 50,
        "max_iter_intra_var": 20,
        "epsilon": 0.0001,
        "intra_em_convergence_eps": 0.0001,
        "inter_convergence_eps": 1e-05,
        "em_convergence_eps": 0.0001,
        "bold_cache_mode": "auto",
        "gpu_cache_safety_margin_gb": 4.0,
        "verbose": True,
    }


def _complete_step3() -> dict:
    return {
        "connect_th": 15.0,
        "epsilon": 0.0001,
        "max_iter_intra_em": 50,
        "max_iter_em": 100,
        "max_iter_lambda": 50,
        "max_iter_m": 50,
        "max_iter_comp": 15,
        "cMSHBM_isolated_component_min_size": 5,
    }


def _top_level_modeB() -> dict:
    """Campaign-spanning fields + schema_version (Mode B / gMSHBM).
    the enable_tf32 toggle lives INSIDE the step0 block, not here."""
    return {
        "schema_version": "2",
        "mode": "modeB_train_prior",
        "variant": "gMSHBM",
        "num_clusters": 300,
        "beta_scalar": 5,
        "w": 50,
        "c": 10,
        "n_grad_components": 100,
        "backend_step0": "cpu",
        "backend_step1": "cpu",
        "backend_step2": "cpu",
        "backend_step3": "cpu",
    }


def _full_v2_modeB() -> dict:
    """Complete v2 Mode B config — every field set to canonical values."""
    return {
        **_top_level_modeB(),
        "step0": _complete_step0(),
        "step1": _complete_step1(),
        "step2": _complete_step2(),
        "step3": _complete_step3(),
    }


def _full_v2_modeA_single() -> dict:
    """Complete v2 Mode A single-subject config — no step2 block."""
    body = _top_level_modeB()
    body["mode"] = "modeA_single"
    return {
        **body,
        "step0": _complete_step0(),
        "step1": _complete_step1(),
        "step3": _complete_step3(),
    }


def _write_config(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "pipeline_config.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────
# Schema-version handling
# ─────────────────────────────────────────────────────────────────────
def test_v1_rejected_with_migration_hint(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["schema_version"] = "1"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError) as excinfo:
        read_pipeline_config(p)
    msg = str(excinfo.value)
    assert "schema_version '1'" in msg
    assert "step0" in msg and "step3" in msg
    assert "sample_modeB" in msg
    assert "sample_modeA_single" in msg
    assert "no default fallback" in msg


def test_unknown_schema_version_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["schema_version"] = "99"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"schema_version must be '2'"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Mode-aware block presence
# ─────────────────────────────────────────────────────────────────────
def test_modeB_complete_config_parses(tmp_path: Path) -> None:
    cfg = read_pipeline_config(_write_config(tmp_path, _full_v2_modeB()))
    assert isinstance(cfg, PipelineConfig)
    assert cfg.mode == "modeB_train_prior"
    assert cfg.step2 is not None
    assert isinstance(cfg.step0, Step0Knobs)
    assert isinstance(cfg.step1, Step1Knobs)
    assert isinstance(cfg.step2, Step2Knobs)
    assert isinstance(cfg.step3, Step3Knobs)


def test_modeA_single_complete_config_parses(tmp_path: Path) -> None:
    cfg = read_pipeline_config(_write_config(tmp_path, _full_v2_modeA_single()))
    assert cfg.mode == "modeA_single"
    assert cfg.step2 is None
    assert isinstance(cfg.step0, Step0Knobs)
    assert isinstance(cfg.step1, Step1Knobs)
    assert isinstance(cfg.step3, Step3Knobs)


def test_modeA_with_step2_block_rejected(tmp_path: Path) -> None:
    """Mode A doesn't run step2; the parser catches this with a
    targeted Mode-A-specific hint (not the generic unknown-key error).
    The hint must name both sample_modeA_single and sample_modeA_batch
    so a user with a Mode B config they're adapting to Mode A has a
    template to copy from."""
    body = _full_v2_modeA_single()
    body["step2"] = _complete_step2()  # extraneous for Mode A
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError) as excinfo:
        read_pipeline_config(p)
    msg = str(excinfo.value)
    assert "does not run step2" in msg
    assert "modeA_single" in msg
    assert "sample_modeA_single" in msg
    assert "sample_modeA_batch" in msg


def test_modeB_missing_step2_block_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    del body["step2"]
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"mode='modeB_train_prior'.*step2"):
        read_pipeline_config(p)


def test_modeA_missing_step3_block_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeA_single()
    del body["step3"]
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"mode='modeA_single'.*step3"):
        read_pipeline_config(p)


def test_missing_step0_block_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    del body["step0"]
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step0"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Missing required field within a step block
# ─────────────────────────────────────────────────────────────────────
def test_missing_step0_field_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    del body["step0"]["smooth_sigma"]
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"missing required key 'step0\.smooth_sigma'"):
        read_pipeline_config(p)


def test_missing_step3_connect_th_rejected(tmp_path: Path) -> None:
    """connect_th's key must be present even though its value can be null."""
    body = _full_v2_modeB()
    del body["step3"]["connect_th"]
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"missing required key 'step3\.connect_th'"):
        read_pipeline_config(p)


def test_missing_top_level_beta_scalar_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    del body["beta_scalar"]
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"missing required key 'beta_scalar'"):
        read_pipeline_config(p)


def test_missing_step0_enable_tf32_rejected(tmp_path: Path) -> None:
    """enable_tf32 now lives inside step0 block, not at top level.
    Omitting it raises the nested-key error."""
    body = _full_v2_modeB()
    del body["step0"]["enable_tf32"]
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"missing required key 'step0\.enable_tf32'"):
        read_pipeline_config(p)


def test_step2_enable_tf32_is_an_unknown_key(tmp_path: Path) -> None:
    """Step 2 lost its TF32 toggle with the dense CuPy port (its GPU
    backend makes one cuBLAS call per run); a config still carrying the
    key is refused with a message that says so, not a bare unknown-key
    error (every pre-fold Mode-B config on disk carries it)."""
    body = _full_v2_modeB()
    body["step2"]["enable_tf32"] = False
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError,
                       match=r"step2\.enable_tf32 was removed.*Delete the key"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Enum quantization on top-level fields
# ─────────────────────────────────────────────────────────────────────
def test_num_clusters_enum_rejects_non_atlas(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["num_clusters"] = 350
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError) as excinfo:
        read_pipeline_config(p)
    msg = str(excinfo.value)
    assert "num_clusters" in msg
    assert "100" in msg and "1000" in msg


def test_step1_split_flag_rejects_nonzero(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step1"]["split_flag"] = "1"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step1\.split_flag"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# connect_th's number|null polymorphism
# ─────────────────────────────────────────────────────────────────────
def test_step3_connect_th_null_survives(tmp_path: Path) -> None:
    """Mode B + dMSHBM is the canonical null case."""
    body = _full_v2_modeB()
    body["variant"] = "dMSHBM"  # dMSHBM doesn't use connect_th
    body["step3"]["connect_th"] = None
    cfg = read_pipeline_config(_write_config(tmp_path, body))
    assert cfg.step3.connect_th is None


def test_step3_connect_th_int_widens_to_float(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step3"]["connect_th"] = 20
    cfg = read_pipeline_config(_write_config(tmp_path, body))
    assert cfg.step3.connect_th == 20.0
    assert isinstance(cfg.step3.connect_th, float)


def test_step3_connect_th_rejects_bool(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step3"]["connect_th"] = True
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"connect_th.*bool"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Unknown-key rejection
# ─────────────────────────────────────────────────────────────────────
def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step_0"] = {}  # typo: should be "step0"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"unknown key.*top level.*step_0"):
        read_pipeline_config(p)


def test_unknown_step0_key_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step0"]["smooth_simga"] = 3.0  # typo
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"unknown key.*step0.*smooth_simga"):
        read_pipeline_config(p)


def test_unknown_step1_key_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step1"]["thresholds"] = 0.2  # typo
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"unknown key.*step1"):
        read_pipeline_config(p)


def test_unknown_step2_key_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step2"]["max_iter_inner"] = 5  # typo
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"unknown key.*step2"):
        read_pipeline_config(p)


def test_unknown_step3_key_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step3"]["connect_threshold"] = 15.0  # typo
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"unknown key.*step3"):
        read_pipeline_config(p)


def test_comment_field_rejected(tmp_path: Path) -> None:
    """Underscore-prefixed comment fields are also rejected — round-1
    contract. Project creator must keep the config clean."""
    body = _full_v2_modeB()
    body["_comment"] = "this is a note"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"unknown key.*_comment"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Strict bool/int/float type guards on nested blocks
# ─────────────────────────────────────────────────────────────────────
def test_step0_save_geodesic_distance_rejects_string(tmp_path: Path) -> None:
    """Stringified "false" must NOT silently flip to True."""
    body = _full_v2_modeB()
    body["step0"]["save_geodesic_distance"] = "false"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step0\.save_geodesic_distance.*bool"):
        read_pipeline_config(p)


def test_step2_verbose_rejects_string(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step2"]["verbose"] = "true"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step2\.verbose.*bool"):
        read_pipeline_config(p)


def test_step0_smooth_sigma_rejects_bool(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step0"]["smooth_sigma"] = True
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step0\.smooth_sigma.*bool"):
        read_pipeline_config(p)


def test_step2_max_iter_em_rejects_bool(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step2"]["max_iter_em"] = True
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step2\.max_iter_em.*bool"):
        read_pipeline_config(p)


def test_step3_epsilon_int_widens_to_float(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step3"]["epsilon"] = 1
    cfg = read_pipeline_config(_write_config(tmp_path, body))
    assert cfg.step3.epsilon == 1.0
    assert isinstance(cfg.step3.epsilon, float)


def test_step2_epsilon_rejects_string(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step2"]["epsilon"] = "0.001"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step2\.epsilon"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Top-level type guards
# ─────────────────────────────────────────────────────────────────────
def test_step0_enable_tf32_rejects_non_bool(tmp_path: Path) -> None:
    """enable_tf32 moved into step block (2026-06); rejects non-bool
    with the nested-key error prefix."""
    body = _full_v2_modeB()
    body["step0"]["enable_tf32"] = "true"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step0\.enable_tf32.*bool"):
        read_pipeline_config(p)


def test_beta_scalar_rejects_bool(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["beta_scalar"] = True
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"beta_scalar.*bool"):
        read_pipeline_config(p)


def test_w_negative_rejected(tmp_path: Path) -> None:
    """Top-level w >= 0 contract — w is a weight, negative is nonsense."""
    body = _full_v2_modeB()
    body["w"] = -1
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"\bw must be >= 0"):
        read_pipeline_config(p)


def test_c_negative_rejected(tmp_path: Path) -> None:
    """Top-level c >= 0 contract — c is a weight, negative is nonsense."""
    body = _full_v2_modeB()
    body["c"] = -3
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"\bc must be >= 0"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Mode × variant constraint
# ─────────────────────────────────────────────────────────────────────
def test_cmshbm_modeb_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["variant"] = "cMSHBM"
    p = _write_config(tmp_path, body)
    with pytest.raises(NotImplementedError, match=r"cMSHBM"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Sample project sanity (the in-tree templates parse)
# ─────────────────────────────────────────────────────────────────────
def test_sample_modeB_parses() -> None:
    repo_root = _find_repo_root()
    p = repo_root / "projects" / "sample_modeB" / "pipeline_config.json"
    if not p.exists():
        pytest.skip(f"sample_modeB not present at {p}")
    cfg = read_pipeline_config(p)
    assert cfg.mode == "modeB_train_prior"
    assert cfg.variant == "gMSHBM"
    assert cfg.num_clusters == 300
    assert cfg.step2 is not None  # Mode B has step2


def test_sample_modeA_single_parses() -> None:
    repo_root = _find_repo_root()
    p = repo_root / "projects" / "sample_modeA_single" / "pipeline_config.json"
    if not p.exists():
        pytest.skip(f"sample_modeA_single not present at {p}")
    cfg = read_pipeline_config(p)
    assert cfg.mode == "modeA_single"
    assert cfg.step2 is None  # Mode A omits step2
    # connect_th was changed to explicit 15.0 (gMSHBM canonical) per
    # the round-2 reviewer feedback — exercises the "project creator
    # writes concrete number, __post_init__ resolution branch unused"
    # contract.
    assert cfg.step3.connect_th == 15.0


def test_sample_modeA_batch_parses() -> None:
    repo_root = _find_repo_root()
    p = repo_root / "projects" / "sample_modeA_batch" / "pipeline_config.json"
    if not p.exists():
        pytest.skip(f"sample_modeA_batch not present at {p}")
    cfg = read_pipeline_config(p)
    assert cfg.mode == "modeA_batch"
    assert cfg.step2 is None
    assert cfg.step3.connect_th == 15.0


# ─────────────────────────────────────────────────────────────────────
# Strict-type enum guards (top-level)
# ─────────────────────────────────────────────────────────────────────
def test_num_clusters_rejects_float(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["num_clusters"] = 100.0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"num_clusters.*int.*float"):
        read_pipeline_config(p)


def test_num_clusters_rejects_bool(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["num_clusters"] = True
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"num_clusters.*bool"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Enum checks at parser layer (not validator)
# ─────────────────────────────────────────────────────────────────────
def test_step0_emb_output_format_rejects_invalid(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step0"]["emb_output_format"] = "hdf5"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step0\.emb_output_format.*one of"):
        read_pipeline_config(p)


def test_step1_profile_dtype_rejects_invalid(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step1"]["profile_dtype"] = "float16"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step1\.profile_dtype.*one of"):
        read_pipeline_config(p)


def test_step2_bold_cache_mode_rejects_invalid(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step2"]["bold_cache_mode"] = "lazy"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step2\.bold_cache_mode.*one of"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Validator range / positivity checks
# ─────────────────────────────────────────────────────────────────────
def test_step0_smooth_sigma_negative_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step0"]["smooth_sigma"] = -1.0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step0\.smooth_sigma.*positive"):
        read_pipeline_config(p)


def test_step0_sub_FC_zero_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step0"]["sub_FC"] = 0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step0\.sub_FC.*positive"):
        read_pipeline_config(p)


def test_step1_threshold_out_of_range_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step1"]["threshold"] = 1.5
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step1\.threshold.*\[0, 1\]"):
        read_pipeline_config(p)


def test_step1_radius_mask_zero_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step1"]["radius_mask_radius_mm"] = 0.0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step1\.radius_mask_radius_mm.*positive"):
        read_pipeline_config(p)


def test_step2_max_iter_em_zero_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step2"]["max_iter_em"] = 0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step2\.max_iter_em.*positive"):
        read_pipeline_config(p)


def test_step2_gpu_cache_safety_margin_negative_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step2"]["gpu_cache_safety_margin_gb"] = -1.0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step2\.gpu_cache_safety_margin_gb.*>= 0"):
        read_pipeline_config(p)


def test_step3_connect_th_negative_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step3"]["connect_th"] = -5.0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step3\.connect_th.*>= 0"):
        read_pipeline_config(p)


def test_step3_epsilon_zero_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step3"]["epsilon"] = 0.0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"step3\.epsilon.*positive"):
        read_pipeline_config(p)


def test_step3_cMSHBM_min_size_zero_rejected_at_parse(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["step3"]["cMSHBM_isolated_component_min_size"] = 0
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError,
                       match=r"step3\.cMSHBM_isolated_component_min_size.*positive"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# Non-default round-trip — every field set to a non-canonical value
# ─────────────────────────────────────────────────────────────────────
def test_full_v2_roundtrip_non_default(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body.update({
        "num_clusters": 400,
        "beta_scalar": 7,
        "w": 80,
        "c": 20,
        "n_grad_components": 150,
    })
    body["step0"] = {
        "sub_FC": 120, "sub_verts": 250,
        "block_a": 4, "block_b": 12,
        "smooth_sigma": 3.0, "K_hop": 4,
        "watershed_steps": 60, "watershed_frac": 0.8,
        "downsample": 4.0,
        "emb_output_format": "both",
        "save_geodesic_distance": True,
        "save_edge_density": True,
        "enable_tf32": True,
    }
    body["step1"] = {
        "threshold": 0.15,
        "split_flag": "0",
        "profile_dtype": "float32",
        "reduction_dtype": "float32",
        "radius_mask_radius_mm": 25.0,
    }
    body["step2"] = {
        "max_iter_inter": 12,
        "max_iter_intra_em": 20,
        "max_iter_em": 150,
        "max_iter_m": 60,
        "max_iter_intra_var": 25,
        "epsilon": 5e-5,
        "intra_em_convergence_eps": 5e-5,
        "inter_convergence_eps": 5e-6,
        "em_convergence_eps": 5e-5,
        "bold_cache_mode": "eager_bitpacked",
        "gpu_cache_safety_margin_gb": 6.0,
        "verbose": False,
    }
    body["step3"] = {
        "connect_th": 12.0,
        "epsilon": 5e-5,
        "max_iter_intra_em": 60,
        "max_iter_em": 120,
        "max_iter_lambda": 60,
        "max_iter_m": 60,
        "max_iter_comp": 20,
        "cMSHBM_isolated_component_min_size": 8,
    }
    cfg = read_pipeline_config(_write_config(tmp_path, body))

    # Campaign-spanning
    assert cfg.num_clusters == 400
    assert cfg.beta_scalar == 7
    assert (cfg.w, cfg.c) == (80, 20)
    assert cfg.n_grad_components == 150

    # step0 — all 13 (including save_edge_density + enable_tf32)
    s0 = cfg.step0
    assert (s0.sub_FC, s0.sub_verts, s0.block_a, s0.block_b) == (120, 250, 4, 12)
    assert s0.smooth_sigma == 3.0 and s0.K_hop == 4
    assert (s0.watershed_steps, s0.watershed_frac, s0.downsample) == (60, 0.8, 4.0)
    assert s0.emb_output_format == "both"
    assert s0.save_geodesic_distance is True
    assert s0.save_edge_density is True
    assert s0.enable_tf32 is True

    # step1 — all 5
    s1 = cfg.step1
    assert s1.threshold == 0.15 and s1.split_flag == "0"
    assert s1.profile_dtype == "float32" and s1.reduction_dtype == "float32"
    assert s1.radius_mask_radius_mm == 25.0

    # step2 — all 12
    s2 = cfg.step2
    assert s2 is not None
    assert (s2.max_iter_inter, s2.max_iter_intra_em, s2.max_iter_em) == (12, 20, 150)
    assert (s2.max_iter_m, s2.max_iter_intra_var) == (60, 25)
    assert s2.epsilon == 5e-5
    assert s2.intra_em_convergence_eps == 5e-5
    assert s2.inter_convergence_eps == 5e-6
    assert s2.em_convergence_eps == 5e-5
    assert s2.bold_cache_mode == "eager_bitpacked"
    assert s2.gpu_cache_safety_margin_gb == 6.0
    assert s2.verbose is False

    # step3 — all 8
    s3 = cfg.step3
    assert s3.connect_th == 12.0 and s3.epsilon == 5e-5
    assert (s3.max_iter_intra_em, s3.max_iter_em) == (60, 120)
    assert (s3.max_iter_lambda, s3.max_iter_m, s3.max_iter_comp) == (60, 60, 20)
    assert s3.cMSHBM_isolated_component_min_size == 8


# ─────────────────────────────────────────────────────────────────────
# Knobs are now non-default — direct construction without args fails
# ─────────────────────────────────────────────────────────────────────
def test_knobs_cannot_construct_without_args() -> None:
    """Step{N}Knobs() with no args must TypeError under strict mode —
    documents that there is no implicit default fallback."""
    with pytest.raises(TypeError):
        Step0Knobs()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Step1Knobs()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Step2Knobs()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Step3Knobs()  # type: ignore[call-arg]


# ─────────────────────────────────────────────────────────────────────
# num_clusters × backend_step2='gpu' constraint
#
# The GPU init kernel keeps a fixed number of per-lane accumulators;
# ``step2_em_iter_master._kernels_gpu.check_dims`` owns the ceiling. Refusing it here
# means the run does not die in the Session ctor, i.e. after step 0/1
# and the whole cohort's packed-BOLD decode.
# ─────────────────────────────────────────────────────────────────────
_SPARSE_MAX_CLUSTERS = 512
_SPARSE_MAX_D_GRAD = 11772


def test_the_parser_sparse_limits_match_the_kernel_constants() -> None:
    """The parser mirrors them as literals so it stays stdlib-only."""
    pytest.importorskip("cupy")
    from arealmshbm.step2_em_iter_master._kernels_gpu import (
        MAX_CLUSTERS, MAX_D_GRAD,
    )
    assert MAX_CLUSTERS == _SPARSE_MAX_CLUSTERS
    assert MAX_D_GRAD == _SPARSE_MAX_D_GRAD


def test_reading_a_gpu_config_imports_neither_cupy_nor_numba(
        tmp_path: Path) -> None:
    """``Pipeline.__init__`` parses JSON; the kernel stack costs ~1.7 s."""
    body = _full_v2_modeB()
    body["backend_step2"] = "gpu"
    p = _write_config(tmp_path, body)
    code = (
        "import sys, json;"
        "from arealmshbm.pipeline.config import read_pipeline_config;"
        f"read_pipeline_config(r{str(p)!r});"
        "print(json.dumps([m for m in ('cupy', 'numba') if m in sys.modules]))"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


def test_num_clusters_over_the_sparse_limit_rejected(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["backend_step2"] = "gpu"
    body["num_clusters"] = next(v for v in (600, 700, 800, 900, 1000)
                                if v > _SPARSE_MAX_CLUSTERS)
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError,
                       match=r"backend_step2='gpu'.*num_clusters <="):
        read_pipeline_config(p)


def test_num_clusters_at_the_sparse_limit_accepted(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["backend_step2"] = "gpu"
    body["num_clusters"] = max(v for v in (100, 200, 300, 400, 500)
                               if v <= _SPARSE_MAX_CLUSTERS)
    p = _write_config(tmp_path, body)
    assert read_pipeline_config(p).backend_step2 == "gpu"


def test_n_grad_components_over_the_sparse_limit_rejected(
        tmp_path: Path) -> None:
    """``connect_u`` stages one float per component in shared memory."""
    body = _full_v2_modeB()
    body["backend_step2"] = "gpu"
    body["n_grad_components"] = _SPARSE_MAX_D_GRAD + 1
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError,
                       match=r"backend_step2='gpu'.*n_grad_components <="):
        read_pipeline_config(p)
    # At the limit, and on the CPU backend, it parses.
    body["n_grad_components"] = _SPARSE_MAX_D_GRAD
    assert read_pipeline_config(_write_config(tmp_path, body)) is not None
    body["n_grad_components"] = _SPARSE_MAX_D_GRAD + 1
    body["backend_step2"] = "cpu"
    assert read_pipeline_config(_write_config(tmp_path, body)) is not None
    # ``connect_u`` is gMSHBM-only, so the limit is scoped to the variant.
    body["backend_step2"] = "gpu"
    body["variant"] = "dMSHBM"
    assert read_pipeline_config(_write_config(tmp_path, body)) is not None


def test_the_sparse_cluster_limit_is_scoped_to_backend_step2(
        tmp_path: Path) -> None:
    """A big L on the CPU backend is untouched."""
    body = _full_v2_modeB()
    body["num_clusters"] = 1000
    p = _write_config(tmp_path, body)
    assert read_pipeline_config(p).num_clusters == 1000


def test_backend_step2_gpu_sparse_is_not_an_alias(tmp_path: Path) -> None:
    """``'gpu_sparse'`` was the P-layout backend's name while the dense
    CuPy port held ``'gpu'``; it is rejected like any unknown value."""
    body = _full_v2_modeB()
    body["backend_step2"] = "gpu_sparse"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"backend_step2 must be one of"):
        read_pipeline_config(p)


# ─────────────────────────────────────────────────────────────────────
# w=0 × backend_step3='gpu_sparse' constraint
#
# w=0 is not a "prior off" switch: the dense E-step still evaluates
# 0*log(theta)=NaN outside supp(theta), while the candidate-set backend
# only visits supp(theta) and would silently answer differently.
# ─────────────────────────────────────────────────────────────────────
def test_w_zero_rejected_with_gpu_sparse(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["w"] = 0
    body["backend_step3"] = "gpu_sparse"
    p = _write_config(tmp_path, body)
    with pytest.raises(ValueError, match=r"gpu_sparse.*requires\s+w > 0"):
        read_pipeline_config(p)


def test_w_zero_accepted_on_cpu(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["w"] = 0
    p = _write_config(tmp_path, body)
    assert read_pipeline_config(p).w == 0


def test_w_positive_accepted_with_gpu_sparse(tmp_path: Path) -> None:
    body = _full_v2_modeB()
    body["backend_step3"] = "gpu_sparse"
    p = _write_config(tmp_path, body)
    cfg = read_pipeline_config(p)
    assert cfg.w == 50 and cfg.backend_step3 == "gpu_sparse"
