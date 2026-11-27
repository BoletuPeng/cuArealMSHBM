"""Asset contract of the step-0 inputs bundle.

``Step0Pipeline.load_inputs`` reads the shipped bundle and computes
nothing itself: no staged bundle is a hard ``FileNotFoundError``, and a
bundle of another format version a hard ``ValueError``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

import json

import pytest

from arealmshbm.step0_pipeline import pipeline as step0_pipeline
from arealmshbm.step0_pipeline.config import Step0Config
from arealmshbm.step0_pipeline.inputs_bundle import (
    BUNDLE_VERSION, bundle_dir, load_inputs_bundle,
)


def _cfg(tmp_path):
    return Step0Config(project_dir=tmp_path, sub_id="sub-001",
                       sess_list=("ses-01",))


def test_bundle_dir_sits_under_the_precomputed_root(monkeypatch, tmp_path):
    """One fixed directory under the precomputed root the avg_mesh
    bundles share."""
    monkeypatch.setenv("MSHBM_PRECOMPUTED_ROOT", str(tmp_path))
    assert bundle_dir() == tmp_path / "step0_inputs"


def test_load_inputs_missing_bundle_is_hard_error(monkeypatch, tmp_path):
    """No bundle staged: ``load_inputs`` raises, naming the bundle and
    framing it as an asset."""
    monkeypatch.setenv("MSHBM_PRECOMPUTED_ROOT", str(tmp_path))
    # A fresh process memo, so an earlier test's load cannot pre-empt
    # the miss and the miss cannot leak into later tests.
    monkeypatch.setattr(step0_pipeline, "_INPUTS_CPU", None)
    with pytest.raises(FileNotFoundError) as excinfo:
        step0_pipeline.Step0Pipeline(_cfg(tmp_path)).load_inputs()
    msg = str(excinfo.value)
    assert str(tmp_path / "step0_inputs") in msg, \
        "must name the missing bundle"
    assert "asset" in msg, "must frame it as a shipped asset, not a cache"
    assert step0_pipeline._INPUTS_CPU is None, "a miss must not be memoised"


def test_bundle_version_mismatch_is_hard_error(monkeypatch, tmp_path):
    """A staged bundle of another format version is rejected by name."""
    monkeypatch.setenv("MSHBM_PRECOMPUTED_ROOT", str(tmp_path))
    (tmp_path / "step0_inputs").mkdir()
    (tmp_path / "step0_inputs" / "manifest.json").write_text(
        json.dumps({"version": BUNDLE_VERSION + "-other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        load_inputs_bundle()
