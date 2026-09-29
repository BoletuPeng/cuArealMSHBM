"""test_driver_prewarm_deps.py — the driver's pre-step-0 dependency checks.

Every ``gpu`` backend needs CuPy, a CUDA device and NVRTC;
``backend_step1='gpu'`` needs nvCOMP; ``backend_step2='gpu'`` needs
``psutil``; ``backend_step3='gpu'`` with a variant that runs
``check_connectedness`` needs the cooperative CC kernel (CUDA toolkit at
``CUDA_PATH`` + a device that supports cooperative launch).
``Pipeline.run()`` checks all of them in its prewarm phase, before it
starts the step-2 prewarm thread, so a missing dependency fails before
step 0, not after steps 0-2. No GPU needed: the GPU-side calls are
stubbed and step 0 is replaced by a sentinel raise.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from arealmshbm.pipeline import driver
from arealmshbm.pipeline.driver import Pipeline, _require_cuda_device

_CC_MOD = "arealmshbm.check_connectedness.connectedness_gpu"


class _ReachedStep0(Exception):
    """Raised in place of step 0: the prewarm block ran to completion."""


def _write_project(root: Path, *, mode_a: bool, variant: str,
                   **backends: str) -> Path:
    """Project tree ``Pipeline.run()`` walks up to the prewarm block:
    the in-tree sample config, empty BOLD files (only their existence is
    checked before step 0) and, for Mode A, an empty prior file."""
    repo_root = Path(__file__).resolve().parents[3]
    sample = repo_root / "projects" / (
        "sample_modeA_single" if mode_a else "sample_modeB")
    project = root / "tiny_project"
    project.mkdir()
    cfg = json.loads(
        (sample / "pipeline_config.json").read_text(encoding="utf-8"))
    cfg["variant"] = variant
    cfg.update(backends)
    (project / "pipeline_config.json").write_text(
        json.dumps(cfg), encoding="utf-8")
    sids = ("1",) if mode_a else ("1", "2")
    for sid in sids:
        for hemi in ("lh", "rh"):
            (project / f"{hemi}.{sid}.func.gii").write_bytes(b"")
    (project / "bold_inputs.json").write_text(json.dumps({
        "schema_version": "1",
        "dataset_name": "prewarm_deps_fixture",
        "targ_mesh": "fsaverage6",
        "seed_mesh": "fsaverage3",
        "subjects": [{"id": sid, "sessions": [
            {"id": "1",
             "lh": str(project / f"lh.{sid}.func.gii"),
             "rh": str(project / f"rh.{sid}.func.gii")},
        ]} for sid in sids],
    }), encoding="utf-8")
    return project


def _pipeline(project: Path) -> Pipeline:
    pipe = Pipeline(project)
    if pipe.config.is_mode_a:
        prior = pipe.layout.prior_path(pipe.config.variant,
                                       pipe.config.beta_scalar,
                                       pipe._prior_includes_beta())
        prior.parent.mkdir(parents=True, exist_ok=True)
        prior.write_bytes(b"")
    return pipe


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub every GPU-side prewarm and stop ``run()`` at step 0.
    ``calls`` records which dependency checks the driver made."""
    calls: dict = {"cuda": 0, "cc": 0, "step2": 0}

    def _cuda():
        calls["cuda"] += 1

    def _step0(self):
        raise _ReachedStep0

    def _cc():
        calls["cc"] += 1

    def _step2(*, background=True):
        calls["step2"] += 1

    monkeypatch.setattr(Pipeline, "_run_step0_all_subjects_maybe_tf32",
                        _step0)
    monkeypatch.setattr(driver, "_require_cuda_device", _cuda)
    fake_cc = types.ModuleType(_CC_MOD)
    fake_cc.prewarm_connectedness_gpu = _cc
    monkeypatch.setitem(sys.modules, _CC_MOD, fake_cc)
    from arealmshbm.pipeline import step2_runners
    monkeypatch.setattr(step2_runners, "prewarm_step2_gpu", _step2)
    monkeypatch.setattr(step2_runners, "join_step2_prewarm",
                        lambda timeout=30.0: None)
    return calls


def test_mode_b_step2_gpu_missing_psutil_fails_before_step0(
        tmp_path: Path, stubs: dict,
        monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _pipeline(_write_project(tmp_path, mode_a=False,
                                    variant="gMSHBM", backend_step2="gpu"))
    monkeypatch.setitem(sys.modules, "psutil", None)
    with pytest.raises(ImportError, match="psutil"):
        pipe.run()
    assert stubs["step2"] == 0


def test_mode_b_step2_gpu_with_psutil_prewarms(
        tmp_path: Path, stubs: dict,
        monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _pipeline(_write_project(tmp_path, mode_a=False,
                                    variant="gMSHBM", backend_step2="gpu"))
    monkeypatch.setitem(sys.modules, "psutil", types.ModuleType("psutil"))
    with pytest.raises(_ReachedStep0):
        pipe.run()
    assert stubs == {"cuda": 1, "cc": 0, "step2": 1}


def test_mode_a_ignores_backend_step2_gpu(
        tmp_path: Path, stubs: dict,
        monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _pipeline(_write_project(tmp_path, mode_a=True,
                                    variant="gMSHBM", backend_step2="gpu"))
    monkeypatch.setitem(sys.modules, "psutil", None)
    with pytest.raises(_ReachedStep0):
        pipe.run()
    assert stubs == {"cuda": 0, "cc": 0, "step2": 0}


@pytest.mark.parametrize("mode_a,variant,backends", [
    (False, "gMSHBM", {"backend_step0": "gpu"}),
    (True, "gMSHBM", {"backend_step1": "gpu"}),
    (False, "gMSHBM", {"backend_step2": "gpu"}),
    (True, "dMSHBM", {"backend_step3": "gpu"}),
    (False, "dMSHBM", {"backend_step3": "gpu"}),
])
def test_gpu_backend_missing_cupy_fails_before_step0(
        tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch,
        mode_a: bool, variant: str, backends: dict) -> None:
    pipe = _pipeline(_write_project(tmp_path, mode_a=mode_a,
                                    variant=variant, **backends))
    monkeypatch.setattr(driver, "_require_cuda_device",
                        _require_cuda_device)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match="cupy"):
        pipe.run()
    assert stubs == {"cuda": 0, "cc": 0, "step2": 0}


@pytest.mark.parametrize("mode_a", [True, False])
def test_step1_gpu_missing_nvcomp_fails_before_step0(
        tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch,
        mode_a: bool) -> None:
    from arealmshbm.data_io import _nvcomp_batched
    pipe = _pipeline(_write_project(tmp_path, mode_a=mode_a,
                                    variant="gMSHBM", backend_step1="gpu",
                                    backend_step2="gpu"))
    monkeypatch.setitem(sys.modules, "psutil", types.ModuleType("psutil"))
    monkeypatch.setitem(sys.modules, "nvidia.nvcomp", None)
    monkeypatch.setattr(_nvcomp_batched, "_CACHE", {})
    with pytest.raises(ImportError, match="nvidia-nvcomp-cu12"):
        pipe.run()
    assert stubs["step2"] == 0


@pytest.mark.parametrize("mode_a,variant", [
    (True, "gMSHBM"), (True, "cMSHBM"), (False, "gMSHBM"),
])
def test_step3_gpu_cc_failure_fails_before_step0(
        tmp_path: Path, stubs: dict, monkeypatch: pytest.MonkeyPatch,
        mode_a: bool, variant: str) -> None:
    pipe = _pipeline(_write_project(tmp_path, mode_a=mode_a,
                                    variant=variant, backend_step2="gpu",
                                    backend_step3="gpu"))

    def _no_coop():
        raise RuntimeError("device 0 does not support cooperative launch")

    monkeypatch.setitem(sys.modules, "psutil", types.ModuleType("psutil"))
    monkeypatch.setattr(sys.modules[_CC_MOD], "prewarm_connectedness_gpu",
                        _no_coop)
    with pytest.raises(RuntimeError, match="cooperative launch"):
        pipe.run()
    assert stubs["step2"] == 0


@pytest.mark.parametrize("mode_a", [True, False])
def test_step3_gpu_dmshbm_skips_cc(tmp_path: Path, stubs: dict,
                                   mode_a: bool) -> None:
    pipe = _pipeline(_write_project(tmp_path, mode_a=mode_a,
                                    variant="dMSHBM", backend_step3="gpu"))
    with pytest.raises(_ReachedStep0):
        pipe.run()
    assert stubs["cc"] == 0


@pytest.mark.parametrize("mode_a,variant", [
    (True, "gMSHBM"), (True, "cMSHBM"), (False, "gMSHBM"),
])
def test_step3_gpu_cc_variants_prewarm(tmp_path: Path, stubs: dict,
                                       mode_a: bool, variant: str) -> None:
    pipe = _pipeline(_write_project(tmp_path, mode_a=mode_a,
                                    variant=variant, backend_step3="gpu"))
    with pytest.raises(_ReachedStep0):
        pipe.run()
    assert stubs["cc"] == 1


@pytest.mark.parametrize("mode_a", [True, False])
def test_cpu_backends_check_nothing(tmp_path: Path, stubs: dict,
                                    monkeypatch: pytest.MonkeyPatch,
                                    mode_a: bool) -> None:
    from arealmshbm.data_io import _nvcomp_batched
    pipe = _pipeline(_write_project(tmp_path, mode_a=mode_a,
                                    variant="gMSHBM"))
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setitem(sys.modules, "nvidia.nvcomp", None)
    monkeypatch.setattr(_nvcomp_batched, "_CACHE", {})
    with pytest.raises(_ReachedStep0):
        pipe.run()
    assert stubs == {"cuda": 0, "cc": 0, "step2": 0}
