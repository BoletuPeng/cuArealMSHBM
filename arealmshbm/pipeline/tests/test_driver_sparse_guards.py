# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""The driver's early ``seed_mesh`` guards for the P-layout GPU backends
(step 2 ``'gpu'``, step 3 ``'gpu_sparse'``).

Both bake a compile-time ``ceil(D/8) <= 256`` limit,
which only ``seed_mesh='fsaverage3'`` satisfies. Step 2's rule lives in
``Step2Config.__post_init__`` and step 3's in the ``gpu_sparse`` session
ctor — but neither sees ``seed_mesh`` (it comes from
``bold_inputs.json``) and both fire only after steps 0-1 (step 3: 0-2)
have already burned minutes of wall time. ``Pipeline.__init__`` therefore
re-checks it in ``_validate_inputs_vs_config``, alongside the other
config↔bold_inputs cross-checks; these tests pin that it does, for both
steps, in the mode(s) each one runs in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arealmshbm.pipeline.driver import Pipeline


def _write_project(root: Path, *, mode_a: bool, seed_mesh: str,
                   **backends: str) -> Path:
    """Smallest project tree ``Pipeline.__init__`` accepts.

    ``pipeline_config.json`` is cribbed verbatim from the matching
    in-tree sample so the fixture tracks schema drift, then patched with
    the ``backend_step*`` values under test (top-level keys present in
    every mode, so Mode A can carry ``backend_step2`` without the
    Mode-A-forbidden step2 block).
    """
    repo_root = Path(__file__).resolve().parents[3]
    sample = repo_root / "projects" / (
        "sample_modeA_single" if mode_a else "sample_modeB")

    project = root / "tiny_project"
    project.mkdir()
    cfg = json.loads(
        (sample / "pipeline_config.json").read_text(encoding="utf-8"))
    cfg.update(backends)
    (project / "pipeline_config.json").write_text(
        json.dumps(cfg), encoding="utf-8")
    (project / "bold_inputs.json").write_text(json.dumps({
        "schema_version": "1",
        "dataset_name": "sparse_guard_fixture",
        "targ_mesh": "fsaverage6",
        "seed_mesh": seed_mesh,
        # Mode A wants exactly 1 subject, Mode B at least 2.
        "subjects": [{"id": sid, "sessions": [
            {"id": "1",
             "lh": str(project / f"lh.{sid}.func.gii"),
             "rh": str(project / f"rh.{sid}.func.gii")},
        ]} for sid in (("1",) if mode_a else ("1", "2"))],
    }), encoding="utf-8")
    return project


def test_step3_gpu_sparse_rejects_non_fsaverage3_mode_a(
        tmp_path: Path) -> None:
    project = _write_project(tmp_path, mode_a=True, seed_mesh="fsaverage4",
                             backend_step3="gpu_sparse")
    with pytest.raises(ValueError, match=r"backend_step3='gpu_sparse'.*"
                                         r"seed_mesh='fsaverage3'"):
        Pipeline(project)


def test_step3_gpu_sparse_rejects_non_fsaverage3_mode_b(
        tmp_path: Path) -> None:
    """Step 3 runs in Mode B too, so the guard must not be Mode-A only."""
    project = _write_project(tmp_path, mode_a=False, seed_mesh="fsaverage4",
                             backend_step3="gpu_sparse")
    with pytest.raises(ValueError, match=r"backend_step3='gpu_sparse'"):
        Pipeline(project)


def test_step2_gpu_rejects_non_fsaverage3(tmp_path: Path) -> None:
    """The step-2 guard, pinned alongside its step-3 twin."""
    project = _write_project(tmp_path, mode_a=False, seed_mesh="fsaverage4",
                             backend_step2="gpu")
    with pytest.raises(ValueError, match=r"backend_step2='gpu' requires"):
        Pipeline(project)


def test_mode_a_ignores_backend_step2_gpu(tmp_path: Path) -> None:
    """Mode A never runs step 2, so a leftover ``backend_step2`` there is
    inert — the step-2 guard must not fire on it even on fsaverage4."""
    project = _write_project(tmp_path, mode_a=True, seed_mesh="fsaverage4",
                             backend_step2="gpu")
    assert Pipeline(project).config.backend_step2 == "gpu"


@pytest.mark.parametrize("mode_a", [True, False])
def test_fsaverage3_passes_both_guards(tmp_path: Path, mode_a: bool) -> None:
    """With the canonical seed mesh both guards are silent and
    construction succeeds."""
    backends = {"backend_step3": "gpu_sparse"}
    if not mode_a:
        backends["backend_step2"] = "gpu"
    project = _write_project(tmp_path, mode_a=mode_a, seed_mesh="fsaverage3",
                             **backends)
    assert Pipeline(project).inputs.seed_mesh == "fsaverage3"
