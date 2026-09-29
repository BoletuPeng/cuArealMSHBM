"""test_step1_gpu_routing.py — the guards on ``backend=`` in
:func:`arealmshbm.pipeline.step1_runners.run_generate_profiles`.

  * ``profile_dtype_reduce`` other than float32 is REJECTED, not
    silently ignored — the fused leaf is fp32 throughout.
  * An unknown backend name is rejected.
  * ``backend='gpu'`` routes to the fused whole-subject leaf, full
    stop: the stage pipeline is the CPU backend's and must not be
    constructed here.
  * ``prewarm_step1_gpu`` -- the first thing the driver does for
    ``backend_step1='gpu'`` -- raises a missing nvCOMP synchronously,
    before any thread starts.

No GPU needed: the routing is decided before any device work.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arealmshbm.pipeline import step1_runners


def _project(tmp_path: Path) -> Path:
    """Minimal project: one subject, one session, empty fMRI lists."""
    d = tmp_path / "data_list" / "fMRI_list"
    d.mkdir(parents=True)
    (d / "lh_sub1_sess1.txt").write_text("lh_run1.func.gii\n", encoding="utf-8")
    (d / "rh_sub1_sess1.txt").write_text("rh_run1.func.gii\n", encoding="utf-8")
    return tmp_path


def _call(project: Path, **over):
    kwargs = dict(
        project_dir=project, subjects=["1"], sessions=["1"],
        seed_mesh="fsaverage3", targ_mesh="fsaverage6",
        backend="gpu", verbose=False,
    )
    kwargs.update(over)
    return step1_runners.run_generate_profiles(**kwargs)


def test_gpu_prewarm_fails_fast_without_nvcomp(monkeypatch):
    """A missing ``nvidia-nvcomp-cu12`` must be raised from the prewarm
    itself, for both the inline and the background form -- not swallowed
    as a best-effort warm-up failure and rediscovered after step 0."""
    import sys
    from arealmshbm.data_io import _nvcomp_batched as nb

    monkeypatch.setitem(sys.modules, "nvidia.nvcomp", None)
    saved = dict(nb._CACHE)
    nb._CACHE.clear()
    try:
        for background in (False, True):
            with pytest.raises(ImportError, match="nvidia-nvcomp-cu12"):
                step1_runners.prewarm_step1_gpu(None, background=background)
    finally:
        nb._CACHE.clear()
        nb._CACHE.update(saved)


def test_gpu_backend_rejects_a_non_fp32_reduction_dtype(tmp_path):
    with pytest.raises(ValueError, match="computes in float32"):
        _call(_project(tmp_path), profile_dtype_reduce=np.float64)


def test_unknown_backend_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown backend"):
        _call(_project(tmp_path), backend="opencl")


def test_gpu_backend_always_uses_the_fused_leaf(tmp_path, monkeypatch):
    """``backend='gpu'`` is the fused leaf — and the hooks are passed
    straight through rather than dropped."""
    project = _project(tmp_path)
    seen = {}

    def _boom(**kw):                     # the CPU stage pipeline
        raise AssertionError("stage pipeline constructed on backend='gpu'")

    def _fused(**kw):
        seen.update(kw)
        kw["write_handles"].append("handle")
        return [("1", Path("sub1.b2nd"))]

    class _Acc:
        def add(self, sess_index, packed_dev):
            pass

    import arealmshbm.pipeline._step1_bold_prefetcher as pf
    import arealmshbm.pipeline._step1_stage_pipeline as sp

    monkeypatch.setattr(pf, "Step1BoldPrefetcher", _boom)
    monkeypatch.setattr(sp, "Step1GenerateProfilesStagePipeline", _boom)
    monkeypatch.setattr(
        step1_runners, "_run_generate_profiles_gpu_fused", _fused)

    acc = _Acc()
    handles: list = []
    out = _call(project, avg_accumulator=acc, write_handles=handles)

    assert out == [("1", Path("sub1.b2nd"))]
    assert seen["on_packed"] == acc.add and seen["write_handles"] is handles
    assert handles == ["handle"]
