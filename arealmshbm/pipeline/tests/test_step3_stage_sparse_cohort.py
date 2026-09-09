"""test_step3_stage_sparse_cohort.py — stage LOAD loads the cohort once.

On ``gpu_sparse`` the group prior / spatial masks / candidate layout are
cohort constants; stage LOAD reads them once before the subject loop and
hands the same :class:`Step3SparseCohort` to every ``Step3Pipeline``.
Both tests replace ``Step3Pipeline`` with a recording stub and the
cohort loader with a counting fake, so no data store is touched — but
the coordinator still opens cupy streams on this backend, hence the
cupy skip.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("cupy", reason="gpu_sparse coordinator creates cupy streams")

from arealmshbm.pipeline import _step3_stage_pipeline as sp
from arealmshbm.step3_pipeline import Step3Config


class _StubPipe:
    """Records the cohort it was handed; every stage method is a no-op."""

    def __init__(self, cfg, *, precomputed_gradient_mat=None,
                 sparse_cohort=None):
        self.cfg = cfg
        self.sparse_cohort = sparse_cohort

    def load_inputs(self):
        return None

    def build_session(self, inputs):
        return None

    def run(self):
        return None

    def save(self, result):
        return None

    def close(self):
        return None


def _configs(n):
    return [
        Step3Config(project_dir="/tmp/nonexistent", num_session=6,
                    num_clusters=300, subid=i + 1, backend="gpu_sparse")
        for i in range(n)
    ]


def _coordinator(n):
    return sp.Step3StagePipeline(
        subject_ids=[str(i + 1) for i in range(n)],
        configs=_configs(n), gradients={}, backend="gpu_sparse",
    )


def _run_with_timeout(coord, timeout=60.0):
    """Run the coordinator on a helper thread; return its exception.

    A cohort failure that does not reach ``qDone`` would hang ``run()``
    forever, so the assertion has to be "finished", not just "raised".
    """
    box = {}

    def _target():
        try:
            coord.run()
        except BaseException as e:      # noqa: BLE001 — that's the assertion
            box["err"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"Step3StagePipeline.run() hung for {timeout}s"
    return box.get("err")


def test_stage_load_builds_cohort_once_and_shares_it(monkeypatch):
    pipes = []

    class _Recording(_StubPipe):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
            pipes.append(self)

    cohort = object()
    calls = []

    def _fake_loader(cfg, **kw):
        calls.append(cfg)
        return cohort

    monkeypatch.setattr(sp, "Step3Pipeline", _Recording)
    monkeypatch.setattr(
        "arealmshbm.step3_pipeline.sparse_inputs.load_step3_sparse_cohort",
        _fake_loader)

    assert _run_with_timeout(_coordinator(3)) is None
    assert len(calls) == 1
    assert len(pipes) == 3
    assert all(p.sparse_cohort is cohort for p in pipes)


def test_cohort_load_failure_surfaces_from_run(monkeypatch):
    def _boom(cfg, **kw):
        raise RuntimeError("cohort load failed")

    monkeypatch.setattr(sp, "Step3Pipeline", _StubPipe)
    monkeypatch.setattr(
        "arealmshbm.step3_pipeline.sparse_inputs.load_step3_sparse_cohort",
        _boom)

    err = _run_with_timeout(_coordinator(2))
    assert isinstance(err, RuntimeError)
    assert "cohort load failed" in str(err)


def test_dense_backend_gets_no_cohort(monkeypatch):
    pipes = []

    class _Recording(_StubPipe):
        def __init__(self, cfg, **kw):
            super().__init__(cfg, **kw)
            pipes.append(self)

    def _never(cfg, **kw):
        raise AssertionError("dense backend must not load a sparse cohort")

    monkeypatch.setattr(sp, "Step3Pipeline", _Recording)
    monkeypatch.setattr(
        "arealmshbm.step3_pipeline.sparse_inputs.load_step3_sparse_cohort",
        _never)

    coord = sp.Step3StagePipeline(
        subject_ids=["1", "2"],
        configs=[Step3Config(project_dir="/tmp/nonexistent", num_session=6,
                             num_clusters=300, subid=i + 1,
                             backend="gpu_full") for i in range(2)],
        gradients={}, backend="gpu_full",
    )
    assert _run_with_timeout(coord) is None
    assert [p.sparse_cohort for p in pipes] == [None, None]
