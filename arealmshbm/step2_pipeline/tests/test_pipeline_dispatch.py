"""test_pipeline_dispatch.py

The step-2 backend dispatch + the sparse outer/intra/EM control flow,
tested without cupy and without a project on disk.

Everything the sparse path touches is faked: ``load_step2_sparse_inputs``
returns a namespace with the handful of attributes the pipeline logs,
``Step2SparseSession`` is a recorder that scripts its own costs, and the
save is intercepted. What is actually asserted is the plumbing the
design contract pins (``docs/step2_sparse_design.md`` §4/§6):

* ``run()`` selects the sparse triple for ``backend='gpu'`` and
  the dense (CPU) triple for ``'cpu'``;
* the exact call sequence and per-iteration call counts on the Session
  (``initialize_state`` once, ``reset_inter`` per inter iter,
  ``reset_intra`` per intra iter, ``run_iter`` per EM iter,
  ``intra_closure`` per intra iter, ``inter_closure`` per inter iter);
* the convergence arithmetic — intra, inter and the EM body — matches
  ``run_em`` / ``vmf_clustering_batch`` term for term;
* ``Record`` accumulates one entry per inter iter, ``iter_inter`` /
  ``iter_intra`` track, progress emits fire with the same payloads;
* the Params handed to the writer carry exactly the saved keys and
  never the three per-subject ones;
* ``backend='cpu'`` still walks ``load_inputs`` → ``initialize_params``
  → ``run_em`` untouched.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

import arealmshbm.step2_em_iter_master as _eim
import arealmshbm.step2_io as _sio
from arealmshbm.step2_pipeline.config import Step2Config
from arealmshbm.step2_pipeline.pipeline import Step2Pipeline
from arealmshbm.step2_pipeline.vmf_clustering_batch import em_body_sparse


N, T, L, D, S = 8, 2, 4, 6, 2


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
def _fake_inputs() -> SimpleNamespace:
    return SimpleNamespace(
        layout=SimpleNamespace(P=N * 2, L=L, N=N, n_lh=N // 2),
        S=S, T=T, N=N, D=D, D_grad=3, n_lh=N // 2, n_rh=N // 2,
        mtc=np.zeros((D, L), dtype=np.float64), dim=D - 1,
        bold_reader=None, packed_host=None, grad_reader=None,
        timings={"cohort": 0.001, "profiles": 0.002},
    )


class FakeSession:
    """Records every call; scripts ``run_iter`` / ``intra_closure`` costs.

    ``em_costs`` is the per-``run_iter`` ``cost_S`` sequence (reused
    cyclically once exhausted, with the last value repeated so the EM
    body converges); ``intra_costs`` likewise for ``intra_closure``.
    """

    def __init__(self, inputs, **kw) -> None:
        self.inputs = inputs
        self.kwargs = dict(kw)
        self.S = int(inputs.S)
        self.calls: List[str] = []
        self.timings: Dict[str, float] = {"K3": 0.001, "K8": 0.002}
        self.kappa = 900.0
        self._em_i = 0
        self._intra_i = 0
        # Two run_iter calls per EM body: the second is within eps of the
        # first, so em_body_sparse stops at iter_em == 2.
        self.em_costs: List[np.ndarray] = [
            np.array([-1.0e6, -2.0e6]), np.array([-1.0e6, -2.0e6]),
        ]
        # First intra closure -100, second within eps -> intra converges
        # at iter 2. Third+ repeat the second.
        self.intra_costs: List[float] = [-100.0, -100.0]

    # ── §4 API ──
    def initialize_state(self) -> None:
        self.calls.append("initialize_state")

    def reset_inter(self) -> None:
        self.calls.append("reset_inter")

    def reset_intra(self) -> None:
        self.calls.append("reset_intra")

    def run_iter(self) -> Tuple[int, np.ndarray]:
        self.calls.append("run_iter")
        c = self.em_costs[min(self._em_i, len(self.em_costs) - 1)]
        self._em_i += 1
        return 3, np.asarray(c, dtype=np.float64)

    def intra_closure(self) -> float:
        self.calls.append("intra_closure")
        v = self.intra_costs[min(self._intra_i, len(self.intra_costs) - 1)]
        self._intra_i += 1
        return float(v)

    def inter_closure(self) -> None:
        self.calls.append("inter_closure")

    def export_params(self) -> Dict[str, Any]:
        self.calls.append("export_params")
        import scipy.sparse as sp
        theta = sp.csc_matrix(
            np.eye(N, L, dtype=np.float64))
        return {
            "sigma": np.full((1, L), 1.0, dtype=np.float32),
            "epsil": np.full((1, L), 2.0, dtype=np.float32),
            "kappa": np.full((1, L), 3.0, dtype=np.float32),
            "mu": np.zeros((L, D), dtype=np.float32),
            "cost_em": np.array([-1.0e6, -2.0e6], dtype=np.float64),
            "theta": theta,
        }


class Recorder:
    """Captures ``_save_params`` calls.

    Patched onto the class wrapped in ``staticmethod`` — that is what
    ``Step2Pipeline._save_params`` is, so ``self._save_params(a, b)``
    must not grow a ``self`` argument.
    """

    def __init__(self) -> None:
        self.saved: List[Tuple[Dict[str, Any], Any]] = []

    def hook(self):
        saved = self.saved

        def _save_params(Params, path) -> None:
            saved.append((dict(Params), path))

        return staticmethod(_save_params)


@pytest.fixture
def sparse_env(monkeypatch, tmp_path):
    """Patch the two lazy imports the sparse path makes + the writer."""
    sess_holder: Dict[str, FakeSession] = {}
    inputs = _fake_inputs()

    def _loader(cfg, *, overlap: bool = True):
        return inputs

    def _session_factory(inp, **kw):
        s = FakeSession(inp, **kw)
        sess_holder["sess"] = s
        return s

    monkeypatch.setattr(_sio, "load_step2_sparse_inputs", _loader,
                        raising=False)
    monkeypatch.setattr(_eim, "Step2SparseSession", _session_factory,
                        raising=False)
    rec = Recorder()
    monkeypatch.setattr(Step2Pipeline, "_save_params", rec.hook(),
                        raising=True)
    return SimpleNamespace(inputs=inputs, sess=sess_holder, rec=rec,
                           tmp=tmp_path)


def _cfg(tmp_path, **kw) -> Step2Config:
    base = dict(project_dir=tmp_path, num_sub=S, num_session=T,
                num_clusters=L, mode="gMSHBM", beta_scalar=5.0,
                backend="gpu", out_dir=tmp_path / "out", verbose=False,
                max_iter_inter=2, max_iter_intra_em=3, max_iter_em=5)
    base.update(kw)
    return Step2Config(**base)


# ─────────────────────────────────────────────────────────────────────
# em_body_sparse — convergence semantics
# ─────────────────────────────────────────────────────────────────────
class _CostSession:
    def __init__(self, costs: List[List[float]]) -> None:
        self.costs = [np.asarray(c, dtype=np.float64) for c in costs]
        self.S = int(self.costs[0].shape[0])
        self.i = 0

    def run_iter(self):
        c = self.costs[min(self.i, len(self.costs) - 1)]
        self.i += 1
        return 2, c


def test_em_body_first_iter_never_stops() -> None:
    """iter_em == 1 has no previous cost — it can never converge, even
    when the next iter is identical."""
    s = _CostSession([[-1.0, -1.0], [-1.0, -1.0]])
    r = em_body_sparse(s, max_iter_em=5, em_convergence_eps=1e-4,
                       verbose=False)
    assert r.em_iters == 2
    assert r.converged is True


def test_em_body_needs_every_subject_to_converge() -> None:
    """One lagging subject keeps the whole batch going — the dense body
    tests ``per_sub_ok.sum() == S``, not ``any``."""
    s = _CostSession([[-1.0, -1.0], [-1.0, -2.0], [-1.0, -2.0]])
    r = em_body_sparse(s, max_iter_em=5, em_convergence_eps=1e-4,
                       verbose=False)
    assert r.em_iters == 3
    assert r.converged is True


def test_em_body_hits_the_cap_without_converging() -> None:
    s = _CostSession([[-1.0], [-2.0], [-4.0], [-8.0], [-16.0]])
    r = em_body_sparse(s, max_iter_em=4, em_convergence_eps=1e-4,
                       verbose=False)
    assert r.em_iters == 4
    assert r.converged is False
    assert r.m_iters_per_em == [2, 2, 2, 2]
    assert r.cost_S.dtype == np.float64


def test_em_body_rel_matches_the_dense_formula() -> None:
    """``rel = |‖cost - prev‖| / prev`` — signed denominator, absolute
    value applied twice, exactly as ``vmf_clustering_batch`` does it."""
    prev, cur = -1.0e6, -1.0e6 * (1.0 + 5e-5)
    s = _CostSession([[prev], [cur]])
    r = em_body_sparse(s, max_iter_em=5, em_convergence_eps=1e-4,
                       verbose=False)
    assert r.converged is True          # 5e-5 <= 1e-4
    s = _CostSession([[prev], [-1.0e6 * (1.0 + 5e-4)]])
    r = em_body_sparse(s, max_iter_em=2, em_convergence_eps=1e-4,
                       verbose=False)
    assert r.converged is False         # 5e-4 > 1e-4


# ─────────────────────────────────────────────────────────────────────
# run() dispatch
# ─────────────────────────────────────────────────────────────────────
def test_run_dispatches_to_the_sparse_triple(sparse_env, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path))
    calls: List[str] = []

    def _boom(*a, **k):
        calls.append("dense")
        raise AssertionError(
            "dense path must not run for backend='gpu'")

    pipe.load_inputs = _boom          # type: ignore[assignment]
    pipe.initialize_params = _boom    # type: ignore[assignment]
    pipe.run_em = _boom               # type: ignore[assignment]
    res = pipe.run()
    assert calls == []
    assert res.Params_final_path.name == "Params_Final.mat"


def test_run_dispatches_to_the_dense_triple(monkeypatch, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path, backend="cpu"))
    seen: List[str] = []

    def _sparse_boom(*a, **k):
        raise AssertionError("sparse path must not run for backend='cpu'")

    monkeypatch.setattr(Step2Pipeline, "load_inputs_sparse", _sparse_boom)
    monkeypatch.setattr(Step2Pipeline, "initialize_params_sparse",
                        _sparse_boom)
    monkeypatch.setattr(Step2Pipeline, "_run_em_sparse", _sparse_boom)

    sentinel = object()

    def _li(self):
        seen.append("load_inputs")
        return sentinel

    def _ip(self, inputs):
        seen.append("initialize_params")
        assert inputs is sentinel
        return {}

    def _re(self, Params, inputs):
        seen.append("run_em")
        from arealmshbm.step2_pipeline.pipeline import Step2Result
        return Step2Result(Params_final_path=tmp_path / "x.mat",
                           inter_iters=1, intra_em_iters_per_inter=[1],
                           final_cost=0.0)

    monkeypatch.setattr(Step2Pipeline, "load_inputs", _li)
    monkeypatch.setattr(Step2Pipeline, "initialize_params", _ip)
    monkeypatch.setattr(Step2Pipeline, "run_em", _re)
    pipe.run()
    assert seen == ["load_inputs", "initialize_params", "run_em"]


# ─────────────────────────────────────────────────────────────────────
# load_inputs_sparse / initialize_params_sparse
# ─────────────────────────────────────────────────────────────────────
def test_load_inputs_sparse_forwards_loader_timings(sparse_env, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path))
    got = pipe.load_inputs_sparse()
    assert got is sparse_env.inputs
    assert "load_inputs" in pipe.timings
    assert pipe.timings["load_inputs.cohort"] == pytest.approx(0.001)
    assert pipe.timings["load_inputs.profiles"] == pytest.approx(0.002)


def test_initialize_params_sparse_is_host_bookkeeping_only(
        sparse_env, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path))
    P = pipe.initialize_params_sparse(sparse_env.inputs)
    assert set(P) == {"ini_val", "iter_inter", "Record"}
    assert P["iter_inter"] == 0 and P["Record"] == []
    assert P["ini_val"] > 0.0
    assert "initialize_params" in pipe.timings
    # No host compose on this path.
    assert "initialize_params.compose" not in pipe.timings


# ─────────────────────────────────────────────────────────────────────
# _run_em_sparse control flow
# ─────────────────────────────────────────────────────────────────────
def test_sparse_call_sequence_and_counts(sparse_env, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path))
    res = pipe.run()
    sess = sparse_env.sess["sess"]
    calls = sess.calls

    # FakeSession: EM body stops at 2 run_iters; intra converges at
    # iter 2; max_iter_inter=2 with a constant cost, so the outer loop
    # converges on rel_outer == 0 at inter 2.
    assert calls[0] == "initialize_state"
    assert calls.count("initialize_state") == 1
    assert res.inter_iters == 2
    assert res.intra_em_iters_per_inter == [2, 2]
    assert calls.count("reset_inter") == 2
    assert calls.count("reset_intra") == 4          # 2 inter × 2 intra
    assert calls.count("run_iter") == 8             # 4 intra × 2 EM iters
    assert calls.count("intra_closure") == 4
    assert calls.count("inter_closure") == 2
    assert calls.count("export_params") == 1
    assert calls[-1] == "export_params"

    # Ordering within one inter iter: reset_inter precedes the first
    # reset_intra, and inter_closure comes after the last intra_closure.
    first_inter = calls.index("reset_inter")
    assert calls.index("reset_intra") > first_inter
    assert (calls.index("inter_closure")
            > max(i for i, c in enumerate(calls[:calls.index("inter_closure")])
                  if c == "intra_closure"))


def test_sparse_session_ctor_kwargs(sparse_env, tmp_path) -> None:
    cfg = _cfg(tmp_path)
    pipe = Step2Pipeline(cfg)
    pipe.run()
    kw = sparse_env.sess["sess"].kwargs
    assert kw["mode"] == cfg.mode
    assert kw["num_clusters"] == cfg.num_clusters
    assert kw["dim"] == sparse_env.inputs.dim
    assert kw["beta_internal"] == cfg.beta_internal
    assert kw["eps_m_step"] == cfg.epsilon
    assert kw["max_iter_m"] == cfg.max_iter_m
    assert kw["eps_intra_var"] == cfg.epsilon
    assert kw["max_iter_intra_var"] == cfg.max_iter_intra_var
    assert kw["bold_cache_mode"] == cfg.bold_cache_mode
    assert kw["bold_cache_safety_margin_gb"] == cfg.gpu_cache_safety_margin_gb
    assert sparse_env.sess["sess"].inputs is sparse_env.inputs


def test_sparse_record_and_iteration_bookkeeping(sparse_env, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path))
    pipe.run()
    Params, _path = sparse_env.rec.saved[0]
    assert Params["Record"] == [-100.0, -100.0]     # one per inter iter
    assert Params["iter_inter"] == 2
    assert Params["iter_intra"] == 2
    assert Params["cost_intra"] == -100.0
    assert Params["cost_inter"] == -100.0


def test_sparse_intra_convergence_uses_the_dense_arithmetic(
        sparse_env, tmp_path) -> None:
    """intra stops when ``|Δcost| / cost_prev <= eps``; a cost that keeps
    moving by more than eps runs to ``max_iter_intra_em``."""
    pipe = Step2Pipeline(_cfg(tmp_path, max_iter_inter=1,
                              max_iter_intra_em=4))
    # Patch the scripted intra costs before the run by pre-seeding the
    # factory: easiest is to run once, then inspect. Instead, override
    # via a subclass of the fake through the module-level factory.
    import arealmshbm.step2_em_iter_master as eim

    orig = eim.Step2SparseSession

    def _factory(inp, **kw):
        s = orig(inp, **kw)
        s.intra_costs = [-100.0, -200.0, -400.0, -800.0]
        return s

    eim.Step2SparseSession = _factory        # type: ignore[attr-defined]
    try:
        res = pipe.run()
    finally:
        eim.Step2SparseSession = orig        # type: ignore[attr-defined]
    assert res.intra_em_iters_per_inter == [4]


def test_sparse_inter_convergence_stops_early(sparse_env, tmp_path) -> None:
    """A constant inter cost gives rel_outer == 0 <= eps at inter 2, so
    a cap of 10 must still stop at 2."""
    pipe = Step2Pipeline(_cfg(tmp_path, max_iter_inter=10))
    res = pipe.run()
    assert res.inter_iters == 2
    assert res.final_cost == -100.0


def test_sparse_progress_emits(sparse_env, tmp_path, monkeypatch) -> None:
    events: List[Tuple[str, tuple, dict]] = []

    class _Emitter:
        def emit_state(self, *a, **k):
            events.append(("state", a, k))

        def emit_iter(self, *a, **k):
            events.append(("iter", a, k))

    pipe = Step2Pipeline(_cfg(tmp_path))
    pipe.progress = _Emitter()          # type: ignore[assignment]
    pipe.run()
    states = [e for e in events if e[0] == "state"]
    iters = [e for e in events if e[0] == "iter"]
    assert states[0][1][2] == "running"
    # 2 inter emits (no iter_intra) + 4 intra emits (with iter_intra).
    outer = [e for e in iters if "iter_intra" not in e[2]]
    inner = [e for e in iters if "iter_intra" in e[2]]
    assert len(outer) == 2 and len(inner) == 4
    assert outer[0][2] == {"iter_inter": 1, "max_inter": 2}
    assert inner[0][2] == {"iter_inter": 1, "max_inter": 2,
                           "iter_intra": 1, "max_intra": 3}


def test_sparse_failure_emits_failed_state(sparse_env, tmp_path) -> None:
    events: List[Tuple[str, tuple, dict]] = []

    class _Emitter:
        def emit_state(self, *a, **k):
            events.append(("state", a, k))

        def emit_iter(self, *a, **k):
            pass

    import arealmshbm.step2_em_iter_master as eim
    orig = eim.Step2SparseSession

    def _boom(inp, **kw):
        raise RuntimeError("device OOM")

    eim.Step2SparseSession = _boom       # type: ignore[attr-defined]
    pipe = Step2Pipeline(_cfg(tmp_path))
    pipe.progress = _Emitter()           # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="device OOM"):
            pipe.run()
    finally:
        eim.Step2SparseSession = orig    # type: ignore[attr-defined]
    assert events[-1][1][2] == "failed"
    assert "RuntimeError: device OOM" in events[-1][2]["error"]


# ─────────────────────────────────────────────────────────────────────
# Saved Params + the theta knob
# ─────────────────────────────────────────────────────────────────────
def test_sparse_saved_params_keys(sparse_env, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path))
    pipe.run()
    Params, path = sparse_env.rec.saved[0]
    assert path == tmp_path / "out" / "Params_Final.mat"
    assert set(Params) == {
        "ini_val", "iter_inter", "iter_intra", "Record",
        "sigma", "epsil", "kappa", "mu", "theta",
        "cost_em", "cost_intra", "cost_inter",
    }
    # The three per-subject keys the dense path strips never exist here.
    for k in ("s_lambda", "s_psi", "s_t_nu"):
        assert k not in Params
    assert Params["mu"].shape == (L, D)          # internal (L, D) layout
    assert Params["sigma"].shape == (1, L)
    assert Params["cost_em"].shape == (S,)


def test_theta_reaches_the_writer_verbatim(sparse_env, tmp_path) -> None:
    """``_run_em_sparse`` hands ``export_params()['theta']`` over as-is.

    The densify belongs to ``_save.save_params_final`` — see
    ``test_save.py::test_a_sparse_theta_is_densified``. Densifying here
    too would be a second, divergent copy of that policy.
    """
    pipe = Step2Pipeline(_cfg(tmp_path))
    pipe.run()
    Params, _ = sparse_env.rec.saved[0]
    assert hasattr(Params["theta"], "toarray")     # the csc, untouched


# ─────────────────────────────────────────────────────────────────────
# Timings
# ─────────────────────────────────────────────────────────────────────
def test_sparse_timing_keys(sparse_env, tmp_path) -> None:
    pipe = Step2Pipeline(_cfg(tmp_path))
    res = pipe.run()
    for k in ("load_inputs", "initialize_params", "session_ctor",
              "init_device", "em_total", "closure_total", "run_em",
              "save", "total"):
        assert k in res.timings, k
    # Per-kernel Session timings are forwarded under session.*
    assert res.timings["session.K3"] == pytest.approx(0.001)
    assert res.timings["session.K8"] == pytest.approx(0.002)


def test_session_timings_are_forwarded_verbatim(sparse_env, tmp_path) -> None:
    """No unit conversion in the forwarding — the Session normalises.

    ``kernel.<name>`` entries used to be scaled by 1e-3 here on the
    assumption they were milliseconds; the Session now emits seconds for
    every key, so any scale in the pipeline would corrupt them.
    """
    pipe = Step2Pipeline(_cfg(tmp_path))
    res = pipe.run()
    for k, v in sparse_env.sess["sess"].timings.items():
        assert res.timings[f"session.{k}"] == pytest.approx(float(v)), k


def test_em_iters_total_counts_run_iter_calls(sparse_env, tmp_path) -> None:
    """``Step2Result.em_iters_total`` == the number of ``run_iter`` calls."""
    pipe = Step2Pipeline(_cfg(tmp_path))
    res = pipe.run()
    n_run_iter = sparse_env.sess["sess"].calls.count("run_iter")
    assert res.em_iters_total == n_run_iter
    # Strictly more than the number of EM bodies (2 run_iters per body here).
    assert res.em_iters_total == 2 * sum(res.intra_em_iters_per_inter)


# ─────────────────────────────────────────────────────────────────────
# __exit__ frees the pools on the gpu backend only
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("backend,expect", [("cpu", False), ("gpu", True)])
def test_exit_pool_free_scope(monkeypatch, tmp_path, backend, expect) -> None:
    freed: List[str] = []

    class _Pool:
        def free_all_blocks(self) -> None:
            freed.append("x")

    fake_cp = SimpleNamespace(
        get_default_memory_pool=lambda: _Pool(),
        get_default_pinned_memory_pool=lambda: _Pool(),
    )
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cp)
    with Step2Pipeline(_cfg(tmp_path, backend=backend)):
        pass
    assert bool(freed) is expect


