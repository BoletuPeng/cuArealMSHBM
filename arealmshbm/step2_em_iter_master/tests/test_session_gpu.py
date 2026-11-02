"""test_session_gpu.py — Step2SparseSession vs the CPU master.

Three families:

1. a synthetic **binary** cohort (the on-disk BOLD contract) written as a
   project and loaded through ``load_step2_sparse_inputs``, so the Session
   pulls it through the production reader; small enough to drive the CPU
   ``Step2EmIterSession`` in a second, run for one EM iteration in both
   gMSHBM and dMSHBM;
2. run-to-run determinism — two Sessions on the same inputs must agree with
   ``np.array_equal`` on every field;
3. the bench project (skipped when absent), with the loose structural bars of
   ``docs/step2_sparse_design.md`` §7.3.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.data_io.profile_io import (  # noqa: E402
    profile_path, write_subject_profile_tnd,
)
from arealmshbm.step2_em_iter_master import (  # noqa: E402
    Step2EmIterSession, Step2SparseSession,
)
from arealmshbm.step2_em_iter_master.tests._loaders import (  # noqa: E402
    InMemoryGradientLoader, InMemoryProfileLoader,
)
from arealmshbm.step2_init.boundary_mask import (  # noqa: E402
    build_step2_boundary_mask,
)
from arealmshbm.step2_io.load_subject_profiles import (  # noqa: E402
    _widen_normalize_bitpacked_to_f32_NTD_kernel,
)
from arealmshbm.step2_io.tests._synthetic_project import (  # noqa: E402
    SEED, TARG, load_project, make_project,
)
from arealmshbm.step2_em_iter_master.tests import _bench_fixture as BF  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# synthetic cohort
# ─────────────────────────────────────────────────────────────────────
S, T, N, L, D, D_GRAD = 2, 2, 128, 16, 264, 8
N_LH, L_LH = N // 2, L // 2
DB = (D + 7) // 8


def _synthetic(seed, root):
    """Draw a random binary cohort, write it as a project under ``root``
    and load it the way the pipeline does (tiny fake mesh, no medial
    wall)."""
    rng = np.random.default_rng(seed)
    bits = (rng.random((S, T, N, D)) < 0.10).astype(np.uint8)
    packed = np.packbits(bits, axis=-1, bitorder="little")
    assert packed.shape == (S, T, N, DB)

    # One 0/1 mask per hemisphere, >= 1 cell per row; the CPU master gets
    # the dense mask the pipeline builds from the same two blocks.
    lh_b, rh_b = (np.zeros((N_LH, L_LH), dtype=np.float32) for _ in range(2))
    for blk in (lh_b, rh_b):
        for row in blk:
            k = int(rng.integers(1, 5))
            row[rng.choice(L_LH, size=k, replace=False)] = 1.0
    bm = build_step2_boundary_mask(lh_b, rh_b, "gMSHBM")

    mtc = rng.standard_normal((D, L))
    mtc /= np.linalg.norm(mtc, axis=0, keepdims=True)
    grad = rng.standard_normal((S, N, D_GRAD)).astype(np.float32)

    cfg, refs = make_project(
        root, S=S, T=T, n_h=N_LH, L_h=L_LH, D=D, n_grad=D_GRAD, mw=(),
        bold=bits, boundary=(lh_b, rh_b), mtc=mtc, grad=grad)
    inputs = load_project(cfg, refs)
    # The exact CPU BOLD: widen the same bits with the production kernel.
    bold = np.zeros((S, N, T, D), dtype=np.float32)
    for s in range(S):
        _widen_normalize_bitpacked_to_f32_NTD_kernel(
            np.ascontiguousarray(packed[s].transpose(1, 0, 2)), bold[s], D)
    grad_STND = np.ascontiguousarray(
        np.broadcast_to(grad[:, None], (S, T, N, D_GRAD)).copy())
    return inputs, bm, mtc, bold, grad_STND


def _cpu_one_iter(mode, bm, mtc, bold, grad_STND, dim, ini_val, beta, n_iter=1):
    from arealmshbm.step2_init.compose_init_state import compose_init_state
    bold_loader = InMemoryProfileLoader(bold, T)
    grad_loader = (InMemoryGradientLoader(grad_STND, T)
                   if mode == "gMSHBM" else None)
    mtc_LD = np.ascontiguousarray(mtc.astype(np.float32).T)
    s_lambda, theta = compose_init_state(
        bold_loader=bold_loader, group_mtc=mtc, boundary_mask=bm,
        num_clusters=L, ini_val=ini_val)
    Params = {
        "ini_val": ini_val,
        "sigma": (ini_val * np.ones(L, dtype=np.float32)).reshape(1, L),
        "epsil": (ini_val * np.ones(L, dtype=np.float32)).reshape(1, L),
        "kappa": (ini_val * np.ones(L, dtype=np.float32)).reshape(1, L),
        "mu": mtc_LD.copy(),
        "s_psi": np.broadcast_to(mtc_LD, (S, L, D)).copy(),
        "s_t_nu": np.broadcast_to(mtc_LD, (S, T, L, D)).copy(),
        "s_lambda": s_lambda,
        "theta": theta,
    }
    sess = Step2EmIterSession(
        bold_loader=bold_loader, grad_loader=grad_loader, num_sub=S,
        N=N, T=T, D=D, D_grad=D_GRAD, boundary_mask=bm,
        s_psi=Params["s_psi"], sigma=Params["sigma"], mode=mode, dim=dim,
        num_clusters=L, ini_val=ini_val, beta_internal=beta, n_lh=N_LH,
        eps_m_step=1e-4, max_iter_m=50)
    sess.upload_initial_state(Params)
    sess.refresh_s_psi_sigma(Params["s_psi"], Params["sigma"])
    m_iters = []
    for _ in range(n_iter):
        m, _tot = sess.run_iter(Params)
        m_iters.append(m)
    Params["cost_em"] = sess.get_cost_S().copy()
    # the pre-iteration init state, for the bit-exactness check
    init0 = {"s_lambda": s_lambda, "theta": theta}
    return Params, m_iters, init0


def _gpu_session(inputs, mode, dim, ini_val, beta):
    return Step2SparseSession(
        inputs, mode=mode, num_clusters=L, dim=dim, ini_val=ini_val,
        beta_internal=beta, eps_m_step=1e-4, max_iter_m=50,
        eps_intra_var=1e-4, max_iter_intra_var=20)


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _relmax(a, b, eps=1e-30):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / (np.abs(b) + eps)))


@pytest.mark.parametrize("mode", ["gMSHBM", "dMSHBM"])
def test_one_em_iter_matches_cpu(mode, tmp_path):
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, bm, mtc, bold, grad_STND = _synthetic(0, tmp_path)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))
    beta = 5000.0 if mode == "gMSHBM" else 0.0

    g = _gpu_session(inputs, mode, dim, ini_val, beta)
    g.initialize_state()
    gh0 = g.sync_to_host(("s_lambda", "theta"))
    m_gpu, cost_gpu = g.run_iter()
    gh = g.sync_to_host(("s_t_nu", "s_lambda", "theta"))

    Params, m_cpu, init0 = _cpu_one_iter(mode, bm, mtc, bold, grad_STND, dim,
                                         ini_val, beta)

    # init state is bit-equal to the CPU ``compose_init_state`` output
    # (design §1.2). (``Params['s_lambda']`` has since been overwritten by
    # the CPU iteration, so the comparison uses the pre-iteration
    # snapshot.)  ``theta`` is compared on P only: the P-layout carries no
    # out-of-P cell, where ``compose_init_state`` writes f32(eps).
    assert np.array_equal(_bits(gh0["s_lambda"]), _bits(init0["s_lambda"]))
    assert np.array_equal(_bits(gh0["theta"]),
                          _bits(init0["theta"] * (bm != 0)))
    assert m_gpu == m_cpu[0], (m_gpu, m_cpu)
    assert _relmax(gh["s_t_nu"], Params["s_t_nu"]) <= 5e-3
    assert _relmax(gh["theta"], Params["theta"] * (bm != 0)) <= 5e-3
    assert _relmax(gh["s_lambda"], Params["s_lambda"]) <= 5e-3
    kap_cpu = float(np.asarray(Params["kappa"]).ravel()[0])
    assert abs(g.kappa - kap_cpu) / kap_cpu <= 1e-4, (g.kappa, kap_cpu)
    assert _relmax(cost_gpu, Params["cost_em"]) <= 5e-3


_SPIN_SRC = r"""
extern "C" __global__ void _test_spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
"""


def test_ctor_fences_the_current_stream_not_the_null_stream(tmp_path, monkeypatch):
    """The eager ingest refills a page-locked slot once its H2D is fenced.

    Built inside a non-blocking stream with work queued on it, the H2D
    copies sit behind that work; a fence on the legacy default stream
    returns immediately and lets the next subject's refill overwrite the
    slot under them. The ring is pinned to ONE slot so subject 2 refills
    subject 1's: with the default depth a cohort no larger than the ring
    never refills anything and the fence is not exercised.

    The refill has to land while subject 1's copy is still queued, so the
    reader is a double: asked for subject 1 it queues the stream work
    itself, right ahead of that copy, and it fills the slot with a memcpy
    of the same bytes. With every core busy the production reader's
    decode takes 0.3 s against the ~0.1 s of queued work, and work queued
    ahead of the whole constructor can run out before the first copy is
    issued; a wrong fence then passes.
    """
    import arealmshbm.step2_em_iter_master.session_gpu as sg
    from arealmshbm.initialize_concentration import initialize_concentration
    monkeypatch.setattr(sg, "_INGEST_DEPTH", 1)
    # The per-subject payload must clear 64 KiB: CUDA inlines smaller H2D
    # copies into the pushbuffer, which hides the reuse hazard entirely.
    n_wide = 4096
    rng = np.random.default_rng(3)
    bits = (rng.random((S, T, n_wide, D)) < 0.10).astype(np.uint8)
    packed = np.packbits(bits, axis=-1, bitorder="little")
    assert packed[0].nbytes > (1 << 16)
    blk = np.zeros((n_wide // 2, L_LH), dtype=np.float32)
    blk[:, 0] = 1.0
    mtc = rng.standard_normal((D, L))
    mtc /= np.linalg.norm(mtc, axis=0, keepdims=True)
    cfg, refs = make_project(
        tmp_path, S=S, T=T, n_h=n_wide // 2, L_h=L_LH, D=D, n_grad=D_GRAD,
        mw=(), bold=bits, boundary=(blk, blk), mtc=mtc)

    base = load_project(cfg, refs)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))
    spin = cp.RawKernel(_SPIN_SRC, "_test_spin")
    stream = cp.cuda.Stream(non_blocking=True)

    def memcpy_reader(s_1, out):
        np.copyto(out, packed[s_1 - 1])

    def queueing_reader(s_1, out):
        if s_1 == 1:
            # On the ingest worker's thread, whose current stream is not
            # the constructor's.
            with stream:
                spin((1,), (32,), (np.int64(300_000_000),))   # ~0.1 s
        memcpy_reader(s_1, out)

    def _build(reader):
        return Step2SparseSession(
            replace(base, bold_reader=reader), mode="dMSHBM", num_clusters=L,
            dim=dim, ini_val=ini_val, beta_internal=0.0, eps_m_step=1e-4,
            max_iter_m=50, eps_intra_var=1e-4, max_iter_intra_var=20,
            bold_cache_mode="eager_bitpacked")

    # The Session's first kernel launch in a process blocks until the work
    # already queued has run, so subject 1's copy would land before the
    # refill: build once without that work first.
    _build(memcpy_reader)
    with stream:
        g = _build(queueing_reader)
        got = cp.asnumpy(g._packed)
    stream.synchronize()
    assert np.array_equal(got, packed)


def test_sync_to_host_rejects_an_unknown_field(tmp_path):
    """ValueError, not KeyError."""
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(4, tmp_path)
    dim = D - 1
    g = _gpu_session(inputs, "dMSHBM", dim,
                     float(initialize_concentration(dim)), 0.0)
    with pytest.raises(ValueError, match="unknown field"):
        g.sync_to_host(("not_a_field",))


@pytest.mark.parametrize("ini_val_case", ["natural", "underflow"])
def test_init_state_matches_compose_init_state(ini_val_case, tmp_path):
    """K1 on the synthetic cohort is bit-equal to ``compose_init_state``
    (design §1.2). The ``underflow`` case scales the logits (``ini_val``
    500) until the reference carries fp32-subnormal cells and masked rows
    whose every in-mask cell underflows, so the init's §1.10 stores
    (subnormal ``exp`` kept, all-underflow row left 0) are covered on
    device without the bench fixture.
    """
    from arealmshbm.initialize_concentration import initialize_concentration
    from arealmshbm.step2_init.compose_init_state import compose_init_state
    inputs, bm, mtc, bold, _g = _synthetic(1, tmp_path)
    dim = D - 1
    ini_val = (float(initialize_concentration(dim)) if ini_val_case == "natural"
               else 500.0)
    g = _gpu_session(inputs, "gMSHBM", dim, ini_val, 5000.0)
    g.initialize_state()
    h = g.sync_to_host(("s_lambda", "theta"))
    s_lambda, theta = compose_init_state(
        bold_loader=InMemoryProfileLoader(bold, T), group_mtc=mtc,
        boundary_mask=bm, num_clusters=L, ini_val=ini_val)
    if ini_val_case == "underflow":
        sl = s_lambda[0]
        assert ((sl != 0) & ((_bits(sl) & 0x7f800000) == 0)).any(), \
            "the fixture must contain subnormal init cells"
        assert ((sl == 0).all(axis=1) & (bm != 0).any(axis=1)).any(), \
            "the fixture must contain all-underflow rows"
    assert np.array_equal(_bits(h["s_lambda"]), _bits(s_lambda))
    assert np.array_equal(_bits(h["theta"]), _bits(theta * (bm != 0)))


def test_determinism(tmp_path):
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(2, tmp_path)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))
    fields = ("s_t_nu", "s_lambda_P", "theta_P", "log_theta", "lv_sum",
              "log_connect", "scr", "rmax", "s_psi", "sigma", "epsil", "mu")
    runs = []
    for _ in range(2):
        g = _gpu_session(inputs, "gMSHBM", dim, ini_val, 5000.0)
        g.initialize_state()
        costs = [g.run_iter()[1] for _ in range(3)]
        g.intra_closure()
        g.inter_closure()
        h = g.sync_to_host(fields)
        h["cost"] = np.concatenate(costs)
        h["kappa"] = np.float64(g.kappa)
        runs.append(h)
    for k in runs[0]:
        a = np.asarray(runs[0][k])
        b = np.asarray(runs[1][k])
        assert np.array_equal(a.view(_u(a.dtype)), b.view(_u(b.dtype))), k


def _u(dt):
    return {4: np.uint32, 8: np.uint64, 1: np.uint8}[np.dtype(dt).itemsize]


def test_bold_cache_mode_stream_matches_eager(tmp_path, monkeypatch):
    """Eager, stream with the host cache, and stream decoding through
    the reader on every visit (host cache over budget) agree bit for
    bit; only the first keeps no host copy at all."""
    pytest.importorskip("psutil")
    from arealmshbm.initialize_concentration import initialize_concentration
    import arealmshbm.step2_em_iter_master.session_gpu as sg
    inputs, _bm, _mtc, _bold, _g = _synthetic(3, tmp_path)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))

    def _build(cache):
        return Step2SparseSession(
            inputs, mode="gMSHBM", num_clusters=L, dim=dim, ini_val=ini_val,
            beta_internal=5000.0, eps_m_step=1e-4, max_iter_m=50,
            eps_intra_var=1e-4, max_iter_intra_var=20, bold_cache_mode=cache)

    out = []
    for cache, fraction in (("eager_bitpacked", 0.5), ("stream", 0.5),
                            ("stream", 1e-15)):
        monkeypatch.setattr(sg, "_HOST_CACHE_FRACTION", fraction)
        if fraction < 0.5:
            with pytest.warns(RuntimeWarning, match="host BOLD cache"):
                g = _build(cache)
        else:
            g = _build(cache)
        assert g.bold_cache_mode == cache
        if cache == "eager_bitpacked":
            assert g._host_cache is None and g._pinned_bold is None
        else:
            assert g._pinned_bold is not None
            assert (g._host_cache is not None) == (fraction == 0.5)
        g.initialize_state()
        for _ in range(2):
            g.run_iter()
        out.append(g.sync_to_host(("s_lambda_P", "theta_P", "s_t_nu")))
    for k in out[0]:
        for o in out[1:]:
            assert np.array_equal(out[0][k].view(np.uint32),
                                  o[k].view(np.uint32)), k


def test_eager_ingest_reads_each_subject_once_and_keeps_no_host_copy(tmp_path):
    """The eager ctor pulls every subject through the reader exactly
    once into the device cache; no host cache or staging slot outlives
    the constructor."""
    from arealmshbm.initialize_concentration import initialize_concentration
    base, _bm, _mtc, _bold, _g = _synthetic(6, tmp_path)
    calls = []
    real = base.bold_reader

    def counting(s_1, out):
        calls.append(s_1)
        real(s_1, out)

    inputs = replace(base, bold_reader=counting)
    dim = D - 1
    g = Step2SparseSession(
        inputs, mode="dMSHBM", num_clusters=L, dim=dim,
        ini_val=float(initialize_concentration(dim)), beta_internal=0.0,
        eps_m_step=1e-4, max_iter_m=50, eps_intra_var=1e-4,
        max_iter_intra_var=20, bold_cache_mode="eager_bitpacked")
    assert sorted(calls) == list(range(1, S + 1))
    assert g._host_cache is None and g._pinned_bold is None
    expect = np.empty((S, T, N, DB), dtype=np.uint8)
    for s in range(S):
        real(s + 1, expect[s])
    assert np.array_equal(cp.asnumpy(g._packed), expect)


@pytest.mark.parametrize("cache, fraction", [("eager_bitpacked", 0.5),
                                             ("stream", 0.5),
                                             ("stream", 1e-15)])
def test_ctor_propagates_a_reader_failure(tmp_path, monkeypatch, cache, fraction):
    """A subject the reader rejects fails the constructor on every
    residency path. Subject 2's file holds one session instead of T,
    which the loader (it opens subject 1 only) cannot see: the reader's
    header check raises when the Session pulls that subject — through
    worker threads on the eager ring and the stream-mode host cache,
    whose exception must surface, not be swallowed."""
    if cache == "stream":
        pytest.importorskip("psutil")
    import arealmshbm.step2_em_iter_master.session_gpu as sg
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(7, tmp_path)
    write_subject_profile_tnd(profile_path(tmp_path, "2", TARG, SEED),
                              np.zeros((1, N, D), dtype=np.uint8))
    monkeypatch.setattr(sg, "_HOST_CACHE_FRACTION", fraction)
    dim = D - 1
    with pytest.raises(ValueError, match="sub 2 packed shape"):
        Step2SparseSession(
            inputs, mode="dMSHBM", num_clusters=L, dim=dim,
            ini_val=float(initialize_concentration(dim)), beta_internal=0.0,
            eps_m_step=1e-4, max_iter_m=50, eps_intra_var=1e-4,
            max_iter_intra_var=20, bold_cache_mode=cache)


def test_export_params_shapes(tmp_path):
    from arealmshbm.initialize_concentration import initialize_concentration
    import scipy.sparse as sp
    inputs, bm, _mtc, _bold, _g = _synthetic(4, tmp_path)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))
    g = _gpu_session(inputs, "gMSHBM", dim, ini_val, 5000.0)
    g.initialize_state()
    g.run_iter()
    p = g.export_params()
    assert p["sigma"].shape == (1, L) and p["sigma"].dtype == np.float32
    assert p["epsil"].shape == (1, L)
    assert p["kappa"].shape == (1, L)
    assert p["mu"].shape == (L, D) and p["mu"].dtype == np.float32
    assert p["cost_em"].shape == (S,) and p["cost_em"].dtype == np.float64
    assert sp.issparse(p["theta"]) and p["theta"].shape == (N, L)
    dense = np.asarray(p["theta"].todense())
    assert np.all(dense[bm == 0] == 0.0)
    h = g.sync_to_host(("theta",))
    assert np.allclose(dense, h["theta"], rtol=0, atol=0)


def test_ctor_rejects_oversized_eager_cache(tmp_path):
    """An explicit ``eager_bitpacked`` that does not fit must raise (design §2.2)."""
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(5, tmp_path)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))
    free_gb = cp.cuda.Device().mem_info[0] / 2 ** 30
    with pytest.raises(ValueError, match="eager_bitpacked"):
        Step2SparseSession(
            inputs, mode="gMSHBM", num_clusters=L, dim=dim, ini_val=ini_val,
            beta_internal=5000.0, eps_m_step=1e-4, max_iter_m=50,
            eps_intra_var=1e-4, max_iter_intra_var=20,
            bold_cache_mode="eager_bitpacked",
            bold_cache_safety_margin_gb=free_gb + 8.0)


# ─────────────────────────────────────────────────────────────────────
# bench project (skipped when absent)
# ─────────────────────────────────────────────────────────────────────
def test_bench_matches_cpu_structurally():
    BF.skip_unless_bench("proj")
    import copy
    fx = BF.get_fixture("proj", 1)
    cfg = fx.cfg
    g = Step2SparseSession(
        fx.inputs, mode=cfg.mode, num_clusters=fx.L, dim=fx.dim,
        ini_val=fx.ini_val, beta_internal=float(cfg.beta_internal),
        eps_m_step=float(cfg.epsilon), max_iter_m=int(cfg.max_iter_m),
        eps_intra_var=float(cfg.epsilon),
        max_iter_intra_var=int(cfg.max_iter_intra_var))
    g.initialize_state()
    m_gpu = [g.run_iter()[0] for _ in range(3)]
    gh = g.sync_to_host(("theta", "s_lambda"))

    P = {k: (fx.Params[k].copy() if isinstance(fx.Params[k], np.ndarray)
             else copy.deepcopy(fx.Params[k])) for k in fx.Params}
    c = Step2EmIterSession(
        bold_loader=fx.cpu_inputs.bold_loader,
        grad_loader=fx.cpu_inputs.grad_loader,
        num_sub=fx.S, N=fx.N, T=fx.T, D=fx.D, D_grad=fx.D_grad,
        boundary_mask=fx.boundary_mask, s_psi=P["s_psi"], sigma=P["sigma"],
        mode=cfg.mode, dim=fx.dim, num_clusters=fx.L, ini_val=fx.ini_val,
        beta_internal=float(cfg.beta_internal), n_lh=fx.n_lh,
        eps_m_step=float(cfg.epsilon), max_iter_m=int(cfg.max_iter_m))
    c.upload_initial_state(P)
    c.refresh_s_psi_sigma(P["s_psi"], P["sigma"])
    m_cpu = [c.run_iter(P)[0] for _ in range(3)]

    assert m_gpu == m_cpu, (m_gpu, m_cpu)
    flips = int((gh["theta"].argmax(1) != P["theta"].argmax(1)).sum())
    assert flips <= 500, flips
    alive_g = int((gh["s_lambda"][0].sum(1) != 0).sum())
    alive_c = int((P["s_lambda"][0].sum(1) != 0).sum())
    assert abs(alive_g - alive_c) <= 100, (alive_g, alive_c)
    kap_cpu = float(np.asarray(P["kappa"]).ravel()[0])
    assert abs(g.kappa - kap_cpu) / kap_cpu <= 2e-2, (g.kappa, kap_cpu)
