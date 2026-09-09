"""test_session_gpu.py — Step2SparseSession vs the CPU master.

Three families:

1. a synthetic **binary** cohort (the on-disk BOLD contract) small enough to
   drive the CPU ``Step2EmIterSession`` in a second, run for one EM iteration
   in both gMSHBM and dMSHBM;
2. run-to-run determinism — two Sessions on the same inputs must agree with
   ``np.array_equal`` on every field;
3. the bench project (skipped when absent), with the loose structural bars of
   ``docs/step2_sparse_design.md`` §7.3.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.step2_em_iter_master import (  # noqa: E402
    Step2EmIterSession, Step2SparseSession,
)
from arealmshbm.step2_io import InMemoryGradientLoader, InMemoryProfileLoader  # noqa: E402
from arealmshbm.step2_io.load_subject_profiles import (  # noqa: E402
    _widen_normalize_bitpacked_to_f32_NTD_kernel,
)
from arealmshbm.step2_io.sparse_inputs import Step2SparseInputs  # noqa: E402
from arealmshbm.step2_io.sparse_layout import build_step2_layout_dense  # noqa: E402
from arealmshbm.step2_em_iter_master.tests import _bench_fixture as BF  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# synthetic cohort
# ─────────────────────────────────────────────────────────────────────
S, T, N, L, D, D_GRAD = 2, 2, 128, 16, 264, 8
N_LH, L_LH = N // 2, L // 2
DB = (D + 7) // 8


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    bits = (rng.random((S, T, N, D)) < 0.10).astype(np.uint8)
    packed = np.packbits(bits, axis=-1, bitorder="little")
    assert packed.shape == (S, T, N, DB)

    # Block-diagonal 0/1 mask, >= 1 cell per row.
    bm = np.zeros((N, L), dtype=np.float32)
    for n in range(N):
        lo, hi = (0, L_LH) if n < N_LH else (L_LH, L)
        k = int(rng.integers(1, 5))
        cols = rng.choice(np.arange(lo, hi), size=k, replace=False)
        bm[n, cols] = 1.0
    layout = build_step2_layout_dense(bm)

    mtc = rng.standard_normal((D, L))
    mtc /= np.linalg.norm(mtc, axis=0, keepdims=True)
    grad = rng.standard_normal((S, N, D_GRAD)).astype(np.float32)

    inputs = Step2SparseInputs.from_arrays(layout, packed, mtc,
                                           grad_SNDg=grad)
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
        num_clusters=L)
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


def _relmax(a, b, eps=1e-30):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / (np.abs(b) + eps)))


@pytest.mark.parametrize("mode", ["gMSHBM", "dMSHBM"])
def test_one_em_iter_matches_cpu(mode):
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, bm, mtc, bold, grad_STND = _synthetic(0)
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

    # init state is bit-exact vs the CPU ``compose_init_state`` output
    # (``Params['s_lambda']`` has since been overwritten by the CPU iteration,
    # so the comparison uses the pre-iteration snapshot).  ``theta`` is
    # compared on P only: the P-layout carries no out-of-P cell, where
    # ``compose_init_state`` writes f32(eps).
    assert np.array_equal(gh0["s_lambda"], init0["s_lambda"])
    assert np.array_equal(gh0["theta"], init0["theta"] * (bm != 0))
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


def test_ctor_fences_the_current_stream_not_the_null_stream():
    """The pinned staging slot is reused per subject inside the ctor.

    Built inside a non-blocking stream that already has work queued, the
    H2D copies sit behind that work; a fence on the legacy default stream
    returns immediately and lets the host refill the slot under them.
    """
    from arealmshbm.initialize_concentration import initialize_concentration
    # The per-subject payload must clear 64 KiB: CUDA inlines smaller H2D
    # copies into the pushbuffer, which hides the reuse hazard entirely.
    n_wide = 4096
    rng = np.random.default_rng(3)
    bits = (rng.random((S, T, n_wide, D)) < 0.10).astype(np.uint8)
    packed = np.packbits(bits, axis=-1, bitorder="little")
    assert packed[0].nbytes > (1 << 16)
    bm = np.zeros((n_wide, L), dtype=np.float32)
    bm[np.arange(n_wide), np.where(np.arange(n_wide) < n_wide // 2,
                                   0, L_LH)] = 1.0
    mtc = rng.standard_normal((D, L))
    mtc /= np.linalg.norm(mtc, axis=0, keepdims=True)
    inputs = Step2SparseInputs.from_arrays(
        build_step2_layout_dense(bm), packed, mtc)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))

    def _build():
        return Step2SparseSession(
            inputs, mode="dMSHBM", num_clusters=L, dim=dim, ini_val=ini_val,
            beta_internal=0.0, eps_m_step=1e-4, max_iter_m=50,
            eps_intra_var=1e-4, max_iter_intra_var=20,
            bold_cache_mode="eager_bitpacked")

    spin = cp.RawKernel(_SPIN_SRC, "_test_spin")
    spin((1,), (32,), (np.int64(1),))            # compile off the timed path
    # Warm the memory pool with this Session's exact block sizes: a pool
    # miss calls cudaMalloc, which synchronizes the device and would drain
    # the queued spin before the ctor's copies ever queue behind it.
    warm = _build()
    del warm
    cp.cuda.get_current_stream().synchronize()

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        spin((1,), (32,), (np.int64(300_000_000),))   # ~0.1 s of stream work
        g = _build()
        got = cp.asnumpy(g._packed)
    stream.synchronize()
    assert np.array_equal(got, packed)


def test_sync_to_host_rejects_an_unknown_field():
    """Same contract as the dense GPU session: ValueError, not KeyError."""
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(4)
    dim = D - 1
    g = _gpu_session(inputs, "dMSHBM", dim,
                     float(initialize_concentration(dim)), 0.0)
    with pytest.raises(ValueError, match="unknown field"):
        g.sync_to_host(("not_a_field",))


def test_init_state_matches_compose_init_state():
    from arealmshbm.initialize_concentration import initialize_concentration
    from arealmshbm.step2_init.compose_init_state import compose_init_state
    inputs, bm, mtc, bold, _g = _synthetic(1)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))
    g = _gpu_session(inputs, "gMSHBM", dim, ini_val, 5000.0)
    g.initialize_state()
    h = g.sync_to_host(("s_lambda", "theta"))
    s_lambda, theta = compose_init_state(
        bold_loader=InMemoryProfileLoader(bold, T), group_mtc=mtc,
        boundary_mask=bm, num_clusters=L)
    assert np.array_equal(h["s_lambda"], s_lambda)
    assert np.array_equal(h["theta"], theta * (bm != 0))


def test_determinism():
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(2)
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


def test_bold_cache_mode_stream_matches_eager():
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(3)
    dim = D - 1
    ini_val = float(initialize_concentration(dim))
    out = []
    for cache in ("eager_bitpacked", "stream"):
        g = Step2SparseSession(
            inputs, mode="gMSHBM", num_clusters=L, dim=dim, ini_val=ini_val,
            beta_internal=5000.0, eps_m_step=1e-4, max_iter_m=50,
            eps_intra_var=1e-4, max_iter_intra_var=20, bold_cache_mode=cache)
        assert g.bold_cache_mode == cache
        g.initialize_state()
        for _ in range(2):
            g.run_iter()
        out.append(g.sync_to_host(("s_lambda_P", "theta_P", "s_t_nu")))
    for k in out[0]:
        assert np.array_equal(out[0][k].view(np.uint32),
                              out[1][k].view(np.uint32)), k


def test_export_params_shapes():
    from arealmshbm.initialize_concentration import initialize_concentration
    import scipy.sparse as sp
    inputs, bm, _mtc, _bold, _g = _synthetic(4)
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


def test_ctor_rejects_oversized_eager_cache():
    """An explicit ``eager_bitpacked`` that does not fit must raise (design §2.2)."""
    from arealmshbm.initialize_concentration import initialize_concentration
    inputs, _bm, _mtc, _bold, _g = _synthetic(5)
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
