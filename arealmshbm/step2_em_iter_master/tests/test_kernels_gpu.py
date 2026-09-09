"""test_kernels_gpu.py — per-kernel parity of the step-2 ``gpu`` backend.

Each test feeds **identical inputs** to the CUDA kernel and to the CPU numba
reference (or to an fp64 evaluation of the same contraction) and compares on
the P-cells. The bars are the ones in ``docs/step2_sparse_design.md`` §7.1.

Tests that need the bench project skip when it is absent
(``MSHBM_STEP2_BENCH_DIR``, default ``testdata/step2_bench``); everything that
only needs cupy runs anywhere with a GPU.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.step2_em_iter_master import _kernels_gpu as K  # noqa: E402
from arealmshbm.step2_em_iter_master.tests import _bench_fixture as BF  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# cupy-only tests (no bench data needed)
# ─────────────────────────────────────────────────────────────────────
def test_device_invad_matches_invad_numba():
    """Device ``invad`` vs ``invad_numba(1174, .)`` over a randomized sweep.

    The port is bit-exact on the **asymptotic** branch only; the 7-point
    ``array_equal`` this test replaced happened to sample only round numbers
    that agree.  On the secant branch it inherits device-vs-msvcrt ``log``
    ULP differences through the iteration.  Measured over this fixed 4000-point
    sweep at ``dim = 1174`` (and reproduced on two other seeds):

    * ~490-520 of 4000 differ at all, essentially all of them <= 6.5e-13
      relative, in the band ``rbar in [0.11, 0.55]`` i.e. kappa 130-900 --
      which straddles the production kappa range;
    * about **1 in 4000** lands on a secant *early-exit* divergence, where one
      side takes an extra iteration and the roots differ by ~3e-4 relative
      (here ``rbar = 0.34147`` -> 453.5874 vs 453.7225, and that one point is
      also the only ``np.float32(kappa)`` mismatch).  This is looser than the
      "invad verified bit-exact" the design's 1.9/9 still claim, and it is
      what the bars below pin.  It stays 67x inside design 7.3's
      ``|kappa - kappa_cpu| / kappa_cpu <= 2e-2``.
    """
    from arealmshbm.m_step._invad import invad_numba
    rng = np.random.default_rng(20260903)
    rb = rng.uniform(1e-9, 1.0 - 1e-9, 4000).astype(np.float64)
    out = cp.empty(rb.size, dtype=cp.float64)
    K.module().get_function("probe_invad")(
        (K.grid(rb.size, 256),), (256,),
        (cp.asarray(rb), out, np.float64(1174.0), np.int32(rb.size)))
    got = cp.asnumpy(out)
    ref = np.array([invad_numba(1174.0, float(r)) for r in rb], dtype=np.float64)
    assert np.array_equal(np.isfinite(got), np.isfinite(ref))
    fin = np.isfinite(ref) & (ref != 0.0)
    rel = np.abs(got[fin] - ref[fin]) / np.abs(ref[fin])
    n_loose = int(np.count_nonzero(rel > 1e-12))
    assert n_loose <= 3, (n_loose, float(rel.max()))
    assert float(rel.max()) <= 1e-3, float(rel.max())
    n32 = int(np.count_nonzero(np.float32(got) != np.float32(ref)))
    assert n32 <= 3, n32


def test_module_is_one_object_across_threads():
    """``module()`` publishes one RawModule under a lock.

    The driver's prewarm calls it from a daemon thread while the main thread
    can be inside ``Step2SparseSession.__init__``; two builders would mean two
    concurrent NVRTC compiles of the identical source on a cold cupy cache.
    """
    import threading
    got = []
    err = []

    def work():
        try:
            got.append(K.module())
        except Exception as e:                      # pragma: no cover
            err.append(e)

    ths = [threading.Thread(target=work) for _ in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert not err, err
    assert len(got) == 8 and all(m is got[0] for m in got)
    assert got[0] is K.module()


def test_warmup_forces_the_nvrtc_compile():
    """``warmup_step2_gpu`` must ask for a symbol, not just build the
    lazy RawModule -- ``cp.RawModule(backend='nvrtc')`` compiles nothing in its
    constructor, so a ``module()``-only prewarm leaves the whole compile inside
    the Session ctor (measured fresh-process ``session_ctor``: 1.02 s with no
    prewarm, 0.12 s with the module-only prewarm, 0.067 s with this one).
    """
    from arealmshbm.step2_em_iter_master import warmup_step2_gpu
    warmup_step2_gpu()
    warmup_step2_gpu()               # idempotent
    # a symbol lookup after the warmup must not have to compile anything
    import time
    t0 = time.perf_counter()
    fn = K.module().get_function("estep_row1")
    assert fn is not None
    assert time.perf_counter() - t0 < 0.05


def test_device_cdln_bitexact():
    from arealmshbm.em_stop_criterion._cdln import _cdln_single
    k = np.array([403.6113960568694, 553.0, 900.0, 1200.0, 2464.0],
                 dtype=np.float64)
    out = cp.empty(k.size, dtype=cp.float64)
    K.module().get_function("probe_cdln")(
        (1,), (32,), (cp.asarray(k), out, np.float64(586.0), np.int32(k.size)))
    ref = np.array([_cdln_single(float(x), 586.0) for x in k], dtype=np.float64)
    assert np.array_equal(cp.asnumpy(out).view(np.uint64), ref.view(np.uint64))


def test_f32_rn_subnormal_bit_patterns():
    """``f32_rn`` / ``f32_to_f64`` survive CuPy's unconditional ``-ftz=true``."""
    xs = np.array([1e-40, 5e-41, 1e-44, 5e-39, 1.4e-45, -1e-40, -7e-46,
                   0.0, 1.0, 1.17549435e-38, 1.1754943e-38, 3.4e38],
                  dtype=np.float64)
    out = cp.empty(xs.size, dtype=cp.float32)
    mod = K.module()
    mod.get_function("probe_f32_rn")(
        (1,), (64,), (cp.asarray(xs), out, np.int32(xs.size)))
    got = cp.asnumpy(out)
    ref = xs.astype(np.float32)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
    assert np.any(got.view(np.uint32) & 0x7f800000 == 0) and np.any(got != 0), \
        "the fixture must actually contain subnormals"

    back = cp.empty(xs.size, dtype=cp.float64)
    mod.get_function("probe_f32_to_f64")(
        (1,), (64,), (cp.asarray(ref), back, np.int32(xs.size)))
    assert np.array_equal(cp.asnumpy(back), ref.astype(np.float64))


def test_mstep_inplace_matches_pingpong():
    """K4c must read ``old`` before the in-place store (eps = 0, >= 2 iter_m).

    Aliasing ``st_old`` with ``s_t_nu`` without that ordering makes
    ``cos == 1`` for every column and silently ends the M-step after one
    iteration.
    """
    mod = K.module()
    S, T, L, D = 2, 2, 8, 64
    rng = np.random.default_rng(7)
    xdot = cp.asarray(rng.standard_normal((S, T, L, D)).astype(np.float32))
    spsi = cp.asarray(rng.standard_normal((S, L, D)).astype(np.float32))
    st0 = cp.asarray(rng.standard_normal((S, T, L, D)).astype(np.float32))
    kap = cp.full(1, np.float32(3.5), dtype=cp.float32)
    cos_a = cp.zeros((S, T, L), dtype=cp.float32)
    cos_b = cp.zeros((S, T, L), dtype=cp.float32)

    st_in = st0.copy()
    fn = mod.get_function("mstep_fused_body")
    cos_first = None
    for it in range(3):
        fn((S * T * L,), (K.FUSED_BLOCK,),
           (kap, xdot, spsi, st_in, st_in, cos_a,
            np.int32(T), np.int32(L), np.int32(D)),
           shared_mem=D * 4)
        if it == 0:
            cos_first = cp.asnumpy(cos_a).copy()
    cos_inplace = cp.asnumpy(cos_a).copy()
    st_inplace = cp.asnumpy(st_in)
    # Iteration 1 starts from a random s_t_nu: a correct cosine is nowhere
    # near 1.  Clobbering ``old`` before the read would make it exactly
    # ``sum_d nv^2 == 1`` for every column and stop the M-step after one pass.
    assert float(np.abs(cos_first - 1.0).max()) > 0.1, cos_first

    # Ping-pong reference: distinct old/new buffers, copied back each iter.
    old = st0.copy()
    new = cp.empty_like(old)
    for _ in range(3):
        fn((S * T * L,), (K.FUSED_BLOCK,),
           (kap, xdot, spsi, old, new, cos_b,
            np.int32(T), np.int32(L), np.int32(D)),
           shared_mem=D * 4)
        cp.copyto(old, new)
    assert np.array_equal(st_inplace.view(np.uint32),
                          cp.asnumpy(old).view(np.uint32))
    assert np.array_equal(cos_inplace.view(np.uint32),
                          cp.asnumpy(cos_b).view(np.uint32))
    assert np.isfinite(cos_inplace).all()


# ─────────────────────────────────────────────────────────────────────
# bench-backed context
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ctx():
    BF.skip_unless_bench("proj")
    from arealmshbm.step2_em_iter_master import Step2SparseSession
    fx = BF.get_fixture("proj", 1)
    cfg = fx.cfg
    sess = Step2SparseSession(
        fx.inputs, mode=cfg.mode, num_clusters=fx.L, dim=fx.dim,
        ini_val=fx.ini_val, beta_internal=float(cfg.beta_internal),
        eps_m_step=float(cfg.epsilon), max_iter_m=int(cfg.max_iter_m),
        eps_intra_var=float(cfg.epsilon),
        max_iter_intra_var=int(cfg.max_iter_intra_var))
    sess.initialize_state()
    init_state = sess.sync_to_host(("s_lambda", "theta"))
    sess.run_iter()                       # iteration 1 -- runs K7
    k7 = sess.sync_to_host(("rmax_dense", "s_t_nu", "u", "u_sq", "grad_sq",
                            "n_alive"))
    k7["kappa"] = sess.kappa
    used = sess.sync_to_host(("theta_P", "log_theta"))   # theta iteration 2 reads
    _m2, cost2 = sess.run_iter()          # iteration 2 -- no K7 (theta_out == 0)
    return {"fx": fx, "sess": sess, "init": init_state, "k7": k7,
            "theta_used": used, "bold": fx.bold_f32(0),
            "cost2": np.asarray(cost2).copy()}


def test_row_stats_exact(ctx):
    """K0: the reconstructed ``(bit - mean) * inv`` is bit-equal to the CPU widen."""
    fx, sess = ctx["fx"], ctx["sess"]
    bold = ctx["bold"]
    h = sess.sync_to_host(("row_mean", "row_inv", "n_alive"))
    rm, ri = h["row_mean"][0], h["row_inv"][0]
    rng = np.random.default_rng(0)
    rows = rng.choice(fx.N, 512, replace=False)
    for n in rows:
        for t in range(fx.T):
            bits = np.unpackbits(fx.packed[0, t, n],
                                 bitorder="little")[:fx.D].astype(np.float32)
            rec = (bits - rm[t, n]).astype(np.float32) * ri[t, n]
            assert np.array_equal(rec, bold[n, t]), (n, t)
    na_ref = (np.abs(bold).sum(axis=2) != 0).sum(axis=1).astype(np.int32)
    assert np.array_equal(h["n_alive"][0], na_ref)


def test_init_matches_compose_init_state(ctx):
    """K1: hard labels / medial / s_lambda / theta equal ``compose_init_state``."""
    fx = ctx["fx"]
    init = ctx["init"]
    assert np.array_equal(init["s_lambda"], fx.Params["s_lambda"])
    theta_ref = fx.Params["theta"].copy()
    theta_ref[fx.boundary_mask == 0] = 0.0      # P-layout carries no out-of-P cell
    assert np.array_equal(init["theta"], theta_ref)


def test_x_dot_sl_bits_vs_fp64(ctx):
    """K3 on a sample of parcels against an fp64 evaluation of the same sum."""
    fx, sess = ctx["fx"], ctx["sess"]
    mod = K.module()
    lay = fx.layout
    T, N, D, Db, L = fx.T, fx.N, fx.D, fx.Db, fx.L
    sl = cp.asarray(fx.layout.gather(
        np.abs(np.random.default_rng(3).standard_normal((N, L))
               ).astype(np.float32) * (fx.boundary_mask != 0)))
    dev = {k: cp.asarray(getattr(lay, k)) for k in
           ("col_ptr", "csc_row", "csc_pidx")}
    out = cp.zeros((T, L, D), dtype=cp.float32)
    h = sess.sync_to_host(("row_mean", "row_inv"))
    rm = cp.asarray(h["row_mean"][0])
    ri = cp.asarray(h["row_inv"][0])
    packed = cp.asarray(fx.packed[0])
    args = (packed, rm, ri, dev["col_ptr"], dev["csc_row"], dev["csc_pidx"],
            sl, out, np.int32(N), np.int32(L), np.int32(D), np.int32(Db),
            np.int32(K.XDOT_CHUNK))
    smem = K.XDOT_CHUNK * 8          # member weight + row id (no byte stage)
    fn = mod.get_function("x_dot_sl_bits")
    fn((T * L,), (K.XDOT_BLOCK,), args, shared_mem=smem)
    got = cp.asnumpy(out)

    out2 = cp.zeros((T, L, D), dtype=cp.float32)
    fn((T * L,), (K.XDOT_BLOCK,),
       (packed, rm, ri, dev["col_ptr"], dev["csc_row"], dev["csc_pidx"], sl,
        out2, np.int32(N), np.int32(L), np.int32(D), np.int32(Db),
        np.int32(K.XDOT_CHUNK)), shared_mem=smem)
    assert np.array_equal(cp.asnumpy(out).view(np.uint32),
                          cp.asnumpy(out2).view(np.uint32)), "K3 not deterministic"

    bold = ctx["bold"]
    sl_host = cp.asnumpy(sl)
    rng = np.random.default_rng(11)
    parcels = rng.choice(L, 6, replace=False)
    worst = 0.0
    worst_sgemm = 0.0
    for l in int_list(parcels):
        i0, i1 = lay.col_ptr[l], lay.col_ptr[l + 1]
        mem = lay.csc_row[i0:i1]
        w = sl_host[lay.csc_pidx[i0:i1]].astype(np.float64)
        ref = np.einsum("n,ntd->td", w, bold[mem].astype(np.float64))
        sgemm = np.einsum("n,ntd->td", w.astype(np.float32),
                          bold[mem]).astype(np.float64)
        den = max(np.abs(ref).max(), 1e-3)
        worst = max(worst, float(np.abs(got[:, l, :] - ref).max() / den))
        worst_sgemm = max(worst_sgemm, float(np.abs(sgemm - ref).max() / den))
    # Design §7.1 / §9: the bit form differs two O(sum w) quantities, so it is
    # a few times less accurate than a plain fp32 dot; both are far inside the
    # kappa sensitivity budget.
    assert worst <= 1e-4, (worst, worst_sgemm)
    assert worst <= 10.0 * max(worst_sgemm, 1e-12), (worst, worst_sgemm)


def int_list(a):
    return [int(x) for x in np.asarray(a).ravel()]


def _cpu_spatial_connect(fx, s_lambda_NL, grad_N_Dg):
    """The CPU Phase-C kernel on one subject (grad replicated over T)."""
    from arealmshbm.step2_em_iter_master._kernels import (
        _spatial_connect_per_subject_numba,
    )
    T, N, L, Dg = fx.T, fx.N, fx.L, fx.D_grad
    grad_TND = np.ascontiguousarray(
        np.broadcast_to(grad_N_Dg, (T, N, Dg)).copy())
    out = np.empty((N, L), dtype=np.float32)
    _spatial_connect_per_subject_numba(
        grad_TND, np.ascontiguousarray(s_lambda_NL), out,
        fx.n_lh, fx.L_lh,
        np.empty((L, Dg), np.float32), np.empty((Dg, L), np.float32),
        np.empty(L, np.float32), np.empty(N, np.float32),
        np.empty(L, np.float32), np.empty((N, L), np.float32))
    return out


def test_connect_matches_cpu(ctx):
    """K5a/K5b: sum_lambda / u_sq / grad_sq exact, log_connect <= 1e-5 rel."""
    fx, sess = ctx["fx"], ctx["sess"]
    mod = K.module()
    L, Dg, T = fx.L, fx.D_grad, fx.T
    h = sess.sync_to_host(("s_lambda", "s_lambda_P", "grad_sq"))
    sl_NL = h["s_lambda"][0]
    # Force one empty parcel so the 0/0 -> NaN path is exercised.
    lay = fx.layout
    dead_l = int(np.argmin(np.diff(lay.col_ptr)[np.diff(lay.col_ptr) > 0]))
    dead_l = int(np.flatnonzero(np.diff(lay.col_ptr) > 0)[dead_l])
    sl_NL = sl_NL.copy()
    sl_NL[:, dead_l] = 0.0
    sl_P = cp.asarray(lay.gather(sl_NL))

    dev = {k: cp.asarray(getattr(lay, k)) for k in
           ("col_ptr", "csc_row", "csc_pidx", "p_row", "col")}
    act_p = cp.asarray(np.arange(lay.P, dtype=np.int32))
    grad = cp.asarray(fx.grad[0])
    grad_sq = cp.asarray(h["grad_sq"][0])
    sum_lambda = cp.zeros(L, dtype=cp.float32)
    u_LD = cp.zeros((L, Dg), dtype=cp.float32)
    u_sq = cp.zeros(L, dtype=cp.float32)
    lc = cp.zeros(lay.P, dtype=cp.float32)
    mod.get_function("connect_u")(
        (L,), (K.CONNECT_BLOCK,),
        (dev["col_ptr"], dev["csc_row"], dev["csc_pidx"], sl_P, grad,
         sum_lambda, u_LD, u_sq, np.int32(Dg)), shared_mem=Dg * 4)
    mod.get_function("connect_scv_P")(
        (K.grid(lay.P, 256),), (256,),
        (act_p, dev["p_row"], dev["col"], grad, grad_sq, u_LD, u_sq, lc,
         np.int32(lay.P), np.int32(Dg), np.int32(T)))

    ref = _cpu_spatial_connect(fx, sl_NL, fx.grad[0])
    ref_P = lay.gather(ref)
    got = cp.asnumpy(lc)

    sl_ref = sl_NL.sum(axis=0, dtype=np.float32)   # ascending-n fp32 serial
    got_sl = cp.asnumpy(sum_lambda)
    assert np.array_equal(got_sl, np.add.reduce(sl_NL, axis=0, dtype=np.float32))
    gsq_ref = np.einsum("nd,nd->n", fx.grad[0], fx.grad[0], dtype=np.float32)
    assert np.abs(cp.asnumpy(grad_sq) - gsq_ref).max() <= 1e-3 * np.abs(gsq_ref).max()

    nan_g = np.isnan(got)
    nan_c = np.isnan(ref_P)
    assert nan_g.any(), "the synthetic empty parcel must produce NaN"
    assert np.array_equal(nan_g, nan_c)
    fin = ~nan_g
    d = np.abs(got[fin].astype(np.float64) - ref_P[fin].astype(np.float64))
    scale = float(np.abs(ref_P[fin]).max())
    # ``vmf`` cancels three O(1) quantities down to O(0.01), so the only
    # meaningful bar is relative to the operand scale, not to the result.
    print(f"log_connect: max|d|={d.max():.3e} scale={scale:.3e} "
          f"rel={d.max() / scale:.3e}")
    assert d.max() / scale <= 1e-5, (d.max(), scale)


def test_acc_P_vs_fp64(ctx):
    """K6: lv_sum on the active support vs an fp64 ``X @ s_t_nu``."""
    fx, sess = ctx["fx"], ctx["sess"]
    h = sess.sync_to_host(("lv_sum", "s_t_nu", "active"))
    lay = fx.layout
    bold = ctx["bold"]
    act = h["active"].astype(bool)
    nu = h["s_t_nu"][0].astype(np.float64)          # (T, L, D)
    rng = np.random.default_rng(5)
    rows = rng.choice(np.flatnonzero(np.diff(lay.row_ptr) > 0), 256,
                      replace=False)
    worst = 0.0
    for n in int_list(rows):
        p0, p1 = lay.row_ptr[n], lay.row_ptr[n + 1]
        X = bold[n].astype(np.float64)              # (T, D)
        for p in range(p0, p1):
            if not act[p]:
                continue
            l = lay.col[p]
            ref = float(np.einsum("td,td->", X, nu[:, l, :]))
            worst = max(worst, abs(float(h["lv_sum"][p]) - ref)
                        / max(abs(ref), 1e-3))
    assert worst <= 3e-4, worst


def _cpu_estep_reference(lay, bm, lv_sum_P, theta_P, log_connect_P, n_alive,
                         kappa_f32, cdln, beta_f32, has_spatial):
    """Dense (N, L) TRANSCRIPTION of the CPU E-step chain.

    Transcribed op-for-op from ``step2_em_iter_master/_kernels.py``:

    * lines 686-728 -- ``_fused_estep_per_subject_NTD``'s post-sgemm chain:
      the ``log_vmf == 0`` row scan that builds ``tmp_idx`` (NOT the device's
      ``n_alive == 0`` shortcut), the fp32 ``log_vmf + log(theta)`` compose
      with ``-inf`` at ``theta <= 0``, the **UNCONDITIONAL** ``+ beta*lc``
      add that follows that branch, the nanmax row max with its NaN/inf -> 0
      fallback, and the fp64 ``exp(v - row_max)``;
    * lines 730-739 -- the dense dead-column pass over all N;
    * lines 746-800 -- the fp32 cost chain (``row_sum`` and ``c`` serial and
      ascending in ``l``, ``slc_raw = f32(scr) * bm``, the three LOG_EPS20
      floors, the ``(beta*slc)*lc`` grouping).

    E.1 is NOT transcribed -- the caller runs the real numba
    ``_phase_e1_normalize_per_subject_kernel`` on this function's fp64 scratch.

    The P-vectors are scattered into dense (N, L) arrays exactly as the
    kernels compute them: ``lv_sum`` / ``log_connect`` are only written at
    cells of the active support (K6 and K5b), and ``estep_row1`` gates its
    reads on the same ``theta > 0`` predicate, so they are scattered there and
    left 0 elsewhere.  With ``theta_out == 0`` every out-of-P and inactive
    cell is ``lam = -inf`` on both sides.
    """
    f32 = np.float32
    N, L = lay.N, lay.L
    on_P = theta_P > 0
    theta = np.zeros((N, L), dtype=np.float32)
    theta[lay.p_row, lay.col] = theta_P
    lvs = np.zeros((N, L), dtype=np.float32)
    lvs[lay.p_row[on_P], lay.col[on_P]] = lv_sum_P[on_P]
    lc = np.zeros((N, L), dtype=np.float32)
    lc[lay.p_row[on_P], lay.col[on_P]] = log_connect_P[on_P]
    if has_spatial:
        # the CPU's dense log_connect carries -inf on cross-hemisphere cells
        cross = np.zeros((N, L), dtype=bool)
        cross[:lay.n_lh, lay.L_lh:] = True
        cross[lay.n_lh:, :lay.L_lh] = True
        lc[cross] = -np.inf

    cdln_add = (n_alive.astype(np.float32) * cdln).astype(np.float32)
    log_vmf = (kappa_f32 * lvs + cdln_add[:, None]).astype(np.float32)
    row_has_zero = (log_vmf == f32(0.0)).any(axis=1)          # _kernels.py:690

    with np.errstate(divide="ignore", invalid="ignore"):
        ltheta = np.where(theta > f32(0.0),
                          np.log(theta.astype(np.float64)).astype(np.float32),
                          f32(-np.inf))
        v = np.where(theta > f32(0.0), (log_vmf + ltheta).astype(np.float32),
                     f32(-np.inf))
        if has_spatial:                                       # _kernels.py:717
            v = (v + (beta_f32 * lc).astype(np.float32)).astype(np.float32)
        rmax = np.nanmax(np.where(np.isnan(v), -np.inf, v), axis=1)
    rmax = rmax.astype(np.float32)
    rmax = np.where(np.isnan(rmax) | np.isinf(rmax), f32(0.0), rmax)
    with np.errstate(over="ignore"):
        scr = np.exp(v.astype(np.float64) - rmax.astype(np.float64)[:, None])

    cs = np.nansum(scr, axis=0)                               # _kernels.py:730
    scr[:, cs == 0.0] = 0.0

    # ---- cost (fp32, serial ascending l) --------------------------------
    slc_raw = (scr.astype(np.float32) * bm).astype(np.float32)
    row_sum = np.zeros(N, dtype=np.float32)
    for l in range(L):
        row_sum = (row_sum + slc_raw[:, l]).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        slc = np.where(row_sum[:, None] > f32(0.0),
                       (slc_raw / row_sum[:, None]).astype(np.float32),
                       f32(0.0))
    slc = np.where(np.isnan(slc), f32(0.0), slc)
    slc = np.where(row_has_zero[:, None], f32(0.0), slc)
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = np.where(theta > f32(0.0),
                      np.log(theta.astype(np.float64)).astype(np.float32),
                      K.LOG_EPS20_F32)
        ls = np.where(slc > f32(0.0),
                      np.log(slc.astype(np.float64)).astype(np.float32),
                      K.LOG_EPS20_F32)
    lt = np.where(np.isinf(lt), K.LOG_EPS20_F32, lt)
    ls = np.where(np.isinf(ls), K.LOG_EPS20_F32, ls)
    lcc = np.where(np.isnan(lc) | np.isinf(lc), K.LOG_EPS20_F32, lc)
    c = np.zeros(N, dtype=np.float32)
    for l in range(L):
        c = (c + (slc[:, l] * log_vmf[:, l]).astype(np.float32)).astype(np.float32)
        c = (c + (slc[:, l] * lt[:, l]).astype(np.float32)).astype(np.float32)
        c = (c - (slc[:, l] * ls[:, l]).astype(np.float32)).astype(np.float32)
        if has_spatial:
            c = (c + ((beta_f32 * slc[:, l]).astype(np.float32)
                      * lcc[:, l]).astype(np.float32)).astype(np.float32)
    return {"scr": scr, "rmax": rmax, "log_vmf": log_vmf,
            "row_cost": c, "tmp_idx": row_has_zero}


def _cpu_phase_e1(scr_NL_f64, row_has_zero, bm):
    """The REAL numba ``_phase_e1_normalize_per_subject_kernel`` on the scratch."""
    from arealmshbm.step2_em_iter_master._kernels import (
        _phase_e1_normalize_per_subject_kernel,
    )
    N, L = scr_NL_f64.shape
    tmp_idx = np.ascontiguousarray(
        np.broadcast_to(row_has_zero[:, None], (N, L)).copy())
    out = np.zeros((N, L), dtype=np.float32)
    _phase_e1_normalize_per_subject_kernel(
        np.ascontiguousarray(scr_NL_f64), tmp_idx,
        np.ascontiguousarray(bm, dtype=np.float32), out)
    return out


def _active_csc(lay, active_P):
    """Host build of the (act_col_ptr, act_csc_row, act_csc_pidx) triple."""
    flag = active_P[lay.csc_pidx]
    incl = np.cumsum(flag.astype(np.int32))
    col_ptr = np.zeros(lay.L + 1, dtype=np.int32)
    col_ptr[1:] = incl[lay.col_ptr[1:] - 1]
    col_ptr[1:][lay.col_ptr[1:] == 0] = 0
    return (col_ptr,
            lay.csc_row[flag].astype(np.int32),
            lay.csc_pidx[flag].astype(np.int32))


def test_estep_rows_match_cpu(ctx):
    """K8/K9/K10 vs the CPU E-step chain fed the kernels' own lv_sum.

    Design 7.1: ``scr`` within 4 ulp, ``s_lambda`` within 1 fp32 ulp with an
    identical zero pattern, per-row cost within 1e-6 rel.  The reference is
    the CPU transcription in :func:`_cpu_estep_reference` plus the real numba
    E.1 kernel -- not a restatement of the GPU kernel's own semantics.
    """
    fx, sess = ctx["fx"], ctx["sess"]
    lay = fx.layout
    h = sess.sync_to_host(("lv_sum", "log_connect", "n_alive", "scr",
                           "rmax", "log_vmf", "s_lambda_P"))
    theta_P = ctx["theta_used"]["theta_P"]
    kappa = sess.kappa
    from arealmshbm.em_stop_criterion._cdln import _cdln_single
    cdln = np.float32(_cdln_single(kappa, float(fx.dim) * 0.5 - 1.0))
    ref = _cpu_estep_reference(
        lay, fx.boundary_mask, h["lv_sum"], theta_P, h["log_connect"],
        h["n_alive"][0], np.float32(kappa), cdln,
        np.float32(fx.cfg.beta_internal), 1)

    # the CPU's ``log_vmf == 0`` row scan must select the same rows as the
    # device's ``n_alive == 0`` shortcut (design 1.1 consequence).
    assert np.array_equal(ref["tmp_idx"], h["n_alive"][0] == 0)
    assert np.array_equal(ref["log_vmf"][lay.p_row, lay.col].view(np.uint32),
                          h["log_vmf"].view(np.uint32))
    assert np.array_equal(ref["rmax"].view(np.uint32),
                          h["rmax"].view(np.uint32))

    ref_scr = ref["scr"][lay.p_row, lay.col]
    got_scr = h["scr"]
    assert np.array_equal(np.isnan(got_scr), np.isnan(ref_scr))
    ok = np.isfinite(ref_scr) & np.isfinite(got_scr)
    ulp = np.abs(got_scr[ok] - ref_scr[ok]) / np.maximum(
        np.abs(np.spacing(ref_scr[ok])), 1e-320)
    assert float(np.max(ulp)) <= 4.0, float(np.max(ulp))

    sl_ref = _cpu_phase_e1(ref["scr"], ref["tmp_idx"], fx.boundary_mask)
    ref_sl = sl_ref[lay.p_row, lay.col]
    got_sl = h["s_lambda_P"][0]
    assert np.array_equal(got_sl == 0, ref_sl == 0)
    nz = got_sl != 0
    d_ulp = np.abs(got_sl[nz].astype(np.float64) - ref_sl[nz].astype(np.float64)) \
        / np.spacing(ref_sl[nz].astype(np.float64))
    assert float(np.max(d_ulp)) <= 1.0, float(np.max(d_ulp))

    # per-row cost: the device folds the fp32 row costs through an fp64 tree
    # (design 1.6), so compare against the fp64 sum of the CPU row costs.
    cost_ref = float(ref["row_cost"].astype(np.float64).sum())
    cost_gpu = float(ctx["cost2"][0])
    assert abs(cost_gpu - cost_ref) <= 1e-6 * abs(cost_ref), \
        (cost_gpu, cost_ref, abs(cost_gpu - cost_ref) / abs(cost_ref))


def test_estep_survives_an_emptied_parcel(ctx):
    """K5 -> K8 -> K9 -> K10 with one parcel synthetically emptied.

    Zeroing every member weight of one parcel drives ``sum_lambda[l] = 0 ->
    inv_l = +inf -> u = 0*inf = NaN -> log_connect = NaN`` for that whole
    same-hemisphere column.  On the device the column's in-P cells are
    inactive (``theta = 0``), so ``estep_row1`` writes ``lam = -inf`` with no
    ``beta*lc`` term and ``scr = 0``; the CPU gets ``-inf + beta*NaN = NaN``
    and zeroes the column in its dead-column pass.  The two must agree on
    ``s_lambda`` -- that equivalence is the invariant documented at
    ``estep_row1``, and it is what makes K9's NaN sweep over the ACTIVE CSC
    alone sufficient.
    """
    fx, sess = ctx["fx"], ctx["sess"]
    mod = K.module()
    lay = fx.layout
    N, L, P, T, Dg = fx.N, fx.L, lay.P, fx.T, fx.D_grad
    h = sess.sync_to_host(("s_lambda", "grad_sq", "n_alive", "lv_sum"))

    sizes = np.diff(lay.col_ptr)
    dead_l = int(np.flatnonzero(sizes > 0)[int(np.argmin(sizes[sizes > 0]))])
    sl_NL = h["s_lambda"][0].copy()
    sl_NL[:, dead_l] = 0.0
    theta_NL = (sl_NL != 0).astype(np.float32) * np.float32(0.5)
    theta_P = lay.gather(theta_NL)
    sl_P = cp.asarray(lay.gather(sl_NL))
    active_P = theta_P > 0
    act_col_ptr, act_csc_row, act_csc_pidx = _active_csc(lay, active_P)
    act_p = np.flatnonzero(active_P).astype(np.int32)
    n_act = int(act_p.size)

    dev = {k: cp.asarray(getattr(lay, k)) for k in ("row_ptr", "col", "p_row")}
    d_actp = cp.asarray(act_col_ptr)
    d_pidx = cp.asarray(act_csc_pidx)
    theta_d = cp.asarray(theta_P)
    with np.errstate(divide="ignore"):
        log_theta_d = cp.asarray(
            np.where(theta_P > 0, np.log(theta_P.astype(np.float64)),
                     -np.inf).astype(np.float32))
    grad = cp.asarray(fx.grad[0])
    grad_sq = cp.asarray(h["grad_sq"][0])
    sum_lambda = cp.zeros(L, dtype=cp.float32)
    u_LD = cp.zeros((L, Dg), dtype=cp.float32)
    u_sq = cp.zeros(L, dtype=cp.float32)
    lc = cp.zeros(P, dtype=cp.float32)
    mod.get_function("connect_u")(
        (L,), (K.CONNECT_BLOCK,),
        (d_actp, cp.asarray(act_csc_row), d_pidx, sl_P, grad, sum_lambda,
         u_LD, u_sq, np.int32(Dg)), shared_mem=Dg * 4)
    mod.get_function("connect_scv_P")(
        (K.grid(max(n_act, 1), 256),), (256,),
        (cp.asarray(act_p), dev["p_row"], dev["col"], grad, grad_sq, u_LD,
         u_sq, lc, np.int32(n_act), np.int32(Dg), np.int32(T)))
    assert float(cp.asnumpy(sum_lambda)[dead_l]) == 0.0
    assert np.isnan(cp.asnumpy(u_sq)[dead_l]), "the emptied parcel must go NaN"

    lv_sum = cp.asarray(h["lv_sum"])
    n_alive = cp.asarray(h["n_alive"][0])
    kappa = sess.kappa
    from arealmshbm.em_stop_criterion._cdln import _cdln_single
    cdln = np.float32(_cdln_single(kappa, float(fx.dim) * 0.5 - 1.0))
    kf = cp.asarray(np.array([np.float32(kappa)], dtype=np.float32))
    cd = cp.asarray(np.array([cdln], dtype=np.float32))
    beta = np.float32(fx.cfg.beta_internal)
    log_vmf = cp.zeros(P, dtype=cp.float32)
    scr = cp.zeros(P, dtype=cp.float64)
    rmax = cp.zeros(N, dtype=cp.float32)
    out_sl = cp.zeros(P, dtype=cp.float32)
    cost_part = cp.zeros(K.grid(N, K.ROW_BLOCK), dtype=cp.float64)
    mod.get_function("estep_row1")(
        (K.grid(N, K.ROW_BLOCK),), (K.ROW_BLOCK,),
        (dev["row_ptr"], dev["col"], lv_sum, theta_d, log_theta_d, lc,
         n_alive, kf, cd, rmax, beta, np.int32(1), np.int32(0), np.int32(N),
         log_vmf, scr, rmax))
    mod.get_function("dead_col")((L,), (256,), (d_actp, d_pidx, scr))
    mod.get_function("estep_row2")(
        (K.grid(N, K.ROW_BLOCK),), (K.ROW_BLOCK,),
        (dev["row_ptr"], scr, theta_d, log_vmf, lc, n_alive, beta,
         np.int32(1), np.float32(K.LOG_EPS20_F32), np.int32(N), out_sl,
         cost_part))
    got = cp.asnumpy(out_sl)
    assert np.isfinite(got).all(), "an emptied parcel must not poison s_lambda"
    assert np.isfinite(cp.asnumpy(cost_part)).all(), "cost must stay non-NaN"

    ref = _cpu_estep_reference(
        lay, fx.boundary_mask, cp.asnumpy(lv_sum), theta_P, cp.asnumpy(lc),
        cp.asnumpy(n_alive), np.float32(kappa), cdln, beta, 1)
    sl_ref = _cpu_phase_e1(ref["scr"], ref["tmp_idx"], fx.boundary_mask)
    ref_P = sl_ref[lay.p_row, lay.col]
    assert np.array_equal(got == 0, ref_P == 0)
    nz = got != 0
    assert float(np.max(np.abs(got[nz].astype(np.float64)
                               - ref_P[nz].astype(np.float64))
                        / np.spacing(ref_P[nz].astype(np.float64)))) <= 1.0


def test_theta_e2_and_active_support(ctx):
    """K12: emulated fp32 mean, log_theta and the rebuilt active lists."""
    fx, sess = ctx["fx"], ctx["sess"]
    lay = fx.layout
    h = sess.sync_to_host(("s_lambda_P", "theta_P", "log_theta", "active"))
    sl = h["s_lambda_P"]
    ref = np.zeros(lay.P, dtype=np.float32)
    for s in range(fx.S):
        ref = (ref + sl[s]).astype(np.float32)
    ref = (ref * np.float32(1.0 / fx.S)).astype(np.float32)
    assert np.array_equal(h["theta_P"].view(np.uint32), ref.view(np.uint32))
    lt_ref = np.where(ref > 0, np.log(ref.astype(np.float64)),
                      -np.inf).astype(np.float32)
    assert np.array_equal(h["log_theta"].view(np.uint32),
                          lt_ref.view(np.uint32))
    # The compaction flag is the UNION supp(theta) | supp(s_lambda), not
    # supp(theta): for S >= 2 a cell whose only weight is 2^-149 rounds
    # theta to +0 (exact half-way tie) while s_lambda stays nonzero, and
    # K3 / K5a must still visit it.  At S=1 inv_S_f32 == 1.0f so the two
    # sets coincide and this fixture pins the theta half exactly.
    act_ref = (ref != 0) | (sl != 0).any(axis=0)
    assert np.array_equal(h["active"].astype(bool), act_ref)
    assert int(h["active"].sum()) == sess.n_active


def test_theta_e2_keeps_subnormals():
    """§1.10: a subnormal ``s_lambda`` must survive E.2 with numpy's bits."""
    mod = K.module()
    P = 8
    vals = np.array([1e-40, 5e-41, 1.4e-45, 7e-46, 1e-38, 0.0, 3e-45, 1.0],
                    dtype=np.float32)
    sl = cp.asarray(vals.reshape(1, P))
    theta = cp.zeros(P, dtype=cp.float32)
    log_theta = cp.zeros(P, dtype=cp.float32)
    active = cp.zeros(P, dtype=cp.int32)
    mod.get_function("theta_mean_logtheta")(
        (1,), (32,), (sl, theta, log_theta, active, np.int32(1), np.int64(P),
                      np.float32(1.0)))
    ref = (np.float32(0.0) + vals).astype(np.float32) * np.float32(1.0)
    assert np.array_equal(cp.asnumpy(theta).view(np.uint32),
                          ref.view(np.uint32))
    assert np.array_equal(cp.asnumpy(active).astype(bool), ref != 0)


def test_theta_e2_active_covers_the_subnormal_tie():
    """S >= 2: ``theta`` can round to +0 while ``s_lambda`` is still nonzero.

    ``acc = 2^-149``, ``acc * f32(1/2) = 2^-150`` is an exact half-way tie and
    rounds half-to-even to +0.  The compaction flag must be the union of
    ``supp(theta)`` and ``supp(s_lambda)`` or K3 / K5a silently drop a member
    the CPU visits (design 2.3; measured 158 such cells at EM iter 1 on the
    S=2 bench).
    """
    mod = K.module()
    P = 4
    tiny = np.float32(1.4012984643e-45)          # 2^-149
    sl_h = np.zeros((2, P), dtype=np.float32)
    sl_h[0, 0] = tiny                            # tie -> theta == 0
    sl_h[0, 1] = np.float32(2.0) * tiny          # 2^-148 -> theta == 2^-149
    sl_h[0, 2] = np.float32(1.0)
    # column 3 stays all-zero
    sl = cp.asarray(sl_h)
    theta = cp.zeros(P, dtype=cp.float32)
    log_theta = cp.zeros(P, dtype=cp.float32)
    active = cp.zeros(P, dtype=cp.int32)
    mod.get_function("theta_mean_logtheta")(
        (1,), (32,), (sl, theta, log_theta, active, np.int32(2), np.int64(P),
                      np.float32(0.5)))
    th = cp.asnumpy(theta)
    act = cp.asnumpy(active).astype(bool)
    assert th[0] == np.float32(0.0), th[0]        # the tie really rounds to 0
    assert th[1] == tiny
    assert act[0] and act[1] and act[2], act      # s_lambda != 0 keeps them on
    assert not act[3]                             # nothing anywhere -> off


def test_rmax_out_matches_dense_reference(ctx):
    """K7: the iteration-1 out-of-P row max, incl. the cross-hemisphere rule."""
    fx, sess = ctx["fx"], ctx["sess"]
    lay = fx.layout
    h = ctx["k7"]
    bold = ctx["bold"]
    kappa_f32 = np.float32(h["kappa"])
    from arealmshbm.em_stop_criterion._cdln import _cdln_single
    cdln = np.float32(_cdln_single(h["kappa"], float(fx.dim) * 0.5 - 1.0))
    theta_out = np.float32(np.finfo(np.float64).eps)
    log_theta_out = np.float32(math.log(float(theta_out)))
    beta = np.float32(fx.cfg.beta_internal)

    nu = h["s_t_nu"][0].transpose(0, 2, 1).reshape(fx.T * fx.D, fx.L)
    rng = np.random.default_rng(21)
    rows = rng.choice(fx.N, 48, replace=False)
    worst = 0.0
    for n in int_list(rows):
        X = bold[n].reshape(-1)
        lv = X.astype(np.float64) @ nu.astype(np.float64)
        cross = fx.grad[0][n].astype(np.float64) @ h["u"].T.astype(np.float64)
        inP = np.zeros(fx.L, dtype=bool)
        inP[lay.col[lay.row_ptr[n]:lay.row_ptr[n + 1]]] = True
        same = (np.arange(fx.L) < fx.L_lh) == (n < fx.n_lh)
        best = -np.inf
        for l in range(fx.L):
            if inP[l] or not same[l]:
                continue
            lam = (kappa_f32 * np.float32(lv[l])
                   + np.float32(h["n_alive"][0][n]) * cdln) + log_theta_out
            vmf = (np.float32(2.0) * np.float32(cross[l])
                   - h["grad_sq"][0][n]) - h["u_sq"][l]
            lc = np.float32(0.0)
            for _ in range(fx.T):
                lc = np.float32(lc + vmf)
            lam = np.float32(lam + np.float32(beta * lc))
            if lam == lam and lam > best:
                best = float(lam)
        got = float(h["rmax_dense"][n])
        if not np.isfinite(best):
            assert not np.isfinite(got)
            continue
        worst = max(worst, abs(got - best) / max(abs(best), 1.0))
    assert worst <= 1e-3, worst


def test_outer_leaves_match_numba(ctx):
    """K13 vs ``intra_subject_var_loop`` / ``intra_em_cost`` / ``inter_subject_var``.

    Bars are design 7.1's: ``psi`` ``np.array_equal``, ``sigma`` / ``mu`` /
    ``epsil`` <= 1e-9 rel, cost <= 1e-10 rel.  ``intra_sigma_dot`` and
    ``cost_terms_perl`` are block-per-parcel fixed trees (design 3, K13b/K13d),
    so they are reassociated with respect to the numba leaves' serial fp64
    loops; MEASURED on this fixture (S=1, bench ``proj``) they still come out
    at rel 0.0 for sigma / mu / cost, with ``psi`` bit-equal.  ``inter_eps_dot``
    is deliberately left serial -- see the comment on that kernel: a tree there
    moves ``epsil`` to 5.40e-07 and shifts the whole S=1 outer-EM trajectory.
    """
    from arealmshbm.step2_em_outer import (
        intra_subject_var_loop, inter_subject_var, intra_em_cost_step2,
    )
    fx, sess = ctx["fx"], ctx["sess"]
    cfg = fx.cfg
    h = sess.sync_to_host(("s_t_nu", "s_psi", "sigma", "epsil", "mu",
                           "cost_em"))
    P = {"s_t_nu": h["s_t_nu"], "s_psi": h["s_psi"],
         "sigma": h["sigma"].reshape(1, -1), "epsil": h["epsil"].reshape(1, -1),
         "mu": h["mu"], "cost_em": h["cost_em"]}
    psi_new, sigma_new, _ = intra_subject_var_loop(
        s_t_nu=P["s_t_nu"], s_psi_in=P["s_psi"], sigma_in=P["sigma"],
        epsil=P["epsil"], mu=P["mu"], dim=fx.dim, ini_val=fx.ini_val,
        epsilon=float(cfg.epsilon), max_iter=int(cfg.max_iter_intra_var))
    Pc = dict(P, s_psi=psi_new, sigma=sigma_new)
    cost_ref = intra_em_cost_step2(Pc, dim=fx.dim)
    mu_ref, eps_ref = inter_subject_var(
        s_psi=psi_new, prev_mu=P["mu"], prev_epsil=P["epsil"], dim=fx.dim,
        ini_val=fx.ini_val)

    cost_gpu = sess.intra_closure()
    g1 = sess.sync_to_host(("s_psi", "sigma"))
    sess.inter_closure()
    g2 = sess.sync_to_host(("mu", "epsil"))

    assert np.array_equal(g1["s_psi"], psi_new)
    assert _rel(g1["sigma"], np.asarray(sigma_new).ravel()) <= 1e-9
    assert abs(cost_gpu - cost_ref) <= 1e-10 * abs(cost_ref)
    assert _rel(g2["mu"], mu_ref) <= 1e-9
    assert _rel(g2["epsil"], np.asarray(eps_ref).ravel()) <= 1e-9


def test_outer_leaves_zero_norm_column():
    """§1.8: ``nrm == 0`` writes psi = 0 and cos = 0 (NOT the M-step's NaN)."""
    mod = K.module()
    S, T, L, D = 2, 2, 4, 32
    rng = np.random.default_rng(1)
    nu = rng.standard_normal((S, T, L, D)).astype(np.float32)
    nu[0, :, 1, :] = 0.0
    psi_prev = rng.standard_normal((S, L, D)).astype(np.float32)
    sigma = np.full(L, np.float32(2.0), dtype=np.float32)
    eps_mu = rng.standard_normal((L, D)).astype(np.float32)
    eps_mu[1] = 0.0                       # sigma*0 + 0 -> zero column at (0,1)
    d_nu = cp.asarray(nu)
    d_prev = cp.asarray(psi_prev)
    d_new = cp.empty((S, L, D), dtype=cp.float32)
    d_cos = cp.full((L, S), np.float32(9.0), dtype=cp.float32)
    mod.get_function("intra_psi_iter")(
        (S * L,), (256,),
        (d_nu, d_prev, d_new, cp.asarray(sigma), cp.asarray(eps_mu), d_cos,
         np.int32(S), np.int32(T), np.int32(L), np.int32(D)),
        shared_mem=D * 4)
    new = cp.asnumpy(d_new)
    cos = cp.asnumpy(d_cos)
    assert np.array_equal(new[0, 1], np.zeros(D, dtype=np.float32))
    assert cos[1, 0] == np.float32(0.0)
    assert np.isfinite(new).all() and np.isfinite(cos).all()


def _rel(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def test_e1_store_keeps_subnormals():
    """§1.10 round-trip: an E.1 quotient of ~1e-40 must store numpy's bits.

    A native fp64->fp32 convert flushes it to +0 under CuPy's ``-ftz=true``,
    which would kill the cell for the rest of the pipeline (``theta == 0`` is
    absorbing).
    """
    mod = K.module()
    P, N, L = 3, 1, 3
    row_ptr = cp.asarray(np.array([0, P], dtype=np.int32))
    scr_h = np.array([1.0, 1e-40, 7e-46], dtype=np.float64)
    scr = cp.asarray(scr_h)
    theta = cp.asarray(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    log_vmf = cp.zeros(P, dtype=cp.float32)
    log_connect = cp.zeros(P, dtype=cp.float32)
    n_alive = cp.asarray(np.array([6], dtype=np.int32))
    out = cp.zeros(P, dtype=cp.float32)
    part = cp.zeros(1, dtype=cp.float64)
    mod.get_function("estep_row2")(
        (1,), (K.ROW_BLOCK,),
        (row_ptr, scr, theta, log_vmf, log_connect, n_alive,
         np.float32(0.0), np.int32(0), np.float32(K.LOG_EPS20_F32),
         np.int32(N), out, part))
    rs = scr_h.sum()
    ref = (scr_h / rs).astype(np.float32)
    got = cp.asnumpy(out)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (got, ref)
    assert (ref.view(np.uint32)[1] & 0x7f800000) == 0 and ref[1] != 0, \
        "the fixture must land in the fp32 subnormal band"


def test_check_dims_limits():
    """The four static limits of the backend (design §3 / §9)."""
    K.check_dims(1175, 147, 300, 1174, 100)                # the production shape
    with pytest.raises(ValueError, match="fsaverage3"):
        K.check_dims(2563, 321, 300, 2562, 100)            # fsaverage4 seed mesh
    with pytest.raises(ValueError, match="num_clusters"):
        K.check_dims(1175, 147, 1024, 1174, 100)
    with pytest.raises(ValueError, match="n_grad_components"):
        K.check_dims(1175, 147, 300, 1174, K.MAX_D_GRAD + 1)
    K.check_dims(1175, 147, 300, 1174, K.MAX_D_GRAD)
    with pytest.raises(ValueError, match="dim/2"):
        K.check_dims(41, 6, 300, 40, 100)                  # v = 19 < 25


def test_connect_u_shared_limit_is_exact():
    """``MAX_D_GRAD`` is the largest ``D_grad`` ``connect_u`` can launch with.

    Pinned against the compiled kernel's static shared usage and the device's
    per-block default so an edit to ``sh_w`` / ``sh_n`` cannot silently move
    the ceiling the config validators mirror as a literal.
    """
    fn = K.module().get_function("connect_u")
    static = int(fn.shared_size_bytes)
    limit = int(cp.cuda.Device().attributes["MaxSharedMemoryPerBlock"])
    assert static == K.CONNECT_STATIC_SHARED
    assert 4 * K.MAX_D_GRAD + static <= limit
    assert limit < 4 * (K.MAX_D_GRAD + 1) + static
