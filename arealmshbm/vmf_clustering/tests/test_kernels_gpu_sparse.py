# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Kernel-level checks of ``_kernels_gpu_sparse`` against the CPU numba
reference kernels on sub-001 (skipped without cupy / the data store).

Each test feeds *identical* inputs (gathered at the candidate set) to
both sides and compares the P-cell outputs; the dense CPU outputs are
also checked to be zero outside P (the support invariant the sparse
backend relies on).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arealmshbm.vmf_clustering.tests._sub001_fixture import (
    load_sub001, skip_unless_cupy, skip_unless_sub001,
)


def _row_stats_host(packed_TND: np.ndarray, D: int):
    """Mirror of the device row-stat formula (design §1)."""
    T, N, Db = packed_TND.shape
    pop = np.unpackbits(packed_TND, axis=-1, bitorder="little").sum(axis=-1).astype(np.int64)
    mean_d = pop / float(D)
    post = pop - float(D) * mean_d * mean_d
    inv = np.where(post > 0, 1.0 / np.sqrt(np.where(post > 0, post, 1.0)), 1.0)
    inv = np.where((pop == 0) | (pop == D), 0.0, inv)
    return mean_d.astype(np.float32), inv.astype(np.float32)


@pytest.fixture(scope="module")
def ctx():
    skip_unless_cupy()
    fx = skip_unless_sub001()
    import cupy as cp
    from arealmshbm.vmf_clustering import _kernels_gpu_sparse as K
    from arealmshbm.vmf_clustering.sparse_layout import layout_to_device
    from arealmshbm.data_io.bitpacked_norm import unpack_normalize_packed_NTD_host

    lay_h = fx.layout
    lay = layout_to_device(lay_h)
    p_row_m = K.p_row_m_of(lay)
    N, L, T, D = lay_h.N, lay_h.L, fx.packed_TND.shape[0], fx.D
    # Dense fp32 BOLD rows (CPU reference) — 2.3 GB, built once.
    ds_NTD = unpack_normalize_packed_NTD_host(
        np.ascontiguousarray(np.transpose(fx.packed_TND, (1, 0, 2))), D)
    mean_h, inv_h = _row_stats_host(fx.packed_TND, D)
    theta = fx.theta
    mu = fx.Params["mu"]
    s_t_nu_DLT = np.ascontiguousarray(np.broadcast_to(mu[..., None], (D, L, T)))
    s_t_nu_TLD = np.ascontiguousarray(np.transpose(s_t_nu_DLT, (2, 1, 0)))
    # Reference acc via the CPU fused sgemm.
    acc_ref = ds_NTD.reshape(N, T * D) @ np.ascontiguousarray(
        np.transpose(s_t_nu_DLT, (2, 0, 1))).reshape(T * D, L)
    return dict(
        fx=fx, cp=cp, K=K, lay_h=lay_h, lay=lay, p_row_m=p_row_m,
        N=N, L=L, T=T, D=D, ds_NTD=ds_NTD, mean_h=mean_h, inv_h=inv_h,
        theta=theta, mu=mu, s_t_nu_TLD=s_t_nu_TLD, acc_ref=acc_ref,
        packed_dev=cp.asarray(fx.packed_TND),
        mean_dev=cp.asarray(mean_h), inv_dev=cp.asarray(inv_h),
    )


def test_acc_bits_matches_sgemm(ctx):
    cp, K, lay = ctx["cp"], ctx["K"], ctx["lay"]
    T, L, D = ctx["T"], ctx["L"], ctx["D"]
    stnu = cp.asarray(ctx["s_t_nu_TLD"])
    S = cp.empty((T, L), dtype=cp.float64)
    allzero = cp.empty((T, L), dtype=cp.uint8)
    anynan = cp.empty((T, L), dtype=cp.uint8)
    col_zero = cp.empty(L, dtype=cp.uint8)
    col_nan = cp.empty(L, dtype=cp.uint8)
    K.stnu_col_stats(stnu, S, allzero, anynan, col_zero, col_nan)
    assert not col_zero.any() and not col_nan.any()
    np.testing.assert_allclose(S.get(), ctx["s_t_nu_TLD"].astype(np.float64).sum(axis=2),
                               rtol=1e-12, atol=1e-12)
    acc = cp.empty(lay["P"], dtype=cp.float32)
    K.acc_bits(lay, ctx["packed_dev"], ctx["mean_dev"], ctx["inv_dev"], stnu, S, D, acc)
    cp.cuda.Device().synchronize()
    acc_h = acc.get()
    ref = ctx["lay_h"].gather(ctx["acc_ref"])
    err = np.abs(acc_h - ref)
    scale = np.maximum(np.abs(ref), 1e-3)
    print(f"acc_bits: max abs {err.max():.3e}  max rel {(err / scale).max():.3e}")
    assert (err / scale).max() < 3e-4
    # determinism
    acc2 = cp.empty_like(acc)
    K.acc_bits(lay, ctx["packed_dev"], ctx["mean_dev"], ctx["inv_dev"], stnu, S, D, acc2)
    assert cp.array_equal(acc, acc2)
    # timing
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        K.acc_bits(lay, ctx["packed_dev"], ctx["mean_dev"], ctx["inv_dev"], stnu, S, D, acc)
    cp.cuda.Device().synchronize()
    print(f"acc_bits: {(time.perf_counter() - t0) / 10 * 1e3:.3f} ms/call")


def _cpu_estep(ctx, s_lambda, acc, kappa_f32, cdln_T, col_zero, log_theta, scv, sxv, bm):
    from arealmshbm.V_lambda import _kernels as VK
    from arealmshbm.vmf_clustering import _kernels as EK
    lay_h = ctx["lay_h"]
    N, L = ctx["N"], ctx["L"]
    fx = ctx["fx"]
    sp = fx.setting_params
    M = lay_h.M_active
    V_lam = np.zeros((M, L), dtype=np.float32)
    nbh_NM = np.ascontiguousarray(lay_h.neighborhood, dtype=np.int64)
    row_idx_active = lay_h.row_idx_active.astype(np.int64)
    # CPU candidate index from the V_lambda setup (column-major).
    from arealmshbm.V_lambda.setup import build_candidate_index
    row_idx, col_idx = build_candidate_index(fx.theta, fx.theta)
    VK.v_lambda_potts_closeform_fused_full_lam_f32(
        nbh_NM, s_lambda, row_idx_active, row_idx, col_idx, V_lam,
        np.empty(M, np.float32), np.empty(M, np.float32))
    inv_active = lay_h.inv_active.astype(np.int64)
    out = np.empty((N, L), np.float32)
    V_temp = np.empty((N, L), np.float32)
    drift = EK.fused_v_lambda_assemble_softmax_drift_f32(
        acc, kappa_f32, cdln_T, col_zero, log_theta, np.float32(sp["w"]),
        V_lam, inv_active, np.float32(sp["c"]), np.asarray(sp["beta"], np.float32),
        scv, sxv, bm, s_lambda, np.empty(L, np.float32), V_temp, out)
    return out, V_temp, drift


def test_estep_iteration_matches_cpu(ctx):
    cp, K, lay, lay_h = ctx["cp"], ctx["K"], ctx["lay"], ctx["lay_h"]
    fx = ctx["fx"]; sp = fx.setting_params
    N, L, T = ctx["N"], ctx["L"], ctx["T"]
    from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32
    from arealmshbm.spatial_priors.spatial_connect import ConnectSession
    kappa = np.full(L, fx.ini_val, dtype=np.float64)
    cdln = np.empty(L, np.float32)
    cdln_general_to_f32(kappa, int(sp["dim"]), cdln)
    cdln_T = (cdln * np.float32(T)).astype(np.float32)
    kappa_f32 = kappa.astype(np.float32)
    col_zero = np.zeros(L, dtype=np.bool_)
    theta = ctx["theta"]
    log_theta = np.where(theta > 0, np.log(theta, where=theta > 0, out=np.full_like(theta, -np.inf)),
                         np.float32(-np.inf)).astype(np.float32)
    conn = ConnectSession(fx.grad, N, L)
    _, scv = conn.compute(theta)
    scv = np.ascontiguousarray(scv)
    sxv = np.zeros((N, L), np.float32)
    bm = fx.boundary_mask
    acc = np.ascontiguousarray(ctx["acc_ref"], dtype=np.float32)

    # ── CPU: two λ-iterations from s_lambda = theta ──
    out1, Vt1, d1 = _cpu_estep(ctx, theta, acc, kappa_f32, cdln_T, col_zero, log_theta, scv, sxv, bm)
    out2, Vt2, d2 = _cpu_estep(ctx, out1, acc, kappa_f32, cdln_T, col_zero, log_theta, scv, sxv, bm)
    # support invariant
    assert np.count_nonzero(out1) == np.count_nonzero(lay_h.gather(out1))
    assert np.count_nonzero(out2) == np.count_nonzero(lay_h.gather(out2))

    # ── GPU ──
    ws = K.EStepWorkspace(lay_h.M_active)
    dev = lambda a: cp.asarray(np.ascontiguousarray(a))
    lam = dev(lay_h.gather(theta))
    acc_P = dev(lay_h.gather(acc))
    log_theta_P = dev(lay_h.gather(log_theta))
    scv_P = dev(lay_h.gather(scv))
    sxv_P = dev(lay_h.gather(sxv))
    bm_P = dev(lay_h.gather(bm))
    poison = cp.zeros(lay_h.M_active, dtype=cp.uint8)
    V_temp = cp.empty(lay_h.P, dtype=cp.float32)
    out = cp.empty(lay_h.P, dtype=cp.float32)
    beta = dev(np.asarray(sp["beta"], np.float32))
    args = (lay, ws, None, acc_P, dev(kappa_f32), dev(cdln_T), dev(col_zero.astype(np.uint8)),
            log_theta_P, sp["w"], sp["c"], beta, scv_P, sxv_P, bm_P, poison, V_temp, out)
    g1 = K.estep_iteration(*(args[:2] + (lam,) + args[3:]))
    out_h = out.get(); Vt_h = V_temp.get()
    ref_out = lay_h.gather(out1); ref_Vt = lay_h.gather(Vt1)
    assert np.array_equal(Vt_h, ref_Vt), "V_lambda must be bit-identical to the CPU kernel"
    diff = np.abs(out_h - ref_out)
    print(f"estep iter1: max |Δout| {diff.max():.3e}, n(Δ≠0) {np.count_nonzero(diff)}, "
          f"drift cpu {d1:.12e} gpu {g1:.12e}")
    assert diff.max() <= 2e-7
    assert abs(g1 - d1) <= 1e-9 * max(1.0, abs(d1))
    # second iteration from the GPU output
    out_b = cp.empty_like(out)
    g2 = K.estep_iteration(*(args[:2] + (out,) + args[3:-1] + (out_b,)))
    diff2 = np.abs(out_b.get() - lay_h.gather(out2))
    print(f"estep iter2: max |Δout| {diff2.max():.3e}, drift cpu {d2:.12e} gpu {g2:.12e}")
    assert diff2.max() <= 4e-7
    assert abs(g2 - d2) <= 1e-8 * max(1.0, abs(d2))
    # determinism + timing
    out_c = cp.empty_like(out)
    g3 = K.estep_iteration(*(args[:2] + (out,) + args[3:-1] + (out_c,)))
    assert g3 == g2 and cp.array_equal(out_b, out_c)
    t0 = time.perf_counter()
    for _ in range(20):
        K.estep_iteration(*(args[:2] + (out,) + args[3:-1] + (out_c,)))
    print(f"estep_iteration: {(time.perf_counter() - t0) / 20 * 1e3:.3f} ms/iter (incl. sync)")


def test_argmax_labels(ctx):
    cp, K, lay, lay_h = ctx["cp"], ctx["K"], ctx["lay"], ctx["lay_h"]
    from arealmshbm.data_io import derive_labels
    theta = ctx["theta"]
    lh, rh = derive_labels(theta)
    labels = cp.empty(lay_h.N, dtype=cp.int32)
    K.argmax_labels(lay, cp.asarray(lay_h.gather(theta)), labels)
    assert np.array_equal(labels.get(), np.concatenate([lh, rh]).astype(np.int32))


def test_connect_prior(ctx):
    cp, K, lay, lay_h = ctx["cp"], ctx["K"], ctx["lay"], ctx["lay_h"]
    fx = ctx["fx"]; N, L = ctx["N"], ctx["L"]
    from arealmshbm.spatial_priors.spatial_connect import ConnectSession
    rng = np.random.default_rng(0)
    # a non-trivial s_lambda: theta with random per-row perturbation
    sl = ctx["theta"] * rng.uniform(0.5, 1.5, size=ctx["theta"].shape).astype(np.float32)
    conn = ConnectSession(fx.grad, N, L)
    u_ref, scv_ref = conn.compute(sl)
    grad = cp.asarray(fx.grad)
    grad_sq = cp.asarray(conn._grad_sq_norms)
    u = cp.empty((L, fx.grad.shape[1]), cp.float32)
    u_sq = cp.empty(L, cp.float32)
    scv = cp.empty(lay_h.P, cp.float32)
    K.connect_prior(lay, ctx["p_row_m"], cp.asarray(lay_h.gather(sl)), grad, grad_sq, u, u_sq, scv)
    np.testing.assert_allclose(u.get(), u_ref, rtol=1e-5, atol=1e-6)
    ref = lay_h.gather(scv_ref)
    got = scv.get()
    assert np.array_equal(np.isinf(ref), np.isinf(got))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(got[fin], ref[fin], rtol=2e-5, atol=2e-4)
    print(f"connect: max |Δscv| {np.abs(got[fin] - ref[fin]).max():.3e} "
          f"(|scv| ~ {np.abs(ref[fin]).mean():.3e})")


def test_xyz_prior(ctx):
    cp, K, lay, lay_h = ctx["cp"], ctx["K"], ctx["lay"], ctx["lay_h"]
    fx = ctx["fx"]; N, L = ctx["N"], ctx["L"]
    from arealmshbm.spatial_priors.spatial_xyz import XyzSession, compute_unit_sphere_xyz
    sphere_bil = np.concatenate([fx.lh_sphere["vertices"].T, fx.rh_sphere["vertices"].T], axis=0)
    xyz = XyzSession(sphere_bil, L)
    sphere_dev = cp.asarray(compute_unit_sphere_xyz(sphere_bil))
    rng = np.random.default_rng(1)
    sl = ctx["theta"] * rng.uniform(0.5, 1.5, size=ctx["theta"].shape).astype(np.float32)
    lam = cp.asarray(lay_h.gather(sl))
    s_muc = cp.empty((3, L), cp.float32); cdln3 = cp.empty(L, cp.float32)
    g32 = cp.empty(L, cp.float32); sxv = cp.empty(lay_h.P, cp.float32)
    for gamma in (np.zeros(L), rng.choice([0.0, 1000.0, 2000.0], size=L)):
        gamma = np.ascontiguousarray(gamma, dtype=np.float64)
        muc_ref, sxv_ref = xyz.compute(sl, gamma)
        K.xyz_prior(lay, ctx["p_row_m"], lam, sphere_dev, cp.asarray(gamma), s_muc, cdln3, g32, sxv)
        np.testing.assert_allclose(s_muc.get(), muc_ref, rtol=1e-5, atol=1e-6)
        ref = lay_h.gather(sxv_ref); got = sxv.get()
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-3)
        print(f"xyz gamma∈{set(np.unique(gamma))}: max |Δsxv| {np.abs(got - ref).max():.3e}")


def test_em_stop_cost(ctx):
    cp, K, lay, lay_h = ctx["cp"], ctx["K"], ctx["lay"], ctx["lay_h"]
    fx = ctx["fx"]; sp = fx.setting_params; N, L, T = ctx["N"], ctx["L"], ctx["T"]
    from arealmshbm.em_stop_criterion import _kernels as SK
    from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32
    from arealmshbm.spatial_priors.spatial_connect import ConnectSession
    rng = np.random.default_rng(2)
    theta = ctx["theta"]
    sl = theta * rng.uniform(0.5, 1.5, size=theta.shape).astype(np.float32)
    kappa = np.full(L, 900.0)
    cdln = np.empty(L, np.float32); cdln_general_to_f32(kappa, int(sp["dim"]), cdln)
    kappa_f32 = kappa.astype(np.float32)
    acc = np.ascontiguousarray(ctx["acc_ref"], dtype=np.float32)
    ltc = np.empty_like(theta); SK.log_with_neginf_floor(theta, ltc)
    Vt = rng.uniform(0, 3, size=theta.shape).astype(np.float32) * (theta > 0)
    _, scv = ConnectSession(fx.grad, N, L).compute(sl)
    scv = np.ascontiguousarray(scv)
    scv_clean = np.empty_like(scv); out = np.empty(1, np.float64)
    SK.fused_em_stop_assemble_cleanup_cost_f32(
        acc, kappa_f32, cdln, T, sl, ltc, Vt, scv, scv_clean,
        np.float32(sp["w"]), np.float32(sp["c"]), np.asarray(sp["beta"], np.float32), out)
    ws = K.EMStopWorkspace(lay_h.P)
    scv_P = cp.asarray(lay_h.gather(scv))
    cost = K.em_stop_cost(lay, ws, cp.asarray(lay_h.gather(acc)), cp.asarray(kappa_f32),
                          cp.asarray(cdln), T, cp.asarray(lay_h.gather(sl)),
                          cp.asarray(lay_h.gather(ltc)), cp.asarray(lay_h.gather(Vt)), scv_P,
                          sp["w"], sp["c"], cp.asarray(np.asarray(sp["beta"], np.float32)))
    print(f"em_stop cost cpu {out[0]:.15e} gpu {cost:.15e}")
    assert abs(cost - out[0]) <= 1e-9 * abs(out[0])
    assert np.array_equal(scv_P.get(), lay_h.gather(scv_clean))


def test_intra_psi(ctx):
    cp, K = ctx["cp"], ctx["K"]
    fx = ctx["fx"]; sp = fx.setting_params; L, T, D = ctx["L"], ctx["T"], ctx["D"]
    from arealmshbm.intra_em import intra_subject_var, intra_em_cost
    rng = np.random.default_rng(3)
    mu = ctx["mu"]
    s_t_nu_DLT = (mu[..., None] + rng.normal(0, 0.01, size=(D, L, T))).astype(np.float32)
    sigma, epsil = fx.Params["sigma"], fx.Params["epsil"]
    psi_ref = intra_subject_var(s_t_nu_DLT, sigma, epsil, mu)
    cost_ref = intra_em_cost(psi_ref, s_t_nu_DLT, mu, sigma, epsil, 12.5, int(sp["dim"]))
    stnu = cp.asarray(np.ascontiguousarray(np.transpose(s_t_nu_DLT, (2, 1, 0))))
    mu_LD = cp.asarray(np.ascontiguousarray(mu.T))
    psi_LD = cp.empty((L, D), cp.float32)
    pc1 = cp.empty(L, cp.float64); pc2 = cp.empty(L, cp.float64)
    summed_LD = cp.empty((L, D), cp.float32)
    K.intra_psi(stnu, cp.asarray(sigma), cp.asarray(epsil), mu_LD, psi_LD, summed_LD, pc1, pc2)
    psi = psi_LD.get().T
    # Bit-identical only while T < 8 — above that numpy's pairwise sum
    # blocks and the kernel's serial T sum no longer matches it.
    assert T < 8
    assert np.array_equal(psi, psi_ref), "intra_subject_var must be bit-identical"
    from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32
    cs = np.empty(L, np.float32); ce = np.empty(L, np.float32)
    cdln_general_to_f32(sigma.astype(np.float64), int(sp["dim"]), cs)
    cdln_general_to_f32(epsil.astype(np.float64), int(sp["dim"]), ce)
    term1 = float((sigma.astype(np.float64) * pc1.get()).sum()) + T * float(cs.astype(np.float64).sum())
    term2 = float((epsil.astype(np.float64) * pc2.get()).sum()) + float(ce.astype(np.float64).sum())
    cost = term1 + term2 + 12.5
    print(f"intra cost cpu {cost_ref:.15e} gpu {cost:.15e}")
    assert abs(cost - cost_ref) <= 1e-10 * abs(cost_ref)
