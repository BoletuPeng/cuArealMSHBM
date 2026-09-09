"""test_m_step_gpu_sparse.py

Validation of the ``gpu_sparse`` GPU M-step
(:mod:`arealmshbm.m_step.m_step_gpu_sparse`) against the CPU reference
:class:`arealmshbm.m_step.m_step.MStepSession`.

Real-data checks run on sub-001 (fsaverage6, T=6, N=81924, D=1175,
L=300, P=251502) via the shared fixture; they skip when the profile
store or cupy is missing. The synthetic case is self-contained and only
needs cupy.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arealmshbm.m_step.m_step import MStepSession
from arealmshbm.vmf_clustering.tests._sub001_fixture import (
    skip_unless_cupy, skip_unless_sub001,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _csc_from_dense_support(mask_NL: np.ndarray):
    """Hand-built ``(col_ptr, csc_row, csc_pidx)`` for a dense support.

    Mirrors :func:`arealmshbm.vmf_clustering.sparse_layout.build_candidate_layout_dense`:
    the ``(P,)`` CSR order is the row-major nonzero order of the support,
    and the CSC arrays are that order sorted by ``(col, n)``.
    """
    rows, cols = np.nonzero(mask_NL)              # row-major -> CSR order
    L = mask_NL.shape[1]
    order = np.lexsort((rows, cols)).astype(np.int32)
    col_ptr = np.zeros(L + 1, dtype=np.int64)
    np.cumsum(np.bincount(cols, minlength=L), out=col_ptr[1:])
    return (col_ptr.astype(np.int32),
            np.ascontiguousarray(rows[order], dtype=np.int32),
            np.ascontiguousarray(order, dtype=np.int32),
            rows.astype(np.int64), cols.astype(np.int64))


def _gather_P(x_NL, rows, cols):
    return np.ascontiguousarray(x_NL[rows, cols], dtype=np.float32)


# ---------------------------------------------------------------------
# module-scope sub-001 reference (built once: the host BOLD is 2.3 GB)
# ---------------------------------------------------------------------
class _Ref:
    pass


_REF = None


def _sub001_ref():
    """Fixture + host fp32 BOLD + CPU MStepSession + device inputs."""
    global _REF
    if _REF is not None:
        return _REF
    import cupy as cp
    from arealmshbm.data_io.bitpacked_norm import (
        unpack_normalize_packed_NTD_host,
    )
    from arealmshbm.vmf_clustering.sparse_layout import layout_to_device
    from arealmshbm.m_step import m_step_gpu_sparse as G

    fx = skip_unless_sub001()
    r = _Ref()
    r.fx = fx
    r.T, r.N, r.Db = fx.packed_TND.shape
    r.D = int(fx.D)
    r.L = int(fx.layout.L)
    r.dim = float(fx.setting_params["dim"])
    r.eps = 1e-4
    r.max_iter = 50

    # Host reference BOLD (N, T, D) fp32.
    r.ds_NTD = unpack_normalize_packed_NTD_host(
        np.ascontiguousarray(np.transpose(fx.packed_TND, (1, 0, 2))), r.D)
    r.session = MStepSession(r.ds_NTD, r.dim, r.L, r.T,
                             epsilon=r.eps, max_iter=r.max_iter)

    # Inputs.
    r.s_lambda = np.ascontiguousarray(fx.Params["theta"], dtype=np.float32)
    r.s_lambda_P = fx.layout.gather(r.s_lambda)
    r.s_psi = np.ascontiguousarray(fx.Params["mu"], dtype=np.float32)   # (D, L)
    r.sigma = np.ascontiguousarray(
        np.asarray(fx.Params["sigma"]).ravel(), dtype=np.float32)
    r.s_t_nu_DLT = np.ascontiguousarray(
        np.repeat(r.s_psi[:, :, None], r.T, axis=2))
    r.kappa_init = np.full(r.L, float(fx.ini_val), dtype=np.float64)

    # Device state.
    r.packed_dev = cp.asarray(np.ascontiguousarray(fx.packed_TND))
    r.row_mean_dev, r.row_inv_dev = G.compute_row_stats(r.packed_dev, r.D)
    r.layout_dev = layout_to_device(fx.layout)
    r.s_lambda_P_dev = cp.asarray(r.s_lambda_P)
    r.s_psi_LD_dev = cp.asarray(np.ascontiguousarray(r.s_psi.T))     # (L, D)
    r.sigma_dev = cp.asarray(r.sigma)
    r.s_t_nu_TLD_dev = cp.asarray(
        np.ascontiguousarray(np.transpose(r.s_t_nu_DLT, (2, 1, 0))))
    r.mstep = G.MStepGPU(r.T, r.D, r.L, r.dim, r.eps, r.max_iter,
                         r.packed_dev, r.row_mean_dev, r.row_inv_dev,
                         r.layout_dev)
    _REF = r
    return r


# ---------------------------------------------------------------------
# 1. row statistics
# ---------------------------------------------------------------------
def test_row_stats_vs_host():
    skip_unless_cupy()
    import cupy as cp
    r = _sub001_ref()

    packed = r.fx.packed_TND                       # (T, N, Db)
    bits_pop = np.zeros((r.T, r.N), dtype=np.int64)
    for b in range(r.Db):
        col = packed[:, :, b].astype(np.uint32)
        for k in range(8):
            bits_pop += ((col >> k) & 1).astype(np.int64)

    pop_d = bits_pop.astype(np.float64)
    mean_ref = (pop_d / float(r.D)).astype(np.float32)
    post = pop_d - float(r.D) * (pop_d / float(r.D)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ref = (1.0 / np.sqrt(post)).astype(np.float32)
    zero_rows = (bits_pop == 0) | (bits_pop == r.D)
    inv_ref = np.where(zero_rows, np.float32(0.0), inv_ref).astype(np.float32)

    mean_gpu = cp.asnumpy(r.row_mean_dev)
    inv_gpu = cp.asnumpy(r.row_inv_dev)
    assert np.array_equal(mean_gpu, mean_ref), "row_mean must be bit-exact"

    # ULP distance on row_inv.
    a = mean_gpu  # keep flake quiet
    ulp = np.abs(inv_gpu.view(np.int32).astype(np.int64)
                 - inv_ref.view(np.int32).astype(np.int64))
    print(f"\n[row_stats] rows={r.T * r.N} zero/const rows={int(zero_rows.sum())} "
          f"row_inv max ULP diff={int(ulp.max())}")
    assert int(ulp.max()) <= 1

    # Cross-check against the actual host fp32 rows: row_mean is the value
    # subtracted, row_inv the scale applied (checked on a random sample of
    # non-degenerate rows).
    rng = np.random.default_rng(0)
    ok = np.flatnonzero(~zero_rows[0])
    sample = rng.choice(ok, size=64, replace=False)
    for n in sample:
        bits = np.zeros(r.D, dtype=np.float32)
        for d in range(r.D):
            bits[d] = (packed[0, n, d >> 3] >> (d & 7)) & 1
        ref_row = r.ds_NTD[n, 0, :]
        recon = (bits - mean_gpu[0, n]) * inv_gpu[0, n]
        assert np.allclose(recon, ref_row, rtol=0, atol=1e-7), n
    del a


# ---------------------------------------------------------------------
# 2. x_dot_sl_bits vs the CPU sgemm
# ---------------------------------------------------------------------
def test_x_dot_sl_bits_vs_sgemm():
    skip_unless_cupy()
    import cupy as cp
    from arealmshbm.m_step import m_step_gpu_sparse as G
    r = _sub001_ref()

    out = cp.empty((r.T, r.L, r.D), dtype=cp.float32)
    G.x_dot_sl_bits(r.packed_dev, r.row_mean_dev, r.row_inv_dev,
                    r.layout_dev["col_ptr"], r.layout_dev["csc_row"],
                    r.layout_dev["csc_pidx"], r.s_lambda_P_dev, r.D, out)
    got = cp.asnumpy(out)

    max_abs = 0.0
    max_rel = 0.0
    max_scale_rel = 0.0
    for t in range(r.T):
        ref = (r.ds_NTD[:, t, :].T @ r.s_lambda).astype(np.float32)  # (D, L)
        g = got[t].T                                                # (D, L)
        d = np.abs(g.astype(np.float64) - ref.astype(np.float64))
        max_abs = max(max_abs, float(d.max()))
        scale = float(np.abs(ref).max())
        max_scale_rel = max(max_scale_rel, float(d.max()) / scale)
        m = np.abs(ref) > 1e-3
        if m.any():
            max_rel = max(max_rel,
                          float((d[m] / np.abs(ref[m].astype(np.float64))).max()))
    print(f"\n[x_dot_sl_bits] max_abs={max_abs:.3e} "
          f"max_rel(|ref|>1e-3)={max_rel:.3e} "
          f"max_abs/max|ref|={max_scale_rel:.3e}")

    # The fp32 sgemm is NOT ground truth: it accumulates 81924 fp32 terms
    # per output cell, while the bit kernel folds <=64-member fp32 partials
    # into an fp64 running sum. Anchor both against an fp64 matmul at t=0.
    Xt = r.ds_NTD[:, 0, :]
    ref64 = Xt.T.astype(np.float64) @ r.s_lambda.astype(np.float64)
    ref32 = (Xt.T @ r.s_lambda).astype(np.float32)
    e_sgemm = np.abs(ref32.astype(np.float64) - ref64)
    e_gpu = np.abs(got[0].T.astype(np.float64) - ref64)
    print(f"[x_dot_sl_bits] vs fp64 truth (t=0, scale={np.abs(ref64).max():.3f}): "
          f"sgemm max={e_sgemm.max():.3e} rms={np.sqrt((e_sgemm**2).mean()):.3e} | "
          f"gpu max={e_gpu.max():.3e} rms={np.sqrt((e_gpu**2).mean()):.3e}")

    # Contract: agreement at the SCALE of the output (the elementwise
    # |ref|>1e-3 relative metric is dominated by near-zero cells where the
    # fp32 sgemm reference itself carries the larger absolute error).
    assert max_scale_rel <= 1e-6
    assert e_gpu.max() <= e_sgemm.max()
    assert np.sqrt((e_gpu ** 2).mean()) <= np.sqrt((e_sgemm ** 2).mean())


# ---------------------------------------------------------------------
# 3. full run vs MStepSession.run
# ---------------------------------------------------------------------
def test_full_run_vs_cpu():
    skip_unless_cupy()
    import cupy as cp
    r = _sub001_ref()

    st_cpu, kappa_cpu, iter_cpu = r.session.run(
        r.s_t_nu_DLT, r.s_lambda, r.s_psi, r.sigma, r.kappa_init)
    st_gpu_dev, kappa_gpu, iter_gpu = r.mstep.run(
        r.s_t_nu_TLD_dev, r.s_lambda_P_dev, r.s_psi_LD_dev,
        r.sigma_dev, float(r.fx.ini_val))
    st_gpu = np.transpose(cp.asnumpy(st_gpu_dev), (2, 1, 0))   # -> (D, L, T)

    k_cpu = float(kappa_cpu[0])
    rel_k = abs(k_cpu - kappa_gpu) / abs(k_cpu)
    diff = np.abs(st_gpu.astype(np.float64) - st_cpu.astype(np.float64))
    print(f"\n[full run] iter_m cpu={iter_cpu} gpu={iter_gpu}\n"
          f"           kappa cpu={k_cpu!r} gpu={kappa_gpu!r} rel={rel_k:.3e}\n"
          f"           s_t_nu max|diff|={float(diff.max()):.3e} "
          f"nan cpu={int(np.isnan(st_cpu).sum())} "
          f"gpu={int(np.isnan(st_gpu).sum())}")
    assert iter_cpu == iter_gpu
    assert rel_k <= 1e-6
    assert np.array_equal(np.isnan(st_cpu), np.isnan(st_gpu))
    assert float(np.nanmax(diff)) <= 1e-5

    # Convergence flags derived from cos must agree with the CPU latch.
    cos_gpu = cp.asnumpy(r.mstep._cos_TL)
    flags_gpu = cp.asnumpy(r.mstep._flag_acc)
    print(f"           final-iter cos: min={float(np.nanmin(cos_gpu)):.9f} "
          f"max(1-cos)={float(np.nanmax(1.0 - cos_gpu)):.3e} "
          f"flags={flags_gpu.tolist()} cpu_flags="
          f"{r.session._flag_T_acc.tolist()}")
    assert np.array_equal(flags_gpu.astype(np.int8),
                          r.session._flag_T_acc.astype(np.int8))


# ---------------------------------------------------------------------
# 4. determinism
# ---------------------------------------------------------------------
def test_determinism():
    skip_unless_cupy()
    import cupy as cp
    r = _sub001_ref()

    a_dev, ka, ia = r.mstep.run(r.s_t_nu_TLD_dev, r.s_lambda_P_dev,
                                r.s_psi_LD_dev, r.sigma_dev,
                                float(r.fx.ini_val))
    a = a_dev.copy()
    b_dev, kb, ib = r.mstep.run(r.s_t_nu_TLD_dev, r.s_lambda_P_dev,
                                r.s_psi_LD_dev, r.sigma_dev,
                                float(r.fx.ini_val))
    assert (ia, ka) == (ib, kb)
    # bitwise (stronger than array_equal; also pins NaN payloads)
    assert bool(cp.all(a.view(cp.uint32) == b_dev.view(cp.uint32)))
    assert bool(cp.array_equal(a, b_dev))


# ---------------------------------------------------------------------
# 5. synthetic tiny case (empty parcel, pop == 0 and pop == D rows)
# ---------------------------------------------------------------------
def _synthetic(with_empty_parcel: bool):
    import cupy as cp
    from arealmshbm.data_io.bitpacked_norm import (
        unpack_normalize_packed_NTD_host,
    )
    from arealmshbm.m_step import m_step_gpu_sparse as G

    rng = np.random.default_rng(7)
    N, T, D, L = 40, 2, 20, 4
    bits = (rng.random((T, N, D)) < 0.35).astype(np.uint8)
    bits[:, 0, :] = 0        # pop == 0
    bits[:, 1, :] = 1        # pop == D
    packed_TND = np.ascontiguousarray(
        np.packbits(bits, axis=-1, bitorder="little"))

    # Candidate support: a few members per parcel; parcel L-1 optionally empty.
    mask = np.zeros((N, L), dtype=bool)
    for n in range(N):
        ls = rng.choice(L - 1 if with_empty_parcel else L,
                        size=2, replace=False)
        mask[n, ls] = True
    if not with_empty_parcel:
        mask[0, L - 1] = True

    col_ptr, csc_row, csc_pidx, rows, cols = _csc_from_dense_support(mask)

    s_lambda = np.zeros((N, L), dtype=np.float32)
    s_lambda[mask] = rng.random(int(mask.sum())).astype(np.float32) + 0.1
    s_lambda_P = _gather_P(s_lambda, rows, cols)

    s_psi = rng.standard_normal((D, L)).astype(np.float32)
    s_psi /= np.linalg.norm(s_psi, axis=0, keepdims=True)
    if with_empty_parcel:
        s_psi[:, L - 1] = 0.0                       # -> cn == 0 -> NaN
    sigma = (rng.random(L).astype(np.float32) + 0.5)
    s_t_nu_DLT = np.ascontiguousarray(np.repeat(s_psi[:, :, None], T, axis=2))

    ds_NTD = unpack_normalize_packed_NTD_host(
        np.ascontiguousarray(np.transpose(packed_TND, (1, 0, 2))), D)
    dim, eps, max_iter = float(D - 1), 1e-4, 12
    ses = MStepSession(ds_NTD, dim, L, T, epsilon=eps, max_iter=max_iter)
    kappa_init = np.full(L, 12.0, dtype=np.float64)
    st_cpu, kappa_cpu, iter_cpu = ses.run(
        s_t_nu_DLT, s_lambda, s_psi, sigma, kappa_init)

    packed_dev = cp.asarray(packed_TND)
    rm, ri = G.compute_row_stats(packed_dev, D)
    layout_dev = {
        "col_ptr": cp.asarray(col_ptr), "csc_row": cp.asarray(csc_row),
        "csc_pidx": cp.asarray(csc_pidx), "P": int(csc_row.size),
    }
    ms = G.MStepGPU(T, D, L, dim, eps, max_iter, packed_dev, rm, ri,
                    layout_dev)
    st_dev, kappa_gpu, iter_gpu = ms.run(
        cp.asarray(np.ascontiguousarray(np.transpose(s_t_nu_DLT, (2, 1, 0)))),
        cp.asarray(s_lambda_P),
        cp.asarray(np.ascontiguousarray(s_psi.T)),
        cp.asarray(sigma), 12.0)
    st_gpu = np.transpose(cp.asnumpy(st_dev), (2, 1, 0))
    return (st_cpu, float(kappa_cpu[0]), iter_cpu,
            st_gpu, float(kappa_gpu), iter_gpu)


def test_synthetic_with_empty_parcel():
    skip_unless_cupy()
    st_cpu, k_cpu, i_cpu, st_gpu, k_gpu, i_gpu = _synthetic(True)
    print(f"\n[synthetic empty] iter cpu={i_cpu} gpu={i_gpu} "
          f"kappa cpu={k_cpu!r} gpu={k_gpu!r} "
          f"nan cpu={int(np.isnan(st_cpu).sum())} "
          f"gpu={int(np.isnan(st_gpu).sum())}")
    assert i_cpu == i_gpu
    assert np.array_equal(np.isnan(st_cpu), np.isnan(st_gpu))
    assert np.isnan(st_cpu).any(), "empty parcel must produce NaN"
    assert (np.isnan(k_cpu) and np.isnan(k_gpu)) or \
        abs(k_cpu - k_gpu) / abs(k_cpu) <= 1e-6


def test_synthetic_no_empty_parcel():
    skip_unless_cupy()
    st_cpu, k_cpu, i_cpu, st_gpu, k_gpu, i_gpu = _synthetic(False)
    d = np.abs(st_gpu.astype(np.float64) - st_cpu.astype(np.float64))
    print(f"\n[synthetic dense] iter cpu={i_cpu} gpu={i_gpu} "
          f"kappa rel={abs(k_cpu - k_gpu) / abs(k_cpu):.3e} "
          f"max|diff|={float(d.max()):.3e}")
    assert i_cpu == i_gpu
    assert not np.isnan(st_cpu).any() and not np.isnan(st_gpu).any()
    assert abs(k_cpu - k_gpu) / abs(k_cpu) <= 1e-6
    assert float(d.max()) <= 1e-5


# ---------------------------------------------------------------------
# 6. timing
# ---------------------------------------------------------------------
def test_timing_report():
    skip_unless_cupy()
    import cupy as cp
    from arealmshbm.m_step import m_step_gpu_sparse as G
    r = _sub001_ref()

    out = cp.empty((r.T, r.L, r.D), dtype=cp.float32)

    def _xdot():
        G.x_dot_sl_bits(r.packed_dev, r.row_mean_dev, r.row_inv_dev,
                        r.layout_dev["col_ptr"], r.layout_dev["csc_row"],
                        r.layout_dev["csc_pidx"], r.s_lambda_P_dev, r.D, out)

    for _ in range(3):
        _xdot()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        _xdot()
    cp.cuda.runtime.deviceSynchronize()
    t_xdot = (time.perf_counter() - t0) / 5 * 1e3

    args = (r.s_t_nu_TLD_dev, r.s_lambda_P_dev, r.s_psi_LD_dev,
            r.sigma_dev, float(r.fx.ini_val))
    _, _, iter_m = r.mstep.run(*args)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        r.mstep.run(*args)
    cp.cuda.runtime.deviceSynchronize()
    t_run = (time.perf_counter() - t0) / 5 * 1e3

    # CPU reference wall for context.
    t0 = time.perf_counter()
    r.session.run(r.s_t_nu_DLT, r.s_lambda, r.s_psi, r.sigma, r.kappa_init)
    t_cpu = (time.perf_counter() - t0) * 1e3

    print(f"\n[timing sub-001] x_dot_sl_bits={t_xdot:.3f} ms | "
          f"full run={t_run:.3f} ms (iter_m={iter_m}, "
          f"{(t_run - t_xdot) / max(iter_m, 1):.3f} ms/iter_m) | "
          f"CPU MStepSession.run={t_cpu:.1f} ms")


if __name__ == "__main__":   # pragma: no cover
    pytest.main([__file__, "-q", "-s"])
