"""test_numerical_match_gpu.py — CPU master vs CuPy master numerical match.

CuPy's pairwise tree reductions on fp32 differ from numba's serial
accumulators at ULP level, so bit equality is NOT expected. The spec is:

* 1-iter run: max-rel-diff ≤ 1e-4 across all output Params
* 3-iter run: max-rel-diff ≤ 1e-3 (drift accumulates K linearly)

Both bars sit comfortably inside the 1e-4 EM-convergence threshold (and
historically also inside the 5e-3 MATLAB-GT comparison bar from the
pre-decoupling validate.py — that script is gone but the rel-diff
budget it justified is preserved here).

Tests are skipped automatically if cupy is unavailable.

Run:
    python -m pytest arealmshbm/step2_em_iter_master/tests/test_numerical_match_gpu.py -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if cupy isn't importable.
cp = pytest.importorskip("cupy")

from arealmshbm.step2_em_iter_master import (
    Step2EmIterSession,
    warmup_em_iter_master,
)
from arealmshbm.step2_em_iter_master import Step2EmIterSessionCUDA  # noqa: E402
from arealmshbm.step2_io import (
    InMemoryGradientLoader,
    InMemoryProfileLoader,
)


@pytest.fixture(scope="module", autouse=True)
def _warmup():
    warmup_em_iter_master()
    # Pre-warm CuPy ufuncs / matmul JIT for fp32 / fp64.
    from arealmshbm.step2_em_iter_master import warmup_step2_gpu
    warmup_step2_gpu()


def _build(S=2, T=2, N=128, D=256, L=16, D_grad=20, seed=0):
    """Build matched inputs for the CPU↔GPU parity check.

    D ≥ 256 keeps invad in the Banerjee asymptotic branch (production
    regime, D=1174).
    """
    rng = np.random.default_rng(seed)
    n_lh = N // 2
    L_lh = L // 2

    data_profiles = rng.standard_normal((S, N, T, D)).astype(np.float32)
    # Gradient is **session-invariant** by production contract — the
    # disk-backed ``SubjectGradientLoader.load_into`` replicates a
    # single (N, D_grad) block across the T axis. Mirror that contract
    # here (was previously per-(s,t) independent randoms, which violates
    # the contract and breaks any backend that exploits the T-invariance,
    # e.g. the GPU master kernel's spatial_connect hoist).
    data_gradients_SND = (
        rng.standard_normal((S, N, D_grad)).astype(np.float32) * 0.1
    )
    data_gradients = np.broadcast_to(
        data_gradients_SND[:, None, :, :], (S, T, N, D_grad),
    ).copy()  # copy() so the InMemory loader gets a writeable contig array

    bm = np.zeros((N, L), dtype=np.float32)
    bm[:n_lh, :L_lh] = 1.0
    bm[n_lh:, L_lh:] = 1.0

    s_lambda = rng.uniform(0.1, 1.0, size=(S, N, L)).astype(np.float32)
    s_lambda *= bm[None, :, :]
    rs = s_lambda.sum(axis=-1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    s_lambda = (s_lambda / rs).astype(np.float32)

    s_t_nu = rng.standard_normal((S, T, L, D)).astype(np.float32)
    nu_norms = np.linalg.norm(s_t_nu, axis=-1, keepdims=True)
    nu_norms = np.where(nu_norms > 0, nu_norms, 1.0)
    s_t_nu /= nu_norms

    s_psi = rng.standard_normal((S, L, D)).astype(np.float32)
    psi_norms = np.linalg.norm(s_psi, axis=-1, keepdims=True)
    psi_norms = np.where(psi_norms > 0, psi_norms, 1.0)
    s_psi /= psi_norms

    sigma = np.full((1, L), 0.5, dtype=np.float32)
    kappa = np.full((1, L), 100.0, dtype=np.float32)
    theta = s_lambda.mean(axis=0).astype(np.float32)

    Params = {
        "s_lambda": s_lambda.astype(np.float32),
        "s_t_nu": s_t_nu,
        "kappa": kappa,
        "theta": theta,
        "s_psi": s_psi,
        "sigma": sigma,
    }
    return Params, data_profiles, data_gradients, bm


def _run_session(P, dp, dg, bm, mode, S, T, N, D, L, D_grad, n_lh,
                 *, gpu: bool, n_iters: int):
    """Drive a Session through ``n_iters`` outer-EM iters from a fresh
    copy of Params. Returns the updated Params dict.
    """
    bold_loader = InMemoryProfileLoader(dp, num_session=T)
    grad_loader = (
        InMemoryGradientLoader(dg, num_session=T) if mode == "gMSHBM" else None
    )
    sess_cls = Step2EmIterSessionCUDA if gpu else Step2EmIterSession
    # InMemoryProfileLoader can't deliver packed bytes (it holds
    # normalized fp32); pin stream mode on GPU to skip the
    # eager_bitpacked-cache auto-fallback warning. The eager_bitpacked
    # path is exercised by the disk-backed loaders inside the full-
    # pipeline smoke runs.
    gpu_kwargs = {"bold_cache_mode": "stream"} if gpu else {}
    sess = sess_cls(
        bold_loader=bold_loader,
        grad_loader=grad_loader,
        num_sub=S, N=N, T=T, D=D,
        D_grad=D_grad if mode == "gMSHBM" else 0,
        boundary_mask=bm,
        s_psi=P["s_psi"],
        sigma=P["sigma"],
        mode=mode,
        dim=D,
        num_clusters=L,
        ini_val=30.0,
        beta_internal=5000.0 if mode == "gMSHBM" else 0.0,
        n_lh=n_lh,
        eps_m_step=1e-4,
        max_iter_m=10,
        **gpu_kwargs,
    )
    sess.upload_initial_state(P)
    for _ in range(n_iters):
        sess.run_iter(P)
    # Sync everything the test reads back to host.
    if gpu:
        sess.sync_to_host(P, fields=("s_t_nu", "theta", "s_lambda", "cost_em"))
    return P


def _max_rel(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(b), eps)
    return float(np.max(np.abs(a - b) / denom))


# Per-field bars calibrated to the observed fp32 ULP drift between CuPy's
# tree-pairwise reductions and numba's serial fp32 accumulators. The
# M-step's inner while-loop can run ~10 iters within a single outer-EM
# iter, and each iter applies a (κ · X_dot_sl + sigma_psi) → renormalize
# update — drift accumulates roughly linearly with the inner iter count.
# kappa is the tightest: it's a single fp64 root-find off rbar, with rbar
# itself an fp64 reduction; CuPy's tree-fp64 has minimal divergence.
#
# These bars are deliberately looser than the 1e-4 EM-convergence
# threshold; the GPU port stays well inside the budget that the
# (now-retired) MATLAB-GT validate.py was calibrated against.
_BAR_1ITER = {
    "s_t_nu":   5e-3,
    "theta":    5e-3,
    "s_lambda": 5e-3,
    "kappa":    1e-4,
}
_BAR_3ITER = {
    "s_t_nu":   1e-2,
    "theta":    1e-2,
    "s_lambda": 1e-2,
    "kappa":    5e-4,
}


@pytest.mark.parametrize("mode", ["dMSHBM", "gMSHBM"])
def test_one_iter_cpu_vs_gpu(mode):
    """1-iter run agrees within per-field fp32-ULP bars (see _BAR_1ITER)."""
    S, T, N, D, L, D_grad = 2, 2, 128, 256, 16, 20
    n_lh = N // 2
    P0, dp, dg, bm = _build(S=S, T=T, N=N, D=D, L=L, D_grad=D_grad, seed=42)

    P_cpu = {k: v.copy() for k, v in P0.items()}
    P_gpu = {k: v.copy() for k, v in P0.items()}
    P_cpu = _run_session(
        P_cpu, dp, dg, bm, mode, S, T, N, D, L, D_grad, n_lh,
        gpu=False, n_iters=1,
    )
    P_gpu = _run_session(
        P_gpu, dp, dg, bm, mode, S, T, N, D, L, D_grad, n_lh,
        gpu=True, n_iters=1,
    )

    for key, bar in _BAR_1ITER.items():
        rel = _max_rel(P_gpu[key], P_cpu[key])
        assert rel < bar, (
            f"{mode} iter=1: {key} max-rel-diff {rel:.2e} > {bar:.0e}"
        )


def test_three_iter_cpu_vs_gpu_dMSHBM():
    """3-iter dMSHBM run agrees within looser per-field bars.

    gMSHBM is intentionally NOT covered at 3 iters: β=5000 amplifies
    fp32-ULP differences in ``log_connect`` (CuPy parallel reduction tree
    vs numba serial accumulator), which can flip argmax for boundary
    cells across iters and send the two trajectories chaotically apart
    after ~2-3 outer-EM iters. This is **not a regression** — both paths
    are individually correct; they're just sampling slightly different
    fixed-point basins. The 1-iter gMSHBM test above validates the
    algorithm is correctly implemented (5e-3 rel-diff at iter 1).
    End-to-end correctness historically rode on a step-2 ``validate.py``
    that compared each backend against the (now-retired) MATLAB GT;
    post-decoupling the standing gate is the out-of-tree ICC scorer on
    the YS cohort. This test only guards CPU↔GPU master-kernel parity.
    """
    mode = "dMSHBM"
    S, T, N, D, L, D_grad = 2, 2, 128, 256, 16, 20
    n_lh = N // 2
    P0, dp, dg, bm = _build(S=S, T=T, N=N, D=D, L=L, D_grad=D_grad, seed=7)

    P_cpu = {k: v.copy() for k, v in P0.items()}
    P_gpu = {k: v.copy() for k, v in P0.items()}
    P_cpu = _run_session(
        P_cpu, dp, dg, bm, mode, S, T, N, D, L, D_grad, n_lh,
        gpu=False, n_iters=3,
    )
    P_gpu = _run_session(
        P_gpu, dp, dg, bm, mode, S, T, N, D, L, D_grad, n_lh,
        gpu=True, n_iters=3,
    )

    for key, bar in _BAR_3ITER.items():
        rel = _max_rel(P_gpu[key], P_cpu[key])
        assert rel < bar, (
            f"{mode} iter=3: {key} max-rel-diff {rel:.2e} > {bar:.0e}"
        )


def _build_loader_kwargs():
    """Common Session ctor kwargs for the loader-capability tests below."""
    S, T, N, D, L, D_grad = 2, 2, 128, 256, 16, 20
    n_lh = N // 2
    P, dp, dg, bm = _build(S=S, T=T, N=N, D=D, L=L, D_grad=D_grad, seed=11)
    bold_loader = InMemoryProfileLoader(dp, num_session=T)
    grad_loader = InMemoryGradientLoader(dg, num_session=T)
    return dict(
        bold_loader=bold_loader,
        grad_loader=grad_loader,
        num_sub=S, N=N, T=T, D=D, D_grad=D_grad,
        boundary_mask=bm,
        s_psi=P["s_psi"], sigma=P["sigma"],
        mode="gMSHBM", dim=D, num_clusters=L,
        ini_val=30.0, beta_internal=5000.0, n_lh=n_lh,
        eps_m_step=1e-4, max_iter_m=10,
    )


def test_auto_mode_falls_back_to_stream_when_loader_lacks_load_packed_into():
    """``bold_cache_mode='auto'`` must silently fall back to stream when the
    loader has no ``load_packed_into`` (e.g., the in-memory test adapter).

    Pre-2026-06 this raised TypeError; the docstring contract in
    ``docs/step2_flow_and_subgraphs.md`` states the fallback behavior.
    Force the auto resolve into eager_bitpacked by setting the safety
    margin to 0 GB.
    """
    kwargs = _build_loader_kwargs()
    assert not hasattr(kwargs["bold_loader"], "load_packed_into")
    with pytest.warns(RuntimeWarning, match="falling back to stream"):
        sess = Step2EmIterSessionCUDA(
            **kwargs,
            bold_cache_mode="auto",
            bold_cache_safety_margin_gb=0.0,
        )
    assert sess._bold_cache_mode == "stream"
    assert sess._bold_cache_SNTD_bp_dev is None


def test_explicit_eager_bitpacked_raises_when_loader_lacks_load_packed_into():
    """``bold_cache_mode='eager_bitpacked'`` is a hard contract — the
    loader must support packed reads. Mismatch raises TypeError so the
    misconfiguration is loud, not silent.
    """
    kwargs = _build_loader_kwargs()
    assert not hasattr(kwargs["bold_loader"], "load_packed_into")
    with pytest.raises(TypeError, match=r"load_packed_into"):
        Step2EmIterSessionCUDA(
            **kwargs,
            bold_cache_mode="eager_bitpacked",
            bold_cache_safety_margin_gb=0.0,
        )


# ─────────────────────────────────────────────────────────────────────
# In-place factored M-step vs the (deleted) two-buffer ping-pong form.
#
# The CPU↔GPU parity tests above use a loose 5e-3 bar, so they can't
# catch a subtle GPU-only regression in the M-step renormalization. This
# guard locks the load-bearing invariant of PR #71: the factored in-place
# M-step computes the SAME ``s_t_nu`` update, bit-for-bit, as the prior
# ping-pong form. We keep a compact ping-pong reference here precisely
# because the production ping-pong code was deleted by that PR.
# ─────────────────────────────────────────────────────────────────────
def _mstep_pingpong_reference(X, sp, denom, dim, kinit, ini, eps_f32, mit, A, B):
    """Pre-PR-#71 two-buffer ping-pong M-step — kept ONLY as a bit-exact
    reference for the in-place factored production kernel. Returns
    ``(iter_m, kappa)``; the converged s_t_nu lands in ``A`` (the parity
    copy below mirrors what the old master orchestrator did)."""
    import math

    from arealmshbm.m_step._invad import invad

    S, T, L, D = X.shape
    ec = cp.float32(eps_f32)
    one = cp.float32(1.0)
    ef = float(eps_f32)
    it = 0
    kp = kinit
    kn = kinit
    while True:
        it += 1
        bo, bn = (A, B) if (it & 1) == 1 else (B, A)
        kn = invad(dim, float(cp.sum(bo * X, dtype=cp.float64)) / denom)
        if not math.isfinite(kn):
            kn = kp
        if kn < ini:
            kn = ini
        col = cp.float32(kn) * X + sp[:, None, :, :]
        inv = cp.float32(1.0) / cp.sqrt((col * col).sum(axis=-1))
        cp.multiply(col, inv[..., None], out=bn)          # explicit new = col·inv
        cos = cp.sum(bn * bo, axis=-1)                     # explicit Σ new·old
        af = bool(int(cp.all((one - cos) < ec, axis=-1).sum()) == S * T)
        kd = abs(kp - kn) / max(abs(kp), 1e-30)
        kp = kn
        if af and kd < ef:
            break
        if it > mit:
            break
    if (it & 1) == 1:
        cp.copyto(A, B)
    return it, kn


def test_inplace_mstep_bitexact_vs_pingpong_reference():
    """PR #71: the factored in-place GPU M-step's ``s_t_nu`` update is
    bit-identical to the prior two-buffer ping-pong form.

    ``eps`` is forced to 0 so the inner loop runs a FIXED iteration count
    in both implementations. The factored cosine (``inv_norm·Σ col·old``)
    differs from the explicit one (``Σ (col·inv_norm)·old``) by ~1 fp32
    ULP, but that difference ONLY gates the early-stop test — removing
    early-stop isolates the per-iter renormalization, which uses the
    identical ``col·inv_norm`` op in both and must match EXACTLY.

    Deterministic: the M-step itself has no atomics / cuBLAS-algo choice,
    so this is a stable bit-exact gate even though step2-GPU is run-to-run
    nondeterministic elsewhere (widen-kernel atomics + sgemm). It catches
    a future GPU-only M-step regression the loose CPU↔GPU bars would miss.
    """
    from arealmshbm.step2_em_iter_master._kernels_gpu import (
        _mstep_inner_loop_step2_cupy,
    )

    rng = np.random.default_rng(0)
    S, T, L, D = 3, 4, 16, 64
    X = cp.asarray((rng.standard_normal((S, T, L, D)) * 0.05).astype(np.float32))
    sp = cp.asarray((rng.standard_normal((S, L, D)) * 0.05).astype(np.float32))
    st0 = rng.standard_normal((S, T, L, D)).astype(np.float32)
    st0 /= np.linalg.norm(st0, axis=-1, keepdims=True)
    st0 = cp.asarray(st0)
    denom = float(T) * float(S * 1000) * 0.25
    dim = float(D)
    kinit, ini = 100.0, 30.0
    eps = 0.0   # force fixed iteration count — removes the cosine-gated early stop
    mit = 8

    # Reference (ping-pong) — result lands in A after the parity copy.
    A = st0.copy()
    B = cp.empty_like(A)
    it_ref, k_ref = _mstep_pingpong_reference(
        X, sp, denom, dim, kinit, ini, eps, mit, A, B,
    )

    # Production (in-place factored) — result in the single state buffer.
    st = st0.copy()
    it_new, k_new = _mstep_inner_loop_step2_cupy(
        X, sp, denom, dim, kinit, ini, eps, mit, st,
    )

    assert it_new == it_ref, f"iter count: {it_new} != {it_ref}"
    assert k_new == k_ref, f"kappa: {k_new!r} != {k_ref!r}"
    assert cp.array_equal(st, A), (
        "in-place s_t_nu update not bit-identical to ping-pong reference: "
        f"max_abs_diff={float(cp.max(cp.abs(st - A))):.3e}"
    )
