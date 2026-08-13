"""_kernels.py — streaming EM-iter master + sub-kernels.

The master :func:`em_iter_master_kernel_streaming` is a Python
orchestrator: Phase B (M-step) is one @njit call, and the two outer
S-loops (Phase A.3 and Phase C+D+E.1) iterate in Python, decoding one
subject's BOLD/grad into reused scratch slots between phases via
loader callbacks. Peak BOLD/grad RAM stays at one subject's slab
regardless of cohort size S.

Storage is fp32 throughout; reduction-accumulator dtypes are mixed
fp32/fp64 per the per-site ablation findings below.

Per-call layout (caller responsibilities):

* bold scratch    : (N, T, D)  fp32 C-contig — re-used per subject;
                    loader fills it before each per-s phase
* grad scratch    : (T, N, D_grad) fp32 C-contig (gMSHBM only)
* s_t_nu_STLD     : (S, T, L, D) fp32 C-contig (M-step in/out)
* X_dot_sl_STLD   : (S, T, L, D) fp32 C-contig (Phase A.3 → Phase B)
* sigma_psi_SLD   : (S, L, D)    fp32 C-contig
* s_lambda_SNL    : (S, N, L)    fp32 C-contig (storage)
* s_lambda_NL_f64_scratch : (N, L) fp64 C-contig — single-subject
                    softmax intermediate (Phase D → Phase E.1)
* theta_NL        : (N, L)       fp32 C-contig
* boundary_mask_NL: (N, L)       fp32 C-contig
* log_connect_NL_buf : (N, L)    fp32 C-contig — per-subject scratch (gMSHBM)
* tmp_idx_NL_buf  : (N, L)       bool — per-subject scratch

BOLD NTD layout note
--------------------
BOLD is consumed in (N, T, D) NTD per subject — matches step0/step1/
step3 across the fork. Phase A.3 and Phase D both fuse the per-t inner
loop into ONE big sgemm per subject that reshapes BOLD: (N, T, D) →
(N, T·D) zero-copy. Numba's ``np.dot`` does NOT dispatch arbitrary 2-D
strided slices through cblas_sgemm — it falls through to a generic
loop ~50-100× slower than BLAS at this scale. The "big sgemm per
subject" pattern sidesteps that pitfall. Phase C consumes per-t
``grad[t]`` C-contig slices (TND-shaped); gradients keep TND.

Reduction-accumulator dtype: mixed, per the per-site policy in
``docs/step2_em_iter_master_kernel.md §2``:

* **fp32 acc**: Phase B denom / col-norm / cosine; Phase D log_vmf
  composition + per-subject cost; Phase E.2 theta-mean.
* **fp64 acc (MUST stay fp64)**: Phase B kappa_sum; Phase D softmax
  exp / row_max subtract (β=5000 subnormal); Phase E.1 row-normalize
  ``rs`` (catastrophic on fp32 — see kernel comment).

The CPU wall is identical either way (memory-bound), but the dtype
map informs the GPU port: on a 1:64 fp64:fp32 device, every fp32-safe
site saves a 64× factor.

Master orchestration runs at Python level (no @njit on the master
itself); BLAS sgemm already saturates the CPU. Inner sub-kernels that
don't call BLAS use ``prange`` where the work is embarrassingly
parallel (per-vertex softmax, per-parcel normalize).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numba import njit, prange

from arealmshbm.em_stop_criterion._cdln import _cdln_single
from arealmshbm.m_step._invad import invad_numba


# log(eps_f64^20) ≈ -720.4391 — softmax / cost floor (matches the fused
# E-step kernel's invariant).
_LOG_EPS20 = float(math.log(np.finfo(np.float64).eps ** 20))


# ─────────────────────────────────────────────────────────────────────
# invad — full MATLAB-equivalent algorithm.
#
# Replaced the asymptotic-only formula with the full Bessel-probe +
# secant-polish algorithm at ``arealmshbm.m_step._invad.invad_numba``
# — bit-equivalent to MATLAB ``CBIG_ArealMSHBM_invAd`` (verified to fp64
# ULP on rbar ∈ [0.05, 0.9], D=1175 via mpmath). The previous
# ``_invad_asymptotic_f64`` only matched MATLAB when Bessel overflowed
# (κ > ~880 at D=1175); in early EM iters (κ ~ 553) MATLAB used fzero
# and Python used asymptotic-only — a ~0.5 absolute κ discrepancy that
# compounded over iters.
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Phase A — Precompute X_dot_sl[s, t] = s_lambda[s].T @ X[s, t].
#
# This is the hoisted M-step trick: ``X.T @ s_lambda`` is
# independent of s_t_nu, so we hoist it out of the iter_m inner loop.
# The MATLAB driver recomputes it every iter_m (~2078 GFlops per outer
# EM iter at production scale); we compute it ONCE per outer EM iter.
#
# Output layout (T, L, D) per subject — unit-stride d for the downstream
# M-step inner loop. BOLD is now (S, N, T, D) (NTD layout); the per-(s,t)
# slice would be N-strided, so we fold S·T smaller sgemms into ONE big
# sgemm per subject + a small transpose.
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False, inline='always')
def _compute_X_dot_sl_s_NTD(
    X_NTD_s,           # (N, T, D) fp32 C-contig — BOLD for one subject
    s_lambda_NL_s,     # (N, L)    fp32 C-contig — s_lambda for one subject
    out_TLD_s,         # (T, L, D) fp32 C-contig — X_dot_sl[s] target
):
    """Per-subject big sgemm: ``out[t, l, d] = Σ_n s_lambda[n, l] * X[n, t, d]``.

    Implemented as ONE big sgemm ``(L, N) @ (N, T·D) → (L, T·D)``,
    then transposed to ``(T, L, D)`` into the caller-owned output.

    Equivalent FLOPs to the legacy ``T`` smaller ``(L, N) × (N, D)``
    sgemms (the (S, T, N, D) TND path), but a single BLAS dispatch
    instead of T → marginally better cache reuse of the (L, N) left
    factor, and — critically — works directly on the (S, N, T, D) NTD
    layout without paying for a per-(s, t) C-contig staging copy
    (numba's np.dot does not dispatch strided 2-D slices through
    cblas_sgemm — it falls through to a manual generic loop, ~50-100×
    slower than BLAS at this scale).
    """
    N, T, D = X_NTD_s.shape
    L = out_TLD_s.shape[1]
    # BOLD[s] (N, T, D) C-contig reshape → (N, T*D) C-contig (zero-copy).
    bold_NX_view = X_NTD_s.reshape(N, T * D)
    # s_lambda[s].T is (L, N) F-contig view of (N, L) C-contig; numba
    # dispatches the .T view to cblas_sgemm with the transA flag.
    # Output: (L, T*D) fp32 C-contig, allocated by numba's runtime.
    tmp_LX = np.dot(s_lambda_NL_s.T, bold_NX_view)  # (L, T*D)
    # Reshape (L, T*D) → (L, T, D) (zero-copy) and transpose into the
    # caller's (T, L, D) output. The transpose loop is ~T·L·D ≈ 940 K
    # fp32 writes per subject; auto-vectorized by LLVM at this stride.
    tmp_LTD = tmp_LX.reshape(L, T, D)
    for t in range(T):
        for l in range(L):
            for d in range(D):
                out_TLD_s[t, l, d] = tmp_LTD[l, t, d]


# ─────────────────────────────────────────────────────────────────────
# Phase A — sigma_psi precompute.
# sigma_psi[s, l, d] = sigma[l] * s_psi[s, l, d].
# Done ONCE per outer EM iter (sigma and s_psi don't change within
# vmf_clustering_batch — they're updated by intra/inter_subject_var
# OUTSIDE the EM-iter loop).
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _sigma_psi_SLD_compute(sigma_L, s_psi_SLD, sigma_psi_SLD):
    """``sigma_psi[s, l, d] = sigma[l] * s_psi[s, l, d]``.

    Pure data-parallel write (no reduction → bit-identical regardless
    of thread count). prange over flattened (s, l) for balance.
    """
    S, L, D = s_psi_SLD.shape
    SL = S * L
    for sl in prange(SL):
        s = sl // L
        l = sl - s * L
        sig = sigma_L[l]
        row_p = s_psi_SLD[s, l]
        row_o = sigma_psi_SLD[s, l]
        for d in range(D):
            row_o[d] = sig * row_p[d]


# ─────────────────────────────────────────────────────────────────────
# Phase B — Multi-subject M-step inner loop.
#
# Adapted from the single-subject (T, L, D) prototype
# ``mstep_inner_loop_master_TLD`` to multi-subject (S, T, L, D). Differences:
#   * kappa_sum reduces over (S, T, L, D) (was (T, L, D)).
#   * denominator: T * Σ_{s, n, l} s_lambda — global sum across S.
#   * s_t_nu update: per (S, T, L) — adds an outer S loop.
#   * Convergence flag: per (S, T) — flag_ST_acc[s, t] holds the
#     "all-L-converged" bit for the most recent iter. It is overwritten
#     each iter (not accumulated); only the final iter's flag matters
#     because the outer loop terminates when it shows all-converged.
#     All converged iff ``Σ flag_ST_acc == S * T``.
#   * kappa is still uniform-scalar (MATLAB always materializes
#     ``invAd`` once per iter_m and broadcasts).
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _kappa_sum_reduce_STLD(s_t_nu_STLD, X_dot_sl_STLD):
    """``Σ_{s, t, l, d} s_t_nu[s, t, l, d] * X_dot_sl[s, t, l, d]``.

    fp64 accumulator. (S, T, L, D) C-contig → unit-stride along d.
    Parallel reduction over (s, t, l). Tried ``fastmath=True`` for
    AVX2/FMA emission — no measurable gain at production scale (the
    inner D=1175 loop is already memory-bandwidth-bound). Kept
    fastmath=False to preserve strict associativity.
    """
    S, T, L, D = s_t_nu_STLD.shape
    # Flatten (S, T, L) to a single prange dimension for better balance.
    STL = S * T * L
    acc = 0.0
    for stl in prange(STL):
        s = stl // (T * L)
        rem = stl - s * (T * L)
        t = rem // L
        l = rem - t * L
        row_n = s_t_nu_STLD[s, t, l]
        row_x = X_dot_sl_STLD[s, t, l]
        local = 0.0
        for d in range(D):
            local += row_n[d] * row_x[d]
        acc += local
    return acc


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', inline='always')
def _denom_global_compute(num_session_f64, s_lambda_SNL):
    """``num_session * Σ_{s, n, l} s_lambda[s, n, l]`` (fp64 acc).

    Matches MATLAB ``num_session * sum(sum(sum(Params.s_lambda)))``.

    fp64 accumulator (required at production scale). Each row of
    ``s_lambda`` sums to 1, so the total grows like ``S * N``. For
    production ``S=200`` on ``fsa6`` (``N ≈ 65k``) the sum crosses
    ``2^23`` and the fp32 ULP exceeds the per-element magnitude
    (``~1/L``), stagnating the running sum well short of the true
    value; the biased denom feeds ``rbar = kappa_sum / denom`` and
    directly corrupts ``κ``. Mode A's same reduction in
    ``arealmshbm/m_step/_kernels.py`` and the GPU port's
    ``s_lambda.sum(dtype=cp.float64)`` already accumulate in fp64;
    this is just bringing the CPU master into line with them.

    Implicit promotion: ``acc = 0.0`` is fp64; ``acc += s_lambda[s, n, l]``
    promotes the fp32 operand to fp64 and accumulates in fp64.
    """
    S, N, L = s_lambda_SNL.shape
    acc = 0.0
    for s in range(S):
        for n in range(N):
            row = s_lambda_SNL[s, n]
            for l in range(L):
                acc += row[l]
    return num_session_f64 * acc


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _mstep_fused_update_STLD(
    kappa_f32,
    X_dot_sl_STLD,        # (S, T, L, D) fp32
    sigma_psi_SLD,        # (S, L, D)    fp32
    s_t_nu_old_STLD,      # (S, T, L, D) fp32 — read
    s_t_nu_new_STLD,      # (S, T, L, D) fp32 — write
    eps_f32,
    flag_ST_acc,          # (S, T) int8 per-iter flag (overwritten 0 or 1)
    converge_STL_scratch, # (S, T, L) int8 per-(s,t,l) "converged" tally
):
    """Per-(s, t, l) column update + cosine drift + per-(s, t) flag.

    Parallel over flattened (s, t, l) — each work unit independently
    processes one column. The per-(s,t) convergence flag is computed
    by a small per-(s,t) reduction across L *after* the main parallel
    pass, using a scratch tally to avoid prange-reduction races.

    For each (s, t, l):
        col[d]        = kappa·X_dot_sl[s, t, l, d] + sigma_psi[s, l, d]
        col_norm      = sqrt(Σ col[d]²)            (fp32 acc — M3 ablation-verified safe)
        new[s,t,l,d]  = col[d] / col_norm
        cos           = Σ new · old                (fp32 acc — M4 ablation-verified safe;
                                                    bit-identical at production scale)
    After each (s, t), set ``flag_ST_acc[s, t] = 1`` iff every l has
    ``(1 - cos) < eps``, else ``flag_ST_acc[s, t] = 0``.

    Note: col-norm and cosine accumulators are fp32 per the per-site
    policy in ``docs/step2_em_iter_master_kernel.md §2``. M3 (col-norm)
    yields ≤1.4e-2 max-rel-diff on s_lambda but cost rel ~1.6e-9. M4
    (cosine) is bit-identical at D=1175. The kappa_sum reduction
    (M1, in ``_kappa_sum_reduce_STLD``) MUST stay fp64.
    """
    S, T, L, D = X_dot_sl_STLD.shape
    STL = S * T * L
    one_f32 = np.float32(1.0)
    # Pass 1 (parallel over STL): column update + per-(s,t,l) converge bit.
    for stl in prange(STL):
        s = stl // (T * L)
        rem = stl - s * (T * L)
        t = rem // L
        l = rem - t * L

        row_x = X_dot_sl_STLD[s, t, l]
        row_sp = sigma_psi_SLD[s, l]
        row_old = s_t_nu_old_STLD[s, t, l]
        row_new = s_t_nu_new_STLD[s, t, l]

        # Pass 1: build col into row_new, accumulate L2 norm (M3 fp32 acc).
        acc_n_f32 = np.float32(0.0)
        for d in range(D):
            v = kappa_f32 * row_x[d] + row_sp[d]
            row_new[d] = v
            acc_n_f32 += v * v
        # sqrt in fp64 to avoid extra ULP loss; inv_cn is fp32 anyway.
        cn = math.sqrt(np.float64(acc_n_f32))
        inv_cn = np.float32(1.0 / cn) if cn > 0.0 else np.float32(np.inf)

        # Pass 2: normalize + cosine (M4 fp32 acc).
        acc_c_f32 = np.float32(0.0)
        for d in range(D):
            new_v = row_new[d] * inv_cn
            row_new[d] = new_v
            acc_c_f32 += new_v * row_old[d]
        cos_v = acc_c_f32

        converge_STL_scratch[s, t, l] = np.int8(1) if (one_f32 - cos_v) < eps_f32 else np.int8(0)

    # Pass 2 (per-(s, t), small reduction over L): set flag_ST_acc.
    for s in range(S):
        for t in range(T):
            n_converged = 0
            for l in range(L):
                if converge_STL_scratch[s, t, l]:
                    n_converged += 1
            flag_ST_acc[s, t] = np.int8(1) if n_converged >= L else np.int8(0)


@njit(cache=True, fastmath=False, boundscheck=False, error_model='numpy')
def _mstep_inner_loop_master_step2(
    X_dot_sl_STLD,        # (S, T, L, D) fp32 C-contig
    sigma_psi_SLD,        # (S, L, D)    fp32 C-contig
    denom_f64,            # fp64 scalar — T * Σ s_lambda
    dim_f64,              # fp64 scalar — vMF dim (= D)
    kappa_init_f64,       # fp64 scalar
    ini_val_f64,          # fp64 scalar — lower clamp
    eps_f32,
    max_iter_m,
    s_t_nu_A_STLD,        # (S, T, L, D) fp32 C-contig — ping-pong A
    s_t_nu_B_STLD,        # (S, T, L, D) fp32 C-contig — ping-pong B
    flag_ST_acc,          # (S, T) int8 scratch — zeroed at entry
    converge_STL_scratch, # (S, T, L) int8 scratch for parallel reduce
):
    """Multi-subject M-step inner while-loop master.

    Returns ``(final_idx_int, iter_m_int, kappa_final_f64)``. The
    converged s_t_nu lives in ``s_t_nu_A_STLD`` if ``final_idx == 0``
    else ``s_t_nu_B_STLD``.

    Closed-form invad (Banerjee asymptotic) — for D ≥ ~200 (production
    fsaverage6: D=1174) the Bessel polish branch never activates in
    MATLAB's invAd.

    kappa clamp: if ``kappa < ini_val`` (the invAd output dipped below
    the initial seed), clamp to ini_val. Matches the legacy
    ``kappa_update.py`` boundary check.
    """
    S, T, L, D = X_dot_sl_STLD.shape

    # flag_ST_acc is OVERWRITTEN per m_iter inside ``_mstep_fused_update_STLD``
    # (per-iter MATLAB semantics: 1 if converged THIS iter, else 0). No pre-zero
    # needed — first call writes both branches.

    iter_m = 0
    kappa_prev = kappa_init_f64
    kappa_new = kappa_init_f64
    final_idx = 0
    eps_f64 = np.float64(eps_f32)

    while True:
        iter_m += 1
        if (iter_m & 1) == 1:
            buf_old = s_t_nu_A_STLD
            buf_new = s_t_nu_B_STLD
            final_idx = 1
        else:
            buf_old = s_t_nu_B_STLD
            buf_new = s_t_nu_A_STLD
            final_idx = 0

        # 1. Global kappa reduction.
        kappa_sum_f64 = _kappa_sum_reduce_STLD(buf_old, X_dot_sl_STLD)
        rbar = kappa_sum_f64 / denom_f64
        kappa_new = invad_numba(dim_f64, rbar)

        # Inf-safeguard (MATLAB: kappa_update(==Inf) = prev_kappa). For the
        # closed-form path, kappa is always finite given finite rbar in
        # [0, 1). Just defensively check.
        if not math.isfinite(kappa_new):
            kappa_new = kappa_prev
        # ini_val clamp (MATLAB: if kappa < ini_val: kappa = ini_val).
        if kappa_new < ini_val_f64:
            kappa_new = ini_val_f64

        kappa_f32 = np.float32(kappa_new)

        # 2. Fused per-(s, t, l) update + cosine + flag.
        _mstep_fused_update_STLD(
            kappa_f32, X_dot_sl_STLD, sigma_psi_SLD,
            buf_old, buf_new, eps_f32, flag_ST_acc,
            converge_STL_scratch,
        )

        # 3. Convergence — all-(s, t)-flagged AND kappa drift.
        n_flag = 0
        for s in range(S):
            for t in range(T):
                n_flag += flag_ST_acc[s, t]
        all_flag = (n_flag == S * T)
        kappa_drift = abs(kappa_prev - kappa_new) / kappa_prev
        kappa_prev = kappa_new

        if all_flag and (kappa_drift < eps_f64):
            break
        if iter_m > max_iter_m:
            break

    return final_idx, iter_m, kappa_new


# ─────────────────────────────────────────────────────────────────────
# Phase C — Per-subject spatial_connect_prior (gMSHBM only).
#
# Block-diagonal LH/RH structure (boundary_mask makes s_lambda cross-hemi
# zero; gradient is bilateral). For each (s, t):
#     u_h = (grad_h.T @ s_lambda_h) / sum_n s_lambda_h  -> (D_grad, L/2)
#     cross_h = grad_h @ u_h                            -> (n_h, L/2)
#     vmf_h = -||grad_h||² + 2·cross_h - ||u_h||²       -> (n_h, L/2)
#     log_connect[s] += vmf_h (LH-block) + vmf_h (RH-block)
#     log_connect[s][cross-hemi entries] = -inf
#
# Output: log_connect_SNL (S, N, L) fp32 with cross-hemi = -inf.
#
# Implementation uses np.dot for the 4 sgemms per (s, t); the assemble
# step is a numba loop. ~5 GFlops per (s, t) at fsaverage6/D_grad=100.
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _spatial_connect_per_subject_numba(
    grad_TND_s,           # (T, N, D_grad) fp32 C-contig per subject
    s_lambda_NL_s,        # (N, L)         fp32 C-contig per subject
    log_connect_NL_out,   # (N, L)         fp32 C-contig — output
    n_lh,                 # int — N/2
    L_lh,                 # int — L/2
    scratch_u_LD,         # (L, D_grad) fp32 — full u
    scratch_uupd_DL,      # (D_grad, L) fp32 — u_update (full)
    scratch_sumlam_L,     # (L,)        fp32 — sum_n s_lambda
    scratch_grad_sq,      # (N,)        fp32 — per-vertex ||grad||²
    scratch_u_sq,         # (L,)        fp32 — per-parcel ||u||²
    scratch_cross_NL,     # (N, L)      fp32 — cross-product scratch
):
    """Sum-across-T-sessions of -||grad - u_parcel||² per (n, l), in fp32.

    Implements MATLAB lines 549-565 of
    ``cdgMSHBM_estimate_group_priors_sequential.m`` for one subject
    using the gemm-trick decomposition from
    :class:`arealmshbm.spatial_priors.spatial_connect.ConnectSession`.

    Cross-hemi block exploits ``boundary_mask``: ``s_lambda`` is zero on
    cross-hemi cells (upstream normalization), so the full
    ``grad.T @ s_lambda`` sgemm has zeros in the cross-hemi columns of
    ``u_update``. The same-hemi cells get correct contributions. We
    then mask cross-hemi entries of ``log_connect_NL_out`` to -inf
    explicitly in the assemble step.

    Caller responsibilities:
    * ``log_connect_NL_out`` is fully written (zeroed/overwritten);
      no pre-fill needed.
    * cross-hemi cells of ``log_connect_NL_out`` are set to -inf.
    """
    T = grad_TND_s.shape[0]
    N = grad_TND_s.shape[1]
    D_grad = grad_TND_s.shape[2]
    L = scratch_u_LD.shape[0]
    NEG_INF = np.float32(-np.inf)
    ZERO = np.float32(0.0)
    TWO = np.float32(2.0)

    # Initialize log_connect: same-hemi blocks zero, cross-hemi -inf.
    for n in prange(N):
        in_lh_vert = (n < n_lh)
        for l in range(L):
            in_lh_parc = (l < L_lh)
            if in_lh_vert == in_lh_parc:
                log_connect_NL_out[n, l] = ZERO
            else:
                log_connect_NL_out[n, l] = NEG_INF

    # Per-T-session loop.
    for t in range(T):
        grad_t = grad_TND_s[t]  # (N, D_grad) fp32 C-contig view

        # ── Step 1: u_update[d, l] = Σ_n grad_t[n, d] * s_lambda[n, l]
        # Full sgemm via numba np.dot — dispatches to cblas_sgemm with
        # transA flag (grad_t.T is F-contig view). Cross-hemi columns
        # of u_update are 0 because s_lambda is cross-hemi-zero (the
        # upstream boundary_mask).
        scratch_uupd_DL[:] = np.dot(grad_t.T, s_lambda_NL_s)

        # ── Step 2: sum_lambda per parcel (full L; cross-hemi half is zero
        # numerically, but we don't depend on that — division by zero
        # produces inf which we mask).
        for l in prange(L):
            acc = ZERO
            for n in range(N):
                acc += s_lambda_NL_s[n, l]
            scratch_sumlam_L[l] = acc

        # ── Step 3: u[l, d] = u_update[d, l] / sum_lambda[l].
        # MATLAB-literal: empty parcels (sum_lambda=0, u_update=0) yield
        # 0/0=NaN, which propagates through u_sq → vmf → log_connect →
        # softmax. The dead-column zero step then drops the all-NaN
        # columns. Mirroring MATLAB's behavior here (rather than the
        # earlier "inv=0 if sl==0" guard) is required for bit-for-bit
        # E-step output at empty-parcel cells.
        for l in prange(L):
            sl = scratch_sumlam_L[l]
            inv = np.float32(1.0) / sl   # 0/0=NaN, x/0=inf (numpy error_model)
            for d in range(D_grad):
                scratch_u_LD[l, d] = scratch_uupd_DL[d, l] * inv

        # ── Step 4: per-parcel ||u||².
        for l in prange(L):
            acc = ZERO
            for d in range(D_grad):
                v = scratch_u_LD[l, d]
                acc += v * v
            scratch_u_sq[l] = acc

        # ── Step 5: per-vertex ||grad_t||² (per-session — must recompute).
        for n in prange(N):
            acc = ZERO
            for d in range(D_grad):
                v = grad_t[n, d]
                acc += v * v
            scratch_grad_sq[n] = acc

        # ── Step 6: cross[n, l] = grad_t @ u_LD.T   (full sgemm)
        # u_LD.T is F-contig (D_grad, L) view of (L, D_grad).
        scratch_cross_NL[:] = np.dot(grad_t, scratch_u_LD.T)

        # ── Step 7: assemble same-hemi vmf and accumulate into log_connect.
        # Cross-hemi cells stay at -inf (initialized in Step 0).
        for n in prange(N):
            in_lh_vert = (n < n_lh)
            gsq = scratch_grad_sq[n]
            for l in range(L):
                in_lh_parc = (l < L_lh)
                if in_lh_vert == in_lh_parc:
                    vmf = (TWO * scratch_cross_NL[n, l]
                           - gsq - scratch_u_sq[l])
                    log_connect_NL_out[n, l] += vmf


# ─────────────────────────────────────────────────────────────────────
# Phase D — Per-subject fused E-step (NTD variant).
#
# BOLD lives in (S, N, T, D) layout; the per-(s, t) slice would be
# N-strided and numba's np.dot would fall through to a generic loop.
# Instead, fold both ``t`` and ``d`` into a single contraction:
#
#     log_vmf[n, l] = κ · Σ_t lv[t, n, l]  +  count_alive[n] · cdln_val
#     lv[t, n, l]   = Σ_d X[n, t, d] · s_t_nu[t, l, d]
#
# Step 1: reshape BOLD[s] (N, T, D) → (N, T·D) zero-copy, transpose
#   s_t_nu_TLD[s] (T, L, D) → (T, D, L) into a scratch buffer, then one
#   big sgemm (N, T·D) × (T·D, L) → (N, L) collapses both ``t`` and
#   ``d`` in a single BLAS call → ``lv_sum_NL``.
# Step 2: per-vertex medial-wall scan — vertex n is "alive" for session
#   t iff X[n, t, :] has any nonzero entry. ``count_alive[n]`` is the
#   number of sessions where vertex n carries real BOLD.
# Step 3: ``log_vmf[n, l] = κ · lv_sum[n, l] + count_alive[n] · cdln_val``.
#
# Equivalence to the original per-t per-alive kernel
# ----------------------------------------------------
# The original kernel fired the ``not alive`` branch on per-(t, n) rows
# where any ``lv[t, n, l] * κ == 0`` in fp32. For non-medial vertices
# (BOLD row nonzero), ``lv`` is a sum of D=1175 nonzero products and is
# never exactly zero in fp32 (statistically impossible). For medial
# vertices (BOLD row identically zero), ``lv = 0`` for every l.
#
# So in production data, ``alive_t_n = (BOLD[s, n, t, :] has any
# nonzero entry)`` — a per-(t, n) BOLD-row scan replaces the
# materialize-lv-then-check pattern, with the bonus that we no longer
# need (T, N, L) scratch.
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True, inline='always')
def _fused_estep_per_subject_NTD(
    X_NTD,             # (N, T, D) fp32 C-contig  ← was (T, N, D)
    s_t_nu_TLD,        # (T, L, D) fp32 C-contig — unchanged
    kappa_scalar_f64,
    dim_int,
    theta_NL,          # (N, L) fp32 C-contig
    log_connect_NL,    # (N, L) fp32 C-contig — gMSHBM only; zeros otherwise
    boundary_mask_NL,  # (N, L) fp32 C-contig
    beta_f64,
    has_spatial,       # int — 1 if gMSHBM, 0 otherwise
    log_vmf_NL_buf,    # (N, L) fp32 C-contig — written then read
    s_lambda_NL_out,   # (N, L) fp64 C-contig — output (per-subject scratch; caller normalizes + casts to fp32 storage)
    tmp_idx_NL_out,    # (N, L) bool C-contig — output
    s_t_nu_TDL_scratch,  # (T, D, L) fp32 C-contig — Phase D transpose scratch
    log_lambda_scratch_NL,  # (N, L) fp32 C-contig — Session-owned, reused; zero-init not required (overwritten)
    n_alive_count_N,        # (N,)   int32 C-contig — Session-owned, reused; zero-init'd at entry
):
    """Returns scalar fp64 cost. Writes ``s_lambda_NL_out``,
    ``tmp_idx_NL_out``, and uses ``log_vmf_NL_buf`` and
    ``s_t_nu_TDL_scratch`` as scratch.

    Per-(t, n) alive flag is derived from a BOLD-row nonzero scan;
    medial-wall vertices (all-zero BOLD row) contribute nothing to
    the softmax sum. Equivalent to checking ``lv != 0`` exactly per
    cluster but cheaper (one scan per row vs L scans per row).

    ``s_lambda_NL_out`` is a fp64 SCRATCH (single-subject, ~262 MB
    at fsa6/L=400). The softmax exp() intermediate stays fp64; the
    caller (master kernel) is responsible for running a fp64-precision
    Phase E.1 normalize on this scratch and casting the normalized
    result to fp32 storage via
    :func:`_phase_e1_normalize_per_subject_kernel`. This preserves
    fp64 precision through the whole D → E.1 chain (multiply by bm,
    row-sum, divide), losing precision only at the final fp32 store
    after normalization.
    """
    N = X_NTD.shape[0]
    T = X_NTD.shape[1]
    D = X_NTD.shape[2]
    L = s_t_nu_TLD.shape[1]

    cdln_v = float(dim_int) * 0.5 - 1.0
    cdln_val_f32 = np.float32(_cdln_single(kappa_scalar_f64, cdln_v))
    kappa_f32 = np.float32(kappa_scalar_f64)
    zero_f32 = np.float32(0.0)

    # ── Step 1a: transpose s_t_nu_TLD (T, L, D) → s_t_nu_TDL (T, D, L) ──
    # Required so the big sgemm's right operand is (T·D, L) C-contig.
    # ~T·L·D ≈ 940 K writes per subject; LLVM auto-vectorizes the inner
    # loop (D contiguous on the read side, L contiguous on the write side).
    for t in range(T):
        for l in range(L):
            row_in = s_t_nu_TLD[t, l]
            for d in range(D):
                s_t_nu_TDL_scratch[t, d, l] = row_in[d]

    # ── Step 1b: big sgemm — (N, T·D) × (T·D, L) → (N, L) ──
    # Contracts over both ``t`` and ``d`` in a single BLAS call. The
    # output is exactly ``lv_sum[n, l] = Σ_t Σ_d X[n, t, d] · s_t_nu[t, l, d]``.
    bold_NX_view = X_NTD.reshape(N, T * D)
    stnu_XL_view = s_t_nu_TDL_scratch.reshape(T * D, L)
    lv_sum = np.dot(bold_NX_view, stnu_XL_view)  # (N, L) fp32 C-contig

    # ── Step 1c: per-(t, n) alive — count nonzero BOLD sessions per vertex ──
    # ``n_alive[n]`` = number of sessions ``t`` where ``X[n, t, :]`` has
    # any nonzero entry. For medial-wall vertices (X[n, t, :] all zero
    # for every t), n_alive[n] = 0. For canonical non-medial vertices,
    # n_alive[n] = T. Per-session medial edge cases (a vertex zero in
    # one session but not another) land in 0 < n_alive[n] < T.
    # n_alive_count_N is caller-owned (Session scratch); zero-init here.
    for n in prange(N):
        n_alive_count_N[n] = 0
    for t in range(T):
        for n in prange(N):
            is_alive_t = False
            for d in range(D):
                if X_NTD[n, t, d] != zero_f32:
                    is_alive_t = True
                    break
            if is_alive_t:
                n_alive_count_N[n] += 1

    # ── Step 1d: assemble log_vmf = κ · lv_sum + n_alive · cdln_val ──
    for n in prange(N):
        cdln_add = np.float32(n_alive_count_N[n]) * cdln_val_f32
        for l in range(L):
            log_vmf_NL_buf[n, l] = kappa_f32 * lv_sum[n, l] + cdln_add

    # ── Step 2: tmp_idx + softmax ──
    #
    # log_vmf pre-softmax COMPOSITION (``log_vmf + log(θ) + β·log_connect``)
    # runs in fp32; the exp() itself stays fp64 (β=5000 produces ~1e-242
    # tail values that hard-underflow fp32's 1.18e-38 minimum normal —
    # non-negotiable). See ``docs/step2_em_iter_master_kernel.md §2``.
    # log_lambda_scratch_NL is caller-owned (Session scratch); fully
    # overwritten below so no zero-init needed.
    log_lambda_scratch = log_lambda_scratch_NL
    beta_f32 = np.float32(beta_f64)
    for n in prange(N):
        # Per-row "any zero entry?" → broadcast tmp_idx row.
        # Exact fp32-zero test: log_vmf == 0 iff the (n, l) cell got
        # zero contribution from BOTH κ·lv_sum and n_alive·cdln. For
        # medial vertices this is the normal case (lv_sum=0 because
        # BOLD row is zero; n_alive=0 → cdln_add=0). For non-medial
        # vertices a coincidence ``κ·lv_sum + n_alive·cdln_val == 0``
        # in fp32 would also flip this row to True and zero it; at
        # D=1175 with sums of D nonzero products this is vanishingly
        # rare. If a future regression mysteriously zeros sparse rows,
        # suspect this site first.
        row_has_zero = False
        for l in range(L):
            if log_vmf_NL_buf[n, l] == zero_f32:
                row_has_zero = True
                break
        if row_has_zero:
            for l in range(L):
                tmp_idx_NL_out[n, l] = True
        else:
            for l in range(L):
                tmp_idx_NL_out[n, l] = False

        # Compose log_vmf_aug = log_vmf + log(θ) [+ β·log_connect] in fp32,
        # track row_max (nanmax) in fp32.
        row_max_f32 = np.float32(-np.inf)
        for l in range(L):
            v = log_vmf_NL_buf[n, l]
            theta_v = theta_NL[n, l]
            if theta_v > zero_f32:
                v = v + np.float32(math.log(np.float64(theta_v)))
            else:
                v = np.float32(-np.inf)
            if has_spatial == 1:
                v = v + beta_f32 * log_connect_NL[n, l]
            log_lambda_scratch[n, l] = v
            if v == v and v > row_max_f32:
                row_max_f32 = v
        if not (row_max_f32 == row_max_f32) or math.isinf(row_max_f32):
            row_max_f32 = zero_f32

        # Cast to fp64 only at the exp step (β=5000 subnormal — mandatory).
        # Output stays fp64; the per-subject Phase E.1 normalize kernel
        # reads this fp64 scratch and casts to fp32 storage in one pass.
        row_max_f64 = np.float64(row_max_f32)
        for l in range(L):
            shifted = np.float64(log_lambda_scratch[n, l]) - row_max_f64
            s_lambda_NL_out[n, l] = math.exp(shifted)

    # ── Dead-column zero ──
    for l in prange(L):
        cs = 0.0
        for n in range(N):
            v = s_lambda_NL_out[n, l]
            if v == v:
                cs += v
        if cs == 0.0:
            for n in range(N):
                s_lambda_NL_out[n, l] = 0.0

    # ── Step 3: per-subject cost (fp32 acc — variant A) ──
    #
    # Variant-A ablation: cost is only used by the per-subject EM
    # convergence test (rel-diff vs prev iter). The fp32 cost
    # accumulator contributes ≤8e-7 rel-diff vs fp64 — well below the
    # 1e-4 convergence-test bar. Final Params are bit-identical.
    log_eps20_f32 = np.float32(_LOG_EPS20)
    row_cost = np.zeros(N, dtype=np.float32)
    for n in prange(N):
        row_sum_f32 = np.float32(0.0)
        for l in range(L):
            row_sum_f32 += np.float32(s_lambda_NL_out[n, l]) * boundary_mask_NL[n, l]

        row_has_zero = tmp_idx_NL_out[n, 0]

        c_f32 = np.float32(0.0)
        for l in range(L):
            slc_raw = np.float32(s_lambda_NL_out[n, l]) * boundary_mask_NL[n, l]
            if row_sum_f32 > np.float32(0.0):
                slc = slc_raw / row_sum_f32
            else:
                slc = np.float32(0.0)
            if slc != slc:
                slc = np.float32(0.0)
            if row_has_zero:
                slc = np.float32(0.0)

            theta_v = theta_NL[n, l]
            if theta_v > np.float32(0.0):
                ltheta_f64 = math.log(np.float64(theta_v))
                if math.isinf(ltheta_f64):
                    ltheta = log_eps20_f32
                else:
                    ltheta = np.float32(ltheta_f64)
            else:
                ltheta = log_eps20_f32

            if slc > np.float32(0.0):
                lslc_f64 = math.log(np.float64(slc))
                if math.isinf(lslc_f64):
                    lslc = log_eps20_f32
                else:
                    lslc = np.float32(lslc_f64)
            else:
                lslc = log_eps20_f32

            c_f32 += slc * log_vmf_NL_buf[n, l]
            c_f32 += slc * ltheta
            c_f32 -= slc * lslc

            if has_spatial == 1:
                lc = log_connect_NL[n, l]
                if not (lc == lc):
                    lc = log_eps20_f32
                if math.isinf(lc):
                    lc = log_eps20_f32
                c_f32 += beta_f32 * slc * lc

        row_cost[n] = c_f32

    cost = np.float32(0.0)
    for n in range(N):
        cost += row_cost[n]
    return np.float64(cost)


# ─────────────────────────────────────────────────────────────────────
# Phase E — Normalize + theta update.
#
#   s_lambda_SNL *= boundary_mask     (broadcast)
#   row_sums[s, n] = Σ_l s_lambda[s, n, l]
#   s_lambda[s, n, l] /= row_sums (where row_sums > 0; else 0)
#   tmp_idx[s, n, l] → s_lambda[s, n, l] = 0
#   theta[n, l] = mean_{s} s_lambda[s, n, l]
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _phase_e1_normalize_per_subject_kernel(
    s_lambda_NL_f64_scratch,  # (N, L) fp64 C-contig — IN: Phase D output; modified in-place
    tmp_idx_NL,               # (N, L) bool C-contig — IN
    boundary_mask_NL,         # (N, L) fp32 C-contig — IN
    s_lambda_NL_f32_out,      # (N, L) fp32 C-contig — OUT (one subject's slice of storage)
):
    """Per-subject Phase E.1 normalize, fp32-storage-aware.

    Reads the per-subject Phase D fp64 scratch (raw softmax numerator),
    runs the multiply-by-bm + row-sum + divide + tmp_idx-zero
    sequence ENTIRELY IN FP64, and casts the normalized result to
    fp32 storage in a single pass at the end.

    ⚠️ ``rs`` MUST stay fp64. fp32 cumulative subnormal flips zero
    rows that have only softmax-tail values. See
    ``docs/step2_em_iter_master_kernel.md §2`` (E row-normalize site).
    """
    N, L = s_lambda_NL_f64_scratch.shape
    for n in prange(N):
        row = s_lambda_NL_f64_scratch[n]
        bm_row = boundary_mask_NL[n]
        ti_row = tmp_idx_NL[n]
        out_row = s_lambda_NL_f32_out[n]
        rs = 0.0
        for l in range(L):
            v = row[l] * np.float64(bm_row[l])
            row[l] = v
            rs += v
        if rs > 0.0:
            for l in range(L):
                row[l] = row[l] / rs
        else:
            for l in range(L):
                row[l] = 0.0
        if ti_row[0]:
            for l in range(L):
                row[l] = 0.0
        # Single fp64 → fp32 cast at the end of the row.
        for l in range(L):
            out_row[l] = np.float32(row[l])


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True, inline='always')
def _phase_e2_theta_only_kernel(
    s_lambda_SNL_f32,     # (S, N, L) fp32 C-contig — IN (post-Phase-E.1)
    theta_NL_out,         # (N, L) fp32 C-contig — OUT
):
    """Phase E.2: ``theta[n, l] = mean_s s_lambda[s, n, l]``.

    fp32 accumulator across S (safe for cohort sums at S ≤ ~1k)."""
    S, N, L = s_lambda_SNL_f32.shape
    inv_S_f32 = np.float32(1.0 / float(S))
    for n in prange(N):
        for l in range(L):
            acc_f32 = np.float32(0.0)
            for s in range(S):
                acc_f32 += s_lambda_SNL_f32[s, n, l]
            theta_NL_out[n, l] = acc_f32 * inv_S_f32


# ─────────────────────────────────────────────────────────────────────
# EM-iter master kernel — orchestrates Phases A → E for ONE outer EM
# iter. Callers (Step2EmIterSession.run_iter) handle the convergence
# check between calls. Phase A.3 and Phase C+D+E run per-subject from
# Python with loader callbacks, so peak BOLD/grad RAM stays bounded at
# one subject's slab regardless of S.
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# Warmup — JIT-compile every kernel once at process start.
# ─────────────────────────────────────────────────────────────────────
def warmup() -> None:
    """JIT-compile the master + every sub-kernel. Idempotent."""
    S, T, N, D, L, D_grad = 2, 2, 8, 4, 4, 3
    n_lh = N // 2
    L_lh = L // 2
    rng = np.random.default_rng(0)

    # BOLD layout: (S, N, T, D) — NTD.
    BOLD = rng.standard_normal((S, N, T, D)).astype(np.float32)
    grad = rng.standard_normal((S, T, N, D_grad)).astype(np.float32)
    bm = np.ones((N, L), dtype=np.float32)
    s_lambda_f32 = np.full((S, N, L), 0.25, dtype=np.float32)
    s_t_nu_A = (rng.standard_normal((S, T, L, D)) * 0.1).astype(np.float32)
    s_t_nu_B = np.empty_like(s_t_nu_A)
    theta = np.full((N, L), 0.25, dtype=np.float32)
    cost = np.zeros(S, dtype=np.float64)
    X_dot_sl = np.empty((S, T, L, D), dtype=np.float32)
    sigma_psi = np.empty((S, L, D), dtype=np.float32)
    log_connect = np.zeros((N, L), dtype=np.float32)
    log_vmf = np.zeros((N, L), dtype=np.float32)
    tmp_idx = np.zeros((N, L), dtype=np.bool_)
    flag_ST = np.zeros((S, T), dtype=np.int8)
    converge_STL = np.zeros((S, T, L), dtype=np.int8)
    scratch_u = np.empty((L, D_grad), dtype=np.float32)
    scratch_uupd_DL = np.empty((D_grad, L), dtype=np.float32)
    scratch_sumlam_L = np.empty(L, dtype=np.float32)
    scratch_grad_sq = np.empty(N, dtype=np.float32)
    scratch_u_sq = np.empty(L, dtype=np.float32)
    scratch_cross_NL = np.empty((N, L), dtype=np.float32)
    s_t_nu_TDL_scratch = np.empty((T, D, L), dtype=np.float32)
    sigma = np.full(L, 0.1, dtype=np.float32)
    s_psi = (rng.standard_normal((S, L, D)) * 0.05).astype(np.float32)
    # Per-subject fp64 scratch for Phase D output.
    s_lambda_NL_f64_scratch = np.empty((N, L), dtype=np.float64)
    # Per-subject E-step scratch (Session-promoted in production; the
    # warmup allocates a stub so the signature matches).
    log_lambda_scratch_NL_w = np.empty((N, L), dtype=np.float32)
    n_alive_count_N_w = np.empty(N, dtype=np.int32)

    # Streaming-master warmup — runs the Python orchestrator once on
    # tiny shapes to surface signature errors immediately.
    from arealmshbm.step2_io import (
        InMemoryGradientLoader,
        InMemoryProfileLoader,
    )
    BOLD_w = np.ascontiguousarray(BOLD)  # (S, N, T, D) fp32 C-contig
    grad_w = np.ascontiguousarray(grad)  # (S, T, N, D_grad) fp32 C-contig
    bold_loader_w = InMemoryProfileLoader(BOLD_w, num_session=T)
    grad_loader_w = InMemoryGradientLoader(grad_w, num_session=T)

    def _bold_into(s_1, out):
        bold_loader_w.load_into(s_1, out)

    def _grad_into(s_1, out):
        grad_loader_w.load_into(s_1, out)

    bold_scratch_w = np.empty((N, T, D), dtype=np.float32)
    grad_scratch_w = np.empty((T, N, D_grad), dtype=np.float32)

    em_iter_master_kernel_streaming(
        bold_scratch_w, grad_scratch_w, _bold_into, _grad_into,
        bm, 1,
        s_lambda_f32, s_t_nu_A, s_t_nu_B, theta, cost,
        s_lambda_NL_f64_scratch,
        X_dot_sl, sigma_psi,
        log_connect, log_vmf, tmp_idx, flag_ST, converge_STL,
        scratch_u, scratch_uupd_DL, scratch_sumlam_L,
        scratch_grad_sq, scratch_u_sq, scratch_cross_NL,
        s_t_nu_TDL_scratch,
        log_lambda_scratch_NL_w, n_alive_count_N_w,
        sigma, s_psi,
        D, 50.0, 30.0, 5000.0, np.float32(1e-4), 3, n_lh, L_lh,
    )
    em_iter_master_kernel_streaming(
        bold_scratch_w, grad_scratch_w, _bold_into, _grad_into,
        bm, 0,
        s_lambda_f32, s_t_nu_A, s_t_nu_B, theta, cost,
        s_lambda_NL_f64_scratch,
        X_dot_sl, sigma_psi,
        log_connect, log_vmf, tmp_idx, flag_ST, converge_STL,
        scratch_u, scratch_uupd_DL, scratch_sumlam_L,
        scratch_grad_sq, scratch_u_sq, scratch_cross_NL,
        s_t_nu_TDL_scratch,
        log_lambda_scratch_NL_w, n_alive_count_N_w,
        sigma, s_psi,
        D, 50.0, 30.0, 5000.0, np.float32(1e-4), 3, n_lh, L_lh,
    )


# ─────────────────────────────────────────────────────────────────────
# Streaming master kernel — Python orchestrator.
#
# The two per-subject loops (Phase A.3 and Phase C+D+E) live in Python
# so the caller can decode one subject's BOLD / grad from disk on each
# visit and free the bytes between visits. Phase B (multi-subject M-
# step), Phase A.1 (sigma_psi precompute), Phase A.4 (parity copy-
# back) and Phase E.2 (theta = mean over S) do NOT read BOLD/grad and
# remain numba-only.
#
# s_lambda storage is fp32 throughout the EM loop. Phase A.3 reads
# fp32 directly; Phase D's softmax exp() uses fp64 inside a per-row
# scratch (s_lambda_NL_f64_scratch), with the cast back to fp32
# storage happening at the end of Phase E.1
# (_phase_e1_normalize_per_subject_kernel).
# ─────────────────────────────────────────────────────────────────────


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _phase_a4_copy_b_to_a(s_t_nu_A_STLD, s_t_nu_B_STLD):
    """Copy ping-pong B back into A (odd-parity m_iters)."""
    S, T, L, D = s_t_nu_A_STLD.shape
    STL = S * T * L
    for stl in prange(STL):
        s = stl // (T * L)
        rem = stl - s * (T * L)
        t = rem // L
        l = rem - t * L
        row_a = s_t_nu_A_STLD[s, t, l]
        row_b = s_t_nu_B_STLD[s, t, l]
        for d in range(D):
            row_a[d] = row_b[d]


def em_iter_master_kernel_streaming(
    # ── Streaming sources + per-subject scratch slots ──
    bold_scratch_NTD,        # (N, T, D)        fp32 C-contig — re-used per subject
    grad_scratch_TND,        # (T, N, D_grad)   fp32 C-contig — re-used per subject
    load_bold_into,          # callable(s_1, out) — fills bold_scratch_NTD for subject s_1
    load_grad_into,          # callable(s_1, out) — fills grad_scratch_TND for subject s_1
    # ── Static (caller-owned, never mutated): ──
    boundary_mask_NL,        # (N, L) fp32 C-contig
    has_spatial,             # int — 1 if gMSHBM, 0 otherwise
    # ── Mutable state (in-place updates): ──
    s_lambda_SNL_f32,        # (S, N, L) fp32 C-contig — IN/OUT
    s_t_nu_A_STLD,           # (S, T, L, D) fp32 C-contig — IN (current); will be written
    s_t_nu_B_STLD,           # (S, T, L, D) fp32 C-contig — scratch ping-pong
    theta_NL,                # (N, L) fp32 C-contig — IN (prev) / OUT (this iter)
    cost_S_out,              # (S,) fp64 — OUT (this iter's per-subject cost)
    # ── Per-iter scratch (Session-owned, reused): ──
    s_lambda_NL_f64_scratch, # (N, L) fp64 C-contig — Phase D output scratch (single-subject)
    X_dot_sl_STLD, sigma_psi_SLD,
    log_connect_NL_buf, log_vmf_NL_buf,
    tmp_idx_NL_buf, flag_ST_acc,
    converge_STL_scratch,
    # ── Spatial scratch (gMSHBM): ──
    scratch_u_LD, scratch_uupd_DL, scratch_sumlam_L,
    scratch_grad_sq, scratch_u_sq, scratch_cross_NL,
    # ── Phase D scratch ──
    s_t_nu_TDL_scratch,
    # ── Per-subject E-step scratch (Session-owned, reused per S): ──
    log_lambda_scratch_NL,   # (N, L) fp32 — Phase D log_lambda pre-softmax
    n_alive_count_N,         # (N,)   int32 — Phase D per-vertex alive count
    # ── Per-EM-iter precomputed (Session): ──
    sigma_L, s_psi_SLD,
    # ── Scalars: ──
    dim_int,
    kappa_init_f64, ini_val_f64, beta_f64, eps_f32,
    max_iter_m, n_lh, L_lh,
):
    """Streaming variant of :func:`em_iter_master_kernel`. Returns
    ``(m_iters_int, kappa_final_f64)``.

    Two outer subject loops are lifted to Python so each subject's
    BOLD/grad lives in RAM only for the duration of that subject's
    visit; ``load_bold_into`` / ``load_grad_into`` are callables that
    decode subject ``s`` (1-based) into the pre-allocated scratch
    slots. The Session ctor closes over the loaders to provide these.

    ``s_lambda_SNL_f32`` IS the storage; no separate f32 scratch. Phase
    D's softmax exp() intermediate is computed in fp64 (β=5000 produces
    subnormal tail values that flush fp32) but cast to fp32 on store —
    see :func:`_fused_estep_per_subject_NTD`.

    At fsa6/S=200/T=6/D=1175/D_grad=100 the per-iter peak scratch is:
      * BOLD scratch  ~770 MB
      * grad scratch  ~200 MB (gMSHBM only)
    independent of S; ``s_lambda`` storage is 26 GB at S=200.
    """
    S = int(s_lambda_SNL_f32.shape[0])

    # Phase A.1 — sigma_psi precompute.
    _sigma_psi_SLD_compute(sigma_L, s_psi_SLD, sigma_psi_SLD)

    # Phase A.3 — streamed per subject. Decode BOLD[s] into scratch;
    # s_lambda_SNL_f32[s] is read directly (no cast — storage IS fp32).
    for s in range(S):
        load_bold_into(s + 1, bold_scratch_NTD)
        _compute_X_dot_sl_s_NTD(
            bold_scratch_NTD,
            s_lambda_SNL_f32[s],
            X_dot_sl_STLD[s],
        )

    # Phase B — Multi-subject M-step inner while-loop. Pure compute
    # over X_dot_sl_STLD / sigma_psi_SLD / s_t_nu_A/B_STLD — does NOT
    # touch BOLD/grad/s_lambda. Unchanged.
    _, m_iters, kappa_new = _mstep_inner_loop_master_step2(
        X_dot_sl_STLD,
        sigma_psi_SLD,
        _denom_global_compute(np.float64(s_t_nu_A_STLD.shape[1]),
                              s_lambda_SNL_f32),
        np.float64(dim_int),
        kappa_init_f64,
        ini_val_f64,
        eps_f32,
        max_iter_m,
        s_t_nu_A_STLD,
        s_t_nu_B_STLD,
        flag_ST_acc,
        converge_STL_scratch,
    )
    if (m_iters & 1) == 1:
        _phase_a4_copy_b_to_a(s_t_nu_A_STLD, s_t_nu_B_STLD)

    # Phase C+D+E (combined per-subject loop) — streamed.
    #   Per subject: decode BOLD[s] + (if gMSHBM) grad[s] into scratch,
    #   run Phase C spatial-connect (gMSHBM only) reading s_lambda_f32[s]
    #   directly, and Phase D fused E-step that writes the new
    #   s_lambda_f32[s] (with fp64 softmax intermediate internally).
    for s in range(S):
        load_bold_into(s + 1, bold_scratch_NTD)
        if has_spatial == 1:
            load_grad_into(s + 1, grad_scratch_TND)
            _spatial_connect_per_subject_numba(
                grad_scratch_TND,
                s_lambda_SNL_f32[s],
                log_connect_NL_buf,
                n_lh, L_lh,
                scratch_u_LD, scratch_uupd_DL, scratch_sumlam_L,
                scratch_grad_sq, scratch_u_sq, scratch_cross_NL,
            )
        # Phase D writes to per-subject fp64 scratch.
        cost_S_out[s] = _fused_estep_per_subject_NTD(
            bold_scratch_NTD,
            s_t_nu_A_STLD[s],
            kappa_new,
            np.int64(dim_int),
            theta_NL,
            log_connect_NL_buf,
            boundary_mask_NL,
            beta_f64,
            has_spatial,
            log_vmf_NL_buf,
            s_lambda_NL_f64_scratch,
            tmp_idx_NL_buf,
            s_t_nu_TDL_scratch,
            log_lambda_scratch_NL,
            n_alive_count_N,
        )
        # Phase E.1 — Per-subject normalize: fp64 scratch → fp32 storage.
        _phase_e1_normalize_per_subject_kernel(
            s_lambda_NL_f64_scratch,
            tmp_idx_NL_buf,
            boundary_mask_NL,
            s_lambda_SNL_f32[s],
        )

    # Phase E.2 — theta = mean over S.
    _phase_e2_theta_only_kernel(s_lambda_SNL_f32, theta_NL)

    return m_iters, kappa_new
