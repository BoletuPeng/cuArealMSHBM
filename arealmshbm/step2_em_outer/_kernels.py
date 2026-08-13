"""_kernels.py — @njit kernels for the three step-2 outer-EM closed-form leaves.

Replaces the naive numpy implementations in:
    intra_em_cost.py            → :func:`_intra_em_cost_step2_kernel`
    inter_subject_var.py        → :func:`_inter_subject_var_*_update`
    intra_subject_var_loop.py   → :func:`_intra_subject_var_loop_kernel`

**Internal layout convention** — matches the ``step2_em_iter_master``
family. All arrays are C-contig with ``d`` unit-stride, so the inner
``for d in range(D)`` loops fill cache linearly:

    s_psi    : (S, L, D)     fp32
    s_t_nu   : (S, T, L, D)  fp32
    mu       : (L, D)        fp32
    sigma    : (L,)          fp32
    epsil    : (L,)          fp32
    cost_em  : (S,)          fp64
    theta    : (N, L)        fp32
    s_lambda : (S, N, L)     fp64
    boundary_mask : (N, L)   fp32

All kernels keep the same **fp32 storage / fp64 accumulator** contract
as the rest of step-2.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from arealmshbm.em_stop_criterion._cdln import _cdln_single
from arealmshbm.m_step._invad import invad_numba


# ─────────────────────────────────────────────────────────────────────
# intra_em_cost_step2 — outer-loop convergence cost (L16).
#
# Math (algebraically identical to the numpy form in intra_em_cost.py):
#   term1_data = Σ_l σ[l] · (Σ_{s,t,d} ψ[s,l,d] · ν[s,t,l,d])    fp64
#   term1_cdln = (S * T) · Σ_l Cdln(σ_l, dim)                    fp64
#   term2_data = Σ_l ε[l] · (Σ_{s,d} ψ[s,l,d] · μ[l,d])          fp64
#   term2_cdln = S · Σ_l Cdln(ε_l, dim)                          fp64
#   update_cost = term1_data + term1_cdln + term2_data + term2_cdln + Σ_s cost_em
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _intra_em_cost_step2_kernel(
    psi_SLD,        # (S, L, D)     fp32 C-contig
    nu_STLD,        # (S, T, L, D)  fp32 C-contig
    mu_LD,          # (L, D)        fp32 C-contig
    sigma_L,        # (L,)          fp32
    epsil_L,        # (L,)          fp32
    cost_em_S,      # (S,)          fp64
    dim_f64,        # fp64 scalar
):
    """Single-pass reduction, d-innermost unit-stride. fp64 accumulators."""
    S = psi_SLD.shape[0]
    L = psi_SLD.shape[1]
    D = psi_SLD.shape[2]
    T = nu_STLD.shape[1]

    cdln_v = 0.5 * dim_f64 - 1.0
    term1_per_l = np.empty(L, dtype=np.float64)
    term2_per_l = np.empty(L, dtype=np.float64)
    cdln_sigma_per_l = np.empty(L, dtype=np.float64)
    cdln_epsil_per_l = np.empty(L, dtype=np.float64)

    for l in prange(L):
        # term1: Σ_{s,t,d} ψ[s,l,d] · ν[s,t,l,d]  (fp64 acc, d unit-stride)
        acc1 = 0.0
        for s in range(S):
            psi_row = psi_SLD[s, l]
            for t in range(T):
                nu_row = nu_STLD[s, t, l]
                for d in range(D):
                    acc1 += np.float64(psi_row[d]) * np.float64(nu_row[d])
        term1_per_l[l] = acc1

        # term2: Σ_{s,d} ψ[s,l,d] · μ[l,d]   (d unit-stride)
        acc2 = 0.0
        mu_row = mu_LD[l]
        for s in range(S):
            psi_row = psi_SLD[s, l]
            for d in range(D):
                acc2 += np.float64(psi_row[d]) * np.float64(mu_row[d])
        term2_per_l[l] = acc2

        # Cdln per-l.
        cdln_sigma_per_l[l] = _cdln_single(np.float64(sigma_L[l]), cdln_v)
        cdln_epsil_per_l[l] = _cdln_single(np.float64(epsil_L[l]), cdln_v)

    term1_data = 0.0
    term1_cdln = 0.0
    term2_data = 0.0
    term2_cdln = 0.0
    for l in range(L):
        term1_data += np.float64(sigma_L[l]) * term1_per_l[l]
        term2_data += np.float64(epsil_L[l]) * term2_per_l[l]
        term1_cdln += cdln_sigma_per_l[l]
        term2_cdln += cdln_epsil_per_l[l]
    term1_cdln *= np.float64(S * T)
    term2_cdln *= np.float64(S)

    cost_em_sum = 0.0
    for s in range(S):
        cost_em_sum += cost_em_S[s]

    return term1_data + term1_cdln + term2_data + term2_cdln + cost_em_sum


# ─────────────────────────────────────────────────────────────────────
# inter_subject_var — (μ, ε) closed-form update from per-subject s_psi (L18).
#
#   mu_update[l, d] = Σ_s ψ[s, l, d]                              (L, D)
#   mu_new[l, d]    = mu_update[l, d] / ||mu_update[l, :]||₂      (zero-col → prev_mu)
#
#   ε_input[l]      = Σ_{s, d} ψ[s, l, d] · μ_new[l, d] / S        (L,)
#   ε_new[l]        = invAd(dim, min(ε_input[l], 1.0))   with ini_val/inf fallbacks
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _inter_subject_var_mu_update(
    psi_SLD,        # (S, L, D) fp32 C-contig
    prev_mu_LD,     # (L, D)    fp32 C-contig — fallback for zero-norm
    mu_new_LD,      # (L, D)    fp32 C-contig — OUT
    mu_norms_L,     # (L,)      fp32          — OUT (per-l column norm)
):
    """μ update — d unit-stride, fp64 norm accumulator."""
    S = psi_SLD.shape[0]
    L = psi_SLD.shape[1]
    D = psi_SLD.shape[2]
    for l in prange(L):
        # Σ_s ψ[s, l, d] into mu_new_LD[l] (per-l fp32 row).
        out_row = mu_new_LD[l]
        acc_norm2 = 0.0
        for d in range(D):
            v = 0.0
            for s in range(S):
                v += np.float64(psi_SLD[s, l, d])
            out_row[d] = np.float32(v)
            acc_norm2 += v * v
        nrm = math.sqrt(acc_norm2)
        mu_norms_L[l] = np.float32(nrm)
        if nrm > 0.0:
            inv_nrm_f32 = np.float32(1.0 / nrm)
            for d in range(D):
                out_row[d] *= inv_nrm_f32
        else:
            prev_row = prev_mu_LD[l]
            for d in range(D):
                out_row[d] = prev_row[d]


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy')
def _inter_subject_var_epsil_update(
    psi_SLD,        # (S, L, D) fp32 C-contig
    mu_new_LD,      # (L, D)    fp32 C-contig
    prev_eps_L,     # (L,)      fp32 — fallback for invad(inf)
    epsil_new_L,    # (L,)      fp32 — OUT
    dim_f64,
    ini_val_f64,
):
    """ε update — serial over L (invad is the scalar root-find bottleneck)."""
    S = psi_SLD.shape[0]
    L = psi_SLD.shape[1]
    D = psi_SLD.shape[2]
    inv_S_f64 = 1.0 / np.float64(S)
    for l in range(L):
        acc = 0.0
        mu_row = mu_new_LD[l]
        for s in range(S):
            psi_row = psi_SLD[s, l]
            for d in range(D):
                acc += np.float64(psi_row[d]) * np.float64(mu_row[d])
        eps_in = acc * inv_S_f64
        if eps_in > 1.0:
            eps_in = 1.0
        v = invad_numba(dim_f64, eps_in)
        if v < ini_val_f64:
            v = ini_val_f64
        if not math.isfinite(v):
            v = np.float64(prev_eps_L[l])
        epsil_new_L[l] = np.float32(v)


# ─────────────────────────────────────────────────────────────────────
# intra_subject_var_loop — while-loop wrapping (s_psi, sigma) updates (L17).
#
# Per-iter math (in internal (S, L, D) / (S, T, L, D) layout):
#   ψ_update[s, l, d] = σ[l] · Σ_t ν[s, t, l, d] + ε[l] · μ[l, d]
#   col_norm[s, l]    = sqrt(Σ_d ψ_update²)
#   ψ_new[s, l, d]    = ψ_update[s, l, d] / col_norm[s, l]   (zero → 0)
#
#   cos_per_LS[l, s]  = Σ_d ψ_new · ψ_prev    (d unit-stride)
#   not_yet[s]        = (Σ_l 1{(1 - cos) < ε} < L)
#   flag_psi[s] = 1 where ~not_yet[s] (accumulates across iters)
#
#   σ_input[l]        = Σ_{s,t,d} ψ_new[s,l,d] · ν[s,t,l,d] / (S T)
#   σ_input           = min(σ_input, 1.0)
#   σ_new[l]          = invAd(dim, σ_input[l])  with ini_val / inf fallbacks
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _intra_subject_var_one_iter(
    nu_STLD,        # (S, T, L, D) fp32 C-contig
    psi_prev_SLD,   # (S, L, D)    fp32 C-contig — IN (prev iter's ψ)
    psi_new_SLD,    # (S, L, D)    fp32 C-contig — OUT (this iter's ψ)
    sigma_cur_L,    # (L,)         fp32 — IN (current σ, pre-update)
    eps_mu_LD,      # (L, D)       fp32 — IN (ε[l] · μ[l, d], precomputed)
    flag_psi_S,     # (S,)         int32 — IN/OUT (accumulating)
    cos_per_LS,     # (L, S)       fp32 — scratch
    eps_f32,        # fp32 scalar
):
    """ψ update + cosine drift + per-s flag_psi accumulate (this iter).

    Note: parallelizing on (l, s) means each (l, s) work-unit reads
    nu_STLD with a strided access pattern across (s, t) for a fixed l.
    Inner d loop is unit-stride, which is what matters for the bulk
    work (~85% of FLOPs).
    """
    S = nu_STLD.shape[0]
    T = nu_STLD.shape[1]
    L = nu_STLD.shape[2]
    D = nu_STLD.shape[3]
    one_f32 = np.float32(1.0)

    # Per-(s, l): build ψ_update column (d unit-stride), normalize, accumulate cosine.
    for sl in prange(S * L):
        s = sl // L
        l = sl - s * L
        sigma_l = sigma_cur_L[l]
        eps_mu_row = eps_mu_LD[l]
        new_row = psi_new_SLD[s, l]
        prev_row = psi_prev_SLD[s, l]

        # 1) Build ψ_update column, accumulate fp64 norm².
        acc_n2 = 0.0
        for d in range(D):
            nu_sum = np.float32(0.0)
            for t in range(T):
                nu_sum += nu_STLD[s, t, l, d]
            v = sigma_l * nu_sum + eps_mu_row[d]
            new_row[d] = v
            acc_n2 += np.float64(v) * np.float64(v)
        nrm = math.sqrt(acc_n2)
        if nrm > 0.0:
            inv_nrm = np.float32(1.0 / nrm)
            acc_c = np.float32(0.0)
            for d in range(D):
                vn = new_row[d] * inv_nrm
                new_row[d] = vn
                acc_c += vn * prev_row[d]
            cos_per_LS[l, s] = acc_c
        else:
            for d in range(D):
                new_row[d] = np.float32(0.0)
            cos_per_LS[l, s] = np.float32(0.0)

    # Per-subject flag_psi update.
    for s in range(S):
        n_conv = 0
        for l in range(L):
            if (one_f32 - cos_per_LS[l, s]) < eps_f32:
                n_conv += 1
        if n_conv >= L:
            flag_psi_S[s] = np.int32(1)


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy', parallel=True)
def _intra_subject_var_sigma_update(
    psi_new_SLD,    # (S, L, D)    fp32 — current ψ
    nu_STLD,        # (S, T, L, D) fp32
    sigma_cur_L,    # (L,)         fp32 — pre-update σ (inf-fallback)
    sigma_new_L,    # (L,)         fp32 — OUT
    dim_f64,
    ini_val_f64,
):
    """σ update via ``invAd(dim, min(Σ_{s,t,d} ψ·ν / (S·T), 1.0))``."""
    S = psi_new_SLD.shape[0]
    L = psi_new_SLD.shape[1]
    D = psi_new_SLD.shape[2]
    T = nu_STLD.shape[1]
    inv_ST_f64 = 1.0 / np.float64(S * T)

    for l in prange(L):
        acc = 0.0
        for s in range(S):
            psi_row = psi_new_SLD[s, l]
            for t in range(T):
                nu_row = nu_STLD[s, t, l]
                for d in range(D):
                    acc += np.float64(psi_row[d]) * np.float64(nu_row[d])
        sigma_in = acc * inv_ST_f64
        if sigma_in > 1.0:
            sigma_in = 1.0
        v = invad_numba(dim_f64, sigma_in)
        if v < ini_val_f64:
            v = ini_val_f64
        if not math.isfinite(v):
            v = np.float64(sigma_cur_L[l])
        sigma_new_L[l] = np.float32(v)


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model='numpy')
def _intra_subject_var_loop_kernel(
    nu_STLD,        # (S, T, L, D) fp32 C-contig
    psi_in_SLD,     # (S, L, D)    fp32 C-contig
    sigma_in_L,     # (L,)         fp32
    epsil_L,        # (L,)         fp32
    mu_LD,          # (L, D)       fp32
    # Outputs:
    psi_new_SLD,    # (S, L, D)    fp32 C-contig — OUT
    sigma_new_L,    # (L,)         fp32          — OUT
    flag_psi_S,     # (S,)         int32         — OUT (zero on entry)
    # Scratch (caller-owned, sized once):
    psi_prev_SLD,   # (S, L, D)    fp32 C-contig — scratch
    eps_mu_LD,      # (L, D)       fp32 — scratch
    sigma_cur_L,    # (L,)         fp32 — scratch
    sigma_tmp_L,    # (L,)         fp32 — scratch
    cos_per_LS,     # (L, S)       fp32 — scratch
    # Scalars:
    dim_f64,
    ini_val_f64,
    eps_f32,
    max_iter,
):
    """While-loop master for L17. flag_psi accumulates across iters
    (MATLAB never zeros it on a new iter — once a subject converges
    its bit stays set). Convergence requires both
    ``flag_psi.sum() == S`` AND ``mean_l |σ_cur - σ_new| / σ_cur < ε``.
    """
    S = nu_STLD.shape[0]
    L = nu_STLD.shape[2]
    D = nu_STLD.shape[3]

    # Initialize scratch.
    for s in range(S):
        for l in range(L):
            for d in range(D):
                psi_prev_SLD[s, l, d] = psi_in_SLD[s, l, d]
    for l in range(L):
        sigma_cur_L[l] = sigma_in_L[l]
    # eps_mu_LD: ε[l] · μ[l, d]
    for l in range(L):
        e = epsil_L[l]
        mu_row = mu_LD[l]
        out_row = eps_mu_LD[l]
        for d in range(D):
            out_row[d] = e * mu_row[d]
    for s in range(S):
        flag_psi_S[s] = np.int32(0)

    eps_f64 = np.float64(eps_f32)

    iter_used = 0
    for iter_idx in range(max_iter):
        iter_used = iter_idx + 1
        # 1) ψ update + flag accumulate.
        _intra_subject_var_one_iter(
            nu_STLD, psi_prev_SLD, psi_new_SLD,
            sigma_cur_L, eps_mu_LD, flag_psi_S, cos_per_LS, eps_f32,
        )

        # Slide window: psi_prev ← psi_new for next iter.
        for s in range(S):
            for l in range(L):
                for d in range(D):
                    psi_prev_SLD[s, l, d] = psi_new_SLD[s, l, d]

        # 2) σ update.
        _intra_subject_var_sigma_update(
            psi_new_SLD, nu_STLD, sigma_cur_L, sigma_tmp_L,
            dim_f64, ini_val_f64,
        )

        # 3) convergence check.
        flag_sum = 0
        for s in range(S):
            flag_sum += flag_psi_S[s]
        # Invariant: sigma_cur_L[l] >= ini_val > 0 (enforced upstream —
        # _intra_subject_var_sigma_update clamps every l to ini_val on
        # entry, and iter-1 inherits sigma_in_L from Step2Pipeline's
        # ini_val * np.ones(L)). Division is therefore well-defined; no
        # denominator floor needed.
        rel_acc = 0.0
        for l in range(L):
            rel_acc += abs(np.float64(sigma_cur_L[l] - sigma_tmp_L[l])
                           / np.float64(sigma_cur_L[l]))
        rel_mean = rel_acc / np.float64(L)
        for l in range(L):
            sigma_cur_L[l] = sigma_tmp_L[l]
        for l in range(L):
            sigma_new_L[l] = sigma_cur_L[l]

        if flag_sum == S and rel_mean < eps_f64:
            break

    return iter_used


# ─────────────────────────────────────────────────────────────────────
# Reset operations — numba in-place writes.
#
# ``reset_kappa_uniform``: kappa[:] = ini_val
# ``reset_s_t_nu_from_mtc``: broadcast mtc (L, D) into s_t_nu_STLD across (s, t).
# ``reset_s_psi_from_mtc``: broadcast mtc (L, D) into s_psi_SLD across s.
# ──────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def reset_s_t_nu_from_mtc_STLD(s_t_nu_STLD, mtc_LD):
    """Broadcast ``mtc_LD`` into ``s_t_nu_STLD`` across (s, t). In-place."""
    S = s_t_nu_STLD.shape[0]
    T = s_t_nu_STLD.shape[1]
    L = s_t_nu_STLD.shape[2]
    D = s_t_nu_STLD.shape[3]
    STL = S * T * L
    for stl in prange(STL):
        s = stl // (T * L)
        rem = stl - s * (T * L)
        t = rem // L
        l = rem - t * L
        mtc_row = mtc_LD[l]
        out_row = s_t_nu_STLD[s, t, l]
        for d in range(D):
            out_row[d] = mtc_row[d]


@njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def reset_s_psi_from_mtc_SLD(s_psi_SLD, mtc_LD):
    """Broadcast ``mtc_LD`` into ``s_psi_SLD`` across s. In-place."""
    S = s_psi_SLD.shape[0]
    L = s_psi_SLD.shape[1]
    D = s_psi_SLD.shape[2]
    for sl in prange(S * L):
        s = sl // L
        l = sl - s * L
        mtc_row = mtc_LD[l]
        out_row = s_psi_SLD[s, l]
        for d in range(D):
            out_row[d] = mtc_row[d]


# ─────────────────────────────────────────────────────────────────────
# Warmup — JIT-compile every kernel once at process start.
# ─────────────────────────────────────────────────────────────────────
def warmup_step2_em_outer() -> None:
    """JIT-compile all kernels with tiny dummy inputs."""
    S, T, L, D = 2, 2, 4, 4
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((S, L, D)).astype(np.float32)
    nu = rng.standard_normal((S, T, L, D)).astype(np.float32)
    mu = rng.standard_normal((L, D)).astype(np.float32)
    sigma = np.full(L, 50.0, dtype=np.float32)
    epsil = np.full(L, 50.0, dtype=np.float32)
    cost_em = np.zeros(S, dtype=np.float64)

    _intra_em_cost_step2_kernel(psi, nu, mu, sigma, epsil, cost_em, float(D))

    mu_new = np.empty((L, D), dtype=np.float32)
    mu_norms = np.empty(L, dtype=np.float32)
    _inter_subject_var_mu_update(psi, mu, mu_new, mu_norms)

    eps_new = np.empty(L, dtype=np.float32)
    _inter_subject_var_epsil_update(
        psi, mu_new, epsil, eps_new, float(D), 50.0,
    )

    psi_new = np.empty_like(psi)
    sigma_new = np.empty(L, dtype=np.float32)
    flag = np.zeros(S, dtype=np.int32)
    psi_prev = np.empty_like(psi)
    eps_mu = np.empty((L, D), dtype=np.float32)
    sigma_cur = np.empty(L, dtype=np.float32)
    sigma_tmp = np.empty(L, dtype=np.float32)
    cos_per_LS = np.empty((L, S), dtype=np.float32)
    _intra_subject_var_loop_kernel(
        nu, psi, sigma, epsil, mu,
        psi_new, sigma_new, flag,
        psi_prev, eps_mu, sigma_cur, sigma_tmp, cos_per_LS,
        float(D), 50.0, np.float32(1e-4), 3,
    )

    mtc_LD = rng.standard_normal((L, D)).astype(np.float32)
    s_t_nu_buf = np.empty((S, T, L, D), dtype=np.float32)
    s_psi_buf = np.empty((S, L, D), dtype=np.float32)
    reset_s_t_nu_from_mtc_STLD(s_t_nu_buf, mtc_LD)
    reset_s_psi_from_mtc_SLD(s_psi_buf, mtc_LD)
