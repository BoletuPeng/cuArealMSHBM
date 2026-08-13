"""_kernels.py

Numba kernels for the EM stop-criterion block. Single-core, out-buffer
style (caller pre-allocates outputs).

Public API:
    log_with_neginf_floor
        — ``log(x)`` with ``log(0) → log(eps_f64^20)``. Used once per
          Session to build ``log(theta)`` at construction.
    fused_em_stop_assemble_cleanup_cost_f32
        — single-pass fused EM-stop block; computes ``update_cost`` and
          writes back the cleaned ``spatial_connect_vmf`` in one sweep
          over (N, L).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np


# log(eps_f64 ^ 20) where eps_f64 = 2^-52. ≈ -720.873.
LOG_EPS_POW20 = np.float64(20.0 * np.log(np.finfo(np.float64).eps))


@nb.njit(cache=True, fastmath=False, boundscheck=False, error_model="numpy")
def log_with_neginf_floor(x, out):
    """``out[n, k] = log(x[n, k])`` with ``log(0) -> LOG_EPS_POW20``.

    MATLAB equivalent: ``y = log(x); y(isinf(y)) = log(eps^20)``. theta
    and s_lambda are non-negative in practice, so the only entries hit
    are exact zeros; non-positive inputs uniformly take the floor.
    """
    N, L = x.shape
    floor_f32 = np.float32(LOG_EPS_POW20)
    for n in range(N):
        for k in range(L):
            xnk = x[n, k]
            if xnk <= np.float32(0.0):
                out[n, k] = floor_f32
            else:
                out[n, k] = np.float32(np.log(xnk))


@nb.njit(cache=True, fastmath=False, boundscheck=False, error_model="numpy")
def fused_em_stop_assemble_cleanup_cost_f32(
    acc, kappa, cdln_per_k, T,
    s_lambda, log_theta_cost, V_temp,
    scv_in, scv_clean_out,
    w, c, beta, out_scalar,
):
    """Single-pass fused EM-stop block.

    Per (n, l):
        llp_nl     = T * cdln[l] + kappa[l] * acc[n, l]
        scv_clean  = scv_in[n, l] if isfinite, else LOG_EPS_POW20
        scv_clean_out[n, l] = scv_clean
        log_sl     = log(s_lambda[n, l]) with floor at LOG_EPS_POW20
        term       = sl * (llp_nl + log_theta_cost - w*log_sl
                           - c*V_temp + beta[l]*scv_clean)
        acc_total += term

    No (N, L) intermediate is materialized for ``log_lambda_prop`` —
    the value is computed and consumed entirely within the inner
    loop. The cleaned spatial-connect VMF IS materialized into
    ``scv_clean_out`` because the super-call propagates the cleanup
    back into outer state (mirrors MATLAB's in-place mutation of
    ``Params.spatial_connect_vmf``); we still get the savings from
    one fused pass over scv instead of two.

    Reduction order: per-row fp64 accumulator, then summed into outer
    fp64 ``acc``.

    Inputs:
        acc                 : (N, L) fp32 — X @ s_t_nu sgemm output
        kappa               : (L,)   fp32
        cdln_per_k          : (L,)   fp32
        T                   : int    — number of sessions
        s_lambda            : (N, L) fp32
        log_theta_cost      : (N, L) fp32 — pre-floored log(theta)
        V_temp              : (N, L) fp32 — last lambda iter's V_temp
        scv_in              : (N, L) fp32 — raw spatial_connect_vmf
        w, c                : fp32 scalars
        beta                : (L,)   fp32
    Outputs (caller-provided):
        scv_clean_out       : (N, L) fp32 — scv_in with NaN/Inf -> floor
        out_scalar          : (1,)   fp64 — total cost
    """
    N, L = s_lambda.shape
    Tf = np.float32(T)
    w_f = np.float64(w)
    c_f = np.float64(c)
    floor_f64 = LOG_EPS_POW20
    floor_f32 = np.float32(LOG_EPS_POW20)
    acc_total = np.float64(0.0)
    for n in range(N):
        row_acc = np.float64(0.0)
        for l in range(L):
            # Inline assemble: log_lambda_prop without materialization.
            llp_nl = Tf * cdln_per_k[l] + kappa[l] * acc[n, l]
            # Inline cleanup: write back the cleaned scv for the caller.
            scv_v = scv_in[n, l]
            if not math.isfinite(scv_v):
                scv_v = floor_f32
            scv_clean_out[n, l] = scv_v
            # Inline log(s_lambda) with floor.
            sl_f32 = s_lambda[n, l]
            sl = np.float64(sl_f32)
            if sl_f32 <= np.float32(0.0):
                log_sl = floor_f64
            else:
                log_sl = np.log(sl)
            term_per_nl = (
                np.float64(llp_nl)
                + np.float64(log_theta_cost[n, l])
                - w_f * log_sl
                - c_f * np.float64(V_temp[n, l])
                + np.float64(beta[l]) * np.float64(scv_v)
            )
            row_acc += sl * term_per_nl
        acc_total += row_acc
    out_scalar[0] = acc_total


def warmup() -> None:
    """Compile both kernels with realistic dtypes/shapes once."""
    N, L, T = 4, 6, 3

    x = np.array([[0.0, 0.5, 0.1, 0.2, 0.3, 0.0],
                  [0.4, 0.0, 0.6, 0.0, 0.7, 0.8],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.9, 0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    log_x = np.empty_like(x)
    log_with_neginf_floor(x, log_x)

    acc = np.full((N, L), 0.5, dtype=np.float32)
    kappa = np.full(L, 1000.0, dtype=np.float32)
    cdln_per_k = np.full(L, -10.0, dtype=np.float32)
    s_lam = np.full((N, L), 0.1, dtype=np.float32)
    s_lam[0, 0] = 0.0  # warm the inline log_with_floor branch
    lthc = np.full((N, L), -1.0, dtype=np.float32)
    Vt = np.full((N, L), 0.3, dtype=np.float32)
    vmf = np.array([[1.0, np.nan, np.inf, -np.inf, 2.0, 3.0],
                    [-np.inf, 4.0, 5.0, np.nan, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                    [14.0, np.inf, -np.inf, np.nan, 15.0, 16.0]], dtype=np.float32)
    scv_clean = np.empty_like(vmf)
    beta = np.full(L, 5000.0, dtype=np.float32)
    out_scalar = np.empty(1, dtype=np.float64)
    fused_em_stop_assemble_cleanup_cost_f32(
        acc, kappa, cdln_per_k, T,
        s_lam, lthc, Vt, vmf, scv_clean,
        50.0, 10.0, beta, out_scalar,
    )
