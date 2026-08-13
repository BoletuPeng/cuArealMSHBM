"""_kernels.py

Numba kernels for the E-step λ-loop body. Single-core serial @njit,
out-buffer style, fp32 throughout (fp64 only for the convergence-drift
accumulator).

Per-call buffer shapes (sub-001, fsaverage6, 6 sessions, 300 ROIs):
    N = 81924    — total vertices (bilateral, medial wall included)
    D = 1175     — BOLD time-series length per session
    T = 6        — sessions
    L = 300      — clusters
    M_active     — N rows where theta sums > 0 (here 74947 ≤ N).

Kernels:

| Kernel                                                | Role |
|-------------------------------------------------------|------|
| ``compute_log_theta_with_neginf_cap_f32``             | one-shot at Session __init__: log(θ) with -Inf at θ==0 |
| ``compute_col_zero_mask_f32``                         | column-zero scan on ``acc`` (skip Cdln add for all-zero columns) |
| ``fused_v_lambda_assemble_softmax_drift_f32``         | hot path: V_lambda gather + assemble + softmax + convergence-drift in one row-streaming pass via L-sized L1-resident row buffer. |

All `@njit(cache=True, fastmath=False, boundscheck=False)`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, boolean, float32, float64, int64, void


# ─────────────────────────────────────────────────────────────────────────
# Session-cached one-time kernels
# ─────────────────────────────────────────────────────────────────────────
@njit(void(float32[:, ::1], float32[:, ::1]),
      cache=True, fastmath=False, boundscheck=False)
def compute_log_theta_with_neginf_cap_f32(theta, out):
    """``out[n, l] = log(theta[n, l])`` with ``log(0) -> -inf``.

    MATLAB stores theta as fp32. log(0) in fp32 → -Inf, which propagates
    through the assembled log_vmf and gets caught by the boundary mask /
    softmax row-sum cleanup downstream. Same semantics as MATLAB's
    ``log(theta)`` directly.

    Buffers
    -------
    theta : (N, L) fp32 C-contig — input.
    out   : (N, L) fp32 C-contig — output.
    """
    N, L = theta.shape
    for n in range(N):
        for l in range(L):
            v = theta[n, l]
            if v <= float32(0.0):
                out[n, l] = float32(-math.inf)
            else:
                out[n, l] = float32(math.log(v))


# ─────────────────────────────────────────────────────────────────────────
# Per-iter kernels
# ─────────────────────────────────────────────────────────────────────────
@njit(void(float32[:, ::1], boolean[::1]),
      cache=True, fastmath=False, boundscheck=False)
def compute_col_zero_mask_f32(acc, out_mask):
    """``out_mask[l] = all(acc[:, l] == 0)``.

    Buffers
    -------
    acc      : (N, L) fp32 C-contig — input.
    out_mask : (L,)   bool          — output. True if column is all-zero.

    Semantics — INTENTIONAL DIVERGENCE FROM MATLAB
    ----------------------------------------------
    MATLAB step3 line 587 reads:

        log_vmf(:, sum(log_vmf == 0, 1) == 0) = ...   % LxNxSxT shape

    With ``log_vmf`` as ``(L, N, S, T)``, ``sum(log_vmf == 0, 1)`` reduces
    over the L axis, so the filter is **per-(N, S, T)**: a cell is
    "Cdln-eligible" iff its entire L stripe is non-zero. One dead parcel
    ``l*`` makes every (n, s, t) ineligible (because the ``l*`` slot
    zeros out their L stripes), so MATLAB drops Cdln for ALL cells as
    soon as a single parcel goes dead — even live parcels lose their
    Cdln correction.

    This kernel uses a **per-L** mask: a column is "Cdln-eligible" iff
    its entire N stripe is non-zero. Dead parcels lose Cdln; live
    parcels keep it. This is the dimensionally-correct form (Cdln is a
    per-l constant; the partition function correction must be applied
    per-l). The MATLAB form is most likely an axis confusion in the
    original script.

    Equivalence on production data
    ------------------------------
    On sub-001..006, every parcel keeps a non-zero ``s_t_nu`` after the
    M-step (the ``σ·s_psi`` prior pull guarantees positive entries), so
    no parcel ever goes dead and no column of ``acc`` is identically
    zero. In that regime ``out_mask`` is identically False, MATLAB's
    ``idx`` is identically False, and the two forms produce identical
    output. The semantic divergence only manifests on degenerate inputs
    (e.g., heavily masked data with kappa→0); the MATLAB GT does not
    cover that regime, and we have not seen it in 6 production subjects.
    """
    N, L = acc.shape
    for l in range(L):
        all_zero = True
        for n in range(N):
            if acc[n, l] != float32(0.0):
                all_zero = False
                break
        out_mask[l] = all_zero


# ─────────────────────────────────────────────────────────────────────────
# Mega-fused: V_lambda gather + assemble + softmax + drift, in one
# row-streaming pass.
#
# (1) V_lambda gather (no separate splat). The fused kernel reads
#     V_lambda_active directly via ``inv_active_idx[n]`` (m if row n is
#     theta-active, -1 otherwise). V_temp_out is still written for
#     em_stop_criterion's final-iter consumption.
#
# (2) Boundary-mask early-skip in the exp() pass. ``bm == 0`` ⊂
#     ``theta == 0`` ⊂ ``log_theta == -Inf``, so log_vmf is -Inf at
#     those cells and ``exp(-Inf - rmax) * 0 = 0``. Skipping writes 0
#     directly without calling exp() — identical math, fewer libm calls.
#     ``bm == 0`` is dense in practice (each vertex has only ~7
#     reachable parcels out of L=300).
# ─────────────────────────────────────────────────────────────────────────
@njit(float64(
    float32[:, ::1],   # acc (N, L) fp32
    float32[::1],      # kappa (L,) fp32
    float32[::1],      # cdln_T (L,) fp32
    boolean[::1],      # col_zero_mask (L,)
    float32[:, ::1],   # log_theta (N, L) fp32
    float32,           # w
    float32[:, ::1],   # V_lambda_active (M_active, L) fp32 — direct, no splat
    int64[::1],        # inv_active_idx (N,) — m if active, -1 if inactive
    float32,           # c
    float32[::1],      # beta (L,) fp32
    float32[:, ::1],   # spatial_connect_vmf (N, L) fp32
    float32[:, ::1],   # spatial_xyz_vmf (N, L) fp32
    float32[:, ::1],   # boundary_mask (N, L) fp32
    float32[:, ::1],   # s_lambda_curr (N, L) fp32
    float32[::1],      # row_buf (L,) fp32 scratch
    float32[:, ::1],   # V_temp_out (N, L) fp32 — written for em_stop
    float32[:, ::1],   # out (N, L) fp32 — s_lambda_new
), cache=True, fastmath=False, boundscheck=False)
def fused_v_lambda_assemble_softmax_drift_f32(
    acc, kappa, cdln_T, col_zero_mask,
    log_theta, w,
    V_lambda_active, inv_active_idx,
    c,
    beta, spatial_connect_vmf, spatial_xyz_vmf,
    boundary_mask,
    s_lambda_curr,
    row_buf,
    V_temp_out,
    out,
):
    """Mega-fused: V_lambda gather + assemble + softmax + drift.

    Per-row body is three passes through the L1-resident ``row_buf``:

      A. Gather V_temp via ``inv_active_idx[n]`` indirection (skipping
         splat materialization), assemble log_vmf into ``row_buf``,
         write V_temp_out for em_stop, track rmax.
      B. ``exp(row_buf - rmax) * bmask`` with early-skip at bm == 0.
      C. Normalize → out, accumulate fp64 drift.

    The active vs inactive branch is hoisted OUT of the inner-l loop
    (``is_active = m >= 0`` is row-invariant) so LLVM can vectorize
    each branch's inner loop independently.

    Returns ``mean(|out - s_lambda_curr|)`` as fp64.

    Math invariant: ``bm == 0 ⊂ theta == 0 ⊂ log_theta == -Inf``, so
    log_vmf is -Inf at those cells and ``exp(-Inf - rmax) * 0 = 0``.
    The early-skip writes 0 directly — identical numerical outcome
    without the libm exp() call. ``rmax`` is taken over all cells
    (-Inf < anything, so it doesn't shift).
    """
    N, L = acc.shape
    two_c = float32(2.0) * c
    # drift accumulator MUST stay fp64: an fp32 reduction over (N, L)
    # drifts ~7e-5 absolute, just above the ε=1e-4 convergence threshold
    # (see e_step_lambda.py module docstring). Do not narrow to fp32.
    drift_acc = float64(0.0)
    inv_NL = float64(1.0) / float64(N * L)

    for n in range(N):
        m = inv_active_idx[n]
        rmax = float32(-math.inf)

        # ── Pass A: assemble + V_temp gather + rmax ──
        if m >= 0:
            # Active row: V_temp comes from V_lambda_active[m, :]
            for l in range(L):
                cdln_l = float32(0.0) if col_zero_mask[l] else cdln_T[l]
                vt = V_lambda_active[m, l]
                V_temp_out[n, l] = vt
                v = acc[n, l] * kappa[l] + cdln_l
                v += w * log_theta[n, l]
                v -= two_c * vt
                v += beta[l] * spatial_connect_vmf[n, l]
                v += spatial_xyz_vmf[n, l]
                row_buf[l] = v
                if v > rmax:
                    rmax = v
        else:
            # Inactive row: V_temp = 0 (skip the subtraction term entirely)
            for l in range(L):
                cdln_l = float32(0.0) if col_zero_mask[l] else cdln_T[l]
                V_temp_out[n, l] = float32(0.0)
                v = acc[n, l] * kappa[l] + cdln_l
                v += w * log_theta[n, l]
                v += beta[l] * spatial_connect_vmf[n, l]
                v += spatial_xyz_vmf[n, l]
                row_buf[l] = v
                if v > rmax:
                    rmax = v

        # ── Pass B: exp(row_buf - rmax) * bmask with early-skip ──
        # ~97.61% of cells have bm == 0 on sub-001 → skipped.
        rsum = float32(0.0)
        for l in range(L):
            bm = boundary_mask[n, l]
            if bm == float32(0.0):
                row_buf[l] = float32(0.0)
                continue
            ev = float32(math.exp(row_buf[l] - rmax))
            ev *= bm
            row_buf[l] = ev
            rsum += ev

        # ── Pass C: normalize → out, accumulate fp64 drift ──
        if rsum > float32(0.0) and rsum == rsum:
            inv_rsum = float32(1.0) / rsum
            for l in range(L):
                if col_zero_mask[l]:
                    new_val = float32(0.0)
                else:
                    new_val = row_buf[l] * inv_rsum
                out[n, l] = new_val
                d = float64(new_val) - float64(s_lambda_curr[n, l])
                if d < float64(0.0):
                    d = -d
                drift_acc += d
        else:
            # row sum is 0 or NaN — softmax degenerate; out row is zero.
            # s_lambda_curr is post-softmax (>= 0), so |0 - curr| = curr.
            for l in range(L):
                drift_acc += float64(s_lambda_curr[n, l])
                out[n, l] = float32(0.0)

    return drift_acc * inv_NL


# ─────────────────────────────────────────────────────────────────────────
# Warmup — pre-compile every kernel with realistic dtypes.
# ─────────────────────────────────────────────────────────────────────────
def warmup() -> None:
    """JIT-compile every kernel with sub-001-shape inputs.

    Cheap (~30 ms one-shot). Recommended at process startup so the first
    real call doesn't pay JIT cost.
    """
    N, L, M = 16, 8, 6
    theta = np.ones((N, L), dtype=np.float32)
    out_NL_a = np.empty((N, L), dtype=np.float32)
    out_NL_c = np.empty((N, L), dtype=np.float32)
    out_NL_d = np.empty((N, L), dtype=np.float32)
    out_NL_f = np.empty((N, L), dtype=np.float32)
    mask_L = np.zeros(L, dtype=np.bool_)
    kappa = np.full(L, 0.5, dtype=np.float32)
    cdln_T = np.zeros(L, dtype=np.float32)
    beta = np.zeros(L, dtype=np.float32)
    bmask = np.ones((N, L), dtype=np.float32)
    row_buf = np.empty(L, dtype=np.float32)

    compute_log_theta_with_neginf_cap_f32(theta, out_NL_a)
    compute_col_zero_mask_f32(theta, mask_L)
    # Mega-fused: V_lambda gather + assemble + softmax + drift.
    inv_active_idx = np.full(N, -1, dtype=np.int64)
    inv_active_idx[:M] = np.arange(M, dtype=np.int64)   # first M rows are "active"
    V_lambda_active_warm = np.zeros((M, L), dtype=np.float32)
    V_temp_out_warm = np.empty((N, L), dtype=np.float32)
    _ = fused_v_lambda_assemble_softmax_drift_f32(
        theta, kappa, cdln_T, mask_L,
        out_NL_a, np.float32(50.0),
        V_lambda_active_warm, inv_active_idx, np.float32(10.0),
        beta, out_NL_c, out_NL_d,
        bmask,
        out_NL_a,    # s_lambda_curr
        row_buf,
        V_temp_out_warm,
        out_NL_f,
    )
