"""_kernels_gpu.py

GPU kernels for the E-step λ-loop body. CuPy ports of the two hot
kernels in :mod:`arealmshbm.vmf_clustering._kernels` and
:mod:`arealmshbm.V_lambda._kernels`. All inputs and outputs are
``cupy.ndarray``; no PCIe transfers inside these functions.

Public API:
    vlambda_potts_closeform_fused_cupy
        — CuPy port of ``v_lambda_potts_closeform_fused_f32``.
          Fancy-indexing for the per-candidate gather.
    fused_v_lambda_assemble_softmax_drift_cupy
        — CuPy port of ``fused_v_lambda_assemble_softmax_drift_f32``.

Both kernels are mathematically identical to the CPU versions but not
bit-equal: CuPy's parallel reduction trees vs the CPU's serial
accumulators give fp32 ULP-level drift on output magnitudes ~O(1). The
≥99% vertex-agreement spec absorbs this the same way it absorbs the
CPU-vs-MATLAB drift.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import cupy as cp


# ─────────────────────────────────────────────────────────────────────────
# V_lambda Potts close-form (3-phase fused, GPU)
# ─────────────────────────────────────────────────────────────────────────
def vlambda_potts_closeform_fused_cupy(
    neighborhood_NM: cp.ndarray,    # (N, M1) int64
    lam: cp.ndarray,                # (N, K)  fp32
    row_idx: cp.ndarray,            # (P,)    int64
    col_idx: cp.ndarray,            # (P,)    int64
    V_lam: cp.ndarray,              # (N, K)  fp32 — pre-zeroed at construction
) -> None:
    """Fused 3-phase V_lambda Potts close-form, all on GPU.

    Mirrors :func:`arealmshbm.V_lambda._kernels.v_lambda_potts_closeform_fused_f32`.

    Phase 1 (per-vertex row-sum):
        row_sum[m] = Σ_k lam[m, k]
    Phase 2 (per-vertex neighbor-sum, j>0 mask):
        nbr_sum[m] = Σ_n: j>0 row_sum[j-1]
    Phase 3 (per-candidate close-form):
        V_lam[m, k] = nbr_sum[m] - Σ_n: j>0 lam[j-1, k]

    All three phases run as CuPy primitives. Sub-001 timing on RTX 5090:
    ~1 ms / call (vs ~32 ms CPU; ~32× speedup).

    Inputs/outputs are CuPy device arrays; no host-device transfer.
    Caller pre-zeros ``V_lam`` once at construction (static-candidate-set
    invariant).
    """
    N, M1 = neighborhood_NM.shape
    # Phase 1
    row_sum = lam.sum(axis=1)                                 # (N,) fp32

    # Phase 2 — gather neighbors' row_sums with j>0 mask.
    valid = neighborhood_NM > 0                                # (N, M1) bool
    safe_j = cp.where(valid, neighborhood_NM - 1, 0)           # (N, M1) int64
    # row_sum[safe_j] is (N, M1); zero out invalid entries via valid mask.
    nbr_vals = cp.where(valid, row_sum[safe_j], cp.float32(0.0))
    nbr_sum = nbr_vals.sum(axis=1)                            # (N,) fp32

    # Phase 3 — per-candidate gather + reduce.
    nbh_at_p = neighborhood_NM[row_idx]                        # (P, M1) int64
    valid_p = nbh_at_p > 0                                     # (P, M1) bool
    safe_j_p = cp.where(valid_p, nbh_at_p - 1, 0)              # (P, M1) int64
    # lam[safe_j_p, col_idx[:, None]]: gather (P, M1) values from lam.
    lam_vals = lam[safe_j_p, col_idx[:, None]]                 # (P, M1) fp32
    acc_lam = cp.where(valid_p, lam_vals, cp.float32(0.0)).sum(axis=1)  # (P,)
    # Scatter into V_lam at (row_idx, col_idx).
    V_lam[row_idx, col_idx] = nbr_sum[row_idx] - acc_lam


# ─────────────────────────────────────────────────────────────────────────
# Mega-fused: assemble + softmax + drift (GPU)
# ─────────────────────────────────────────────────────────────────────────
def fused_v_lambda_assemble_softmax_drift_cupy(
    acc: cp.ndarray,                # (N, L) fp32
    kappa: cp.ndarray,              # (L,)   fp32
    cdln_T: cp.ndarray,             # (L,)   fp32
    col_zero_mask: cp.ndarray,      # (L,)   bool
    log_theta: cp.ndarray,          # (N, L) fp32
    w: float,
    V_lambda_active: cp.ndarray,    # (M_active, L) fp32
    inv_active_idx: cp.ndarray,     # (N,) int64 (m if active, -1 if not)
    c: float,
    beta: cp.ndarray,               # (L,)   fp32
    spatial_connect_vmf: cp.ndarray,    # (N, L) fp32
    spatial_xyz_vmf: cp.ndarray,        # (N, L) fp32
    boundary_mask: cp.ndarray,          # (N, L) fp32
    s_lambda_curr: cp.ndarray,          # (N, L) fp32
    V_temp_out: cp.ndarray,             # (N, L) fp32 — written for em_stop
    out: cp.ndarray,                    # (N, L) fp32 — s_lambda_new
) -> float:
    """GPU equivalent of ``fused_v_lambda_assemble_softmax_drift_f32``.

    Returns the fp32-mean drift ``mean(|out - s_lambda_curr|)`` as a
    Python float (synchronized off-device).

    Implementation uses CuPy primitives in a chain that mirrors the
    CPU kernel's three-pass per-row body but with all (N, L) buffers
    materialized between steps; bandwidth-bound. A hand-written
    ``cupy.RawKernel`` collapsing the intermediates would be faster
    if this kernel ever shows up as the dominant hot spot.

    Math:
      Pass A:
          V_temp[n, l] = V_lambda_active[inv_active_idx[n], l] if active
                          else 0
          log_vmf[n, l] = acc[n, l] * kappa[l]
                          + (cdln_T[l] if not col_zero_mask[l] else 0)
                          + w * log_theta[n, l]
                          - 2c * V_temp[n, l]
                          + beta[l] * scv[n, l]
                          + sxv[n, l]
          rmax[n] = max_l log_vmf[n, l]
      Pass B (with bm==0 early-skip baked in via mask):
          ev[n, l] = exp(log_vmf[n, l] - rmax[n]) * boundary_mask[n, l]
          rsum[n] = Σ_l ev[n, l]
      Pass C:
          out[n, l] = ev[n, l] / rsum[n]   (or 0 if rsum<=0/NaN/col_zero)
          drift = mean_{n, l} |out[n, l] - s_lambda_curr[n, l]|

    The bm==0 early-skip in CPU's Pass B is replaced here by
    multiplication by ``boundary_mask`` after exp(); mathematically
    identical (exp(...)*0 = 0). The libm exp() floor doesn't apply on
    GPU — exp is a hardware intrinsic — so the skip wouldn't help speed
    even if we did it.
    """
    two_c = cp.float32(2.0 * c)
    w_f32 = cp.float32(w)

    # Pass A.0 — gather V_temp from V_lambda_active via inv_active_idx.
    # For inactive rows (idx == -1) we want V_temp[n, :] = 0; do this via
    # safe-indexing with mask.
    active = inv_active_idx >= 0                              # (N,) bool
    safe_idx = cp.where(active, inv_active_idx, 0)            # (N,) int64
    V_temp = cp.where(active[:, None],
                       V_lambda_active[safe_idx],
                       cp.float32(0.0))                       # (N, L) fp32
    cp.copyto(V_temp_out, V_temp)

    # Pass A.1 — assemble log_vmf.
    cdln_eff = cp.where(col_zero_mask, cp.float32(0.0), cdln_T)  # (L,) fp32
    log_vmf = (acc * kappa
               + cdln_eff
               + w_f32 * log_theta
               - two_c * V_temp
               + beta * spatial_connect_vmf
               + spatial_xyz_vmf)                             # (N, L) fp32

    # Pass B — softmax with boundary mask.
    rmax = log_vmf.max(axis=1, keepdims=True)                 # (N, 1)
    ev = cp.exp(log_vmf - rmax) * boundary_mask                # (N, L)
    rsum = ev.sum(axis=1, keepdims=True)                       # (N, 1)

    # Pass C — normalize + drift, with rsum>0 / col_zero cleanup.
    safe_rsum = cp.where(rsum > 0, rsum, cp.float32(1.0))
    new_val = cp.where(rsum > 0, ev / safe_rsum, cp.float32(0.0))
    new_val = cp.where(col_zero_mask, cp.float32(0.0), new_val)
    cp.copyto(out, new_val)

    drift = cp.abs(out - s_lambda_curr).mean()
    return float(drift)


# ─────────────────────────────────────────────────────────────────────────
# M-step kernels (GPU)
# ─────────────────────────────────────────────────────────────────────────
def mstep_X_dot_sl_TDL_cupy(
    data_series_NTD_dev: cp.ndarray,    # (N, T, D) fp32 on GPU
    s_lambda_NL_dev: cp.ndarray,        # (N, L)    fp32 on GPU
    out_TDL_dev: cp.ndarray,            # (T, D, L) fp32 on GPU
) -> None:
    """Per-t sgemm batch, GPU. Mirrors :func:`m_step._compute_X_dot_sl_TDL`.

    Uses ``data_series_NTD_dev[:, t, :]`` strided view per t — cuBLAS
    handles the lda fine (just like MKL on CPU). One ``cublasSgemm`` per
    t; could be batched via ``cublasSgemmStridedBatched`` for marginal
    speedup but the per-t loop is already ~24 ms on RTX 5090 for sub-001
    shapes.
    """
    N, T, D = data_series_NTD_dev.shape
    for t in range(T):
        X = data_series_NTD_dev[:, t, :]   # (N, D) strided view
        cp.matmul(X.T, s_lambda_NL_dev, out=out_TDL_dev[t])


def mstep_fused_iter_m_body_cupy(
    kappa_f32,                          # python float
    X_dot_sl_TDL_dev: cp.ndarray,       # (T, D, L) fp32
    sigma_psi_DL_dev: cp.ndarray,       # (D, L)    fp32
    s_t_nu_old_TDL_dev: cp.ndarray,     # (T, D, L) fp32
    s_t_nu_new_TDL_dev: cp.ndarray,     # (T, D, L) fp32 — output
    cos_TL_dev: cp.ndarray,             # (T, L)    fp32 — output
) -> None:
    """Per-iter_m M-step body — TDL-batch shape, GPU.

    Mirrors :func:`m_step._kernels.fused_lambda_X_normalize_TDL_f32`.

    For each (t, l):
        col[d]  = kappa * X_dot_sl[t, d, l] + sigma_psi[d, l]
        col_norm = sqrt(sum_d col[d]^2)
        s_t_nu_new[t, d, l] = col[d] / col_norm
        cos[t, l] = sum_d s_t_nu_new[t, d, l] * s_t_nu_old[t, d, l]

    All vectorized via cupy primitives. Empty parcels (col_norm == 0)
    yield NaN entries via 0 * inf — same as CPU and MATLAB.
    """
    # Pass 1: lambda_X (broadcast + add).
    lambda_X = cp.float32(kappa_f32) * X_dot_sl_TDL_dev + sigma_psi_DL_dev
    # col_norms over D axis: (T, L)
    col_norms_TL = cp.sqrt((lambda_X ** 2).sum(axis=1))
    # Normalize per (t, l): broadcast (T, 1, L)
    inv_norm = cp.float32(1.0) / col_norms_TL          # (T, L); inf at empty parcels
    cp.multiply(lambda_X, inv_norm[:, None, :], out=s_t_nu_new_TDL_dev)
    # Cosine: per (t, l), sum_d new * old
    cp.sum(s_t_nu_new_TDL_dev * s_t_nu_old_TDL_dev, axis=1, out=cos_TL_dev)


# ─────────────────────────────────────────────────────────────────────────
# spatial_xyz_prior kernel (GPU)
# ─────────────────────────────────────────────────────────────────────────
def spatial_xyz_compute_cupy(
    sphere_xyz_unit_lh_dev: cp.ndarray,       # (n_lh, 3) fp32
    sphere_xyz_unit_rh_dev: cp.ndarray,       # (n_lh, 3) fp32
    sphere_xyz_unit_full_dev: cp.ndarray,     # (N, 3)    fp32 — for step 3
    s_lambda_dev: cp.ndarray,                 # (N, L)    fp32
    xyz_gamma_f32_dev: cp.ndarray,            # (L,) fp32
    cdln_per_k_dev: cp.ndarray,               # (L,) fp32 (computed CPU-side via cdln_d3)
    n_lh: int, L_lh: int,
    s_muc_dev: cp.ndarray,                    # (3, L) fp32 — output
    spatial_xyz_vmf_dev: cp.ndarray,          # (N, L) fp32 — output
) -> None:
    """GPU spatial_xyz_prior. Mirrors :class:`spatial_priors.XyzSession.compute`.

    Block-diagonal LH-LH + RH-RH for steps 1+2. Step 3 + 5 are FULL
    sgemm/elementwise. Cdln(xyz_gamma, 3) is supplied by the caller
    (computed CPU-side via ``cdln_d3_to_f32``).
    """
    # Step 1: lambda_X = sphere.T @ s_lambda, block-diagonal.
    s_lam_lh = s_lambda_dev[:n_lh, :L_lh]
    s_lam_rh = s_lambda_dev[n_lh:, L_lh:]
    lambda_X_lh = sphere_xyz_unit_lh_dev.T @ s_lam_lh   # (3, L_lh)
    lambda_X_rh = sphere_xyz_unit_rh_dev.T @ s_lam_rh   # (3, L_lh)

    # Step 2: column-normalize per hemi.
    col_norms_lh = cp.sqrt((lambda_X_lh ** 2).sum(axis=0))   # (L_lh,)
    col_norms_rh = cp.sqrt((lambda_X_rh ** 2).sum(axis=0))
    s_muc_lh = lambda_X_lh / col_norms_lh                   # (3, L_lh)
    s_muc_rh = lambda_X_rh / col_norms_rh
    s_muc_dev[:, :L_lh] = s_muc_lh
    s_muc_dev[:, L_lh:] = s_muc_rh

    # Step 3: FULL sgemm sphere @ s_muc → vmf.
    cp.matmul(sphere_xyz_unit_full_dev, s_muc_dev, out=spatial_xyz_vmf_dev)

    # Step 5: assemble vmf = cdln + xyz_gamma * gamma_cos, then NaN → 0.
    cp.multiply(spatial_xyz_vmf_dev, xyz_gamma_f32_dev,
                 out=spatial_xyz_vmf_dev)
    cp.add(spatial_xyz_vmf_dev, cdln_per_k_dev, out=spatial_xyz_vmf_dev)
    finite = cp.isfinite(spatial_xyz_vmf_dev)
    spatial_xyz_vmf_dev[:] = cp.where(finite, spatial_xyz_vmf_dev,
                                        cp.float32(0.0))


# ─────────────────────────────────────────────────────────────────────────
# spatial_connect_prior kernel (GPU)
# ─────────────────────────────────────────────────────────────────────────
def spatial_connect_compute_cupy(
    grad_lh_dev: cp.ndarray,                  # (n_lh, D) fp32
    grad_rh_dev: cp.ndarray,                  # (n_lh, D) fp32
    grad_sq_norms_dev: cp.ndarray,            # (N,) fp32 — cached
    s_lambda_dev: cp.ndarray,                 # (N, L) fp32
    n_lh: int, L_lh: int, D: int,
    u_dev: cp.ndarray,                        # (L, D) fp32 — output
    spatial_connect_vmf_dev: cp.ndarray,      # (N, L) fp32 — output (cross-hemi pre-filled -Inf)
) -> None:
    """GPU spatial_connect_prior. Mirrors :class:`spatial_priors.ConnectSession.compute`.

    Block-diagonal sgemm + closed-form distance via the gemm trick.
    Cross-hemi cells of ``spatial_connect_vmf`` MUST be pre-filled with
    ``-Inf`` by the caller (constant across calls).
    """
    s_lam_lh = s_lambda_dev[:n_lh, :L_lh]
    s_lam_rh = s_lambda_dev[n_lh:, L_lh:]

    # Step 1a: u_update = grad.T @ s_lambda (block-diag).
    u_update_lh = grad_lh_dev.T @ s_lam_lh    # (D, L_lh)
    u_update_rh = grad_rh_dev.T @ s_lam_rh    # (D, L_lh)
    # Step 1b: sum_lambda per parcel + divide.
    sum_lam_lh = s_lam_lh.sum(axis=0)         # (L_lh,)
    sum_lam_rh = s_lam_rh.sum(axis=0)
    # u[k, d] = u_update[d, k] / sum_lam[k]
    u_dev[:L_lh] = (u_update_lh / sum_lam_lh).T
    u_dev[L_lh:] = (u_update_rh / sum_lam_rh).T

    # Step 2: cross = grad @ u.T (block-diag, written to vmf diagonal blocks).
    u_lh = u_dev[:L_lh]
    u_rh = u_dev[L_lh:]
    cp.matmul(grad_lh_dev, u_lh.T, out=spatial_connect_vmf_dev[:n_lh, :L_lh])
    cp.matmul(grad_rh_dev, u_rh.T, out=spatial_connect_vmf_dev[n_lh:, L_lh:])

    # Step 3: u_sq_norms.
    u_sq = (u_dev ** 2).sum(axis=1)           # (L,)

    # Step 4: assemble vmf = -||grad||^2 + 2*cross - ||u||^2.
    # Block-diagonal: only LH-LH and RH-RH cells. Cross-hemi cells stay -Inf
    # (pre-filled at session __init__).
    block_lh = spatial_connect_vmf_dev[:n_lh, :L_lh]
    block_rh = spatial_connect_vmf_dev[n_lh:, L_lh:]
    grad_sq_lh = grad_sq_norms_dev[:n_lh]
    grad_sq_rh = grad_sq_norms_dev[n_lh:]
    u_sq_lh = u_sq[:L_lh]
    u_sq_rh = u_sq[L_lh:]
    block_lh[:] = (cp.float32(2.0) * block_lh
                    - grad_sq_lh[:, None] - u_sq_lh)
    block_rh[:] = (cp.float32(2.0) * block_rh
                    - grad_sq_rh[:, None] - u_sq_rh)
    # NaN cleanup → -Inf (matches MATLAB)
    NEG_INF = cp.float32(-cp.inf)
    not_finite_lh = ~cp.isfinite(block_lh)
    not_finite_rh = ~cp.isfinite(block_rh)
    block_lh[:] = cp.where(not_finite_lh, NEG_INF, block_lh)
    block_rh[:] = cp.where(not_finite_rh, NEG_INF, block_rh)


def warmup() -> None:
    """JIT-compile every CuPy operation with realistic dtypes/shapes.

    CuPy compiles its ufuncs / reductions on first invocation per dtype
    combo. The calls below force compilation upfront so the first real
    call doesn't pay JIT cost. ~50 ms one-shot.
    """
    N, L, M_act, M1, P = 16, 8, 6, 6, 12
    nbh = cp.zeros((M_act, M1), dtype=cp.int64)
    nbh[0, 0] = 2
    lam = cp.full((M_act, L), 1.0 / L, dtype=cp.float32)
    rows = cp.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=cp.int64)
    cols = cp.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=cp.int64)
    V_lam = cp.zeros((M_act, L), dtype=cp.float32)
    vlambda_potts_closeform_fused_cupy(nbh, lam, rows, cols, V_lam)

    acc = cp.zeros((N, L), dtype=cp.float32)
    kappa = cp.ones(L, dtype=cp.float32)
    cdln_T = cp.zeros(L, dtype=cp.float32)
    col_zero = cp.zeros(L, dtype=cp.bool_)
    log_theta = cp.zeros((N, L), dtype=cp.float32)
    Vlam = cp.zeros((M_act, L), dtype=cp.float32)
    inv_idx = cp.full(N, -1, dtype=cp.int64)
    inv_idx[:M_act] = cp.arange(M_act, dtype=cp.int64)
    beta = cp.zeros(L, dtype=cp.float32)
    scv = cp.zeros((N, L), dtype=cp.float32)
    sxv = cp.zeros((N, L), dtype=cp.float32)
    bmask = cp.ones((N, L), dtype=cp.float32)
    s_lam = cp.zeros((N, L), dtype=cp.float32)
    V_temp_out = cp.empty((N, L), dtype=cp.float32)
    out = cp.empty((N, L), dtype=cp.float32)
    fused_v_lambda_assemble_softmax_drift_cupy(
        acc, kappa, cdln_T, col_zero, log_theta, 50.0,
        Vlam, inv_idx, 10.0, beta, scv, sxv, bmask,
        s_lam, V_temp_out, out,
    )
