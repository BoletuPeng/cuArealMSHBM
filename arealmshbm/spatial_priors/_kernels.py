"""_kernels.py

Numba kernels for the two spatial-prior assemble passes. Single-core,
out-buffer style; every kernel takes pre-allocated input + output
buffers, allocates nothing inside the JIT'd region, returns nothing.

Public API:
    assemble_xyz_vmf     — fused per-(n, l) write of
                           ``cdln_per_k[l] + xyz_gamma[l] * gamma_cos[n, l]``
                           with NaN→0 cleanup. Used by spatial_xyz_prior.
    assemble_connect_vmf — fused per-(n, l) write of
                           ``-||g_n - u_l||²`` via the gemm-trick, plus
                           cross-hemisphere -inf mask. Used by
                           spatial_connect_prior.
    (other small numeric helpers used by the two assemble passes.)

Buffer reference (Mode A, fsaverage6, 300 ROIs):
    N    = 81924  (bilateral vertex count)
    L    = 300    (clusters; LH parcels = L/2, RH = L/2)
    n_lh = N/2    (left-hemisphere vertex count)
    L_lh = L/2    (left-hemisphere parcel count)

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import numba as nb


# ─────────────────────────────────────────────────────────────────────────
# spatial_xyz_prior — final assembly + NaN cleanup
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False)
def assemble_xyz_vmf(vmf, cdln_per_k, xyz_gamma_f32):
    """In-place: ``vmf[n, k] = cdln_per_k[k] + xyz_gamma_f32[k] * vmf[n, k]``,
    then ``NaN → 0``.

    Mirrors MATLAB lines 428-429 of
    ``CBIG_ArealMSHBM_gMSHBM_generate_individual_parcellation.m``:

        Params.spatial_xyz_vmf = bsxfun(@plus, Cdln(xyz_gamma, 3),
                                        bsxfun(@times, xyz_gamma,
                                               spatial_xyz_vmf_tmp));
        Params.spatial_xyz_vmf(isnan(...)) = 0;

    NaN sources: (a) ``Cdln(0, 3) = NaN`` for inactive parcels (xyz_gamma=0);
    (b) ``s_muc[:, k] = NaN`` for empty parcels (lambda_X column = 0).

    Inputs / Outputs:
        vmf : (N, L) float32 — input contains ``gamma_cos[n, k]`` (the
            ``sphere @ s_muc`` cosine matrix); rewritten to the final
            spatial_xyz_vmf with NaN entries zeroed.
    Inputs:
        cdln_per_k    : (L,) float32 — ``Cdln(xyz_gamma[k], 3)``.
            NaN at k where xyz_gamma <= 0.
        xyz_gamma_f32 : (L,) float32 — concentration parameter; 0 for
            inactive parcels (post-cleanup zeros these).
    """
    N, L = vmf.shape
    for n in range(N):
        for k in range(L):
            v = cdln_per_k[k] + xyz_gamma_f32[k] * vmf[n, k]
            # NaN check via ``v != v`` (the standard IEEE-754 test).
            if v != v:
                vmf[n, k] = np.float32(0.0)
            else:
                vmf[n, k] = v


# ─────────────────────────────────────────────────────────────────────────
# spatial_connect_prior — final assembly, cross-hemi mask, NaN cleanup
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False)
def assemble_connect_vmf(vmf, grad_sq, u_sq, n_lh, L_lh):
    """In-place: ``vmf[n, k] = -||grad[n] - u[k]||^2`` via the gemm trick
    ``-(||g||^2 - 2 g·u + ||u||^2)``, with cross-hemisphere mask + NaN→-Inf.

    Mirrors MATLAB lines 761-770 of
    ``CBIG_ArealMSHBM_gMSHBM_generate_individual_parcellation.m``:

        for p = 1:L
            curr_parcel_diff = bsxfun(@minus, grad_data, u(p,:));
            curr_spatial_vmf(:, p) = -1*sum(curr_parcel_diff.^2, 2);
        end
        curr_spatial_vmf(isnan(curr_spatial_vmf)) = -Inf;
        % cross-hemi mask:
        curr_spatial_vmf(1:N/2, L/2+1:L) = -Inf;
        curr_spatial_vmf(N/2+1:end, 1:L/2) = -Inf;

    The caller has already computed ``vmf = grad @ u.T`` (the ``grad·u``
    cross term) and supplies ``grad_sq[n] = ||grad[n]||^2`` (precomputed
    once per session) and ``u_sq[k] = ||u[k]||^2`` (per-call). This kernel
    folds them together into ``-||g - u||^2 = 2 (g·u) - ||g||^2 - ||u||^2``
    and then applies the masks.

    Cross-hemisphere mask:
      LH-vertex (n < n_lh)    × RH-parcel (k >= L_lh) → -Inf
      RH-vertex (n >= n_lh)   × LH-parcel (k <  L_lh) → -Inf

    Inputs / Outputs:
        vmf     : (N, L) float32 — input contains ``grad @ u.T``;
            rewritten to spatial_connect_vmf with masks + NaN cleanup.
    Inputs:
        grad_sq : (N,)  float32 — ``||grad_data[n]||^2`` (per-session, cached).
        u_sq    : (L,)  float32 — ``||u[k]||^2`` (per-call; NaN where parcel empty).
        n_lh    : int   — left-hemi vertex count (= N/2).
        L_lh    : int   — left-hemi parcel count (= L/2).
    """
    N, L = vmf.shape
    NEG_INF = np.float32(-np.inf)
    TWO = np.float32(2.0)
    for n in range(N):
        is_lh_vert = n < n_lh
        gs = grad_sq[n]
        for k in range(L):
            is_lh_parc = k < L_lh
            if is_lh_vert != is_lh_parc:
                vmf[n, k] = NEG_INF
            else:
                v = TWO * vmf[n, k] - gs - u_sq[k]
                if v != v:
                    vmf[n, k] = NEG_INF
                else:
                    vmf[n, k] = v


# ─────────────────────────────────────────────────────────────────────────
# Pre-allocated scratch reductions (small per-call helpers, fp32 contig).
#
# Block-diagonal sgemms are used only for the K-axis-large gemms (step 1
# of xyz: ``sphere.T @ s_lambda``; step 1 of connect: ``grad.T @ s_lambda``)
# where the cross-hemi-zero invariant of ``s_lambda`` makes block-diag
# mathematically exact. Element-wise reductions stay numba only on contig
# fp32; on non-contig slices we use numpy (numpy's reduction
# auto-vectorizes over the contig axis, while a naive numba scalar loop
# on a non-contig slice loses SIMD).
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False)
def col_norms_3xL_f32(lambda_X, out):
    """``out[k] = sqrt(sum_d lambda_X[d, k]^2)`` for d=0..2 (3-row matrix).

    Replaces ``np.sqrt((lambda_X**2).sum(axis=0))`` which allocates a
    (3, L) temp + a (L,) temp every call.

    Inputs:
        lambda_X : (3, L) float32 read-only contiguous.
    Outputs:
        out      : (L,) float32 rewritten.
    """
    three, L = lambda_X.shape
    for k in range(L):
        s = (lambda_X[0, k] * lambda_X[0, k]
             + lambda_X[1, k] * lambda_X[1, k]
             + lambda_X[2, k] * lambda_X[2, k])
        out[k] = np.sqrt(s)


@nb.njit(cache=True, fastmath=False, boundscheck=False, error_model="numpy")
def divide_cols_3xL_inplace_f32(lambda_X, col_norms, out):
    """``out[d, k] = lambda_X[d, k] / col_norms[k]`` for d=0..2.

    Per-call, replaces ``np.divide(lambda_X, col_norms[None, :], out=out)``
    which numpy handles fine but we keep the numba kernel for consistency.
    Produces NaN where col_norms[k] == 0 (matches numpy + IEEE behavior) —
    enabled by ``error_model='numpy'`` so 0/0 → NaN instead of raising
    ZeroDivisionError. Zero col norms occur for empty parcels (lambda_X
    column all-zero); the downstream ``assemble_xyz_vmf`` zero-cleanup
    handles the resulting NaN.

    Inputs:
        lambda_X  : (3, L) float32 read-only contiguous.
        col_norms : (L,)   float32 read-only.
    Outputs:
        out       : (3, L) float32 rewritten.
    """
    three, L = lambda_X.shape
    for k in range(L):
        c = col_norms[k]
        out[0, k] = lambda_X[0, k] / c
        out[1, k] = lambda_X[1, k] / c
        out[2, k] = lambda_X[2, k] / c


@nb.njit(cache=True, fastmath=False, boundscheck=False, error_model="numpy")
def divide_uupdate_f32(u_update, sum_lam, out):
    """``out[k, d] = u_update[d, k] / sum_lam[k]``.

    Replaces ``np.divide(u_update.T, sum_lam[:, None], out=out)`` plus the
    transpose temp. Produces NaN where sum_lam[k] == 0 (empty parcel) —
    enabled by ``error_model='numpy'`` so 0/0 → NaN instead of raising
    ZeroDivisionError. The downstream ``assemble_connect_vmf`` NaN→-Inf
    cleanup handles the resulting NaN columns.

    Inputs:
        u_update : (D, L) float32 read-only.
        sum_lam  : (L,)   float32 read-only.
    Outputs:
        out      : (L, D) float32 rewritten.
    """
    D, L = u_update.shape
    for k in range(L):
        c = sum_lam[k]
        for d in range(D):
            out[k, d] = u_update[d, k] / c


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def row_sq_norms_f32(u, out):
    """``out[k] = sum_d u[k, d]^2`` for k=0..L-1.

    Replaces ``np.sum(u**2, axis=1, dtype=np.float32, out=out)`` plus the
    ``u**2`` temp.

    Inputs:
        u   : (L, D) float32 read-only contiguous.
    Outputs:
        out : (L,) float32 rewritten.
    """
    L, D = u.shape
    for k in range(L):
        s = np.float32(0.0)
        for d in range(D):
            s += u[k, d] * u[k, d]
        out[k] = s


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def cast_f64_to_f32_1d(src, dst):
    """Single-core fp64 -> fp32 cast for a 1D array.

    Inputs:
        src : (L,) float64 read-only.
    Outputs:
        dst : (L,) float32 rewritten.
    """
    L = src.shape[0]
    for k in range(L):
        dst[k] = np.float32(src[k])


# ─────────────────────────────────────────────────────────────────────────
# Workspace fp32 cast helper (used by both Sessions when caller hands fp64).
# Mirrors V_lambda/_kernels.cast_f64_to_f32 — kept here too so each module
# is self-contained.
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False)
def cast_f64_to_f32_2d(src, dst):
    """Single-core fp64 → fp32 cast for a 2D array.

    Inputs:
        src : (N, M) float64 read-only
    Outputs:
        dst : (N, M) float32 rewritten
    """
    N, M = src.shape
    for i in range(N):
        for j in range(M):
            dst[i, j] = np.float32(src[i, j])


# ─────────────────────────────────────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────────────────────────────────────
def warmup() -> None:
    """Compile every kernel with realistic dtypes/shapes once. ~50 ms one-shot."""
    N, L = 4, 6
    n_lh = N // 2
    L_lh = L // 2

    # Full-block kernels (legacy path, retained for the reference functions).
    vmf = np.full((N, L), 0.5, dtype=np.float32)
    cdln_per_k = np.full(L, -1.0, dtype=np.float32)
    cdln_per_k[0] = np.float32(np.nan)
    xyz_gamma = np.array([0.0, 1000.0, 1000.0, 0.0, 1000.0, 0.0], dtype=np.float32)
    assemble_xyz_vmf(vmf, cdln_per_k, xyz_gamma)

    vmf2 = np.full((N, L), 0.1, dtype=np.float32)
    grad_sq = np.full(N, 0.2, dtype=np.float32)
    u_sq = np.full(L, 0.3, dtype=np.float32)
    u_sq[0] = np.float32(np.nan)
    assemble_connect_vmf(vmf2, grad_sq, u_sq, n_lh, L_lh)

    # Pre-allocated reduction kernels.
    lam_X = np.full((3, L), 0.5, dtype=np.float32)
    cn = np.empty(L, dtype=np.float32)
    col_norms_3xL_f32(lam_X, cn)
    divide_cols_3xL_inplace_f32(lam_X, cn, lam_X)

    u_update = np.full((3, L), 0.5, dtype=np.float32)  # D=3 stand-in
    u_buf = np.empty((L, 3), dtype=np.float32)
    sum_lam_full = np.full(L, 0.5, dtype=np.float32)
    divide_uupdate_f32(u_update, sum_lam_full, u_buf)

    u_for_sq = np.full((L, 3), 0.5, dtype=np.float32)
    sq_out = np.empty(L, dtype=np.float32)
    row_sq_norms_f32(u_for_sq, sq_out)

    src1d = np.full(L, 0.4, dtype=np.float64)
    dst1d = np.empty(L, dtype=np.float32)
    cast_f64_to_f32_1d(src1d, dst1d)

    src = np.full((N, L), 0.4, dtype=np.float64)
    dst = np.empty((N, L), dtype=np.float32)
    cast_f64_to_f32_2d(src, dst)
