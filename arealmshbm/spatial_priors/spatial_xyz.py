"""spatial_xyz.py

Spherical-vMF compactness prior for each parcel — the ``spatial_xyz_prior``
term added to ``log_vmf`` in the EM E-step. Activated when
``check_connectedness`` boosts ``Params.xyz_gamma`` for parcels split
into geometrically-distant components. At ``xyz_gamma = 0`` the prior
contributes nothing (``Cdln(0, 3) = NaN`` propagates through the
multiplication and the post-cleanup zeros it).

Public API:
    XyzSession              — caches the bilateral sphere coords once;
                              recomputes the (N, L) prior each call.
    compute_unit_sphere_xyz — load + row-normalize bilateral sphere
                              coords (also used by the GPU full-device
                              path).

Per call:
    1. ``lambda_X = sphere_xyz.T @ s_lambda``                  — (3, L) fp32 gemm
    2. ``s_muc = column-normalize(lambda_X)``                  — (3, L) fp32
    3. ``gamma_cos = sphere_xyz @ s_muc``                      — (N, L) fp32 gemm
    4. ``cdln_per_k = Cdln(xyz_gamma, 3)``                     — (L,) fp32
    5. ``spatial_xyz_vmf = cdln_per_k + xyz_gamma * gamma_cos``— (N, L) fp32
       with ``NaN → 0`` cleanup

Steps 1, 3 are fp32 sgemm via numpy. Step 5 is a fused numba kernel
(:func:`_kernels.assemble_xyz_vmf`). Cdln is a numba leaf
(:func:`_cdln.cdln_d3_to_f32`). All intermediates are fp32; output is fp32.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import _kernels
from ._cdln import cdln_d3_to_f32


# Step functions — one per algorithmic step in the original derivation.
# Historically each was wired into the MATLAB-GT validate.py harness
# (now retired) to bisect intermediate-quantity drift.
def compute_unit_sphere_xyz(vertices: np.ndarray) -> np.ndarray:
    """Mirror of MATLAB lines 412-420 (sphere mesh load + row-normalize).

    Inputs:
        vertices : (N, 3) or (3, N) array of bilateral sphere coordinates.
            Either layout accepted; the result is always (N, 3).
    Returns:
        (N, 3) float32 contiguous, each row a unit vector on S^2.
        Rows that are all-zero (degenerate) stay all-zero (NOT NaN'd) —
        downstream s_muc computation handles the propagation.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape[0] == 3 and v.shape[1] != 3:
        v = v.T
    if v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3) or (3, N); got {v.shape}")
    norms = np.sqrt((v ** 2).sum(axis=1, keepdims=True))
    safe = np.where(norms == 0.0, 1.0, norms)
    out = v / safe
    return np.ascontiguousarray(out, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
# XyzSession.
#
# Two structural exploits driven by step-3 invariants:
#
#  (1) Cross-hemi block-diagonal step 1 sgemm. ``s_lambda`` is cross-hemi
#      zero upstream (boundary_mask), so ``lambda_X = sphere.T @ s_lambda``
#      splits into LH-LH + RH-RH halves and is mathematically exact —
#      ~2x FLOPS reduction on the K-axis-large gemm.
#
#      Step 3 (``sphere @ s_muc``) and step 5 (assemble) stay FULL-buffer.
#      That's not a missed optimization — it's required to match MATLAB
#      at the intermediate cross-hemi cells (s_muc has no cross-hemi
#      structure; every column is the unit-vector mean direction of that
#      parcel, so step 3 produces a finite cosine at every (n, k)
#      including cross-hemi). The intermediate buffer is filled at all
#      cells. Cross-hemi cells of spatial_xyz_vmf are downstream-
#      irrelevant (the EM softmax sees -Inf there from independent
#      sources — ``beta * spatial_connect_vmf`` and ``w * log(theta)``
#      both pin it), but filling them keeps the per-step intermediate
#      shape identical to the original derivation.
#
#  (2) All per-call temporaries pre-allocated. No internal allocation in
#      the hot path; reductions go through dedicated numba kernels in
#      ``_kernels``.
#
# Per call: 2 sgemms (LH + RH halves of step 1) + 2 numba reductions
# (col-norm + divide) + 1 full sgemm (step 3) + 1 numba leaf (Cdln) +
# 1 full numba leaf (assemble). All inputs from caller, all scratch from
# session, no numpy temporaries.
# ─────────────────────────────────────────────────────────────────────────
class XyzSession:
    """Pre-allocated state for the spatial_xyz_prior fast path.

    Construction: takes the bilateral sphere XYZ (N, 3) or (3, N), pre-
    normalizes to unit length (idempotent — feeding already-unit vectors
    gives the same fp32 array), and allocates per-call output buffers.

    Per call: :meth:`compute` runs the four-step pipeline and returns
    ``(s_muc, spatial_xyz_vmf)`` as VIEWS into cached buffers. The next
    ``compute`` overwrites them; copy if you need to keep across calls.

    Hemisphere split (LH = first half, RH = second half) assumes:
      * vertex order is ``[lh_verts; rh_verts]`` with ``n_lh = N // 2``.
      * parcel order is ``[lh_parcels; rh_parcels]`` with ``L_lh = L // 2``.
    These match the step-3 driver's layout (driver lines 246-287 + 232-237).
    """

    __slots__ = (
        "N", "L", "n_lh", "L_lh",
        # Cached sphere — the LH/RH views are zero-copy slices of `sphere_xyz_unit`.
        "sphere_xyz_unit",          # (N, 3) fp32 C-contig — owns memory
        "_sphere_lh",               # (n_lh, 3) view
        "_sphere_rh",               # (n_lh, 3) view
        # Per-call scratch for the LH/RH halves.
        "_lambda_X_lh",             # (3, L_lh) fp32 — scratch
        "_lambda_X_rh",             # (3, L_lh) fp32 — scratch
        "_col_norms_lh",            # (L_lh,)   fp32 — scratch
        "_col_norms_rh",            # (L_lh,)   fp32 — scratch
        "_s_muc_f32",               # (3, L)    fp32 — output (LH cols + RH cols)
        "_xyz_gamma_f64",           # (L,)      fp64 — scratch (Cdln input)
        "_xyz_gamma_f32",           # (L,)      fp32 — scratch (assemble multiplier)
        "_cdln_per_k",              # (L,)      fp32 — scratch
        # Output buffer — full sgemm overwrites every cell on every call;
        # the np.empty allocation here is pure scratch.
        "_spatial_xyz_vmf_f32",     # (N, L)    fp32 C-contig — output
        # Cast destination for fp64 callers.
        "_s_lambda_f32_buf",        # (N, L)    fp32 C-contig — scratch
    )

    def __init__(self, sphere_xyz: np.ndarray, L: int):
        sphere = compute_unit_sphere_xyz(sphere_xyz)   # (N, 3) fp32
        N = sphere.shape[0]
        if N % 2 != 0:
            raise ValueError(
                f"N={N} must be even (bilateral mesh, half LH half RH)"
            )
        if L % 2 != 0:
            raise ValueError(
                f"L={L} must be even (half LH parcels, half RH parcels)"
            )

        self.N = int(N)
        self.L = int(L)
        self.n_lh = int(N // 2)
        self.L_lh = int(L // 2)

        # Sphere + LH/RH views (zero-copy slices).
        self.sphere_xyz_unit = sphere
        self._sphere_lh = sphere[:self.n_lh]
        self._sphere_rh = sphere[self.n_lh:]

        # Per-half scratch.
        self._lambda_X_lh   = np.empty((3, self.L_lh), dtype=np.float32)
        self._lambda_X_rh   = np.empty((3, self.L_lh), dtype=np.float32)
        self._col_norms_lh  = np.empty(self.L_lh, dtype=np.float32)
        self._col_norms_rh  = np.empty(self.L_lh, dtype=np.float32)

        # Full s_muc output.
        self._s_muc_f32     = np.empty((3, L), dtype=np.float32)

        # Cdln scratch.
        self._xyz_gamma_f64 = np.empty(L, dtype=np.float64)
        self._xyz_gamma_f32 = np.empty(L, dtype=np.float32)
        self._cdln_per_k    = np.empty(L, dtype=np.float32)

        # Output buffer. Step 3 below is a FULL sgemm sphere @ s_muc → vmf,
        # so every cell is overwritten on every call — np.empty (uninit)
        # is correct, no memset needed at construction.
        self._spatial_xyz_vmf_f32 = np.empty((N, L), dtype=np.float32)

        # fp64 caller cast destination.
        self._s_lambda_f32_buf = np.empty((N, L), dtype=np.float32)

    # ────────── Per-call API ──────────
    def compute(self,
                s_lambda: np.ndarray,
                xyz_gamma: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full spatial_xyz_prior pipeline (block-diagonal hot path).

        Parameters
        ----------
        s_lambda : (N, L) array, fp32 or fp64.
            Soft posterior. Cast to fp32 (zero-copy if already fp32 contig).
            Cross-hemi entries assumed zero (boundary_mask upstream).
        xyz_gamma : (L,) or (1, L) or (L, 1) array, any float dtype.
            Per-parcel concentration; 0 at inactive parcels (stays 0 across
            calls — monotonic).

        Returns
        -------
        s_muc            : (3, L) float32 view of cached buffer.
        spatial_xyz_vmf  : (N, L) float32 view of cached buffer.
        """
        s_lam = self._stage_s_lambda(s_lambda)

        # Stage xyz_gamma into pre-allocated fp64 + fp32 scratch (no temp alloc).
        gamma_in = np.asarray(xyz_gamma).ravel()
        if gamma_in.shape[0] != self.L:
            raise ValueError(
                f"xyz_gamma length {gamma_in.shape[0]} != L={self.L}"
            )
        np.copyto(self._xyz_gamma_f64, gamma_in, casting="safe")
        _kernels.cast_f64_to_f32_1d(self._xyz_gamma_f64, self._xyz_gamma_f32)

        n_lh = self.n_lh
        L_lh = self.L_lh

        # ── Step 1 (block-diagonal sgemm) ──
        # lambda_X[:, :L_lh] = sphere_lh.T @ s_lambda[:n_lh, :L_lh]
        # lambda_X[:, L_lh:] = sphere_rh.T @ s_lambda[n_lh:, L_lh:]
        # Cross-hemi entries of s_lambda are 0 (boundary_mask), so the diagonal
        # blocks fully recover the full sgemm result on the same-hemi columns.
        # ~2x FLOPS reduction vs the (3, N) @ (N, L) full gemm. Cross-hemi
        # columns of lambda_X stay 0 (pre-zeroed at __init__ via _s_muc_f32 init).
        s_lam_lh = s_lam[:n_lh, :L_lh]
        s_lam_rh = s_lam[n_lh:, L_lh:]
        np.matmul(self._sphere_lh.T, s_lam_lh, out=self._lambda_X_lh)
        np.matmul(self._sphere_rh.T, s_lam_rh, out=self._lambda_X_rh)

        # ── Step 2 (column-normalize per hemi) ──
        # NaN where col is all-zero (parcel got no soft assignment); the NaN
        # propagates through to vmf cross-hemi cells and the cleanup zeros them.
        _kernels.col_norms_3xL_f32(self._lambda_X_lh, self._col_norms_lh)
        _kernels.col_norms_3xL_f32(self._lambda_X_rh, self._col_norms_rh)
        s_muc_lh = self._s_muc_f32[:, :L_lh]
        s_muc_rh = self._s_muc_f32[:, L_lh:]
        _kernels.divide_cols_3xL_inplace_f32(self._lambda_X_lh,
                                              self._col_norms_lh, s_muc_lh)
        _kernels.divide_cols_3xL_inplace_f32(self._lambda_X_rh,
                                              self._col_norms_rh, s_muc_rh)

        # ── Step 3 (FULL sgemm — sphere @ s_muc → vmf) ──
        # Full and not block-diagonal because s_muc has no cross-hemi
        # structure — every column is the unit-vector mean direction of
        # that parcel, and ``sphere_xyz @ s_muc`` produces a finite cosine
        # at every (n, k) including cross-hemi. A block-diag sliced
        # step 3 would leave cross-hemi cells at their pre-init value
        # (which would diverge from the expected finite cosine), and
        # would also write into non-C-contig sub-blocks at step 5 (loses
        # numba auto-vectorization, ~7 ms regression measured during the
        # tier-3 optimization sweep).
        np.matmul(self.sphere_xyz_unit, self._s_muc_f32,
                  out=self._spatial_xyz_vmf_f32)

        # ── Step 4 (Cdln) ──
        cdln_d3_to_f32(self._xyz_gamma_f64, self._cdln_per_k)

        # ── Step 5 (FULL contig assemble — auto-vectorized) ──
        # In-place over the (N, L) C-contig buffer. NaN cleanup matters
        # only for empty parcels (col_norm == 0 → s_muc[:, k] all NaN →
        # vmf[:, k] all NaN); the kernel's ``v != v`` check zeros those
        # columns. Cross-hemi cells are finite (see step 3 comment) and
        # pass through unchanged.
        _kernels.assemble_xyz_vmf(
            self._spatial_xyz_vmf_f32,
            self._cdln_per_k,
            self._xyz_gamma_f32,
        )
        return self._s_muc_f32, self._spatial_xyz_vmf_f32

    # ────────── Internal ──────────
    def _stage_s_lambda(self, s_lambda: np.ndarray) -> np.ndarray:
        """Get an ``(N, L)`` float32 contiguous view of ``s_lambda``.

        Zero-copy fast paths:
          * float32 C-contiguous   (the canonical numpy-side pattern)
          * float32 F-contiguous   (MATLAB→Python boundary; numpy will
                                    materialize a transposed view that
                                    sgemm handles natively)
        Cast paths:
          * float64 C-contig → ``cast_f64_to_f32_2d`` into _s_lambda_f32_buf
          * everything else  → ``np.ascontiguousarray(..., float32)`` (copy)
        """
        if s_lambda.shape != (self.N, self.L):
            raise ValueError(
                f"s_lambda shape mismatch: expected ({self.N}, {self.L}), "
                f"got {s_lambda.shape}"
            )
        if s_lambda.dtype == np.float32 and (
            s_lambda.flags["C_CONTIGUOUS"] or s_lambda.flags["F_CONTIGUOUS"]
        ):
            return s_lambda
        if s_lambda.dtype == np.float64 and s_lambda.flags["C_CONTIGUOUS"]:
            _kernels.cast_f64_to_f32_2d(s_lambda, self._s_lambda_f32_buf)
            return self._s_lambda_f32_buf
        # Fallback for unusual dtypes / non-contig.
        np.copyto(self._s_lambda_f32_buf,
                  np.ascontiguousarray(s_lambda, dtype=np.float32))
        return self._s_lambda_f32_buf
