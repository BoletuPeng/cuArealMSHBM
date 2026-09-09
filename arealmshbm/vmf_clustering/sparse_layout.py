"""sparse_layout.py

Candidate-set (P-layout) description of the step-3 EM state, shared by
the ``gpu_sparse`` backend and its tests.

Why
---
The (N, L) dense buffers the CPU / ``gpu_full`` backends carry are
~1 % populated: ``s_lambda`` is a row-softmax of ``log_vmf`` whose
``w·log(θ)`` term is ``-Inf`` wherever ``θ == 0``, so after the first
E-step ``supp(s_lambda) ⊆ supp(θ)`` (and the initial ``s_lambda`` *is*
``θ``). Every downstream consumer (V_lambda, spatial priors, EM-stop,
M-step, argmax) therefore only ever needs the ``P = nnz(θ)`` candidate
cells. ``P ≈ 251 k`` at fsaverage6 × L=300 vs ``N·L ≈ 24.6 M``.

Layout (all 0-based; ``M`` = active rows = rows of θ with a nonzero)
-------------------------------------------------------------------
    row_idx_active (M,)   int32   full vertex id of active row m (ascending)
    inv_active     (N,)   int32   m for active vertices, -1 otherwise
    row_ptr        (M+1,) int32   CSR over active rows; candidates of row m
                                  live in [row_ptr[m], row_ptr[m+1]) with
                                  **ascending column** order
    col            (P,)   int32   parcel index l
    theta          (P,)   fp32    θ[n, l]  (> 0 by construction)
    bm             (P,)   fp32    boundary_mask[n, l]
    col_ptr        (L+1,) int32   CSC: members of parcel l in
                                  [col_ptr[l], col_ptr[l+1]), ascending n
    csc_row        (P,)   int32   full vertex id n of the member
    csc_pidx       (P,)   int32   index into the CSR arrays (so a P-vector
                                  in CSR order is read as ``x[csc_pidx]``)
    neighborhood   (M, M1) int32  Potts stencil in active space, 1-indexed,
                                  0 = absent (medial wall / missing) —
                                  the transpose of V_lambda's
                                  ``build_neighborhood`` output
    bm_row_ptr     (M+1,) int32   CSR over active rows of the boundary-mask
    bm_col         (Pb,)  int32   support (superset of the θ support); only
                                  consulted when an ``acc`` column is NaN

Invariants checked by :func:`build_candidate_layout_dense`:
    * ``θ ≥ 0``; ``supp(θ) ⊆ supp(boundary_mask)``;
    * no cross-hemisphere candidate (LH vertex × RH parcel or vice versa).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from arealmshbm.V_lambda.setup import build_neighborhood

#: Widest packed profile the ``gpu_sparse`` kernels accept, in bytes.
#: ``acc_bits`` gives each of its 32 lanes ``ACC_MAXB = 8`` bytes, so
#: ``ceil(D/8) <= 8*32 = 256`` (D <= 2048).
MAX_D_BYTES = 256


@dataclass
class CandidateLayout:
    """The candidate-set (P-layout) index bundle for one subject.

    ``N``/``L`` are the dense shape, ``M_active`` the number of rows with
    at least one candidate and ``P`` the number of candidate cells.
    ``row_idx_active``/``inv_active`` map between dense rows and active
    rows; ``row_ptr``/``col`` are the CSR index of the candidate set and
    ``col_ptr``/``csc_row``/``csc_pidx`` its CSC transpose (``csc_pidx``
    points back into CSR order). ``theta`` and ``bm`` carry the prior and
    the boundary mask gathered to P order, ``bm_row_ptr``/``bm_col`` the
    boundary mask's own CSR, and ``neighborhood`` the V_lambda
    neighbourhood table. All index arrays are int32 and C-contiguous.
    """
    N: int
    L: int
    M_active: int
    P: int
    row_idx_active: np.ndarray
    inv_active: np.ndarray
    row_ptr: np.ndarray
    col: np.ndarray
    theta: np.ndarray
    bm: np.ndarray
    col_ptr: np.ndarray
    csc_row: np.ndarray
    csc_pidx: np.ndarray
    neighborhood: np.ndarray
    bm_row_ptr: np.ndarray
    bm_col: np.ndarray

    @property
    def n_lh(self) -> int:
        return self.N // 2

    @property
    def L_lh(self) -> int:
        return self.L // 2

    def csr_rows_of_P(self) -> np.ndarray:
        """(P,) int32 — active-row index m of each CSR entry."""
        return np.repeat(np.arange(self.M_active, dtype=np.int32),
                         np.diff(self.row_ptr))

    def gather(self, x_NL: np.ndarray) -> np.ndarray:
        """Gather a dense (N, L) array at the P cells (CSR order)."""
        rows = self.row_idx_active[self.csr_rows_of_P()]
        return np.ascontiguousarray(x_NL[rows, self.col])


def _csr_from_dense_mask(mask_MxL: np.ndarray):
    """Row-major nonzero positions of a (M, L) boolean array → (row_ptr, col)."""
    rows, cols = np.nonzero(mask_MxL)          # row-major ⇒ ascending col per row
    counts = np.bincount(rows, minlength=mask_MxL.shape[0])
    row_ptr = np.zeros(mask_MxL.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return row_ptr.astype(np.int32), cols.astype(np.int32), rows


def build_candidate_layout_dense(theta: np.ndarray,
                                 boundary_mask: np.ndarray,
                                 lh_vertex_nbors: np.ndarray,
                                 rh_vertex_nbors: np.ndarray,
                                 ) -> CandidateLayout:
    """Reference (dense-input) builder. Slow-ish (~100 ms at fsa6) but
    trivially correct; the fast loader in ``step3_pipeline.sparse_inputs``
    must produce an identical layout (pinned by its tests).
    """
    th = np.ascontiguousarray(theta, dtype=np.float32)
    bmask = np.asarray(boundary_mask)
    N, L = th.shape
    if bmask.shape != (N, L):
        raise ValueError(f"boundary_mask shape {bmask.shape} != ({N}, {L})")
    if N % 2 or L % 2:
        raise ValueError("N and L must be even (bilateral layout)")
    if (th < 0).any():
        raise ValueError("theta must be non-negative")

    row_active = th.sum(axis=1) != 0
    row_idx_active = np.flatnonzero(row_active).astype(np.int32)
    M = int(row_idx_active.size)
    inv_active = np.full(N, -1, dtype=np.int32)
    inv_active[row_idx_active] = np.arange(M, dtype=np.int32)

    th_act = th[row_idx_active]
    bm_act = bmask[row_idx_active]
    row_ptr, col, rows_m = _csr_from_dense_mask(th_act != 0)
    P = int(col.size)
    theta_P = np.ascontiguousarray(th_act[rows_m, col], dtype=np.float32)
    bm_P = np.ascontiguousarray(bm_act[rows_m, col], dtype=np.float32)
    if (bm_P == 0).any():
        raise ValueError("supp(theta) must be inside supp(boundary_mask)")
    n_full = row_idx_active[rows_m]
    n_lh, L_lh = N // 2, L // 2
    if (((n_full < n_lh) & (col >= L_lh)) | ((n_full >= n_lh) & (col < L_lh))).any():
        raise ValueError("theta has cross-hemisphere candidates")

    # CSC: sort P entries by (col, n) — argsort with a stable key.
    order = np.lexsort((n_full, col)).astype(np.int32)
    col_counts = np.bincount(col, minlength=L)
    col_ptr = np.zeros(L + 1, dtype=np.int64)
    np.cumsum(col_counts, out=col_ptr[1:])
    csc_row = np.ascontiguousarray(n_full[order], dtype=np.int32)
    csc_pidx = np.ascontiguousarray(order, dtype=np.int32)

    nbh_M1xM = build_neighborhood(lh_vertex_nbors, rh_vertex_nbors, th)
    if nbh_M1xM.shape[1] != M:
        raise ValueError("neighborhood active count mismatch")
    neighborhood = np.ascontiguousarray(nbh_M1xM.T, dtype=np.int32)

    bm_row_ptr, bm_col, _ = _csr_from_dense_mask(bm_act != 0)

    return CandidateLayout(
        N=N, L=L, M_active=M, P=P,
        row_idx_active=row_idx_active, inv_active=inv_active,
        row_ptr=row_ptr, col=col, theta=theta_P, bm=bm_P,
        col_ptr=col_ptr.astype(np.int32), csc_row=csc_row, csc_pidx=csc_pidx,
        neighborhood=neighborhood,
        bm_row_ptr=bm_row_ptr, bm_col=bm_col,
    )


def layouts_equal(a: CandidateLayout, b: CandidateLayout) -> bool:
    """True iff every scalar and every index/value array matches
    exactly (``np.array_equal``)."""
    for f in ("N", "L", "M_active", "P"):
        if getattr(a, f) != getattr(b, f):
            return False
    for f in ("row_idx_active", "inv_active", "row_ptr", "col", "theta", "bm",
              "col_ptr", "csc_row", "csc_pidx", "neighborhood",
              "bm_row_ptr", "bm_col"):
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            return False
    return True


def layout_to_device(layout: CandidateLayout) -> Dict[str, Any]:
    """Upload every array to the current CuPy device; returns a dict of
    ``cupy.ndarray`` keyed by field name (plus the scalar fields)."""
    import cupy as cp
    out: Dict[str, Any] = {
        "N": layout.N, "L": layout.L, "M_active": layout.M_active, "P": layout.P,
    }
    for f in ("row_idx_active", "inv_active", "row_ptr", "col", "theta", "bm",
              "col_ptr", "csc_row", "csc_pidx", "neighborhood",
              "bm_row_ptr", "bm_col"):
        out[f] = cp.asarray(np.ascontiguousarray(getattr(layout, f)))
    return out


__all__ = [
    "MAX_D_BYTES", "CandidateLayout", "build_candidate_layout_dense",
    "layouts_equal", "layout_to_device",
]
