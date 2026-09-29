"""_layout_oracle.py — dense-input reference builder for the step-3
:class:`~arealmshbm.vmf_clustering.sparse_layout.CandidateLayout`.

The test oracle for
:func:`arealmshbm.step3_pipeline.sparse_inputs.build_candidate_layout_fast`
and the layout builder of the sub-001 fixture. ``layouts_equal`` is the
structural equality the tests assert.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np

from arealmshbm.V_lambda.setup import build_neighborhood
from arealmshbm.vmf_clustering.sparse_layout import CandidateLayout


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
    trivially correct; ``step3_pipeline.sparse_inputs.build_candidate_layout_fast``
    must produce an identical layout.
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
