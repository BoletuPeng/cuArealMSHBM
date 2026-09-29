"""sparse_layout.py

Candidate-set (P-layout) description of the step-3 EM state, shared by
the ``gpu`` backend and its tests.

Why
---
The (N, L) dense buffers the ``cpu`` backend carries are
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

Invariants (enforced by
:func:`arealmshbm.step3_pipeline.sparse_inputs.build_candidate_layout_fast`):
    * ``θ ≥ 0``; ``supp(θ) ⊆ supp(boundary_mask)``;
    * no cross-hemisphere candidate (LH vertex × RH parcel or vice versa).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


#: Widest packed profile the ``gpu`` kernels accept, in bytes.
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


def layout_to_device(layout: CandidateLayout) -> Dict[str, Any]:
    """Upload the index arrays and ``bm`` to the current CuPy device;
    returns a dict of ``cupy.ndarray`` keyed by field name (plus the
    scalar fields). ``theta`` stays on the host: the session derives
    ``log_theta`` and the initial ``s_lambda`` from it itself."""
    import cupy as cp
    out: Dict[str, Any] = {
        "N": layout.N, "L": layout.L, "M_active": layout.M_active, "P": layout.P,
    }
    for f in ("row_idx_active", "inv_active", "row_ptr", "col", "bm",
              "col_ptr", "csc_row", "csc_pidx", "neighborhood",
              "bm_row_ptr", "bm_col"):
        out[f] = cp.asarray(np.ascontiguousarray(getattr(layout, f)))
    return out


__all__ = ["MAX_D_BYTES", "CandidateLayout", "layout_to_device"]
