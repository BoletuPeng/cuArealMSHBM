"""sparse_layout.py — the static P-layout of the step-2 ``gpu`` backend.

``P = {(n, l) : boundary_mask[n, l] != 0}`` is fixed for a whole run: every
per-cell quantity of the EM (``s_lambda``, ``theta``, ``log_connect``,
``lv_sum``, the softmax scratch) lives on a ``(P,)`` vector in **CSR order**
(row-major, columns ascending within a row). A CSC view (``members(l)``,
rows ascending within a column) is carried alongside so parcel-wise
reductions can walk the same cells.

Design contract: ``docs/step2_sparse_design.md`` §2.1. Invariants enforced
by the builders (``ValueError`` otherwise):

* ``N == 2 * n_lh`` and ``L == 2 * L_lh`` (bilateral meshes / parcellations);
* no cross-hemisphere cell (the boundary mask is block-diagonal);
* every stored mask value is exactly ``1.0`` after ``eliminate_zeros`` — the
  radius mask is a 0/1 indicator and the E.1 multiply-by-bm on P is then an
  exact no-op;
* ``P > 0``.

Empty rows are allowed (fsaverage6 / Schaefer-300 has 1 695 of them).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Step2Layout:
    """Static candidate-cell layout ``P = nnz(boundary_mask)``.

    Attributes
    ----------
    N, L, n_lh, L_lh, P : int
    row_ptr  : (N+1,) int32 — CSR over ALL ``N`` rows; row ``n`` owns CSR
               positions ``row_ptr[n]:row_ptr[n+1]`` with ``col`` ascending.
    col      : (P,) int32 — parcel index of CSR position ``p``.
    p_row    : (P,) int32 — vertex index of CSR position ``p``.
    col_ptr  : (L+1,) int32 — CSC; ``members(l)`` occupy
               ``col_ptr[l]:col_ptr[l+1]`` with ``csc_row`` ascending.
    csc_row  : (P,) int32 — vertex index of CSC position ``i``.
    csc_pidx : (P,) int32 — CSR position of CSC position ``i`` (so a CSR-order
               vector ``x`` is read at member ``i`` as ``x[csc_pidx[i]]``).
    """

    N: int
    L: int
    n_lh: int
    L_lh: int
    P: int
    row_ptr: np.ndarray
    col: np.ndarray
    p_row: np.ndarray
    col_ptr: np.ndarray
    csc_row: np.ndarray
    csc_pidx: np.ndarray

    # ── conversions used by tests / A-B scripts ──
    def scatter(self, x_P: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """CSR-order ``(P,)`` (or ``(..., P)``) → dense ``(..., N, L)``."""
        x = np.asarray(x_P)
        out = np.full(x.shape[:-1] + (self.N, self.L), fill, dtype=x.dtype)
        out[..., self.p_row, self.col] = x
        return out

    def gather(self, dense: np.ndarray) -> np.ndarray:
        """Dense ``(..., N, L)`` → CSR-order ``(..., P)``."""
        d = np.asarray(dense)
        return np.ascontiguousarray(d[..., self.p_row, self.col])

    def as_csr_matrix(self, values: np.ndarray | None = None) -> sp.csr_matrix:
        """The layout as an ``(N, L)`` scipy CSR matrix; ``values`` is the
        P-order payload (fp32 ones when omitted)."""
        vals = (np.ones(self.P, dtype=np.float32) if values is None
                else np.asarray(values))
        return sp.csr_matrix((vals, self.col, self.row_ptr),
                             shape=(self.N, self.L))


# ─────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────
def _finish(N: int, L: int, n_lh: int, L_lh: int,
            row_ptr: np.ndarray, col: np.ndarray) -> Step2Layout:
    row_ptr = np.ascontiguousarray(row_ptr, dtype=np.int32)
    col = np.ascontiguousarray(col, dtype=np.int32)
    P = int(col.shape[0])
    if P <= 0:
        raise ValueError("Step2Layout: boundary mask has no nonzero cell")
    if row_ptr.shape != (N + 1,) or int(row_ptr[-1]) != P or int(row_ptr[0]) != 0:
        raise ValueError("Step2Layout: malformed row_ptr")
    counts = np.diff(row_ptr)
    if np.any(counts < 0):
        raise ValueError("Step2Layout: row_ptr must be non-decreasing")
    p_row = np.repeat(np.arange(N, dtype=np.int32), counts).astype(np.int32)
    # Ascending columns within each row (required by every serial-order
    # exactness argument in the design).
    if P > 1:
        same_row = p_row[1:] == p_row[:-1]
        if np.any(same_row & (col[1:] <= col[:-1])):
            raise ValueError("Step2Layout: columns must be strictly ascending "
                             "within a row")
    if np.any(col < 0) or np.any(col >= L):
        raise ValueError("Step2Layout: column index out of range")
    # Block-diagonal (no cross-hemisphere cell).
    lh_vert = p_row < n_lh
    lh_parc = col < L_lh
    if np.any(lh_vert != lh_parc):
        raise ValueError("Step2Layout: boundary mask has cross-hemisphere cells")
    # CSC via scipy's CSR->CSC conversion (a C counting sort, stable): carrying
    # the CSR position as the matrix value yields ``csc_pidx`` directly, and
    # the stability of the sort keeps rows ascending within each column.
    # ~5 ms at P = 586 k vs ~23 ms for ``np.argsort(kind='stable')``.
    csc = sp.csr_matrix(
        (np.arange(P, dtype=np.float64), col, row_ptr), shape=(N, L)
    ).tocsc()
    csc_pidx = csc.data.astype(np.int32)
    csc_row = csc.indices.astype(np.int32)
    col_ptr = csc.indptr.astype(np.int64)
    if not np.array_equal(p_row[csc_pidx], csc_row):
        raise ValueError("Step2Layout: internal CSC inconsistency")
    return Step2Layout(
        N=int(N), L=int(L), n_lh=int(n_lh), L_lh=int(L_lh), P=P,
        row_ptr=row_ptr, col=col, p_row=p_row,
        col_ptr=np.ascontiguousarray(col_ptr, dtype=np.int32),
        csc_row=np.ascontiguousarray(csc_row),
        csc_pidx=np.ascontiguousarray(csc_pidx),
    )


def _hemi_csr(m: Any, name: str) -> sp.csr_matrix:
    if sp.issparse(m):
        # copy=True: without it scipy shares ``indices``/``indptr`` (and
        # ``data`` when the dtype already matches) with the caller, and the
        # in-place eliminate_zeros/sort_indices below would rewrite the
        # caller's matrix.
        c = sp.csr_matrix(m, dtype=np.float64, copy=True)
    else:
        c = sp.csr_matrix(np.asarray(m, dtype=np.float64))
    c.eliminate_zeros()
    c.sort_indices()
    if c.nnz and not np.all(c.data == 1.0):
        bad = c.data[c.data != 1.0]
        raise ValueError(
            f"Step2Layout: {name} boundary mask must be a 0/1 indicator; "
            f"found {bad.size} value(s) != 1.0 (e.g. {bad[:3]})"
        )
    return c


def build_step2_layout(lh_mask: Any, rh_mask: Any) -> Step2Layout:
    """Build the bilateral layout from the two per-hemisphere masks
    ``(n_lh, L_lh)`` and ``(n_rh, L_rh)`` (any scipy-sparse format or dense).

    The bilateral mask is ``blockdiag(lh, rh)``: LH vertices see LH parcels
    ``0..L_lh-1``, RH vertices see RH parcels ``L_lh..L-1``.
    """
    lh = _hemi_csr(lh_mask, "lh")
    rh = _hemi_csr(rh_mask, "rh")
    n_lh, L_lh = lh.shape
    n_rh, L_rh = rh.shape
    if n_lh != n_rh or L_lh != L_rh:
        raise ValueError(
            f"Step2Layout: hemispheres must match (lh {lh.shape}, rh {rh.shape})"
        )
    N = n_lh + n_rh
    L = L_lh + L_rh
    row_ptr = np.empty(N + 1, dtype=np.int64)
    row_ptr[: n_lh + 1] = lh.indptr
    row_ptr[n_lh + 1:] = lh.indptr[-1] + rh.indptr[1:]
    col = np.concatenate([lh.indices.astype(np.int64),
                          rh.indices.astype(np.int64) + L_lh])
    return _finish(N, L, n_lh, L_lh, row_ptr, col)


def build_step2_layout_dense(bm_NL: np.ndarray) -> Step2Layout:
    """Reference builder from the dense ``(N, L)`` mask (tests only)."""
    bm = np.asarray(bm_NL)
    if bm.ndim != 2:
        raise ValueError("boundary mask must be 2-D")
    N, L = bm.shape
    if N % 2 or L % 2:
        raise ValueError("Step2Layout: N and L must be even (bilateral)")
    n_lh, L_lh = N // 2, L // 2
    nz = bm != 0
    vals = bm[nz]
    if vals.size and not np.all(vals == 1.0):
        raise ValueError("Step2Layout: boundary mask must be a 0/1 indicator")
    rows, cols = np.nonzero(nz)            # row-major → cols ascending per row
    row_ptr = np.zeros(N + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(np.bincount(rows, minlength=N))
    return _finish(N, L, n_lh, L_lh, row_ptr, cols.astype(np.int64))


def layouts_equal(a: Step2Layout, b: Step2Layout) -> bool:
    """True iff both layouts have the same shape scalars and the same
    six index arrays (``np.array_equal``)."""
    if (a.N, a.L, a.n_lh, a.L_lh, a.P) != (b.N, b.L, b.n_lh, b.L_lh, b.P):
        return False
    for f in ("row_ptr", "col", "p_row", "col_ptr", "csc_row", "csc_pidx"):
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            return False
    return True


def layout_to_device(layout: Step2Layout) -> Dict[str, Any]:
    """Upload the six index arrays; returns a plain dict (arrays + ints).

    Keys: ``row_ptr, col, p_row, col_ptr, csc_row, csc_pidx`` (cupy int32) and
    ``N, L, n_lh, L_lh, P`` (Python ints).
    """
    import cupy as cp  # local import: CPU-only hosts can still build layouts

    d: Dict[str, Any] = {
        k: cp.asarray(getattr(layout, k))
        for k in ("row_ptr", "col", "p_row", "col_ptr", "csc_row", "csc_pidx")
    }
    d.update(N=layout.N, L=layout.L, n_lh=layout.n_lh, L_lh=layout.L_lh,
             P=layout.P)
    return d
