"""_layout_oracle.py — dense-input reference builder for the step-2
:class:`~arealmshbm.step2_io.sparse_layout.Step2Layout`.

The test oracle for :func:`arealmshbm.step2_io.build_step2_layout`. It
finishes through the library's private ``_finish`` (the shared CSR/CSC
assembly and invariant checks), so the equality it pins is on the CSR
extraction from the mask. ``layouts_equal`` is the structural equality
the tests assert. ``scatter`` / ``gather`` convert between a CSR-order
P-vector and the dense ``(N, L)`` grid.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np

from arealmshbm.step2_io.sparse_layout import Step2Layout, _finish


def build_step2_layout_dense(bm_NL: np.ndarray) -> Step2Layout:
    """Reference builder from the dense ``(N, L)`` mask."""
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


def scatter(lay: Step2Layout, x_P: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """CSR-order ``(P,)`` (or ``(..., P)``) → dense ``(..., N, L)``."""
    x = np.asarray(x_P)
    out = np.full(x.shape[:-1] + (lay.N, lay.L), fill, dtype=x.dtype)
    out[..., lay.p_row, lay.col] = x
    return out


def gather(lay: Step2Layout, dense: np.ndarray) -> np.ndarray:
    """Dense ``(..., N, L)`` → CSR-order ``(..., P)``."""
    d = np.asarray(dense)
    return np.ascontiguousarray(d[..., lay.p_row, lay.col])


def layouts_equal(a: Step2Layout, b: Step2Layout) -> bool:
    """True iff both layouts have the same shape scalars and the same
    six index arrays (``np.array_equal``)."""
    if (a.N, a.L, a.n_lh, a.L_lh, a.P) != (b.N, b.L, b.n_lh, b.L_lh, b.P):
        return False
    for f in ("row_ptr", "col", "p_row", "col_ptr", "csc_row", "csc_pidx"):
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            return False
    return True
