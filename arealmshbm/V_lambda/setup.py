"""setup.py

One-shot setup helpers for the V_lambda module — the per-EM-run setup
that the original MATLAB driver does once before the inner λ-loop.

Public API:
    build_neighborhood    — concat + remap LH/RH ``vertex_nbors`` over
                            the active rows of ``s_lambda``. Returns
                            ``(M1, N_active)`` int64 in MATLAB-shape
                            (1-based, 0 = absent).
    build_candidate_index — flatten the column-major nonzero pattern of
                            ``theta`` over active rows.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np


def build_neighborhood(lh_vertex_nbors: np.ndarray,
                       rh_vertex_nbors: np.ndarray,
                       s_lambda: np.ndarray) -> np.ndarray:
    """Concatenate + remap LH/RH ``vertex_nbors`` over the active rows of
    ``s_lambda``. Returns ``(M1, N_active)`` int64; entries are 1-based
    with 0 marking an absent neighbor (matches the V_lambda kernel's
    sentinel).

    Hemis are concatenated; RH neighbor IDs are offset by
    ``N_lh_active``; medial-wall neighbors collapse to 0.
    """
    N_total = s_lambda.shape[0]
    N_hemi = N_total // 2

    lh_keep = np.flatnonzero(s_lambda[:N_hemi].sum(axis=1) != 0)
    rh_keep = np.flatnonzero(s_lambda[N_hemi:].sum(axis=1) != 0)

    lh_remap = np.zeros(N_hemi + 1, dtype=np.int64)
    lh_remap[lh_keep + 1] = np.arange(1, lh_keep.size + 1, dtype=np.int64)
    rh_remap = np.zeros(N_hemi + 1, dtype=np.int64)
    rh_remap[rh_keep + 1] = np.arange(1, rh_keep.size + 1, dtype=np.int64)

    lh_nbh_keep = np.ascontiguousarray(lh_vertex_nbors[:, lh_keep], dtype=np.int64)
    rh_nbh_keep = np.ascontiguousarray(rh_vertex_nbors[:, rh_keep], dtype=np.int64)

    lh_remapped = lh_remap[np.clip(lh_nbh_keep, 0, N_hemi)]
    rh_remapped = rh_remap[np.clip(rh_nbh_keep, 0, N_hemi)]

    n_lh_active = lh_keep.size
    rh_remapped_global = np.where(rh_remapped > 0, rh_remapped + n_lh_active, 0)

    return np.ascontiguousarray(np.concatenate(
        [lh_remapped, rh_remapped_global], axis=1), dtype=np.int64)


def build_candidate_index(theta: np.ndarray,
                          s_lambda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns 0-based ``(row_idx, col_idx)`` int64 — flattened nonzero
    positions of θ over active rows, in column-major order."""
    keep = s_lambda.sum(axis=1) != 0
    tmp_theta = theta[keep]
    nz = np.asfortranarray(tmp_theta != 0)
    flat_F = np.flatnonzero(nz.ravel(order="F"))
    nrows = tmp_theta.shape[0]
    return ((flat_F % nrows).astype(np.int64),
            (flat_F // nrows).astype(np.int64))
