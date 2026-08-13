"""pipeline_setup.py

Per-EM-call setup arrays the EM body consumes:

    boundary_mask : (N, L) — bilateral block diagonal built from
                    ``spatial_mask_<mesh>.mat`` per-hemi masks.
    neighborhood  : (M1, N_active) — concatenated LH+RH ``vertexNbors``
                    with medial-wall collapse and RH offset; Potts MRF
                    stencil.
    row_idx       : flat 0-indexed positions of θ's nonzero entries
                    over active rows.
    col_idx       : matching column indices for ``row_idx``.

Reuses the helpers in :mod:`arealmshbm.V_lambda.setup` for
neighborhood / row_idx / col_idx; this module adds ``build_boundary_mask``
and the wrapping glue. Potts edge weights (V_same=0, V_diff=1) are baked
into the V_lambda kernel — no edge-weight matrices are produced here.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from arealmshbm.V_lambda.setup import (
    build_neighborhood,
    build_candidate_index,
)


def build_boundary_mask(lh_boundary: np.ndarray,
                        rh_boundary: np.ndarray) -> np.ndarray:
    """Bilateral block-diagonal boundary mask.

    Combines per-hemi sparse boundary masks into a single ``(N, L)``
    block-diagonal where the LH-RH and RH-LH blocks are zero.

    Parameters
    ----------
    lh_boundary : (N_hemi, L_hemi) — LH boundary mask (typically sparse
                  in MATLAB; densified before passing here).
    rh_boundary : (N_hemi, L_hemi) — RH boundary mask.

    Returns
    -------
    boundary_mask : (N, L) fp64 — bilateral.
    """
    lh = np.ascontiguousarray(np.asarray(lh_boundary), dtype=np.float64)
    rh = np.ascontiguousarray(np.asarray(rh_boundary), dtype=np.float64)
    if lh.ndim != 2 or rh.ndim != 2:
        raise ValueError("lh/rh_boundary must be 2D")
    n_lh, l_lh = lh.shape
    n_rh, l_rh = rh.shape

    # MATLAB line 232:
    #   boundary_mask = [lh_boundary zeros(rh.shape); zeros(lh.shape) rh_boundary]
    top = np.concatenate([lh, np.zeros((n_lh, l_rh), dtype=np.float64)], axis=1)
    bot = np.concatenate([np.zeros((n_rh, l_lh), dtype=np.float64), rh], axis=1)
    return np.concatenate([top, bot], axis=0)


def build_pipeline_setup(theta: np.ndarray,
                         s_lambda_init: np.ndarray,
                         lh_boundary: np.ndarray,
                         rh_boundary: np.ndarray,
                         lh_vertex_nbors: np.ndarray,
                         rh_vertex_nbors: np.ndarray) -> Dict[str, Any]:
    """One-stop setup for step-3 EM (Mode A).

    Mirrors MATLAB step3 lines 228-287. Output is a dict ready to merge
    into ``setting_params`` (modulo the scalar params w/c/beta/etc. the
    caller already has).

    Parameters
    ----------
    theta           : (N, L) — group spatial prior; used to derive
                      ``row_idx`` / ``col_idx``.
    s_lambda_init   : (N, L) — initial ``Params.s_lambda`` (= theta).
                      Used to identify active vertices (rows whose sum
                      is non-zero).
    lh_boundary     : (N_hemi, L_hemi) — LH block of spatial_mask.
    rh_boundary     : (N_hemi, L_hemi) — RH block of spatial_mask.
    lh_vertex_nbors : (M0, N_hemi) int — LH 1-indexed vertex neighbor
                      table from the inflated mesh.
    rh_vertex_nbors : (M0, N_hemi) int — same for RH.

    Returns
    -------
    setup : dict with keys
        ``boundary_mask`` — (N, L) fp64
        ``neighborhood``  — (M1, N_active) int64 (1-indexed; 0 = absent)
        ``row_idx``       — (P,) int64 (0-indexed)
        ``col_idx``       — (P,) int64 (0-indexed)
    """
    boundary_mask = build_boundary_mask(lh_boundary, rh_boundary)

    # Active-row neighborhood (mirrors V_lambda/setup.build_neighborhood,
    # which already implements step3 lines 246-287 verbatim).
    neighborhood = build_neighborhood(
        lh_vertex_nbors, rh_vertex_nbors, np.asarray(s_lambda_init),
    )

    row_idx, col_idx = build_candidate_index(
        np.asarray(theta), np.asarray(s_lambda_init),
    )

    return {
        "boundary_mask": boundary_mask,
        "neighborhood": neighborhood,
        "row_idx": row_idx,
        "col_idx": col_idx,
    }
