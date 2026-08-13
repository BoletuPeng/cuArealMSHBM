"""boundary_mask.py

Step-2 leaf L9 — bilateral block-diagonal boundary mask, mode-dispatched.

Mirrors MATLAB
``CBIG_ArealMSHBM_cdgMSHBM_estimate_group_priors_sequential.m`` lines
166-170::

    if strcmp(mode, 'gMSHBM')
        boundary_mask = [lh_boundary zeros(size(rh_boundary));
                         zeros(size(lh_boundary)) rh_boundary]
    else  % dMSHBM
        boundary_mask = [lh_boundary zeros(size(lh_boundary));
                         zeros(size(rh_boundary)) rh_boundary]

For bilateral fsaverage6 (``size(lh_boundary) == size(rh_boundary)``)
both branches produce numerically identical block-diagonal matrices.
The dispatch is preserved for forward-compat with asymmetric meshes
should they ever arise.

The block-diagonal layout reuses the construction in
:mod:`arealmshbm.pipeline_setup.pipeline_setup`. Output is fp32
(step-2 init precision); the existing step-3 helper returns fp64 so we
cast at the boundary.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np


# Only gMSHBM/dMSHBM are wired; cMSHBM xyz-vMF path is not implemented.
_VALID_MODES = ("gMSHBM", "dMSHBM")


def build_step2_boundary_mask(lh_boundary: np.ndarray,
                              rh_boundary: np.ndarray,
                              mode: str) -> np.ndarray:
    """Build the (N_lh + N_rh, L_lh + L_rh) bilateral block-diagonal mask.

    Parameters
    ----------
    lh_boundary : (N_lh, L_lh) dense — LH radius-mask block.
    rh_boundary : (N_rh, L_rh) dense — RH radius-mask block.
    mode        : one of ``'gMSHBM' | 'dMSHBM'``. On bilateral
                  meshes (``N_lh == N_rh and L_lh == L_rh``) the output is
                  identical across modes.

    Returns
    -------
    boundary_mask : (N_lh + N_rh, L_lh + L_rh) fp32 dense block-diagonal.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"mode must be one of {_VALID_MODES}, got {mode!r}")

    lh = np.ascontiguousarray(np.asarray(lh_boundary), dtype=np.float32)
    rh = np.ascontiguousarray(np.asarray(rh_boundary), dtype=np.float32)
    if lh.ndim != 2 or rh.ndim != 2:
        raise ValueError("lh/rh_boundary must be 2D")
    n_lh, l_lh = lh.shape
    n_rh, l_rh = rh.shape

    if mode == "gMSHBM":
        # MATLAB: [lh, zeros(size(rh_boundary)); zeros(size(lh_boundary)), rh]
        # On bilateral meshes the two zero blocks have the same shape.
        top = np.concatenate(
            [lh, np.zeros((n_lh, l_rh), dtype=np.float32)], axis=1)
        bot = np.concatenate(
            [np.zeros((n_rh, l_lh), dtype=np.float32), rh], axis=1)
    else:
        # dMSHBM:
        # [lh, zeros(size(lh_boundary)); zeros(size(rh_boundary)), rh]
        # On bilateral meshes (n_lh == n_rh and l_lh == l_rh) this is
        # numerically identical to the gMSHBM branch — kept distinct
        # only because the MATLAB dMSHBM source uses different zero-block
        # shapes (vestigial of the unwired cMSHBM arm).
        top = np.concatenate(
            [lh, np.zeros((n_lh, l_lh), dtype=np.float32)], axis=1)
        bot = np.concatenate(
            [np.zeros((n_rh, l_rh), dtype=np.float32), rh], axis=1)

    return np.concatenate([top, bot], axis=0)
