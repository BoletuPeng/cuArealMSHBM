"""boundary_mask.py

Step-2 leaf L9 — bilateral block-diagonal boundary mask.

Mirrors the ``boundary_mask`` line of the stock parents
(``CBIG_ArealMSHBM_gMSHBM_estimate_group_priors_parent.m`` line 237 and
``CBIG_ArealMSHBM_dMSHBM_estimate_group_priors_parent.m`` line 225 at
CBIG 2296385)::

    % gMSHBM parent
    boundary_mask = [lh_boundary zeros(size(rh_boundary));
                     zeros(size(lh_boundary)) rh_boundary];
    % dMSHBM parent
    boundary_mask = [lh_boundary zeros(size(lh_boundary));
                     zeros(size(rh_boundary)) rh_boundary];

On bilateral fsaverage6 (``size(lh_boundary) == size(rh_boundary)``)
both forms are the same block-diagonal matrix; this leaf builds the
gMSHBM form for both variants.

The block-diagonal layout is the one step 3's
:func:`arealmshbm.pipeline_setup.build_boundary_mask` builds (in
fp64); this leaf builds it directly in fp32 (step-2 init precision)
and does not import that helper.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np


def build_step2_boundary_mask(lh_boundary: np.ndarray,
                              rh_boundary: np.ndarray) -> np.ndarray:
    """Build the (N_lh + N_rh, L_lh + L_rh) bilateral block-diagonal mask.

    Parameters
    ----------
    lh_boundary : (N_lh, L_lh) dense — LH radius-mask block.
    rh_boundary : (N_rh, L_rh) dense — RH radius-mask block.

    Returns
    -------
    boundary_mask : (N_lh + N_rh, L_lh + L_rh) fp32 dense block-diagonal.
    """
    lh = np.ascontiguousarray(np.asarray(lh_boundary), dtype=np.float32)
    rh = np.ascontiguousarray(np.asarray(rh_boundary), dtype=np.float32)
    if lh.ndim != 2 or rh.ndim != 2:
        raise ValueError("lh/rh_boundary must be 2D")
    n_lh, l_lh = lh.shape
    n_rh, l_rh = rh.shape

    # MATLAB: [lh, zeros(size(rh_boundary)); zeros(size(lh_boundary)), rh]
    top = np.concatenate(
        [lh, np.zeros((n_lh, l_rh), dtype=np.float32)], axis=1)
    bot = np.concatenate(
        [np.zeros((n_rh, l_lh), dtype=np.float32), rh], axis=1)

    return np.concatenate([top, bot], axis=0)
