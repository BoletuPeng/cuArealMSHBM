"""surface_smoothing.py

Per-hemisphere geodesic Gaussian smoothing — the apply half of HCP
Workbench ``src/Files/MetricSmoothingObject.cxx`` (GEO_GAUSS_AREA
branch, ROI variant).

The smoothing operator is a shipped asset: the per-hemi gather
``(gather_W, inv_weight_sum)`` is read from the step-0 inputs bundle
onto ``Step0Inputs`` (the package docstring defines its weights). Per
call, :func:`cifti_smoothing` does only a SpMV + per-row normalize on
that gather — bandwidth-bound and fast.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix


# A SmoothingGather bundle is the per-hemi gather CSR + its per-row
# 1/weight_sum (zero where weight_sum == 0, so multiplying acts as a
# safe divide).
SmoothingGather = Tuple[csr_matrix, np.ndarray]


def _hemi_smooth_apply(gather: SmoothingGather,
                       data: np.ndarray,       # (N_full, K) fp64
                       ) -> np.ndarray:        # (N_full, K) fp64
    """Apply a gather: ``out = (gather_W @ data) * inv_weight_sum``.
    SpMV + ufuncs only; the caller keeps only the ROI rows.
    """
    gather_W, inv_weight_sum = gather
    out = gather_W @ data
    out *= inv_weight_sum[:, None]
    return out


def cifti_smoothing(data: np.ndarray,
                    medial_mask: np.ndarray,
                    *,
                    lh_gather: SmoothingGather,
                    rh_gather: SmoothingGather,
                    ) -> np.ndarray:
    """Drop-in replacement for ``wb_command -cifti-smoothing`` on the
    surface portion of a CIFTI dtseries.

    Parameters
    ----------
    data : (N_cortex, K) array
        Cortex-only column data. Cast to fp64 internally.
    medial_mask : (N_full,) bool array (lh first, rh second)
        ``True`` = medial wall. Smoothing is per-hemi so kernel mass
        does not cross the medial wall or inter-hemi boundary.
    lh_gather, rh_gather : SmoothingGather
        Per-hemi ``(gather_W, inv_weight_sum)`` from ``Step0Inputs``
        (``lh_smooth_gather`` / ``rh_smooth_gather``).

    Returns
    -------
    smoothed : (N_cortex, K) fp32
    """
    data = np.ascontiguousarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (N_cortex, K), got {data.shape}")
    medial_mask = np.asarray(medial_mask).reshape(-1)
    n_full = medial_mask.shape[0]
    n_lh = lh_gather[0].shape[0]
    n_rh = rh_gather[0].shape[0]
    if n_lh + n_rh != n_full:
        raise ValueError(
            f"medial_mask length {n_full} != lh+rh verts {n_lh + n_rh}")
    cortex_bool = ~medial_mask.astype(bool)
    n_cortex = int(cortex_bool.sum())
    if data.shape[0] != n_cortex:
        raise ValueError(
            f"data has {data.shape[0]} rows; expected N_cortex={n_cortex}")

    K = data.shape[1]
    data64 = data.astype(np.float64, copy=False)
    full = np.zeros((n_full, K), dtype=np.float64)
    full[cortex_bool] = data64

    lh_full = full[:n_lh]
    rh_full = full[n_lh:]

    lh_sm = _hemi_smooth_apply(lh_gather, lh_full)
    rh_sm = _hemi_smooth_apply(rh_gather, rh_full)

    out_full = np.concatenate([lh_sm, rh_sm], axis=0)
    return out_full[cortex_bool].astype(np.float32, copy=False)
