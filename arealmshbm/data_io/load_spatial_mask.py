"""load_spatial_mask.py

Load ``spatial_mask_<mesh>.mat`` — the per-hemi 30 mm-radius spatial
masks (``lh_boundary``, ``rh_boundary``) used to build the bilateral
block-diagonal ``boundary_mask``.

Each hemisphere is read by :func:`arealmshbm.data_io.mat5_stream.read_sparse`
(MAT v5/v7 and v7.3) and densified to a C-order fp64 array, because the
CPU EM body uses dense fp32/fp64 multiplication.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .mat5_stream import read_sparse


def load_spatial_mask(mask_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``spatial_mask_<mesh>.mat`` and return ``(lh_boundary, rh_boundary)``.

    Parameters
    ----------
    mask_path : path to ``spatial_mask_<mesh>.mat``.

    Returns
    -------
    lh_boundary, rh_boundary : 2D C-order fp64 arrays, shape
        ``(N_hemi, L_hemi)`` in MATLAB convention.
    """
    return tuple(
        np.asarray(read_sparse(mask_path, k).toarray(order="C"),
                   dtype=np.float64)
        for k in ("lh_boundary", "rh_boundary"))
