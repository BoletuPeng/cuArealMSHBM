"""load_spatial_mask.py

Load ``spatial_mask_<mesh>.mat`` — the per-hemi 30 mm-radius spatial
masks (``lh_boundary``, ``rh_boundary``) used to build the bilateral
block-diagonal ``boundary_mask``.

The MATLAB file holds two sparse arrays. We densify on read because
the EM body uses dense fp32/fp64 multiplication; the sparse advantage
is gone once you densify.

Both v7 (default ``scipy.io`` path) and v7.3 (``h5py`` fallback) input
formats are supported. v7.3 sparse matrices are stored as h5py groups
with CSC-format ``data`` / ``ir`` / ``jc`` and a ``MATLAB_sparse``
group attribute holding the row count — we reconstruct via
``scipy.sparse.csc_matrix`` and densify.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_spatial_mask(mask_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``spatial_mask_<mesh>.mat`` and return ``(lh_boundary, rh_boundary)``.

    Parameters
    ----------
    mask_path : path to ``spatial_mask_<mesh>.mat``.

    Returns
    -------
    lh_boundary, rh_boundary : 2D fp64 arrays. Densified if originally sparse.
        Shape is ``(N_hemi, L_hemi)`` in MATLAB convention.
    """
    p = Path(mask_path)
    if not p.exists():
        raise FileNotFoundError(f"spatial mask not found: {p}")

    try:
        from scipy.io import loadmat
        m = loadmat(p, squeeze_me=False)
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(p, "r") as f:
            lh = _read_v73_boundary(f, "lh_boundary")
            rh = _read_v73_boundary(f, "rh_boundary")
        return lh, rh

    def _to_dense(x):
        # scipy.io.loadmat returns a sparse matrix as scipy.sparse type — densify.
        if hasattr(x, "toarray"):
            return np.asarray(x.toarray(), dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    lh = _to_dense(m["lh_boundary"])
    rh = _to_dense(m["rh_boundary"])
    return lh, rh


def _read_v73_boundary(f, key: str) -> np.ndarray:
    """Read one boundary mask from an open h5py.File.

    Handles both the dense case (h5py.Dataset — un-transpose to MATLAB
    shape) and the sparse case (h5py.Group with CSC ``data`` / ``ir`` /
    ``jc``). The masks ARE saved sparse by
    ``CBIG_ArealMSHBM_generate_radius_mask.m``, so v7.3 files always hit
    the Group branch — the dense branch is kept defensively for
    hand-edited files.
    """
    import h5py
    from scipy.sparse import csc_matrix

    obj = f[key]
    if isinstance(obj, h5py.Group):
        # CSC: ``data`` are nz values, ``ir`` are row indices,
        # ``jc`` are column pointers (length n_cols + 1). Row count is
        # in the parent Group's MATLAB_sparse attribute.
        data = np.asarray(obj["data"]).ravel()
        indices = np.asarray(obj["ir"]).ravel()
        indptr = np.asarray(obj["jc"]).ravel()
        n_rows_attr = obj.attrs.get("MATLAB_sparse")
        if n_rows_attr is None:
            raise ValueError(
                f"v7.3 sparse {key!r}: missing MATLAB_sparse attribute"
            )
        n_rows = int(np.asarray(n_rows_attr).ravel()[0])
        n_cols = int(indptr.shape[0]) - 1
        sp = csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))
        return np.ascontiguousarray(sp.toarray(), dtype=np.float64)
    # Dense: h5py is transposed wrt MATLAB.
    arr = np.asarray(obj)
    return np.ascontiguousarray(arr.T, dtype=np.float64)
