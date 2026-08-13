"""load_subject_gradient.py — gradient .mat reader helper.

Provides the private leaf used by
:class:`arealmshbm.step2_io.subject_loaders.SubjectGradientLoader`
on its legacy ``.mat`` fallback path:

* :func:`_read_emb_100` — read one per-hemi diffusion-embedding ``.mat``
  → ``(V_h, n_components)`` fp32. Handles both legacy MATLAB (v5/v7)
  and v7.3 (HDF5) container formats.

fp32 throughout.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_emb_100(path: str | Path, n_components: int = 100) -> np.ndarray:
    """Load a CBIG diffusion-embedding ``.mat`` → (V_h, n_components) fp32.

    Handles both legacy MATLAB (v5/v7) and v7.3 (HDF5) files. Mirrors
    the dispatch shape in :func:`arealmshbm.data_io.fetch_data._read_gradient_emb`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"gradient mat not found: {p}")
    try:
        from scipy.io import loadmat
        m = loadmat(str(p), squeeze_me=False)
        if "emb" not in m:
            raise KeyError(f"gradient mat missing 'emb' field: {p}")
        emb = np.asarray(m["emb"])
    except (NotImplementedError, ValueError):
        # v7.3 HDF5 — h5py reads transposed wrt MATLAB column-major.
        import h5py
        with h5py.File(str(p), "r") as f:
            if "emb" not in f:
                raise KeyError(f"gradient mat missing 'emb' field: {p}")
            emb = np.asarray(f["emb"]).T

    if emb.ndim != 2:
        raise ValueError(
            f"gradient 'emb' must be 2-D; got shape {emb.shape} from {p}"
        )
    if emb.shape[1] < n_components:
        raise ValueError(
            f"gradient 'emb' has {emb.shape[1]} cols; need >= {n_components} "
            f"(from {p})"
        )
    return np.ascontiguousarray(emb[:, :n_components], dtype=np.float32)


