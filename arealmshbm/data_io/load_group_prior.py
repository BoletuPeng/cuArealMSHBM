"""load_group_prior.py

Load the gMSHBM group prior ``Params_Final.mat`` (Mode A).

The group prior holds the trained ``mu``, ``theta``, ``epsil``, ``sigma``
arrays. In Mode A these are fixed throughout step 3; the EM only updates
subject-level state.

Reads typical v7 ``.mat`` via :func:`scipy.io.loadmat`; falls back to
``h5py`` for the v7.3 case (the HCP-prior files at
``arealmshbm/data/group_priors/HCP_fsaverage6_40sub/<num_ROIs>/gMSHBM/beta<X>/Params_Final.mat``
are v7).

Output dtype is fp32 by intent — the EM hot path is fp32 throughout,
so promoting fp64 priors here would just double memory and cast back at
the first kernel call.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _densify(x):
    """MATLAB-sparse field → dense ndarray; dense fields pass through.

    ``Params.theta`` / ``Params.mu`` may have been written as MATLAB
    sparse arrays by some writer (a MATLAB-side ``sparse()``, say — the
    in-tree step-2 writer always densifies), in which case ``scipy.io.loadmat``
    hands this reader a ``csc_array``/``csc_matrix`` and
    ``np.ascontiguousarray`` on one of those raises ``ValueError:
    setting an array element with a sequence``. Mirrors
    ``load_spatial_mask._to_dense``.
    """
    return x.toarray() if hasattr(x, "toarray") else x


def load_group_prior(prior_path: str | Path) -> Dict[str, np.ndarray]:
    """Read ``Params_Final.mat`` and return the four Mode-A fields.

    Parameters
    ----------
    prior_path : path to ``Params_Final.mat``.

    Returns
    -------
    prior : dict with keys
        ``mu``    — (D, L) fp32 group cluster centroids
        ``theta`` — (N, L) fp32 group spatial prior
        ``epsil`` — (1, L) fp32 inter-subject vMF concentration
        ``sigma`` — (1, L) fp32 intra-subject vMF concentration
    """
    p = Path(prior_path)
    if not p.exists():
        raise FileNotFoundError(f"group prior not found: {p}")

    try:
        from scipy.io import loadmat
        m = loadmat(p, squeeze_me=False)
        params = m["Params"]
        # Older v7 files come back as a structured ndarray; access via field names.
        mu     = np.ascontiguousarray(_densify(params["mu"][0, 0]),    dtype=np.float32)
        theta  = np.ascontiguousarray(_densify(params["theta"][0, 0]), dtype=np.float32)
        epsil  = np.asarray(params["epsil"][0, 0]).reshape(1, -1).astype(np.float32)
        sigma  = np.asarray(params["sigma"][0, 0]).reshape(1, -1).astype(np.float32)
    except (NotImplementedError, ValueError):
        # v7.3 path.
        import h5py
        with h5py.File(p, "r") as f:
            # h5py stores transposed; un-transpose to MATLAB shape.
            mu    = np.ascontiguousarray(np.asarray(f["Params/mu"]).T,    dtype=np.float32)
            theta = np.ascontiguousarray(np.asarray(f["Params/theta"]).T, dtype=np.float32)
            epsil = np.asarray(f["Params/epsil"]).reshape(1, -1).astype(np.float32)
            sigma = np.asarray(f["Params/sigma"]).reshape(1, -1).astype(np.float32)

    return {"mu": mu, "theta": theta, "epsil": epsil, "sigma": sigma}
