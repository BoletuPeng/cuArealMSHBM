"""inputs_bundle.py

Reader for the shipped step-0 inputs bundle — the subject-invariant
part of ``Step0Inputs`` on fsaverage6: sphere + midthickness meshes,
medial mask, smoothing / gradient mesh prep, the Gaussian smoothing
gather (sigma = 2.55 mm), and the cortex neighbor table with its 3-hop
expansion — the CBIG canonical call.

Layout::

    <root>/step0_inputs/
        manifest.json       # version, file inventory
        medial_mask.npy
        lh_sphere_verts.npy
        ...
        lh_smooth_prep_0.npy ... lh_smooth_prep_6.npy   # tuple, 7 arrays
        lh_grad_prep_0.npy   ... lh_grad_prep_3.npy     # tuple, 4 arrays
        lh_smooth_gather_csr_data.npy                   # CSR triple
        lh_smooth_gather_csr_indices.npy
        lh_smooth_gather_csr_indptr.npy
        lh_smooth_gather_inv.npy                        # inv_weight_sum (N,)
        ...

``<root>`` is ``arealmshbm/data/precomputed/``, or
``$MSHBM_PRECOMPUTED_ROOT`` when that is set (the root override the
avg_mesh bundles use).

The bundle is an ASSET, not a rebuildable cache — the contract of
:mod:`arealmshbm.data_io.load_avg_mesh`. It is gitignored and staged
by the installer; runtime never computes one. A missing bundle is a
broken install: a hard ``FileNotFoundError``.

Files are raw ``.npy`` (no compression), so a load is the OS reading
bytes off disk — ~30-100 ms for the ~158 MB bundle on a warm
filesystem.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


BUNDLE_VERSION = "1"


def bundle_dir() -> Path:
    """Directory of the shipped step-0 inputs bundle. Override via env."""
    env = os.getenv("MSHBM_PRECOMPUTED_ROOT")
    if env:
        return Path(env) / "step0_inputs"
    # arealmshbm/step0_pipeline/inputs_bundle.py → arealmshbm/ → data/...
    return (Path(__file__).parent.parent
            / "data" / "precomputed" / "step0_inputs")


def load_inputs_bundle():
    """Load a ``Step0Inputs`` (CPU-only, no GPU mirrors) from the bundle
    at :func:`bundle_dir`.

    Returns
    -------
    inputs : Step0Inputs
        With ``*_gpu`` fields set to ``None``. The caller
        (``Step0Pipeline.load_inputs``) populates GPU mirrors via
        :func:`_build_gpu_mirrors`.

    Raises
    ------
    FileNotFoundError
        If no bundle is staged. This is a broken install, not a
        recoverable cache miss.
    ValueError
        If the staged bundle's format version is not this code's.
    """
    from .pipeline import Step0Inputs

    bundle = bundle_dir()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"step-0 inputs bundle missing: {bundle}. This is a shipped "
            f"asset, not a cache — reinstall to restage it; runtime does "
            f"not compute one.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("version") != BUNDLE_VERSION:
        raise ValueError(
            f"step-0 inputs bundle version mismatch at {bundle}: "
            f"on-disk='{manifest.get('version')}' code='{BUNDLE_VERSION}'. "
            f"Restage the bundle that matches this code.")

    kwargs: dict = dict(manifest["scalars"])

    # Plain arrays.
    for name, fn in manifest["arrays"].items():
        kwargs[name] = np.load(bundle / fn, allow_pickle=False)

    # Tuples.
    for name, fns in manifest["tuples"].items():
        kwargs[name] = tuple(
            np.load(bundle / fn, allow_pickle=False) for fn in fns)

    # CSR matrices.
    for name, d in manifest["csr"].items():
        data = np.load(bundle / d["data"], allow_pickle=False)
        indices = np.load(bundle / d["indices"], allow_pickle=False)
        indptr = np.load(bundle / d["indptr"], allow_pickle=False)
        inv = np.load(bundle / d["inv_weight_sum"], allow_pickle=False)
        csr = csr_matrix((data, indices, indptr),
                          shape=tuple(d["shape"]), copy=False)
        kwargs[name] = (csr, inv)

    # The bundle holds no GPU mirrors — the dataclass defaults them to
    # None; the caller populates them when backend == 'gpu'.
    return Step0Inputs(**kwargs)
