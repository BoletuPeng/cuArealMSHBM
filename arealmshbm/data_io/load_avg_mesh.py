"""load_avg_mesh.py

Load an fsaverage average-mesh bundle — the inflated surface (used by
the connectivity BFS) and the sphere (used by ``spatial_xyz_prior``).

Returns the same fields as the MATLAB ``CBIG_ReadNCAvgMesh``:
    vertices    : (3, N) coords
    vertexNbors : (max_neigh, N) int, 1-indexed, 0 = absent slot
    MARS_label  : (1, N) int — 1 for medial wall, 2 for cortex
    faces       : (F, 3) int, 0-indexed

``vertexNbors`` ordering inside each column is not MATLAB
``MARS2_readSbjMesh``'s ring-order, but only the SET of neighbors
matters for both downstream consumers (V_lambda's Potts stencil uses
every non-zero slot uniformly; ``check_connectedness`` treats it as a
graph adjacency).

**On-disk format is the single source of truth.** Each ``(hemi, mesh,
surface)`` bundle is a shipped ``.npz`` asset under
``arealmshbm/data/precomputed/avg_mesh/`` (override via
``$MSHBM_PRECOMPUTED_ROOT``). There is NO runtime rebuild from raw
FreeSurfer ``surf/`` + ``cortex.label`` — those are build-time inputs,
not a runtime fallback. A missing bundle is a broken install, not a
cache miss: reinstall, or regenerate it out-of-band (the raw-read
builder lives in git history at the commit that assetized this module).

Atlas dir ``MSHBM_ATLAS_DIR`` is still required at runtime, but only for
the live ``label/*.annot`` reads (aparc + Schaefer) in step 1 — never
for mesh geometry, which comes wholly from these bundles.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal

import numpy as np


def _atlas_dir() -> Path:
    env = os.getenv("MSHBM_ATLAS_DIR")
    if not env:
        raise RuntimeError(
            "MSHBM_ATLAS_DIR is unset or empty. Point it at a directory "
            "whose per-mesh subdirs hold `<targ_mesh>/label/` — the "
            "FreeSurfer annot files read live in step 1 "
            "(`<hemi>.aparc.annot` and `<hemi>.Schaefer2018_*.annot`). "
            "Mesh geometry does NOT come from here — it comes from the "
            "shipped avg_mesh .npz bundles. The studio-app runner exports "
            "this from studio-app/python/resources/atlas/; see that dir's "
            "README."
        )
    return Path(env)


# ─────────────────────────────────────────────────────────────────────
# Shipped avg-mesh bundles. Each (hemi, mesh, surface) is one ``.npz``
# (uncompressed — a zip of raw .npy entries) holding vertices / faces /
# MARS_label / vertexNbors.
#
# These are ASSETS, not a rebuildable cache: ``vertexNbors`` has no
# source file (it is derived from faces) and MARS_label is synthesized
# across surf + cortex.label, so a bundle cannot be re-expressed as "read
# file X". The bundles are gitignored (installer's job — keeps GitHub
# small), so a fresh checkout has none; they must be staged by the
# installer, exactly like game textures. A missing bundle is a broken
# install, resolved by reinstalling — never by recomputing at runtime.
#
# Bundle key: (hemi, mesh, surface). Pointing at a different atlas
# revision ⇒ restage the bundles (or redirect via
# ``$MSHBM_PRECOMPUTED_ROOT``).
# ─────────────────────────────────────────────────────────────────────
def _avg_mesh_bundle_dir() -> Path:
    """Root dir holding the shipped avg_mesh bundles. Override via env."""
    env = os.getenv("MSHBM_PRECOMPUTED_ROOT")
    if env:
        return Path(env) / "avg_mesh"
    # arealmshbm/data_io/load_avg_mesh.py → arealmshbm/ → data/...
    return (Path(__file__).parent.parent
            / "data" / "precomputed" / "avg_mesh")


def _avg_mesh_bundle_file(hemi: str, mesh: str, surface: str) -> Path:
    return _avg_mesh_bundle_dir() / f"{hemi}_{mesh}_{surface}.npz"


def load_avg_mesh(hemi: Literal["lh", "rh"],
                  mesh: str = "fsaverage6",
                  surface: Literal["inflated", "sphere"] = "inflated",
                  ) -> Dict[str, np.ndarray]:
    """Load one fsaverage mesh bundle (one hemi, one surface).

    Results are cached in-process by ``(hemi, mesh, surface)``; repeated
    calls within a session avoid the ~5-15 ms bundle read. Callers MUST
    treat the returned arrays as read-only; the project does so already.

    Parameters
    ----------
    hemi    : 'lh' or 'rh'.
    mesh    : 'fsaverage6' (default), 'fsaverage5', etc.
    surface : 'inflated' (connectivity / vertexNbors) or 'sphere'
              (spatial_xyz_prior coords).

    Returns
    -------
    mesh_dict : dict with keys
        ``vertices``    — (3, N) fp64
        ``vertexNbors`` — (max_neigh, N) int64 (1-indexed; 0 = absent)
        ``MARS_label``  — (N,) int64 (1 = medial wall, 2 = cortex)
        ``faces``       — (F, 3) int64 (0-indexed)

    Raises
    ------
    FileNotFoundError
        If the ``.npz`` bundle is not staged. This is a broken install,
        not a recoverable cache miss — reinstall (or restage the bundle).
    """
    if hemi not in ("lh", "rh"):
        raise ValueError(f"hemi must be 'lh' or 'rh'; got {hemi!r}")
    if surface not in ("inflated", "sphere"):
        raise ValueError(f"surface must be 'inflated' or 'sphere'; got {surface!r}")
    return _load_avg_mesh_cached(hemi, mesh, surface)


@lru_cache(maxsize=16)
def _load_avg_mesh_cached(hemi: str, mesh: str, surface: str
                           ) -> Dict[str, np.ndarray]:
    p = _avg_mesh_bundle_file(hemi, mesh, surface)
    if not p.exists():
        raise FileNotFoundError(
            f"avg_mesh bundle missing for ({hemi}, {mesh}, {surface}): {p}. "
            f"This is a shipped asset, not a cache — reinstall to restage "
            f"it. (The raw-FreeSurfer builder that produces these bundles "
            f"lives in git history at the commit that assetized "
            f"load_avg_mesh; runtime does not rebuild from source.)"
        )
    with np.load(p, allow_pickle=False) as z:
        out = {
            "vertices":    z["vertices"],
            "vertexNbors": z["vertexNbors"],
            "MARS_label":  z["MARS_label"],
            "faces":       z["faces"],
        }
    # Freeze so a downstream mutation raises instead of silently poisoning
    # the lru_cache for every subsequent call in this process. Callers
    # that legitimately need to mutate (rare) must copy first.
    for v in out.values():
        v.setflags(write=False)
    return out
