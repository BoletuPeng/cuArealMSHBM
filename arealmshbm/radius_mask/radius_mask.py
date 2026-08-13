"""radius_mask.py

Supercall for the radius_mask supergraph.

CPU path: numba kernels in :mod:`._kernels` called directly.
Dispatch on ``backend='gpu'`` defers to :mod:`.radius_mask_gpu` —
batched pull-based Bellman-Ford on a (V, K) distance matrix per
problem family (parcels for the radius mask, source verts for the
central-sulcus mean). cupy is imported lazily — the CPU path never
touches it.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from ..data_io.load_avg_mesh import load_avg_mesh
from ._common import _build_mesh_csr, _coerce_labels, _read_aparc
from ._kernels import (
    add_spatial_constraint_kernel,
    central_sulcus_kernel,
    classify_central_relevance_kernel,
    truncate_kernel,
    build_parcel_csr_kernel,
)


def _run_hemi(hemi: str, mesh_d: dict, labels: np.ndarray,
              aparc: np.ndarray, radius: float,
              timings: dict, verbose: bool) -> np.ndarray:
    L_h = int(labels.max())
    V = labels.shape[0]
    if verbose:
        print(f"==> {hemi} ...")

    t = time.perf_counter()
    indptr, indices, weights = _build_mesh_csr(mesh_d["vertices"], mesh_d["faces"])
    timings[f"build_csr_{hemi}_s"] = time.perf_counter() - t

    parcel_offs, parcel_inds = build_parcel_csr_kernel(labels, L_h)
    pre_verts = np.flatnonzero(aparc == 25).astype(np.int64)
    post_verts = np.flatnonzero(aparc == 23).astype(np.int64)

    t = time.perf_counter()
    relevant_parcels, relevant_verts = classify_central_relevance_kernel(
        labels, aparc, L_h)
    timings[f"classify_{hemi}_s"] = time.perf_counter() - t

    avg_dis = np.empty((2, L_h), dtype=np.float64)
    t = time.perf_counter()
    central_sulcus_kernel(
        indptr, indices, weights,
        pre_verts, post_verts,
        parcel_offs, parcel_inds,
        L_h,
        relevant_parcels, relevant_verts,
        avg_dis,
    )
    timings[f"central_sulcus_{hemi}_s"] = time.perf_counter() - t

    out_mask = np.zeros((V, L_h), dtype=np.uint8)
    t = time.perf_counter()
    add_spatial_constraint_kernel(
        indptr, indices, weights,
        labels,
        parcel_offs, parcel_inds,
        L_h,
        np.float32(radius),
        out_mask,
    )
    timings[f"add_spatial_constraint_{hemi}_s"] = time.perf_counter() - t

    t = time.perf_counter()
    truncate_kernel(out_mask, avg_dis, labels, aparc)
    timings[f"truncate_{hemi}_s"] = time.perf_counter() - t

    return out_mask


def generate_radius_mask(lh_labels: np.ndarray,
                         rh_labels: np.ndarray,
                         mesh: str,
                         radius,
                         out_dir,
                         dtype: np.dtype = np.float32,
                         verbose: bool = True,
                         backend: str = "cpu",
                         cbig_code_dir: Optional[str] = None,
                         ) -> dict:
    """Per-parcel radius mask + central-sulcus truncation on a
    fsaverage* mesh.

    Pipeline (per hemi, fully numba-jit'd):

      build_mesh_csr → classify_central_relevance → central_sulcus
      → add_spatial_constraint → truncate.

    ``backend='gpu'`` dispatches to :func:`radius_mask_gpu.generate_radius_mask_gpu`.

    Parameters
    ----------
    lh_labels, rh_labels : (V,) int — per-hemi parcel labels, 1..L
                           with 0 = medial wall.
    mesh    : 'fsaverage6' / 'fsaverage5' / 'fsaverage'.
    radius  : float or string mm (e.g. 30 or '30').
    out_dir : path; output written to
              ``<out_dir>/spatial_mask/spatial_mask_<mesh>.mat``.
    dtype   : kept for signature parity; kernels run fp32 internally.
    verbose : print per-stage banners.
    cbig_code_dir : atlas-dir override (legacy name) for the
                    ``<mesh>/label/<hemi>.aparc.annot`` lookup; ``None`` ⇒
                    ``MSHBM_ATLAS_DIR``. Mesh geometry comes from the
                    shipped avg_mesh bundles, not from here.

    Returns a dict with ``lh_boundary``, ``rh_boundary`` (csc_matrix),
    ``mat_path``, and per-stage ``timings``.
    """
    if backend == "gpu":
        from .radius_mask_gpu import generate_radius_mask_gpu
        return generate_radius_mask_gpu(
            lh_labels=lh_labels, rh_labels=rh_labels, mesh=mesh,
            radius=radius, out_dir=out_dir, dtype=dtype, verbose=verbose,
            cbig_code_dir=cbig_code_dir,
        )
    if backend != "cpu":
        raise ValueError(f"generate_radius_mask: unknown backend {backend!r}")
    radius_f = float(radius)
    timings: dict = {}

    lh_mesh = load_avg_mesh("lh", mesh, "inflated")
    rh_mesh = load_avg_mesh("rh", mesh, "inflated")
    lh_aparc = _read_aparc("lh", mesh, cbig_code_dir=cbig_code_dir)
    rh_aparc = _read_aparc("rh", mesh, cbig_code_dir=cbig_code_dir)

    lh_labels = _coerce_labels(lh_labels)
    rh_labels = _coerce_labels(rh_labels)

    if verbose:
        print("1. compute geodesic + per-parcel mask (numba kernels)")
    lh_out = _run_hemi("lh", lh_mesh, lh_labels, lh_aparc, radius_f,
                       timings, verbose)
    rh_out = _run_hemi("rh", rh_mesh, rh_labels, rh_aparc, radius_f,
                       timings, verbose)

    # MATLAB stores boundary as double sparse; mirror that for read-compat.
    lh_sparse = sp.csc_matrix(lh_out.astype(np.float64))
    rh_sparse = sp.csc_matrix(rh_out.astype(np.float64))

    out_dir = Path(out_dir)
    out_subdir = out_dir / "spatial_mask"
    out_subdir.mkdir(parents=True, exist_ok=True)
    mat_path = out_subdir / f"spatial_mask_{mesh}.mat"
    sio.savemat(str(mat_path),
                {"lh_boundary": lh_sparse, "rh_boundary": rh_sparse},
                do_compression=True)

    if verbose:
        print(f"Saved: {mat_path}")

    return {
        "lh_boundary": lh_sparse,
        "rh_boundary": rh_sparse,
        "mat_path": str(mat_path),
        "timings": timings,
    }
