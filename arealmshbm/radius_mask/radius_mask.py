"""radius_mask.py

Supercall for the radius_mask supergraph: one host frame (mesh and
aparc load, label coercion, ``.mat`` write) around a per-hemisphere
body picked by ``backend``.

CPU body: numba kernels in :mod:`._kernels` called directly.
``backend='gpu'`` takes the body from :mod:`.radius_mask_gpu` — a
frontier delta-stepping SSSP for the central-sulcus family and a
batched pull-based Bellman-Ford on a (V, L_h) distance matrix for the
bounded radius mask. cupy is imported lazily — the CPU path never
touches it. Both backends produce byte-for-byte the same decoded
masks.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import scipy.io as sio

from ..data_io.load_avg_mesh import load_avg_mesh
from ._common import (_build_mesh_csr, _coerce_labels, _mask_to_csc,
                      _read_aparc)
from ._kernels import (
    add_spatial_constraint_kernel,
    central_sulcus_kernel,
    classify_central_relevance_kernel,
    truncate_kernel,
    build_parcel_csr_kernel,
)


def _run_hemi(hemi: str, mesh_d: dict, labels: np.ndarray,
              aparc: np.ndarray, radius: float,
              verbose: bool) -> np.ndarray:
    L_h = int(labels.max())
    V = labels.shape[0]
    if verbose:
        print(f"==> {hemi} ...")

    indptr, indices, weights = _build_mesh_csr(
        mesh_d["vertices"], mesh_d["faces"], mesh_d["vertexNbors"])

    parcel_offs, parcel_inds = build_parcel_csr_kernel(labels, L_h)
    pre_verts = np.flatnonzero(aparc == 25).astype(np.int64)
    post_verts = np.flatnonzero(aparc == 23).astype(np.int64)

    relevant_parcels, relevant_verts = classify_central_relevance_kernel(
        labels, aparc, L_h)

    avg_dis = np.empty((2, L_h), dtype=np.float64)
    central_sulcus_kernel(
        indptr, indices, weights,
        pre_verts, post_verts,
        parcel_offs, parcel_inds,
        L_h,
        relevant_parcels, relevant_verts,
        avg_dis,
        numba.get_num_threads(),
    )

    out_mask = np.zeros((V, L_h), dtype=np.uint8)
    add_spatial_constraint_kernel(
        indptr, indices, weights,
        labels,
        parcel_offs, parcel_inds,
        L_h,
        np.float32(radius),
        out_mask,
        numba.get_num_threads(),
    )

    truncate_kernel(out_mask, avg_dis, labels, aparc)

    return out_mask


def generate_radius_mask(lh_labels: np.ndarray,
                         rh_labels: np.ndarray,
                         radius: float,
                         out_dir,
                         verbose: bool = True,
                         backend: str = "cpu",
                         ) -> dict:
    """Per-parcel radius mask + central-sulcus truncation on the
    fsaverage6 mesh.

    Pipeline (per hemi):

      build_mesh_csr → classify_central_relevance → central_sulcus
      → add_spatial_constraint → truncate.

    The mesh / aparc load, the label coercion and the ``.mat`` write
    are the same host code on both backends; ``backend`` picks the
    per-hemisphere body (:func:`_run_hemi`, numba, or
    :func:`.radius_mask_gpu.run_hemi_gpu`).

    Parameters
    ----------
    lh_labels, rh_labels : (V,) int — per-hemi parcel labels, 1..L
                           with 0 = medial wall.
    radius  : mm.
    out_dir : path; output written to
              ``<out_dir>/spatial_mask/spatial_mask.mat``.
    verbose : print per-stage banners.

    Returns a dict with ``lh_boundary``, ``rh_boundary`` (csc_matrix)
    and ``mat_path``.
    """
    if backend == "gpu":
        from .radius_mask_gpu import run_hemi_gpu as run_hemi
    elif backend == "cpu":
        run_hemi = _run_hemi
    else:
        raise ValueError(f"generate_radius_mask: unknown backend {backend!r}")
    radius_f = float(radius)

    lh_mesh = load_avg_mesh("lh", "inflated")
    rh_mesh = load_avg_mesh("rh", "inflated")
    lh_aparc = _read_aparc("lh")
    rh_aparc = _read_aparc("rh")

    lh_labels = _coerce_labels(lh_labels)
    rh_labels = _coerce_labels(rh_labels)

    if verbose:
        print(f"1. compute geodesic + per-parcel mask ({backend})")
    lh_out = run_hemi("lh", lh_mesh, lh_labels, lh_aparc, radius_f, verbose)
    rh_out = run_hemi("rh", rh_mesh, rh_labels, rh_aparc, radius_f, verbose)

    # MATLAB stores boundary as double sparse; mirror that for read-compat.
    lh_sparse = _mask_to_csc(lh_out)
    rh_sparse = _mask_to_csc(rh_out)

    out_subdir = Path(out_dir) / "spatial_mask"
    out_subdir.mkdir(parents=True, exist_ok=True)
    mat_path = out_subdir / "spatial_mask.mat"
    sio.savemat(str(mat_path),
                {"lh_boundary": lh_sparse, "rh_boundary": rh_sparse},
                do_compression=True)

    if verbose:
        print(f"Saved: {mat_path}")

    return {
        "lh_boundary": lh_sparse,
        "rh_boundary": rh_sparse,
        "mat_path": str(mat_path),
    }
