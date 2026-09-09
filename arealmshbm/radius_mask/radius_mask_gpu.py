"""radius_mask_gpu.py

GPU supercall for the radius_mask supergraph. The two heavy SSSP
kernels run different solvers, because the two problems have opposite
shapes. ``add_spatial_constraint`` is K = L_h ≈ 150 problems bounded
at 30 mm — a shallow wave over a fifth of the mesh, which the batched
pull-based Bellman-Ford settles in ~30 sweeps / 8 ms. ``central_sulcus``
is Nsrc ≈ 5000 *unbounded* problems run to full settle, where the same
solver costs 209 sweeps of the whole (V, Nsrc) matrix (1.37 s, at the
card's memory roofline); it runs on the frontier-based
``sssp_batch32`` instead (~50 ms). The remaining small CPU pieces —
parcel CSR build, paracentral/insula classifier, truncate decision
table — stay CPU: heavy control-flow that runs in ~1 ms each, not
worth a kernel.

Hybrid layout per hemi:

    CPU prep:  load mesh + aparc → build_mesh_csr → H2D CSR + labels
               + aparc + parcel_csr.
    GPU:       central_sulcus (batched delta-stepping SSSP, 32 sources
               per CTA, gathering only the relevant-vertex rows), then
               per-source distance reduce → per-parcel mean.
    GPU:       add_spatial_constraint (Bellman-Ford bounded by radius).
    CPU:       classify_central_relevance, truncate (run on D2H'd mask
               + avg_dis; ~1 ms each).

cupy is imported at module top — this file is only loaded via the
``backend == 'gpu'`` dispatch branch in :mod:`.radius_mask`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np
import scipy.io as sio

from ..data_io.load_avg_mesh import load_avg_mesh
from ._common import (_build_mesh_csr, _coerce_labels, _mask_to_csc,
                      _read_aparc)
from ._kernels import (
    build_parcel_csr_kernel,
    classify_central_relevance_kernel,
    truncate_kernel,
)
from ._kernels_gpu import (
    _SSSP_DELTA_MULT,
    bellman_ford_bounded_cupy,
    build_edge_table,
    central_sulcus_distances_cupy,
    init_dist_for_parcels_cupy,
)


def _run_hemi_gpu(hemi: str, mesh_d: dict, labels: np.ndarray,
                   aparc: np.ndarray, radius: float,
                   timings: dict, verbose: bool) -> np.ndarray:
    """Per-hemi: GPU SSSPs (bounded for radius mask + unbounded for
    central sulcus) + CPU classify/truncate. Returns a (V, L_h) uint8
    host array — the post-truncate boundary mask.
    """
    L_h = int(labels.max())
    V = labels.shape[0]
    if verbose:
        print(f"==> {hemi} (gpu) ...")

    # ── Mesh CSR build (CPU) + H2D ──
    t = time.perf_counter()
    indptr_h, indices_h, weights_h = _build_mesh_csr(
        mesh_d["vertices"], mesh_d["faces"], mesh_d["vertexNbors"],
    )
    indptr_dev = cp.asarray(indptr_h)
    indices_dev = cp.asarray(indices_h)
    weights_dev = cp.asarray(weights_h)
    labels_dev = cp.asarray(labels)
    timings[f"build_csr_{hemi}_s"] = time.perf_counter() - t

    parcel_offs, parcel_inds = build_parcel_csr_kernel(labels, L_h)
    pre_verts = np.flatnonzero(aparc == 25).astype(np.int64)
    post_verts = np.flatnonzero(aparc == 23).astype(np.int64)

    # ── Classify (CPU; needs aparc + labels) ──
    t = time.perf_counter()
    relevant_parcels, relevant_verts = classify_central_relevance_kernel(
        labels, aparc, L_h,
    )
    timings[f"classify_{hemi}_s"] = time.perf_counter() - t

    # ── central_sulcus on GPU (batched delta-stepping SSSP) ──
    # Nsrc single-source problems, 32 sources per CTA. Only the rows at
    # ``relevant_verts`` are consumed, so the kernel returns the compact
    # (n_rel, NCOL) gather instead of the full (V, Nsrc) matrix.
    Npre = int(pre_verts.size)
    Npost = int(post_verts.size)
    src_concat = np.concatenate([pre_verts, post_verts]).astype(np.int32)
    rel_verts = np.flatnonzero(relevant_verts).astype(np.int32)

    t = time.perf_counter()
    tab_dev = build_edge_table(indptr_h, indices_h, weights_h)
    delta = _SSSP_DELTA_MULT * float(weights_h.mean(dtype=np.float64))
    dist_rel_dev = central_sulcus_distances_cupy(
        tab_dev, src_concat, rel_verts, V, delta,
    )
    # Per-parcel mean: avg_dis[h, l] = mean over (s_h, v in parcel l) of
    # dist[v, s_h]. Pre slice = cols [0, Npre); post = [Npre, Nsrc).
    # (Columns beyond Nsrc are the block padding — never sliced.)
    sum_pre_rel = cp.asnumpy(
        dist_rel_dev[:, :Npre].astype(cp.float64).sum(axis=1))
    sum_post_rel = cp.asnumpy(
        dist_rel_dev[:, Npre:Npre + Npost].astype(cp.float64).sum(axis=1))
    del dist_rel_dev, tab_dev
    # Row index of each relevant vertex in the gathered matrix. Parcel
    # verts of a relevant parcel are relevant verts by construction
    # (classify_central_relevance_kernel marks the whole parcel), so
    # every lookup below hits a real row.
    row_of = np.full(V, -1, dtype=np.int64)
    row_of[rel_verts] = np.arange(rel_verts.shape[0], dtype=np.int64)
    avg_dis = np.empty((2, L_h), dtype=np.float64)
    Npre_f = float(Npre); Npost_f = float(Npost)
    for l_idx in range(L_h):
        if relevant_parcels[l_idx] == 0:
            avg_dis[0, l_idx] = 0.0
            avg_dis[1, l_idx] = 0.0
            continue
        ps = parcel_offs[l_idx]; pe = parcel_offs[l_idx + 1]
        n_v = pe - ps
        if n_v == 0:
            avg_dis[0, l_idx] = np.nan
            avg_dis[1, l_idx] = np.nan
            continue
        rows = row_of[parcel_inds[ps:pe]]
        avg_dis[0, l_idx] = float(sum_pre_rel[rows].sum()) / (float(n_v) * Npre_f)
        avg_dis[1, l_idx] = float(sum_post_rel[rows].sum()) / (float(n_v) * Npost_f)
    timings[f"central_sulcus_{hemi}_s"] = time.perf_counter() - t

    # ── add_spatial_constraint on GPU (batched BF, bounded by radius) ──
    t = time.perf_counter()
    dist_parc_dev = cp.empty((V, L_h), dtype=cp.float32)
    scratch_parc_dev = cp.empty_like(dist_parc_dev)
    init_dist_for_parcels_cupy(labels_dev, dist_parc_dev)
    bellman_ford_bounded_cupy(
        indptr_dev, indices_dev, weights_dev,
        dist_parc_dev, scratch_parc_dev,
        radius=float(radius), max_iters=500,
    )
    # mask = dist <= radius (fp32 cmp); cast to uint8. Single ufunc.
    out_mask_dev = (dist_parc_dev <= cp.float32(radius)).astype(cp.uint8)
    out_mask = cp.asnumpy(out_mask_dev)
    del dist_parc_dev, scratch_parc_dev, out_mask_dev
    timings[f"add_spatial_constraint_{hemi}_s"] = time.perf_counter() - t

    # ── truncate (CPU; small + heavy control-flow) ──
    t = time.perf_counter()
    truncate_kernel(out_mask, avg_dis, labels, aparc)
    timings[f"truncate_{hemi}_s"] = time.perf_counter() - t
    return out_mask


def generate_radius_mask_gpu(lh_labels: np.ndarray,
                              rh_labels: np.ndarray,
                              mesh: str,
                              radius,
                              out_dir,
                              dtype: np.dtype = np.float32,
                              verbose: bool = True,
                              cbig_code_dir: Optional[str] = None,
                              ) -> dict:
    """GPU port of :func:`radius_mask.generate_radius_mask`. Same output
    schema (``lh_boundary``, ``rh_boundary`` as csc_matrix in
    ``spatial_mask_<mesh>.mat``). ``cbig_code_dir`` is an atlas-dir
    override (legacy name) for the ``<mesh>/label/<hemi>.aparc.annot``
    lookup; ``None`` ⇒ ``MSHBM_ATLAS_DIR``. Mesh geometry comes from the
    shipped avg_mesh bundles, not from here.
    """
    radius_f = float(radius)
    timings: dict = {}

    lh_mesh = load_avg_mesh("lh", mesh, "inflated")
    rh_mesh = load_avg_mesh("rh", mesh, "inflated")
    lh_aparc = _read_aparc("lh", mesh, cbig_code_dir=cbig_code_dir)
    rh_aparc = _read_aparc("rh", mesh, cbig_code_dir=cbig_code_dir)

    lh_labels = _coerce_labels(lh_labels)
    rh_labels = _coerce_labels(rh_labels)

    if verbose:
        print("1. compute geodesic + per-parcel mask (gpu Bellman-Ford)")
    lh_out = _run_hemi_gpu("lh", lh_mesh, lh_labels, lh_aparc, radius_f,
                           timings, verbose)
    rh_out = _run_hemi_gpu("rh", rh_mesh, rh_labels, rh_aparc, radius_f,
                           timings, verbose)

    # Save as csc_matrix(double) — matches the CPU supercall's on-disk schema.
    lh_sparse = _mask_to_csc(lh_out)
    rh_sparse = _mask_to_csc(rh_out)
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
