"""surface_smoothing.py

Per-hemisphere geodesic Gaussian smoothing — port of HCP Workbench
``src/Files/MetricSmoothingObject.cxx`` (GEO_GAUSS_AREA branch, ROI
variant). The geodesic-distance computation uses the polyhedral
geodesic algorithm class from ``GeodesicHelper.cxx`` — modified
Dijkstra over both the 1-ring edge graph and a precomputed across-face
"unfolded" neighbor graph. See ``_geodesic_kernels.py`` for the
algorithm-class details and citations.

Hot-path strategy:
    * Mesh-only precomputation (vertex areas + Layer 1/2 neighbor CSR
      tables) is built **once per hemi at pipeline load_inputs** by
      :func:`_geodesic_kernels.prepare_smoothing_mesh`. The pipeline
      caches the tuple in ``Step0Inputs`` and passes it back in.
    * The per-source bounded Dijkstra + scatter→gather CSR assembly is
      further hoisted into :func:`prepare_smoothing_gather` — the
      result depends only on ``(mesh-prep, roi, sigma)``, all of which
      are subject-invariant for a fixed mesh + cfg.smooth_sigma. The
      pipeline calls this once per hemi at ``load_inputs`` and caches
      the resulting ``(gather_W, inv_weight_sum)`` on ``Step0Inputs``.
      Previously the gather was rebuilt every ``iter_a`` call
      (~3× per subject), redoing 74k Dijkstras for the same answer.
    * Per call, :func:`cifti_smoothing` does only a SpMV + per-row
      normalize on the cached gather — bandwidth-bound and fast.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numba as nb
import numpy as np
from scipy.sparse import csr_matrix

from ._geodesic_kernels import (
    alloc_scratch,
    all_sources_scatter,
    prepare_smoothing_mesh,
)


# A SmoothingPrep bundle is the 7-tuple returned by
# prepare_smoothing_mesh: (va, n1_indptr, n1_idx, n1_dist,
#                          n2_indptr, n2_idx, n2_dist).
SmoothingPrep = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray, np.ndarray]

# A SmoothingGather bundle is what :func:`prepare_smoothing_gather`
# returns: the cached per-hemi gather CSR + its per-row 1/weight_sum
# (zero where weight_sum == 0, so multiplying acts as a safe divide).
SmoothingGather = Tuple[csr_matrix, np.ndarray]


def prepare_smoothing_gather(prep: SmoothingPrep,
                             roi: np.ndarray,         # (N_full,) bool
                             sigma: float,
                             ) -> SmoothingGather:
    """Build the per-hemi gather CSR and inverse weight-sum vector.

    Inputs are all mesh / config quantities: ``prep`` is the mesh-only
    output of :func:`prepare_smoothing_mesh`, ``roi`` is the per-hemi
    cortex mask derived from the medial mask, and ``sigma`` is the
    Gaussian width. None of these vary across the per-block calls or
    across subjects on a fixed mesh, so the pipeline builds this once
    per hemi at ``load_inputs`` and reuses it for every smoothing call.
    """
    va, n1_indptr, n1_idx, n1_dist, n2_indptr, n2_idx, n2_dist = prep
    n_full = va.shape[0]
    cutoff = float(sigma) * 3.0

    cortex_idx = np.where(roi)[0].astype(np.int32)
    roi_u8 = roi.astype(np.uint8)

    T = nb.get_num_threads()
    (output_t, marked_t, changed_t,
     heap_node_t, heap_dist_t, heap_pos_t,
     pop_nodes_t, pop_dists_t,
     out_src_t, out_tgt_t, out_w_t,
     overflow_t) = alloc_scratch(
         n_full=n_full,
         n_sources=cortex_idx.size,
         num_threads=T,
         cap_per_source=512,
     )
    out_count_t = np.zeros(T, dtype=np.int64)

    all_sources_scatter(
        cortex_idx, roi_u8, va, float(sigma), float(cutoff),
        n1_indptr, n1_idx, n1_dist,
        n2_indptr, n2_idx, n2_dist,
        output_t, marked_t, changed_t,
        heap_node_t, heap_dist_t, heap_pos_t,
        pop_nodes_t, pop_dists_t,
        out_src_t, out_tgt_t, out_w_t,
        out_count_t,
        overflow_t,
    )
    if overflow_t.any():
        raise RuntimeError(
            "surface_smoothing scatter overflow: per-thread emission cap "
            "(cap_per_source=512) exceeded. The cap is an in-source constant — "
            "raise the literal `cap_per_source=512` in this function "
            "(prepare_smoothing_gather) or reduce sigma."
        )

    total = int(out_count_t.sum())
    if total <= 0:
        gather_W = csr_matrix((n_full, n_full), dtype=np.float64)
        inv_weight_sum = np.zeros(n_full, dtype=np.float64)
        return gather_W, inv_weight_sum

    src_idx = np.empty(total, dtype=np.int32)
    tgt_idx = np.empty(total, dtype=np.int32)
    w_all = np.empty(total, dtype=np.float64)
    cur = 0
    for t in range(T):
        n_t = int(out_count_t[t])
        if n_t == 0:
            continue
        src_idx[cur:cur + n_t] = out_src_t[t, :n_t]
        tgt_idx[cur:cur + n_t] = out_tgt_t[t, :n_t]
        w_all[cur:cur + n_t] = out_w_t[t, :n_t]
        cur += n_t

    gather_W = csr_matrix(
        (w_all, (tgt_idx, src_idx)), shape=(n_full, n_full))
    weight_sum = np.asarray(gather_W.sum(axis=1)).reshape(-1)
    inv_weight_sum = np.zeros_like(weight_sum)
    safe = weight_sum > 0
    inv_weight_sum[safe] = 1.0 / weight_sum[safe]
    return gather_W, inv_weight_sum


def _hemi_smooth_apply(gather: SmoothingGather,
                       roi: np.ndarray,        # (N_full,) bool
                       data: np.ndarray,       # (N_full, K) fp64
                       ) -> np.ndarray:        # (N_full, K) fp64
    """Apply a cached gather: ``out = (gather_W @ data) * inv_weight_sum``
    then zero non-ROI rows. SpMV + ufuncs only — no Dijkstra here.
    """
    gather_W, inv_weight_sum = gather
    out = gather_W @ data
    out *= inv_weight_sum[:, None]
    out[~roi] = 0.0
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
        Per-hemi cached ``(gather_W, inv_weight_sum)`` from
        :func:`prepare_smoothing_gather`. Built once per hemi at
        ``load_inputs`` — see module docstring for why this hoist is
        sound (gather depends only on mesh + roi + sigma).

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
    lh_roi = cortex_bool[:n_lh]
    rh_roi = cortex_bool[n_lh:]

    lh_sm = _hemi_smooth_apply(lh_gather, lh_roi, lh_full)
    rh_sm = _hemi_smooth_apply(rh_gather, rh_roi, rh_full)

    out_full = np.concatenate([lh_sm, rh_sm], axis=0)
    return out_full[cortex_bool].astype(np.float32, copy=False)
