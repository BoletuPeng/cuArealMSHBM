"""surface_smoothing_gpu.py

CuPy mirror of the per-call hot path in :mod:`.surface_smoothing`. The
Dijkstra / gather assembly stays on CPU (it runs exactly once per hemi
at ``Step0Pipeline.load_inputs`` regardless of backend), and only the
SpMV + per-row scale + ROI mask move to device.

Hot path:
    out[j, k] = sum_i gather_W[j, i] * data[i, k] / weight_sum[j]
                where j outside ROI -> 0.

The CPU path uses fp64 throughout (matches scipy's default SpMV dtype),
but the gather matrix has values in (0, area_max] and the smoothing
output feeds find_minima + watershed, both of which compare values and
do not accumulate. fp32 internal precision is therefore safe; the
correctness assertion is pinned by ``tests/test_gpu_correctness.py``
(max-abs-diff well below the mesh-derivative noise floor).

Precision: fp32 device internals. The returned array is fp32, matching
the CPU contract.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

# Cupy + cupyx imports kept lazy at module-top level — this file is
# only loaded via the ``backend == 'gpu'`` dispatch in
# :mod:`arealmshbm.step0_pipeline.pipeline`.
import cupy as cp
import cupyx.scipy.sparse as cpsp


# A SmoothingGatherGPU bundle mirrors SmoothingGather (the CPU side)
# but with the CSR on device + the inverse weight-sum on device.
SmoothingGatherGPU = Tuple[cpsp.csr_matrix, cp.ndarray]


def prepare_smoothing_gather_gpu(cpu_gather) -> SmoothingGatherGPU:
    """Push a CPU (gather_W, inv_weight_sum) bundle to device as fp32.

    The CPU gather is built once per hemi at ``Step0Pipeline.load_inputs``
    by :func:`.surface_smoothing.prepare_smoothing_gather`. For the GPU
    backend the pipeline calls THIS function right after the CPU build
    and caches the device-resident bundle on ``Step0Inputs``.

    Parameters
    ----------
    cpu_gather : (scipy.sparse.csr_matrix, np.ndarray)
        The CPU side bundle.

    Returns
    -------
    (gather_W_d, inv_weight_sum_d) : both fp32 device-resident.
    """
    gather_W, inv_weight_sum = cpu_gather
    # Build the device CSR as fp32. cupyx accepts a scipy CSR directly
    # for the constructor; setting dtype in advance avoids one device
    # roundtrip vs the .astype path.
    gW_f32 = gather_W.astype(np.float32, copy=False)
    gather_W_d = cpsp.csr_matrix(gW_f32)
    inv_weight_sum_d = cp.asarray(inv_weight_sum, dtype=cp.float32)
    return gather_W_d, inv_weight_sum_d


def cifti_smoothing_gpu(data: Union[np.ndarray, cp.ndarray],
                        medial_mask: np.ndarray,
                        *,
                        lh_gather_gpu: SmoothingGatherGPU,
                        rh_gather_gpu: SmoothingGatherGPU,
                        ) -> cp.ndarray:
    """GPU twin of :func:`.surface_smoothing.cifti_smoothing`.

    Same signature except the gather bundles are device-resident.
    ``data`` may be either a host array or a device ``cp.ndarray``;
    the production caller passes a device array straight from the
    GPU accumulator chain, so ``cp.asarray`` is a no-op alias.

    Returns a device-resident ``cp.ndarray`` (PR #56-pattern
    device-stay refactor). The downstream consumer ``find_minima_gpu``
    + ``watershed_edge_count_gpu`` both accept device input; tests
    cp.asnumpy at their assertion boundary.

    Parameters
    ----------
    data : (N_cortex, K) fp32, host or device
        Cortex-only column data.
    medial_mask : (N_full,) bool
        ``True`` = medial wall. Smoothing is per-hemi; this is just the
        scatter/gather between cortex-only and full-mesh layouts.
    lh_gather_gpu, rh_gather_gpu : SmoothingGatherGPU
        Per-hemi cached device-resident bundles.

    Returns
    -------
    smoothed : (N_cortex, K) device fp32
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (N_cortex, K), got {data.shape}")
    medial_mask = np.asarray(medial_mask).reshape(-1)
    n_full = medial_mask.shape[0]
    n_lh = lh_gather_gpu[0].shape[0]
    n_rh = rh_gather_gpu[0].shape[0]
    if n_lh + n_rh != n_full:
        raise ValueError(
            f"medial_mask length {n_full} != lh+rh verts {n_lh + n_rh}")
    cortex_bool = ~medial_mask.astype(bool)
    n_cortex = int(cortex_bool.sum())
    if data.shape[0] != n_cortex:
        raise ValueError(
            f"data has {data.shape[0]} rows; expected N_cortex={n_cortex}")
    K = data.shape[1]

    # H2D (if data is host) + scatter into full-mesh layout on device.
    # When the caller passes a device cp.ndarray fp32 (production path),
    # cp.asarray is an alias — no copy, no stream sync.
    data_d = cp.asarray(data, dtype=cp.float32)
    full_d = cp.zeros((n_full, K), dtype=cp.float32)
    cortex_bool_d = cp.asarray(cortex_bool)
    full_d[cortex_bool_d] = data_d

    lh_full_d = full_d[:n_lh]
    rh_full_d = full_d[n_lh:]
    lh_roi_d = cortex_bool_d[:n_lh]
    rh_roi_d = cortex_bool_d[n_lh:]

    # Per-hemi: SpMM + scale + ROI mask.
    lh_gW_d, lh_inv_d = lh_gather_gpu
    rh_gW_d, rh_inv_d = rh_gather_gpu
    lh_out_d = lh_gW_d @ lh_full_d
    lh_out_d *= lh_inv_d[:, None]
    lh_out_d[~lh_roi_d] = 0.0
    rh_out_d = rh_gW_d @ rh_full_d
    rh_out_d *= rh_inv_d[:, None]
    rh_out_d[~rh_roi_d] = 0.0

    out_full_d = cp.concatenate([lh_out_d, rh_out_d], axis=0)
    # Re-mask: gather cortex rows → (N_cortex, K) fp32. Output stays on
    # device — find_minima_gpu + watershed_edge_count_gpu consume it
    # directly.
    out_cortex_d = out_full_d[cortex_bool_d]
    return out_cortex_d
