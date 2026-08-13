"""fc_similarity_gpu.py

CuPy GPU port of the fc_similarity supercall. End-to-end device
residency for the per-session compute of one
``compute_FC_simi_block``: BOLD data and t_series live in device
memory for the lifetime of one session's outer loop; the three big
sgemms (``t @ s_series_A``, ``t @ s_series_B``, ``FC_B.T @ FC_A``) run
on cuBLAS sgemm; the per-iter_a ``FC_simi_block`` is returned as a
device-resident ``cp.ndarray`` and handed off straight to
``cifti_gradient_gpu`` — skipping a D2H sync plus the matching H2D
inside the consumer (~24 round-trips per subject on the YS cohort).

Precision contract matches the CPU path in :mod:`.fc_similarity`
**algorithmically, not bitwise** — every accumulator is fp64 and every
stored array is fp32 in both, but two rounding points differ by ≤1 ULP
per element:

    * column-wise mean accumulates in fp64 in both. CPU subtracts the
      fp64 mean from the fp32 cell (producing an fp64 delta, then
      quantizing on store, and accumulating ``ss`` from the fp64 delta).
      GPU casts the mean to fp32 *before* subtracting and accumulates
      ``ss`` from the fp32-stored value. The ≤1-ULP difference between
      the two stored values is the source of the drift.
    * L2 sum-of-squares is fp64 → fp32 sqrt on both — same path.
    * Final per-block normalization is fp32 throughout. CPU does one
      divide by a precomputed outer product ``mag_b @ mag_a``; GPU does
      two sequential broadcast divides. Mathematically equivalent;
      ≤1 ULP per element apart.
    * Matmul is fp32 throughout (cublasSgemm — no Tensor-core fp16
      implicit downcast).

Empirically the cascaded drift is well inside the step-0 RNG /
orientation tolerance documented in
``docs/step0_flow_and_subgraphs.md``; the downstream ICC scorer
(maintained out of tree) is the binding correctness gate after MATLAB
decoupling.

cupy import lives at module top — this file is only loaded via the
``backend == 'gpu'`` dispatch branch in
:mod:`arealmshbm.step0_pipeline.pipeline`, so the CPU path
never pays the cupy import.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import cupy as cp
import numpy as np


def _demean_norm_columns(x_d: cp.ndarray) -> cp.ndarray:
    """Column-wise demean + L2 norm of a (T, K) fp32 device array.

    Backed by a fused RawKernel in ``_kernels_fused_gpu.py`` —
    accumulates mean + SS in fp64 registers across two passes through
    the fp32 buffer, no fp64 intermediate materialization. ~7x faster
    than the prior ``astype(fp64).mean / .sum`` chain at the
    fc_similarity FC_B shape (T=240, K=7833).

    Precision: ``x_d`` post-demean is bit-identical to the prior
    impl per cell (same ``x - float(mean64)`` arithmetic). ``mag``
    differs by ≤1 ULP fp32 due to a different reduction-tree shape;
    well within the step-0 RNG / Lanczos algorithm-class tolerance
    already documented in ``docs/step0_flow_and_subgraphs.md``.
    """
    from ._kernels_fused_gpu import fused_demean_norm_columns_cupy
    return fused_demean_norm_columns_cupy(x_d)


def _demean_norm_rows(x_d: cp.ndarray) -> cp.ndarray:
    """Row-wise demean + L2 norm of a (K, T) fp32 device array.

    Backed by a fused RawKernel — see :func:`_demean_norm_columns`.
    Same precision contract.
    """
    from ._kernels_fused_gpu import fused_demean_norm_rows_cupy
    return fused_demean_norm_rows_cupy(x_d)


def compute_t_series_gpu(
    curr_data_d: cp.ndarray,            # (N_cortex, T) fp32 on device
    randinds_FC: np.ndarray,            # (N2,) int — host indices
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Build ``(t_series_d, mag_t_d)`` on device. Mirrors MATLAB
    lines 317-319 of ``CBIG_SPGrad_RSFC_gradients.m``:
        t_series = curr_data(randinds_FC, :);
        t_series = bsxfun(@minus, t_series, mean(t_series, 2));
        mag_t = sqrt(sum(t_series.^2, 2));

    Returns
    -------
    t_series_d : (N2, T) fp32 device
    mag_t_d    : (N2, 1) fp32 device
    """
    assert curr_data_d.dtype == cp.float32, "curr_data_d must be fp32"
    idx_d = cp.asarray(np.ascontiguousarray(
        np.asarray(randinds_FC).reshape(-1), dtype=np.int64))
    t_series_d = curr_data_d[idx_d].copy()              # (N2, T) fp32
    mag_flat = _demean_norm_rows(t_series_d)
    return t_series_d, mag_flat.reshape(-1, 1)


def compute_FC_simi_block_gpu(
    curr_data_d: cp.ndarray,            # (N_cortex, T) fp32 on device
    t_series_d: cp.ndarray,             # (N2, T)      fp32 on device
    mag_t_d: cp.ndarray,                # (N2, 1)      fp32 on device
    randinds_verts: np.ndarray,         # host int
    block_a_index: int,
    *,
    num_blocks_a: int = 3,
    num_blocks_b: int = 10,
) -> cp.ndarray:
    """GPU twin of :func:`fc_similarity.compute_FC_simi_block`.

    Returns the ``(N_cortex, block_a_size)`` fp32 ``FC_simi_block`` as
    a device-resident ``cp.ndarray``. The downstream consumer
    ``cifti_gradient_gpu`` is itself a CuPy RawKernel, so handing off
    on-device skips one implicit D2H sync plus the matching H2D inside
    that kernel — ~24 round-trips per subject on the YS cohort. All
    intermediates (FC_A, FC_B, s_series_*, scratch demean/norm passes)
    likewise stay on device.
    """
    assert curr_data_d.dtype == cp.float32, "curr_data_d must be fp32"
    assert t_series_d.dtype == cp.float32, "t_series_d must be fp32"
    assert mag_t_d.dtype == cp.float32, "mag_t_d must be fp32"

    n_cortex, T = curr_data_d.shape
    rv = np.ascontiguousarray(np.asarray(randinds_verts).reshape(-1),
                              dtype=np.int64)
    n1 = rv.shape[0]
    a, b = int(num_blocks_a), int(num_blocks_b)
    iter_a = a + (1 if (n1 % a) != 0 else 0)
    iter_b = b + (1 if (n_cortex % b) != 0 else 0)
    block_size_a = n1 // a
    block_size_b = n_cortex // b
    if not (0 <= block_a_index < iter_a):
        raise ValueError(
            f"block_a_index {block_a_index} out of range [0, {iter_a})")

    from ._kernels_fused_gpu import (
        fused_div_demean_norm_columns_cupy,
        fused_div_nanclamp_cupy,
    )

    mt_flat_d = mag_t_d.reshape(-1).astype(cp.float32, copy=False)

    # ---- FC_A for block_a_index (built once, reused across j) -------
    a_start = block_a_index * block_size_a
    a_end = n1 if block_a_index == iter_a - 1 else (block_a_index + 1) * block_size_a
    a_idx_d = cp.asarray(np.ascontiguousarray(rv[a_start:a_end], dtype=np.int64))
    block_a_size = int(a_idx_d.shape[0])

    # s_series_A = curr_data[a_idx, :].T, with column-wise demean + norm
    s_series_A_d = curr_data_d[a_idx_d].T.copy()        # (T, block_a_size) fp32
    mag_s_A_d = _demean_norm_columns(s_series_A_d)

    # FC_A = (t @ s_series_A); divide by (mt ⊗ mag_s_A) + column demean
    # + L2 norm all fused into one RawKernel pass — see
    # _kernels_fused_gpu.fused_div_demean_norm_columns_cupy.
    FC_A_d = t_series_d @ s_series_A_d                  # (N2, block_a_size) fp32
    mag_a_d = fused_div_demean_norm_columns_cupy(FC_A_d, mt_flat_d, mag_s_A_d)

    # Output buffer stays on device — returned as cp.ndarray.
    FC_simi_block_d = cp.zeros((n_cortex, block_a_size), dtype=cp.float32)

    # ---- block-B loop -----------------------------------------------
    for j in range(iter_b):
        b_start = j * block_size_b
        b_end = n_cortex if j == iter_b - 1 else (j + 1) * block_size_b
        # contiguous slice + transpose on device
        s_series_B_d = curr_data_d[b_start:b_end].T.copy()  # (T, kb) fp32
        mag_s_B_d = _demean_norm_columns(s_series_B_d)

        # Same fusion as FC_A above.
        FC_B_d = t_series_d @ s_series_B_d                  # (N2, kb) fp32
        mag_b_d = fused_div_demean_norm_columns_cupy(FC_B_d, mt_flat_d,
                                                     mag_s_B_d)

        # block = (FC_B.T @ FC_A) / (mag_b ⊗ mag_a) with NaN→0 clamp,
        # written directly into the FC_simi_block slice. Fuses 2
        # broadcast divides + cp.nan_to_num + slice copy = 4 kernel
        # launches collapsed into 1. The cp.nan_to_num that previously
        # ran once at the end is now per-slice (mathematically
        # equivalent — see CPU compute_FC_simi_block for the rationale).
        block_d = FC_B_d.T @ FC_A_d                         # (kb, block_a_size)
        fused_div_nanclamp_cupy(
            block_d, mag_b_d, mag_a_d,
            FC_simi_block_d[b_start:b_end, :])

    return FC_simi_block_d
