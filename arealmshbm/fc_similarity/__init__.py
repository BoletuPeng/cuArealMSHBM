"""fc_similarity

Block-wise FC-similarity matrix.

This implements the inner content of the per-scan loop in
``CBIG_SPGrad_RSFC_gradients.m`` (lines 285–402). For each scan we:

    1. Subsample BOLD time courses with ``randinds_FC`` to form
       ``t_series`` (N2, T) and its row-magnitudes ``mag_t``.
    2. For one block-A index ``i`` (column slice over ``randinds_verts``),
       compute the partial FC matrix ``FC_A = (t_series @ s_series) /
       (mag_t @ mag_s)`` of shape (N2, block_a_size).
    3. Loop over block-B (row slices over ``curr_data``), compute
       ``FC_B`` analogously, and accumulate
       ``FC_simi_block[block_b_slice, :] = corr(FC_B^T, FC_A)``
       (Pearson correlation of columns).

The result ``FC_simi_block`` (N_cortex, block_a_size) is what feeds
``surface_gradient`` next.

Public API:
    compute_t_series       — t_series + mag_t for one scan (CPU numba).
    compute_FC_simi_block  — full FC_simi_block for one (scan, block_a)
                             via numpy ``@`` → OpenBLAS sgemm.

GPU twins live in :mod:`.fc_similarity_gpu` (CuPy / cuBLAS sgemm) and
are imported lazily by :mod:`arealmshbm.step0_pipeline.pipeline`
when ``cfg.backend == 'gpu'`` — the CPU path never pays the cupy import.

Precision policy:
    fp32 storage and matmul throughout (matches MATLAB's ``single``
    casts). The mean-subtraction step is internally promoted to fp64
    (matching MATLAB's single-mean semantics, which use double
    accumulation) and the result is cast back to fp32 — verified
    bit-identical ``t_series`` vs MATLAB and 99-th percentile
    element-wise rel diff <= 1e-4 on FC_A / FC_simi_block.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .fc_similarity import compute_t_series, compute_FC_simi_block

__all__ = [
    "compute_t_series",
    "compute_FC_simi_block",
]
