"""test_nan_clamp.py — regression test for the stuck-vertex NaN clamp
on ``compute_FC_simi_block`` (CPU path) and ``compute_FC_simi_block_gpu``
(GPU path).

Failure mode this guards against: a BOLD vertex with zero variance
(a "stuck" channel — flat signal across all T timepoints) produces
``mag_t==0`` after the row-demean / L2-norm step. The subsequent
``FC / mag_t`` divides yield ``0/0 = NaN``. Without the boundary
clamp inside ``compute_FC_simi_block``, our self-implemented
``cifti_gradient`` is NaN-passthrough, so a single stuck seed
contaminates the entire ``(N_cortex, K)`` block and the downstream
``avg_grads → edge_density → diffmap`` chain collapses to zero —
which is exactly the sub-005 regression that motivated the fix.

The test plants a stuck vertex inside an otherwise-finite synthetic
BOLD volume and asserts the output is finite (no NaN, no inf).

Run::

    python -m pytest arealmshbm/fc_similarity/tests/ -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from arealmshbm.fc_similarity import (
    compute_t_series,
    compute_FC_simi_block,
)


_HAS_CUPY = importlib.util.find_spec("cupy") is not None


def _synthetic_bold(n_cortex: int, T: int, rng: np.random.Generator,
                    stuck_idx: int) -> np.ndarray:
    """Build a deterministic ``(N_cortex, T)`` fp32 BOLD with one
    stuck vertex (constant signal → var=0) and otherwise non-trivial
    structure. Stuck vertex value chosen non-zero so the failure mode
    is "stuck", not "all-zero" (those are distinct).
    """
    data = rng.standard_normal((n_cortex, T)).astype(np.float32)
    data[stuck_idx, :] = np.float32(0.42)
    return data


def _run_one_block_cpu(curr_data, randinds_FC, randinds_verts,
                       num_blocks_a=2, num_blocks_b=2,
                       block_a_index=0):
    t_series, mag_t = compute_t_series(curr_data, randinds_FC)
    return compute_FC_simi_block(
        curr_data=curr_data,
        t_series=t_series, mag_t=mag_t,
        randinds_verts=randinds_verts,
        block_a_index=block_a_index,
        num_blocks_a=num_blocks_a, num_blocks_b=num_blocks_b,
    )


def test_stuck_vertex_outside_subsampled_seeds_is_finite():
    """Stuck vertex contaminates ``compute_t_series`` (mag_t==0 at that
    row) but is not itself picked as a subsampled seed. The clamp must
    still drop the NaN propagated into the broadcast division.
    """
    rng = np.random.default_rng(0)
    n_cortex, T = 200, 24
    stuck = 17
    curr_data = _synthetic_bold(n_cortex, T, rng, stuck_idx=stuck)

    randinds_FC = np.arange(n_cortex, dtype=np.int64)
    randinds_verts = np.arange(50, 90, dtype=np.int64)
    assert stuck not in randinds_verts

    out = _run_one_block_cpu(curr_data, randinds_FC, randinds_verts)
    assert out.dtype == np.float32
    assert np.isfinite(out).all(), (
        f"NaN/inf leaked through compute_FC_simi_block: "
        f"NaN={int(np.isnan(out).sum())} inf={int(np.isinf(out).sum())}"
    )


def test_stuck_vertex_inside_subsampled_seeds_is_finite():
    """Now the stuck vertex IS one of the subsampled seeds (worst case:
    s_series_A has a zero column, so the division by ``mag_s_A`` is also
    0/0). The clamp must still emit a finite block.
    """
    rng = np.random.default_rng(1)
    n_cortex, T = 200, 24
    stuck = 60
    curr_data = _synthetic_bold(n_cortex, T, rng, stuck_idx=stuck)

    randinds_FC = np.arange(n_cortex, dtype=np.int64)
    randinds_verts = np.arange(50, 90, dtype=np.int64)
    assert stuck in randinds_verts

    out = _run_one_block_cpu(curr_data, randinds_FC, randinds_verts)
    assert np.isfinite(out).all()


def test_no_stuck_vertex_is_finite_too():
    """Sanity: control case with no stuck vertex. Confirms the test
    fixture itself isn't generating spurious NaN.
    """
    rng = np.random.default_rng(2)
    n_cortex, T = 200, 24
    curr_data = rng.standard_normal((n_cortex, T)).astype(np.float32)

    randinds_FC = np.arange(n_cortex, dtype=np.int64)
    randinds_verts = np.arange(50, 90, dtype=np.int64)

    out = _run_one_block_cpu(curr_data, randinds_FC, randinds_verts)
    assert np.isfinite(out).all()


@pytest.mark.skipif(not _HAS_CUPY,
                    reason="cupy not installed — GPU path not testable")
def test_stuck_vertex_gpu_is_finite():
    """GPU mirror: stuck vertex inside subsampled seeds. The GPU
    function takes device arrays and returns a device array; we
    cp.asnumpy at the boundary to keep the assertion purely numpy.
    """
    import cupy as cp
    from arealmshbm.fc_similarity.fc_similarity_gpu import (
        compute_FC_simi_block_gpu,
    )

    rng = np.random.default_rng(3)
    n_cortex, T = 200, 24
    stuck = 73
    curr_data = _synthetic_bold(n_cortex, T, rng, stuck_idx=stuck)

    randinds_FC = np.arange(n_cortex, dtype=np.int64)
    randinds_verts = np.arange(50, 90, dtype=np.int64)
    assert stuck in randinds_verts

    t_series, mag_t = compute_t_series(curr_data, randinds_FC)

    curr_data_d = cp.asarray(curr_data)
    t_series_d = cp.asarray(t_series)
    mag_t_d = cp.asarray(mag_t)

    out_d = compute_FC_simi_block_gpu(
        curr_data_d=curr_data_d,
        t_series_d=t_series_d, mag_t_d=mag_t_d,
        randinds_verts=randinds_verts,
        block_a_index=0,
        num_blocks_a=2, num_blocks_b=2,
    )
    out = cp.asnumpy(out_d)
    assert np.isfinite(out).all()
