"""Kernel-level gates for :mod:`arealmshbm.fc_similarity._kernels_fused_gpu`:
the fused divide+demean+norm against the equivalent cupy chain and the
shared-memory budget the divide kernel has to refuse. Skipped
when cupy is absent.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


_HAS_CUPY = importlib.util.find_spec("cupy") is not None


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
@pytest.mark.parametrize("T,K", [(7, 5), (241, 313), (749, 313)])
def test_div_demean_norm_matches_the_cupy_chain(T, K):
    """Fused kernel reproduces the chain to fp32 ULP; ``T=749`` runs the
    cooperative ``mr`` staging loop for several iterations, as production does.
    """
    import cupy as cp
    from arealmshbm.fc_similarity._kernels_fused_gpu import (
        fused_div_demean_norm_columns_cupy)

    rng = np.random.default_rng(T * 1000 + K)
    x = rng.standard_normal((T, K), dtype=np.float32) * 17.0
    mr = (rng.random(T, dtype=np.float32) * 3.0 + 0.25)
    mc = (rng.random(K, dtype=np.float32) * 3.0 + 0.25)
    mr_d, mc_d = cp.asarray(mr), cp.asarray(mc)

    got = cp.asarray(x)
    mag_got = fused_div_demean_norm_columns_cupy(got, mr_d, mc_d)

    ref = cp.asarray(x) / (mr_d[:, None] * mc_d[None, :])
    mean64 = ref.astype(cp.float64).mean(axis=0)
    ref -= mean64.astype(cp.float32)[None, :]
    mag_ref = cp.sqrt((ref.astype(cp.float64) ** 2).sum(axis=0)).astype(
        cp.float32)

    def _max_rel(a, b):
        a64, b64 = a.astype(cp.float64), b.astype(cp.float64)
        scale = max(float(cp.abs(b64).max()), 1e-30)
        return float(cp.abs(a64 - b64).max()) / scale

    assert _max_rel(got, ref) <= 1e-6, "demeaned buffer diverges"
    assert _max_rel(mag_got, mag_ref) <= 1e-6, "column norms diverge"


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
def test_div_kernel_guards_its_shared_memory_budget():
    """``T`` at the cap launches; one row over raises a named ValueError."""
    import cupy as cp
    from arealmshbm.fc_similarity._kernels_fused_gpu import (
        fused_div_demean_norm_columns_cupy, _AXIS0_DIV1W_MAX_T)

    K = 64
    mc = cp.ones(K, dtype=cp.float32)
    ok = cp.asarray(np.random.default_rng(0).random(
        (_AXIS0_DIV1W_MAX_T, K), dtype=np.float32))
    fused_div_demean_norm_columns_cupy(
        ok, cp.ones(_AXIS0_DIV1W_MAX_T, dtype=cp.float32), mc)
    cp.cuda.get_current_stream().synchronize()

    over = cp.zeros((_AXIS0_DIV1W_MAX_T + 1, K), dtype=cp.float32)
    with pytest.raises(ValueError, match="dynamic shared memory"):
        fused_div_demean_norm_columns_cupy(
            over, cp.ones(_AXIS0_DIV1W_MAX_T + 1, dtype=cp.float32), mc)


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
def test_in_place_entry_points_reject_non_contiguous_input():
    """The demean kernels write through a flat row-major offset, so a
    non-C-contiguous buffer cannot be mutated in place. Silently copying
    it would return correct magnitudes and leave the caller's array
    un-demeaned, so it must raise instead."""
    import cupy as cp
    from arealmshbm.fc_similarity._kernels_fused_gpu import (
        fused_demean_norm_columns_cupy,
        fused_demean_norm_rows_cupy,
        fused_div_demean_norm_columns_cupy,
    )

    base = cp.asarray(np.random.default_rng(7).standard_normal(
        (16, 24), dtype=np.float32))
    view = base[:, ::2]                       # strided -> not C-contiguous
    assert not view.flags.c_contiguous

    for fn in (fused_demean_norm_columns_cupy, fused_demean_norm_rows_cupy):
        with pytest.raises(ValueError, match="C-contiguous"):
            fn(view)
    with pytest.raises(ValueError, match="C-contiguous"):
        fused_div_demean_norm_columns_cupy(
            view,
            cp.ones(view.shape[0], dtype=cp.float32),
            cp.ones(view.shape[1], dtype=cp.float32))
