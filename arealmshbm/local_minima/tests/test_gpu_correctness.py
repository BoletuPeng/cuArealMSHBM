"""test_gpu_correctness.py — bit-equivalence gate for
:func:`find_minima_gpu` vs the CPU :func:`find_minima` reference.

The K-NN local-minimum check is a pure comparison (no accumulation),
so the GPU port must produce **bit-identical** output to the CPU
reference. Skipped automatically if cupy / a CUDA device is not
available.

Run::

    python -m pytest arealmshbm/local_minima/tests/ -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from arealmshbm.local_minima import find_minima


_HAS_CUPY = importlib.util.find_spec("cupy") is not None


def _synth_K_neighbors(N: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """Build a (N, M) fp64 neighbor table with 1-indexed entries, NaN
    padding in random tail positions. Col 0 = self (i+1). Cols 1..M-1
    are distinct random cortex IDs (1-indexed); some tail positions
    NaN-padded with probability 0.3 per slot to exercise the padding
    handler.
    """
    out = np.empty((N, M), dtype=np.float64)
    out[:, 0] = np.arange(1, N + 1, dtype=np.float64)
    for i in range(N):
        cand = rng.permutation(N - 1) + 1
        cand = cand[cand != (i + 1)][: M - 1]
        for m, c in enumerate(cand, start=1):
            out[i, m] = float(c)
        # pad shortfall + random NaN tail
        n_real = len(cand)
        if n_real < M - 1:
            out[i, 1 + n_real:] = np.nan
        # randomly NaN-pad some real entries from the tail
        n_nan = int(rng.binomial(n_real, 0.3))
        if n_nan > 0:
            out[i, M - n_nan:] = np.nan
    return out


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
@pytest.mark.parametrize("N,K,M,seed", [
    (200, 16, 7, 0),
    (1024, 100, 32, 1),
    (4096, 8, 64, 2),
])
def test_bit_exact_vs_cpu(N, K, M, seed):
    """GPU output is bit-identical to CPU on synthetic Gaussian data
    with a random K-NN table (NaN padding included)."""
    import cupy as cp
    from arealmshbm.local_minima import (
        find_minima_gpu, prepare_K_neighbors_for_gpu,
    )

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N, K)).astype(np.float32)
    K_neighbors = _synth_K_neighbors(N, M, rng)

    out_cpu = find_minima(data, K_neighbors)

    nb_d = prepare_K_neighbors_for_gpu(K_neighbors)
    out_gpu_d = find_minima_gpu(data, nb_d)
    out_gpu = cp.asnumpy(out_gpu_d)

    assert out_cpu.shape == out_gpu.shape == (N, K)
    assert out_cpu.dtype == np.bool_
    assert out_gpu.dtype == np.bool_
    # Strict equality — pure comparison, no rounding.
    assert np.array_equal(out_cpu, out_gpu), (
        f"N={N} K={K} M={M}: GPU and CPU minima disagree at "
        f"{int((out_cpu != out_gpu).sum())} cells"
    )


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
def test_device_input_alias():
    """Production caller hands a device ``cp.ndarray`` straight from
    ``cifti_smoothing_gpu`` — verify the cp.asarray inside is a no-op
    alias (no copy) and the result matches the host-input path.
    """
    import cupy as cp
    from arealmshbm.local_minima import (
        find_minima_gpu, prepare_K_neighbors_for_gpu,
    )

    rng = np.random.default_rng(7)
    N, K, M = 512, 32, 16
    data = rng.standard_normal((N, K)).astype(np.float32)
    K_neighbors = _synth_K_neighbors(N, M, rng)
    nb_d = prepare_K_neighbors_for_gpu(K_neighbors)

    out_host_input_d = find_minima_gpu(data, nb_d)
    out_dev_input_d = find_minima_gpu(cp.asarray(data), nb_d)
    assert cp.array_equal(out_host_input_d, out_dev_input_d)
