"""test_gpu_correctness.py — CI-collectable correctness gate for
:func:`cifti_smoothing_gpu`.

Asserts the GPU cupyx fp32 SpMM matches the CPU numpy fp64 SpMV
reference on the real fsa6 gather (built by the production
``prepare_smoothing_gather`` at sigma=2.55) to max-abs-diff < 1e-3.
A silent regression in the gather-CSR build or the SpMM+scale pass
fires on the next test run instead of waiting for an end-to-end
pipeline run. The fp32 design choice is justified by this drift
bound — downstream find_minima + watershed compare values without
accumulation, so the GPU/CPU drift is well inside the value-comparison
noise floor.

Skipped automatically if cupy / a CUDA device is not available.

Run::

    python -m pytest arealmshbm/surface_smoothing/tests/ -v

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


_HAS_CUPY = importlib.util.find_spec("cupy") is not None

# Tests that read the CBIG midthickness atlas point at the env-configured
# CBIG checkout; if ``CBIG_CODE_DIR`` is unset, the atlas-dependent tests
# are skipped (the GPU vs. CPU equivalence tests don't depend on this
# atlas, they consume the precomputed step0_inputs cache instead).
_CBIG_ENV = os.environ.get("CBIG_CODE_DIR")
_CBIG = Path(_CBIG_ENV) if _CBIG_ENV else None
_ATLAS = (_CBIG / "utilities" / "matlab" / "speedup_gradients" /
          "utilities" / "fs6_surface_template") if _CBIG else None
_HAS_ATLAS = bool(_ATLAS and _ATLAS.exists())
_SIGMA = 2.55


# fp64-CPU vs fp32-GPU smoothing of Gaussian data: max-abs is bounded
# by the gather-weight propagation of fp32 round-off; 1e-3 is the
# loose-but-meaningful bar (downstream find_minima+watershed compare
# values, no accumulation, so 1e-3 is well inside the noise floor).
_MAX_ABS = 1e-3
_MAX_REL = 5e-3


@pytest.fixture(scope="module")
def _state():
    if not _HAS_CUPY:
        pytest.skip("cupy not installed")
    if not _HAS_ATLAS:
        pytest.skip(f"CBIG midthickness atlas not at {_ATLAS}")

    from arealmshbm.data_io import load_avg_mesh
    from arealmshbm.surface_io import read_surface_mesh
    from arealmshbm.surface_smoothing import (
        prepare_smoothing_mesh, prepare_smoothing_gather, cifti_smoothing,
    )
    from arealmshbm.surface_smoothing.surface_smoothing_gpu import (
        prepare_smoothing_gather_gpu, cifti_smoothing_gpu,
    )

    try:
        lh_avg = load_avg_mesh("lh", "fsaverage6", "sphere")
        rh_avg = load_avg_mesh("rh", "fsaverage6", "sphere")
    except FileNotFoundError as e:
        pytest.skip(f"fsaverage6 sphere bundle not staged: {e}")
    n_lh = lh_avg["vertices"].shape[1]
    medial_mask = np.concatenate([
        (lh_avg["MARS_label"].ravel() == 1),
        (rh_avg["MARS_label"].ravel() == 1),
    ]).astype(bool)
    cortex = ~medial_mask
    n_cortex = int(cortex.sum())

    lh_mid = read_surface_mesh(_ATLAS / "fsaverage6.L.midthickness.surf.gii")
    rh_mid = read_surface_mesh(_ATLAS / "fsaverage6.R.midthickness.surf.gii")
    lh_prep = prepare_smoothing_mesh(lh_mid["vertices"], lh_mid["faces"])
    rh_prep = prepare_smoothing_mesh(rh_mid["vertices"], rh_mid["faces"])
    lh_gather = prepare_smoothing_gather(lh_prep, cortex[:n_lh], _SIGMA)
    rh_gather = prepare_smoothing_gather(rh_prep, cortex[n_lh:], _SIGMA)
    lh_gather_gpu = prepare_smoothing_gather_gpu(lh_gather)
    rh_gather_gpu = prepare_smoothing_gather_gpu(rh_gather)

    return dict(
        cifti_smoothing=cifti_smoothing,
        cifti_smoothing_gpu=cifti_smoothing_gpu,
        lh_gather=lh_gather, rh_gather=rh_gather,
        lh_gather_gpu=lh_gather_gpu, rh_gather_gpu=rh_gather_gpu,
        medial_mask=medial_mask, n_cortex=n_cortex,
    )


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
@pytest.mark.skipif(not _HAS_ATLAS, reason=f"missing atlas {_ATLAS}")
@pytest.mark.parametrize("K", [108, 150])
def test_cpu_gpu_smoothing_agree(_state, K):
    """fp32 GPU smoothing must agree with fp64 CPU smoothing within
    tight relative tolerance on the real fsa6 gather. Asserts shape,
    dtype, finiteness, and per-element max-abs / max-rel drift.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((_state["n_cortex"], K)).astype(np.float32)

    out_cpu = _state["cifti_smoothing"](
        data, _state["medial_mask"],
        lh_gather=_state["lh_gather"], rh_gather=_state["rh_gather"],
    )
    import cupy as cp
    out_gpu_d = _state["cifti_smoothing_gpu"](
        data, _state["medial_mask"],
        lh_gather_gpu=_state["lh_gather_gpu"],
        rh_gather_gpu=_state["rh_gather_gpu"],
    )
    # GPU function returns a device cp.ndarray (PR #56-pattern device-stay
    # refactor — output feeds find_minima_gpu/watershed_gpu directly).
    out_gpu = cp.asnumpy(out_gpu_d)

    assert out_cpu.shape == out_gpu.shape
    assert out_cpu.dtype == out_gpu.dtype == np.float32
    assert np.isfinite(out_gpu).all()

    max_abs = float(np.max(np.abs(out_cpu - out_gpu)))
    denom = max(float(np.max(np.abs(out_cpu))), 1e-30)
    rel = max_abs / denom
    assert max_abs < _MAX_ABS, (
        f"K={K}: max_abs diff {max_abs:.3e} >= {_MAX_ABS:.0e} "
        f"(rel={rel:.3e})"
    )
    assert rel < _MAX_REL, f"K={K}: rel diff {rel:.3e} >= {_MAX_REL:.0e}"


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
@pytest.mark.skipif(not _HAS_ATLAS, reason=f"missing atlas {_ATLAS}")
def test_cifti_smoothing_gpu_device_input_alias(_state):
    """Device-input path: production caller hands a cp.ndarray straight
    from the upstream GPU accumulator. ``cp.asarray(data, dtype=cp.float32)``
    must alias (no copy) so the kernel sees identical input — verify
    by comparing output against the host-input path.

    Tolerance note: cuSPARSE SpMM is non-deterministic across calls
    (~1 ULP fp32 ≈ 1.2e-7 max abs diff on the fsa6 gather), so this is
    NOT a bit-equality check. atol=1e-5 sits 100× above that
    non-determinism floor and ~100× below the cross-precision drift
    bound, so a real device-input contract break (wrong path / wrong
    dtype / copy-on-wrong-device) would surface as a much larger
    divergence.
    """
    import cupy as cp
    rng = np.random.default_rng(0)
    data = rng.standard_normal((_state["n_cortex"], 108)).astype(np.float32)

    out_host_in_d = _state["cifti_smoothing_gpu"](
        data, _state["medial_mask"],
        lh_gather_gpu=_state["lh_gather_gpu"],
        rh_gather_gpu=_state["rh_gather_gpu"],
    )
    out_dev_in_d = _state["cifti_smoothing_gpu"](
        cp.asarray(data), _state["medial_mask"],
        lh_gather_gpu=_state["lh_gather_gpu"],
        rh_gather_gpu=_state["rh_gather_gpu"],
    )
    assert out_host_in_d.shape == out_dev_in_d.shape
    assert out_host_in_d.dtype == out_dev_in_d.dtype == cp.float32
    max_abs = float(cp.abs(out_host_in_d - out_dev_in_d).max())
    assert max_abs < 1e-5, (
        f"device-input vs host-input max_abs={max_abs:.3e} >= 1e-5 — "
        "the cp.asarray alias path likely broke (wrong dtype / copy / "
        "device drift)"
    )
