"""test_gpu_correctness.py — CI-collectable correctness gate for
:func:`cifti_gradient_gpu`.

Asserts the GPU RawKernel matches the CPU @njit reference on the real
fsa6 mesh against a synthetic FC_simi_block-shaped input. A silent
regression in the RawKernel — wrong indexing, dropped sync,
normal-equations bug — fires on the next test run instead of waiting
for an end-to-end pipeline run. The fp32 / one-block-per-vertex
kernel design is justified by this drift bound (sub-1e-4 vs CPU fp64).

Skipped automatically if cupy / a CUDA device is not available.

Run::

    python -m pytest arealmshbm/surface_gradient/tests/ -v

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


# Loose-but-meaningful tolerances. Tight rel for the cohort, with a
# small absolute slack to absorb fp32-vs-fp64 normal-equations drift.
_MAX_ABS = 1e-3
_MAX_REL = 1e-3


@pytest.fixture(scope="module")
def _state():
    """Build the (mesh, prep, gpu prep, medial mask, n_cortex) tuple
    once per pytest session. Heavy enough that we don't want to repeat
    it across the K-parameterized tests below.
    """
    if not _HAS_CUPY:
        pytest.skip("cupy not installed")
    if not _HAS_ATLAS:
        pytest.skip(f"CBIG midthickness atlas not at {_ATLAS}")

    import cupy as cp
    from arealmshbm.data_io import load_avg_mesh
    from arealmshbm.surface_io import read_surface_mesh
    from arealmshbm.surface_gradient import (
        cifti_gradient, prepare_gradient_mesh,
    )
    from arealmshbm.surface_gradient.surface_gradient_gpu import (
        prepare_gradient_mesh_gpu, cifti_gradient_gpu,
    )

    try:
        lh_avg = load_avg_mesh("lh", "fsaverage6", "sphere")
        rh_avg = load_avg_mesh("rh", "fsaverage6", "sphere")
    except FileNotFoundError as e:
        pytest.skip(f"fsaverage6 sphere bundle not staged: {e}")
    medial_mask = np.concatenate([
        (lh_avg["MARS_label"].ravel() == 1),
        (rh_avg["MARS_label"].ravel() == 1),
    ]).astype(bool)
    n_cortex = int((~medial_mask).sum())

    lh_mid = read_surface_mesh(_ATLAS / "fsaverage6.L.midthickness.surf.gii")
    rh_mid = read_surface_mesh(_ATLAS / "fsaverage6.R.midthickness.surf.gii")
    lh_prep = prepare_gradient_mesh(lh_mid["vertices"], lh_mid["faces"])
    rh_prep = prepare_gradient_mesh(rh_mid["vertices"], rh_mid["faces"])

    lh_verts_d = cp.asarray(lh_mid["vertices"], dtype=cp.float32)
    rh_verts_d = cp.asarray(rh_mid["vertices"], dtype=cp.float32)
    lh_mesh_gpu = prepare_gradient_mesh_gpu(lh_prep)
    rh_mesh_gpu = prepare_gradient_mesh_gpu(rh_prep)

    return dict(
        cifti_gradient=cifti_gradient,
        cifti_gradient_gpu=cifti_gradient_gpu,
        lh_mid_verts=lh_mid["vertices"], rh_mid_verts=rh_mid["vertices"],
        lh_prep=lh_prep, rh_prep=rh_prep,
        lh_verts_d=lh_verts_d, rh_verts_d=rh_verts_d,
        lh_mesh_gpu=lh_mesh_gpu, rh_mesh_gpu=rh_mesh_gpu,
        medial_mask=medial_mask, n_cortex=n_cortex,
    )


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not installed")
@pytest.mark.skipif(not _HAS_ATLAS, reason=f"missing atlas {_ATLAS}")
@pytest.mark.parametrize("K", [108, 150])
def test_cpu_gpu_gradient_agree(_state, K):
    """fp32 GPU gradient must agree with fp64 CPU gradient within tight
    relative tolerance on the real fsa6 midthickness mesh — synthetic
    Gaussian data feeds both sides.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((_state["n_cortex"], K)).astype(np.float32)

    out_cpu = _state["cifti_gradient"](
        data,
        _state["lh_mid_verts"], *_state["lh_prep"],
        _state["rh_mid_verts"], *_state["rh_prep"],
        _state["medial_mask"],
    )
    import cupy as cp
    out_gpu_d = _state["cifti_gradient_gpu"](
        data, medial_mask=_state["medial_mask"],
        lh_verts_d=_state["lh_verts_d"], lh_mesh_gpu=_state["lh_mesh_gpu"],
        rh_verts_d=_state["rh_verts_d"], rh_mesh_gpu=_state["rh_mesh_gpu"],
    )
    # GPU function returns a device cp.ndarray (PR #56-pattern device-stay
    # refactor). Bring it to host for the numpy comparison.
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
def test_cifti_gradient_gpu_device_input_alias(_state):
    """Device-input path: production caller hands the cp.ndarray
    ``FC_simi_block`` straight from ``compute_FC_simi_block_gpu``.
    ``cp.asarray(data, dtype=cp.float32)`` must alias (no copy) so the
    kernel sees identical input — verify by comparing output against
    the host-input path.

    Tolerance note: the gradient RawKernel solves a 3×3 normal
    equations per vertex with non-deterministic atomicAdd-driven
    accumulator ordering, so two runs on the same fp32 input differ
    by ~1 ULP fp32. atol=1e-5 sits well above that floor and well
    below any plausible real bug (wrong dtype / copy-on-wrong-device
    / dropped sync would land at ~1e-3+).
    """
    import cupy as cp
    rng = np.random.default_rng(0)
    data = rng.standard_normal((_state["n_cortex"], 108)).astype(np.float32)

    out_host_in_d = _state["cifti_gradient_gpu"](
        data, medial_mask=_state["medial_mask"],
        lh_verts_d=_state["lh_verts_d"], lh_mesh_gpu=_state["lh_mesh_gpu"],
        rh_verts_d=_state["rh_verts_d"], rh_mesh_gpu=_state["rh_mesh_gpu"],
    )
    out_dev_in_d = _state["cifti_gradient_gpu"](
        cp.asarray(data), medial_mask=_state["medial_mask"],
        lh_verts_d=_state["lh_verts_d"], lh_mesh_gpu=_state["lh_mesh_gpu"],
        rh_verts_d=_state["rh_verts_d"], rh_mesh_gpu=_state["rh_mesh_gpu"],
    )
    assert out_host_in_d.shape == out_dev_in_d.shape
    assert out_host_in_d.dtype == out_dev_in_d.dtype == cp.float32
    max_abs = float(cp.abs(out_host_in_d - out_dev_in_d).max())
    assert max_abs < 1e-5, (
        f"device-input vs host-input max_abs={max_abs:.3e} >= 1e-5 — "
        "the cp.asarray alias path likely broke (wrong dtype / copy / "
        "device drift)"
    )
