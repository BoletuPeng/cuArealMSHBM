"""test_config_backend.py

``Step2Config`` backend enum + the ``gpu`` backend's static limits.

The backend value is load-bearing plumbing: ``'gpu'`` selects the
P-layout path (a different ``load_inputs`` / ``initialize_params`` /
``run_em`` triple), ``'cpu'`` the numba reference. These tests pin the
enum membership, the rejection message and the catalog's own copy of the
enum, so a future rename cannot silently demote a GPU run to CPU, and
the three static kernel limits of the GPU backend (seed mesh,
``num_clusters``, ``n_grad_components``) are refused at config time
rather than after step 0/1. ``'gpu_sparse'``, the name the P-layout
backend shipped under while the dense CuPy port still held ``'gpu'``,
is rejected like any other unknown spelling — no alias.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arealmshbm.step2_pipeline.config import _VALID_BACKENDS, Step2Config


def _cfg(**kw) -> Step2Config:
    base = dict(project_dir="proj", num_sub=1, num_session=6,
                num_clusters=300, mode="gMSHBM", verbose=False)
    base.update(kw)
    return Step2Config(**base)


# ── backend enum ──
def test_backend_enum_is_exactly_the_two_values() -> None:
    assert _VALID_BACKENDS == {"cpu", "gpu"}


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_backend_accepted(backend: str) -> None:
    assert _cfg(backend=backend).backend == backend


def test_backend_default_is_cpu() -> None:
    assert _cfg().backend == "cpu"


@pytest.mark.parametrize("backend", ["gpu_dense", "gpu_sparse"])
def test_unknown_backend_rejected(backend: str) -> None:
    """``'gpu_sparse'`` was the P-layout backend's name while the dense
    CuPy port held ``'gpu'``; it is not an alias now."""
    with pytest.raises(ValueError, match="backend must be one of"):
        _cfg(backend=backend)


def test_no_alias_for_a_wrong_spelling() -> None:
    """No aliases: a config written against a name that does not exist
    must fail loudly rather than fall back to a working backend."""
    with pytest.raises(ValueError):
        _cfg(backend="cuda")


def test_gpu_requires_the_fsaverage3_seed_mesh() -> None:
    """The GPU kernels need ceil(D/8) <= 256, i.e. fsaverage3."""
    with pytest.raises(ValueError, match="seed_mesh='fsaverage3'"):
        _cfg(backend="gpu", seed_mesh="fsaverage4")
    assert _cfg(backend="gpu").seed_mesh == "fsaverage3"
    # The guard is scoped to the GPU backend only.
    assert _cfg(backend="cpu", seed_mesh="fsaverage4").backend == "cpu"


_SPARSE_MAX_CLUSTERS = 512
_SPARSE_MAX_D_GRAD = 11772


def test_the_config_sparse_limits_match_the_kernel_constants() -> None:
    """``Step2Config`` mirrors them as literals so it stays cupy-free."""
    pytest.importorskip("cupy")
    from arealmshbm.step2_em_iter_master._kernels_gpu import (
        MAX_CLUSTERS, MAX_D_GRAD,
    )
    assert MAX_CLUSTERS == _SPARSE_MAX_CLUSTERS
    assert MAX_D_GRAD == _SPARSE_MAX_D_GRAD


def test_gpu_rejects_more_clusters_than_the_kernels_take() -> None:
    """``init_hard_labels``'s per-lane accumulators cap L (``check_dims``).

    Checked at config time, not at Session-ctor time: the ctor only runs
    after step 0/1 and the whole cohort's packed-BOLD decode.
    """
    with pytest.raises(ValueError, match="num_clusters <="):
        _cfg(backend="gpu", num_clusters=_SPARSE_MAX_CLUSTERS + 1)
    assert _cfg(backend="gpu",
                num_clusters=_SPARSE_MAX_CLUSTERS
                ).num_clusters == _SPARSE_MAX_CLUSTERS
    # Scoped to the GPU backend only.
    assert _cfg(backend="cpu",
                num_clusters=_SPARSE_MAX_CLUSTERS + 1).backend == "cpu"


def test_gpu_rejects_more_grad_components_than_connect_u_takes() -> None:
    """``connect_u`` stages one float per component in shared memory."""
    with pytest.raises(ValueError, match="n_grad_components <="):
        _cfg(backend="gpu", n_grad_components=_SPARSE_MAX_D_GRAD + 1)
    assert _cfg(backend="gpu",
                n_grad_components=_SPARSE_MAX_D_GRAD
                ).n_grad_components == _SPARSE_MAX_D_GRAD
    # dMSHBM never launches ``connect_u``; the CPU backend is untouched.
    assert _cfg(backend="gpu", mode="dMSHBM",
                n_grad_components=_SPARSE_MAX_D_GRAD + 1).mode == "dMSHBM"
    assert _cfg(backend="cpu",
                n_grad_components=_SPARSE_MAX_D_GRAD + 1).backend == "cpu"


# ── the catalog still describes the driver-facing enum ──
def test_catalog_backend_step2_values_unchanged() -> None:
    """``PipelineConfig.backend_step2`` offers the same two values as
    step0/step1 (``_VALID_BACKENDS_STEP012`` is shared by the three)."""
    repo = Path(__file__).resolve().parents[3]
    catalog = repo / "lib" / "hyperparameters" / "step2.json"
    with open(catalog, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = next(e for e in data["entries"] if e["key"] == "backend_step2")
    assert entry["values"] == ["cpu", "gpu"]
    assert entry["default"] == "cpu"

    from arealmshbm.pipeline.config import _VALID_BACKENDS_STEP012
    assert set(_VALID_BACKENDS_STEP012) == {"cpu", "gpu"}
