"""_sub001_fixture.py — real-data fixture for the ``gpu_sparse`` kernels.

Loads sub-001 (fsaverage6, T=6, L=300) from the local profile store and
builds the :class:`CandidateLayout`. Cached in-process so a test module
pays the ~1.5 s load once.

Skips (pytest) when the store — or the gitignored ``avg_mesh`` bundle
the load needs — is absent, so the suite stays green on machines
without the data and in a fresh worktree.

Usage::

    from arealmshbm.vmf_clustering.tests._sub001_fixture import load_sub001
    fx = load_sub001()          # None → pytest.skip in tests
    fx.layout                   # CandidateLayout (host numpy)
    fx.packed_TND               # (T, N, D_bytes) uint8, MW rows zeroed
    fx.D                        # unpacked D
    fx.theta, fx.boundary_mask  # dense (N, L)
    fx.lh_sphere / fx.rh_sphere # mesh dicts (vertices (3, n), vertexNbors (6, n))
    fx.lh_inflated / fx.rh_inflated
    fx.grad                     # (N, 100) fp32
    fx.Params                   # initialize_params output
    fx.setting_params           # dim, w, c, beta, epsilon, connect_th, ...

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

SUB001_DIR = Path(os.environ.get("MSHBM_SUB001_DIR",
                                 "testdata/modeA/sub-001"))


@dataclass
class Sub001Fixture:
    layout: Any
    packed_TND: np.ndarray
    D: int
    theta: np.ndarray
    boundary_mask: np.ndarray
    lh_sphere: Dict[str, np.ndarray]
    rh_sphere: Dict[str, np.ndarray]
    lh_inflated: Dict[str, np.ndarray]
    rh_inflated: Dict[str, np.ndarray]
    grad: np.ndarray
    Params: Dict[str, Any]
    setting_params: Dict[str, Any]
    ini_val: float


_CACHE: Optional[Sub001Fixture] = None


def missing_avg_mesh(*surfaces: str) -> Optional[str]:
    """Message naming the un-staged ``avg_mesh`` bundle, or ``None``.

    The bundles are gitignored assets, so a fresh worktree has none and
    ``load_avg_mesh`` raises ``FileNotFoundError``. These real-data tests
    must skip on that, not error.
    """
    from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
    try:
        for surface in surfaces:
            load_avg_mesh("lh", "fsaverage6", surface)
            load_avg_mesh("rh", "fsaverage6", surface)
    except FileNotFoundError as exc:
        return f"avg_mesh bundle not staged: {exc}"
    return None


def sub001_available() -> bool:
    return ((SUB001_DIR / "cohort.json").exists()
            and missing_avg_mesh("inflated", "sphere") is None)


def load_sub001() -> Optional[Sub001Fixture]:
    """Build (or return the cached) fixture; None when data is missing."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not sub001_available():
        return None

    from arealmshbm.step3_pipeline import Step3Config, Step3Pipeline
    from arealmshbm.vmf_clustering.sparse_layout import (
        build_candidate_layout_dense,
    )

    cfg = Step3Config(
        project_dir=SUB001_DIR, num_session=6, num_clusters=300, subid=1,
        mesh="fsaverage6", w=50.0, c=10.0, beta_scalar=5.0, backend="cpu",
    )
    pipe = Step3Pipeline(cfg)
    inp = pipe.load_inputs()
    layout = build_candidate_layout_dense(
        inp.Params["theta"], inp.boundary_mask,
        inp.lh_inflated["vertexNbors"], inp.rh_inflated["vertexNbors"],
    )
    # (N, T, Db) → (T, N, Db) on-disk layout used by the sparse backend.
    packed_TND = np.ascontiguousarray(
        np.transpose(inp.data["series"], (1, 0, 2)))
    _CACHE = Sub001Fixture(
        layout=layout,
        packed_TND=packed_TND,
        D=int(inp.data["D_unpacked"]),
        theta=inp.Params["theta"],
        boundary_mask=np.ascontiguousarray(inp.boundary_mask, dtype=np.float32),
        lh_sphere=inp.lh_sphere, rh_sphere=inp.rh_sphere,
        lh_inflated=inp.lh_inflated, rh_inflated=inp.rh_inflated,
        grad=np.ascontiguousarray(inp.data["gradient_mat"], dtype=np.float32),
        Params=inp.Params,
        setting_params=inp.setting_params,
        ini_val=float(inp.ini_val),
    )
    return _CACHE


def skip_unless_sub001():
    """pytest helper: returns the fixture or skips the calling test."""
    import pytest
    if not (SUB001_DIR / "cohort.json").exists():
        pytest.skip(f"sub-001 profile store not found at {SUB001_DIR}")
    missing = missing_avg_mesh("inflated", "sphere")
    if missing is not None:
        pytest.skip(missing)
    return load_sub001()


def skip_unless_cupy():
    import pytest
    try:
        import cupy  # noqa: F401
    except Exception:
        pytest.skip("cupy not available")
