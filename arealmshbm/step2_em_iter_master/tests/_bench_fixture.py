"""_bench_fixture.py — shared fixture for the step-2 sparse GPU tests.

Loads the on-disk bench projects (``testdata/step2_bench/proj`` for S=1,
``proj2`` for S=2; override the root with ``MSHBM_STEP2_BENCH_DIR``) once
per process and exposes both

* the :class:`~arealmshbm.step2_io.sparse_inputs.Step2SparseInputs` the
  production loader (``load_step2_sparse_inputs``) reads from the
  project, plus the bit-packed bytes and gradients its readers deliver,
  kept as the kernel tests' reference copies, and
* the matching **CPU-side reference objects** — the real
  ``Step2Pipeline.load_inputs()`` / ``initialize_params()`` output, so each
  kernel test can compare against the numba kernels on identical inputs.

Everything is module-cached: the disk read plus the CPU ``compose_init_state``
costs ~1.5 s and is paid once.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from arealmshbm.vmf_clustering.tests._sub001_fixture import missing_avg_mesh

BENCH_ROOT = Path(os.environ.get("MSHBM_STEP2_BENCH_DIR", r"testdata/step2_bench"))

_CACHE: Dict[str, Any] = {}


class BenchFixture:
    """Everything the kernel tests need for one bench project."""

    __slots__ = ("S", "T", "N", "D", "Db", "L", "n_lh", "L_lh",
                 "D_grad", "dim", "layout", "inputs", "cpu_inputs", "Params",
                 "packed", "grad", "boundary_mask", "ini_val", "cfg")

    def __init__(self, project_dir: Path, S: int, mode: str = "gMSHBM"):
        from arealmshbm.step2_io import load_step2_sparse_inputs
        from arealmshbm.step2_pipeline import Step2Config, Step2Pipeline

        cfg = Step2Config(
            project_dir=project_dir,
            num_sub=S,
            num_session=6,
            num_clusters=300,
            mode=mode,
            beta_scalar=5,
            backend="cpu",
            out_dir=project_dir / "_fixture_out",
            verbose=False,
        )
        self.cfg = cfg
        pipe = Step2Pipeline(cfg)
        cpu_inputs = pipe.load_inputs()
        Params = pipe.initialize_params(cpu_inputs)

        self.cpu_inputs = cpu_inputs
        self.Params = Params
        self.ini_val = float(Params["ini_val"])
        self.boundary_mask = cpu_inputs.boundary_mask
        self.dim = int(cpu_inputs.dim)

        self.S = S
        self.T = int(cpu_inputs.T)
        self.N = int(cpu_inputs.N)
        self.D = int(cpu_inputs.D)
        self.Db = (self.D + 7) // 8
        self.L = int(cfg.num_clusters)
        self.n_lh = int(cpu_inputs.n_lh)
        self.L_lh = self.L // 2
        self.D_grad = int(cpu_inputs.D_grad)

        # The Session's inputs, and what their readers deliver: a kernel
        # test's Session and its CPU reference start from the same bytes.
        self.inputs = load_step2_sparse_inputs(cfg)
        self.layout = self.inputs.layout
        self.packed = np.empty((S, self.T, self.N, self.Db), dtype=np.uint8)
        for s in range(S):
            self.inputs.bold_reader(s + 1, self.packed[s])
        self.grad = None
        if self.inputs.grad_reader is not None:
            self.grad = np.empty((S, self.N, self.D_grad), dtype=np.float32)
            for s in range(S):
                self.inputs.grad_reader(s + 1, self.grad[s])

    # ── convenience ──
    def bold_f32(self, s: int) -> np.ndarray:
        """The exact CPU ``(N, T, D)`` fp32 BOLD slab for subject ``s`` (0-based)."""
        from arealmshbm.step2_io.load_subject_profiles import (
            _widen_normalize_bitpacked_to_f32_NTD_kernel,
        )
        out = np.zeros((self.N, self.T, self.D), dtype=np.float32)
        _widen_normalize_bitpacked_to_f32_NTD_kernel(
            np.ascontiguousarray(self.packed[s].transpose(1, 0, 2)), out, self.D)
        return out


def bench_available(name: str = "proj") -> bool:
    return ((BENCH_ROOT / name / "cohort.json").exists()
            and missing_avg_mesh("inflated") is None)


def get_fixture(name: str = "proj", S: int = 1, mode: str = "gMSHBM") -> BenchFixture:
    key = f"{name}:{S}:{mode}"
    fx = _CACHE.get(key)
    if fx is None:
        fx = BenchFixture(BENCH_ROOT / name, S, mode=mode)
        _CACHE[key] = fx
    return fx


def skip_unless_bench(name: str = "proj") -> None:
    if not (BENCH_ROOT / name / "cohort.json").exists():
        pytest.skip(f"bench project {BENCH_ROOT / name} not present "
                    f"(set MSHBM_STEP2_BENCH_DIR)")
    missing = missing_avg_mesh("inflated")
    if missing is not None:
        pytest.skip(missing)


__all__ = ["BENCH_ROOT", "BenchFixture", "bench_available", "get_fixture",
           "missing_avg_mesh", "skip_unless_bench"]
