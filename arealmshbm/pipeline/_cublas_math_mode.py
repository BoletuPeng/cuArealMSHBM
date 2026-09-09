"""_cublas_math_mode.py — scoped cuBLAS math-mode toggle.

``NVIDIA_TF32_OVERRIDE=1`` is a process-wide env var read at cuBLAS
library init; setting it after ``import cupy`` is too late, and once
set it affects every step in the unified-driver pipeline. When we
want TF32 active for one specific step but strict fp32 for the
others, we need the runtime ``cublasSetMathMode`` API instead.

Empirical (see ``docs/precision_impact_analysis.md``):
* DEFAULT_MATH on the current GPU → ~27 TFLOPS for 8192³ sgemm
* TENSOR_OP_MATH (= TF32 in CUDA 11+) → ~51 TFLOPS for the same shape
* setMathMode restore is clean — adjacent unrelated ops see the
  prior mode

The constant naming is legacy: CUDA 11+ collapses ``CUBLAS_TF32_TENSOR_OP_MATH``
and ``CUBLAS_TENSOR_OP_MATH`` to the same integer (1). CuPy still ships
the legacy name; we use it directly so the import doesn't break on
older cupy builds.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def cublas_tf32_scope() -> Iterator[None]:
    """Enable TF32 tensor-core math on the current device's cuBLAS handle
    inside the ``with`` block; restore the prior math mode on exit.

    All ``cupy.matmul`` / ``cupy.einsum`` / ``cupy.tensordot`` calls
    dispatching through cuBLAS use TF32 (10-bit mantissa truncation
    on the matmul operands; fp32 accumulators) for the scope. Other
    GPU ops (elementwise, reductions, transcendentals) are unaffected.

    Safe to use even if the scope contains no GPU work — the toggle
    happens once on entry and once on exit. cuBLAS math mode is
    per-handle and cupy keeps one handle per (device, thread), so the
    scope covers only the calling thread's cuBLAS work — a gemm issued
    from another thread is unaffected, and re-entrant toggling on the
    same thread would race.

    Raises if cupy isn't importable — caller's responsibility to gate
    on the GPU-backend check.
    """
    import cupy as cp
    from cupy.cuda import cublas

    handle = cp.cuda.device.get_cublas_handle()
    prior_mode = cublas.getMathMode(handle)
    # CUBLAS_TENSOR_OP_MATH = 1 (CUDA 11+ aliases CUBLAS_TF32_TENSOR_OP_MATH
    # to the same integer; we use the legacy name for cupy compatibility).
    cublas.setMathMode(handle, cublas.CUBLAS_TENSOR_OP_MATH)
    try:
        yield
    finally:
        cublas.setMathMode(handle, prior_mode)
