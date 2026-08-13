"""watershed_gpu.py

CuPy GPU port of :func:`watershed_algorithm_repaired`. Production
exposes only the fused boundary-count form (:func:`watershed_edge_count_gpu`)
since the step-0 pipeline never needs the full per-vertex label array
on host — only the (N,) ``sum(labels == 0, axis=1)``. See
``_kernels_gpu.py`` for the underlying RawKernels and the algorithmic
transformation (random-shuffle → Jacobi convergence) notes.

Precision / semantics contract vs CPU
-------------------------------------
* **Boundary mask (label == 0)** — empirically equivalent to the CPU
  ``watershed_algorithm_repaired`` to within the same algorithm-class
  drift documented in ``docs/step0_flow_and_subgraphs.md``. The
  downstream consumer is
  ``edge_density = sum(label == 0, axis=1) / num_sample_S``, which
  reads only the boundary mask, not the catchment id values.
* **Label IDs** — NOT bit-equal to CPU. CPU uses a per-column RNG
  shuffle seeded by ``k``; GPU uses atomic-counter rank order. Both
  assign distinct positive ints to distinct minima; the partition
  geometry (which vertex belongs to which catchment basin) is
  determined by mesh + edge_metrics, not by the id permutation.
* **Determinism** — GPU output is deterministic in (boundary mask,
  partition structure). Label int VALUES at non-boundary vertices
  may differ across runs (the atomic-counter assignment is
  GPU-arbitrary). Any downstream parity check should compare
  ``labels == 0`` boundary masks, not raw label values.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import Union

import numpy as np
import cupy as cp


def prepare_neighbors_for_gpu(neighbors: np.ndarray) -> cp.ndarray:
    """One-time conversion of the (N, R+1) fp64 1-indexed-with-NaN-pad
    neighbor table into a (N, R) int32 0-indexed-with-(-1)-pad device
    array suitable for repeated calls to
    :func:`watershed_edge_count_gpu`.

    The fsa6 step-0 pipeline calls watershed ``iter_a=3..4`` times per
    subject against the SAME ``ti.neighbors_table``; lifting this
    preprocessing out of the per-iter_a call hides ~3 ms/sub of host
    work (the np.isnan + astype(int32) chain).
    """
    # NaN sentinel is replaced with -1 in fp64 BEFORE the int32 cast —
    # float→int on NaN is implementation-defined at the C level (numpy
    # typically yields INT_MIN but it's not spec'd). Doing np.where in
    # float space, then casting, keeps every astype input well-defined.
    nb_1idx = np.ascontiguousarray(neighbors[:, 1:])
    nan_mask = np.isnan(nb_1idx)
    nb_int = np.where(nan_mask, -1, nb_1idx - 1).astype(np.int32)
    return cp.asarray(nb_int)


def watershed_edge_count_gpu(
        edge_metrics: Union[np.ndarray, cp.ndarray],
        minima: Union[np.ndarray, cp.ndarray],
        neighbors: np.ndarray,
        stepnum: int,
        fracmaxh: float,
        *,
        neighbors_int_d: cp.ndarray = None) -> np.ndarray:
    """Device-side fused watershed + edge-count.

    Runs the same threshold-sweep watershed kernel as the CPU
    :func:`watershed_algorithm_repaired`, but reduces straight to the
    per-vertex boundary count on device (``sum(labels == 0, axis=0)``
    over the K-axis after the kernel's internal (K, N) layout) — so
    only the (N,) int64 edge count crosses PCIe, never the full (N, K)
    int32 label array. At K=125 N=75k this saves ~3 ms per call (D2H is
    PCIe-5 bound at ~32 GB/s).

    ``edge_metrics`` and ``minima`` accept both host and device input
    (PR #56-pattern device-stay refactor). The production caller hands
    off the device cp.ndarray straight from
    ``cifti_smoothing_gpu`` / ``find_minima_gpu``; ``cp.asarray`` then
    aliases (no copy, no sync). Host inputs still work via a one-shot
    H2D, preserving the legacy contract for tests / ad-hoc callers.

    Parameters
    ----------
    edge_metrics : (N, K) fp32 host or device
    minima : (N, K) bool host or device
    neighbors : (N, R+1) fp64 numpy
        Used only if ``neighbors_int_d`` is None.
    stepnum : int
    fracmaxh : float
    neighbors_int_d : (N, R) int32 cupy.ndarray, optional
        Pre-converted neighbor table from
        :func:`prepare_neighbors_for_gpu`. If given, skips the per-call
        host preprocessing of ``neighbors``.

    Returns
    -------
    edge_count : (N,) int64 numpy
        ``sum(labels == 0, axis=1)`` — vertex-wise count of columns
        in which this vertex was a watershed boundary.
    """
    from ._kernels_gpu import run_watershed_gpu

    assert edge_metrics.ndim == 2
    assert edge_metrics.dtype == np.float32, (
        f"edge_metrics must be fp32; got {edge_metrics.dtype}")
    assert minima.shape == edge_metrics.shape
    # bool guard is load-bearing: ``.view(np.uint8)`` / ``.view(cp.uint8)``
    # below assumes 1-byte itemsize. An int32 / fp32 slip-in silently
    # reinterprets bytes and produces a wrong-shape buffer that crashes
    # opaquely inside the kernel call.
    assert minima.dtype == np.bool_, (
        f"minima must be bool; got {minima.dtype}")
    N, _K = int(edge_metrics.shape[0]), int(edge_metrics.shape[1])

    if neighbors_int_d is None:
        assert neighbors is not None and neighbors.shape[0] == N
        nb_d = prepare_neighbors_for_gpu(neighbors)
    else:
        assert neighbors_int_d.shape[0] == N
        nb_d = neighbors_int_d

    # H2D (if host) + transpose. cp.asarray with explicit dtype is the
    # alias-friendly path: a device fp32 input becomes a no-op alias;
    # a host fp32 input triggers one H2D. bool→uint8 reinterpret runs
    # on device for both host- and device-input paths.
    em_d = cp.asarray(edge_metrics, dtype=cp.float32).T.copy()
    # Reinterpret bool as uint8 (same itemsize, alias). ``np.uint8`` is
    # the same dtype object as ``cp.uint8`` (cupy reuses numpy dtypes),
    # so a single ``.view(np.uint8)`` call works for both numpy and
    # cupy inputs — host views H2D via the cp.asarray below; device
    # views stay device-resident.
    min_d = cp.asarray(minima.view(np.uint8)).T.copy()

    # min/max compute on device; ``float()`` materialises each scalar
    # to host so np.arange below can run on host fp64. Two implicit
    # syncs per call (one per ``.min/.max`` reduction's float()
    # block) — net still wins over the old host-side min/max because
    # we no longer pay the (N, K) H2D for edge_metrics ahead of those
    # reductions.
    em_min64 = float(em_d.min())
    em_max64 = float(em_d.max())
    stoph = em_max64 * float(fracmaxh)
    step = (em_max64 - em_min64) / float(stepnum)
    hiter_np = np.arange(em_min64, stoph + step / 2.0, step).astype(np.float32)
    hiter_d = cp.asarray(hiter_np)

    label_kn_d = run_watershed_gpu(em_d, min_d, nb_d, hiter_d)

    # Reduce on device: (K, N) → (N,) int64 via sum(label==0) over axis 0.
    edge_count_d = (label_kn_d == 0).sum(axis=0).astype(cp.int64)
    return cp.asnumpy(edge_count_d)


