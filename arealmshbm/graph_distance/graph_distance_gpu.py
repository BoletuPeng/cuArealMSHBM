"""graph_distance_gpu.py

GPU implementation of :func:`gradient_geodesic_distance` — same
signature, precision contract and output shape as the CPU function.

The solver is the per-source Δ-stepping SSSP in
:mod:`._kernels_gpu`. It replaced a batched pull-based
Bellman-Ford in 2026-09: both reach the same fixed point of
``d[u] = min_v fl(d[v] + w(v, u))`` and were bit-identical, so the
older one was dropped rather than kept as a knob — hence the two rows
in the timing table of ``docs/step0_flow_and_subgraphs.md``.

Every entry point returns ``D[dest, source]``; a per-source SSSP fills
``D[source, dest]`` and fp32 addition is not reassociative, so the
transpose pass is not optional.

The preconditions are hard — there is nothing to fall back to, so each
one raises ``ValueError``: valence ``M <= EDGE_SLOTS``, ``N <= MAX_N``
(the kernel's static shared-memory budget), neighbour slots inside the
1-indexed range ``[0, N]``, a symmetric neighbour table, and
non-negative non-NaN weights.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _shape_check(verts, vertex_nbors, grad_data) -> int:
    """Shared input validation; returns N. Lifted from the CPU wrapper
    so both backends fail with the same error messages."""
    if vertex_nbors.ndim != 2:
        raise ValueError(
            f"vertex_nbors must be 2D; got {vertex_nbors.shape}")
    N = vertex_nbors.shape[0]
    if grad_data.shape != (N,):
        raise ValueError(
            f"grad_data shape {grad_data.shape} must equal (N,) = ({N},)")
    if verts.ndim != 2 or verts.shape[0] != N or verts.shape[1] != 3:
        raise ValueError(
            f"verts must be (N, 3) = ({N}, 3); got {verts.shape}")
    return N


def _delta_for(grad_data_d, delta_mult: float) -> float:
    """Validate ``grad_data_d`` and return the Δ-stepping bucket width.

    A negative or NaN weight is rejected here: the kernel orders
    distances by their uint32 bit pattern and terminates only on strict
    decreases. Δ never changes the result, only the bucketing, so an
    approximate mean suffices (1.0 if the field is all zero).
    """
    import cupy as cp

    stats = cp.stack([grad_data_d.min(), grad_data_d.mean()]).get()
    g_min, mean_w = float(stats[0]), float(stats[1])
    if not (g_min >= 0.0):     # also catches NaN
        raise ValueError(
            "graph_distance_gpu: the solver requires non-negative "
            f"grad_data (edge weights); got min(grad_data) = {g_min!r}. "
            "Negative weights invert the kernel's uint32 distance "
            "ordering and remove its termination guarantee."
        )
    delta = delta_mult * mean_w
    return delta if delta > 0.0 else 1.0


def _check_topology(vertex_nbors_d, N: int, M: int) -> None:
    """Enforce the solver's hard topology preconditions: valence, the
    shared-memory bound on ``N`` (applied by the module build the probe
    triggers), slot range and symmetry -- the last two from one probe
    launch over the neighbour table.
    """
    from ._kernels_gpu import EDGE_SLOTS, probe_topology_cupy

    if M > EDGE_SLOTS:
        raise ValueError(
            f"graph_distance_gpu: valence M={M} exceeds the packed edge "
            f"table's EDGE_SLOTS={EDGE_SLOTS}."
        )
    flags = probe_topology_cupy(vertex_nbors_d, N, M)
    if flags & 2:
        raise ValueError(
            f"graph_distance_gpu: vertex_nbors has a slot outside the "
            f"valid 1-indexed range [0, {N}] (0 = empty slot); the "
            "solver would read out of bounds."
        )
    if flags & 1:
        raise ValueError(
            "graph_distance_gpu: vertex_nbors is not a symmetric relation. "
            "The kernel pushes v -> nbors[v], so an asymmetric table would "
            "solve the transposed graph."
        )


def _gpu_device_delta(vertex_nbors_d, grad_data_d, *, cap: int):
    """Δ-stepping solve over the gradient-weighted topology; returns
    the normalized ``(N, N)`` fp32 ``D[dest, source]`` matrix."""
    import cupy as cp
    from ._kernels_gpu import (
        DEFAULT_BLOCKS_PER_SM,
        DEFAULT_DELTA_MULT,
        DEFAULT_TPB,
        build_edge_table_cupy,
        get_delta_module,
        sm_count,
    )

    N, M = int(vertex_nbors_d.shape[0]), int(vertex_nbors_d.shape[1])
    mod = get_delta_module(N, M, tpb=DEFAULT_TPB, cap=cap)
    k_sssp = mod.get_function("sssp_delta_rows")
    k_tr = mod.get_function("transpose_scale")

    grid = (DEFAULT_BLOCKS_PER_SM * sm_count(),)
    block = (DEFAULT_TPB,)
    n_tiles = (N + 31) // 32

    work = cp.empty((N, N), dtype=cp.float32)
    gmax_d = cp.empty(1, dtype=cp.uint32)

    try:
        delta = _delta_for(grad_data_d, DEFAULT_DELTA_MULT)
        tab = build_edge_table_cupy(vertex_nbors_d, grad_data_d, N, M, mod)
        gmax_d.fill(0)
        k_sssp(grid, block,
               (work, tab, cp.int32(N), cp.int32(N),
                cp.float32(delta), gmax_d))
        del tab
        # gmax_d holds max|dist| as a raw fp32 bit pattern; distances
        # are non-negative, so the uint ordering matches the float.
        g_max = float(gmax_d.get().view(np.float32)[0])
        if not np.isfinite(g_max):
            raise RuntimeError(
                "graph_distance_gpu: disconnected mesh — at least one "
                "source-target pair is unreachable on the gradient-"
                "weighted graph. The CBIG step-0 caller assumes a "
                "connected icosphere; check vertex_nbors."
            )
        # Normalisation: fp64 reciprocal of the max rounded once to
        # fp32, then one fp32 multiply.
        inv = cp.float32(1.0 / g_max) if g_max > 0.0 else cp.float32(1.0)
        D = cp.empty((N, N), dtype=cp.float32)
        k_tr((n_tiles, n_tiles), (32, 8), (work, D, cp.int32(N), inv))
    finally:
        del work
    return D


def gradient_geodesic_distance_gpu_device(vertex_nbors_d,
                                          grad_data_d,
                                          *,
                                          cap: Optional[int] = None,
                                          ):
    """Device-resident variant: ``vertex_nbors_d`` and ``grad_data_d``
    are already on the GPU; returns the normalized ``(N, N)`` fp32
    distance matrix as a ``cupy.ndarray`` (also on device).

    The unified pipeline driver uses this path so the (N, N) matrix
    flows straight into the GPU diffusion_map's ``exp(-D / D.max())``
    transform without a host round-trip. The topology preconditions are
    checked before the solve. ``cap`` sizes the Δ solver's compacted
    frontier worklist; overflowing it is result-neutral, so it is a
    throughput knob only (default ``_kernels_gpu.DEFAULT_CAP``).
    """
    N, M = int(vertex_nbors_d.shape[0]), int(vertex_nbors_d.shape[1])
    _check_topology(vertex_nbors_d, N, M)
    from ._kernels_gpu import DEFAULT_CAP
    return _gpu_device_delta(vertex_nbors_d, grad_data_d,
                             cap=DEFAULT_CAP if cap is None else int(cap))


def gradient_geodesic_distance_gpu(verts: np.ndarray,
                                     vertex_nbors: np.ndarray,
                                     grad_data: np.ndarray,
                                     ) -> np.ndarray:
    """Drop-in replacement for the CPU ``gradient_geodesic_distance``.

    Same parameter contract as the CPU function. ``verts`` is accepted
    for API parity but unused inside (the algorithm only needs the
    graph topology + per-vertex gradient values).

    Returns the normalized (N, N) fp32 distance matrix as a numpy array
    on the host. Internal transfers are H2D (vertex_nbors + grad_data
    on launch) and D2H (the (N, N) matrix on return). For the pipeline
    fast path that avoids the D2H entirely see
    :func:`gradient_geodesic_distance_gpu_device`.
    """
    import cupy as cp

    _shape_check(verts, vertex_nbors, grad_data)
    # ``cp.asarray`` handles the numpy → device copy + dtype cast; the
    # CBIG vertex_nbors / grad_data are already C-contig on the host so
    # the result is contiguous without a separate ascontiguousarray pass.
    vn_d = cp.asarray(vertex_nbors, dtype=cp.int32)
    gd_d = cp.asarray(grad_data, dtype=cp.float32)

    D_d = gradient_geodesic_distance_gpu_device(vn_d, gd_d)

    return cp.asnumpy(D_d)
