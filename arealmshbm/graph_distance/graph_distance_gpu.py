"""graph_distance_gpu.py

GPU implementation of :func:`gradient_geodesic_distance`. Mirrors the
CPU API in ``graph_distance.py`` — same signature, same precision
contract, same output shape — but the all-pairs SSSP is replaced by
batched pull-based Bellman-Ford on the (N, N) distance matrix.

See ``_kernels_gpu.py`` for kernel-level documentation of:

  * The ``D[v, s]`` layout invariant (coalesced warp reads over s).
  * Pre-fused edge weights with +inf sentinel for absent slots
    (eliminates the per-iter `if slot == 0` branch in the hot loop).
  * Bank-conflict-free broadcast access to per-block shared neighbor
    + weight cache.
  * Templated M for full unroll of the slot loop.

Two entry points:

  * :func:`gradient_geodesic_distance_gpu` — numpy in, numpy out;
    drop-in replacement for the CPU function.
  * :func:`gradient_geodesic_distance_gpu_device` — device-array in,
    device-array out; lets the step-0 pipeline keep the (N, N) fp32
    distance matrix on-device for the subgraph C ``diffusion_map``
    that immediately consumes it (saves a ~670 MB D2H per hemi).

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


def gradient_geodesic_distance_gpu_device(vertex_nbors_d,
                                            grad_data_d,
                                            *,
                                            max_iters: Optional[int] = None,
                                            ):
    """Device-resident variant: ``vertex_nbors_d`` and ``grad_data_d``
    are already on the GPU; returns the normalized ``(N, N)`` fp32
    distance matrix as a ``cupy.ndarray`` (also on device).

    The unified pipeline driver uses this path so the (N, N) matrix
    flows straight into the GPU diffusion_map's ``exp(-D / D.max())``
    transform without a host round-trip.
    """
    import cupy as cp
    from ._kernels_gpu import (
        _BF_BLOCK_S,
        MAX_ITERS,
        get_bf_iter_kernel,
        init_distance_cupy,
        precompute_edge_weights_cupy,
    )

    N, M = int(vertex_nbors_d.shape[0]), int(vertex_nbors_d.shape[1])
    if max_iters is None:
        max_iters = MAX_ITERS

    # 1. Precompute (g_v + g_u) / 2 — small (N × M × 4 B); +inf
    #    sentinel for absent slots so the iter kernel needs no
    #    per-slot branch on validity.
    edge_weights_d = precompute_edge_weights_cupy(
        vertex_nbors_d, grad_data_d, N, M
    )

    # 2. Initialize D[v, s] = (v == s ? 0 : +inf). Single-buffer BF —
    #    the Gauss-Seidel kernel reads + writes the same array, halving
    #    iteration count vs Jacobi double-buffer on icosphere graphs.
    D = init_distance_cupy(N)

    # 3. Bellman-Ford loop (in-place / Gauss-Seidel).
    bf_kernel = get_bf_iter_kernel(M)
    changed_flag = cp.zeros(1, dtype=cp.int32)
    grid_x = (N + _BF_BLOCK_S - 1) // _BF_BLOCK_S
    grid_y = N
    grid = (grid_x, grid_y)
    block = (_BF_BLOCK_S,)

    # ``changed_flag.get()`` per-iter syncs at ~50 µs — cheap. We could
    # batch it (sync every K iters and pay K-1 wasted iters at most),
    # but at ~3-5 ms/iter the sync is well under 2 % overhead and
    # batching makes the convergence-iter-count reporting fuzzier.
    iters_done = 0
    for it in range(max_iters):
        changed_flag.fill(0)
        bf_kernel(grid, block,
                  (D, vertex_nbors_d, edge_weights_d,
                   np.int32(N), changed_flag))
        if int(changed_flag.get()) == 0:
            iters_done = it + 1
            break
    else:
        raise RuntimeError(
            f"graph_distance_gpu: BF did not converge within "
            f"max_iters={max_iters}. The icosphere graph diameter is "
            f"empirically ~30-50; non-convergence suggests a "
            f"disconnected mesh or a topology bug upstream."
        )

    # 4. Connectivity guard — if any cell is still +inf the mesh was
    #    disconnected. cupy's max() promotes to fp64 for the result
    #    scalar; we only need the finite check.
    g_max = cp.abs(D).max()
    if not cp.isfinite(g_max):
        raise RuntimeError(
            "graph_distance_gpu: disconnected mesh — at least one "
            "source-target pair is unreachable on the gradient-weighted "
            "graph. The CBIG step-0 caller assumes a connected "
            "icosphere; check vertex_nbors."
        )

    # 5. Normalize in place.
    if float(g_max) > 0.0:
        inv = cp.float32(1.0 / float(g_max))
        D *= inv

    return D


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

    N = _shape_check(verts, vertex_nbors, grad_data)
    # ``cp.asarray`` handles the numpy → device copy + dtype cast; the
    # CBIG vertex_nbors / grad_data are already C-contig on the host so
    # the result is contiguous without a separate ascontiguousarray pass.
    vn_d = cp.asarray(vertex_nbors, dtype=cp.int32)
    gd_d = cp.asarray(grad_data, dtype=cp.float32)

    D_d = gradient_geodesic_distance_gpu_device(vn_d, gd_d)

    # D2H copy. At fsa6 (N=12962) this is ~670 MB → ~150 ms on PCIe 5.0
    # (the dominant cost of the host-return variant). The device-resident
    # variant above skips this.
    return cp.asnumpy(D_d)
