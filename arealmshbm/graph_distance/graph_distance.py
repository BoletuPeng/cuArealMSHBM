"""graph_distance.py

All-pairs geodesic distance on a vertex-attached gradient graph.

Direct port of ``CBIG_SPGrad_create_graph.m`` + MATLAB ``distances``.

The MATLAB source builds an undirected graph where every entry of
``vertexNbors`` (excluding the 0 sentinel) contributes an edge whose
weight is the average of the two endpoint gradient values:

  for n = 1 : size(vertexNbors, 1)            % outer = neighbor slot
      neighbors  = vertexNbors(n, :);
      mask       = neighbors ~= 0;
      grad_w     = (grad_data(mask) + grad_data(neighbors(mask))) / 2;
      G = [G ; addedges(verts(mask), neighbors(mask), grad_w)];

For an undirected mesh each (u, v) appears twice (once as ``u``'s
neighbor of ``v`` and once as ``v``'s neighbor of ``u``). The weight
``w_uv = (g_u + g_v)/2`` is symmetric in (u, v).

Pairwise shortest distances are then normalised to ``[0, 1]`` per the
MATLAB rule ``lh_dist = single(lh_dist / max(max(abs(lh_dist))))``.

Implementation
--------------
Numba ``@njit(parallel=True)`` with ``prange`` over source vertices.
Each thread owns one row of the (N, N) distance matrix and runs a
single-source Dijkstra with its own scratch buffers (binary min-heap +
position back-reference). The 24-core box runs 24 SSSPs concurrently,
turning the ~30 s scipy single-thread baseline into a ~2 s run.

Edge weights are computed on-the-fly inside the SSSP loop (one fp32
add + multiply per relaxation), so we don't materialise a CSR weight
matrix. The neighbor list itself is just the input ``vertex_nbors``
(N, max_neigh) int32 1-indexed table, used directly.

Precision contract
------------------
- ``vertex_nbors`` int32 1-indexed; ``grad_data`` fp32.
- Heap priorities + distance buffers are fp32 (the per-vertex
  gradient values are fp32 and the longest Dijkstra path on the
  12962-vert icosphere stays well within fp32 dynamic range).
- Output ``dist`` is fp32, normalised so ``dist.max() == 1``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import numba as nb
from numba import prange


# ─────────────────────────────────────────────────────────────────────────
# Indexed binary min-heap helpers — same pattern as
# surface_smoothing/_geodesic_kernels.py but specialised to fp32 keys.
# Each helper takes (heap_node, heap_key, heap_pos, heap_size, ...).
#   heap_node : int32 (N,)
#   heap_key  : fp32  (N,)
#   heap_pos  : int32 (N,) — heap position of each vertex; -1 if absent.
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False, inline="always")
def _heap_push(heap_node, heap_key, heap_pos, heap_size, node, key):
    i = heap_size
    heap_node[i] = node
    heap_key[i] = key
    heap_pos[node] = i
    while i > 0:
        parent = (i - 1) >> 1
        if heap_key[parent] > heap_key[i]:
            pn = heap_node[parent]
            pk = heap_key[parent]
            heap_node[parent] = heap_node[i]
            heap_key[parent] = heap_key[i]
            heap_node[i] = pn
            heap_key[i] = pk
            heap_pos[heap_node[parent]] = parent
            heap_pos[heap_node[i]] = i
            i = parent
        else:
            break
    return heap_size + 1


@nb.njit(cache=True, fastmath=False, boundscheck=False, inline="always")
def _heap_pop(heap_node, heap_key, heap_pos, heap_size):
    """Remove and discard min; caller already read top via heap_node[0]/heap_key[0]."""
    last = heap_size - 1
    heap_pos[heap_node[0]] = -1
    if last == 0:
        return 0
    heap_node[0] = heap_node[last]
    heap_key[0] = heap_key[last]
    heap_pos[heap_node[0]] = 0
    new_size = last
    i = 0
    while True:
        l = 2 * i + 1
        r = 2 * i + 2
        smallest = i
        if l < new_size and heap_key[l] < heap_key[smallest]:
            smallest = l
        if r < new_size and heap_key[r] < heap_key[smallest]:
            smallest = r
        if smallest == i:
            break
        sn = heap_node[smallest]
        sk = heap_key[smallest]
        heap_node[smallest] = heap_node[i]
        heap_key[smallest] = heap_key[i]
        heap_node[i] = sn
        heap_key[i] = sk
        heap_pos[heap_node[smallest]] = smallest
        heap_pos[heap_node[i]] = i
        i = smallest
    return new_size


@nb.njit(cache=True, fastmath=False, boundscheck=False, inline="always")
def _heap_decrease(heap_node, heap_key, heap_pos, idx_in_heap, new_key):
    i = idx_in_heap
    heap_key[i] = new_key
    while i > 0:
        parent = (i - 1) >> 1
        if heap_key[parent] > heap_key[i]:
            pn = heap_node[parent]
            pk = heap_key[parent]
            heap_node[parent] = heap_node[i]
            heap_key[parent] = heap_key[i]
            heap_node[i] = pn
            heap_key[i] = pk
            heap_pos[heap_node[parent]] = parent
            heap_pos[heap_node[i]] = i
            i = parent
        else:
            break


# ─────────────────────────────────────────────────────────────────────────
# All-pairs Dijkstra, prange over sources.
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, parallel=True, fastmath=False, boundscheck=False)
def _all_pairs_dijkstra(vertex_nbors, grad_data, dist):
    """Fill ``dist[s, t]`` with shortest path from s to t for every s.

    ``vertex_nbors`` is (N, M) int32 1-indexed; 0 marks absent slot.
    ``grad_data`` is (N,) fp32. Edge weight w(i, j) = (g_i + g_j)/2.
    ``dist`` is (N, N) fp32, written in full.
    """
    N = vertex_nbors.shape[0]
    M = vertex_nbors.shape[1]
    INF = np.float32(np.inf)

    for s in prange(N):
        # Per-thread scratch (allocated inside prange body so each
        # parallel iteration gets its own buffers).
        heap_node = np.empty(N, dtype=np.int32)
        heap_key = np.empty(N, dtype=np.float32)
        heap_pos = np.full(N, -1, dtype=np.int32)
        d = np.full(N, INF, dtype=np.float32)
        visited = np.zeros(N, dtype=np.uint8)

        d[s] = np.float32(0.0)
        heap_size = _heap_push(heap_node, heap_key, heap_pos, 0,
                               s, np.float32(0.0))

        while heap_size > 0:
            u = heap_node[0]
            du = heap_key[0]
            heap_size = _heap_pop(heap_node, heap_key, heap_pos, heap_size)
            if visited[u] != 0:
                continue
            visited[u] = 1
            gu = grad_data[u]

            # Relax each neighbor slot.
            for slot in range(M):
                v1 = vertex_nbors[u, slot]
                if v1 == 0:
                    continue
                v = v1 - 1   # 1-indexed → 0-indexed
                if visited[v] != 0:
                    continue
                # Edge weight w(u, v) = (grad_data[u] + grad_data[v]) / 2.
                w = (gu + grad_data[v]) * np.float32(0.5)
                cand = du + w
                if cand < d[v]:
                    d[v] = cand
                    pos = heap_pos[v]
                    if pos < 0:
                        heap_size = _heap_push(
                            heap_node, heap_key, heap_pos,
                            heap_size, v, cand,
                        )
                    else:
                        _heap_decrease(heap_node, heap_key, heap_pos,
                                       pos, cand)

        # Copy row into output. Unreachable targets keep their INF
        # value — the caller checks for any leftover inf and raises,
        # rather than silently substituting zero (which would create
        # a fake zero-distance shortcut and propagate into diffusion_map).
        for t in range(N):
            dist[s, t] = d[t]


@nb.njit(cache=True, parallel=True, fastmath=False, boundscheck=False)
def _row_max(dist):
    """Return global max of |dist|. fp32 input; fp32 reduction."""
    N = dist.shape[0]
    row_maxes = np.zeros(N, dtype=np.float32)
    for i in prange(N):
        m = np.float32(0.0)
        for j in range(N):
            v = dist[i, j]
            if v < 0:
                v = -v
            if v > m:
                m = v
        row_maxes[i] = m
    g = np.float32(0.0)
    for i in range(N):
        if row_maxes[i] > g:
            g = row_maxes[i]
    return g


@nb.njit(cache=True, parallel=True, fastmath=False, boundscheck=False)
def _scale_inplace(dist, inv):
    """In-place ``dist *= inv``."""
    N = dist.shape[0]
    for i in prange(N):
        for j in range(N):
            dist[i, j] = dist[i, j] * inv


# ─────────────────────────────────────────────────────────────────────────
# Public entry point — kept as a thin Python signature-validator; the
# heavy lifting is in the @njit kernel above so the GIL is released
# across the prange. (numba prange requires a numba-compiled call site,
# so the kernel itself must be @njit; this wrapper does not add
# meaningful overhead — Python-side cost is shape validation only.)
# ─────────────────────────────────────────────────────────────────────────
def gradient_geodesic_distance(verts: np.ndarray,
                               vertex_nbors: np.ndarray,
                               grad_data: np.ndarray,
                               ) -> np.ndarray:
    """Build the gradient-weighted graph and return all-pairs
    geodesic distances normalised to ``[0, 1]``.

    Parameters
    ----------
    verts : (N, 3) fp32
        Vertex coordinates. Currently unused inside the function
        (graph topology comes from ``vertex_nbors``); kept in the
        signature to mirror MATLAB's ``sbjMesh`` argument and for an
        optional shape sanity check.
    vertex_nbors : (N, max_neigh) int32
        1-indexed neighbor table. Slot value ``0`` marks an absent
        neighbor.
    grad_data : (N,) fp32
        Gradient density at each vertex.

    Returns
    -------
    dist : (N, N) fp32
        Normalised geodesic distance, symmetric (up to fp tolerance),
        zero diagonal, ``dist.max() == 1`` after normalisation by the
        global maximum.
    """
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

    vn = np.ascontiguousarray(vertex_nbors, dtype=np.int32)
    gd = np.ascontiguousarray(grad_data, dtype=np.float32)

    dist = np.empty((N, N), dtype=np.float32)
    _all_pairs_dijkstra(vn, gd, dist)

    g = float(_row_max(dist))
    if not np.isfinite(g):
        # _row_max returned inf → at least one source-target pair was
        # unreachable. Refuse to silently zero-fill (which would inject
        # a fake zero-distance edge into the diffusion-map input).
        raise RuntimeError(
            "graph_distance: disconnected mesh — at least one source-target "
            "pair is unreachable on the gradient-weighted graph. The CBIG "
            "step-0 caller assumes a connected icosphere; check vertex_nbors."
        )
    if g > 0.0:
        _scale_inplace(dist, np.float32(1.0 / g))
    return dist
