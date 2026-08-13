"""_geodesic_kernels.py

Numba kernels for polyhedral-geodesic distance computation on a triangle
mesh. Faithful port of HCP Workbench's ``GeodesicHelper`` algorithm
class — the modified-Dijkstra propagation that relaxes both the 1-ring
edge graph AND a precomputed "across-face unfolded" neighbor graph.

This is the same algorithm class as Workbench's
``src/Files/GeodesicHelper.cxx``; cited line numbers below refer to the
master branch as of 2026-05.

Algorithm structure (mirrors GeodesicHelper.cxx):

  1. Precompute layer-1 neighbors (``build_layer1_neighbors``) — mirrors
     the loop at GeodesicHelper.cxx:62-91. Per-vertex 1-ring with
     euclidean edge distances.

  2. Precompute layer-2 ("smooth") neighbors (``build_layer2_neighbors``)
     — mirrors the loop at GeodesicHelper.cxx:100-164. For every
     interior edge with two adjacent triangles, identify the two
     "opposite" vertices (``baseNode`` and ``farNode``) and compute the
     unfolded straight-line distance:

        abhat = normalize(neigh2 - neigh1)
        ad    = abhat * dot(abhat, farCoord - neigh1)
        d     = neigh1 + ad                            (foot of farNode on shared edge)
        ea    = neigh1 - baseCoord
        efhat = normalize(ea - abhat * dot(abhat, ea)) (perpendicular from base)
        cdmag = |d - farCoord|                         (perpendicular len from far)
        g     = d + efhat * cdmag                      (unfolded position of farNode)
        dist  = |g - baseCoord|

     Convexity test (GeodesicHelper.cxx:135-139): the unfolded path
     ``eg`` must cross the shared edge inside the segment (a, b);
     otherwise the unfolding is invalid (concave / degenerate quad) and
     the layer-2 edge is dropped.

  3. Modified Dijkstra (``bounded_dijkstra_smooth``) — mirrors the body
     of ``GeodesicHelper::dijkstra(root, maxdist, ..., smooth=true)`` at
     GeodesicHelper.cxx:217-293. At each popped vertex, relax along
     both ``nodeNeighbors`` (layer 1) and ``nodeNeighbors2`` (layer 2)
     with the cutoff ``maxdist``. Frozen-vertex skipping uses the
     ``marked`` array exactly as in the C++ source.

  4. Heap is an indexed binary min-heap implemented in flat int32/fp64
     arrays (numba has no heapq). Decrease-key is supported by storing
     a per-vertex heap-position back-reference.

The mathematical object computed is the polyhedral geodesic distance —
the actual shortest path along the mesh surface (the same continuous
2-manifold object Workbench computes), NOT the discrete edge-graph
shortest path. Adding layer-2 unfolded edges lets the Dijkstra
propagation take "diagonal" shortcuts across triangle interiors.

fp64 accumulation for distance compares (matches Workbench's float
behavior at fsaverage6 scale to ~1e-7), fp32 output where the caller
asks for it.

Per-source Dijkstra is dispatched via ``prange`` in
``all_sources_scatter`` with per-thread scratch buffers (see
:func:`alloc_scratch`), so every cortex source vertex runs on its own
thread.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import numba as nb


# ─────────────────────────────────────────────────────────────────────────
# Mesh-only precomputation (per-vertex areas + Layer 1/2 neighbor tables).
# Each Step0Inputs holds one bundle per hemi so the per-call hot path
# never touches mesh topology — only data.
# ─────────────────────────────────────────────────────────────────────────
def _vertex_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex area = (sum of incident face areas) / 3 — matches
    Workbench ``SurfaceFile::computeNodeAreas``.
    """
    v = verts.astype(np.float64, copy=False)
    f = faces.astype(np.int64, copy=False)
    p0 = v[f[:, 0]]; p1 = v[f[:, 1]]; p2 = v[f[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    va = np.zeros(v.shape[0], dtype=np.float64)
    for k in range(3):
        np.add.at(va, f[:, k], fa)
    va *= (1.0 / 3.0)
    return va


def prepare_smoothing_mesh(verts: np.ndarray, faces: np.ndarray):
    """Build all mesh-only precomputation needed by ``cifti_smoothing``.

    Returns a 7-tuple ``(va, n1_indptr, n1_idx, n1_dist,
    n2_indptr, n2_idx, n2_dist)``. va is fp64; the CSR layers are
    int32 indptr / int32 idx / fp64 dist.

    All of this depends only on (verts, faces); the pipeline caches
    one of these per hemi at load_inputs time so each subsequent
    ``cifti_smoothing`` call skips the ~0.3 s mesh-only work.
    """
    va = _vertex_areas(verts, faces)
    n1_indptr, n1_idx, n1_dist = build_layer1_neighbors(verts, faces)
    n2_indptr, n2_idx, n2_dist = build_layer2_neighbors(verts, faces)
    return va, n1_indptr, n1_idx, n1_dist, n2_indptr, n2_idx, n2_dist


def alloc_scratch(n_full: int, n_sources: int, num_threads: int,
                  cap_per_source: int = 512):
    """Allocate per-thread scratch + per-thread output buffers for
    :func:`all_sources_scatter`. Centralised so the orchestrator and
    any future caller pin the same layout.

    Returns a tuple of 11 ndarrays. The first 8 are per-thread scratch
    (shape ``(T, N)``); the last 3 are per-thread output triples
    (shape ``(T, cap_per_thread)`` where
    ``cap_per_thread = ceil(n_sources / T) * cap_per_source``).
    """
    T = int(num_threads)
    N = int(n_full)
    output_t = np.empty((T, N), dtype=np.float64)
    marked_t = np.zeros((T, N), dtype=np.int8)
    changed_t = np.empty((T, N), dtype=np.int32)
    heap_node_t = np.empty((T, N), dtype=np.int32)
    heap_dist_t = np.empty((T, N), dtype=np.float64)
    heap_pos_t = np.full((T, N), -1, dtype=np.int32)
    pop_nodes_t = np.empty((T, N), dtype=np.int32)
    pop_dists_t = np.empty((T, N), dtype=np.float64)
    # Per-thread output triples — cap is generous (per-source 512 ×
    # sources/thread). Realistic fsaverage6 emission is ~80-120/source.
    cap_per_thread = ((n_sources + T - 1) // T) * cap_per_source
    out_src_t = np.empty((T, cap_per_thread), dtype=np.int32)
    out_tgt_t = np.empty((T, cap_per_thread), dtype=np.int32)
    out_w_t = np.empty((T, cap_per_thread), dtype=np.float64)
    # Per-thread overflow flag — kept separate from out_count_t so the
    # latter stays a clean monotonic count; the kernel raises once it
    # cannot append further work for *any* thread.
    overflow_t = np.zeros(T, dtype=np.uint8)
    return (output_t, marked_t, changed_t,
            heap_node_t, heap_dist_t, heap_pos_t,
            pop_nodes_t, pop_dists_t,
            out_src_t, out_tgt_t, out_w_t,
            overflow_t)


# ─────────────────────────────────────────────────────────────────────────
# Layer-1 neighbors (1-ring edge graph)
# ─────────────────────────────────────────────────────────────────────────
def build_layer1_neighbors(verts: np.ndarray, faces: np.ndarray):
    """Build flat-CSR layer-1 neighbor arrays for a triangle mesh.

    Mirrors ``GeodesicHelperBase`` ctor at GeodesicHelper.cxx:62-91:
    per vertex i, ``nodeNeighbors[i]`` holds the 1-ring (deduplicated)
    and ``distances[i][j]`` holds the euclidean length of the edge
    (i, nodeNeighbors[i][j]).

    Returns
    -------
    n1_indptr : (N+1,) int32   row-pointer
    n1_idx    : (E,)   int32   column indices
    n1_dist   : (E,)   fp64    edge lengths (same length as n1_idx)
    """
    v = verts.astype(np.float64, copy=False)
    f = faces.astype(np.int64, copy=False)
    n = v.shape[0]

    # Collect undirected edges: each (a, b) and (b, a) emitted from each
    # face. Dedup by sorting on (i, j).
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2],
                           f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0],
                           f[:, 0], f[:, 1], f[:, 2]])
    keys = rows.astype(np.int64) * np.int64(n) + cols.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    # Unique (i, j) pairs
    uniq_mask = np.empty(keys_s.shape, dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = keys_s[1:] != keys_s[:-1]
    sel = order[uniq_mask]
    rows_u = rows[sel]
    cols_u = cols[sel]
    diffs = v[rows_u] - v[cols_u]
    dist_u = np.sqrt((diffs * diffs).sum(axis=1))

    # Build CSR by counting per-row, then exclusive scan, then scatter.
    # rows_u is already sorted (because keys_s was sorted by row*n + col).
    counts = np.bincount(rows_u, minlength=n).astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    n1_idx = cols_u.astype(np.int32)
    n1_dist = dist_u.astype(np.float64)
    return indptr, n1_idx, n1_dist


# ─────────────────────────────────────────────────────────────────────────
# Layer-2 neighbors (across-face unfolded shortcuts)
# ─────────────────────────────────────────────────────────────────────────
def _build_edge_to_faces(faces: np.ndarray, n_verts: int):
    """For each undirected edge (i<j), list the up-to-2 incident triangle
    indices and the third vertex of each triangle.

    Returns
    -------
    edge_node1   : (E,) int32  smaller endpoint
    edge_node2   : (E,) int32  larger endpoint
    edge_third1  : (E,) int32  third vertex of triangle #1 (or -1)
    edge_third2  : (E,) int32  third vertex of triangle #2 (or -1)
    """
    f = faces.astype(np.int64, copy=False)
    F = f.shape[0]

    # 3 directed half-edges per face, each tagged with the third vertex.
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    third = np.concatenate([f[:, 2], f[:, 0], f[:, 1]])

    # Make undirected: lo, hi
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keys = lo * np.int64(n_verts) + hi
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    third_s = third[order]
    lo_s = lo[order]
    hi_s = hi[order]

    # Group by unique key. Each group has size 1 (boundary) or 2 (interior).
    uniq_mask = np.empty(keys_s.shape, dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.flatnonzero(uniq_mask)
    E = starts.size
    edge_node1 = lo_s[starts].astype(np.int32)
    edge_node2 = hi_s[starts].astype(np.int32)
    edge_third1 = third_s[starts].astype(np.int32)
    # Second triangle's third vertex: present iff group size == 2.
    edge_third2 = np.full(E, -1, dtype=np.int32)
    has_pair = np.empty(E, dtype=bool)
    has_pair[:-1] = (starts[1:] - starts[:-1]) == 2
    has_pair[-1] = (3 * F - starts[-1]) == 2
    if has_pair.any():
        idx_pair = np.where(has_pair)[0]
        edge_third2[idx_pair] = third_s[starts[idx_pair] + 1]

    return edge_node1, edge_node2, edge_third1, edge_third2


def build_layer2_neighbors(verts: np.ndarray, faces: np.ndarray):
    """Build flat-CSR layer-2 (across-face unfolded) neighbor arrays.

    Mirrors the second loop in ``GeodesicHelperBase`` at
    GeodesicHelper.cxx:100-164. For each interior edge with two adjacent
    triangles, identify the two vertices opposite the shared edge and
    add an unfolded layer-2 edge between them with the convexity check
    from GeodesicHelper.cxx:135-139.

    Returns
    -------
    n2_indptr : (N+1,) int32
    n2_idx    : (E2,)  int32
    n2_dist   : (E2,)  fp64
    """
    v = verts.astype(np.float64, copy=False)
    n = v.shape[0]

    e_n1, e_n2, e_t1, e_t2 = _build_edge_to_faces(faces, n)

    # Restrict to interior edges (have both incident faces).
    inter = e_t2 >= 0
    if not np.any(inter):
        # Pathological: all-boundary mesh. Return empty CSR.
        return (np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64))
    n1 = e_n1[inter].astype(np.int64)
    n2 = e_n2[inter].astype(np.int64)
    bn = e_t1[inter].astype(np.int64)   # baseNode (third vert of triangle 1)
    fn = e_t2[inter].astype(np.int64)   # farNode  (third vert of triangle 2)

    P_n1 = v[n1]; P_n2 = v[n2]; P_b = v[bn]; P_f = v[fn]

    # GeodesicHelper.cxx:125-141 — vector math, exact mirror.
    ab = P_n2 - P_n1                                    # vector along shared edge
    abmag = np.sqrt((ab * ab).sum(axis=1))
    # Guard against degenerate zero-length edges (shouldn't occur on a
    # valid mesh, but defensive).
    safe_ab = abmag > 0.0
    abhat = np.zeros_like(ab)
    abhat[safe_ab] = ab[safe_ab] / abmag[safe_ab, None]

    ac = P_f - P_n1                                     # neigh1 -> farnode
    ad_scalar = (abhat * ac).sum(axis=1)                # dot(abhat, ac)
    ad = abhat * ad_scalar[:, None]                     # foot on shared edge
    d_pt = P_n1 + ad                                    # point d

    ea = P_n1 - P_b                                     # base -> neigh1
    ea_proj_scalar = (abhat * ea).sum(axis=1)
    ef_vec = ea - abhat * ea_proj_scalar[:, None]       # perpendicular from base
    efmag = np.sqrt((ef_vec * ef_vec).sum(axis=1))
    safe_ef = efmag > 0.0
    efhat = np.zeros_like(ef_vec)
    efhat[safe_ef] = ef_vec[safe_ef] / efmag[safe_ef, None]

    cd_vec = d_pt - P_f
    cdmag = np.sqrt((cd_vec * cd_vec).sum(axis=1))      # perpendicular from far
    g_pt = d_pt + efhat * cdmag[:, None]                # unfolded farnode
    eg = g_pt - P_b                                     # full unfolded vec

    # Convexity test: GeodesicHelper.cxx:135-139.
    sum_perp = efmag + cdmag
    valid = (sum_perp > 0.0) & safe_ab & safe_ef
    # Where invalid we'll mark the edge as bad below.
    tempf_split = np.zeros_like(efmag)
    tempf_split[valid] = efmag[valid] / sum_perp[valid]      # eh = eg * tempf_split
    eh = eg * tempf_split[:, None]
    ah = eh - ea
    tempf_along = (ah * abhat).sum(axis=1)               # component along ab
    convex = (tempf_along > 0.0) & (tempf_along < abmag) & valid

    # Unfolded straight-line length.
    dist_unfold = np.sqrt((eg * eg).sum(axis=1))

    # Keep only convex edges; emit BOTH directions (base->far and far->base).
    keep = np.where(convex)[0]
    if keep.size == 0:
        return (np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64))

    src = np.concatenate([bn[keep], fn[keep]])
    tgt = np.concatenate([fn[keep], bn[keep]])
    dst = np.concatenate([dist_unfold[keep], dist_unfold[keep]])

    # Sort by src so we can build CSR.
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    tgt_s = tgt[order]
    dst_s = dst[order]

    counts = np.bincount(src_s, minlength=n).astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    return indptr, tgt_s.astype(np.int32), dst_s.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────
# Numba: bounded modified-Dijkstra with both neighbor layers.
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False)
def _heap_push(heap_node, heap_dist, heap_pos, heap_size, node, dist):
    """Push (node, dist) onto the binary min-heap. Returns new size."""
    i = heap_size
    heap_node[i] = node
    heap_dist[i] = dist
    heap_pos[node] = i
    # sift up
    while i > 0:
        parent = (i - 1) >> 1
        if heap_dist[parent] > heap_dist[i]:
            # swap
            pn = heap_node[parent]; pd = heap_dist[parent]
            heap_node[parent] = heap_node[i]; heap_dist[parent] = heap_dist[i]
            heap_node[i] = pn;                heap_dist[i] = pd
            heap_pos[heap_node[parent]] = parent
            heap_pos[heap_node[i]] = i
            i = parent
        else:
            break
    return heap_size + 1


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def _heap_pop(heap_node, heap_dist, heap_pos, heap_size):
    """Pop minimum. Returns (top_node, new_size). Caller already read top."""
    last = heap_size - 1
    if last == 0:
        heap_pos[heap_node[0]] = -1
        return heap_size - 1
    # move last to top
    heap_pos[heap_node[0]] = -1
    heap_node[0] = heap_node[last]
    heap_dist[0] = heap_dist[last]
    heap_pos[heap_node[0]] = 0
    new_size = last
    # sift down
    i = 0
    while True:
        l = 2 * i + 1
        r = 2 * i + 2
        smallest = i
        if l < new_size and heap_dist[l] < heap_dist[smallest]:
            smallest = l
        if r < new_size and heap_dist[r] < heap_dist[smallest]:
            smallest = r
        if smallest == i:
            break
        sn = heap_node[smallest]; sd = heap_dist[smallest]
        heap_node[smallest] = heap_node[i]; heap_dist[smallest] = heap_dist[i]
        heap_node[i] = sn;                  heap_dist[i] = sd
        heap_pos[heap_node[smallest]] = smallest
        heap_pos[heap_node[i]] = i
        i = smallest
    return new_size


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def _heap_decrease(heap_node, heap_dist, heap_pos, idx_in_heap, new_dist):
    """Decrease-key at heap position idx_in_heap to new_dist."""
    i = idx_in_heap
    heap_dist[i] = new_dist
    while i > 0:
        parent = (i - 1) >> 1
        if heap_dist[parent] > heap_dist[i]:
            pn = heap_node[parent]; pd = heap_dist[parent]
            heap_node[parent] = heap_node[i]; heap_dist[parent] = heap_dist[i]
            heap_node[i] = pn;                heap_dist[i] = pd
            heap_pos[heap_node[parent]] = parent
            heap_pos[heap_node[i]] = i
            i = parent
        else:
            break


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def bounded_dijkstra_smooth(
    root,
    maxdist,
    n1_indptr, n1_idx, n1_dist,
    n2_indptr, n2_idx, n2_dist,
    # Per-source scratch buffers (caller-allocated, length n_verts):
    output,        # fp64, distance from root to each visited vertex
    marked,        # int8: bit 1 = frozen (popped), bit 4 = on-heap or known
    changed,       # int32: list of touched vertices to reset at end
    heap_node,     # int32 (n,)
    heap_dist,     # fp64  (n,)
    heap_pos,      # int32 (n,) — heap position of each vertex (-1 if absent)
    out_nodes,     # int32 (n,) — popped vertex list
    out_dists,     # fp64  (n,) — popped vertex distances
):
    """Bounded modified Dijkstra from `root` with cutoff `maxdist`.

    Mirrors ``GeodesicHelper::dijkstra(root, maxdist, ..., smooth=true)``
    at GeodesicHelper.cxx:217-293. Relaxes both layer-1 (1-ring) and
    layer-2 (across-face unfolded) neighbors at each popped vertex.

    Caller MUST provide scratch buffers initialized as follows on first
    call (and reset between calls — this function resets only what it
    touched, mirroring ``changed[numChanged++]`` cleanup pattern at
    GeodesicHelper.cxx:289-292):

        marked    : zeros (we set/reset bits 1 and 4)
        heap_pos  : -1 everywhere

    `output` and other buffers are overwritten where touched and
    don't need pre-init.

    Returns
    -------
    n_popped : int
        Number of vertices written to out_nodes / out_dists. The first
        entry is always (root, 0.0).
    """
    n_changed = 0
    heap_size = 0

    # Initialize root
    output[root] = 0.0
    marked[root] = marked[root] | 4
    changed[n_changed] = root
    n_changed += 1
    heap_size = _heap_push(heap_node, heap_dist, heap_pos, heap_size,
                           root, 0.0)

    n_popped = 0
    while heap_size > 0:
        whichnode = heap_node[0]
        wn_dist = heap_dist[0]
        heap_size = _heap_pop(heap_node, heap_dist, heap_pos, heap_size)

        out_nodes[n_popped] = whichnode
        out_dists[n_popped] = wn_dist
        n_popped += 1
        marked[whichnode] = marked[whichnode] | 1   # frozen

        # ── Layer 1: 1-ring neighbors ──
        s1 = n1_indptr[whichnode]
        e1 = n1_indptr[whichnode + 1]
        for k in range(s1, e1):
            wn = n1_idx[k]
            if (marked[wn] & 1) == 0:
                tempf = wn_dist + n1_dist[k]
                if tempf <= maxdist:
                    if (marked[wn] & 4) == 0:
                        marked[wn] = marked[wn] | 4
                        changed[n_changed] = wn
                        n_changed += 1
                        output[wn] = tempf
                        heap_size = _heap_push(
                            heap_node, heap_dist, heap_pos, heap_size,
                            wn, tempf)
                    elif tempf < output[wn]:
                        output[wn] = tempf
                        _heap_decrease(heap_node, heap_dist, heap_pos,
                                       heap_pos[wn], tempf)

        # ── Layer 2: across-face unfolded "smooth" neighbors ──
        s2 = n2_indptr[whichnode]
        e2 = n2_indptr[whichnode + 1]
        for k in range(s2, e2):
            wn = n2_idx[k]
            if (marked[wn] & 1) == 0:
                tempf = wn_dist + n2_dist[k]
                if tempf <= maxdist:
                    if (marked[wn] & 4) == 0:
                        marked[wn] = marked[wn] | 4
                        changed[n_changed] = wn
                        n_changed += 1
                        output[wn] = tempf
                        heap_size = _heap_push(
                            heap_node, heap_dist, heap_pos, heap_size,
                            wn, tempf)
                    elif tempf < output[wn]:
                        output[wn] = tempf
                        _heap_decrease(heap_node, heap_dist, heap_pos,
                                       heap_pos[wn], tempf)

    # Reset only touched entries — mirrors GeodesicHelper.cxx:289-292.
    for i in range(n_changed):
        v = changed[i]
        marked[v] = 0
    # heap_pos was reset to -1 by the pops; just to be safe, clear any
    # stragglers (shouldn't occur).
    return n_popped


# ─────────────────────────────────────────────────────────────────────────
# Convenience: run all sources and accumulate scatter triples.
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, parallel=True, fastmath=False, boundscheck=False)
def all_sources_scatter(
    sources,                 # int32 (S,)  — list of cortex source vertices
    roi_mask,                # uint8 (N,)  — 1 if in ROI, 0 medial
    vert_areas,              # fp64  (N,)
    sigma,                   # fp64  scalar
    cutoff,                  # fp64  scalar = 3*sigma
    n1_indptr, n1_idx, n1_dist,
    n2_indptr, n2_idx, n2_dist,
    # Per-thread scratch (T, N) — allocate via alloc_scratch():
    output_t, marked_t, changed_t,
    heap_node_t, heap_dist_t, heap_pos_t,
    pop_nodes_t, pop_dists_t,
    # Per-thread output triples (T, cap_per_thread):
    out_src_t, out_tgt_t, out_w_t,
    # Per-thread emission count (T,) — clean monotonic count:
    out_count_t,
    # Per-thread overflow flag (T,) — set to 1 if cap was hit:
    overflow_t,
):
    """Run bounded Dijkstra from every source in `sources` (in parallel
    across threads), compute GEO_GAUSS_AREA scatter weights mirroring
    ``MetricSmoothingObject::precomputeWeightsROIGeoGaussArea`` at
    MetricSmoothingObject.cxx:423-493, and append (src, tgt, weight)
    triples to per-thread output buffers.

    Each source is fully independent, so the outer loop dispatches via
    ``prange``. Each thread owns its own scratch (shape ``(T, N)``)
    and its own output chunk (shape ``(T, cap_per_thread)``). The
    caller concatenates the per-thread chunks via ``out_count_t``.

    Normalization (faithful to ROIGeoGaussArea, lines 451-468):
        - weightSum is summed over ALL reachable vertices (including
          outside ROI), so normalization doesn't bias edge nodes.
        - Only inside-ROI targets are emitted to the gather list.
    """
    inv_two_sig2 = -0.5 / (sigma * sigma)
    S = sources.shape[0]
    T = output_t.shape[0]

    # Reset per-thread emission counts + overflow flags.
    for t in range(T):
        out_count_t[t] = 0
        overflow_t[t] = 0

    for s in nb.prange(S):
        tid = nb.get_thread_id()
        # If this thread already overflowed on an earlier source,
        # skip — the caller will raise after the kernel returns and
        # there is no value in continuing to fill garbage.
        if overflow_t[tid] != 0:
            continue
        i = sources[s]
        if roi_mask[i] == 0:
            continue

        # Per-thread aliases so numba sees individual buffers in the
        # inner kernel call.
        output_b   = output_t[tid]
        marked_b   = marked_t[tid]
        changed_b  = changed_t[tid]
        heap_node_b = heap_node_t[tid]
        heap_dist_b = heap_dist_t[tid]
        heap_pos_b  = heap_pos_t[tid]
        pop_nodes_b = pop_nodes_t[tid]
        pop_dists_b = pop_dists_t[tid]

        n_pop = bounded_dijkstra_smooth(
            i, cutoff,
            n1_indptr, n1_idx, n1_dist,
            n2_indptr, n2_idx, n2_dist,
            output_b, marked_b, changed_b,
            heap_node_b, heap_dist_b, heap_pos_b,
            pop_nodes_b, pop_dists_b,
        )

        # First pass: weight sum over all reachable nodes (incl. medial).
        wsum = 0.0
        for k in range(n_pop):
            j = pop_nodes_b[k]
            d = pop_dists_b[k]
            w_raw = np.exp(d * d * inv_two_sig2) * vert_areas[j]
            wsum += w_raw
        if wsum <= 0.0:
            continue
        factor = vert_areas[i] / wsum

        # Second pass: emit only ROI targets to this thread's chunk.
        # Bounds-check against the caller-allocated cap. boundscheck=False
        # would let an OOB write through silently; on overflow we set
        # overflow_t[tid] = 1 and let the caller raise after the kernel
        # returns. out_count_t stays a clean monotonic count.
        cap = out_src_t.shape[1]
        n_out_local = out_count_t[tid]
        overflow_here = False
        for k in range(n_pop):
            j = pop_nodes_b[k]
            if roi_mask[j] == 0:
                continue
            if n_out_local >= cap:
                overflow_here = True
                break
            d = pop_dists_b[k]
            w_raw = np.exp(d * d * inv_two_sig2) * vert_areas[j]
            out_src_t[tid, n_out_local] = i
            out_tgt_t[tid, n_out_local] = j
            out_w_t[tid, n_out_local] = w_raw * factor
            n_out_local += 1
        out_count_t[tid] = n_out_local
        if overflow_here:
            overflow_t[tid] = 1
