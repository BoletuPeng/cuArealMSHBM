"""test_dijkstra_kernels.py — the two CPU Dijkstra kernels of radius_mask.

``add_spatial_constraint_kernel`` and ``central_sulcus_kernel`` run on
the indexed heap of ``graph_distance._heap`` (decrease-key). The oracle
below is the lazy-deletion formulation (a push per relaxation, stale
pops skipped), single-threaded. Settled Dijkstra distances do not depend
on the heap, so ``out_mask`` and ``avg_dis`` must agree bit for bit:

  * on an icosphere with Voronoi parcels, and on seeded random graphs
    (continuous fp32 weights, and integer weights for heavy key ties);
  * with a radius that covers most of the graph;
  * with one relevant parcel, so central_sulcus breaks out early with a
    full heap and the thread's next source must start from a clean one;
  * at the default thread count and at one thread. Every sum of these
    fp32 distances is exact in fp64, so the thread partition of the
    ``avg_dis`` accumulation cannot move a bit.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numba
import numpy as np
import pytest
from numba import njit

from arealmshbm.radius_mask._common import _build_mesh_csr
from arealmshbm.radius_mask._kernels import (
    add_spatial_constraint_kernel, build_parcel_csr_kernel,
    central_sulcus_kernel)
from arealmshbm.radius_mask.tests._mesh_fixture import _nbors_from_faces


# ─────────────────────────────────────────────────────────────────────
# Oracle: lazy-deletion binary heap + single-thread Dijkstra loops
# ─────────────────────────────────────────────────────────────────────

@njit(inline="always")
def _lazy_push(keys, vals, n, key, val):
    keys[n] = key
    vals[n] = val
    i = n
    while i > 0:
        p = (i - 1) >> 1
        if keys[p] > keys[i]:
            tk = keys[p]; keys[p] = keys[i]; keys[i] = tk
            tv = vals[p]; vals[p] = vals[i]; vals[i] = tv
            i = p
        else:
            break
    return n + 1


@njit(inline="always")
def _lazy_pop(keys, vals, n):
    rk = keys[0]
    rv = vals[0]
    n -= 1
    keys[0] = keys[n]
    vals[0] = vals[n]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        s = i
        if l < n and keys[l] < keys[s]:
            s = l
        if r < n and keys[r] < keys[s]:
            s = r
        if s == i:
            break
        tk = keys[i]; keys[i] = keys[s]; keys[s] = tk
        tv = vals[i]; vals[i] = vals[s]; vals[s] = tv
        i = s
    return rk, rv, n


@njit
def _add_spatial_oracle(indptr, indices, weights, labels, parcel_offs,
                        parcel_inds, L, radius, out_mask):
    V = labels.shape[0]
    E = indptr[V]
    radius_f = np.float32(radius)
    dist = np.full(V, np.float32(1e30), dtype=np.float32)
    hk = np.empty(E, dtype=np.float32)
    hv = np.empty(E, dtype=np.int32)
    for l_idx in range(L):
        l = l_idx + 1
        n_heap = 0
        for k_p in range(parcel_offs[l_idx], parcel_offs[l_idx + 1]):
            v = parcel_inds[k_p]
            out_mask[v, l_idx] = 1
            is_b = False
            for k_e in range(indptr[v], indptr[v + 1]):
                if labels[indices[k_e]] != l:
                    is_b = True
                    break
            if is_b:
                dist[v] = np.float32(0.0)
                n_heap = _lazy_push(hk, hv, n_heap,
                                    np.float32(0.0), np.int32(v))
        while n_heap > 0:
            d, v_i32, n_heap = _lazy_pop(hk, hv, n_heap)
            v = v_i32
            if d > dist[v]:
                continue
            out_mask[v, l_idx] = 1
            for k_e in range(indptr[v], indptr[v + 1]):
                u = indices[k_e]
                nd = d + weights[k_e]
                if nd <= radius_f and nd < dist[u]:
                    dist[u] = nd
                    n_heap = _lazy_push(hk, hv, n_heap, nd, np.int32(u))
        dist[:] = np.float32(1e30)


@njit
def _central_sulcus_oracle(indptr, indices, weights, pre_verts, post_verts,
                           parcel_offs, parcel_inds, L, relevant_parcels,
                           relevant_verts, out_avg):
    V = indptr.shape[0] - 1
    E = indptr[V]
    Npre = pre_verts.shape[0]
    Npost = post_verts.shape[0]
    n_relevant_total = 0
    for v in range(V):
        if relevant_verts[v] == 1:
            n_relevant_total += 1
    dist = np.full(V, np.float32(1e30), dtype=np.float32)
    hk = np.empty(E, dtype=np.float32)
    hv = np.empty(E, dtype=np.int32)
    accum_pre = np.zeros(V, dtype=np.float64)
    accum_post = np.zeros(V, dtype=np.float64)
    for s_idx in range(Npre + Npost):
        if s_idx < Npre:
            src = pre_verts[s_idx]
            is_pre = True
        else:
            src = post_verts[s_idx - Npre]
            is_pre = False
        dist[src] = np.float32(0.0)
        n_heap = _lazy_push(hk, hv, 0, np.float32(0.0), np.int32(src))
        n_relevant_settled = 0
        while n_heap > 0:
            d, v_i32, n_heap = _lazy_pop(hk, hv, n_heap)
            v = v_i32
            if d > dist[v]:
                continue
            if relevant_verts[v] == 1:
                n_relevant_settled += 1
                if is_pre:
                    accum_pre[v] += np.float64(d)
                else:
                    accum_post[v] += np.float64(d)
                if n_relevant_settled >= n_relevant_total:
                    break
            for k_e in range(indptr[v], indptr[v + 1]):
                u = indices[k_e]
                nd = d + weights[k_e]
                if nd < dist[u]:
                    dist[u] = nd
                    n_heap = _lazy_push(hk, hv, n_heap, nd, np.int32(u))
        dist[:] = np.float32(1e30)
    for l_idx in range(L):
        if relevant_parcels[l_idx] == 0:
            out_avg[0, l_idx] = 0.0
            out_avg[1, l_idx] = 0.0
            continue
        ps = parcel_offs[l_idx]
        pe = parcel_offs[l_idx + 1]
        n_v = pe - ps
        if n_v == 0:
            out_avg[0, l_idx] = np.nan
            out_avg[1, l_idx] = np.nan
            continue
        sp_l = 0.0
        spo_l = 0.0
        for k in range(ps, pe):
            sp_l += accum_pre[parcel_inds[k]]
            spo_l += accum_post[parcel_inds[k]]
        out_avg[0, l_idx] = sp_l / (np.float64(n_v) * np.float64(Npre))
        out_avg[1, l_idx] = spo_l / (np.float64(n_v) * np.float64(Npost))


# ─────────────────────────────────────────────────────────────────────
# Graphs
# ─────────────────────────────────────────────────────────────────────

def _icosphere_graph():
    from arealmshbm.icosphere import make_icosphere
    verts, faces = make_icosphere(400, radius=100.0)
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    V = verts.shape[0]
    indptr, indices, w = _build_mesh_csr(verts, faces,
                                         _nbors_from_faces(faces, V))
    rng = np.random.default_rng(7)
    L = 24
    centres = verts[rng.choice(V, L, replace=False)]
    d2 = ((verts[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    labels = (np.argmin(d2, axis=1) + 1).astype(np.int64)
    labels[rng.choice(V, 5, replace=False)] = 0         # medial wall
    return indptr, indices, w, labels, L, rng


def _random_graph(seed, V, deg, integer_weights):
    rng = np.random.default_rng(seed)
    src = np.concatenate([rng.integers(0, V, V * deg // 2), np.arange(V)])
    dst = np.concatenate([rng.integers(0, V, V * deg // 2),
                          (np.arange(V) + 1) % V])        # ring: connected
    keep = src != dst
    a = np.minimum(src[keep], dst[keep])
    b = np.maximum(src[keep], dst[keep])
    key = np.unique(a.astype(np.int64) * V + b)
    a, b = key // V, key % V
    if integer_weights:
        w = rng.integers(1, 4, a.size).astype(np.float32)
    else:
        w = rng.uniform(0.1, 5.0, a.size).astype(np.float32)
    r = np.concatenate([a, b])
    c = np.concatenate([b, a])
    ww = np.concatenate([w, w])
    o = np.lexsort((c, r))
    r, c, ww = r[o], c[o], ww[o]
    indptr = np.zeros(V + 1, np.int64)
    np.add.at(indptr, r + 1, 1)
    indptr = np.cumsum(indptr)
    L = 37
    labels = rng.integers(0, L + 1, V).astype(np.int64)
    labels[rng.choice(V, L, replace=False)] = np.arange(1, L + 1)
    return (indptr, c.astype(np.int64), np.ascontiguousarray(ww, np.float32),
            labels, L, rng)


GRAPHS = {
    "icosphere": _icosphere_graph,
    "random_cont": lambda: _random_graph(1, 5000, 6, False),
    "random_int_ties": lambda: _random_graph(2, 20000, 8, True),
}


@pytest.fixture(scope="module", params=sorted(GRAPHS))
def graph(request):
    return GRAPHS[request.param]()


@pytest.fixture(params=["default", "one"])
def threads(request):
    n0 = numba.get_num_threads()
    numba.set_num_threads(n0 if request.param == "default" else 1)
    yield
    numba.set_num_threads(n0)


def _radius_covering_most(indptr, weights):
    V = indptr.shape[0] - 1
    return float(np.float32(weights.mean(dtype=np.float64) * np.sqrt(V)))


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wide", [False, True])
def test_add_spatial_constraint_matches_lazy_heap(graph, threads, wide):
    indptr, indices, weights, labels, L, _ = graph
    offs, inds = build_parcel_csr_kernel(labels, L)
    V = labels.shape[0]
    radius = (_radius_covering_most(indptr, weights) if wide
              else 3.0 * float(weights.mean(dtype=np.float64)))
    got = np.zeros((V, L), dtype=np.uint8)
    add_spatial_constraint_kernel(indptr, indices, weights, labels, offs,
                                  inds, L, np.float32(radius), got,
                                  numba.get_num_threads())
    want = np.zeros((V, L), dtype=np.uint8)
    _add_spatial_oracle(indptr, indices, weights, labels, offs, inds, L,
                        np.float32(radius), want)
    assert np.array_equal(got, want)
    if wide:
        assert want.mean() > 0.5


@pytest.mark.parametrize("n_relevant", [1, "many"])
def test_central_sulcus_matches_lazy_heap(graph, threads, n_relevant):
    indptr, indices, weights, labels, L, rng0 = graph
    rng = np.random.default_rng(int(rng0.integers(1 << 30)))
    V = labels.shape[0]
    offs, inds = build_parcel_csr_kernel(labels, L)
    pre = rng.choice(V, 60, replace=False).astype(np.int64)
    post = rng.choice(V, 45, replace=False).astype(np.int64)
    rp = np.zeros(L, dtype=np.uint8)
    if n_relevant == 1:
        sizes = np.diff(offs)
        live = np.flatnonzero(sizes > 0)
        rp[live[np.argmin(sizes[live])]] = 1              # the smallest parcel
    else:
        rp[rng.random(L) < 0.4] = 1
    rv = np.zeros(V, dtype=np.uint8)
    lab = labels > 0
    rv[lab] = rp[labels[lab] - 1]

    got = np.empty((2, L), dtype=np.float64)
    central_sulcus_kernel(indptr, indices, weights, pre, post, offs, inds,
                          L, rp, rv, got, numba.get_num_threads())
    want = np.empty((2, L), dtype=np.float64)
    _central_sulcus_oracle(indptr, indices, weights, pre, post, offs, inds,
                           L, rp, rv, want)
    assert np.array_equal(got, want, equal_nan=True)
    assert np.count_nonzero(want) >= 2 * int(rp.sum())
