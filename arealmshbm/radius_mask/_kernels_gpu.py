"""_kernels_gpu.py

GPU kernels for the radius_mask supercall. The CPU code in
:mod:`._kernels` runs a serial Dijkstra-with-binary-heap per parcel
(or per central-sulcus source), parallelised across parcels via numba
``prange``. That structure doesn't map well to a GPU — the priority
queue is inherently sequential.

The GPU port restates the same problem as **batched pull-based
Bellman-Ford** on a (V, K) distance matrix, where K is the number of
parallel SSSP problems (parcels for add_spatial_constraint, sources
for central_sulcus):

    init:  dist[v, k] = 0 if v is a source of problem k, else +inf
    iterate: each thread owns one (v, k) cell. Read v's CSR neighbour
             list; compute min over neighbours of dist[u, k] + w(u, v);
             clamp at the radius (when applicable). Loop until no cell
             changes for one full sweep, OR a hard iter cap.

For fsaverage6 + radius=30 mm the graph diameter within the relaxation
front is ~30 hops, so ~30 BF iterations converge. Each iteration is a
single elementwise kernel over (V × K) cells, each touching
``avg_deg ≈ 6`` neighbour reads — embarrassingly parallel, no atomics.

Kernels:
    bellman_ford_bounded_step  — one BF pull-relax pass (RawKernel).
        Used by add_spatial_constraint with ``has_bound=1`` (radius
        clamp) and by central_sulcus with ``has_bound=0``.

    init_dist_for_parcels      — seed dist[v, l] = 0 where labels[v]==l+1.
    init_dist_for_sources      — seed dist[v, k] = 0 at given source verts.

Equivalent to the CPU's "every parcel vert is in mask + Dijkstra from
boundary outward" formulation: setting dist=0 at every parcel vert is
bit-equivalent because the only edges that escape the parcel are at
boundary verts (interior parcel verts have all-parcel neighbours, so
their dist=0 doesn't propagate beyond the parcel before they're
already settled).

cupy is imported at module top — only loaded via the lazy import in
:mod:`.radius_mask_gpu`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import cupy as cp


_BF_BLOCK = 256


_bf_step_kernel = cp.RawKernel(r"""
extern "C" __global__
void bellman_ford_bounded_step(
        const long long* __restrict__ indptr,   // (V+1,) int64
        const long long* __restrict__ indices,  // (E,)   int64
        const float*     __restrict__ weights,  // (E,)   fp32
        const float*     __restrict__ dist_in,  // (V, K) fp32
        float*           __restrict__ dist_out, // (V, K) fp32
        int*             __restrict__ changed_flag,
        int V, int K, float radius, int has_bound) {
    // One thread = one (v, k) cell. (V, K) C-contig so stride is 1 in K;
    // adjacent threads in a warp share v, walking k = 0, 1, ... -- they
    // hit the same CSR slot of v, so the indptr/indices reads coalesce.
    //
    // ``long long`` (not ``long``) on the int64 host arrays: CUDA on
    // Windows uses LLP64 where ``long`` is 32-bit; ``long long`` is
    // always 64-bit. A ``long*`` over an int64 host array would read
    // every other 4 bytes, silently scrambling CSR.

    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)V * (long long)K;
    if (idx >= total) return;
    const int v = (int)(idx / (long long)K);
    const int k = (int)(idx % (long long)K);

    float best = dist_in[(size_t)v * (size_t)K + (size_t)k];
    long long e_start = indptr[v];
    long long e_end   = indptr[v + 1];
    for (long long e = e_start; e < e_end; e++) {
        int u = (int)indices[e];
        float w = weights[e];
        float cand = dist_in[(size_t)u * (size_t)K + (size_t)k] + w;
        // Bounded relax: drop candidates that overshoot radius. Without
        // the bound (central_sulcus) the clamp is disabled.
        if (has_bound && cand > radius) continue;
        if (cand < best) best = cand;
    }
    float prev = dist_in[(size_t)v * (size_t)K + (size_t)k];
    if (best < prev) {
        atomicOr(changed_flag, 1);
    }
    dist_out[(size_t)v * (size_t)K + (size_t)k] = best;
}
""", "bellman_ford_bounded_step")


_init_parcels_kernel = cp.RawKernel(r"""
extern "C" __global__
void init_dist_for_parcels(const long long* __restrict__ labels,
                            float* __restrict__ dist,
                            int V, int L_h) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)V * (long long)L_h;
    if (idx >= total) return;
    const int v = (int)(idx / (long long)L_h);
    const int l = (int)(idx % (long long)L_h);
    long long lbl = labels[v];
    // labels are 1-indexed; l is 0-indexed. Match: labels[v] == l + 1.
    if (lbl == (long long)(l + 1)) {
        dist[idx] = 0.0f;
    } else {
        dist[idx] = __int_as_float(0x7f800000);  // +inf
    }
}
""", "init_dist_for_parcels")


_init_sources_kernel = cp.RawKernel(r"""
extern "C" __global__
void init_dist_for_sources(float* __restrict__ dist, int V, int Nsrc) {
    // Fill (V, Nsrc) with +inf. Source seeding (dist[src[k], k] = 0)
    // happens in a separate 1-thread-per-source kernel below.
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)V * (long long)Nsrc;
    if (idx < total) {
        dist[idx] = __int_as_float(0x7f800000);
    }
}
""", "init_dist_for_sources")


_seed_sources_kernel = cp.RawKernel(r"""
extern "C" __global__
void seed_sources_at_diagonal(const long long* __restrict__ src_verts,
                               float* __restrict__ dist,
                               int Nsrc, int K) {
    // dist[src_verts[k], k] = 0 for k in [0, Nsrc). K = stride along v-axis.
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Nsrc) return;
    long long v = src_verts[k];
    dist[(size_t)v * (size_t)K + (size_t)k] = 0.0f;
}
""", "seed_sources_at_diagonal")


def init_dist_for_parcels_cupy(labels_dev: cp.ndarray,
                                dist_dev: cp.ndarray) -> None:
    """Fill ``dist_dev`` ((V, L_h) fp32) with 0 at parcel verts and +inf
    elsewhere. labels are 1-indexed (0 = medial wall, not seeded).
    """
    V, L_h = dist_dev.shape
    total = V * L_h
    grid = (total + _BF_BLOCK - 1) // _BF_BLOCK
    _init_parcels_kernel((grid,), (_BF_BLOCK,),
                          (labels_dev, dist_dev, cp.int32(V), cp.int32(L_h)))


def init_dist_for_sources_cupy(src_verts_dev: cp.ndarray,
                                dist_dev: cp.ndarray) -> None:
    """Fill ``dist_dev`` ((V, Nsrc) fp32) with +inf, then set
    ``dist[src_verts[k], k] = 0`` for each source k.
    """
    V, Nsrc = dist_dev.shape
    total = V * Nsrc
    grid = (total + _BF_BLOCK - 1) // _BF_BLOCK
    _init_sources_kernel((grid,), (_BF_BLOCK,),
                          (dist_dev, cp.int32(V), cp.int32(Nsrc)))
    # _seed_sources_kernel takes (..., Nsrc, K) where K is the (V, K)
    # column stride. Each source k seeds dist[src_verts[k], k] = 0;
    # the diagonal-seeded SSSP family this enables (one source per
    # column) implies K == Nsrc here. The signature keeps K explicit
    # so the kernel is reusable if a future caller pads dist with
    # extra columns.
    grid_seed = (Nsrc + _BF_BLOCK - 1) // _BF_BLOCK
    _seed_sources_kernel((grid_seed,), (_BF_BLOCK,),
                          (src_verts_dev, dist_dev,
                           cp.int32(Nsrc), cp.int32(Nsrc)))


def bellman_ford_bounded_cupy(
    indptr_dev: cp.ndarray,
    indices_dev: cp.ndarray,
    weights_dev: cp.ndarray,
    dist_dev: cp.ndarray,
    scratch_dev: cp.ndarray,
    radius: float,
    max_iters: int = 500,
) -> int:
    """Pull-based Bellman-Ford until convergence on ``dist_dev`` ((V, K)).

    Uses ping-pong between ``dist_dev`` and ``scratch_dev`` (caller owns
    both, same shape + dtype). On return, ``dist_dev`` holds the final
    distances. Bounded variant: candidate ``d + w > radius`` is dropped,
    which both confines the wave to a ball and accelerates convergence.

    Set ``radius`` to ``+inf`` (or any large value) for an unbounded
    SSSP — useful for central_sulcus where we run to full settle.

    Returns the number of iterations executed (caller can sanity-check
    against ``max_iters``).
    """
    V, K = dist_dev.shape
    if scratch_dev.shape != dist_dev.shape or scratch_dev.dtype != dist_dev.dtype:
        raise ValueError("bellman_ford: scratch must match dist shape + dtype")
    total = V * K
    grid = (total + _BF_BLOCK - 1) // _BF_BLOCK
    has_bound = 1 if cp.isfinite(cp.float32(radius)).item() else 0
    radius_f = cp.float32(radius if has_bound else 1.0)

    changed_flag = cp.zeros(1, dtype=cp.int32)
    cur = dist_dev
    nxt = scratch_dev
    for it in range(max_iters):
        changed_flag[:] = 0
        _bf_step_kernel((grid,), (_BF_BLOCK,),
                         (indptr_dev, indices_dev, weights_dev,
                          cur, nxt, changed_flag,
                          cp.int32(V), cp.int32(K),
                          radius_f, cp.int32(has_bound)))
        cur, nxt = nxt, cur
        if int(changed_flag.item()) == 0:
            # Make sure the final result lives in dist_dev (caller's buffer).
            if cur is not dist_dev:
                dist_dev[...] = cur
            return it + 1
    # Hit max_iters with changed_flag != 0 — relaxation did not converge.
    # On fsaverage6 + radius=30 mm this should converge in ~30 iters; an
    # unbounded SSSP on the same mesh in ~100. Hitting the 500-iter cap
    # means either a CSR-build bug or a much larger mesh than supported.
    # Loud failure beats silently truncated distances downstream.
    raise RuntimeError(
        f"bellman_ford_bounded_cupy did not converge in {max_iters} "
        f"iterations (V={V}, K={K}, has_bound={has_bound}, "
        f"radius={float(radius)!r}). Distances may be truncated; the "
        "downstream mask / mean-distance will be wrong. Investigate the "
        "mesh CSR or raise max_iters."
    )
