"""_kernels_gpu.py

GPU kernels for the radius_mask supercall. The CPU code in
:mod:`._kernels` runs a serial Dijkstra-with-binary-heap per parcel
(or per central-sulcus source), parallelised across parcels via numba
``prange``. That structure doesn't map well to a GPU — the priority
queue is inherently sequential.

Two solvers live here, because the leaf's two SSSP families have
opposite shapes.

**add_spatial_constraint** — K = L_h ≈ 150 problems, each bounded at
30 mm — runs as a **batched pull-based Bellman-Ford** on a (V, K)
distance matrix:

    init:  dist[v, k] = 0 if v is a source of problem k, else +inf
    iterate: each thread owns one (v, k) cell. Read v's CSR neighbour
             list; compute min over neighbours of dist[u, k] + w(u, v);
             clamp at the radius. Loop until no cell changes for one
             full sweep, OR a hard iter cap.

The radius confines the wave to ~30 hops, so ~30 sweeps of a 25 MB
matrix converge — 6 ms, embarrassingly parallel, no atomics.

**central_sulcus** — Nsrc ≈ 5000 problems, unbounded, run to full
settle — is the opposite: the same solver needs 209 sweeps of a 819 MB
matrix (1.37 s, DRAM-roofline-bound). It runs on ``sssp_batch32``, a
frontier delta-stepping SSSP with 32 sources per CTA; see the second
block comment below.

Kernels:
    bellman_ford_bounded_step  — one BF pull-relax pass (RawKernel),
        driving the bounded radius mask.

    init_dist_for_parcels      — seed dist[v, l] = 0 where labels[v]==l+1.

    sssp_batch32               — frontier delta-stepping SSSP, 32
        sources per CTA, gathering only the relevant-vertex rows.
        Limits: vertex valence <= 8 (packed edge table), V < 2^31
        (int32 ids), and — the binding one — **V <= 90080**, because
        the frontier bitmask is RING x ceil(V/32) words of *static*
        shared memory against a 48 KB cap. fsaverage6 and fsaverage5
        fit; fsaverage (V=163842) does not and raises a named
        ValueError from :func:`_sssp_module` (use ``backend='cpu'``).

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


# =====================================================================
# central_sulcus, second generation: batched delta-stepping SSSP
# =====================================================================
# The pull-based Bellman-Ford above solves the central-sulcus family
# (Nsrc ~ 5000 single-source problems, unbounded, run to full settle)
# by sweeping the whole (V, Nsrc) matrix once per hop of the graph
# diameter. On fsaverage6 that is 209 sweeps over 819 MB x 7 accesses
# = ~1.2 TB of DRAM traffic per hemi, and it measures at 1.37 s -- the
# kernel already runs at the card's memory roofline, so the only way
# down is to stop touching cells that cannot change.
#
# ``sssp_batch32`` does that with a frontier. One CTA owns a BATCH = 32
# **column block** of the same (V, Nsrc) matrix and drives a
# delta-stepping SSSP for those 32 sources simultaneously:
#
#   * one warp per frontier vertex, ``lane`` = source within the block.
#     A vertex's 32 distances are 128 contiguous bytes, so every load,
#     store and atomicMin in the relax loop is a single coalesced
#     transaction. This is why the batch is exactly 32 and why the
#     source axis is padded to a multiple of 32 (block alignment).
#   * the frontier is a RING of RING_N shared-memory bitmasks over the
#     V vertices -- one per delta-bucket -- shared by all 32 sources of
#     the block. Amortizing the mask scan over 32 sources is the second
#     half of the win: the sources of a block are neighbours on the
#     cortex (consecutive pre/postcentral vertex ids), so their balls
#     overlap and the union frontier is barely wider than one source's.
#
# Why a ring and not the textbook near/far pair: with ``delta`` at or
# above the largest edge weight, a relaxation out of bucket ``i``
# lands in bucket ``i``, ``i+1`` or ``i+2`` -- never further -- because
# the relaxed vertex's own distance is already >= i*delta and one edge
# adds less than delta. So RING_N = 4 slots are provably enough to hold
# every scheduled vertex, a bucket advance is an index increment, and
# the whole "min-reduce the far pile, then rescan it to promote"
# machinery disappears -- that machinery rescans the entire far pile on
# every advance, and measured an order of magnitude slower than the
# ring (see docs/step1_flow_and_subgraphs.md).
# ``termination``: a vertex is only ever scheduled inside the ring
# window, so RING_N consecutive empty buckets mean the queue is empty.
#
# Precision: bit-identical to the pull-BF. Every relaxation is
# ``fl(d[v] + w)`` followed by a min; min is associative, commutative
# and idempotent, so the fixed point of ``d[u] = min_v fl(d[v] + w)``
# does not depend on the relaxation order and any monotone schedule
# reaches it. atomicMin on the uint32 reinterpretation orders exactly
# like the float for non-negative values (the weights here are
# euclidean distances > 0; distances start at 0 or +inf = 0x7f800000,
# the largest such pattern). Bucketing only reorders relaxations, so
# the kernel is also run-to-run deterministic, and the bucket index is
# free to be approximate -- a vertex scheduled too early is relaxed
# again later, a vertex scheduled late is still inside the ring.
#
# Only the distances at ``relevant_verts`` are consumed downstream, so
# the kernel gathers those rows into a compact (n_rel, NCOL) matrix at
# the end of each block and recycles the (V, 32) work buffer.

# Slots reserved per vertex in the packed edge table. 8 x 8 B = 64 B,
# one L2 line per vertex, read as NPAIR x LDG.128. fsaverage* meshes
# have valence <= 6; a denser mesh is refused at the table build.
_EDGE_SLOTS = 8
# Sources per CTA. 32 fp32 = one 128 B coalesced transaction per warp
# access; do not change without re-deriving the alignment argument.
_SSSP_BATCH = 32
# Bucket ring depth. 4 = the provable 3 (bucket i, i+1, i+2) rounded up
# to a power of two so the modulo is a mask.
_SSSP_RING = 4
# Swept on the fsaverage6 bench. The kernel is latency-bound, not
# occupancy-bound (24.6 KB shared/CTA allows 4 CTAs/SM at every TPB
# tried), so the largest block wins; CAP is flat across 512..4096 --
# the worklist only has to keep the warps fed; and the DELTA optimum
# is a broad floor around x10 (narrow buckets pay in rounds, wide ones
# in re-relaxations). blocks_per_sm only caps the grid, which Nsrc/32
# is already below.
_SSSP_TPB = 1024
_SSSP_CAP = 1024
_SSSP_BLOCKS_PER_SM = 4
# Bucket width in units of the mean edge weight. With delta >= the
# largest edge weight the RING window is exact (see above); a smaller
# delta stays correct -- the schedule clamps to the ring and relaxes
# early -- but pays extra re-relaxations.
_SSSP_DELTA_MULT = 10.0


_SSSP_SRC = r"""
#define NW    %(NW)d
#define CAP   %(CAP)d
#define TPB   %(TPB)d
#define SLOTS %(SLOTS)d
#define RING  %(RING)d
#define BATCH 32
#define NPAIR (SLOTS / 2)
#define INF_BITS 0x7f800000u

// One CTA per 32-source column block; grid-strided over the blocks.
//
//   tab   (V, SLOTS) int2   {neighbour id, fp32 weight bits}, id < 0 = pad
//   src   (NB * 32,) int32  source vertex of each column (the padded
//                           tail repeats a real source; the caller
//                           slices it back off)
//   dist  (gridDim.x, V, 32) fp32  per-CTA work buffer
//   out   (n_rel, NCOL) fp32       gathered distances at relevant verts
//   rel   (n_rel,) int32           relevant vertex ids
extern "C" __global__ void sssp_batch32(
        const int2*  __restrict__ tab,
        const int*   __restrict__ src,
        float*       __restrict__ dist,
        float*       __restrict__ out,
        const int*   __restrict__ rel,
        int V, int NB, int n_rel, int NCOL, float inv_delta)
{
    __shared__ unsigned int mask[RING][NW];
    __shared__ int          list[CAP];
    __shared__ int          cnt[2];

    const int tid   = threadIdx.x;
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int nwarp = TPB >> 5;
    float* const base = dist + (size_t)blockIdx.x * (size_t)V * BATCH;
    const long long cells = (long long)V * BATCH;

    for (int b = blockIdx.x; b < NB; b += gridDim.x) {
        // ---- init: work buffer = +inf, ring empty, 32 sources seeded ----
        for (long long i = tid; i < cells; i += TPB)
            __stcg(base + i, __int_as_float(INF_BITS));
        for (int w = tid; w < RING * NW; w += TPB) (&mask[0][0])[w] = 0u;
        if (tid == 0) { cnt[0] = 0; cnt[1] = 0; }
        __syncthreads();
        if (tid < BATCH) {
            const int s = src[b * BATCH + tid];
            __stcg(base + (size_t)s * BATCH + tid, 0.0f);
            atomicOr(&mask[0][s >> 5], 1u << (s & 31));
        }
        __syncthreads();

        int bi = 0;        // current bucket index
        int empties = 0;   // consecutive empty buckets drained
        int it = 0;
        while (true) {
            // cnt is double buffered on the round parity so this round's
            // count is published and the next round's counter reset
            // behind a single barrier.
            const int cur = it & 1;
            unsigned int* const mcur = mask[bi & (RING - 1)];

            // ---- drain the current bucket into the compacted worklist ----
            for (int w = tid; w < NW; w += TPB) {
                unsigned int word = mcur[w];
                if (word == 0u) continue;
                const int p = atomicAdd(&cnt[cur], __popc(word));
                unsigned int kept = 0u;
                int k = 0;
                while (word) {
                    const int bpos = __ffs(word) - 1;
                    word &= (word - 1u);
                    // Overflow is not an error: the bit stays set and the
                    // vertex is picked up on a later round.
                    if (p + k < CAP) list[p + k] = (w << 5) + bpos;
                    else             kept |= (1u << bpos);
                    k++;
                }
                mcur[w] = kept;
            }
            if (tid == 0) cnt[cur ^ 1] = 0;
            __syncthreads();
            const int ncnt = cnt[cur];
            const int n = ncnt < CAP ? ncnt : CAP;
            it++;

            if (n == 0) {
                // Nothing scheduled in this bucket: step the ring. A
                // relaxation can only schedule into buckets bi..bi+2, so
                // RING empty buckets in a row means the queue is empty.
                if (++empties == RING) break;
                bi++;
                __syncthreads();
                continue;
            }
            empties = 0;

            // ---- relax: one warp per frontier vertex, lane = source ----
            for (int i = warp; i < n; i += nwarp) {
                const int v = list[i];
                const float dv = __ldcg(base + (size_t)v * BATCH + lane);
                const int4* p = (const int4*)(tab + (size_t)v * SLOTS);
                int4 blk[NPAIR];
                #pragma unroll
                for (int q = 0; q < NPAIR; q++) blk[q] = __ldg(p + q);
                #pragma unroll
                for (int m = 0; m < SLOTS; m++) {
                    const int4 e = blk[m >> 1];
                    const int   u  = (m & 1) ? e.z : e.x;
                    const float ew = __int_as_float((m & 1) ? e.w : e.y);
                    if (u < 0) continue;          // warp-uniform: pad slot
                    const float nd = dv + ew;
                    const unsigned int ndb = __float_as_uint(nd);
                    unsigned int improved = 0u;
                    // Filter read first: a plain compare is far cheaper
                    // than a failed atomic, and a stale value can only be
                    // an over-estimate (worst case a redundant atomic).
                    if (ndb < __float_as_uint(
                                __ldcg(base + (size_t)u * BATCH + lane))) {
                        const unsigned int old = atomicMin(
                            (unsigned int*)(base + (size_t)u * BATCH + lane),
                            ndb);
                        if (ndb < old) improved = 1u;
                    }
                    if (__ballot_sync(0xffffffffu, improved)) {
                        // Schedule at the earliest bucket any lane wants;
                        // a lane that wanted a later one is simply relaxed
                        // early and re-scheduled if it improves again.
                        unsigned int mymin = improved ? ndb : INF_BITS;
                        #pragma unroll
                        for (int off = 16; off; off >>= 1) {
                            const unsigned int o =
                                __shfl_xor_sync(0xffffffffu, mymin, off);
                            mymin = o < mymin ? o : mymin;
                        }
                        if (lane == 0) {
                            int slot = (int)(__uint_as_float(mymin) * inv_delta)
                                       - bi;
                            if (slot < 0) slot = 0;
                            if (slot > RING - 1) slot = RING - 1;
                            atomicOr(&mask[(bi + slot) & (RING - 1)][u >> 5],
                                     1u << (u & 31));
                        }
                    }
                }
            }
            __syncthreads();
        }

        // ---- gather the rows the caller actually consumes ----
        for (int i = warp; i < n_rel; i += nwarp)
            out[(size_t)i * NCOL + (size_t)b * BATCH + lane] =
                __ldcg(base + (size_t)rel[i] * BATCH + lane);
        __syncthreads();
    }
}
"""


_sssp_module_cache: dict = {}


# Static shared memory available to one CTA. ptxas rejects anything
# above this ("uses too much shared data"), so the frontier bitmask —
# RING x ceil(V/32) words — puts a hard ceiling on V.
_SMEM_LIMIT = 48 * 1024


def _sssp_smem_bytes(V: int, cap: int) -> int:
    """Static shared bytes one ``sssp_batch32`` CTA needs at this shape:
    the RING x NW frontier bitmask, the CAP worklist, and cnt[2]."""
    return _SSSP_RING * ((V + 31) // 32) * 4 + cap * 4 + 2 * 4


def _sssp_module(V: int, cap: int, tpb: int):
    """Compile (once per shape) the batched delta-stepping SSSP module.

    ``NW`` (the bitmask word count) sizes a ``__shared__`` array, so it
    must be a compile-time constant; the module is therefore keyed on
    ``V`` as well as on the two tuning constants.

    That bitmask is O(V) **static shared memory**, which caps the mesh
    this solver can take at V <= 90080 (48 KB / RING / 4 B x 32):
    fsaverage6 (40962 -> 24.6 KB) and fsaverage5 (10242 -> 6.3 KB) fit,
    fsaverage (163842 -> 82 KB) does not and is refused here rather
    than at ptxas. Use ``backend='cpu'`` for such a mesh.
    """
    need = _sssp_smem_bytes(V, cap)
    if need > _SMEM_LIMIT:
        v_max = (_SMEM_LIMIT - cap * 4 - 8) // (_SSSP_RING * 4) * 32
        raise ValueError(
            f"sssp_batch32: mesh too large for the shared-memory frontier "
            f"(V={V} needs {need} B of static shared memory per CTA, limit "
            f"{_SMEM_LIMIT} B; the ceiling at cap={cap} is V<={v_max}). "
            f"Use backend='cpu' for this mesh.")
    key = (V, cap, tpb, _EDGE_SLOTS, _SSSP_RING)
    mod = _sssp_module_cache.get(key)
    if mod is None:
        nw = (V + 31) // 32
        src = _SSSP_SRC % {"NW": nw, "CAP": cap, "TPB": tpb,
                           "SLOTS": _EDGE_SLOTS, "RING": _SSSP_RING}
        mod = cp.RawModule(code=src)
        mod.compile()
        _sssp_module_cache[key] = mod
    return mod


def build_edge_table(indptr, indices, weights) -> cp.ndarray:
    """Pack a CSR mesh adjacency into the ``(V, _EDGE_SLOTS, 2)`` int32
    edge table the SSSP kernel reads: ``[neighbour id, fp32 weight bits]``
    per slot, ``id = -1`` in the padding slots.

    Built on the host (a few ms of numpy) and uploaded once per hemi.
    """
    import numpy as np

    if indptr.ndim != 1 or indices.ndim != 1 or weights.ndim != 1:
        raise ValueError("build_edge_table: indptr/indices/weights must be 1-D")
    if indices.shape[0] != weights.shape[0]:
        raise ValueError(
            f"build_edge_table: indices ({indices.shape[0]}) and weights "
            f"({weights.shape[0]}) must have equal length")
    if weights.dtype != np.float32:
        raise ValueError(
            f"build_edge_table: weights must be float32; got {weights.dtype}")
    V = int(indptr.shape[0]) - 1
    counts = np.diff(indptr)
    if V > 0 and int(counts.max()) > _EDGE_SLOTS:
        raise ValueError(
            f"build_edge_table: max vertex valence {int(counts.max())} exceeds "
            f"the packed table's {_EDGE_SLOTS} slots")
    if V >= (1 << 31) - 1:
        raise ValueError(f"build_edge_table: V={V} does not fit int32")

    tab = np.zeros((V, _EDGE_SLOTS, 2), dtype=np.int32)
    tab[:, :, 0] = -1
    rows = np.repeat(np.arange(V, dtype=np.int64), counts)
    slots = (np.arange(indices.shape[0], dtype=np.int64)
             - np.repeat(indptr[:-1], counts))
    tab[rows, slots, 0] = indices.astype(np.int32)
    tab[rows, slots, 1] = weights.view(np.int32)
    return cp.asarray(np.ascontiguousarray(tab))


def central_sulcus_distances_cupy(
    tab_dev: cp.ndarray,
    src_verts,
    rel_verts,
    V: int,
    delta: float,
    blocks_per_sm: int = _SSSP_BLOCKS_PER_SM,
    tpb: int = _SSSP_TPB,
    cap: int = _SSSP_CAP,
) -> cp.ndarray:
    """Geodesic distance from every source vertex to every relevant
    vertex, as an ``(n_rel, NCOL)`` fp32 device array.

    ``NCOL = ceil(len(src_verts) / 32) * 32`` -- the column axis is the
    source axis padded up to a whole number of 32-source blocks. The
    padded tail columns repeat ``src_verts[-1]``; the caller slices
    ``[:, :len(src_verts)]`` (or the pre / post halves) back off.

    ``delta`` is the bucket width in mm -- a pure scheduling knob (see
    the block comment above: the fixed point does not depend on it).
    At ``delta >= max edge weight`` the RING window is exact; below
    that the schedule clamps into the ring, which only costs extra
    re-relaxations. Callers pass ``_SSSP_DELTA_MULT * mean_weight``
    (~5.6x the max edge weight on fsaverage6).
    """
    import numpy as np

    if tab_dev.shape != (V, _EDGE_SLOTS, 2) or tab_dev.dtype != cp.int32:
        raise ValueError(
            f"central_sulcus_distances_cupy: tab must be ({V}, {_EDGE_SLOTS}, "
            f"2) int32; got {tab_dev.shape} {tab_dev.dtype}")
    if not (delta > 0.0) or not np.isfinite(delta):
        raise ValueError(
            f"central_sulcus_distances_cupy: delta must be finite and "
            f"positive; got {delta!r}")
    src_verts = np.ascontiguousarray(src_verts, dtype=np.int32)
    rel_verts = np.ascontiguousarray(rel_verts, dtype=np.int32)
    n_src = int(src_verts.shape[0])
    n_rel = int(rel_verts.shape[0])
    if n_src == 0 or n_rel == 0:
        raise ValueError(
            "central_sulcus_distances_cupy: empty source or relevant set "
            f"(n_src={n_src}, n_rel={n_rel})")
    # Vertex ids index straight into the (V, 32) work buffer and into
    # the shared frontier bitmask; an out-of-range id would be a silent
    # out-of-bounds write, not an error. Two int32 min/max on <=5k-element
    # host arrays -- microseconds.
    for name, ids in (("src_verts", src_verts), ("rel_verts", rel_verts)):
        lo = int(ids.min())
        hi = int(ids.max())
        if lo < 0 or hi >= V:
            raise ValueError(
                f"central_sulcus_distances_cupy: {name} must hold vertex "
                f"ids in [0, {V}); got range [{lo}, {hi}]")
    nb = (n_src + _SSSP_BATCH - 1) // _SSSP_BATCH
    ncol = nb * _SSSP_BATCH
    src_pad = np.empty(ncol, dtype=np.int32)
    src_pad[:n_src] = src_verts
    src_pad[n_src:] = src_verts[-1]

    n_sm = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    grid = min(nb, max(1, blocks_per_sm * n_sm))
    dist = cp.empty((grid, V, _SSSP_BATCH), dtype=cp.float32)
    out = cp.empty((n_rel, ncol), dtype=cp.float32)

    ker = _sssp_module(V, cap, tpb).get_function("sssp_batch32")
    ker((grid,), (tpb,),
        (tab_dev, cp.asarray(src_pad), dist, out, cp.asarray(rel_verts),
         cp.int32(V), cp.int32(nb), cp.int32(n_rel), cp.int32(ncol),
         np.float32(1.0 / float(delta))))
    del dist
    return out
