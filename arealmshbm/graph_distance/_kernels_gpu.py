"""_kernels_gpu.py

Per-source Δ-stepping SSSP for ``gradient_geodesic_distance`` — one CTA
per source, distance rows in L2-resident global memory, the frontier as
a pair of shared-memory bitmasks drained through a compacted int32
worklist. The GPU solver since 2026-09, when it replaced a bit-identical
batched pull-based Bellman-Ford. Run-to-run deterministic: ``min`` and
``atomicMin`` are order-free, so the fixed point of
``d[u] = min_v fl(d[v] + w(v, u))`` does not depend on the relaxation
schedule.

Preconditions are hard limits, all rejected with ``ValueError`` by
:mod:`.graph_distance_gpu`: non-negative non-NaN weights (distances
compare as uint32 bit patterns, and the round loop terminates only on
strict decreases), a symmetric neighbour table (this kernel pushes
``v -> nbors[v]``), ``M <= EDGE_SLOTS`` and ``N <= MAX_N`` (the static
shared-memory budget). Rows come out source-major, so ``transpose_scale`` is mandatory: the published
matrix is ``D[dest, source]`` and fp32 addition is not reassociative.

Every access to a distance row must keep the ``.cg`` cache operator:
``atomicMin`` executes at L2 without updating L1, so a plain load could
serve a stale row value.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import cupy as cp


# Slots reserved per vertex in the packed edge table. 8 × 8 B = 64 B is
# one L2 line, so a vertex's whole adjacency is 2 sectors / 3 LDG.128.
EDGE_SLOTS = 8

# Throughput-only knobs; none of them can move a bit of the result.
# BLOCKS_PER_SM is a deliberate under-fill of the occupancy the shared
# memory would allow: it is the largest grid whose live set of distance
# rows still fits in L2.
DEFAULT_TPB = 64
DEFAULT_CAP = 512
DEFAULT_BLOCKS_PER_SM = 16
DEFAULT_DELTA_MULT = 3.0

# Static shared memory per CTA (kernel 2): two NW-word frontier
# bitmasks, the int32 worklist and 16 B of bookkeeping. The 48 KiB
# static limit therefore bounds N; MAX_N is that bound at DEFAULT_CAP
# (188352, above fsaverage7's 163842, so no supported mesh reaches it)
# and :func:`get_delta_module` refuses the actual (N, cap) pair. The
# worklist was int16 -- N < 32768 -- while a Bellman-Ford fallback
# existed for larger meshes; widened when that fallback was removed
# (2026-09): bit-neutral, same register count, same wall.
_SMEM_STATIC_LIMIT = 48 * 1024


def _static_smem_bytes(nw: int, cap: int) -> int:
    return 8 * nw + 4 * cap + 16


MAX_N = 32 * ((_SMEM_STATIC_LIMIT - 4 * DEFAULT_CAP - 16) // 8)

_SRC = r"""
#define NW   %(NW)d
#define CAP  %(CAP)d
#define TPB  %(TPB)d
#define MSLOT %(MSLOT)d
#define SLOTS %(SLOTS)d
#define NPAIR %(NPAIR)d
#define NREAD %(NREAD)d
#define INF_BITS 0x7f800000u

// ─────────────────────────────────────────────────────────────────────
// Kernel 0 — neighbour-table topology probe.
//
// The relax step PUSHES along `v -> nbors[v]`, so an asymmetric table
// describes the transposed graph and is a hard error (see the module
// docstring); a slot outside [0, N] would be an out-of-bounds row read
// in every kernel below. One thread per vertex, MSLOT x MSLOT
// compares, reported as a flag word: bit 1 (= 2) for a slot outside
// [0, N], bit 0 (= 1) for a listed edge (v, u) with no matching
// (u, v). The range test guards this kernel's own back-edge read, so
// it runs first. At N = 12962 / M = 6 that is 466k int loads, paid
// once per call on the neighbour table only -- not once per gradient
// field.
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void probe_topology(
        const int* __restrict__ vertex_nbors,   // (N, MSLOT) int32, 1-indexed
        int N,
        int*       __restrict__ bad)            // (1,) int32 flag word, pre-zeroed
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    #pragma unroll
    for (int m = 0; m < MSLOT; m++) {
        const int u1 = vertex_nbors[v * MSLOT + m];
        if (u1 == 0) continue;
        if (u1 < 0 || u1 > N) { atomicOr(bad, 2); return; }
        const int u = u1 - 1;
        int back = 0;
        #pragma unroll
        for (int q = 0; q < MSLOT; q++)
            if (vertex_nbors[u * MSLOT + q] == v + 1) back = 1;
        if (!back) { atomicOr(bad, 1); return; }
    }
}


// ─────────────────────────────────────────────────────────────────────
// Kernel 1 — fuse vertex_nbors + grad_data into the packed edge table.
//
// tab[v * SLOTS + m] = { u, bitcast<int>((g[v] + g[u]) * 0.5f) } for the
// m-th neighbour of v, and { -1, 0 } for absent / padding slots. One
// thread per vertex: SLOTS is a compile-time 8 so the store is a clean
// 64 B block and the loop unrolls away.
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void build_edge_table(
        const int*   __restrict__ vertex_nbors,   // (N, MSLOT) int32, 1-indexed
        const float* __restrict__ grad_data,      // (N,) fp32
        int2*        __restrict__ tab,            // (N, SLOTS) {u, w-bits}
        int N)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    const float gv = grad_data[v];
    int2* out = tab + (size_t)v * SLOTS;
    #pragma unroll
    for (int m = 0; m < SLOTS; m++) {
        int2 e;
        e.x = -1;
        e.y = 0;
        if (m < MSLOT) {
            const int u1 = vertex_nbors[v * MSLOT + m];
            if (u1 != 0) {
                const int u = u1 - 1;
                e.x = u;
                e.y = __float_as_int((gv + grad_data[u]) * 0.5f);
            }
        }
        out[m] = e;
    }
}


// ─────────────────────────────────────────────────────────────────────
// Kernel 2 — per-source Δ-stepping SSSP.
//
// grid : any number of CTAs; each grid-strides over sources [0, NS).
//        Sized as blocks_per_sm × SM count so the live set of distance
//        rows stays inside L2 instead of thrashing it.
// block: TPB threads (64). Shared state per CTA:
//        mnear[NW], mfar[NW]  frontier bitmasks   (2 × 1624 B)
//        list[CAP]            compacted worklist  (2048 B)
//        cnt[2], minfar, smax bookkeeping
//        => 5312 B/CTA at the shipped NW/CAP (1624 + 1624 + 2048 + 16),
//        confirmed by the compiled kernel (38 registers, 5312 B), i.e.
//        19 CTAs/SM by shared memory (100 KB/SM) and 24 by threads.
//        The launch uses 16/SM on purpose — see DEFAULT_BLOCKS_PER_SM.
//
// gmax accumulates the global max |dist| as a raw fp32 bit pattern via
// atomicMax on uint32. Distances are non-negative so the IEEE-754 bit
// pattern orders exactly like the float, and +inf (0x7f800000) is the
// largest value — which doubles as the disconnected-mesh detector.
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void sssp_delta_rows(
        float*        __restrict__ Drows,   // (NS, N) fp32 source-major work buffer
        const int2*   __restrict__ tab,     // (N, SLOTS) packed edges
        int N, int NS, float delta,
        unsigned int* __restrict__ gmax)    // (1,) uint32, pre-zeroed
{
    __shared__ unsigned int mnear[NW];
    __shared__ unsigned int mfar[NW];
    __shared__ int          list[CAP];
    __shared__ int          cnt[2];
    __shared__ unsigned int minfar;
    __shared__ unsigned int smax;

    const int tid   = threadIdx.x;
    const int nhalf = N >> 1;

    for (int sid = blockIdx.x; sid < NS; sid += gridDim.x) {
        float* dist = Drows + (size_t)sid * (size_t)N;

        // ---- init: row = +inf, masks empty, source seeded in `mnear` ----
        // float2 stores halve the instruction count, but only when the
        // row base is 8 B aligned. Row s starts at byte 4*s*N, so that
        // holds for every s iff N is even (the fsaverage6 icosphere has
        // N = 12962). Odd N takes the scalar path.
        if ((N & 1) == 0) {
            float2* d2 = (float2*)dist;
            const float2 inf2 = make_float2(__int_as_float(INF_BITS),
                                            __int_as_float(INF_BITS));
            for (int i = tid; i < nhalf; i += TPB) __stcg(d2 + i, inf2);
        } else {
            for (int i = tid; i < N; i += TPB)
                __stcg(dist + i, __int_as_float(INF_BITS));
        }
        for (int w = tid; w < NW; w += TPB) { mnear[w] = 0u; mfar[w] = 0u; }
        __syncthreads();
        if (tid == 0) {
            __stcg(dist + sid, 0.0f);
            mnear[sid >> 5] = 1u << (sid & 31);
            cnt[0] = 0; cnt[1] = 0; smax = 0u;
        }
        __syncthreads();

        unsigned int thrb = __float_as_uint(delta);   // bucket bound, as bits
        int it = 0;
        while (true) {
            // cnt is double buffered on the round parity so the reset of
            // the *next* round's counter can be issued before the barrier
            // that publishes this round's count — 2 barriers per round
            // instead of 3.
            const int cur = it & 1;

            // ---- drain `mnear` into the compacted worklist ----
            // Purely shared-memory work: no global traffic at all.
            for (int w = tid; w < NW; w += TPB) {
                unsigned int word = mnear[w];
                if (word == 0u) continue;
                int p = atomicAdd(&cnt[cur], __popc(word));
                unsigned int kept = 0u;
                int k = 0;
                while (word) {
                    const int b = __ffs(word) - 1;
                    word &= (word - 1u);
                    // Overflow is not an error: the bit stays set and the
                    // vertex is picked up on a later round.
                    if (p + k < CAP) list[p + k] = (w << 5) + b;
                    else             kept |= (1u << b);
                    k++;
                }
                mnear[w] = kept;
            }
            if (tid == 0) cnt[cur ^ 1] = 0;
            __syncthreads();
            const int ncnt = cnt[cur];
            const int n = ncnt < CAP ? ncnt : CAP;

            if (n == 0) {
                // ---- bucket advance ----
                // pass A: min-reduce the tentative distances still in `mfar`.
                if (tid == 0) minfar = INF_BITS;
                __syncthreads();
                unsigned int lmin = INF_BITS;
                for (int w = tid; w < NW; w += TPB) {
                    unsigned int word = mfar[w];
                    while (word) {
                        const int b = __ffs(word) - 1;
                        word &= (word - 1u);
                        const unsigned int dv =
                            __float_as_uint(__ldcg(dist + ((w << 5) + b)));
                        if (dv < lmin) lmin = dv;
                    }
                }
                if (lmin != INF_BITS) atomicMin(&minfar, lmin);
                __syncthreads();
                const unsigned int mf = minfar;   // warp-uniform after the barrier
                if (mf == INF_BITS) break;        // both piles empty → converged
                thrb = __float_as_uint(__uint_as_float(mf) + delta);
                // pass B: promote everything at or below the new bound. The
                // `<=` (not `<`) guarantees the minimum itself moves, so the
                // loop always makes progress even for delta == 0.
                for (int w = tid; w < NW; w += TPB) {
                    unsigned int word = mfar[w];
                    if (word == 0u) continue;
                    unsigned int kept = 0u, mv = 0u;
                    while (word) {
                        const int b = __ffs(word) - 1;
                        word &= (word - 1u);
                        const unsigned int dv =
                            __float_as_uint(__ldcg(dist + ((w << 5) + b)));
                        if (dv <= thrb) mv   |= (1u << b);
                        else            kept |= (1u << b);
                    }
                    mfar[w] = kept;
                    if (mv) atomicOr(&mnear[w], mv);
                }
                it++;
                __syncthreads();
                continue;
            }

            // ---- relax ----
            for (int i = tid; i < n; i += TPB) {
                const int v = list[i];
                const float dv = __ldcg(dist + v);
                // NPAIR × 16 B covers all MSLOT neighbours (two {u,w} pairs
                // per int4); the whole 64 B vertex block is 2 L2 sectors.
                const int4* p = (const int4*)(tab + (size_t)v * SLOTS);
                int4 blk[NPAIR];
                #pragma unroll
                for (int q = 0; q < NPAIR; q++) blk[q] = __ldg(p + q);
                #pragma unroll
                for (int m = 0; m < NREAD; m++) {
                    const int4 e = blk[m >> 1];
                    const int   u  = (m & 1) ? e.z : e.x;
                    const float ew = __int_as_float((m & 1) ? e.w : e.y);
                    if (u < 0) continue;
                    const unsigned int ndb = __float_as_uint(dv + ew);
                    // Filter read first: ~83 percent of candidate relaxations
                    // and a plain compare is far cheaper than a failed atomic.
                    // A stale value here can only be an over-estimate, so the
                    // worst case is a redundant atomicMin.
                    if (ndb >= __float_as_uint(__ldcg(dist + u))) continue;
                    const unsigned int old = atomicMin((unsigned int*)(dist + u), ndb);
                    if (ndb < old) {
                        const unsigned int bit = 1u << (u & 31);
                        if (ndb < thrb) atomicOr(&mnear[u >> 5], bit);
                        else            atomicOr(&mfar[u >> 5], bit);
                    }
                }
            }
            it++;
            __syncthreads();
        }

        // ---- row max, folded into the global max ----
        // The row was just written so it is still L2-hot; doing it here
        // avoids a separate 672 MB cupy reduction pass over the matrix.
        unsigned int lm = 0u;
        for (int i = tid; i < N; i += TPB) {
            const unsigned int d = __float_as_uint(__ldcg(dist + i));
            if (d > lm) lm = d;
        }
        atomicMax(&smax, lm);
        __syncthreads();
        if (tid == 0) atomicMax(gmax, smax);
        __syncthreads();
        if (tid == 0) smax = 0u;
    }
}


// ─────────────────────────────────────────────────────────────────────
// Kernel 3 — 32×32 tiled transpose with the normalisation multiply
// fused into the store (saves a full 1.34 GB read+write pass).
//
// The 33-wide shared tile skews the column stride by one bank so the
// transposed read ``tile[threadIdx.x][threadIdx.y + j]`` is conflict
// free; both the load and the store are 128 B coalesced.
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void transpose_scale(
        const float* __restrict__ in,    // (N, N) source-major
        float*       __restrict__ out,   // (N, N) destination-major
        int N, float inv)
{
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < 32; j += 8)
        if (x < N && (y + j) < N)
            tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * N + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < 32; j += 8)
        if (x < N && (y + j) < N)
            out[(size_t)(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j] * inv;
}
"""


_MODULE_CACHE: dict = {}


def get_delta_module(N: int, M: int, tpb: int = DEFAULT_TPB,
                     cap: int = DEFAULT_CAP):
    """Compile (and cache) the Δ-stepping module for this graph shape.

    The shape parameters are baked in as macros, so the frontier
    bitmasks are statically sized shared arrays and the slot loops
    unroll; one module per ``(NW, M, tpb, cap)``.
    """
    if not (1 <= M <= EDGE_SLOTS):
        raise ValueError(
            f"delta-stepping kernel supports valence 1..{EDGE_SLOTS}; got M={M}")
    nw = (N + 31) // 32
    smem = _static_smem_bytes(nw, cap)
    if smem > _SMEM_STATIC_LIMIT:
        raise ValueError(
            f"delta-stepping kernel: N={N} (cap={cap}) needs {smem} B of "
            f"static shared memory per CTA, over the {_SMEM_STATIC_LIMIT} B "
            f"limit -- N <= {MAX_N} at the default cap.")
    npair = (M + 1) // 2
    key = (nw, M, tpb, cap)
    mod = _MODULE_CACHE.get(key)
    if mod is None:
        subs = dict(NW=nw, CAP=cap, TPB=tpb, MSLOT=M, SLOTS=EDGE_SLOTS,
                    NPAIR=npair, NREAD=2 * npair)
        # Token substitution rather than %-formatting: the CUDA source
        # carries literal '%' characters in comments.
        src = _SRC
        for name, val in subs.items():
            src = src.replace("%%(%s)d" % name, str(val))
        mod = cp.RawModule(code=src)
        # Force the JIT now so a compile error surfaces at build time.
        mod.get_function("sssp_delta_rows")
        _MODULE_CACHE[key] = mod
    return mod


def build_edge_table_cupy(vertex_nbors_d: cp.ndarray,
                          grad_data_d: cp.ndarray,
                          N: int, M: int,
                          mod=None) -> cp.ndarray:
    """Return the packed ``(N, EDGE_SLOTS, 2)`` int32 edge table."""
    if mod is None:
        mod = get_delta_module(N, M)
    tab = cp.empty((N, EDGE_SLOTS, 2), dtype=cp.int32)
    k = mod.get_function("build_edge_table")
    block = 128
    k(((N + block - 1) // block,), (block,),
      (vertex_nbors_d, grad_data_d, tab, cp.int32(N)))
    return tab


def probe_topology_cupy(vertex_nbors_d: cp.ndarray,
                        N: int, M: int,
                        mod=None) -> int:
    """Flag word over the 1-indexed neighbour table: bit 1 (2) set iff
    some slot lies outside ``[0, N]``, bit 0 (1) iff the listed relation
    is not symmetric; ``0`` for a table the solver accepts. Building the
    module first also applies the shared-memory bound on ``N``.
    """
    if mod is None:
        mod = get_delta_module(N, M)
    bad = cp.zeros(1, dtype=cp.int32)
    k = mod.get_function("probe_topology")
    block = 128
    k(((N + block - 1) // block,), (block,),
      (vertex_nbors_d, cp.int32(N), bad))
    return int(bad.get()[0])


def sm_count() -> int:
    """Multiprocessor count of the current device (grid sizing input)."""
    return int(cp.cuda.Device().attributes["MultiProcessorCount"])
