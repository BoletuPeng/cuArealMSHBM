"""_kernels_gpu.py

GPU port of the watershed flooding kernel — CuPy RawKernels.

Replaces the numba @njit prange-over-K implementation in
:mod:`watershed_repaired` with a block-per-column GPU layout that
exploits the K-column independence at much wider parallelism than
the CPU's 24-core prange:

  * **Block-per-column** — one CUDA block handles one watershed column.
    K ≈ 125 columns per call (one per ``randinds_verts`` seed in
    block_a). 60 SMs × 4 blocks/SM resident ⇒ ~240 concurrent blocks,
    so one outer-iter_a's worth of columns fits in ~1 wave.

  * **(K, N) layout, host-transposed** — the kernel reads
    ``em[k, v]`` and ``label_k[neighbor_of_v]``. With (K, N) layout
    threads in a warp iterate over adjacent vertices for the same
    column, so the 32 fp32 loads collapse to 1 HBM transaction per
    warp instead of 1 transaction per thread (which is what the
    raw (N, K) layout would give — stride K between thread loads).

  * **Single-kernel threshold sweep** — the n_h=50 threshold loop
    lives inside the kernel; 1 launch per call instead of 50. Block-
    local ``__syncthreads`` is sufficient because one block ↔ one
    column ↔ no cross-block dependency.

  * **One-Jacobi-pass per threshold** — the CPU path visits within-
    step actives in shuffled order, letting labels propagate
    transitively across many ring-1 hops in one pass; on GPU we
    instead double-buffer through ``proposal`` and run ONE Jacobi
    iter per threshold step. Multi-hop chains naturally propagate
    over consecutive thresholds (the water keeps rising), so by the
    end of the n_h=50 sweep all reachable labels have spread. The
    shuffle-order semantic is replaced with a deterministic 1-pass
    Jacobi semantic. Empirically on sub-001 the two agree on the
    (labels==0) watershed-boundary mask to within ~0.5 pp of step3
    final vertex agreement (98.69% CPU → 98.22% GPU, both well above
    the 97.5% spec bar). The downstream consumer is
    ``edge_density = sum(labels == 0, axis=1)``, which reads ONLY
    the boundary mask — not label IDs (so backend-arbitrary label
    numbering does not affect the metric).

  * **Block size 1024** — measured optimum at K=125, N=75k on
    RTX 5070. With 1024 threads/block each thread handles N/1024=75
    vertices per pass; 1 block per SM exploits the SM's full thread
    budget vs 4 blocks of 256 ping-ponging context. Sweep:
    256→40 ms, 512→22 ms, 768→16 ms, 1024→14.6 ms per call.

  * **Encoded proposal** — single int32 buffer per column:
        prop == 0  → no proposal this iter
        prop  > 0  → propose label = prop
        prop == -1 → propose watershed
    Halves the workspace memory vs separate label-proposal + ws-
    proposal buffers.

Memory (K=125, N=75000):
    em        : K * N * 4 =  37 MB (read-only)
    minima    : K * N * 1 =  9.3 MB (read-only)
    neighbors : N * R * 4 = 1.7 MB (read-only, R=6)
    hiter     : n_h * 4   = negligible
    label     : K * N * 4 =  37 MB (rw)
    ws        : K * N * 1 =  9.3 MB (rw)
    proposal  : K * N * 4 =  37 MB (workspace)
    ─────────────────────
    total                  ≈ 132 MB

Determinism: the seed kernel assigns minima labels via atomic-counter
order, which is GPU-arbitrary. But the downstream consumer reads
``(labels == 0).sum(axis=1)`` which is invariant under label-value
permutation: the watershed-boundary GEOMETRY depends on which
catchments overlap, not on which integer was used to mark each
catchment. The Jacobi propagation likewise produces a deterministic
boundary mask given a (modulo-permutation) seed — multi-iter Jacobi
to convergence does not depend on within-iter thread schedule order
since Phase-1 reads happen entirely before Phase-2 writes via
``__syncthreads``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import cupy as cp


# ─────────────────────────────────────────────────────────────────────
# Seed kernel — for each column k, assign sequential labels (starting
# at 1) to vertices marked as local minima.
#
# The atomic counter ``s_rank`` is incremented in GPU-arbitrary order
# across threads. The resulting label assignment differs from CPU's
# RNG-shuffle assignment, but BOTH paths assign distinct positive ints
# to distinct minima and zeros elsewhere — the downstream pipeline
# reads only (label == 0), which is permutation-invariant.
#
# Block (256, 1), grid (K, 1). No shared mem beyond the rank counter.
# ─────────────────────────────────────────────────────────────────────
_seed_kernel_src = r"""
extern "C" __global__
void watershed_seed(
        const unsigned char* __restrict__ minima,  // (K, N) row-major
        int* __restrict__ label,                   // (K, N) row-major, init 0
        int N) {
    const int k = blockIdx.x;
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const unsigned char* min_k = minima + (size_t)k * (size_t)N;
    int* label_k = label + (size_t)k * (size_t)N;

    __shared__ int s_rank;
    if (tid == 0) s_rank = 0;
    __syncthreads();

    for (int v = tid; v < N; v += B) {
        if (min_k[v]) {
            int rk = atomicAdd(&s_rank, 1);
            label_k[v] = rk + 1;   // 1-indexed catchment id
        }
    }
}
"""


_seed_kernel = cp.RawKernel(_seed_kernel_src, "watershed_seed")
_SEED_BLOCK = 256


# ─────────────────────────────────────────────────────────────────────
# Main kernel — for each column k, sweep the n_h thresholds and
# Jacobi-propagate labels to vertices that just became active.
#
# Algorithm: one Jacobi pass per threshold step (propose + commit).
# Multi-hop label propagation chains spread across consecutive
# thresholds — the water rising naturally drives transitive
# propagation without an inner convergence loop. We deliberately
# DROPPED the convergence loop because:
#
#   * Adding it made the kernel ~3x slower on real-data sub-002
#     (340 ms vs 100 ms watershed wall on the cohort).
#   * Empirically it changed the step3 final-vertex agreement by
#     <0.1 pp on sub-001 (98.22% with 1 pass vs 98.20% with up to
#     32 passes per threshold — both well above the 97.5% bar).
#   * The downstream consumer only reads (labels == 0) — the
#     boundary geometry is determined by the mesh + edge_metrics,
#     not by within-threshold race-equivalent label propagation
#     fast-paths.
#
# Block (1024, 1), grid (K, 1). 1 block per SM gives the SM the
# full 1024-thread budget for one column (vs 4 blocks of 256
# ping-ponging context). No shared mem beyond barriers.
# ─────────────────────────────────────────────────────────────────────
_main_kernel_src = r"""
extern "C" __global__
void watershed_main(
        const float* __restrict__ em,           // (K, N) row-major
        const float* __restrict__ hiter,        // (n_h,)
        int n_h,
        const int* __restrict__ neighbors_rn,   // (R, N) int32 row-major — TRANSPOSED!
        int R,
        int N,
        int* __restrict__ label,                // (K, N) inout
        unsigned char* __restrict__ ws,         // (K, N) inout
        int* __restrict__ proposal) {           // (K, N) workspace
    const int k = blockIdx.x;
    const int tid = threadIdx.x;
    const int B = blockDim.x;

    const float* em_k = em + (size_t)k * (size_t)N;
    int* label_k = label + (size_t)k * (size_t)N;
    unsigned char* ws_k = ws + (size_t)k * (size_t)N;
    int* prop_k = proposal + (size_t)k * (size_t)N;

    // Neighbor table is (R, N) row-major — adjacent threads reading
    // neighbors_rn[r][v_thread] = neighbors_rn[r * N + v_thread] hit
    // stride-1, fully coalesced per warp transaction. Compared to the
    // prior (N, R) layout (24-byte stride between threads, scattered)
    // this halves HBM transactions on the 6-neighbor lookup hot path.
    for (int ih = 0; ih < n_h; ih++) {
        const float h = hiter[ih];

        // Phase 1: each thread proposes for its slice of vertices.
        for (int v = tid; v < N; v += B) {
            if (label_k[v] != 0 || ws_k[v]) continue;
            if (em_k[v] >= h) continue;

            int lo = 0, hi = 0;
            bool seen = false;
            #pragma unroll
            for (int r = 0; r < 6; r++) {     // R hardcoded to 6 for unroll
                if (r >= R) break;
                const int nb = neighbors_rn[(size_t)r * (size_t)N
                                             + (size_t)v];
                if (nb < 0) continue;
                const int lab = label_k[nb];
                if (lab == 0) continue;
                if (!seen) {
                    lo = lab; hi = lab; seen = true;
                } else {
                    if (lab < lo) lo = lab;
                    if (lab > hi) hi = lab;
                }
            }
            if (!seen) continue;
            if (lo == hi) {
                prop_k[v] = lo;
            } else {
                prop_k[v] = -1;
            }
        }
        __syncthreads();

        // Phase 2: each thread commits proposals for its slice.
        for (int v = tid; v < N; v += B) {
            const int p = prop_k[v];
            if (p > 0) {
                label_k[v] = p;
                prop_k[v] = 0;
            } else if (p < 0) {
                ws_k[v] = 1;
                prop_k[v] = 0;
            }
        }
        __syncthreads();
    }
}
"""


_main_kernel = cp.RawKernel(_main_kernel_src, "watershed_main")
# Block size 1024 — measured 3× faster than 256 on RTX 5070 at K=125,
# N=75k. The win comes from filling the SM with one big block instead
# of 4 small ones: each SM caps at ~1024 threads regardless, but with
# 1 block of 1024 we get more L1/shared coordination per column than
# with 4 blocks of 256 ping-ponging context. 256/512/768/1024 sweep
# on real-shape synthetic data: 40 / 22 / 16 / 14.6 ms per call.
_MAIN_BLOCK = 1024


def run_watershed_gpu(
        em_kn_d: cp.ndarray,
        minima_kn_d: cp.ndarray,
        neighbors_int_d: cp.ndarray,
        hiter_d: cp.ndarray) -> cp.ndarray:
    """Run the seed + threshold-sweep kernels on already-transposed
    device buffers.

    Parameters
    ----------
    em_kn_d : (K, N) fp32 cupy.ndarray
        Edge-metrics transposed to column-major-equivalent for warp
        coalescing across vertices.
    minima_kn_d : (K, N) uint8 cupy.ndarray
        Local-minima mask transposed; same layout as em.
    neighbors_int_d : (N, R) int32 cupy.ndarray
        Ring-1 neighbors, 0-indexed, ``-1`` for absent slots.
    hiter_d : (n_h,) fp32 cupy.ndarray
        Threshold values.

    Returns
    -------
    labels_kn_d : (K, N) int32 cupy.ndarray
        Catchment ids (0 = watershed boundary, >0 = catchment).
        Caller transposes back to (N, K).
    """
    assert em_kn_d.dtype == cp.float32 and em_kn_d.ndim == 2
    assert minima_kn_d.dtype == cp.uint8 and minima_kn_d.shape == em_kn_d.shape
    assert neighbors_int_d.dtype == cp.int32 and neighbors_int_d.ndim == 2
    assert hiter_d.dtype == cp.float32 and hiter_d.ndim == 1
    if not em_kn_d.flags.c_contiguous:
        em_kn_d = cp.ascontiguousarray(em_kn_d)
    if not minima_kn_d.flags.c_contiguous:
        minima_kn_d = cp.ascontiguousarray(minima_kn_d)

    K, N = int(em_kn_d.shape[0]), int(em_kn_d.shape[1])
    n_h = int(hiter_d.size)

    # Transpose neighbors to (R, N) row-major for coalesced reads —
    # see kernel docstring. The transpose is ~1.7 MB (6 × 75k × 4)
    # and runs once per call.
    if neighbors_int_d.shape[0] != int(em_kn_d.shape[1]):
        raise ValueError(
            f"neighbors_int_d.shape[0]={neighbors_int_d.shape[0]} "
            f"must equal N={N}")
    R = int(neighbors_int_d.shape[1])
    neighbors_rn_d = cp.ascontiguousarray(neighbors_int_d.T)

    label_d = cp.zeros((K, N), dtype=cp.int32)
    ws_d = cp.zeros((K, N), dtype=cp.uint8)
    proposal_d = cp.zeros((K, N), dtype=cp.int32)

    # Seed pass — assigns 1-indexed catchment ids to minima.
    _seed_kernel((K,), (_SEED_BLOCK,), (minima_kn_d, label_d, N))

    # Main threshold sweep — block-per-column, all n_h thresholds
    # handled inside one kernel launch.
    _main_kernel((K,), (_MAIN_BLOCK,),
                 (em_kn_d, hiter_d, n_h,
                  neighbors_rn_d, R, N,
                  label_d, ws_d, proposal_d))

    return label_d
