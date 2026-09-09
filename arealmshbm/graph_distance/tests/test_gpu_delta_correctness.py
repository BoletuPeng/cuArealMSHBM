# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Correctness gate for the Δ-stepping GPU solver — the only GPU solver
for ``gradient_geodesic_distance`` since 2026-09 — plus the dispatcher's
precondition contracts.

The oracle is the batched pull-based Bellman-Ford that Δ-stepping
replaced, transcribed into this file (kernels, host loop and the same
normalisation arithmetic) so the bit-identity that justified the
replacement stays under test after the production copy was deleted.
Both solvers reach the same fixed point of
``d[u] = min_v fl(d[v] + w(v, u))``, so the gate is bit-identity, not a
tolerance: a tolerance would hide a dropped update. Skipped without
cupy / a CUDA device.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_CUPY = importlib.util.find_spec("cupy") is not None


def _has_device() -> bool:
    if not _HAS_CUPY:
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_GPU = _has_device()
pytestmark = pytest.mark.skipif(not _GPU, reason="cupy / CUDA device unavailable")


# ─────────────────────────────────────────────────────────────────────
# Reference solver — batched pull-based Bellman-Ford.
#
# ``D[v, s]`` is destination-outer / source-inner, i.e. already the
# ``D[dest, source]`` orientation the production entry point publishes,
# so no transpose pass belongs here. Kernels are built lazily so the
# module still imports without cupy.
# ─────────────────────────────────────────────────────────────────────
_BF_BLOCK_S = 256
_BF_MAX_ITERS = 300

_BF_PRECOMPUTE_SRC = r"""
extern "C" __global__
void precompute_edge_weights(
        const int*   __restrict__ vertex_nbors,  // (N, M) int32, 1-indexed; 0=absent
        const float* __restrict__ grad_data,     // (N,) fp32
        float*       __restrict__ edge_weights,  // (N, M) fp32 out
        int N, int M) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * M;
    if (idx >= total) return;
    const int v = idx / M;
    const int slot = idx - v * M;
    const int u1 = vertex_nbors[idx];
    if (u1 == 0) {
        // +inf sentinel: the iter kernel's `cand < best` never fires on
        // an absent slot, so the hot loop needs no validity branch.
        edge_weights[idx] = __int_as_float(0x7f800000);
        return;
    }
    const int u = u1 - 1;  // 1-idx -> 0-idx
    edge_weights[idx] = (grad_data[v] + grad_data[u]) * 0.5f;
}
"""

_BF_INIT_SRC = r"""
extern "C" __global__
void init_distance_matrix(float* __restrict__ D, int N) {
    const long long total = (long long)N * (long long)N;
    const long long idx = (long long)blockIdx.x * (long long)blockDim.x
                         + (long long)threadIdx.x;
    if (idx >= total) return;
    const int v = (int)(idx / (long long)N);
    const int s = (int)(idx - (long long)v * (long long)N);
    D[idx] = (v == s) ? 0.0f : __int_as_float(0x7f800000);  // +inf
}
"""

# Gauss-Seidel: reads and writes the same buffer. The races are benign
# for SSSP — relaxation is monotone, so a stale read is still a valid
# upper bound and only ever delays convergence.
_BF_ITER_SRC = r"""
#define BF_M %d
#define BF_BLOCK_S %d

extern "C" __global__
void bf_iter(
        float*       __restrict__ D,              // (N, N) fp32 D[v, s]; IN-PLACE
        const int*   __restrict__ vertex_nbors,   // (N, M) int32, 1-idx
        const float* __restrict__ edge_weights,   // (N, M) fp32 (g_v+g_u)/2
        int N,
        int*         __restrict__ changed_flag) {
    __shared__ int   s_nbors[BF_M];
    __shared__ float s_weights[BF_M];

    const int v = blockIdx.y;
    const int s = blockIdx.x * BF_BLOCK_S + threadIdx.x;

    if (threadIdx.x < BF_M) {
        const int slot = threadIdx.x;
        s_nbors[slot]   = vertex_nbors[v * BF_M + slot];
        s_weights[slot] = edge_weights[v * BF_M + slot];
    }
    __syncthreads();

    if (s >= N) return;

    const size_t self_idx = (size_t)v * (size_t)N + (size_t)s;
    const float prev = D[self_idx];
    float best = prev;

    #pragma unroll
    for (int slot = 0; slot < BF_M; slot++) {
        const int u1 = s_nbors[slot];
        if (u1 == 0) continue;
        const int u = u1 - 1;
        const float w = s_weights[slot];
        const float cand = D[(size_t)u * (size_t)N + (size_t)s] + w;
        if (cand < best) best = cand;
    }

    if (best < prev) {
        D[self_idx] = best;
        atomicOr(changed_flag, 1);
    }
}
"""

_BF_CACHE: dict = {}


def _bf_kernels(M: int):
    """Compile + cache the three reference kernels for valence ``M``."""
    import cupy as cp

    k = _BF_CACHE.get(M)
    if k is None:
        k = (cp.RawKernel(_BF_PRECOMPUTE_SRC, "precompute_edge_weights"),
             cp.RawKernel(_BF_INIT_SRC, "init_distance_matrix"),
             cp.RawKernel(_BF_ITER_SRC % (M, _BF_BLOCK_S), "bf_iter"))
        _BF_CACHE[M] = k
    return k


def _bf_reference(vertex_nbors_d, grad_data_d):
    """Reference solve; returns the normalized (N, N) fp32 D[dest, source]."""
    import cupy as cp

    N, M = int(vertex_nbors_d.shape[0]), int(vertex_nbors_d.shape[1])
    k_pre, k_init, k_iter = _bf_kernels(M)

    edge_weights = cp.empty((N, M), dtype=cp.float32)
    total = N * M
    k_pre(((total + _BF_BLOCK_S - 1) // _BF_BLOCK_S,), (_BF_BLOCK_S,),
          (vertex_nbors_d, grad_data_d, edge_weights, N, M))

    D = cp.empty((N, N), dtype=cp.float32)
    total = N * N
    k_init(((total + _BF_BLOCK_S - 1) // _BF_BLOCK_S,), (_BF_BLOCK_S,), (D, N))

    changed_flag = cp.zeros(1, dtype=cp.int32)
    grid = ((N + _BF_BLOCK_S - 1) // _BF_BLOCK_S, N)
    for _ in range(_BF_MAX_ITERS):
        changed_flag.fill(0)
        k_iter(grid, (_BF_BLOCK_S,),
               (D, vertex_nbors_d, edge_weights, np.int32(N), changed_flag))
        if int(changed_flag.get()[0]) == 0:
            break
    else:
        raise RuntimeError(f"reference BF did not converge in {_BF_MAX_ITERS}")

    g_max = cp.abs(D).max()
    if not cp.isfinite(g_max):
        raise RuntimeError("reference BF: disconnected mesh")
    # Same normalisation arithmetic as the production path: fp64
    # reciprocal of the max rounded once to fp32, then one fp32 multiply.
    if float(g_max) > 0.0:
        D *= cp.float32(1.0 / float(g_max))
    return D


# ``vertex_nbors`` follows the CBIG contract: (N, M) int32, 1-indexed,
# 0 = absent slot.
def _torus_grid(nx: int, ny: int, m_slots: int = 6):
    """Wrapped nx x ny lattice: valence 4, padded to ``m_slots``."""
    n = nx * ny
    vn = np.zeros((n, m_slots), dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            v = i * ny + j
            nbrs = [((i + 1) % nx) * ny + j,
                    ((i - 1) % nx) * ny + j,
                    i * ny + (j + 1) % ny,
                    i * ny + (j - 1) % ny]
            for s, u in enumerate(sorted(set(nbrs))):
                vn[v, s] = u + 1
    return vn


def _ragged_ring(n: int, m_slots: int = 6, rng=None):
    """Ring plus random chords, with 1..m_slots used slots per vertex."""
    rng = rng or np.random.default_rng(0)
    adj = [set() for _ in range(n)]
    for v in range(n):
        adj[v].add((v + 1) % n)
        adj[(v + 1) % n].add(v)
    for _ in range(n // 3):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b and len(adj[a]) < m_slots and len(adj[b]) < m_slots:
            adj[a].add(b)
            adj[b].add(a)
    vn = np.zeros((n, m_slots), dtype=np.int32)
    for v in range(n):
        for s, u in enumerate(sorted(adj[v])):
            vn[v, s] = u + 1
    return vn


def _dev(vn_d, g_d, **kw):
    from arealmshbm.graph_distance import gradient_geodesic_distance_gpu_device
    return gradient_geodesic_distance_gpu_device(vn_d, g_d, **kw)


def _run(vn, grad, **kw):
    import cupy as cp
    return _dev(cp.asarray(vn, dtype=cp.int32),
                cp.asarray(grad, dtype=cp.float32), **kw)


def _run_bf(vn, grad):
    import cupy as cp
    return _bf_reference(cp.asarray(vn, dtype=cp.int32),
                         cp.asarray(grad, dtype=cp.float32))


@pytest.mark.parametrize("builder,args,seed", [
    (_torus_grid, (7, 9), 0),
    (_ragged_ring, (257,), 2),
    (_ragged_ring, (1031,), 3),
])
def test_delta_bit_equals_bf(builder, args, seed):
    """The production solver must reproduce the reference bit-for-bit."""
    import cupy as cp
    rng = np.random.default_rng(seed)
    vn = builder(*args)
    n = vn.shape[0]
    grad = rng.random(n, dtype=np.float32) * np.float32(0.9)
    # Exactly-zero gradients produce zero-weight edges, which are present
    # in the real step-0 input.
    grad[rng.random(n) < 0.15] = np.float32(0.0)

    d_new = _run(vn, grad)
    d_old = _run_bf(vn, grad)
    assert d_new.shape == (n, n)
    assert d_new.dtype == cp.float32
    assert bool(cp.array_equal(d_new, d_old)), (
        "delta solver diverged from the reference; max|diff| = "
        f"{float(cp.abs(d_new - d_old).max()):.3e}")
    # The normalisation is one fp32 multiply by fp32(1/max), so the peak
    # lands within an ulp of 1.0 rather than exactly on it.
    assert float(cp.abs(cp.diag(d_new)).max()) == 0.0
    assert float(d_new.max()) == pytest.approx(1.0, rel=1e-6)


def test_delta_is_deterministic():
    """atomicMin / atomicOr are order-free, so repeated runs must agree
    bit-for-bit even though block scheduling varies."""
    import cupy as cp
    rng = np.random.default_rng(7)
    vn = _torus_grid(11, 13)
    grad = (rng.random(vn.shape[0], dtype=np.float32) * np.float32(0.5))
    first = _run(vn, grad)
    for _ in range(3):
        assert bool(cp.array_equal(_run(vn, grad), first))


def test_worklist_overflow_is_result_neutral():
    """Overflowing the compacted frontier worklist is a throughput event,
    not an error: the un-drained bits stay in the near mask and are picked
    up on the next round. A tiny ``cap`` forces that branch."""
    import cupy as cp
    rng = np.random.default_rng(5)
    vn = _torus_grid(16, 16)
    grad = rng.random(vn.shape[0], dtype=np.float32)
    ref = _run_bf(vn, grad)
    for cap in (32, 512):
        got = _run(vn, grad, cap=cap)
        assert bool(cp.array_equal(got, ref)), f"cap={cap} changed the result"


def test_zero_gradient_field():
    """All-zero gradients -> Δ collapses to 0, exercising the ``<=``
    promote rule that keeps the bucket loop making progress."""
    import cupy as cp
    vn = _torus_grid(6, 7)
    grad = np.zeros(vn.shape[0], dtype=np.float32)
    got = _run(vn, grad)
    assert float(cp.abs(got).max()) == 0.0
    assert bool(cp.array_equal(got, _run_bf(vn, grad)))


def test_high_valence_rejected():
    """A slot count above the packed edge table's is a hard limit, not a
    routing decision — there is no second solver."""
    from arealmshbm.graph_distance._kernels_gpu import EDGE_SLOTS
    n, m = 64, EDGE_SLOTS + 4
    vn = np.zeros((n, m), dtype=np.int32)
    for v in range(n):
        vn[v, 0] = (v + 1) % n + 1
        vn[v, 1] = (v - 1) % n + 1
    grad = np.random.default_rng(17).random(n, dtype=np.float32)
    with pytest.raises(ValueError, match="EDGE_SLOTS"):
        _run(vn, grad)


def test_oversized_mesh_rejected():
    """Past ``MAX_N`` the frontier bitmasks outgrow the kernel's static
    shared-memory budget; refused at the module build, before any
    ``(N, N)`` allocation. (The bound used to be the int16 worklist's
    32768; the widened worklist is pinned bit-identical to the oracle
    at N=33642 in ``docs/step0_flow_and_subgraphs.md``.)"""
    from arealmshbm.graph_distance._kernels_gpu import MAX_N
    n = MAX_N + 32
    idx = np.arange(n)
    vn = np.zeros((n, 6), dtype=np.int32)
    vn[:, 0] = (idx - 1) % n + 1
    vn[:, 1] = (idx + 1) % n + 1
    grad = np.full(n, 0.3, dtype=np.float32)
    with pytest.raises(ValueError, match="shared memory"):
        _run(vn, grad)


def test_disconnected_mesh_raises():
    """Two components -> +inf survives the solve -> hard error rather than
    a silently injected zero-distance shortcut."""
    a = _torus_grid(5, 5)
    b = _torus_grid(5, 5)
    n = a.shape[0]
    vn = np.zeros((2 * n, 6), dtype=np.int32)
    vn[:n] = a
    vn[n:] = np.where(b > 0, b + n, 0)
    grad = np.full(2 * n, 0.3, dtype=np.float32)
    with pytest.raises(RuntimeError, match="disconnected"):
        _run(vn, grad)


def test_negative_gradient_rejected():
    """Negative weights invert the kernel's uint32 distance ordering and
    remove the termination argument for its unbounded round loop, so the
    solver refuses them up front instead of launching."""
    vn = _torus_grid(6, 7)
    grad = np.full(vn.shape[0], 0.3, dtype=np.float32)
    grad[:3] = np.float32(-0.4)
    with pytest.raises(ValueError, match="non-negative"):
        _run(vn, grad)
    nan = np.full(vn.shape[0], 0.3, dtype=np.float32)
    nan[1] = np.float32("nan")
    with pytest.raises(ValueError, match="non-negative"):
        _run(vn, nan)


def test_numpy_entry_point_matches_device():
    """The numpy-in/numpy-out wrapper must agree with the device path."""
    import cupy as cp
    from arealmshbm.graph_distance import gradient_geodesic_distance_gpu
    rng = np.random.default_rng(23)
    vn = _torus_grid(8, 9)
    n = vn.shape[0]
    grad = rng.random(n, dtype=np.float32)
    verts = rng.random((n, 3)).astype(np.float32)
    host = gradient_geodesic_distance_gpu(verts, vn, grad)
    assert np.array_equal(host, cp.asnumpy(_run(vn, grad)))


def _asym_ring(n: int, n_chords: int, m_slots: int = 6, seed: int = 0):
    """Bidirectional ring plus ``n_chords`` one-way chords, which make the
    listed relation asymmetric — the case where a push solver and a pull
    solver disagree on the graph itself. Every other builder here is
    symmetric."""
    rng = np.random.default_rng(seed)
    adj = [[(v + 1) % n, (v - 1) % n] for v in range(n)]
    added = 0
    while added < n_chords:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a == b or b in adj[a] or len(adj[a]) >= m_slots:
            continue
        adj[a].append(b)          # one direction only
        added += 1
    vn = np.zeros((n, m_slots), dtype=np.int32)
    for v in range(n):
        for s, u in enumerate(sorted(adj[v])):
            vn[v, s] = u + 1
    return vn


def test_asymmetric_table_rejected():
    """The kernel pushes ``v -> nbors[v]``, so an asymmetric table would
    solve the transposed graph. It is refused, not routed."""
    rng = np.random.default_rng(31)
    vn = _asym_ring(400, 120)
    grad = rng.random(vn.shape[0], dtype=np.float32) * np.float32(0.9)
    with pytest.raises(ValueError, match="symmetric"):
        _run(vn, grad)


@pytest.mark.parametrize("bad", ["negative", "past_N"])
def test_out_of_range_slot_rejected(bad):
    """A slot outside the 1-indexed range ``[0, N]`` is an out-of-bounds
    device read, so it is rejected before the solve."""
    vn = _torus_grid(7, 9)
    grad = np.full(vn.shape[0], 0.2, dtype=np.float32)
    vn[3, 0] = -1 if bad == "negative" else vn.shape[0] + 5
    with pytest.raises(ValueError, match="outside the valid 1-indexed range"):
        _run(vn, grad)


def test_production_down_sphere_delta_equals_bf():
    """The shape step 0 actually solves: the icosphere ``_subgraph_B``
    builds for an fsaverage6 cohort at ``downsample=3.2`` (N=12962,
    valence 6), on a gradient field with the zero-weight edges the real
    interpolated edge density contains."""
    import cupy as cp
    from arealmshbm.icosphere import make_icosphere
    from arealmshbm.mesh_topology import compute_topology

    n_target = int(round((2 * 40962) / (2.0 * 3.2)))
    verts, faces = make_icosphere(n_target, radius=100.0)
    vn, _vf = compute_topology(verts, faces)
    n = vn.shape[0]
    assert (n, vn.shape[1]) == (12962, 6)

    rng = np.random.default_rng(101)
    grad = (rng.random(n, dtype=np.float32) * np.float32(0.35))
    grad[rng.random(n) < 0.10] = np.float32(0.0)

    vn_d = cp.asarray(vn, dtype=cp.int32)
    g_d = cp.asarray(grad)
    d_new = _dev(vn_d, g_d)
    d_old = _bf_reference(vn_d, g_d)
    assert bool(cp.array_equal(d_new, d_old))
