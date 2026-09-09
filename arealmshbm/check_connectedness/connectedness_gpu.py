"""connectedness_gpu.py

CuPy / CUDA-C port of the step-3 ``check_connectedness`` block
(``docs/step3_sparse_design.md`` §2.6).

Reproduces the CPU chain

    compute_components_general(...)  -> parcel_components (L,) fp64, NaN if empty
    component_distance(...)          -> eucli              (L,) fp32

plus the decision logic of
``arealmshbm.vmf_clustering.vmf_clustering._check_connectedness_step``
(distributed mask, ``xyz_gamma += 1000``, ``max_connectedness`` /
``max_components``).

Design
------
* **No floating-point ``atomicAdd``.** The only float atomics are
  ``atomicMin`` / ``atomicMax`` done through int reinterpretation of
  non-negative floats -- exact and order-independent, so the backend is
  run-to-run bit-reproducible. Integer ``atomicAdd`` (component counts,
  bucket cursors) is exact; bucket *order* varies but every consumer is a
  min/max over the bucket, which is order-independent.
* **Connected components** use hook-to-min + pointer jumping
  (Soman/Kishore/Narayanan). Component ids are *root vertex indices*, so
  they differ from the CPU's encounter-order ids -- only the partition is
  contractual, and both downstream consumers (distinct-count per parcel,
  component membership for the distance) depend on the partition only.
  The loop runs entirely on device inside one cooperative-groups kernel
  (``grid.sync()``), so ``step()`` needs no host round-trip for the
  convergence flag. A non-cooperative fallback (host-driven loop) is
  compiled too and used when cooperative launch is unavailable; that
  path costs extra D2H copies per call.
* ``--fmad=false`` so ``dx*dx + dy*dy + dz*dz`` rounds like numba's
  non-contracted fp32 (the CPU kernel is ``fastmath=False``).
  ``--use_fast_math`` is never used; ``sqrtf`` stays IEEE
  correctly-rounded.
* Everything is pre-allocated in ``__init__``; the per-call methods
  allocate nothing and issue exactly one small D2H copy (``step`` only).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import threading

import numpy as np

import cupy as cp


_THREADS = 256
_TILE = 256

# ---------------------------------------------------------------------------
# Module A -- connected components (compiled twice: cooperative + fallback)
# ---------------------------------------------------------------------------
_CC_COOP_SRC = r'''
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void k_cc_coop(const int* __restrict__ labels,
                                     const int* __restrict__ nbors,
                                     int Ntot, int M1,
                                     int* comp, int* changed)
{
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    volatile int* chg = (volatile int*)changed;

    for (int v = tid; v < Ntot; v += stride) comp[v] = v;
    grid.sync();

    for (int it = 0; it < 4096; ++it) {
        if (tid == 0) *chg = 0;
        grid.sync();

        int loc = 0;
        for (int v = tid; v < Ntot; v += stride) {
            const int lv = labels[v];
            const int pv = comp[v];
            for (int k = 0; k < M1; ++k) {
                const int u = nbors[k * Ntot + v];
                if (u < 0) continue;
                if (labels[u] != lv) continue;
                const int pu = comp[u];
                if (pu == pv) continue;
                const int hi = pv > pu ? pv : pu;
                const int lo = pv < pu ? pv : pu;
                atomicMin(&comp[hi], lo);
                loc = 1;
            }
        }
        if (loc) *chg = 1;
        grid.sync();
        const int ch = *chg;
        grid.sync();
        if (!ch) break;

        for (int v = tid; v < Ntot; v += stride) {
            int c = comp[v];
            while (comp[c] != c) c = comp[c];
            comp[v] = c;
        }
        grid.sync();
    }
}
'''

_CC_PLAIN_SRC = r'''
extern "C" __global__ void k_cc_init(int Ntot, int* comp, int* changed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int v = tid; v < Ntot; v += stride) comp[v] = v;
    if (tid == 0) changed[0] = 0;
}

extern "C" __global__ void k_cc_hook(const int* __restrict__ labels,
                                     const int* __restrict__ nbors,
                                     int Ntot, int M1,
                                     int* comp, int* changed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    int loc = 0;
    for (int v = tid; v < Ntot; v += stride) {
        const int lv = labels[v];
        const int pv = comp[v];
        for (int k = 0; k < M1; ++k) {
            const int u = nbors[k * Ntot + v];
            if (u < 0) continue;
            if (labels[u] != lv) continue;
            const int pu = comp[u];
            if (pu == pv) continue;
            const int hi = pv > pu ? pv : pu;
            const int lo = pv < pu ? pv : pu;
            atomicMin(&comp[hi], lo);
            loc = 1;
        }
    }
    if (loc) changed[0] = 1;
}

extern "C" __global__ void k_cc_compress(int Ntot, int* comp)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int v = tid; v < Ntot; v += stride) {
        int c = comp[v];
        while (comp[c] != c) c = comp[c];
        comp[v] = c;
    }
}
'''

# ---------------------------------------------------------------------------
# Module B -- counts, boundary, bucketing, distance, decision
# ---------------------------------------------------------------------------
_MAIN_SRC = r'''
#define INF32 1e30f
#define TILE 256

__device__ __forceinline__ void atomicMinF(float* addr, float val) {
    // Non-negative floats order identically to their int bit patterns.
    atomicMin((int*)addr, __float_as_int(val));
}
__device__ __forceinline__ void atomicMaxF(float* addr, float val) {
    atomicMax((int*)addr, __float_as_int(val));
}

extern "C" __global__ void k_reset(int Ntot, int L, float* cmin,
                                   int* comp_count, float* eucli, int* bnd_cnt)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int v = tid; v < Ntot; v += stride) cmin[v] = INF32;
    for (int p = tid; p < L; p += stride) {
        comp_count[p] = 0;
        eucli[p] = 0.0f;
        bnd_cnt[p] = 0;
    }
}

/* One atomicAdd per component root -> number of distinct components per parcel. */
extern "C" __global__ void k_count(const int* __restrict__ labels,
                                   const int* __restrict__ comp,
                                   int Ntot, int L, int* comp_count)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int v = tid; v < Ntot; v += stride) {
        if (comp[v] != v) continue;
        const int l = labels[v];
        if (l >= 1 && l <= L) atomicAdd(&comp_count[l - 1], 1);
    }
}

extern "C" __global__ void k_pc(const int* __restrict__ comp_count, int L,
                                double* pc, unsigned char* is_single)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int p = tid; p < L; p += stride) {
        const int c = comp_count[p];
        pc[p] = (c > 0) ? (double)c : __hiloint2double(0x7ff80000, 0x00000000);
        is_single[p] = (c == 1) ? 1 : 0;
    }
}

/* Two-vertex-thick boundary with the sentinel substitution of
   fused_hemi_distance (fake = n_lh*2 + 1 applied at lookup time), fused with
   the zero-singles / RH re-index step. Emits the GLOBAL parcel id of each
   boundary vertex (LH 1..n_lh, RH n_lh+1..L) so LH/RH buckets never mix. */
extern "C" __global__ void k_boundary(const int* __restrict__ labels,
                                      const int* __restrict__ nbors,
                                      const unsigned char* __restrict__ is_single,
                                      int Ntot, int M1, int n_hemi, int n_lh,
                                      int* bg, int* bnd_cnt)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int fake = n_lh * 2 + 1;
    for (int v = tid; v < Ntot; v += stride) {
        const int lv = labels[v];
        int lzv = 0;
        if (lv != 0 && !is_single[lv - 1]) lzv = (v < n_hemi) ? lv : (lv - n_lh);
        const int own_tmp = (lzv != 0) ? lzv : fake;
        int diff = 0;
        for (int k = 0; k < M1; ++k) {
            const int u = nbors[k * Ntot + v];
            if (u < 0) continue;
            const int lu = labels[u];
            int lzu = 0;
            if (lu != 0 && !is_single[lu - 1]) lzu = (u < n_hemi) ? lu : (lu - n_lh);
            const int nl_tmp = (lzu != 0) ? lzu : fake;
            diff += nl_tmp - own_tmp;
        }
        const int g = (diff != 0 && lzv != 0) ? lv : 0;
        bg[v] = g;
        if (g > 0) atomicAdd(&bnd_cnt[g - 1], 1);
    }
}

/* Exclusive scan over L buckets (L ~ 300); serial in one thread. */
extern "C" __global__ void k_scan(const int* __restrict__ bnd_cnt, int L,
                                  int* start, int* cursor)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int s = 0;
    for (int p = 0; p < L; ++p) {
        start[p] = s;
        cursor[p] = s;
        s += bnd_cnt[p];
    }
    start[L] = s;
    cursor[L] = s;
}

extern "C" __global__ void k_scatter(const int* __restrict__ bg,
                                     const int* __restrict__ comp,
                                     const float* __restrict__ xyz,
                                     int Ntot, int* cursor,
                                     float* bx, float* by, float* bz,
                                     int* bcomp, int* bpar)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int v = tid; v < Ntot; v += stride) {
        const int g = bg[v];
        if (g <= 0) continue;
        const int i = atomicAdd(&cursor[g - 1], 1);
        bx[i] = xyz[3 * v + 0];
        by[i] = xyz[3 * v + 1];
        bz[i] = xyz[3 * v + 2];
        bcomp[i] = comp[v];
        bpar[i] = g;
    }
}

/* comp_min_sq[c] = min over pairs (i in c, j in another component of the same
   parcel) of the fp32 squared distance. One block per parcel, tiled in smem. */
extern "C" __global__ void k_pairs(const int* __restrict__ start,
                                   const float* __restrict__ bx,
                                   const float* __restrict__ by,
                                   const float* __restrict__ bz,
                                   const int* __restrict__ bcomp,
                                   int L, float* cmin)
{
    const int g = blockIdx.x;
    if (g >= L) return;
    const int s = start[g];
    const int size = start[g + 1] - s;
    if (size < 2) return;

    __shared__ float sx[TILE];
    __shared__ float sy[TILE];
    __shared__ float sz[TILE];
    __shared__ int   sc[TILE];

    for (int ib = 0; ib < size; ib += TILE) {
        const int i = ib + threadIdx.x;
        const bool act = (i < size);
        float xi = 0.f, yi = 0.f, zi = 0.f;
        int ci = -1;
        if (act) {
            xi = bx[s + i]; yi = by[s + i]; zi = bz[s + i]; ci = bcomp[s + i];
        }
        float m = INF32;
        for (int jb = 0; jb < size; jb += TILE) {
            __syncthreads();
            const int j = jb + threadIdx.x;
            if (j < size) {
                sx[threadIdx.x] = bx[s + j];
                sy[threadIdx.x] = by[s + j];
                sz[threadIdx.x] = bz[s + j];
                sc[threadIdx.x] = bcomp[s + j];
            }
            __syncthreads();
            const int cnt = (size - jb) < TILE ? (size - jb) : TILE;
            if (act) {
                for (int t = 0; t < cnt; ++t) {
                    if (sc[t] == ci) continue;
                    const float dx = xi - sx[t];
                    const float dy = yi - sy[t];
                    const float dz = zi - sz[t];
                    const float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < m) m = d2;
                }
            }
        }
        if (act && m < INF32) atomicMinF(&cmin[ci], m);
        __syncthreads();
    }
}

/* eucli[p] = max over the parcel's components c of sqrt(comp_min_sq[c]).
   Every component of the parcel owns at least one boundary vertex, so the
   max over boundary vertices covers exactly the CPU's max over `uniq`. */
extern "C" __global__ void k_reduce(const int* __restrict__ start, int L,
                                    const int* __restrict__ bcomp,
                                    const int* __restrict__ bpar,
                                    const float* __restrict__ cmin,
                                    float* eucli)
{
    const int total = start[L];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < total; i += stride) {
        const float v = cmin[bcomp[i]];
        if (v < INF32) atomicMaxF(&eucli[bpar[i] - 1], sqrtf(v));
    }
}

/* distrib mask + xyz_gamma update + the two host-visible scalars. One block. */
extern "C" __global__ void k_step(const float* __restrict__ eucli,
                                  const double* __restrict__ pc,
                                  double* xyz_gamma, int L,
                                  double connect_th, double comp_th,
                                  double* out2)
{
    __shared__ double se[256];
    __shared__ double sc[256];
    __shared__ int    sa[256];
    const int t = threadIdx.x;
    const double NEG_INF = -1.0 / 0.0;
    double me = NEG_INF;
    double mc = NEG_INF;
    int any = 0;
    for (int p = t; p < L; p += blockDim.x) {
        const double e = (double)eucli[p];
        const double c = pc[p];
        const bool d = (e > connect_th) || (c > comp_th);
        if (d) {
            any = 1;
            if (e > me) me = e;
            if (isfinite(c) && c > mc) mc = c;
            xyz_gamma[p] += 1000.0;
        }
    }
    se[t] = me; sc[t] = mc; sa[t] = any;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (t < off) {
            if (se[t + off] > se[t]) se[t] = se[t + off];
            if (sc[t + off] > sc[t]) sc[t] = sc[t + off];
            sa[t] |= sa[t + off];
        }
        __syncthreads();
    }
    if (t == 0) {
        if (sa[0]) { out2[0] = se[0]; out2[1] = sc[0]; }
        else       { out2[0] = 0.0;   out2[1] = comp_th; }
    }
}
'''


_MODULES_LOCK = threading.Lock()
_MODULES = None


def _compiled_modules():
    """``(mod_main, mod_plain, mod_coop, k_cc_coop)`` — compiled once per process.

    The step-3 stage pipeline builds one :class:`ConnectednessGPU` per
    subject on the LOAD thread while EM workers run on other streams, so
    the NVRTC compiles and the cooperative-launch probe must not repeat
    per instance. ``k_cc_coop`` is ``None`` when the probe launch fails.
    Not keyed by device — this is a single-GPU deployment.
    """
    global _MODULES
    with _MODULES_LOCK:
        if _MODULES is not None:
            return _MODULES
        opts = ('--fmad=false',)
        mod = cp.RawModule(code=_MAIN_SRC, options=opts, backend='nvrtc')
        mod_plain = cp.RawModule(code=_CC_PLAIN_SRC, options=opts,
                                 backend='nvrtc')
        # Cooperative CC keeps the whole convergence loop on device (no D2H).
        mod_coop = None
        k_coop = None
        try:
            mod_coop = cp.RawModule(code=_CC_COOP_SRC, options=opts,
                                    backend='nvrtc',
                                    enable_cooperative_groups=True)
            k = mod_coop.get_function('k_cc_coop')
            probe_lab = cp.zeros(8, dtype=cp.int32)
            probe_nb = cp.full(8, -1, dtype=cp.int32)
            probe_comp = cp.empty(8, dtype=cp.int32)
            probe_changed = cp.zeros(1, dtype=cp.int32)
            k((1,), (32,), (probe_lab, probe_nb, np.int32(8), np.int32(1),
                            probe_comp, probe_changed))
            # Sync only the probe's own stream: a device-wide sync here
            # would also drain every concurrent EM worker stream.
            cp.cuda.get_current_stream().synchronize()
            k_coop = k
        except Exception:                                   # pragma: no cover
            mod_coop = None
            k_coop = None
        _MODULES = (mod, mod_plain, mod_coop, k_coop)
        return _MODULES


def _pack_nbors(lh_nbors: np.ndarray, rh_nbors: np.ndarray) -> np.ndarray:
    """(M1, 2n) int32, 0-based GLOBAL neighbor index, -1 = absent."""
    m_lh, n = lh_nbors.shape
    m_rh, n_rh = rh_nbors.shape
    if n != n_rh:
        raise ValueError(f"LH/RH must share vertex count (got {n} vs {n_rh})")
    M1 = max(m_lh, m_rh)
    out = np.full((M1, 2 * n), -1, dtype=np.int32)
    lh = np.asarray(lh_nbors, dtype=np.int64)
    rh = np.asarray(rh_nbors, dtype=np.int64)
    out[:m_lh, :n] = np.where(lh > 0, lh - 1, -1).astype(np.int32)
    out[:m_rh, n:] = np.where(rh > 0, rh - 1 + n, -1).astype(np.int32)
    return np.ascontiguousarray(out)


def _xyz_rows(vertices: np.ndarray, n: int) -> np.ndarray:
    """(n, 3) fp32 -- matches the CPU's ``float32(vertices.T)``."""
    v = np.asarray(vertices)
    if v.shape[0] == 3 and v.shape[1] == n:
        return np.ascontiguousarray(v.T, dtype=np.float32)
    if v.shape == (n, 3):
        return np.ascontiguousarray(v, dtype=np.float32)
    raise ValueError(f"vertices must be (3, {n}) or ({n}, 3); got {v.shape}")


class ConnectednessGPU:
    """Device-resident ``check_connectedness`` for the step-3 EM loop.

    Parameters
    ----------
    lh_vertex_nbors, rh_vertex_nbors : (max_neigh, n) int, MATLAB 1-indexed,
        0 = absent -- exactly as stored in the mesh dicts.
    lh_vertices, rh_vertices : (3, n) (or (n, 3)) float -- sphere coords.
    num_parcel : int (``L``; LH owns 1..L/2, RH owns L/2+1..L).
    connect_th : float -- ``eucli > connect_th`` marks a distributed parcel.
    components_threshold : int -- ``parcel_components > this`` likewise.

    All device buffers are allocated here; ``components_and_distance`` and
    ``step`` allocate nothing.
    """

    def __init__(self, lh_vertex_nbors, rh_vertex_nbors,
                 lh_vertices, rh_vertices,
                 num_parcel: int, connect_th: float,
                 components_threshold: int):
        n = int(lh_vertex_nbors.shape[1])
        self.n_hemi = n
        self.Ntot = 2 * n
        self.L = int(num_parcel)
        self.n_lh = self.L // 2
        self.connect_th = float(connect_th)
        self.components_threshold = float(components_threshold)

        nb = _pack_nbors(lh_vertex_nbors, rh_vertex_nbors)
        self.M1 = int(nb.shape[0])
        self._nbors = cp.asarray(nb)
        xyz = np.concatenate([_xyz_rows(lh_vertices, n),
                              _xyz_rows(rh_vertices, n)], axis=0)
        self._xyz = cp.asarray(np.ascontiguousarray(xyz, dtype=np.float32))

        Nt, L = self.Ntot, self.L
        self._comp = cp.empty(Nt, dtype=cp.int32)
        self._changed = cp.zeros(1, dtype=cp.int32)
        self._comp_count = cp.zeros(L, dtype=cp.int32)
        self._pc = cp.zeros(L, dtype=cp.float64)
        self._is_single = cp.zeros(L, dtype=cp.uint8)
        self._bg = cp.zeros(Nt, dtype=cp.int32)
        self._bnd_cnt = cp.zeros(L, dtype=cp.int32)
        self._start = cp.zeros(L + 1, dtype=cp.int32)
        self._cursor = cp.zeros(L + 1, dtype=cp.int32)
        self._bx = cp.empty(Nt, dtype=cp.float32)
        self._by = cp.empty(Nt, dtype=cp.float32)
        self._bz = cp.empty(Nt, dtype=cp.float32)
        self._bcomp = cp.empty(Nt, dtype=cp.int32)
        self._bpar = cp.empty(Nt, dtype=cp.int32)
        self._cmin = cp.empty(Nt, dtype=cp.float32)
        self._eucli = cp.zeros(L, dtype=cp.float32)
        self._out2 = cp.zeros(2, dtype=cp.float64)

        self._mod, self._mod_plain, _, self._k_cc_coop = _compiled_modules()
        self._k_reset = self._mod.get_function('k_reset')
        self._k_count = self._mod.get_function('k_count')
        self._k_pc = self._mod.get_function('k_pc')
        self._k_boundary = self._mod.get_function('k_boundary')
        self._k_scan = self._mod.get_function('k_scan')
        self._k_scatter = self._mod.get_function('k_scatter')
        self._k_pairs = self._mod.get_function('k_pairs')
        self._k_reduce = self._mod.get_function('k_reduce')
        self._k_step = self._mod.get_function('k_step')

        self._k_cc_init = self._mod_plain.get_function('k_cc_init')
        self._k_cc_hook = self._mod_plain.get_function('k_cc_hook')
        self._k_cc_compress = self._mod_plain.get_function('k_cc_compress')

        self._grid = ((Nt + _THREADS - 1) // _THREADS,)
        nsm = int(cp.cuda.Device().attributes['MultiProcessorCount'])
        self._coop_grid = (min(self._grid[0], 2 * nsm),)

    # -----------------------------------------------------------------
    @property
    def uses_cooperative_cc(self) -> bool:
        """True when the CC convergence loop runs device-side (no D2H)."""
        return self._k_cc_coop is not None

    def _cc(self, labels_dev):
        if self._k_cc_coop is not None:
            self._k_cc_coop(self._coop_grid, (_THREADS,),
                            (labels_dev, self._nbors,
                             np.int32(self.Ntot), np.int32(self.M1),
                             self._comp, self._changed))
            return
        # Fallback: host-driven loop (extra D2H per iteration).
        self._k_cc_init(self._grid, (_THREADS,),
                        (np.int32(self.Ntot), self._comp, self._changed))
        for _ in range(4096):
            self._changed.fill(0)
            self._k_cc_hook(self._grid, (_THREADS,),
                            (labels_dev, self._nbors,
                             np.int32(self.Ntot), np.int32(self.M1),
                             self._comp, self._changed))
            if int(self._changed.get()[0]) == 0:
                break
            self._k_cc_compress(self._grid, (_THREADS,),
                                (np.int32(self.Ntot), self._comp))

    def components_and_distance(self, labels_dev):
        """Return ``(parcel_components (L,) fp64, eucli (L,) fp32)``.

        Both are device arrays **owned by this object** -- overwritten on the
        next call. ``labels_dev`` is a bilateral ``(N,)`` int32 device array
        (LH first, RH after; 0 = medial wall).
        """
        Nt, L = self.Ntot, self.L
        g, blk = self._grid, (_THREADS,)

        self._k_reset(g, blk, (np.int32(Nt), np.int32(L), self._cmin,
                               self._comp_count, self._eucli, self._bnd_cnt))
        self._cc(labels_dev)
        self._k_count(g, blk, (labels_dev, self._comp, np.int32(Nt),
                               np.int32(L), self._comp_count))
        self._k_pc((1,), blk, (self._comp_count, np.int32(L), self._pc,
                               self._is_single))
        self._k_boundary(g, blk, (labels_dev, self._nbors, self._is_single,
                                  np.int32(Nt), np.int32(self.M1),
                                  np.int32(self.n_hemi), np.int32(self.n_lh),
                                  self._bg, self._bnd_cnt))
        self._k_scan((1,), (1,), (self._bnd_cnt, np.int32(L),
                                  self._start, self._cursor))
        self._k_scatter(g, blk, (self._bg, self._comp, self._xyz, np.int32(Nt),
                                 self._cursor, self._bx, self._by, self._bz,
                                 self._bcomp, self._bpar))
        self._k_pairs((L,), (_TILE,), (self._start, self._bx, self._by,
                                       self._bz, self._bcomp, np.int32(L),
                                       self._cmin))
        self._k_reduce(g, blk, (self._start, np.int32(L), self._bcomp,
                                self._bpar, self._cmin, self._eucli))
        return self._pc, self._eucli

    def step(self, labels_dev, xyz_gamma_dev):
        """``_check_connectedness_step`` semantics.

        Updates ``xyz_gamma_dev`` (L,) fp64 in place (``+= 1000`` at
        distributed parcels) and returns
        ``(max_connectedness, max_components)`` as Python floats. Exactly one
        device->host copy (the 2-element scalar buffer).
        """
        self.components_and_distance(labels_dev)
        self._k_step((1,), (256,), (self._eucli, self._pc, xyz_gamma_dev,
                                    np.int32(self.L),
                                    np.float64(self.connect_th),
                                    np.float64(self.components_threshold),
                                    self._out2))
        out = self._out2.get()
        return float(out[0]), float(out[1])
