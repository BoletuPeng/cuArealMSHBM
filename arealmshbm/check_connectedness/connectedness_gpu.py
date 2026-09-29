"""connectedness_gpu.py

CuPy / CUDA-C port of the step-3 ``check_connectedness`` block
(``docs/step3_sparse_design.md`` §2.6).

Reproduces the CPU chain

    compute_components_general(...)  -> parcel_components (L,) fp64, NaN if empty
    component_distance(...)          -> eucli              (L,) fp32

plus the decision logic of
``arealmshbm.vmf_clustering.vmf_clustering._check_connectedness_step``
(distributed mask, ``xyz_gamma += 1000``, ``max_connectedness`` /
``max_components``), and -- when ``isolated_component_min_size`` is set
(cMSHBM) -- the ``remove_isolated_surface_components`` pre-predicate that
the CPU chain applies to the argmax labels before the components /
distance test.

Design
------
* **No floating-point ``atomicAdd``.** The only float atomics are
  ``atomicMin`` / ``atomicMax`` done through int reinterpretation of
  non-negative floats -- exact and order-independent, so the backend is
  run-to-run bit-reproducible. Integer ``atomicAdd`` / ``atomicExch``
  (component counts, bucket cursors, the pre-predicate's member lists and
  votes) are exact; bucket and list *order* varies but every consumer is
  order-independent (a min/max over the bucket, a histogram).
* **Connected components** use hook-to-min + pointer jumping
  (Soman/Kishore/Narayanan). Component ids are *root vertex indices*, so
  they differ from the CPU's encounter-order ids -- only the partition is
  contractual, and both downstream consumers (distinct-count per parcel,
  component membership for the distance) depend on the partition only.
  The loop runs entirely on device inside one cooperative-groups kernel
  (``grid.sync()``), so ``step()`` needs no host round-trip for the
  convergence flag. Cooperative launch is a requirement of this backend:
  the device must support it, and CuPy compiles the kernel with
  ``enable_cooperative_groups=True``, which links ``cudadevrt`` from the
  CUDA toolkit (``CUDA_PATH``). A box without either fails at
  construction; use ``backend_step3='cpu'`` there.
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
from typing import Optional

import numpy as np

import cupy as cp


_THREADS = 256
_TILE = 256

# ---------------------------------------------------------------------------
# Module A -- connected components (cooperative-groups kernel)
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

/* ---- cMSHBM pre-predicate: remove_isolated_surface_components on device ----
   Runs on the hook-to-min partition of ``labels_in`` (``comp[v]`` is the
   root vertex of v's component). ``k_ri_push`` threads every component's
   members into a per-root linked list (``head[root]`` / ``next[v]``),
   ``k_ri_small`` lists the roots of the components with fewer than ``thr``
   members, and ``k_ri_relabel`` gives each of those one block: a shared-
   memory histogram of the members' neighbours' labels (0 and the
   component's own label excluded), the most frequent label wins, the
   smallest on ties. Every vote reads ``labels_in``, the counts are
   integers and the list / slot orders (which vary run to run) only
   permute commutative operations, so the result is bit-identical to the
   host function and run-to-run reproducible. O(size * M1 + L) per small
   component, flat in ``thr``. */
extern "C" __global__ void k_ri_push(const int* __restrict__ comp, int Ntot,
                                     int* head, int* next)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int v = tid; v < Ntot; v += stride) next[v] = atomicExch(&head[comp[v]], v);
}

extern "C" __global__ void k_ri_small(const int* __restrict__ head,
                                      const int* __restrict__ next,
                                      int Ntot, int thr, int* small, int* nsmall)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int r = tid; r < Ntot; r += stride) {
        int n = 0;                              /* head[r] < 0: r is not a root */
        for (int v = head[r]; v >= 0 && n < thr; v = next[v]) ++n;
        if (n > 0 && n < thr) small[atomicAdd(nsmall, 1)] = r;
    }
}

extern "C" __global__ void k_ri_relabel(const int* __restrict__ labels_in,
                                        const int* __restrict__ nbors,
                                        const int* __restrict__ head,
                                        const int* __restrict__ next,
                                        const int* __restrict__ small,
                                        const int* __restrict__ nsmall,
                                        int Ntot, int M1, int L, int* labels_out)
{
    extern __shared__ int hist[];               /* L counts; label l at hist[l - 1] */
    __shared__ unsigned long long s_best;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warp = blockDim.x >> 5;
    const int ns = nsmall[0];
    for (int s = blockIdx.x; s < ns; s += gridDim.x) {
        const int r = small[s];
        const int self_label = labels_in[r];    /* the root is a member */
        for (int p = threadIdx.x; p < L; p += blockDim.x) hist[p] = 0;
        if (threadIdx.x == 0) s_best = 0ULL;
        __syncthreads();
        /* warp w takes every n_warp-th member, lane k that member's k-th slot */
        int i = 0;
        for (int v = head[r]; v >= 0; v = next[v], ++i) {
            if (i % n_warp != warp) continue;
            for (int k = lane; k < M1; k += 32) {
                const int u = nbors[k * Ntot + v];
                if (u < 0) continue;
                const int c = labels_in[u];
                if (c == 0 || c == self_label) continue;
                atomicAdd(&hist[c - 1], 1);
            }
        }
        __syncthreads();
        /* key = (count, L - label): its max is the most frequent label, the
           smallest one on ties */
        unsigned long long mine = 0ULL;
        for (int p = threadIdx.x; p < L; p += blockDim.x) {
            const int h = hist[p];
            if (h > 0) {
                const unsigned long long key =
                    ((unsigned long long)h << 32) | (unsigned long long)(L - 1 - p);
                if (key > mine) mine = key;
            }
        }
        if (mine) atomicMax(&s_best, mine);
        __syncthreads();
        if (s_best != 0ULL && threadIdx.x == 0) {   /* no candidate: label kept */
            const int best = L - (int)(s_best & 0xffffffffULL);
            for (int v = head[r]; v >= 0; v = next[v]) labels_out[v] = best;
        }
        __syncthreads();
    }
}
'''


_MODULES_LOCK = threading.Lock()
_MODULES = None


def _compiled_modules():
    """``(mod_main, k_cc_coop)`` — compiled once per process.

    The step-3 stage pipeline builds one :class:`ConnectednessGPU` per
    subject on the LOAD thread while EM workers run on other streams, so
    the NVRTC compiles must not repeat per instance. Not keyed by
    device — this is a single-GPU deployment.
    """
    global _MODULES
    with _MODULES_LOCK:
        if _MODULES is not None:
            return _MODULES
        dev = cp.cuda.Device()
        if not dev.attributes['CooperativeLaunch']:
            name = cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()
            raise RuntimeError(
                f"GPU {dev.id} ({name}) does not support cooperative launch, "
                f"which backend_step3='gpu' needs for gMSHBM / cMSHBM "
                f"(check_connectedness). Use backend_step3='cpu'.")
        if cp.cuda.get_cuda_path() is None:
            raise RuntimeError(
                "backend_step3='gpu' needs a CUDA toolkit at CUDA_PATH for "
                "gMSHBM / cMSHBM: the cooperative connected-components "
                "kernel (check_connectedness) links cudadevrt and includes "
                "cooperative_groups.h from it. Set CUDA_PATH to the "
                "toolkit root or use backend_step3='cpu'.")
        opts = ('--fmad=false',)
        mod = cp.RawModule(code=_MAIN_SRC, options=opts, backend='nvrtc')
        # Cooperative CC keeps the whole convergence loop on device (no D2H).
        mod_coop = cp.RawModule(code=_CC_COOP_SRC, options=opts,
                                backend='nvrtc',
                                enable_cooperative_groups=True)
        _MODULES = (mod, mod_coop.get_function('k_cc_coop'))
        return _MODULES


def prewarm_connectedness_gpu() -> None:
    """Check cooperative-launch support and the CUDA toolkit, and compile
    and link the cooperative connected-components kernel now, so a device
    or toolkit that cannot run it fails before step 0. The main module
    compiles at its first launch."""
    _compiled_modules()


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
    isolated_component_min_size : int or None -- when set, ``step`` first
        relabels every same-label component smaller than this to its
        neighbours' mode label (cMSHBM's
        :func:`remove_isolated_surface_components`; the argument is
        ``abs_threshold``) and runs the predicate on the cleaned copy.
        The caller's labels are not modified. ``None`` = no pre-pass.

    All device buffers are allocated here; ``components_and_distance``,
    ``remove_isolated`` and ``step`` allocate nothing. Labels are always a
    C-contiguous ``(2n,)`` int32 device array (LH first, RH after; 0 =
    medial wall) that this object does not own.
    """

    def __init__(self, lh_vertex_nbors, rh_vertex_nbors,
                 lh_vertices, rh_vertices,
                 num_parcel: int, connect_th: float,
                 components_threshold: int,
                 isolated_component_min_size: Optional[int] = None):
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

        self.isolated_component_min_size: Optional[int] = None
        if isolated_component_min_size is not None:
            thr = int(isolated_component_min_size)
            if thr <= 0:
                raise ValueError(
                    f"isolated_component_min_size must be positive (got {thr})")
            self.isolated_component_min_size = thr
            # A component has at most Ntot members, so any larger threshold
            # is the same predicate (and fits the kernel's int argument).
            self._ri_thr = np.int32(min(thr, Nt + 1))
            self._ri_head = cp.empty(Nt, dtype=cp.int32)
            self._ri_next = cp.empty(Nt, dtype=cp.int32)
            self._ri_small = cp.empty(Nt, dtype=cp.int32)
            self._ri_nsmall = cp.zeros(1, dtype=cp.int32)
            self._labels_clean = cp.empty(Nt, dtype=cp.int32)

        self._mod, self._k_cc_coop = _compiled_modules()
        self._k_reset = self._mod.get_function('k_reset')
        self._k_count = self._mod.get_function('k_count')
        self._k_pc = self._mod.get_function('k_pc')
        self._k_boundary = self._mod.get_function('k_boundary')
        self._k_scan = self._mod.get_function('k_scan')
        self._k_scatter = self._mod.get_function('k_scatter')
        self._k_pairs = self._mod.get_function('k_pairs')
        self._k_reduce = self._mod.get_function('k_reduce')
        self._k_step = self._mod.get_function('k_step')
        self._k_ri_push = self._mod.get_function('k_ri_push')
        self._k_ri_small = self._mod.get_function('k_ri_small')
        self._k_ri_relabel = self._mod.get_function('k_ri_relabel')

        self._grid = ((Nt + _THREADS - 1) // _THREADS,)
        nsm = int(cp.cuda.Device().attributes['MultiProcessorCount'])
        self._coop_grid = (min(self._grid[0], 2 * nsm),)

    # -----------------------------------------------------------------
    def _check_labels(self, labels_dev):
        if (labels_dev.dtype != cp.int32 or labels_dev.shape != (self.Ntot,)
                or not labels_dev.flags.c_contiguous):
            raise ValueError(
                f"labels must be a C-contiguous ({self.Ntot},) int32 device "
                f"array; got {labels_dev.dtype} {labels_dev.shape}")

    def _cc(self, labels_dev):
        self._k_cc_coop(self._coop_grid, (_THREADS,),
                        (labels_dev, self._nbors,
                         np.int32(self.Ntot), np.int32(self.M1),
                         self._comp, self._changed))

    def components_and_distance(self, labels_dev):
        """Return ``(parcel_components (L,) fp64, eucli (L,) fp32)``.

        Both are device arrays **owned by this object** -- overwritten on the
        next call. ``labels_dev`` is a bilateral ``(N,)`` int32 device array
        (LH first, RH after; 0 = medial wall).
        """
        self._check_labels(labels_dev)
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

    def remove_isolated(self, labels_dev):
        """Device ``remove_isolated_surface_components(labels, nbors, thr)``.

        ``labels_dev`` is a bilateral ``(N,)`` int32 device array (LH first,
        RH after; 0 = medial wall) and is left untouched. Returns the cleaned
        copy, a device array **owned by this object** -- overwritten on the
        next call (and not accepted back as input: the votes must read a
        buffer the relabel does not write). Bit-identical to the host
        function applied per hemisphere: the packed neighbour table has no
        cross-hemisphere edge, so one bilateral pass is the two per-hemi
        passes. Requires ``isolated_component_min_size``.
        """
        if self.isolated_component_min_size is None:
            raise ValueError("ConnectednessGPU built without "
                             "isolated_component_min_size")
        self._check_labels(labels_dev)
        if cp.may_share_memory(labels_dev, self._labels_clean):
            raise ValueError("labels_dev is the buffer remove_isolated returns")
        Nt = self.Ntot
        g, blk = self._grid, (_THREADS,)
        self._cc(labels_dev)
        self._ri_head.fill(-1)
        self._ri_nsmall.fill(0)
        self._k_ri_push(g, blk, (self._comp, np.int32(Nt), self._ri_head,
                                 self._ri_next))
        self._k_ri_small(g, blk, (self._ri_head, self._ri_next, np.int32(Nt),
                                  self._ri_thr, self._ri_small, self._ri_nsmall))
        cp.copyto(self._labels_clean, labels_dev)
        self._k_ri_relabel(g, blk, (labels_dev, self._nbors, self._ri_head,
                                    self._ri_next, self._ri_small,
                                    self._ri_nsmall, np.int32(Nt),
                                    np.int32(self.M1), np.int32(self.L),
                                    self._labels_clean),
                           shared_mem=4 * self.L)
        return self._labels_clean

    def step(self, labels_dev, xyz_gamma_dev):
        """``_check_connectedness_step`` semantics.

        With ``isolated_component_min_size`` set, the predicate runs on
        :meth:`remove_isolated`'s cleaned copy of ``labels_dev`` (cMSHBM);
        the cleaned labels stay internal, as they do on the CPU chain.
        Updates ``xyz_gamma_dev`` (L,) fp64 in place (``+= 1000`` at
        distributed parcels) and returns
        ``(max_connectedness, max_components)`` as Python floats. Exactly one
        device->host copy (the 2-element scalar buffer).
        """
        if self.isolated_component_min_size is not None:
            labels_dev = self.remove_isolated(labels_dev)
        self.components_and_distance(labels_dev)
        self._k_step((1,), (256,), (self._eucli, self._pc, xyz_gamma_dev,
                                    np.int32(self.L),
                                    np.float64(self.connect_th),
                                    np.float64(self.components_threshold),
                                    self._out2))
        out = self._out2.get()
        return float(out[0]), float(out[1])
