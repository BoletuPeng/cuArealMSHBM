"""component_distance.py

Connectedness check + per-parcel component distance, called inside the
EM body to decide which parcels need the spatial-xyz pull.

Public API:
    compute_components_general    — BFS over parcel labels on the
                                    bilateral mesh; returns per-parcel
                                    component count.
    component_distance            — pairwise distance between components
                                    inside each parcel.

Implementation notes:
    Single-core serial @njit, out-buffer style; outputs and scratch
    pre-allocated in a per-(mesh, num_parcel) ``Workspace``. fp32 along
    the distance path; integer arrays stay int64. Mesh-level caches are
    keyed by ``id(arr)``; clear with :func:`clear_mesh_caches` when
    swapping meshes within one process.

    The MATLAB ``component_distance.m`` docstring carries an example
    whose arithmetic disagrees with the code — the function returns
    ``max_c (min_{c'≠c} pair_min(c, c'))``, not max-of-pairwise-mins.
    The Python implementation faithfully reproduces the code.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Numba-jitted hot kernels — see _kernels.py.
from . import _kernels


# ─────────────────────────────────────────────────────────────────────────
# LH/RH hemi-level parallelism.
#
# Five hot kernels (cc_from_edges, count_per_parcel, zero_singles_lh,
# zero_singles_and_reindex_rh, fused_hemi_distance) are independent per
# hemisphere and together dominate a check_connectedness call (~3.6 ms of
# 4.3 ms, measured at fsaverage6 × 300 parcels on sub-001 GT labels). We
# dispatch RH to a persistent worker thread and run LH on the main thread,
# getting ~1.85× per-call speedup (chain length 1.83 ms vs full 4.27 ms
# wall — the cap is the longer of {LH chain, RH chain} plus ~0.5 ms Python
# overhead).
#
# nogil=True on the kernels (see ``_kernels.py``) is the prerequisite —
# without it the worker would block on the GIL and the two hemis would
# serialize anyway.
#
# Disable via ``CHECK_CONN_SERIAL=1`` for batch jobs that already saturate
# CPU cores with outer subject-level multiprocessing (preserves the
# original kernel-docstring contract). The env var is read ONCE at module
# import — changing it later has no effect; embedded callers that want to
# toggle at runtime should set ``_PARALLEL_HEMI = False`` on this module
# (which makes ``_get_rh_worker`` return None on every subsequent call).
#
# Worker thread is created lazily on the first ``_dispatch_lh_rh`` call,
# not at import. Subprocesses spawned by an outer multiprocessing layer
# that never invoke check_connectedness pay no OS-thread cost; the first
# real call pays a one-time ~50–100 µs creation. Concurrent first-call
# contenders are serialized by ``_RH_WORKER_LOCK`` — guarantees one
# worker, never two, even under threaded outer loops.
#
# Reentrancy: the module-level workspace cache (``_WORKSPACE_CACHE``) and
# this single-worker executor are NOT safe for concurrent top-level
# ``component_distance`` calls from multiple application threads — they
# would race on workspace slots and serialize on the one-worker pool. The
# in-tree pipeline is single-threaded at the call site; future multi-
# threaded outer loops must give each thread its own workspace + worker
# or fall back to ``CHECK_CONN_SERIAL=1``.
# ─────────────────────────────────────────────────────────────────────────
_PARALLEL_HEMI = os.environ.get("CHECK_CONN_SERIAL", "0") != "1"
_RH_WORKER: ThreadPoolExecutor | None = None
_RH_WORKER_LOCK = threading.Lock()


def _get_rh_worker() -> ThreadPoolExecutor | None:
    """Lazy accessor for the persistent RH worker.

    Returns ``None`` when ``CHECK_CONN_SERIAL=1`` was set at module
    import — callers fall back to in-thread serial dispatch. Otherwise
    creates the worker on first call and caches it for the lifetime of
    the process. Double-checked locking keeps the fast path branch-free
    (one None-check + one return) while still serializing concurrent
    first-call contenders.
    """
    global _RH_WORKER
    if not _PARALLEL_HEMI:
        return None
    if _RH_WORKER is not None:
        return _RH_WORKER
    with _RH_WORKER_LOCK:
        if _RH_WORKER is None:
            _RH_WORKER = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="check_conn_rh",
            )
    return _RH_WORKER


def _dispatch_lh_rh(lh_fn, rh_fn) -> None:
    """Run ``lh_fn()`` on the main thread and ``rh_fn()`` on the persistent
    worker, then join. Falls back to serial when threading is disabled.

    Either function may raise; the worker's exception propagates from
    ``fut.result()``. We wait on RH even if LH raised, so a half-finished
    worker doesn't leak into the next call's closure. If BOTH raise, the
    RH exception is the one that propagates (LH preserved via
    ``__context__`` per the standard try/finally chaining semantics).
    """
    worker = _get_rh_worker()
    if worker is None:
        lh_fn()
        rh_fn()
        return
    fut = worker.submit(rh_fn)
    try:
        lh_fn()
    finally:
        fut.result()


# ─────────────────────────────────────────────────────────────────────────
# Mesh-level caches — derived once per (mesh) and reused across calls.
# Keyed by ``id(arr)``: the caller is expected to keep the source mesh array
# alive (true for the typical case of one cached fsaverage6 mesh per process).
#
# Caller contract: hold a strong reference to ``vertex_nbors`` /
# ``vertices`` for as long as you want the cached derivative reused. If
# the source array is GC'd and a new array is allocated at the same id
# slot with the same N, the shape check below will silently hit a stale
# cache. Mesh switches MUST go through ``clear_mesh_caches()``. The
# Step3Pipeline pins meshes for the process lifetime, so this is a
# non-issue in production.
# ─────────────────────────────────────────────────────────────────────────
_MESH_EDGE_CACHE: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
_MESH_XYZ_T_CACHE: dict[int, np.ndarray] = {}     # (N, 3) float32
_MESH_NBORS64_CACHE: dict[int, np.ndarray] = {}   # (max_neigh, N) int64


def _get_mesh_edges(vertex_nbors: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (src_const, dst_const, N): every (vertex, valid_neighbor) pair on
    the mesh, 0-indexed, sentinel slots filtered out. Computed once per mesh."""
    key = id(vertex_nbors)
    cached = _MESH_EDGE_CACHE.get(key)
    if cached is not None and cached[2] == vertex_nbors.shape[1]:
        return cached
    N = vertex_nbors.shape[1]
    max_neigh = vertex_nbors.shape[0]
    src = np.repeat(np.arange(N, dtype=np.int64), max_neigh)
    dst = vertex_nbors.T.reshape(-1).astype(np.int64) - 1     # 0-indexed; -1 = absent
    valid = dst >= 0
    src = src[valid].copy()    # densify so the kernel sees contiguous int64
    dst = dst[valid].copy()
    _MESH_EDGE_CACHE[key] = (src, dst, N)
    return src, dst, N


def _get_xyz_T(vertices_3xN: np.ndarray) -> np.ndarray:
    """Return (N, 3) float32 row-major sphere coords. fp32 is the canonical
    type throughout the distance pipeline."""
    key = id(vertices_3xN)
    cached = _MESH_XYZ_T_CACHE.get(key)
    if cached is not None and cached.shape[0] == vertices_3xN.shape[1]:
        return cached
    if vertices_3xN.shape[0] == 3:
        out = np.ascontiguousarray(vertices_3xN.T, dtype=np.float32)
    else:
        out = np.ascontiguousarray(vertices_3xN, dtype=np.float32)
    _MESH_XYZ_T_CACHE[key] = out
    return out


def _get_nbors64(vertex_nbors: np.ndarray) -> np.ndarray:
    """Return (max_neigh, N) int64 contiguous neighbor table."""
    key = id(vertex_nbors)
    cached = _MESH_NBORS64_CACHE.get(key)
    if cached is not None and cached.shape == vertex_nbors.shape:
        return cached
    out = np.ascontiguousarray(vertex_nbors, dtype=np.int64)
    _MESH_NBORS64_CACHE[key] = out
    return out


def clear_mesh_caches() -> None:
    """Drop ALL mesh-keyed caches: edges, xyz_T, nbors64, AND the workspace.

    All four caches are mesh-derived and share the same invalidation event
    (caller switches mesh / source array goes out of scope). Calling this
    after a mesh switch is mandatory; calling it spuriously is harmless
    (next call rebuilds; ~50-100 us cost).
    """
    _MESH_EDGE_CACHE.clear()
    _MESH_XYZ_T_CACHE.clear()
    _MESH_NBORS64_CACHE.clear()
    _WORKSPACE_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────────
# Workspace — every scratch + output buffer needed for one component_distance
# call, sized to one (LH/RH mesh, num_parcel) tuple. Allocated lazily on first
# call and reused on every subsequent call with the same mesh.
#
# Memory cost per workspace (fsaverage6, num_parcel=300):
#   ≈ 2 × (N · int64 × 8 buffers + N · float32 × 4 buffers) ≈ 4.6 MB
#   plus tiny outputs (300 floats etc.). Negligible vs the existing mesh
#   caches and well within L3.
# ─────────────────────────────────────────────────────────────────────────
class _Workspace:
    """All scratch + output buffers for one (LH/RH mesh, num_parcel) tuple.

    Caller never instantiates directly; `_get_workspace(...)` constructs and
    caches one per ``(id(lh_nbors), id(rh_nbors), num_parcel)`` key. All
    fields are owned numpy arrays — never views of caller data.

    Sizes (all dtype as marked):
        N                 : int — vertices per hemisphere
        n_lh              : int — parcels per hemisphere = num_parcel // 2

    Per-hemi CC scratch (×2: lh_, rh_):
        parent       (N,)            int64
        rank         (N,)            int8
        comp_map     (N,)            int64
        ci           (N,)            int64
        n_comp       (1,)            int64    — written by kernel; read with ``int(arr[0])``

    Per-hemi count_per_parcel scratch (×2):
        comp_to_label (N+1,)         int64    — sized to upper bound on n_comp+1
        parcel_seen   (n_lh,)        bool_
        count         (n_lh,)        float64

    Top-level outputs:
        parcel_components (num_parcel,)  float64  — eventually concatenated lh/rh count
        eucli_dist        (num_parcel,)  float32  — final result

    Single-comp lookup + zeroed labels:
        is_single         (num_parcel+1,) bool_
        lh_labels64       (N,)            int64    — caller's labels copied as int64
        rh_labels64       (N,)            int64
        lh_zeroed         (N,)            int64    — single-comp parcels zeroed
        rh_zeroed_local   (N,)            int64    — zeroed AND re-indexed to [1, n_lh]

    Per-hemi fused_hemi_distance scratch (×2):
        labels_bnd   (N,)            int64
        ci_masked    (N,)            int64
        parcel_size  (n_lh+2,)       int64
        start        (n_lh+2,)       int64
        cursor       (n_lh+2,)       int64
        bnd_x        (N,)            float32
        bnd_y        (N,)            float32
        bnd_z        (N,)            float32
        bnd_ci       (N,)            int64
        local        (N,)            int64
        uniq         (N,)            int64
        comp_min_sq  (N,)            float32
        eucli        (n_lh,)         float32
    """

    __slots__ = (
        "N", "n_lh", "num_parcel",
        # CC LH
        "lh_parent", "lh_rank", "lh_comp_map", "lh_ci", "lh_n_comp",
        # CC RH
        "rh_parent", "rh_rank", "rh_comp_map", "rh_ci", "rh_n_comp",
        # count LH
        "lh_comp_to_label", "lh_parcel_seen", "lh_count",
        # count RH
        "rh_comp_to_label", "rh_parcel_seen", "rh_count",
        # Top-level
        "parcel_components", "eucli_dist",
        "is_single", "lh_labels64", "rh_labels64",
        "lh_zeroed", "rh_zeroed_local",
        # fused_hemi_distance LH
        "lh_labels_bnd", "lh_ci_masked",
        "lh_parcel_size", "lh_start", "lh_cursor",
        "lh_bnd_x", "lh_bnd_y", "lh_bnd_z", "lh_bnd_ci",
        "lh_local", "lh_uniq", "lh_comp_min_sq", "lh_eucli",
        # fused_hemi_distance RH
        "rh_labels_bnd", "rh_ci_masked",
        "rh_parcel_size", "rh_start", "rh_cursor",
        "rh_bnd_x", "rh_bnd_y", "rh_bnd_z", "rh_bnd_ci",
        "rh_local", "rh_uniq", "rh_comp_min_sq", "rh_eucli",
    )

    def __init__(self, N: int, num_parcel: int):
        self.N = N
        self.n_lh = num_parcel // 2
        self.num_parcel = num_parcel

        n_lh = self.n_lh

        # CC scratch + output (×2)
        for h in ("lh", "rh"):
            setattr(self, f"{h}_parent",   np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_rank",     np.empty(N, dtype=np.int8))
            setattr(self, f"{h}_comp_map", np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_ci",       np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_n_comp",   np.empty(1, dtype=np.int64))

        # count scratch + output (×2)
        for h in ("lh", "rh"):
            setattr(self, f"{h}_comp_to_label", np.empty(N + 1, dtype=np.int64))
            setattr(self, f"{h}_parcel_seen",   np.empty(n_lh, dtype=np.bool_))
            setattr(self, f"{h}_count",         np.empty(n_lh, dtype=np.float64))

        # Top-level outputs + intermediate caches
        self.parcel_components = np.empty(num_parcel, dtype=np.float64)
        self.eucli_dist        = np.zeros(num_parcel, dtype=np.float32)
        self.is_single         = np.zeros(num_parcel + 1, dtype=np.bool_)
        self.lh_labels64       = np.empty(N, dtype=np.int64)
        self.rh_labels64       = np.empty(N, dtype=np.int64)
        self.lh_zeroed         = np.empty(N, dtype=np.int64)
        self.rh_zeroed_local   = np.empty(N, dtype=np.int64)

        # Per-hemi distance scratch (×2)
        for h in ("lh", "rh"):
            setattr(self, f"{h}_labels_bnd",  np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_ci_masked",   np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_parcel_size", np.empty(n_lh + 2, dtype=np.int64))
            setattr(self, f"{h}_start",       np.empty(n_lh + 2, dtype=np.int64))
            setattr(self, f"{h}_cursor",      np.empty(n_lh + 2, dtype=np.int64))
            setattr(self, f"{h}_bnd_x",       np.empty(N, dtype=np.float32))
            setattr(self, f"{h}_bnd_y",       np.empty(N, dtype=np.float32))
            setattr(self, f"{h}_bnd_z",       np.empty(N, dtype=np.float32))
            setattr(self, f"{h}_bnd_ci",      np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_local",       np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_uniq",        np.empty(N, dtype=np.int64))
            setattr(self, f"{h}_comp_min_sq", np.empty(N, dtype=np.float32))
            setattr(self, f"{h}_eucli",       np.empty(n_lh, dtype=np.float32))

        # Tripwire: any typo in __slots__ vs the loops above leaves a slot
        # unset and getattr raises AttributeError here -- catches the mismatch
        # at first construction rather than on a hot-path call site later.
        for _slot in self.__slots__:
            getattr(self, _slot)


_WORKSPACE_CACHE: dict[tuple[int, int, int], _Workspace] = {}


def _get_workspace(lh_mesh: dict, rh_mesh: dict, num_parcel: int) -> _Workspace:
    """Lazy-initialize and cache a Workspace per (LH mesh, RH mesh, num_parcel)."""
    lh_n = lh_mesh["vertexNbors"]
    rh_n = rh_mesh["vertexNbors"]
    N = lh_n.shape[1]
    if N != rh_n.shape[1]:
        raise ValueError(f"LH/RH must share N (got {lh_n.shape[1]} vs {rh_n.shape[1]})")
    key = (id(lh_n), id(rh_n), num_parcel)
    ws = _WORKSPACE_CACHE.get(key)
    if ws is not None and ws.N == N and ws.num_parcel == num_parcel:
        return ws
    ws = _Workspace(N, num_parcel)
    _WORKSPACE_CACHE[key] = ws
    return ws


# ─────────────────────────────────────────────────────────────────────────
# Per-parcel component count — bilateral public entry.
#
# ``compute_components_general`` allocates ad-hoc per-call buffers; the
# component-distance fast path below goes through the workspace and pays
# no per-call allocation. Used in-tree by ``vmf_clustering`` (gMSHBM /
# cMSHBM component-count predicate).
# ─────────────────────────────────────────────────────────────────────────
def compute_components_general(lh_labels: np.ndarray,
                               rh_labels: np.ndarray,
                               lh_vertex_nbors: np.ndarray,
                               rh_vertex_nbors: np.ndarray,
                               num_parcel: int,
                               return_ci: bool = False):
    """Per-parcel component count.

    Returns ``parcel_components`` (num_parcel,) float64 with NaN for empty
    parcels. With ``return_ci=True``, also returns ``(lh_ci, rh_ci)`` so the
    caller can pass them to ``component_distance(..., lh_ci_full=, rh_ci_full=)``
    to reuse work.

    LH and RH cc + count chains are dispatched on parallel threads (see the
    ``_dispatch_lh_rh`` block at module top); ``cc_from_edges`` /
    ``count_per_parcel`` carry ``nogil=True`` so the worker actually runs.

    Internal cc invocation is inlined here (rather than via a separate
    components helper) so we skip the ``np.bincount`` sizes step — sizes
    are unused here; n_comp comes from the kernel's out buffer. The
    cc-calling sites here and in ``component_distance`` maintain the
    cc_from_edges argument list independently; if the kernel signature
    changes, both need to update in lockstep.
    """
    n_lh = num_parcel // 2
    lh_labels64 = np.ascontiguousarray(lh_labels, dtype=np.int64)
    rh_labels64 = np.ascontiguousarray(rh_labels, dtype=np.int64)

    lh_src, lh_dst, lh_N = _get_mesh_edges(lh_vertex_nbors)
    rh_src, rh_dst, rh_N = _get_mesh_edges(rh_vertex_nbors)

    # Per-call scratch — small (~1.6 MB total for fsaverage6). Could share a
    # workspace with component_distance to skip these allocs, but the API
    # split keeps it simple; the malloc cost is ~50 µs vs ~1 ms of cc work.
    lh_parent = np.empty(lh_N, dtype=np.int64)
    lh_rank = np.empty(lh_N, dtype=np.int8)
    lh_comp_map = np.empty(lh_N, dtype=np.int64)
    lh_ci = np.empty(lh_N, dtype=np.int64)
    lh_n_comp_out = np.empty(1, dtype=np.int64)

    rh_parent = np.empty(rh_N, dtype=np.int64)
    rh_rank = np.empty(rh_N, dtype=np.int8)
    rh_comp_map = np.empty(rh_N, dtype=np.int64)
    rh_ci = np.empty(rh_N, dtype=np.int64)
    rh_n_comp_out = np.empty(1, dtype=np.int64)

    lh_parcel_seen = np.empty(n_lh, dtype=np.bool_)
    rh_parcel_seen = np.empty(n_lh, dtype=np.bool_)
    lh_counts = np.empty(n_lh, dtype=np.float64)
    rh_counts = np.empty(n_lh, dtype=np.float64)

    def _lh_chain():
        _kernels.cc_from_edges(lh_src, lh_dst, lh_labels64,
                                lh_parent, lh_rank, lh_comp_map,
                                lh_ci, lh_n_comp_out)
        lh_n_comp = int(lh_n_comp_out[0])
        lh_comp_to_label = np.empty(lh_n_comp + 1, dtype=np.int64)
        _kernels.count_per_parcel(lh_labels64, lh_ci, lh_n_comp, n_lh, 0,
                                   lh_comp_to_label, lh_parcel_seen, lh_counts)

    def _rh_chain():
        _kernels.cc_from_edges(rh_src, rh_dst, rh_labels64,
                                rh_parent, rh_rank, rh_comp_map,
                                rh_ci, rh_n_comp_out)
        rh_n_comp = int(rh_n_comp_out[0])
        rh_comp_to_label = np.empty(rh_n_comp + 1, dtype=np.int64)
        _kernels.count_per_parcel(rh_labels64, rh_ci, rh_n_comp, n_lh, n_lh,
                                   rh_comp_to_label, rh_parcel_seen, rh_counts)

    _dispatch_lh_rh(_lh_chain, _rh_chain)

    out = np.concatenate([lh_counts, rh_counts])
    if return_ci:
        return out, lh_ci, rh_ci
    return out


# ─────────────────────────────────────────────────────────────────────────
# Top-level entry — workspace-backed fast path.
# ─────────────────────────────────────────────────────────────────────────
def component_distance(lh_labels: np.ndarray,
                       rh_labels: np.ndarray,
                       lh_mesh: dict,
                       rh_mesh: dict,
                       num_parcel: int,
                       parcel_components: np.ndarray | None = None,
                       lh_ci_full: np.ndarray | None = None,
                       rh_ci_full: np.ndarray | None = None,
                       copy_output: bool = True) -> np.ndarray:
    """Mirror of CBIG_ArealMSHBM_component_distance.

    Parameters
    ----------
    lh_labels, rh_labels : (N,) int. Medial wall = 0.
        LH parcel ids ∈ [1, num_parcel/2]; RH ∈ [num_parcel/2+1, num_parcel].
    lh_mesh, rh_mesh : dict-like with keys ``vertices`` (3, N) and
        ``vertexNbors`` (max_neigh, N). Vertices on the **sphere** surface,
        matching MATLAB's ``CBIG_ReadNCAvgMesh(..., 'sphere', 'cortex')``.
        Internally cached as float32 and int64 contiguous.
    num_parcel : int.
    parcel_components : (num_parcel,) optional. If given, skips the internal
        count step (CC still re-runs unless ``lh_ci_full`` / ``rh_ci_full``
        are also supplied). The fast path is the default — leave this None.
    lh_ci_full, rh_ci_full : (N,) optional. Sibling shortcut to ``parcel_components``.
    copy_output : bool. If True (default), the returned array is a fresh copy
        the caller can keep around. If False, returns a view into the workspace
        — *the next call to component_distance with the same mesh will overwrite
        it*. Use ``copy_output=False`` only inside a tight EM loop where you
        consume eucli_dist before the next call.

    Returns
    -------
    eucli_dist : (num_parcel,) float32. (Note: previous versions returned
        float64. Audited against MATLAB GT before the MATLAB-decoupling
        cleanup: <1 ULP drift end-to-end.)
    """
    ws = _get_workspace(lh_mesh, rh_mesh, num_parcel)
    n_lh = num_parcel // 2

    lh_nbors64 = _get_nbors64(lh_mesh["vertexNbors"])
    rh_nbors64 = _get_nbors64(rh_mesh["vertexNbors"])
    lh_xyz_T   = _get_xyz_T(lh_mesh["vertices"])    # (N, 3) float32
    rh_xyz_T   = _get_xyz_T(rh_mesh["vertices"])
    lh_src, lh_dst, _ = _get_mesh_edges(lh_mesh["vertexNbors"])
    rh_src, rh_dst, _ = _get_mesh_edges(rh_mesh["vertexNbors"])

    # Stage labels into the workspace as int64 (no-op if already int64 & contiguous).
    # casting="safe" tripwire: accepts widening int casts (int8/16/32 → int64) but
    # rejects float → int (would silently truncate). The pipeline call sites
    # supply int64 labels already; this only fires for future external callers.
    np.copyto(ws.lh_labels64, lh_labels, casting="safe")
    np.copyto(ws.rh_labels64, rh_labels, casting="safe")

    # ----- Phase 1 (slow path only): cc + count per hemi, dispatched parallel.
    # Hot path from vmf_clustering_*.py supplies both ``parcel_components`` and
    # ``lh_ci_full`` / ``rh_ci_full`` — Phase 1 is skipped entirely. The slow
    # path pays one fork-join here; cc+count is ~1.0 ms/hemi serial → ~1.0 ms
    # parallel.
    need_cc = lh_ci_full is None or rh_ci_full is None
    need_count = parcel_components is None
    # ``count_per_parcel`` reads ws.lh_ci / ws.lh_n_comp, which are only
    # written by ``cc_from_edges`` against the workspace. If the caller
    # supplied lh_ci_full / rh_ci_full (external buffers) but not pc, count
    # would otherwise read stale workspace data — must re-run cc so the
    # workspace slots are populated for the current labels. This matches
    # the pre-PR semantics (cc unconditional whenever pc was missing) and
    # only fires on the (pc=None, ci=given) sibling-shortcut path; in-tree
    # callers pass both, so the hot path is unaffected.
    #
    # Forward pointer: if the shortcut path ever becomes hot, the cleaner
    # fix is to teach ``count_per_parcel`` to accept external (ci, n_comp)
    # rather than reading the workspace slots — eliminates the ~50 µs of
    # redundant cc work here (cc result then unused; fhd still consumes
    # the caller's lh_ci_full). Kernel signature change is out of scope
    # for this fixup.
    if need_count:
        need_cc = True
    if need_cc or need_count:
        def _lh_cc_count():
            if need_cc:
                _kernels.cc_from_edges(
                    lh_src, lh_dst, ws.lh_labels64,
                    ws.lh_parent, ws.lh_rank, ws.lh_comp_map,
                    ws.lh_ci, ws.lh_n_comp,
                )
            if need_count:
                _kernels.count_per_parcel(
                    ws.lh_labels64, ws.lh_ci, int(ws.lh_n_comp[0]), n_lh, 0,
                    ws.lh_comp_to_label, ws.lh_parcel_seen, ws.lh_count,
                )

        def _rh_cc_count():
            if need_cc:
                _kernels.cc_from_edges(
                    rh_src, rh_dst, ws.rh_labels64,
                    ws.rh_parent, ws.rh_rank, ws.rh_comp_map,
                    ws.rh_ci, ws.rh_n_comp,
                )
            if need_count:
                _kernels.count_per_parcel(
                    ws.rh_labels64, ws.rh_ci, int(ws.rh_n_comp[0]), n_lh, n_lh,
                    ws.rh_comp_to_label, ws.rh_parcel_seen, ws.rh_count,
                )

        _dispatch_lh_rh(_lh_cc_count, _rh_cc_count)

        if need_cc:
            if lh_ci_full is None:
                lh_ci_full = ws.lh_ci
            if rh_ci_full is None:
                rh_ci_full = ws.rh_ci
        if need_count:
            ws.parcel_components[:n_lh] = ws.lh_count
            ws.parcel_components[n_lh:] = ws.rh_count
            pc = ws.parcel_components
        else:
            pc = np.asarray(parcel_components, dtype=np.float64)
    else:
        pc = np.asarray(parcel_components, dtype=np.float64)

    # Early-out if no parcel is distributed.
    if not np.any(pc > 1):
        ws.eucli_dist[:] = np.float32(0.0)
        return ws.eucli_dist.copy() if copy_output else ws.eucli_dist

    # ----- Build is_single lookup (serial, main thread; cheap).
    ws.is_single[0] = False
    np.equal(pc, 1, out=ws.is_single[1:])

    # ----- Phase 2: zero_singles + fused_hemi_distance per hemi, parallel.
    # This is the hot phase — each hemi chain is ~0.94 ms (0.025 zero +
    # 0.92 fhd) serial; running in parallel collapses to ~0.94 ms wall, the
    # main saving over the 1.85 ms serial total.
    def _lh_zero_fhd():
        _kernels.zero_singles_lh(ws.lh_labels64, ws.is_single, ws.lh_zeroed)
        _kernels.fused_hemi_distance(
            lh_nbors64, lh_xyz_T, ws.lh_zeroed, lh_ci_full, n_lh,
            ws.lh_labels_bnd, ws.lh_ci_masked,
            ws.lh_parcel_size, ws.lh_start, ws.lh_cursor,
            ws.lh_bnd_x, ws.lh_bnd_y, ws.lh_bnd_z, ws.lh_bnd_ci,
            ws.lh_local, ws.lh_uniq, ws.lh_comp_min_sq,
            ws.lh_eucli,
        )

    def _rh_zero_fhd():
        _kernels.zero_singles_and_reindex_rh(ws.rh_labels64, ws.is_single, n_lh,
                                              ws.rh_zeroed_local)
        _kernels.fused_hemi_distance(
            rh_nbors64, rh_xyz_T, ws.rh_zeroed_local, rh_ci_full, n_lh,
            ws.rh_labels_bnd, ws.rh_ci_masked,
            ws.rh_parcel_size, ws.rh_start, ws.rh_cursor,
            ws.rh_bnd_x, ws.rh_bnd_y, ws.rh_bnd_z, ws.rh_bnd_ci,
            ws.rh_local, ws.rh_uniq, ws.rh_comp_min_sq,
            ws.rh_eucli,
        )

    _dispatch_lh_rh(_lh_zero_fhd, _rh_zero_fhd)

    # ----- Concatenate into the workspace's eucli_dist buffer.
    ws.eucli_dist[:n_lh] = ws.lh_eucli
    ws.eucli_dist[n_lh:] = ws.rh_eucli
    return ws.eucli_dist.copy() if copy_output else ws.eucli_dist
