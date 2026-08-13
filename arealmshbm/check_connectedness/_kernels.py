"""_kernels.py

Numba kernels for component_distance. Single-core, out-buffer, fp32
along the distance path.

Design rules (all kernels obey):

* **Serial.** No ``parallel=True``, no ``prange``. Per-call latency is
  sub-millisecond, so cores are reserved for outer subject-level
  multiprocessing.
* **Out-buffer style.** Every kernel receives pre-allocated output AND
  scratch arrays as parameters. Nothing is allocated inside a kernel;
  nothing is returned except trivial scalars. Caller owns all memory.
  See the *Buffer reference* below.
* **fp32 throughout the distance path.** Vertex coordinates, distance
  intermediates, per-component min² and the final eucli output are all
  float32. Integer arrays (labels, ci, edge endpoints, parcel offsets)
  stay int64.

================================ Buffer reference ============================

E              = number of candidate edges in mesh (≈ N · max_neigh,
                 minus absent slots; for fsaverage6: ~245k)
N              = number of vertices in one hemisphere (fsaverage6: 40962)
max_neigh      = max number of neighbors per vertex (fsaverage6: 6)
n_parcels      = parcels per hemisphere (Mode-A 300-ROI run: n_parcels=150)
n_comp         = total connected components produced by `cc_from_edges`
                 on a given (mesh, labels) pair (varies; bounded above by N)
max_parcel_sz  = max boundary verts in any one multi-component parcel
                 (varies; bounded above by N)

== `cc_from_edges` (union-find with path halving + union by rank)
   inputs:    src         (E,)            int64    read-only
              dst         (E,)            int64    read-only
              labels      (N,)            int64    read-only
   scratch:   parent      (N,)            int64    rewritten
              rank        (N,)            int8     rewritten
              comp_map    (N,)            int64    rewritten
   outputs:   ci          (N,)            int64    rewritten — 1-indexed comp id
              n_comp_out  (1,)            int64    rewritten — total comp count

== `boundary` (two-vertex-thick boundary detection, sum-of-differences)
   inputs:    vertex_nbors (max_neigh, N) int64    read-only — MATLAB 1-indexed
              labels       (N,)           int64    read-only
   outputs:   out          (N,)           int64    rewritten — boundary verts → 0

== `count_per_parcel` (component-count per parcel, NaN for empty)
   inputs:    labels        (N,)             int64    read-only
              ci            (N,)             int64    read-only
              n_comp        scalar           int      hint for sizing the
                                                       comp_to_label scratch
              n_parcels     scalar           int
              label_offset  scalar           int      0 for LH, n_lh for RH
   scratch:   comp_to_label (≥ n_comp+1,)  int64    rewritten
              parcel_seen   (n_parcels,)   bool_    rewritten
   outputs:   count_out     (n_parcels,)   float64  rewritten — NaN where empty

== `fused_hemi_distance` (boundary + parcel grouping + min/max distance)
   inputs:    vertex_nbors    (max_neigh, N)        int64    read-only
              vertices_xyz_T  (N, 3)                float32  read-only
              labels_zeroed   (N,)                  int64    read-only
              ci_full         (N,)                  int64    read-only
              num_parcels_hemi scalar               int
   scratch:   labels_bnd      (N,)                  int64    rewritten
              ci_masked       (N,)                  int64    rewritten
              parcel_size     (num_parcels_hemi+2,) int64    rewritten
              start           (num_parcels_hemi+2,) int64    rewritten
              cursor          (num_parcels_hemi+2,) int64    rewritten
              bnd_x           (N,)                  float32  rewritten
              bnd_y           (N,)                  float32  rewritten
              bnd_z           (N,)                  float32  rewritten
              bnd_ci          (N,)                  int64    rewritten
              local           (N,)                  int64    rewritten
              uniq            (N,)                  int64    rewritten
              comp_min_sq     (N,)                  float32  rewritten
   outputs:   out_eucli       (num_parcels_hemi,)   float32  rewritten

==============================================================================

Equivalence to MATLAB:
* ``cc_from_edges``: union-find produces the same vertex partition as
  scipy's ``connected_components`` (component-ID numbering is encounter
  order rather than scipy order; only the partition matters downstream).
* ``boundary``: literal port of
  CBIG_ArealMSHBM_BuildTwoVertThickBoundary's sum-of-differences
  criterion.
* ``count_per_parcel``: each connected component contains vertices of
  exactly one label, so a single O(N) component→label map plus an
  O(n_comp) count gives the same result as
  ``np.unique(ci[labels == k]).size``.
* ``fused_hemi_distance``: numerical equivalent of the MATLAB chain
      labels_tmp  = where(labels_zeroed==0, fake, labels_zeroed)
      up_labels   = boundary(vertex_nbors, labels_tmp)
      labels_bnd  = labels_zeroed * (up_labels == 0)
      ci_masked   = where(labels_zeroed==0, 0, ci_full)
      eucli[p]    = max_{c ∈ p} (min_{c'≠c, c' ∈ p} pair_min(c, c'))
  with all intermediates inlined and per-vertex xyz held in float32.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────
# 1. Connected components (union-find, path halving, union by rank)
# ─────────────────────────────────────────────────────────────────────────
# ``nogil=True`` lets ``component_distance`` / ``compute_components_general``
# dispatch LH and RH on two threads. Numba defaults to nogil=False;
# without this flag the worker would block on the GIL and the two hemis
# would serialize anyway. The kernels remain single-threaded internally —
# the docstring's "Serial" contract is about per-kernel threading, not
# about top-level orchestration.
#
# Five kernels are on the parallel-dispatch path:
#   * Phase 1 (cc + count, slow path only):
#         cc_from_edges, count_per_parcel
#   * Phase 2 (always-on):
#         zero_singles_lh, zero_singles_and_reindex_rh, fused_hemi_distance
# ``boundary`` also carries nogil=True for consistency, but as of this
# commit it has no production caller — it is exercised only by ``warmup()``
# (the standalone ``build_two_vert_thick_boundary`` helper that used to call
# it was removed as dead code). Keeping it nogil means any future caller
# that wants to thread boundary work needs no kernel-side change.
@njit(cache=True, nogil=True, fastmath=False)
def cc_from_edges(src, dst, labels, parent, rank, comp_map, ci, n_comp_out):
    N = labels.shape[0]
    for v in range(N):
        parent[v] = v
        rank[v] = 0
        comp_map[v] = -1

    n_edges = src.shape[0]
    for e in range(n_edges):
        u = src[e]
        v = dst[e]
        if labels[u] != labels[v]:
            continue
        # find(u) with path halving
        x = u
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        ru = x
        # find(v)
        x = v
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        rv = x
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1

    n_comp = 0
    for v in range(N):
        x = v
        while parent[x] != x:
            x = parent[x]
        r = x
        if comp_map[r] < 0:
            comp_map[r] = n_comp
            n_comp += 1
        ci[v] = comp_map[r] + 1
    n_comp_out[0] = n_comp


# ─────────────────────────────────────────────────────────────────────────
# 2. Two-vertex-thick boundary
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=False)
def boundary(vertex_nbors, labels, out):
    """Standalone boundary detector — RAW-LABEL contract.

    Mirrors MATLAB CBIG_ArealMSHBM_BuildTwoVertThickBoundary on raw labels
    (medial wall = 0). Out: same labels with boundary verts set to 0;
    medial-wall verts STAY 0. A medial neighbor (nl == 0) contributes 0 to
    the diff sum (we ``continue``); MATLAB achieves the same by masking
    ``lh_temp(lh_full_labels==0) = 0`` after the bsxfun subtraction.

    NOT interchangeable with the inlined boundary inside
    ``fused_hemi_distance`` — that one uses a different contract
    (pre-sentineled labels with medial = ``num_parcel + 1``) to mirror
    MATLAB ``component_distance.m``'s line 60-65 which substitutes
    medial → ``num_parcel + 1`` BEFORE calling BuildTwoVertThickBoundary.
    The two outputs differ for vertices in distributed parcels adjacent to
    medial / single-comp parcels. Do NOT try to deduplicate without first
    re-validating the GT — the bit-equal pipeline result depends on each
    kernel matching its own caller's contract.
    """
    max_neigh, N = vertex_nbors.shape
    for v in range(N):
        own = labels[v]
        out[v] = own
        if own == 0:
            continue
        diff_sum = 0
        for k in range(max_neigh):
            nb = vertex_nbors[k, v]
            if nb == 0:
                continue
            nl = labels[nb - 1]
            if nl == 0:
                continue
            diff_sum += (nl - own)
        if diff_sum != 0:
            out[v] = 0


# ─────────────────────────────────────────────────────────────────────────
# 3. Per-parcel unique-component-count via comp→label mapping
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=False)
def count_per_parcel(labels, ci, n_comp, n_parcels, label_offset,
                     comp_to_label, parcel_seen, count_out):
    N = labels.shape[0]
    # Reset scratch + output
    for c in range(n_comp + 1):
        comp_to_label[c] = -1
    for p in range(n_parcels):
        parcel_seen[p] = False
        count_out[p] = 0.0

    # Map each component to its (single) label and mark non-empty parcels.
    for v in range(N):
        lbl = labels[v]
        if lbl > label_offset and lbl <= label_offset + n_parcels:
            parcel_seen[lbl - label_offset - 1] = True
            c = ci[v]
            if c >= 1 and c <= n_comp:
                if comp_to_label[c] < 0:
                    comp_to_label[c] = lbl

    # Count per parcel.
    for c in range(1, n_comp + 1):
        lbl = comp_to_label[c]
        if lbl > label_offset and lbl <= label_offset + n_parcels:
            count_out[lbl - label_offset - 1] += 1.0

    # NaN-fill empty parcels.
    for p in range(n_parcels):
        if not parcel_seen[p]:
            count_out[p] = np.nan


# ─────────────────────────────────────────────────────────────────────────
# 4. Fused per-hemisphere distance computation (fp32 distance arithmetic)
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=False)
def fused_hemi_distance(
    vertex_nbors, vertices_xyz_T, labels_zeroed, ci_full, num_parcels_hemi,
    labels_bnd, ci_masked,
    parcel_size, start, cursor,
    bnd_x, bnd_y, bnd_z, bnd_ci,
    local, uniq, comp_min_sq,
    out_eucli,
):
    """Fused per-hemi distance — SENTINELED-LABEL contract.

    Step 1 (inlined boundary detector) uses a different contract from the
    standalone ``boundary`` kernel above: it mirrors the *internal*
    composition in MATLAB ``component_distance.m`` lines 60-65, which
    substitutes medial-wall labels with the sentinel ``num_parcel + 1``
    BEFORE calling BuildTwoVertThickBoundary. Here ``fake = num_parcel + 1``
    is applied at neighbor-lookup time so a medial / single-comp neighbor
    (label 0 in ``labels_zeroed``) DOES contribute to ``diff_sum`` — that
    contribution is what marks distributed-parcel boundary verts adjacent
    to single-comp parcels as boundary. The standalone ``boundary`` kernel
    works on raw labels and ``continue``s on medial neighbors, so its
    output differs at exactly those verts. Do NOT try to deduplicate the
    two boundary impls — see the standalone's docstring for the rationale.
    """
    max_neigh, N = vertex_nbors.shape
    fake = num_parcels_hemi * 2 + 1

    # ----- Step 1: boundary detection + ci masking + labels_bnd construction.
    for v in range(N):
        own_zeroed = labels_zeroed[v]
        own_tmp = own_zeroed if own_zeroed != 0 else fake
        diff_sum = 0
        for k in range(max_neigh):
            nb = vertex_nbors[k, v]
            if nb == 0:
                continue
            nl_zeroed = labels_zeroed[nb - 1]
            nl_tmp = nl_zeroed if nl_zeroed != 0 else fake
            diff_sum += (nl_tmp - own_tmp)
        is_boundary = diff_sum != 0
        if is_boundary and own_zeroed != 0:
            labels_bnd[v] = own_zeroed
        else:
            labels_bnd[v] = 0
        ci_masked[v] = ci_full[v] if own_zeroed != 0 else 0

    # ----- Step 2: zero outputs and parcel buckets.
    for p in range(num_parcels_hemi):
        out_eucli[p] = np.float32(0.0)
    n_buckets = num_parcels_hemi + 2
    for p in range(n_buckets):
        parcel_size[p] = 0
        start[p] = 0

    # ----- Step 3: count boundary verts per parcel; build start[] offsets.
    for v in range(N):
        p = labels_bnd[v]
        if p > 0:
            parcel_size[p] += 1
    for p in range(1, n_buckets):
        start[p] = start[p - 1] + parcel_size[p - 1]

    # ----- Step 4: bucket-sort boundary verts into bnd_x/y/z/ci by parcel.
    for p in range(n_buckets):
        cursor[p] = start[p]
    for v in range(N):
        p = labels_bnd[v]
        if p > 0:
            i = cursor[p]
            bnd_x[i] = np.float32(vertices_xyz_T[v, 0])
            bnd_y[i] = np.float32(vertices_xyz_T[v, 1])
            bnd_z[i] = np.float32(vertices_xyz_T[v, 2])
            bnd_ci[i] = ci_masked[v]
            cursor[p] = i + 1

    # ----- Step 5: per-parcel inner loop. For each parcel:
    #   compress ci → local indices; track per-component min² over (i,j) pairs
    #   in different components; eucli[p-1] = sqrt(max over components).
    INF32 = np.float32(1e30)
    for p in range(1, num_parcels_hemi + 1):
        s_lo = start[p]
        s_hi = start[p + 1]
        size = s_hi - s_lo
        if size < 2:
            continue

        # Compress ci → local 0-based indices for this parcel.
        n_uniq = 0
        for i in range(size):
            c = bnd_ci[s_lo + i]
            slot = -1
            for k in range(n_uniq):
                if uniq[k] == c:
                    slot = k
                    break
            if slot < 0:
                slot = n_uniq
                uniq[n_uniq] = c
                n_uniq += 1
            local[i] = slot

        if n_uniq < 2:
            continue

        for k in range(n_uniq):
            comp_min_sq[k] = INF32

        for i in range(size):
            li = local[i]
            xi = bnd_x[s_lo + i]
            yi = bnd_y[s_lo + i]
            zi = bnd_z[s_lo + i]
            for j in range(i + 1, size):
                lj = local[j]
                if li == lj:
                    continue
                dx = xi - bnd_x[s_lo + j]
                dy = yi - bnd_y[s_lo + j]
                dz = zi - bnd_z[s_lo + j]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < comp_min_sq[li]:
                    comp_min_sq[li] = d2
                if d2 < comp_min_sq[lj]:
                    comp_min_sq[lj] = d2

        max_min = np.float32(0.0)
        for k in range(n_uniq):
            v_ = np.sqrt(comp_min_sq[k])
            if v_ > max_min:
                max_min = v_
        out_eucli[p - 1] = max_min


# ─────────────────────────────────────────────────────────────────────────
# 5. Zero single-component parcels in-place (LH path).
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=False)
def zero_singles_lh(labels_in, is_single, out):
    """For each vertex, write 0 if its parcel has exactly one component, else
    keep the label. ``is_single`` must be size num_parcel+1 with index 0 == False.

    Sizes: labels_in,out (N,) int64; is_single (num_parcel+1,) bool.
    """
    N = labels_in.shape[0]
    for v in range(N):
        lbl = labels_in[v]
        if is_single[lbl]:
            out[v] = 0
        else:
            out[v] = lbl


# ─────────────────────────────────────────────────────────────────────────
# 6. Zero single-component parcels AND re-index RH labels to [1, n_lh].
# ─────────────────────────────────────────────────────────────────────────
@njit(cache=True, nogil=True, fastmath=False)
def zero_singles_and_reindex_rh(labels_in, is_single, n_lh, out):
    """RH variant: zero single-comp parcels (and medial wall) AND subtract n_lh
    from kept labels so RH parcel ids land in [1, n_lh] for the per-hemi kernel.

    Sizes: labels_in,out (N,) int64; is_single (num_parcel+1,) bool.
    """
    N = labels_in.shape[0]
    for v in range(N):
        lbl = labels_in[v]
        if lbl == 0 or is_single[lbl]:
            out[v] = 0
        else:
            out[v] = lbl - n_lh


# ─────────────────────────────────────────────────────────────────────────
# Warmup helper — compile all kernels with tiny dummy buffers. Idempotent.
# After this returns, the @njit dispatchers have a cached compiled version
# matching (int64, int64, int64, int64, int8, int64, int64, int64) etc.
# ─────────────────────────────────────────────────────────────────────────
def warmup() -> None:
    src = np.array([0, 1], dtype=np.int64)
    dst = np.array([1, 2], dtype=np.int64)
    labels = np.array([1, 1, 1], dtype=np.int64)
    parent = np.empty(3, dtype=np.int64)
    rank = np.empty(3, dtype=np.int8)
    comp_map = np.empty(3, dtype=np.int64)
    ci = np.empty(3, dtype=np.int64)
    n_comp_out = np.empty(1, dtype=np.int64)
    cc_from_edges(src, dst, labels, parent, rank, comp_map, ci, n_comp_out)

    nbors = np.array([[2, 3, 0], [1, 3, 0], [1, 2, 0]], dtype=np.int64).T
    out = np.empty(3, dtype=np.int64)
    boundary(nbors, labels, out)

    comp_to_label = np.empty(int(n_comp_out[0]) + 1, dtype=np.int64)
    parcel_seen = np.empty(2, dtype=np.bool_)
    count_out = np.empty(2, dtype=np.float64)
    count_per_parcel(labels, ci, int(n_comp_out[0]), 2, 0,
                     comp_to_label, parcel_seen, count_out)

    xyz = np.zeros((3, 3), dtype=np.float32)
    labels_bnd = np.empty(3, dtype=np.int64)
    ci_masked = np.empty(3, dtype=np.int64)
    parcel_size = np.empty(4, dtype=np.int64)
    start = np.empty(4, dtype=np.int64)
    cursor = np.empty(4, dtype=np.int64)
    bnd_x = np.empty(3, dtype=np.float32)
    bnd_y = np.empty(3, dtype=np.float32)
    bnd_z = np.empty(3, dtype=np.float32)
    bnd_ci = np.empty(3, dtype=np.int64)
    local = np.empty(3, dtype=np.int64)
    uniq = np.empty(3, dtype=np.int64)
    comp_min_sq = np.empty(3, dtype=np.float32)
    out_eucli = np.empty(2, dtype=np.float32)
    fused_hemi_distance(
        nbors, xyz, labels, ci, 2,
        labels_bnd, ci_masked,
        parcel_size, start, cursor,
        bnd_x, bnd_y, bnd_z, bnd_ci,
        local, uniq, comp_min_sq,
        out_eucli,
    )

    is_single = np.array([False, True, False], dtype=np.bool_)
    out_lbl = np.empty(3, dtype=np.int64)
    zero_singles_lh(labels, is_single, out_lbl)
    zero_singles_and_reindex_rh(labels, is_single, 1, out_lbl)
