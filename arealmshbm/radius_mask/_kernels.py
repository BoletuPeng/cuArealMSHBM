"""_kernels.py

Numba kernels for the radius_mask supercall. All compute lives here.
Six kernels — each appears exactly once.

    _hp_push / _hp_pop
        Binary min-heap on parallel arrays (fp32 keys = distance,
        int32 values = vertex idx). Inlined into the Dijkstra kernels.

    build_parcel_csr_kernel(labels, L) -> (offs, inds)
        Group vertex ids by parcel label. CSR layout consumed by the
        parcel-iterating kernels below.

    classify_central_relevance_kernel(labels, aparc_orig, L_h)
        -> (relevant_parcels, relevant_verts)
        Identify parcels whose ``avg_dis`` is consumed by the truncate
        step (paracentral / insula sets). Drives early termination of
        the central_sulcus Dijkstras.

    add_spatial_constraint_kernel(csr, labels, parcel_csr, L, radius, OUT mask)
        For each parcel l, fused boundary detection + bounded multi-
        source Dijkstra (heap seeded from boundary verts, terminate
        at d > radius) + out-mask write. ``prange`` over parcels.

    central_sulcus_kernel(csr, pre, post, parcel_csr, L,
                          relevant_parcels, relevant_verts, OUT avg)
        For each src ∈ pre ∪ post, run single-source Dijkstra with
        early termination once all relevant verts settle, accumulate
        distances at relevant verts inline. ``prange`` over sources.

    truncate_kernel(boundary, avg_dis, labels, aparc_orig)
        Aparc relabel (4→19, 29→19, 32→30) + parcel-aparc overlap →
        central-bin classification + per-set keep-mask application,
        all in one pass.

Layout / precision:
    * Edge weights, Dijkstra distances, vertex coords post-MARS — fp32.
    * Per-parcel mean accumulator (avg_dis) — fp64.
    * Heap sized to E = indptr[V] (upper bound on lazy-deletion pushes).
    * Per-thread scratch (heap, dist, touched, accum) preallocated
      inside each kernel.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import numba
from numba import njit, prange


# ─────────────────────────────────────────────────────────────────────
# Binary min-heap (parallel arrays)
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def _hp_push(keys, vals, n, key, val):
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


@njit(cache=True, inline="always")
def _hp_pop(keys, vals, n):
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


# ─────────────────────────────────────────────────────────────────────
# Parcel→vertices CSR build
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True)
def build_parcel_csr_kernel(labels, L):
    """Group vertex ids by parcel label.

    Parameters
    ----------
    labels : (V,) int64 — parcel ids 1..L; 0 = medial wall (skipped).
    L      : int        — number of parcels.

    Returns
    -------
    parcel_offs : (L+1,) int64 — offsets into parcel_inds.
    parcel_inds : (sum_l |P_l|,) int64 — vertex ids grouped by parcel.
    """
    V = labels.shape[0]
    counts = np.zeros(L, dtype=np.int64)
    for v in range(V):
        l = labels[v]
        if l > 0:
            counts[l - 1] += 1
    offs = np.zeros(L + 1, dtype=np.int64)
    for i in range(L):
        offs[i + 1] = offs[i] + counts[i]
    inds = np.empty(offs[L], dtype=np.int64)
    cursor = np.zeros(L, dtype=np.int64)
    for i in range(L):
        cursor[i] = offs[i]
    for v in range(V):
        l = labels[v]
        if l > 0:
            inds[cursor[l - 1]] = v
            cursor[l - 1] += 1
    return offs, inds


# ─────────────────────────────────────────────────────────────────────
# add_spatial_constraint — bounded multi-src Dijkstra per parcel
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, parallel=True)
def add_spatial_constraint_kernel(
    indptr, indices, weights,
    labels,
    parcel_offs, parcel_inds,
    L,
    radius,
    out_mask,
):
    """Per-parcel boundary mask via bounded multi-source Dijkstra.

    For each parcel l in 1..L: mark every parcel vertex; identify
    boundary verts (parcel verts with at least one non-parcel neighbor)
    and seed the heap with d=0; run bounded Dijkstra terminating when
    popped d > radius; mark every reached vertex.

    Inputs:
        indptr / indices / weights : CSR mesh adjacency (fp32 weights).
        labels                     : (V,) int64 — parcel ids 1..L; 0=MW.
        parcel_offs, parcel_inds   : parcel→verts CSR.
        L, radius                  : int / fp32 (mm).
        out_mask                   : (V, L) uint8 — pre-zeroed, written.
    """
    V = labels.shape[0]
    E = indptr[V]
    T = numba.get_num_threads()
    radius_f = np.float32(radius)

    dist_T = np.full((T, V), np.float32(1e30), dtype=np.float32)
    hk_T = np.empty((T, E), dtype=np.float32)
    hv_T = np.empty((T, E), dtype=np.int32)
    touched_T = np.empty((T, V), dtype=np.int32)

    for l_idx in prange(L):
        tid = numba.get_thread_id()
        l = l_idx + 1
        dist = dist_T[tid]
        hk = hk_T[tid]
        hv = hv_T[tid]
        touched = touched_T[tid]
        n_touched = 0
        n_heap = 0

        ps = parcel_offs[l_idx]
        pe = parcel_offs[l_idx + 1]

        # Init pass: mark every parcel vertex; identify boundary verts;
        # seed Dijkstra with dist=0 from each boundary vert.
        for k_p in range(ps, pe):
            v = parcel_inds[k_p]
            out_mask[v, l_idx] = 1
            is_b = False
            for k_e in range(indptr[v], indptr[v + 1]):
                u = indices[k_e]
                if labels[u] != l:
                    is_b = True
                    break
            if is_b:
                if dist[v] >= np.float32(1e30):
                    touched[n_touched] = v
                    n_touched += 1
                dist[v] = np.float32(0.0)
                n_heap = _hp_push(hk, hv, n_heap,
                                   np.float32(0.0), np.int32(v))

        # Bounded Dijkstra
        while n_heap > 0:
            d, v_i32, n_heap = _hp_pop(hk, hv, n_heap)
            v = v_i32
            if d > dist[v]:
                continue
            out_mask[v, l_idx] = 1
            for k_e in range(indptr[v], indptr[v + 1]):
                u = indices[k_e]
                w = weights[k_e]
                nd = d + w
                if nd <= radius_f and nd < dist[u]:
                    if dist[u] >= np.float32(1e30):
                        touched[n_touched] = u
                        n_touched += 1
                    dist[u] = nd
                    n_heap = _hp_push(hk, hv, n_heap, nd, np.int32(u))

        # Reset only touched entries (cheaper than V-wide fill for small parcels).
        for k in range(n_touched):
            dist[touched[k]] = np.float32(1e30)


# ─────────────────────────────────────────────────────────────────────
# Relevance classifier — paracentral / insula parcel-set membership
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True)
def classify_central_relevance_kernel(labels, aparc_orig, L_h):
    """Identify parcels whose ``avg_dis`` is actually consumed by
    truncate (paracentral / insula sets only). Drives early-termination
    of central_sulcus_kernel.

    Re-implements the overlap → central → central_bin pipeline from
    truncate_kernel up to the binarization step (the duplication is
    intentional — it keeps both kernels self-contained and runs in
    ~1 ms).

    Returns:
        relevant_parcels : (L_h,) uint8 — 1 if avg_dis is consumed.
        relevant_verts   : (V,)  uint8 — 1 if vertex is in a relevant parcel.
    """
    V = labels.shape[0]
    aparc = np.empty(V, dtype=np.int64)
    aparc_max_v = 0
    for v in range(V):
        a = aparc_orig[v]
        if a == 4 or a == 29:
            a = 19
        elif a == 32:
            a = 30
        aparc[v] = a
        if a > aparc_max_v:
            aparc_max_v = a
    aparc_max = aparc_max_v

    overlap = np.zeros((L_h, aparc_max), dtype=np.float64)
    for v in range(V):
        l = labels[v]
        a = aparc[v]
        if l >= 1 and a >= 1:
            overlap[l - 1, a - 1] += 1.0
    for l in range(L_h):
        rs = 0.0
        for a in range(aparc_max):
            rs += overlap[l, a]
        if rs > 0.0:
            inv = 1.0 / rs
            for a in range(aparc_max):
                overlap[l, a] *= inv

    col_idx_arr = np.empty(6, dtype=np.int64)
    col_idx_arr[0] = 17   # paracentral
    col_idx_arr[1] = 22   # postcentral
    col_idx_arr[2] = 24   # precentral
    col_idx_arr[3] = 35   # insula
    col_idx_arr[4] = 18   # sf/cmf/po
    col_idx_arr[5] = 29   # sp/sm
    central = np.zeros((L_h, 6), dtype=np.float64)
    for l in range(L_h):
        for c in range(6):
            ci = col_idx_arr[c]
            if ci < aparc_max:
                central[l, c] = overlap[l, ci]

    for l in range(L_h):
        rm = 0.0
        for c in range(6):
            x = central[l, c]
            if x == x and x > rm:
                rm = x
        if rm > 0.0:
            inv = 1.0 / rm
            for c in range(6):
                x = central[l, c]
                if x == x:
                    central[l, c] = x * inv
                else:
                    central[l, c] = 0.0
        else:
            for c in range(6):
                central[l, c] = 0.0

    for l in range(L_h):
        c1 = central[l, 1]
        c2 = central[l, 2]
        if c1 == 0.0 and c2 != 0.0:
            central[l, 2] = 1.0
        if c1 != 0.0 and c2 == 0.0:
            central[l, 1] = 1.0

    relevant_parcels = np.zeros(L_h, dtype=np.uint8)
    for l in range(L_h):
        if central[l, 0] >= 0.999 or central[l, 3] >= 0.999:
            relevant_parcels[l] = 1

    relevant_verts = np.zeros(V, dtype=np.uint8)
    for v in range(V):
        l = labels[v]
        if l > 0 and relevant_parcels[l - 1] == 1:
            relevant_verts[v] = 1

    return relevant_parcels, relevant_verts


# ─────────────────────────────────────────────────────────────────────
# central_sulcus — per-source Dijkstra with early termination + accum
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, parallel=True)
def central_sulcus_kernel(
    indptr, indices, weights,
    pre_verts,
    post_verts,
    parcel_offs, parcel_inds,
    L,
    relevant_parcels,
    relevant_verts,
    out_avg,
):
    """Per-parcel mean geodesic distance to {pre, post}-central verts.

    For each src in pre_verts ∪ post_verts: run single-source Dijkstra,
    accumulate distance at relevant verts inline (per-thread), early-
    terminate once all ``relevant_verts`` have been popped (their final
    distance is fixed at pop time). Reduce per-thread accumulators →
    per-relevant-parcel mean.

    For each relevant parcel l, ``out_avg[h, l] = mean over (s, v) of
    distance(s, v)`` where s ∈ pre/post_verts (h=0/1) and v ∈ verts of
    parcel l. Non-relevant parcel entries are zeroed (``truncate``
    ignores them; see classify_central_relevance_kernel).

    Inputs:
        indptr / indices / weights : CSR mesh adjacency (fp32 weights).
        pre_verts, post_verts      : (Npre,), (Npost,) int64 source verts.
        parcel_offs, parcel_inds   : parcel→verts CSR.
        L                          : int.
        relevant_parcels, relevant_verts : from classify_central_relevance_kernel.
        out_avg                    : (2, L) fp64 — row 0 = pre, row 1 = post.
    """
    V = indptr.shape[0] - 1
    E = indptr[V]
    T = numba.get_num_threads()
    Npre = pre_verts.shape[0]
    Npost = post_verts.shape[0]
    Nsrc = Npre + Npost

    n_relevant_total = 0
    for v in range(V):
        if relevant_verts[v] == 1:
            n_relevant_total += 1

    dist_T = np.full((T, V), np.float32(1e30), dtype=np.float32)
    hk_T = np.empty((T, E), dtype=np.float32)
    hv_T = np.empty((T, E), dtype=np.int32)
    touched_T = np.empty((T, V), dtype=np.int32)
    accum_pre_T = np.zeros((T, V), dtype=np.float64)
    accum_post_T = np.zeros((T, V), dtype=np.float64)

    for s_idx in prange(Nsrc):
        tid = numba.get_thread_id()
        if s_idx < Npre:
            src = pre_verts[s_idx]
            is_pre = True
        else:
            src = post_verts[s_idx - Npre]
            is_pre = False

        dist = dist_T[tid]
        hk = hk_T[tid]
        hv = hv_T[tid]
        touched = touched_T[tid]
        n_touched = 0

        dist[src] = np.float32(0.0)
        touched[n_touched] = src
        n_touched += 1
        n_heap = _hp_push(hk, hv, 0, np.float32(0.0), np.int32(src))
        n_relevant_settled = 0

        while n_heap > 0:
            d, v_i32, n_heap = _hp_pop(hk, hv, n_heap)
            v = v_i32
            if d > dist[v]:
                continue
            if relevant_verts[v] == 1:
                n_relevant_settled += 1
                if is_pre:
                    accum_pre_T[tid, v] += np.float64(d)
                else:
                    accum_post_T[tid, v] += np.float64(d)
                if n_relevant_settled >= n_relevant_total:
                    break
            for k_e in range(indptr[v], indptr[v + 1]):
                u = indices[k_e]
                w = weights[k_e]
                nd = d + w
                if nd < dist[u]:
                    if dist[u] >= np.float32(1e30):
                        touched[n_touched] = u
                        n_touched += 1
                    dist[u] = nd
                    n_heap = _hp_push(hk, hv, n_heap, nd, np.int32(u))

        # Reset only the touched verts (O(touched), not O(V)).
        for k in range(n_touched):
            dist[touched[k]] = np.float32(1e30)

    # Reduce per-thread accumulators (only relevant verts have non-zero
    # entries; iterate all V to keep the kernel simple — V is 40k, cheap).
    accum_pre = np.zeros(V, dtype=np.float64)
    accum_post = np.zeros(V, dtype=np.float64)
    for v in range(V):
        if relevant_verts[v] == 0:
            continue
        sp = 0.0
        spo = 0.0
        for t in range(T):
            sp += accum_pre_T[t, v]
            spo += accum_post_T[t, v]
        accum_pre[v] = sp
        accum_post[v] = spo

    Npre_f = np.float64(Npre)
    Npost_f = np.float64(Npost)
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
            v = parcel_inds[k]
            sp_l += accum_pre[v]
            spo_l += accum_post[v]
        out_avg[0, l_idx] = sp_l / (np.float64(n_v) * Npre_f)
        out_avg[1, l_idx] = spo_l / (np.float64(n_v) * Npost_f)


# ─────────────────────────────────────────────────────────────────────
# truncate — anatomical correction via aparc-overlap classification
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True)
def truncate_kernel(
    boundary,
    avg_dis,
    labels,
    aparc_orig,
):
    """Anatomical truncation of per-parcel boundary masks.

    For each of 6 anatomical parcel sets (paracentral / postcentral /
    precentral / insula / sf-cmf-po / sp-sm), compute a 6-column
    keep-mask and zero aparc-bin regions of the boundary for parcels
    in that set whose keep-mask column is 0. avg_dis enters as the
    pre-vs-post tiebreaker for the paracentral and insula sets.

    Aparc relabel: 4 → 19, 29 → 19, 32 → 30.

    Inputs:
        boundary   : (V, L_h) uint8 — INOUT, mutated.
        avg_dis    : (2, L_h) fp64 — row 0 = pre dis, row 1 = post dis.
        labels     : (V,) int64.
        aparc_orig : (V,) int64 — DK aparc, 1-indexed.
    """
    V = labels.shape[0]
    L_h = boundary.shape[1]

    # ── Step 1: relabel aparc ──
    aparc = np.empty(V, dtype=np.int64)
    aparc_max_v = 0
    for v in range(V):
        a = aparc_orig[v]
        if a == 4 or a == 29:
            a = 19
        elif a == 32:
            a = 30
        aparc[v] = a
        if a > aparc_max_v:
            aparc_max_v = a
    aparc_max = aparc_max_v

    # ── Step 2: parcel-aparc overlap, row-normalized ──
    overlap = np.zeros((L_h, aparc_max), dtype=np.float64)
    for v in range(V):
        l = labels[v]
        a = aparc[v]
        if l >= 1 and a >= 1:
            overlap[l - 1, a - 1] += 1.0
    for l in range(L_h):
        rs = 0.0
        for a in range(aparc_max):
            rs += overlap[l, a]
        if rs > 0.0:
            inv = 1.0 / rs
            for a in range(aparc_max):
                overlap[l, a] *= inv

    # ── Step 3: extract `central` (L_h, 6) at columns [18,23,25,36,19,30] ──
    col_idx_arr = np.empty(6, dtype=np.int64)
    col_idx_arr[0] = 17   # paracentral (18 - 1)
    col_idx_arr[1] = 22   # postcentral (23 - 1)
    col_idx_arr[2] = 24   # precentral  (25 - 1)
    col_idx_arr[3] = 35   # insula      (36 - 1)
    col_idx_arr[4] = 18   # sf/cmf/po   (19 - 1)
    col_idx_arr[5] = 29   # sp/sm       (30 - 1)
    central = np.zeros((L_h, 6), dtype=np.float64)
    for l in range(L_h):
        for c in range(6):
            ci = col_idx_arr[c]
            if ci < aparc_max:
                central[l, c] = overlap[l, ci]

    # ── Step 4: row-normalize central by row-max (NaN-safe) ──
    for l in range(L_h):
        rm = 0.0
        for c in range(6):
            x = central[l, c]
            if x == x and x > rm:
                rm = x
        if rm > 0.0:
            inv = 1.0 / rm
            for c in range(6):
                x = central[l, c]
                if x == x:
                    central[l, c] = x * inv
                else:
                    central[l, c] = 0.0
        else:
            for c in range(6):
                central[l, c] = 0.0

    # ── Step 5: manual fixes (lines 223-224 / 230-231) ──
    for l in range(L_h):
        c1 = central[l, 1]
        c2 = central[l, 2]
        if c1 == 0.0 and c2 != 0.0:
            central[l, 2] = 1.0
        if c1 != 0.0 and c2 == 0.0:
            central[l, 1] = 1.0

    # ── Step 6: binarize central_bin ──
    central_bin = np.zeros((L_h, 6), dtype=np.uint8)
    for l in range(L_h):
        for c in range(6):
            if central[l, c] >= 0.999:
                central_bin[l, c] = 1

    aparc_bin_per_col_arr = np.empty(6, dtype=np.int64)
    aparc_bin_per_col_arr[0] = 18
    aparc_bin_per_col_arr[1] = 23
    aparc_bin_per_col_arr[2] = 25
    aparc_bin_per_col_arr[3] = 36
    aparc_bin_per_col_arr[4] = 19
    aparc_bin_per_col_arr[5] = 30

    # ── Step 7: per-set keep-mask, apply to boundary ──
    keep_l = np.zeros(6, dtype=np.uint8)
    for i in range(6):
        for l in range(L_h):
            if central_bin[l, i] != 1:
                continue
            for c in range(6):
                keep_l[c] = 0
            if i == 0 or i == 3:
                # paracentral or insula — t = (central > 0.2); force [4,5]=1;
                #   pre/post side flip.
                for c in range(6):
                    if central[l, c] > 0.2:
                        keep_l[c] = 1
                keep_l[4] = 1
                keep_l[5] = 1
                pre_dis = avg_dis[0, l]
                post_dis = avg_dis[1, l]
                pre_or_post = (pre_dis - post_dis) < 0.0
                if pre_or_post:
                    keep_l[1] = 0
                    keep_l[2] = 1
                else:
                    keep_l[1] = 1
                    keep_l[2] = 0
            elif i == 1:
                # postcentral — t = (central >= 0.999); force [0,3,5]=1.
                for c in range(6):
                    if central[l, c] >= 0.999:
                        keep_l[c] = 1
                keep_l[0] = 1
                keep_l[3] = 1
                keep_l[5] = 1
            elif i == 2:
                # precentral — t = (central >= 0.999); force [0,3,4]=1.
                for c in range(6):
                    if central[l, c] >= 0.999:
                        keep_l[c] = 1
                keep_l[0] = 1
                keep_l[3] = 1
                keep_l[4] = 1
            elif i == 4:
                # sf/cmf/po — keep all, zero col 1 (postcentral).
                for c in range(6):
                    keep_l[c] = 1
                keep_l[1] = 0
            else:
                # i == 5: sp/sm — keep all, zero col 2 (precentral).
                for c in range(6):
                    keep_l[c] = 1
                keep_l[2] = 0

            for c in range(6):
                if keep_l[c] == 0:
                    bin_id = aparc_bin_per_col_arr[c]
                    for v in range(V):
                        if aparc[v] == bin_id:
                            boundary[v, l] = 0
