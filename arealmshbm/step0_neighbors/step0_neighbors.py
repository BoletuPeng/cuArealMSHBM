"""step0_neighbors.py

Public entry points: ``neighbors_exclude_medial`` and
``find_K_neighbors``. Numba @njit ports of the corresponding CBIG
SPGrad utilities.

Index conventions
-----------------
Both functions use **1-indexed** vertex IDs throughout, with **NaN**
denoting "absent slot" (boundary, removed-medial, or pad). This
matches MATLAB's CBIG-SPGrad convention so the output is identical
to ``neighbors_table.npy`` / ``K_neighbors.npy`` byte-for-byte.

Shape conventions (Python row-major, transposed from MATLAB)
------------------------------------------------------------
- Input ``orig_neighbors`` : (N, K+1) fp64
- Input ``medial_mask`` : (N,) bool
- Output of ``neighbors_exclude_medial`` : (N_cortex, K+1) fp64
- Output of ``find_K_neighbors``: (N_cortex, P_final) fp64

Note on the sink-vertex trick (port of the MATLAB algorithm)
------------------------------------------------------------
``CBIG_SPGrad_find_neighbors`` appends a synthetic sink row at index
N+1 whose entries are all "N+1". All NaN entries are first replaced
by N+1, the iterative expansion runs, and at the end we drop the
appended row and turn N+1 entries back to NaN.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# helpers (also @njit)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sort_rows(arr: np.ndarray) -> np.ndarray:
    """In-place per-row ascending sort. ``np.sort(arr, axis=1)`` is not
    supported in numba's nopython mode, so we sort each row via the 1D
    overload."""
    for i in range(arr.shape[0]):
        arr[i, :] = np.sort(arr[i, :])
    return arr


@njit(cache=True)
def _drop_all_sentinel_cols(arr: np.ndarray, sentinel: float) -> np.ndarray:
    """Drop columns where every row equals ``sentinel``.

    Because arrays here are sorted ascending and sentinel is the max
    value, "min(col) == sentinel" iff "every entry in col == sentinel".
    """
    n_rows, n_cols = arr.shape
    if n_cols == 0:
        return arr
    keep = np.empty(n_cols, dtype=np.bool_)
    n_keep = 0
    for j in range(n_cols):
        col_min = arr[0, j]
        for i in range(1, n_rows):
            v = arr[i, j]
            if v < col_min:
                col_min = v
        keep[j] = col_min != sentinel
        if keep[j]:
            n_keep += 1
    out = np.empty((n_rows, n_keep), dtype=arr.dtype)
    k = 0
    for j in range(n_cols):
        if keep[j]:
            for i in range(n_rows):
                out[i, k] = arr[i, j]
            k += 1
    return out


@njit(cache=True, inline="always")
def _searchsorted_right(sorted_arr: np.ndarray, v: float) -> int:
    """Manual ``np.searchsorted(sorted_arr, v, side='right')``.

    Returns insertion index such that everything left is <= v.
    """
    lo = 0
    hi = sorted_arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if sorted_arr[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, inline="always")
def _is_member_sorted(sorted_arr: np.ndarray, v: float) -> bool:
    """Branchless-ish binary search membership test (np.isin scalar)."""
    lo = 0
    hi = sorted_arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        m = sorted_arr[mid]
        if m == v:
            return True
        if m < v:
            lo = mid + 1
        else:
            hi = mid
    return False


# ---------------------------------------------------------------------------
# neighbors_exclude_medial — port of CBIG_SPGrad_neighbors_exclude_medial.m
# ---------------------------------------------------------------------------

@njit(cache=True)
def neighbors_exclude_medial(orig_neighbors: np.ndarray,
                             medial_mask: np.ndarray) -> np.ndarray:
    """Drop medial-wall vertices and reindex remaining IDs contiguously."""
    n = orig_neighbors.shape[0]
    cols = orig_neighbors.shape[1]

    # Build 1-indexed sorted medial set and count.
    n_med = 0
    for i in range(n):
        if medial_mask[i]:
            n_med += 1
    medial_set = np.empty(n_med, dtype=np.float64)
    k = 0
    for i in range(n):
        if medial_mask[i]:
            medial_set[k] = float(i + 1)
            k += 1
    # medial_set is naturally sorted ascending (i increases).

    # Drop rows where mask==1.
    n_cortex = n - n_med
    out = np.empty((n_cortex, cols), dtype=np.float64)
    r = 0
    for i in range(n):
        if not medial_mask[i]:
            for j in range(cols):
                out[r, j] = orig_neighbors[i, j]
            r += 1

    # For each entry: NaN-out medial members, else decrement by
    # number of medial IDs strictly less than the value (searchsorted
    # 'right' on a strictly-increasing medial_set).
    for i in range(n_cortex):
        for j in range(cols):
            v = out[i, j]
            if np.isnan(v):
                continue
            if n_med > 0 and _is_member_sorted(medial_set, v):
                out[i, j] = np.nan
            else:
                if n_med > 0:
                    shift = _searchsorted_right(medial_set, v)
                    out[i, j] = v - float(shift)
    return out


# ---------------------------------------------------------------------------
# find_K_neighbors — port of CBIG_SPGrad_find_neighbors.m
# ---------------------------------------------------------------------------

@njit(cache=True)
def find_K_neighbors(neighbors: np.ndarray, K: int) -> np.ndarray:
    """K-hop neighbor expansion with NaN sentinel padding."""
    if K < 1:
        raise ValueError("K must be >= 1")

    N = neighbors.shape[0]
    cols = neighbors.shape[1]
    sentinel = float(N + 1)
    Nplus = N + 1
    ring = cols - 1  # ring-1 width (drop self col)

    # Build (N+1, cols) nbr with NaNs replaced by sentinel and a sink
    # row of all sentinels appended.
    nbr = np.empty((Nplus, cols), dtype=np.float64)
    for i in range(N):
        for j in range(cols):
            v = neighbors[i, j]
            nbr[i, j] = sentinel if np.isnan(v) else v
    for j in range(cols):
        nbr[N, j] = sentinel

    # K_neighbors initial = nbr[:, 1:]
    K_neighbors = nbr[:, 1:].copy()

    if K > 1:
        curr_K_neighbors = _sort_rows(nbr[:, 1:].copy())

        curr_K = 1
        while curr_K != K:
            num_neigh = curr_K_neighbors.shape[1]
            tmp_curr_K_neighbors = np.empty((Nplus, 0), dtype=np.float64)

            for nn in range(num_neigh):
                # curr_neigh[i, :] = nbr[curr_K_neighbors[i,nn]-1, 1:]
                # (sentinel rows index the sink at row N).
                # The algorithmic invariant is curr_K_neighbors ∈ [1, N+1]
                # → idx0 ∈ [0, N] which is in range for nbr of shape
                # (N+1, cols). Any violation indicates an upstream
                # corruption; surface it loudly rather than clipping into
                # an arbitrary row.
                curr_neigh = np.empty((Nplus, ring), dtype=np.float64)
                for i in range(Nplus):
                    idx0 = int(curr_K_neighbors[i, nn]) - 1
                    if idx0 < 0 or idx0 > N:
                        raise ValueError(
                            "find_K_neighbors: neighbor index out of range "
                            "[1, N+1] — upstream neighbor table corrupted."
                        )
                    curr_neigh[i, :] = nbr[idx0, 1:]

                # Mask entries already in K_neighbors → sentinel.
                Kc = K_neighbors.shape[1]
                for i in range(Nplus):
                    for j in range(ring):
                        v = curr_neigh[i, j]
                        if v == sentinel:
                            continue
                        for c in range(Kc):
                            if K_neighbors[i, c] == v:
                                curr_neigh[i, j] = sentinel
                                break
                curr_neigh = _drop_all_sentinel_cols(_sort_rows(curr_neigh), sentinel)

                if nn == 0:
                    tmp_curr_K_neighbors = curr_neigh.copy()
                else:
                    Tc = tmp_curr_K_neighbors.shape[1]
                    cnc = curr_neigh.shape[1]
                    for i in range(Nplus):
                        for j in range(cnc):
                            v = curr_neigh[i, j]
                            if v == sentinel:
                                continue
                            for c in range(Tc):
                                if tmp_curr_K_neighbors[i, c] == v:
                                    curr_neigh[i, j] = sentinel
                                    break
                    curr_neigh = _drop_all_sentinel_cols(_sort_rows(curr_neigh),
                                                        sentinel)
                    tmp_curr_K_neighbors = np.concatenate(
                        (tmp_curr_K_neighbors, curr_neigh), axis=1)

                # K_neighbors = sort(hstack([K_neighbors, curr_neigh]), axis=1)
                merged = np.concatenate((K_neighbors, curr_neigh), axis=1)
                K_neighbors = _drop_all_sentinel_cols(_sort_rows(merged), sentinel)

            curr_K_neighbors = tmp_curr_K_neighbors
            curr_K += 1

    # Exclude self.
    Kc = K_neighbors.shape[1]
    for i in range(Nplus):
        s = nbr[i, 0]
        for j in range(Kc):
            if K_neighbors[i, j] == s:
                K_neighbors[i, j] = sentinel
    K_neighbors = _drop_all_sentinel_cols(_sort_rows(K_neighbors), sentinel)

    # Prepend self col, replace remaining sentinels with NaN, drop sink.
    Kc = K_neighbors.shape[1]
    final = np.empty((N, 1 + Kc), dtype=np.float64)
    for i in range(N):
        final[i, 0] = nbr[i, 0]
        for j in range(Kc):
            v = K_neighbors[i, j]
            final[i, 1 + j] = np.nan if v == sentinel else v
    return final
