"""remove_isolated.py

Relabel small isolated connected components of a surface parcellation by
mode-vote of their neighbours.

For each connected component of vertices sharing a parcel label, if the
component has fewer than ``abs_threshold`` vertices, reassign those
vertices to the mode of their neighbours' labels (excluding the
component's own label and the medial-wall label 0). All reassignment
decisions are computed against the entry-state labels and written into
a fresh copy, so they are order-invariant.

Public API:
    remove_isolated_surface_components(lh_labels, rh_labels, lh_mesh,
                                       rh_mesh, abs_threshold=5)
        -> (lh_labels_new, rh_labels_new)

Tie-break:
    ``mode`` of an int array breaks ties on the smallest value (MATLAB
    convention). ``np.argmax(np.bincount(arr))`` matches this — ``argmax``
    returns the first index on ties.

Empty-vote edge case:
    If all neighbour labels are filtered out, MATLAB's ``mode([])``
    returns NaN and the assignment coerces. Python int64 cannot hold
    NaN, so we leave the label unchanged. The branch is unreachable on
    a real cortical mesh (every interior vert has ≥3 distinct-label
    non-medial neighbours).

Component identification reuses the union-find kernel from
:mod:`arealmshbm.check_connectedness._kernels`
(``cc_from_edges``): that kernel only unions vertices sharing a label,
so its output partitions the mesh into per-label connected components.
Component IDs may be numbered differently from MATLAB's
``CBIG_ComputeConnectedComponentsFromSurface``, but the partition is
identical, which is what the size / membership tests use.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from arealmshbm.check_connectedness._kernels import cc_from_edges


def _build_edges(vertex_nbors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (src, dst) 0-indexed edge list for ``cc_from_edges``.

    ``vertex_nbors`` is the MATLAB convention (max_neigh, N) int64 with
    1-indexed neighbour ids and ``0`` marking absent slots. Output is
    densified to int64 contiguous arrays as the kernel expects.
    """
    if vertex_nbors.ndim != 2:
        raise ValueError(
            f"vertex_nbors must be 2D (max_neigh, N); got {vertex_nbors.shape}"
        )
    max_neigh, N = vertex_nbors.shape
    nbors = np.ascontiguousarray(vertex_nbors, dtype=np.int64)
    src = np.repeat(np.arange(N, dtype=np.int64), max_neigh)
    dst = nbors.T.reshape(-1) - 1                  # 1-indexed → 0-indexed; -1 = absent
    valid = dst >= 0
    return src[valid].copy(), dst[valid].copy()


def _components(
    labels: np.ndarray, vertex_nbors: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Run the union-find component kernel on the label-restricted graph.

    Returns ``(ci, n_comp)`` where ``ci[v] ∈ [1, n_comp]`` is the
    component id of vertex ``v``. Each component contains vertices of
    exactly one label by construction.
    """
    N = labels.shape[0]
    src, dst = _build_edges(vertex_nbors)
    parent = np.empty(N, dtype=np.int64)
    rank = np.empty(N, dtype=np.int8)
    comp_map = np.empty(N, dtype=np.int64)
    ci = np.empty(N, dtype=np.int64)
    n_comp_out = np.empty(1, dtype=np.int64)
    cc_from_edges(src, dst, labels, parent, rank, comp_map, ci, n_comp_out)
    return ci, int(n_comp_out[0])


def _mode_smallest_tiebreak(arr: np.ndarray) -> int:
    """``mode(arr)`` with smallest-value tiebreak (MATLAB ``mode``).

    ``arr`` is a non-empty int array of strictly positive labels; ``0``
    must already be filtered out by the caller. ``np.argmax`` on the
    bincount returns the first (= smallest) index on ties, matching
    MATLAB's convention.
    """
    counts = np.bincount(arr)
    return int(np.argmax(counts))


def remove_isolated_surface_components(
    labels: np.ndarray,
    vertex_nbors: np.ndarray,
    abs_threshold: int = 5,
) -> np.ndarray:
    """Reassign tiny same-label connected components to a neighbour-mode label.

    Parameters
    ----------
    labels : (N,) array, integer-typed.
        Parcel labels; ``0`` denotes medial wall (or any vertex to be
        treated as label-less). Cast to int64 internally.
    vertex_nbors : (max_neigh, N) array, integer-typed.
        MATLAB convention: 1-indexed neighbour vertex ids; ``0`` marks
        absent slots. Cast to int64 internally.
    abs_threshold : int, default 5.
        Components with size strictly less than this are reassigned.
        MATLAB's default and CBIG's only call site both use 5.

    Returns
    -------
    new_labels : (N,) int64 array.
        ``labels`` with tiny components reassigned. Original input is
        not mutated.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got {labels.shape}")
    N = labels.shape[0]
    nbors = np.ascontiguousarray(vertex_nbors, dtype=np.int64)
    if nbors.shape[1] != N:
        raise ValueError(
            f"vertex_nbors second axis ({nbors.shape[1]}) must equal "
            f"labels length ({N})"
        )
    if int(abs_threshold) <= 0:
        raise ValueError(f"abs_threshold must be positive; got {abs_threshold}")

    ci, n_comp = _components(labels, nbors)

    # Component sizes; sizes[0] is unused since cc_from_edges emits ci ∈ [1, n_comp].
    sizes = np.bincount(ci, minlength=n_comp + 1)

    # Bucket vertices by component id once (avoids n_comp * O(N) np.where).
    order = np.argsort(ci, kind="stable")
    sorted_ci = ci[order]
    # boundaries[c] = first index in `order` where sorted_ci == c.
    boundaries = np.searchsorted(sorted_ci, np.arange(n_comp + 2))

    new_labels = labels.copy()

    for c in range(1, n_comp + 1):
        sz = int(sizes[c])
        if sz == 0 or sz >= int(abs_threshold):
            continue
        idx = order[boundaries[c]: boundaries[c + 1]]
        self_label = int(labels[idx[0]])

        # Gather neighbour vertex ids for this component (1-indexed; 0 = absent).
        neigh = nbors[:, idx].T.reshape(-1)
        neigh = neigh[neigh != 0] - 1                   # → 0-indexed
        if neigh.size == 0:
            continue                                     # isolated; no neighbours at all
        nlabels = labels[neigh]
        cand = nlabels[(nlabels != self_label) & (nlabels != 0)]
        if cand.size == 0:
            continue                                     # MATLAB returns NaN; we keep label
        new_labels[idx] = _mode_smallest_tiebreak(cand)

    return new_labels
