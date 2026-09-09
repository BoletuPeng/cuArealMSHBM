"""_common.py

Helpers shared by the CPU (:mod:`.radius_mask`) and GPU
(:mod:`.radius_mask_gpu`) supercalls. Splitting these out removes ~80
lines of verbatim duplication between the two sibling files and keeps
the MARS area-rescale (load-bearing — without it the geodesic distances
drift ~1.45x and per-parcel Jaccard collapses to ~0.6 on fsaverage6) in
one place.

Four helpers:
    _read_aparc(hemi, mesh, cbig_code_dir=None) -> (V,) int64
        Resolve & parse a FreeSurfer aparc.annot at
        ``<atlas>/<mesh>/label/<hemi>.aparc.annot``. Returns labels
        offset by +1 (so precentral=25, postcentral=23, insula=36 — the
        MATLAB convention used downstream; -1 unmapped becomes 0).

    _build_mesh_csr(vertices, faces, vertex_nbors) -> (indptr, indices, weights)
        Undirected fp32 CSR adjacency for a triangle mesh, weighted by
        Euclidean distance on the inflated surface after MARS area
        rescale to 4*pi*100^2 mm^2. ``indptr`` / ``indices`` int64,
        ``weights`` fp32 C-contig. The adjacency comes from the mesh
        bundle's ``vertexNbors`` table (1-indexed, 0 = absent slot);
        ``faces`` is used only for the area rescale factor.

    _mask_to_csc(mask) -> csc_matrix
        (V, L) uint8 boundary mask -> the fp64 double-sparse the .mat
        schema stores, without the dense fp64 round trip.

    _coerce_labels(labels) -> (V,) int64
        Normalize to 1-indexed (0 = medial wall) int64. Negative values
        (the -1 unmapped sentinel) collapse to 0.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ..data_io.annot_io import read_annot_labels
from ..data_io.load_avg_mesh import _atlas_dir


def _read_aparc(hemi: str, mesh: str,
                cbig_code_dir: Optional[str] = None) -> np.ndarray:
    """Read fsaverage* aparc.annot as a (V,) integer label vector with
    MATLAB's 1-indexed convention (insula=36, precentral=25,
    postcentral=23). Unmapped vertices (-1) become 0.
    """
    base = Path(cbig_code_dir) if cbig_code_dir else _atlas_dir()
    p = base / mesh / "label" / f"{hemi}.aparc.annot"
    if not p.exists():
        raise FileNotFoundError(f"aparc.annot for {hemi} on {mesh} not found: {p}")
    # numpy-only parser; equal to nibabel's ``read_annot(...)[0]``
    # without pulling the whole nibabel import into this leaf.
    # See data_io/annot_io.py.
    return read_annot_labels(p).astype(np.int64) + 1


def _build_mesh_csr(vertices: np.ndarray,
                    faces: np.ndarray,
                    vertex_nbors: np.ndarray,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected CSR adjacency (both directions per edge) for a triangle
    mesh, fp32 weights = euclidean distance on the inflated surface
    after MARS area-rescale to 4*pi*100^2 mm^2. Returns
    ``(indptr, indices, weights)``.

    The MARS rescale is load-bearing — without it geodesic distances
    differ from MATLAB by ~1.45x and per-parcel Jaccard collapses to
    ~0.6 on fsaverage6.

    Adjacency is read straight out of the bundle's ``vertexNbors``
    ``(max_neigh, V)`` table (1-indexed, 0 = absent slot) instead of
    being rebuilt by deduplicating the 6*F directed face edges. The
    two agree exactly on a closed triangle mesh — every directed edge
    (v, u) of a face is an entry of ``vertexNbors[:, v]`` and vice
    versa — but the table route is a per-column sort instead of an
    ``np.unique`` over the 6*F directed edge keys. ``faces`` is still
    needed, for the surface-area rescale factor only.

    The face route also *guaranteed* three properties the geodesics
    depend on and a raw table cannot be assumed to have — no self-loop,
    no duplicate neighbour, and symmetry (an undirected graph). Those
    are therefore checked here, exactly: the symmetry test looks each
    directed edge's reverse up in the target's own <= max_neigh-slot
    column rather than sorting the whole edge key set.
    """
    V_xyz = np.asarray(vertices)
    if V_xyz.shape[0] == 3 and V_xyz.shape[1] != 3:
        V_xyz = V_xyz.T
    V_xyz = np.ascontiguousarray(V_xyz, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    # Same arithmetic as ``np.cross`` + ``np.linalg.norm(axis=1)``,
    # spelled out per component: both evaluate the cross terms in the
    # same order and reduce the 3-element row sum left-to-right, so the
    # area (and hence ``factor``, and hence every fp32 weight below) is
    # bit-identical — it just avoids np.cross's generic axis machinery.
    v0 = V_xyz[F[:, 0]]
    e1 = V_xyz[F[:, 1]] - v0
    e2 = V_xyz[F[:, 2]] - v0
    cx = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
    cy = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
    cz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    area = 0.5 * np.sqrt(cx * cx + cy * cy + cz * cz).sum()
    factor = float(np.sqrt(4.0 * np.pi * 100.0 * 100.0 / area))
    V_fp32 = (V_xyz * factor).astype(np.float32)
    n = V_fp32.shape[0]

    nb = np.asarray(vertex_nbors, dtype=np.int64)
    if nb.ndim != 2 or nb.shape[1] != n:
        raise ValueError(
            f"_build_mesh_csr: vertex_nbors must be (max_neigh, {n}); "
            f"got {nb.shape}"
        )
    if nb.min() < 0 or nb.max() > n:
        raise ValueError(
            "_build_mesh_csr: vertex_nbors must hold 1-indexed vertex ids "
            f"in [0, {n}] (0 = absent slot); got range "
            f"[{int(nb.min())}, {int(nb.max())}]"
        )
    # Sort each column ascending with absent slots (0) pushed to the
    # bottom, then read row-major: each vertex's valid neighbours land
    # contiguously and in ascending id order — exactly CSR with
    # ``sort_indices()`` applied.
    filled = np.where(nb > 0, nb, np.int64(n + 1))
    filled.sort(axis=0)
    flat = np.ascontiguousarray(filled.T).ravel()
    keep = flat <= n
    indices = flat[keep] - 1
    counts = (nb > 0).sum(axis=0)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    # ── the three invariants the retired faces+np.unique route gave for
    # free, and which the geodesics silently depend on. All exact, all
    # vectorized; the reverse-edge test only scans each endpoint's
    # <= max_neigh column.
    # No duplicate neighbour: after the sort a repeat is an adjacent
    # equal pair below the pad sentinel.
    dup = (filled[:-1] == filled[1:]) & (filled[1:] <= n)
    if dup.any():
        bad = int(np.argmax(dup.any(axis=0)))
        raise ValueError(
            f"_build_mesh_csr: vertex_nbors column {bad} lists the same "
            f"neighbour twice; the adjacency must be a simple graph (the "
            f"mesh bundle is corrupt)")
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    if (indices == rows).any():
        bad = int(rows[np.argmax(indices == rows)])
        raise ValueError(
            f"_build_mesh_csr: vertex_nbors column {bad} lists itself as a "
            f"neighbour; self-loops are not allowed (the mesh bundle is "
            f"corrupt)")
    # Symmetry: every directed edge (r, c) must have its reverse in c's
    # own <= max_neigh-slot column. filled is 1-indexed, hence rows + 1.
    if not (filled[:, indices] == (rows + 1)).any(axis=0).all():
        e = int(np.argmin((filled[:, indices] == (rows + 1)).any(axis=0)))
        raise ValueError(
            f"_build_mesh_csr: vertex_nbors is not symmetric — edge "
            f"({int(rows[e])} -> {int(indices[e])}) has no reverse entry. "
            f"The adjacency must be undirected (the mesh bundle is corrupt)")

    # ``src`` is a run-length expansion, not a gather — np.repeat beats
    # fancy indexing. The squared-distance reduction keeps the original
    # order: fp32 products accumulated left-to-right in fp64, exactly
    # what ``np.sum(diff * diff, axis=1, dtype=np.float64)`` does.
    src = np.repeat(V_fp32, counts, axis=0)
    dst = V_fp32[indices]
    d0 = src[:, 0] - dst[:, 0]
    d1 = src[:, 1] - dst[:, 1]
    d2 = src[:, 2] - dst[:, 2]
    acc = (d0 * d0).astype(np.float64)
    acc += (d1 * d1).astype(np.float64)
    acc += (d2 * d2).astype(np.float64)
    weights = np.sqrt(acc).astype(np.float32)

    return (
        indptr,
        np.ascontiguousarray(indices, dtype=np.int64),
        np.ascontiguousarray(weights, dtype=np.float32),
    )


def _mask_to_csc(mask: np.ndarray) -> sp.csc_matrix:
    """``(V, L)`` uint8 boundary mask -> the fp64 ``csc_matrix`` the
    ``.mat`` schema stores (MATLAB keeps ``boundary`` as double sparse).

    Identical output to ``sp.csc_matrix(mask.astype(np.float64))`` --
    same indptr / ascending indices / all-ones data -- but built from a
    single transposed ``flatnonzero`` instead of a dense fp64 scan of
    all V*L cells.
    """
    if mask.ndim != 2:
        raise ValueError(f"_mask_to_csc: mask must be 2-D; got {mask.shape}")
    if mask.dtype != np.uint8:
        raise ValueError(f"_mask_to_csc: mask must be uint8; got {mask.dtype}")
    V, L = mask.shape
    # Column-major order == row-major order of the transpose.
    flat = np.flatnonzero(np.ascontiguousarray(mask.T))
    indices = (flat % V).astype(np.int32)
    counts = np.bincount(flat // V, minlength=L)
    indptr = np.zeros(L + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    data = np.ones(indices.shape[0], dtype=np.float64)
    return sp.csc_matrix((data, indices, indptr), shape=(V, L))


def _coerce_labels(labels: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int64).ravel().copy()
    out[out < 0] = 0
    return out
