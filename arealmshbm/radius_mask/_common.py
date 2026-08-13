"""_common.py

Helpers shared by the CPU (:mod:`.radius_mask`) and GPU
(:mod:`.radius_mask_gpu`) supercalls. Splitting these out removes ~80
lines of verbatim duplication between the two sibling files and keeps
the MARS area-rescale (load-bearing — without it the geodesic distances
drift ~1.45x and per-parcel Jaccard collapses to ~0.6 on fsaverage6) in
one place.

Three helpers:
    _read_aparc(hemi, mesh, cbig_code_dir=None) -> (V,) int64
        Resolve & parse a FreeSurfer aparc.annot at
        ``<atlas>/<mesh>/label/<hemi>.aparc.annot``. Returns labels
        offset by +1 (so precentral=25, postcentral=23, insula=36 — the
        MATLAB convention used downstream; -1 unmapped becomes 0).

    _build_mesh_csr(vertices, faces) -> (indptr, indices, weights)
        Undirected fp32 CSR adjacency for a triangle mesh, weighted by
        Euclidean distance on the inflated surface after MARS area
        rescale to 4*pi*100^2 mm^2. ``indptr`` / ``indices`` int64,
        ``weights`` fp32 C-contig.

    _coerce_labels(labels) -> (V,) int64
        Normalize to 1-indexed (0 = medial wall) int64. Negative values
        (nibabel's -1 unmapped sentinel) collapse to 0.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ..data_io.load_avg_mesh import _atlas_dir


def _read_aparc(hemi: str, mesh: str,
                cbig_code_dir: Optional[str] = None) -> np.ndarray:
    """Read fsaverage* aparc.annot as a (V,) integer label vector with
    MATLAB's 1-indexed convention (insula=36, precentral=25,
    postcentral=23). Unmapped vertices (-1) become 0.
    """
    import nibabel.freesurfer.io as fsio

    base = Path(cbig_code_dir) if cbig_code_dir else _atlas_dir()
    p = base / mesh / "label" / f"{hemi}.aparc.annot"
    if not p.exists():
        raise FileNotFoundError(f"aparc.annot for {hemi} on {mesh} not found: {p}")
    labels, _, _ = fsio.read_annot(str(p))
    return labels.astype(np.int64) + 1


def _build_mesh_csr(vertices: np.ndarray,
                    faces: np.ndarray,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected CSR adjacency (both directions per edge) for a triangle
    mesh, fp32 weights = euclidean distance on the inflated surface
    after MARS area-rescale to 4*pi*100^2 mm^2. Returns
    ``(indptr, indices, weights)``.

    The MARS rescale is load-bearing — without it geodesic distances
    differ from MATLAB by ~1.45x and per-parcel Jaccard collapses to
    ~0.6 on fsaverage6.
    """
    V_xyz = np.asarray(vertices)
    if V_xyz.shape[0] == 3 and V_xyz.shape[1] != 3:
        V_xyz = V_xyz.T
    V_xyz = np.ascontiguousarray(V_xyz, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    e1 = V_xyz[F[:, 1]] - V_xyz[F[:, 0]]
    e2 = V_xyz[F[:, 2]] - V_xyz[F[:, 0]]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1).sum()
    factor = float(np.sqrt(4.0 * np.pi * 100.0 * 100.0 / area))
    V_fp32 = (V_xyz * factor).astype(np.float32)
    n = V_fp32.shape[0]

    e_a = F[:, [0, 1, 2]].ravel()
    e_b = F[:, [1, 2, 0]].ravel()
    rows = np.concatenate([e_a, e_b])
    cols = np.concatenate([e_b, e_a])
    diff = V_fp32[rows] - V_fp32[cols]
    w = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64)).astype(np.float32)

    # Each interior edge is shared by 2 triangles -> dedupe per direction.
    key = rows.astype(np.int64) * np.int64(n) + cols.astype(np.int64)
    _, uidx = np.unique(key, return_index=True)
    rows = rows[uidx]; cols = cols[uidx]; w = w[uidx]

    csr = sp.csr_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32)
    csr.sort_indices()
    return (
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        np.ascontiguousarray(csr.data, dtype=np.float32),
    )


def _coerce_labels(labels: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int64).ravel().copy()
    out[out < 0] = 0
    return out
