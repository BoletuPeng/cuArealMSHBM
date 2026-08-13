"""interpolate_sphere.py

Barycentric sphere-to-sphere linear interpolation.

Direct port of CBIG / MARS ``MARS_linearInterpolate.m`` +
``MARS_linearInterpolateAux.c`` + ``MARS_findFaces.h``. Numba @njit
implementation; the only Python-orchestrated step is the
``scipy.spatial.cKDTree`` nearest-vertex query (tech-stack exception).
All face scoring, radial projection, barycentric maths and
interpolation runs in a single parallel kernel.

Algorithm (per target ``p``):

  1. Find nearest source vertex ``v*`` via 3D KD-tree.
  2. For each face incident to ``v*`` (its 1-ring), radially project
     ``p`` onto the face plane (``proj = s * p`` with ``s = <v0,n>/<p,n>``;
     mirrors MATLAB ``projectPoint`` in ``MARS_vec3D.h``). Compute
     signed barycentric weights via the face normal; score by L1
     deviation outside [0, 1]. The min-score face wins; ties are
     broken by first-found.
  3. Interpolate using **unsigned-area** weights (matches MATLAB
     ``MARS_linearInterp.h::linearInterp``): three sub-triangle areas
     ``A_k = area(proj, v_i, v_j)`` divided by their sum. Inside the
     triangle these equal the standard bary weights; outside they
     soften the extrapolation. The weights are non-negative and sum
     to 1 by construction, so no clamp/renormalise step is needed.
  4. A near-unity weight (≥ 0.9999999) snaps to a hard corner
     (matches MATLAB's ``MIN_AREA_TOL``).

The output preserves all leading dims of ``data``.

Face-search strategy vs MATLAB
------------------------------
MATLAB's ``MARS_findFaces`` (MARS_findFaces.h) implements a greedy
best-first walk through the source mesh's vertex graph: from the
KD-nearest seed, it tests each 1-ring face with an ``isInTriangle``
check (signed-bary with a small slop tolerance); if none passes it
marches to the nearest unvisited neighbor and retries, with
``cur_thresh *= 10`` adaptive relaxation and ``prev_seeds``
backtracking as fallbacks. This port omits the walk and uses only
the KD-nearest seed's 1-ring, picking the min-score face inside it.

Both paths use the same final interpolation formula (unsigned-area
weights with the same ``MIN_AREA_TOL = 0.9999999`` corner snap), so
they only differ when they pick a different face. On the actual
fsaverage6 ↔ icosphere setups used by step 0 (both directions),
head-to-head comparison against MATLAB MARS_linearInterpolate on
identical inputs gives:

  * downsample (fsaverage6 → icosphere): max ``|py − ml|`` = 1.6e−4
    on synthetic N(0,1) data (~0.016 % of data std), 0 entries
    above 1e−3.
  * upsample (icosphere → fsaverage6): max ``|py − ml|`` = 3.6e−4
    (~0.036 % of std), 0 entries above 1e−3.

The face-choice disagreement that does occur is concentrated on
edge-tie targets (one signed-bary coord ≈ 0, target lies on a
shared edge between two faces), not on the pentagon Voronoi-cell
irregularity the walk was originally designed to defend against;
the top max-diff targets are 16-50 % of the sphere radius from the
nearest pentagon centre. The walk-with-backtrack is dead code on
well-formed Delaunay-like sphere triangulations because Voronoi(v*)
is contained in the 1-ring face fan of v* by construction. The
residual 1e−4-scale diff is fp32 precision on the
near-zero-bary contribution of edge-tied face picks, well below the
99th-percentile rel-diff bar of 1e−3.

Precision contract
------------------
- External arrays stored fp32.
- Internal projection / barycentric maths run in fp64 for numerical
  robustness near triangle edges; final values cast to ``data.dtype``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────
# Vertex→faces table (counting/scatter)
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True)
def build_vertex_faces(faces: np.ndarray, num_vertices: int) -> np.ndarray:
    """Build a (N, max_faces_per_vert) 1-indexed table of incident faces.

    Mirrors MATLAB ``mesh.vertexFaces``. Slot value ``0`` denotes
    "no face".
    """
    F = faces.shape[0]
    counts = np.zeros(num_vertices, dtype=np.int32)
    for f in range(F):
        for k in range(3):
            counts[faces[f, k]] += 1
    max_faces = 0
    for i in range(num_vertices):
        if counts[i] > max_faces:
            max_faces = counts[i]
    out = np.zeros((num_vertices, max_faces), dtype=np.int32)
    cursor = np.zeros(num_vertices, dtype=np.int32)
    for f in range(F):
        for k in range(3):
            v = faces[f, k]
            out[v, cursor[v]] = f + 1   # 1-indexed
            cursor[v] += 1
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-target face scoring + interpolation kernel.
# Computes (chosen_face, chosen_bary[3]) per target and interpolates
# all P leading channels into out[:, m].
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True, fastmath=False)
def _score_and_interp_batch(p_arr,           # (M, 3) fp32
                            face_table,      # (M, K) int32, 0=padding
                            faces,           # (F, 3) int32
                            verts,           # (N, 3) fp32
                            data_flat,       # (P, N) any float
                            out_flat):       # (P, M) same dtype as data_flat
    M = p_arr.shape[0]
    K = face_table.shape[1]
    P = data_flat.shape[0]
    BARY_TOL = 0.9999999

    for m in prange(M):
        px = np.float64(p_arr[m, 0])
        py = np.float64(p_arr[m, 1])
        pz = np.float64(p_arr[m, 2])

        best_score = np.inf
        best_face = 0
        best_b0 = 0.0
        best_b1 = 0.0
        best_b2 = 0.0

        for k in range(K):
            f1 = face_table[m, k]
            if f1 <= 0:
                continue
            f = f1 - 1   # 0-indexed face
            v0i = faces[f, 0]
            v1i = faces[f, 1]
            v2i = faces[f, 2]

            v0x = np.float64(verts[v0i, 0]); v0y = np.float64(verts[v0i, 1]); v0z = np.float64(verts[v0i, 2])
            v1x = np.float64(verts[v1i, 0]); v1y = np.float64(verts[v1i, 1]); v1z = np.float64(verts[v1i, 2])
            v2x = np.float64(verts[v2i, 0]); v2y = np.float64(verts[v2i, 1]); v2z = np.float64(verts[v2i, 2])

            # Edges and full normal (unnormalised; bary formulae are
            # invariant to scale of `n`).
            ex1 = v1x - v0x; ey1 = v1y - v0y; ez1 = v1z - v0z
            ex2 = v2x - v0x; ey2 = v2y - v0y; ez2 = v2z - v0z
            nx = ey1 * ez2 - ez1 * ey2
            ny = ez1 * ex2 - ex1 * ez2
            nz = ex1 * ey2 - ey1 * ex2

            dot_v0n = v0x * nx + v0y * ny + v0z * nz
            dot_pn = px * nx + py * ny + pz * nz
            # Parallel-plane guard: the radial projection ``s = <v0,n>/<p,n>``
            # blows up when ``<p,n>`` is near zero (target point lies in the
            # face's plane). On a sphere of radius ~100 with non-degenerate
            # faces, ``|dot_pn|`` ≈ |n| * radius ≈ a few hundred; values
            # below 1e-12 indicate a degenerate face — skip rather than
            # produce a wildly displaced projection.
            if abs(dot_pn) < 1e-12:
                continue
            s = dot_v0n / dot_pn

            prx = s * px; pry = s * py; prz = s * pz

            # Signed bary via face-normal sub-triangle normals.
            full_n_sq = nx * nx + ny * ny + nz * nz
            if full_n_sq == 0.0:
                full_n_sq = 1e-30

            # n0 = (v1 - proj) x (v2 - proj)
            ax = v1x - prx; ay = v1y - pry; az = v1z - prz
            bx = v2x - prx; by = v2y - pry; bz = v2z - prz
            n0x = ay * bz - az * by
            n0y = az * bx - ax * bz
            n0z = ax * by - ay * bx
            sb0 = (n0x * nx + n0y * ny + n0z * nz) / full_n_sq

            # n1 = (v2 - proj) x (v0 - proj)
            ax = v2x - prx; ay = v2y - pry; az = v2z - prz
            bx = v0x - prx; by = v0y - pry; bz = v0z - prz
            n1x = ay * bz - az * by
            n1y = az * bx - ax * bz
            n1z = ax * by - ay * bx
            sb1 = (n1x * nx + n1y * ny + n1z * nz) / full_n_sq

            # n2 = (v0 - proj) x (v1 - proj)
            ax = v0x - prx; ay = v0y - pry; az = v0z - prz
            bx = v1x - prx; by = v1y - pry; bz = v1z - prz
            n2x = ay * bz - az * by
            n2y = az * bx - ax * bz
            n2z = ax * by - ay * bx
            sb2 = (n2x * nx + n2y * ny + n2z * nz) / full_n_sq

            lo = sb0 if sb0 < sb1 else sb1
            if sb2 < lo:
                lo = sb2
            hi = sb0 if sb0 > sb1 else sb1
            if sb2 > hi:
                hi = sb2
            score = 0.0
            if lo < 0.0:
                score -= lo
            if hi > 1.0:
                score += hi - 1.0

            # Unsigned-area sub-triangle weights.
            # A0 = 0.5 * |(v1 - proj) x (v2 - proj)| = 0.5 * |n0|, etc.
            A0 = 0.5 * np.sqrt(n0x * n0x + n0y * n0y + n0z * n0z)
            A1 = 0.5 * np.sqrt(n1x * n1x + n1y * n1y + n1z * n1z)
            A2 = 0.5 * np.sqrt(n2x * n2x + n2y * n2y + n2z * n2z)
            totalA = A0 + A1 + A2
            if totalA == 0.0:
                totalA = 1e-30
            ub0 = A0 / totalA
            ub1 = A1 / totalA
            ub2 = A2 / totalA

            if score < best_score:
                best_score = score
                best_face = f
                best_b0 = ub0
                best_b1 = ub1
                best_b2 = ub2

        # Sanity check: best_score stays ``np.inf`` iff every face slot in
        # face_table[m, :] was skipped (no candidate face, or every
        # candidate was degenerate/parallel to the radial). Under the
        # KDTree + closed-mesh invariant this is unreachable, but with
        # ``boundscheck=False`` a silent garbage emit (best_face=0,
        # best_b*=0) would otherwise propagate downstream. Refuse.
        if best_score == np.inf:
            raise RuntimeError(
                "interpolate_sphere: no valid face found for a target point "
                "(empty face_table row or all candidates degenerate). "
                "Check that source_vertex_faces is built from a closed mesh."
            )

        # NOTE: ``best_b{0,1,2}`` are unsigned-area weights ``A_k / Σ A_k``
        # with ``A_k = 0.5 * ||cross||`` (non-negative) and Σ = 1 by
        # construction, so they are ALWAYS in [0, 1]. The clamp/renormalise
        # branch from the original draft was wired against unsigned weights
        # and therefore unreachable; the MATLAB MIN_AREA_TOL snap below
        # remains the sole post-processing.

        # MATLAB MIN_AREA_TOL = 0.9999999 — snap a near-1 weight to a hard corner.
        if best_b0 > BARY_TOL:
            best_b0 = 1.0; best_b1 = 0.0; best_b2 = 0.0
        elif best_b1 > BARY_TOL:
            best_b0 = 0.0; best_b1 = 1.0; best_b2 = 0.0
        elif best_b2 > BARY_TOL:
            best_b0 = 0.0; best_b1 = 0.0; best_b2 = 1.0

        f0 = faces[best_face, 0]
        f1v = faces[best_face, 1]
        f2v = faces[best_face, 2]
        for p in range(P):
            out_flat[p, m] = (data_flat[p, f0] * best_b0
                              + data_flat[p, f1v] * best_b1
                              + data_flat[p, f2v] * best_b2)


# ─────────────────────────────────────────────────────────────────────
# Public API — thin orchestrator: validate, KD-tree, dispatch kernel.
# ─────────────────────────────────────────────────────────────────────
def linear_interpolate_sphere(target_points: np.ndarray,
                              source_verts: np.ndarray,
                              source_faces: np.ndarray,
                              source_vertex_faces: Optional[np.ndarray],
                              data: np.ndarray,
                              ) -> np.ndarray:
    """Barycentric linear interpolation from a source mesh onto target
    points, both lying on a sphere of approximately the same radius.

    Parameters
    ----------
    target_points : (M, 3) fp32
    source_verts : (N, 3) fp32
    source_faces : (F, 3) int32, 0-indexed
    source_vertex_faces : (N, max_faces_per_vert) int32 or None
        1-indexed; ``0`` marks an absent slot. Built on the fly if
        ``None``.
    data : (..., N) any float
        Vertex-attached values; leading dims preserved.

    Returns
    -------
    vals : (..., M) same dtype as ``data``
    """
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError(f"target_points must be (M, 3); got {target_points.shape}")
    if source_verts.ndim != 2 or source_verts.shape[1] != 3:
        raise ValueError(f"source_verts must be (N, 3); got {source_verts.shape}")
    if source_faces.ndim != 2 or source_faces.shape[1] != 3:
        raise ValueError(f"source_faces must be (F, 3); got {source_faces.shape}")
    N = source_verts.shape[0]
    M = target_points.shape[0]
    if data.shape[-1] != N:
        raise ValueError(
            f"data trailing dim {data.shape[-1]} must equal num source verts {N}")

    pts32 = np.ascontiguousarray(target_points, dtype=np.float32)
    src32 = np.ascontiguousarray(source_verts, dtype=np.float32)
    faces32 = np.ascontiguousarray(source_faces, dtype=np.int32)

    if source_vertex_faces is None:
        vertex_faces = build_vertex_faces(faces32, N)
    else:
        if source_vertex_faces.shape[0] != N:
            raise ValueError(
                f"source_vertex_faces row count {source_vertex_faces.shape[0]}"
                f" != num verts {N}")
        vertex_faces = np.ascontiguousarray(source_vertex_faces, dtype=np.int32)

    # KD-tree nearest-vertex query (tech-stack exception: scipy/C).
    tree = cKDTree(src32.astype(np.float64, copy=False))
    _, nearest_idx = tree.query(pts32.astype(np.float64, copy=False), k=1)
    face_table = np.ascontiguousarray(vertex_faces[nearest_idx], dtype=np.int32)

    # Reshape data to (P, N) flat-channel form; preserve leading dims on output.
    leading = data.shape[:-1]
    data_flat = np.ascontiguousarray(data.reshape(-1, N))
    out_flat = np.empty((data_flat.shape[0], M), dtype=data.dtype)

    _score_and_interp_batch(pts32, face_table, faces32, src32, data_flat, out_flat)

    return out_flat.reshape(leading + (M,))
