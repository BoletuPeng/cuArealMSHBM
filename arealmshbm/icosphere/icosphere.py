"""icosphere.py — numba @njit port of class-I geodesic icosahedron.

Public entry: ``make_icosphere(num_target_verts, radius=100.0)`` — IS
the @njit Dispatcher. Returns a tuple ``(vertices, faces)``.

Algorithm
---------
T = ceil(sqrt((N - 2) / 10)). Build the 12-vert / 20-face unit
icosahedron, then for each face: place T-1 points on each edge and
(T-1)(T-2)/2 interior points on a barycentric (i,j,k) grid; tessellate
into T^2 small triangles. Project all non-corner verts to the unit
sphere, then scale to radius. Yields V = 10*T^2 + 2, F = 20*T^2.

The two dict lookups in the original Python code are replaced with
flat ndarrays for numba compatibility:

    edge_lookup : (12, 12) int32, init -1, written by sorted-pair on
                  first encounter. Edge IDs assigned in the same
                  face-major order as the original dict-insertion code.
    gid         : (T+1, T+1) int64, init -1, recomputed inside each
                  face. Indexed by (i, j) with k = T - i - j.

Validation contract:
    V == 10*T^2 + 2, F == 20*T^2, E == 30*T^2 (Euler char = 2),
    every edge in exactly 2 faces, every vert at radius (fp32 1e-4 rel).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _icosahedron_unit_njit():
    """Regular icosahedron inscribed in the unit sphere (12 verts, 20 faces).

    Vertex coords are unit-length; face winding is outward CCW.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    raw = np.array([
        [-1.0,  phi,  0.0], [ 1.0,  phi,  0.0],
        [-1.0, -phi,  0.0], [ 1.0, -phi,  0.0],
        [ 0.0, -1.0,  phi], [ 0.0,  1.0,  phi],
        [ 0.0, -1.0, -phi], [ 0.0,  1.0, -phi],
        [ phi,  0.0, -1.0], [ phi,  0.0,  1.0],
        [-phi,  0.0, -1.0], [-phi,  0.0,  1.0],
    ], dtype=np.float64)
    for i in range(12):
        nrm = math.sqrt(raw[i, 0] ** 2 + raw[i, 1] ** 2 + raw[i, 2] ** 2)
        inv = 1.0 / nrm
        raw[i, 0] *= inv; raw[i, 1] *= inv; raw[i, 2] *= inv

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return raw, faces


@njit(cache=True)
def make_icosphere(num_target_verts, radius=100.0):
    """Build a class-I geodesic icosahedron with V >= num_target_verts.

    Returns
    -------
    vertices : (V, 3) float32, all on the sphere of given radius.
    faces    : (F, 3) int32, 0-indexed.
    """
    if num_target_verts < 12:
        T = 1
    else:
        T = int(math.ceil(math.sqrt((num_target_verts - 2) / 10.0)))
        if T < 1:
            T = 1
    radius_f = float(radius)

    base_verts, base_faces = _icosahedron_unit_njit()
    n_base_v = base_verts.shape[0]
    n_base_f = base_faces.shape[0]

    # --- Edge enumeration in face-major sorted-pair order (matches old
    # dict insertion order). edge_lookup is the dict replacement.
    edge_lookup = np.full((n_base_v, n_base_v), -1, dtype=np.int32)
    next_edge_id = 0
    for fi in range(n_base_f):
        for side in range(3):
            a = base_faces[fi, side]
            b = base_faces[fi, (side + 1) % 3]
            lo = a if a < b else b
            hi = b if a < b else a
            if edge_lookup[lo, hi] < 0:
                edge_lookup[lo, hi] = next_edge_id
                next_edge_id += 1

    # --- Allocate vertex array.
    base_offset_edge = n_base_v
    n_edge_pts = 30 * (T - 1)
    base_offset_face = base_offset_edge + n_edge_pts
    interior_per_face = (T - 1) * (T - 2) // 2  # 0 for T in {1, 2}
    total_v = n_base_v + n_edge_pts + 20 * interior_per_face

    verts = np.zeros((total_v, 3), dtype=np.float64)
    for i in range(n_base_v):
        verts[i, 0] = base_verts[i, 0]
        verts[i, 1] = base_verts[i, 1]
        verts[i, 2] = base_verts[i, 2]

    # --- Fill edge points (T-1 per edge), each edge oriented lo->hi.
    edge_emitted = np.zeros((n_base_v, n_base_v), dtype=np.uint8)
    for fi in range(n_base_f):
        for side in range(3):
            a = base_faces[fi, side]
            b = base_faces[fi, (side + 1) % 3]
            lo = a if a < b else b
            hi = b if a < b else a
            if edge_emitted[lo, hi] == 0:
                edge_emitted[lo, hi] = 1
                eid = edge_lookup[lo, hi]
                for k in range(1, T):
                    t = k / T
                    one_t = 1.0 - t
                    px = one_t * base_verts[lo, 0] + t * base_verts[hi, 0]
                    py = one_t * base_verts[lo, 1] + t * base_verts[hi, 1]
                    pz = one_t * base_verts[lo, 2] + t * base_verts[hi, 2]
                    inv = 1.0 / math.sqrt(px * px + py * py + pz * pz)
                    row = base_offset_edge + eid * (T - 1) + (k - 1)
                    verts[row, 0] = px * inv
                    verts[row, 1] = py * inv
                    verts[row, 2] = pz * inv

    # --- Tessellate each face.
    faces_out = np.zeros((20 * T * T, 3), dtype=np.int64)
    next_face_row = 0

    # Per-face (i,j)->global id grid; reset each face.
    gid = np.full((T + 1, T + 1), -1, dtype=np.int64)

    for face_id in range(n_base_f):
        A = base_faces[face_id, 0]
        B = base_faces[face_id, 1]
        C = base_faces[face_id, 2]

        for ii in range(T + 1):
            for jj in range(T + 1):
                gid[ii, jj] = -1
        gid[T, 0] = A
        gid[0, T] = B
        gid[0, 0] = C

        # Edge AB (k=0): walk j=1..T-1, i = T-j.
        ab_lo = A if A < B else B; ab_hi = B if A < B else A
        ab_eid = edge_lookup[ab_lo, ab_hi]
        ab_rev = (A > B)
        for j in range(1, T):
            i = T - j
            stored_k = i if ab_rev else j
            gid[i, j] = base_offset_edge + ab_eid * (T - 1) + (stored_k - 1)

        # Edge BC (i=0): walk k=1..T-1, j = T-k.
        bc_lo = B if B < C else C; bc_hi = C if B < C else B
        bc_eid = edge_lookup[bc_lo, bc_hi]
        bc_rev = (B > C)
        for k in range(1, T):
            j = T - k
            stored_k = j if bc_rev else k
            gid[0, j] = base_offset_edge + bc_eid * (T - 1) + (stored_k - 1)

        # Edge CA (j=0): walk i=1..T-1, k = T-i.
        ca_lo = C if C < A else A; ca_hi = A if C < A else C
        ca_eid = edge_lookup[ca_lo, ca_hi]
        ca_rev = (C > A)
        for i in range(1, T):
            k = T - i
            stored_k = k if ca_rev else i
            gid[i, 0] = base_offset_edge + ca_eid * (T - 1) + (stored_k - 1)

        # Interior pts in original (i, j) iteration order.
        face_interior_base = base_offset_face + face_id * interior_per_face
        local_idx = 0
        pAx = base_verts[A, 0]; pAy = base_verts[A, 1]; pAz = base_verts[A, 2]
        pBx = base_verts[B, 0]; pBy = base_verts[B, 1]; pBz = base_verts[B, 2]
        pCx = base_verts[C, 0]; pCy = base_verts[C, 1]; pCz = base_verts[C, 2]
        for i in range(1, T - 1):
            for j in range(1, T - i):
                global_id = face_interior_base + local_idx
                local_idx += 1
                gid[i, j] = global_id
                k = T - i - j
                px = (i * pAx + j * pBx + k * pCx) / T
                py = (i * pAy + j * pBy + k * pCy) / T
                pz = (i * pAz + j * pBz + k * pCz) / T
                inv = 1.0 / math.sqrt(px * px + py * py + pz * pz)
                verts[global_id, 0] = px * inv
                verts[global_id, 1] = py * inv
                verts[global_id, 2] = pz * inv

        # Tessellate face into T^2 small triangles.
        for i in range(T):
            for j in range(T - i):
                a_id = gid[i, j]
                b_id = gid[i + 1, j]
                c_id = gid[i, j + 1]
                if b_id >= 0 and c_id >= 0:
                    faces_out[next_face_row, 0] = a_id
                    faces_out[next_face_row, 1] = b_id
                    faces_out[next_face_row, 2] = c_id
                    next_face_row += 1
                if i + j < T - 1:
                    faces_out[next_face_row, 0] = gid[i + 1, j + 1]
                    faces_out[next_face_row, 1] = c_id  # (i, j+1)
                    faces_out[next_face_row, 2] = b_id  # (i+1, j)
                    next_face_row += 1

    # --- Final scaling to requested radius and dtype cast.
    verts_scaled = np.empty((total_v, 3), dtype=np.float32)
    for i in range(total_v):
        verts_scaled[i, 0] = np.float32(verts[i, 0] * radius_f)
        verts_scaled[i, 1] = np.float32(verts[i, 1] * radius_f)
        verts_scaled[i, 2] = np.float32(verts[i, 2] * radius_f)

    faces_i32 = np.empty((20 * T * T, 3), dtype=np.int32)
    for i in range(20 * T * T):
        faces_i32[i, 0] = np.int32(faces_out[i, 0])
        faces_i32[i, 1] = np.int32(faces_out[i, 1])
        faces_i32[i, 2] = np.int32(faces_out[i, 2])

    return verts_scaled, faces_i32
