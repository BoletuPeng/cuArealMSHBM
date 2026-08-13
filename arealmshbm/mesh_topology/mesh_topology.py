"""mesh_topology.py

Public entry: ``compute_topology(vertices, faces)`` — returns the two
per-vertex incidence tables the rest of step 0 actually consumes.

Returns
-------
(vertex_nbors, vertex_faces) — both ``(N, max_*)`` int32 1-indexed
(0 = absent slot).

Reproduces ``MARS_convertFaces2VertNbors`` and
``MARS_convertFaces2FacesOfVert``. The other ``sbjMesh`` fields the
MATLAB struct exposes (faceAreas, metricVerts, vertexDistSq2Nbors,
surface_scaling_factor) are not consumed by the Python step-0
pipeline and have been dropped.

Neighbor ordering: matches ``AddNeighbors`` in the C source — for
every face (v0, v1, v2) processed in face-array order, the pairs
(v0, v1), (v0, v2), (v1, v2) are appended in that order if not
already present. Identical neighbor sets are guaranteed; the column
order is implementation-defined and downstream consumers are
order-independent.

Closed-mesh assumption: ``max_cap`` is sized from the per-vertex
incident-face count, which equals the 1-ring degree only on a closed
mesh (every edge shared by exactly two triangles). All meshes step 0
consumes (fsaverage6 sphere/midthickness, downsampled icospheres) are
closed, so the allocation is exact. On an open mesh a boundary vertex
has ``n_neigh = n_faces + 1`` and would overflow ``vertex_nbors``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def compute_topology(vertices: np.ndarray,
                     faces: np.ndarray):
    """Build the two per-vertex incidence tables.

    Parameters
    ----------
    vertices : (N, 3) float (any precision; only ``shape[0]`` is used)
    faces    : (F, 3) int  0-indexed

    Returns
    -------
    vertex_nbors : (N, max_neigh) int32 1-indexed (0 = absent)
    vertex_faces : (N, max_faces) int32 1-indexed (0 = absent)
    """
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]

    # First pass: count incident faces per vertex (= neighbor capacity).
    fac_per_vert = np.zeros(n_verts, dtype=np.int32)
    for f in range(n_faces):
        fac_per_vert[faces[f, 0]] += 1
        fac_per_vert[faces[f, 1]] += 1
        fac_per_vert[faces[f, 2]] += 1

    max_cap = 0
    for i in range(n_verts):
        if fac_per_vert[i] > max_cap:
            max_cap = int(fac_per_vert[i])

    # vertex_nbors: AddNeighbors-style (dedup membership scan).
    vertex_nbors = np.zeros((n_verts, max_cap), dtype=np.int32)
    nb_count = np.zeros(n_verts, dtype=np.int32)

    # vertex_faces: face-of-vertex append (no dedup needed; each face
    # contributes once per vertex slot).
    vertex_faces = np.zeros((n_verts, max_cap), dtype=np.int32)
    f_count = np.zeros(n_verts, dtype=np.int32)

    for f in range(n_faces):
        v0 = int(faces[f, 0])
        v1 = int(faces[f, 1])
        v2 = int(faces[f, 2])
        # vertex_faces (1-indexed face id)
        vertex_faces[v0, f_count[v0]] = f + 1; f_count[v0] += 1
        vertex_faces[v1, f_count[v1]] = f + 1; f_count[v1] += 1
        vertex_faces[v2, f_count[v2]] = f + 1; f_count[v2] += 1

        # vertex_nbors (AddNeighbors edges (v0,v1), (v0,v2), (v1,v2))
        for (a, b) in ((v0, v1), (v0, v2), (v1, v2)):
            ca = nb_count[a]
            found = False
            for k in range(ca):
                if vertex_nbors[a, k] == b + 1:
                    found = True
                    break
            if not found:
                vertex_nbors[a, ca] = b + 1
                nb_count[a] = ca + 1
                cb = nb_count[b]
                vertex_nbors[b, cb] = a + 1
                nb_count[b] = cb + 1

    return vertex_nbors, vertex_faces
