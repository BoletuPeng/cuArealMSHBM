"""mesh_topology

Build the per-vertex incidence tables of the CBIG SPGrad ``sbjMesh``
struct (see ``CBIG_SPGrad_read_surf_mesh.m``), transposed to
vertex-major:

    vertexNbors            (N, max_neigh)         int32 1-indexed (0 = absent)
    vertexFaces            (N, max_faces)         int32 1-indexed (0 = absent)

Public API:
    compute_topology(vertices, faces) -> (vertex_nbors, vertex_faces)

The neighbor ordering matches the MATLAB C-mex
``MARS_convertFaces2VertNbors`` algorithm: walk the face matrix in
order, and for each face (v0, v1, v2) add the three edges (v0-v1,
v0-v2, v1-v2) as undirected neighbor pairs only the first time each
edge is seen. This yields a fan-style ring-walk per vertex when faces
are stored in FreeSurfer's CCW convention.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .mesh_topology import compute_topology


__all__ = [
    "compute_topology",
]
