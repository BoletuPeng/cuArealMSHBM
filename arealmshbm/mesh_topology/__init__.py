"""mesh_topology

Build the per-vertex / per-face mesh topology that the CBIG SPGrad
pipeline ships in its ``sbjMesh`` struct (see
``CBIG_SPGrad_read_surf_mesh.m``):

    vertexNbors            (max_neigh, N)         int   1-indexed (0 = absent)
    vertexFaces            (max_faces, N)         int   1-indexed (0 = absent)
    faceAreas              (F,)                   fp32
    metricVerts            (3, N)                 fp32  rescaled to r=100 sphere
    surface_scaling_factor scalar                 fp32
    vertexDistSq2Nbors     (max_neigh, N)         fp32

Public API:
    compute_topology(vertices, faces) -> dict

The neighbor ordering matches the MATLAB C-mex
``MARS_convertFaces2VertNbors`` algorithm: walk the face matrix in
order, and for each face (v0, v1, v2) add the three edges (v0-v1,
v0-v2, v1-v2) as undirected neighbor pairs only the first time each
edge is seen. This yields a fan-style ring-walk per vertex when faces
are stored in FreeSurfer's CCW convention. ``data_io/load_avg_mesh.py``
uses lexicographic ordering — *don't* import this module to satisfy
that one (and vice versa); see that module's WARNING.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .mesh_topology import compute_topology


__all__ = [
    "compute_topology",
]
