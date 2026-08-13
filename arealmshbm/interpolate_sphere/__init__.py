"""interpolate_sphere

Barycentric linear interpolation of vertex-attached data from one
sphere mesh to a set of target points on a sphere of (approximately)
the same radius. Port of CBIG / MARS ``MARS_linearInterpolate.m``
plus the C MEX search routine ``MARS_linearInterpolateAux.c`` /
``MARS_findFaces.h``.

Used by step-0 subgraph B (downsample full-resolution gradient edge
density onto the icosphere) and subgraph D (upsample diffusion-map
embedding back onto the full sphere).

Public API:
    linear_interpolate_sphere(target_points, source_verts, source_faces,
                              source_vertex_faces, data) -> np.ndarray

External shapes:
    target_points       : (M, 3) fp32 — query points on the target sphere.
    source_verts        : (N, 3) fp32 — vertex coordinates of the source mesh.
    source_faces        : (F, 3) int32, 0-indexed.
    source_vertex_faces : (N, max_faces_per_vert) int32 1-indexed,
                          ``0`` marks an absent slot. Pass ``None`` to
                          derive on the fly from ``source_faces``.
    data                : (..., N) any float dtype — values attached to
                          source vertices; leading dims are preserved.
    output              : (..., M) same dtype as ``data``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .interpolate_sphere import (
    linear_interpolate_sphere,
    build_vertex_faces,
)


__all__ = [
    "linear_interpolate_sphere",
    "build_vertex_faces",
]
