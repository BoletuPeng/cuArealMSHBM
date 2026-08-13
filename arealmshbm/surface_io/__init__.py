"""surface_io

Surface mesh readers for step-0 (gradient prior). Detects the file
format from extension and dispatches to the right backend.

Public API:
    read_surface_mesh(path) -> {'vertices': (N,3) fp32,
                                'faces':    (F,3) int32 0-indexed}

Supported formats:
    - ``.surf.gii`` (GIFTI)         — read via ``nibabel.gifti``;
                                      darray 0 = pointset (verts),
                                      darray 1 = triangle (faces, 0-indexed).
    - ``.sphere`` (FreeSurfer bin)  — read via
                                      ``nibabel.freesurfer.io.read_geometry``;
                                      faces are 0-indexed in the FS format.

Both readers normalize to fp32 vertices and int32 0-indexed faces — the
common contract every step-0 leaf consumes.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .surface_io import read_surface_mesh


__all__ = [
    "read_surface_mesh",
]
