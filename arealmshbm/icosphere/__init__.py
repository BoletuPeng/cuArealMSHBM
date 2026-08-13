"""icosphere

Generate a class-I geodesic icosahedron (a.k.a. icosphere) sized to a
target vertex count, mirroring the topology of the wb_command
``-surface-create-sphere`` output that the SPGrad pipeline uses for
downsampling.

For a frequency-T subdivision of the regular icosahedron:
    V = 10·T² + 2
    F = 20·T²
    E = 30·T²

Workbench's ``-surface-create-sphere N`` chooses the minimum T such
that V(T) ≥ N. For N = 12801 this is T = 36 → V = 12962, matching the
on-disk ``lh_down_sphere_verts.npy``.

This module implements **frequency-T (class I) geodesic
subdivision**, not power-of-2 recursive subdivision — recursive can
only land on T ∈ {1, 2, 4, 8, 16, 32, 64, ...}, which doesn't include
36. Vertex *positions* will not match workbench's because workbench's
subdivision-traversal order and projection details are
implementation-defined; the spec only requires same vert count,
correct topology, and unit-100 sphere placement.

Public API:
    make_icosphere(num_target_verts, radius=100.0) -> (vertices, faces)

`make_icosphere` is a numba @njit Dispatcher (not a Python wrapper).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .icosphere import make_icosphere


__all__ = [
    "make_icosphere",
]
