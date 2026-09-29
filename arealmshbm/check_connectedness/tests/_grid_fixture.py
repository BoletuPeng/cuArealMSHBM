"""_grid_fixture.py — synthetic 4-neighbour grid meshes for the
connectedness and postprocessing tests.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np


def grid_mesh(rows, cols, x0=0.0, spacing=10.0):
    """4-neighbour rectangular grid; returns (vertices (3, n), nbors (4, n)).

    ``nbors`` follows the mesh-dict convention: MATLAB 1-indexed neighbour
    ids, 0 = absent slot.
    """
    n = rows * cols
    verts = np.zeros((3, n), dtype=np.float64)
    nbors = np.zeros((4, n), dtype=np.int64)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            verts[0, v] = x0 + c * spacing
            verts[1, v] = r * spacing
            k = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    nbors[k, v] = rr * cols + cc + 1     # 1-indexed
                    k += 1
    return verts, nbors


def mesh(verts, nbors):
    return {"vertices": verts, "vertexNbors": nbors}
