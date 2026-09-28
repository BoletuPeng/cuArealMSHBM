"""_mesh_fixture.py — face-based neighbour table for the radius_mask tests.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np

def _nbors_from_faces(faces, n):
    """(max_neigh, V) 1-indexed neighbour table, 0 = absent slot — the
    layout ``load_avg_mesh`` bundles ship."""
    F = np.asarray(faces, dtype=np.int64)
    a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    adj = [set() for _ in range(n)]
    for u, v in zip(a.tolist(), b.tolist()):
        adj[u].add(v)
        adj[v].add(u)
    m = max(len(s) for s in adj)
    tab = np.zeros((m, n), dtype=np.int64)
    for v, s in enumerate(adj):
        # deliberately NOT sorted: the builder must sort per column.
        for k, u in enumerate(sorted(s, reverse=True)):
            tab[k, v] = u + 1
    return tab

