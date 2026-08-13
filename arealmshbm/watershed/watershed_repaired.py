"""watershed_repaired.py

Repaired numba CPU kernel — per-column threshold-sweep watershed
flooding visiting **all** K columns. The unrepaired upstream (WashU)
loop had a dead ``else`` in its 4-way chunking, silently skipping the
trailing ``K mod 4`` columns — they were reported as all-boundary, a
uniform ``+3/375 ≈ 0.008`` floor on ``edge_density``. This kernel
drops the chunking entirely.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>, repair of WashU
``watershed_algorithm_all_par_cifti.m``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def watershed_algorithm_repaired(edge_metrics: np.ndarray,
                                 minima: np.ndarray,
                                 neighbors: np.ndarray,
                                 stepnum: int,
                                 fracmaxh: float) -> np.ndarray:
    """Threshold-sweep watershed flooding (per-column), all K cols."""
    N, K = edge_metrics.shape
    R = neighbors.shape[1] - 1   # ring-1 neighbor cols (drop self)

    minh_val = np.float64(edge_metrics.min())
    maxh_val = np.float64(edge_metrics.max())
    stoph = maxh_val * np.float64(fracmaxh)
    step = (maxh_val - minh_val) / np.float64(stepnum)

    hiter = np.arange(minh_val, stoph + step / 2.0, step).astype(np.float32)
    n_h = hiter.shape[0]

    labels = np.zeros((N, K), dtype=np.int32)

    for k in prange(K):
        np.random.seed(k)

        label = np.zeros(N, dtype=np.int32)
        watershed_zones = np.zeros(N, dtype=np.bool_)

        n_seed = 0
        for i in range(N):
            if minima[i, k]:
                n_seed += 1
        if n_seed > 0:
            ids = np.empty(n_seed, dtype=np.int32)
            for j in range(n_seed):
                ids[j] = j + 1
            np.random.shuffle(ids)
            j = 0
            for i in range(N):
                if minima[i, k]:
                    label[i] = ids[j]
                    j += 1

        active = np.empty(N, dtype=np.int64)

        for ih in range(n_h):
            h = hiter[ih]

            n_active = 0
            for i in range(N):
                if (edge_metrics[i, k] < h
                        and label[i] == 0
                        and not watershed_zones[i]):
                    active[n_active] = i
                    n_active += 1
            if n_active == 0:
                continue

            np.random.shuffle(active[:n_active])

            for ai in range(n_active):
                v = active[ai]
                lo = np.int32(0)
                hi = np.int32(0)
                seen = False
                for r in range(1, R + 1):
                    nb_f = neighbors[v, r]
                    if np.isnan(nb_f):
                        continue
                    nb = int(nb_f) - 1
                    lab = label[nb]
                    if lab == 0:
                        continue
                    if not seen:
                        lo = lab
                        hi = lab
                        seen = True
                    else:
                        if lab < lo:
                            lo = lab
                        if lab > hi:
                            hi = lab
                if not seen:
                    continue
                if lo == hi:
                    label[v] = lo
                else:
                    watershed_zones[v] = True

        for i in range(N):
            labels[i, k] = label[i]

    return labels
