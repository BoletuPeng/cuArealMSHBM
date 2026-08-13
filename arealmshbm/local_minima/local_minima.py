"""local_minima.py

Per-column local-minima on a K-nearest-neighbor mesh graph. Direct
numba port of ``CBIG_SPGrad_find_minima.m``. A vertex
is a local minimum on column k iff its value is strictly less than
every K-hop neighbor's value on the same column.

The MATLAB form sentinels NaN-padded slots by appending an ``+inf``
row to the data. We replicate that exactly.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# fastmath=False on purpose: the neighbor table uses NaN as a padding
# sentinel and the hot loop branches on ``np.isnan(nb_f)``; fastmath
# allows the compiler to assume no NaNs and elide that check on some
# numba versions, which would mis-treat padding slots as valid -1
# neighbors and corrupt the minima mask.
@njit(cache=True, parallel=True, fastmath=False)
def find_minima(data: np.ndarray, K_neighbors: np.ndarray) -> np.ndarray:
    """Locate per-column local minima of ``data`` on a K-NN graph.

    Parameters
    ----------
    data : (N, K) fp32
        Smoothed gradient metric stack. Each column is independent.
    K_neighbors : (N, M) fp64
        Per-vertex neighbor table. Column 0 is self (1-indexed cortex
        ID); columns 1..M-1 hold K-hop neighbor IDs (1-indexed; NaN
        for absent slots).

    Returns
    -------
    minima : (N, K) bool
        ``True`` where the vertex is strictly less than every K-hop
        neighbor on that column.
    """
    N, K = data.shape
    n_cols = K_neighbors.shape[1]   # col 0 = self, cols 1.. = neighbors

    out = np.ones((N, K), dtype=np.bool_)

    for i in prange(N):
        v = data[i]                                  # (K,)
        for k in range(K):
            is_min = True
            for m in range(1, n_cols):
                nb_f = K_neighbors[i, m]
                if np.isnan(nb_f):
                    # padding slot acts like data[+inf row] = +inf →
                    # data[i,k] < +inf is True, doesn't change is_min
                    continue
                nb_idx = int(nb_f) - 1               # 1-indexed → 0-indexed
                if v[k] >= data[nb_idx, k]:
                    is_min = False
                    break
            out[i, k] = is_min

    return out
