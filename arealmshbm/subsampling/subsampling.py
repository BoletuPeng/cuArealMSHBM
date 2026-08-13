"""subsampling.py

Random-index draw for the FC speed-up subsampling.

    np.random.seed(scan_idx) ; np.random.permutation(num_vertices)

Seeded per scan so each scan's draws are deterministic and
reproducible across Python runs. The MATLAB Mersenne-twister sequence
is not reproducible inside numpy; downstream consumers tolerate any
uniform permutation (algorithm-class equivalence).

RNG isolation: numba's ``@njit`` RNG state is independent of the
Python-side numpy RNG — calls to ``np.random.seed`` inside this
kernel do NOT mutate the caller's process-global numpy state, so
co-located callers (e.g. step 3 running in the same process as
step 0 inside the unified driver) are unaffected.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def set_downsample_params(
    num_vertices: int,
    sub_verts: int,
    sub_FC: int,
    scan_idx: int,
):
    """Draw the (randinds_verts, randinds_FC) pair for one scan.

    Both outputs are 0-indexed int64 arrays of lengths
    ``round(num_vertices / sub_verts)`` and ``round(num_vertices / sub_FC)``.
    """
    n1 = int(round(num_vertices / sub_verts))
    n2 = int(round(num_vertices / sub_FC))
    np.random.seed(scan_idx)
    perm_v = np.random.permutation(num_vertices).astype(np.int64)
    perm_f = np.random.permutation(num_vertices).astype(np.int64)
    return perm_v[:n1], perm_f[:n2]
