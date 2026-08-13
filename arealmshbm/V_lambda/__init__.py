"""V_lambda

Potts close-form MRF smoothness term (the V_lambda factor in the gMSHBM
E-step). Potts weights ``V_same=0, V_diff=1`` are baked into the kernel —
no edge-weight matrices are taken or stored.

Public API:
    Session               — caches neighborhood once; recomputes
                            V_lambda from theta + s_lambda each call.
    build_neighborhood    — concat + remap LH/RH ``vertex_nbors`` into a
                            single bilateral neighborhood.
    build_candidate_index — flatten column-major nonzero theta sites into
                            ``(row_idx, col_idx)`` pairs the kernel sweeps.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .v_lambda import Session
from .setup import (
    build_neighborhood,
    build_candidate_index,
)

__all__ = [
    "Session",
    "build_neighborhood",
    "build_candidate_index",
]
