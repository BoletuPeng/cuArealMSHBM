"""step2_init

Step-2 group-prior init leaves (Mode B).

Two production entry points cover the MATLAB
``CBIG_ArealMSHBM_cdgMSHBM_estimate_group_priors_sequential``
initialization block (lines 117-180):

    build_step2_boundary_mask — mode-dispatched bilateral block-diagonal
                                radius mask (MATLAB lines 166-170)
    compose_init_state        — fused per-sub argmax + theta + final
                                ``(S, N, L) fp32`` Params["s_lambda"]
                                buffer. Replaces the older split path
                                (init_s_lambda → init_theta_from_s_lambda
                                → fp64→fp32 cast + (N, L, S) → (S, N, L)
                                transpose). Streams per subject from a
                                ``SubjectProfileLoader``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .boundary_mask import build_step2_boundary_mask
from .compose_init_state import compose_init_state

__all__ = [
    "build_step2_boundary_mask",
    "compose_init_state",
]
