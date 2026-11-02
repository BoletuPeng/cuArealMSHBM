"""step2_init

Step-2 group-prior init leaves (Mode B).

Two production entry points cover the stock CBIG initialisation
(``CBIG_ArealMSHBM_{g,d}MSHBM_estimate_group_priors_parent.m``: the
boundary mask and the mask / normalise / ``theta_num`` block;
``..._child.m``: the ``ini_s_lambda`` block):

    build_step2_boundary_mask — mode-dispatched bilateral block-diagonal
                                radius mask
    compose_init_state        — fused per-sub soft masked posterior +
                                theta, written as the final
                                ``(S, N, L) fp32`` Params["s_lambda"]
                                buffer. Streams per subject from a
                                ``SubjectProfileLoader``.
    init_kappa_T              — the fp32 logit scale both backends use.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .boundary_mask import build_step2_boundary_mask
from .compose_init_state import compose_init_state, init_kappa_T

__all__ = [
    "build_step2_boundary_mask",
    "compose_init_state",
    "init_kappa_T",
]
