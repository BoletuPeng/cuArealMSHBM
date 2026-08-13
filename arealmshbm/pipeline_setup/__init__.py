"""pipeline_setup

Derived setup arrays the EM body needs but the raw inputs don't carry:
boundary_mask, neighborhood, row_idx/col_idx. Potts edge weights are
baked into the V_lambda kernel so no V_same/V_diff matrices flow.

Public API:
    build_pipeline_setup — assemble all derived arrays in one call.
    build_boundary_mask  — combine per-hemi sparse masks into one (N, L).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .pipeline_setup import build_pipeline_setup, build_boundary_mask

__all__ = ["build_pipeline_setup", "build_boundary_mask"]
