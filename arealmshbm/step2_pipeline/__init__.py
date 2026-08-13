"""step2_pipeline

End-to-end Python step-2 group-prior estimation (Mode B). Coupled
multi-subject EM that produces ``Params = (mu, theta, epsil, sigma,
kappa)``. Mirrors MATLAB's
``CBIG_ArealMSHBM_cdgMSHBM_estimate_group_priors_sequential.m`` under
the ``mode ∈ {gMSHBM, dMSHBM}`` dispatch; cMSHBM is not wired.

The EM-body subgraph is implemented by the fused numba master at
:mod:`arealmshbm.step2_em_iter_master`. The pipeline wrapper
:func:`vmf_clustering_batch` is a thin Python orchestrator over
``Step2EmIterSession``.

Public API:
    Step2Config         — pipeline-config dataclass (mode + cohort + caps).
    Step2Pipeline       — load → init → EM → save lifecycle.
    Step2Inputs         — bundled outputs of load_inputs().
    Step2Result         — bundled outputs of run().
    vmf_clustering_batch — the EM-body subgraph (used by Pipeline + tests).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .config import Step2Config
from .pipeline import Step2Pipeline, Step2Inputs, Step2Result
from .vmf_clustering_batch import vmf_clustering_batch, VmfBatchResult

__all__ = [
    "Step2Config",
    "Step2Pipeline",
    "Step2Inputs",
    "Step2Result",
    "vmf_clustering_batch",
    "VmfBatchResult",
]
