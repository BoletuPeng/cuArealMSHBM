"""step2_pipeline

End-to-end Python step-2 group-prior estimation (Mode B). Coupled
multi-subject EM that produces ``Params = (mu, theta, epsil, sigma,
kappa)``. Mirrors MATLAB's
``CBIG_ArealMSHBM_cdgMSHBM_estimate_group_priors_sequential.m`` under
the ``mode ∈ {gMSHBM, dMSHBM}`` dispatch; cMSHBM is not wired.

Two EM-body implementations sit behind ``Step2Config.backend``:

* ``'cpu'`` — the fused numba master at
  :mod:`arealmshbm.step2_em_iter_master`, driven by
  :func:`vmf_clustering_batch` over ``Step2Inputs``.
* ``'gpu'`` — the P-layout / bit-packed CuPy backend
  (``Step2SparseSession``), driven by :func:`em_body_sparse` over
  ``Step2SparseInputs``. Contract: ``docs/step2_sparse_design.md``.

Public API:
    Step2Config         — pipeline-config dataclass (mode + cohort + caps).
    Step2Pipeline       — load → init → EM → save lifecycle.
    Step2Inputs         — bundled outputs of load_inputs() (CPU backend).
    Step2SparseInputs   — bundled outputs of load_inputs_sparse() ('gpu').
    Step2Result         — bundled outputs of run().
    vmf_clustering_batch — the EM-body subgraph, CPU backend.
    em_body_sparse       — the EM-body subgraph, 'gpu' backend.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from arealmshbm.step2_io.sparse_inputs import Step2SparseInputs

from .config import Step2Config
from .pipeline import Step2Pipeline, Step2Inputs, Step2Result
from .vmf_clustering_batch import (
    SparseBatchResult,
    VmfBatchResult,
    em_body_sparse,
    vmf_clustering_batch,
)

__all__ = [
    "Step2Config",
    "Step2Pipeline",
    "Step2Inputs",
    "Step2SparseInputs",
    "Step2Result",
    "vmf_clustering_batch",
    "VmfBatchResult",
    "em_body_sparse",
    "SparseBatchResult",
]
