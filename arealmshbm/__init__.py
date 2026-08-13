"""arealmshbm

Python implementation of the Kong-2022 Areal multi-session
hierarchical Bayesian model (Areal-MSHBM) for individual-specific
cortical parcellation. Both Mode A (HCP-prior personalisation) and
Mode B (self-trained prior) are supported end-to-end.

Top-level entry: the project-driven driver under
:mod:`arealmshbm.pipeline`. The per-step super-calls re-exported below
are kept for targeted regression / profile runs:

    Step0Config / Step0Pipeline — RSFC gradient + diffusion embedding.
    Step2Config / Step2Pipeline — group prior estimation (Mode B).
    Step3Config / Step3Pipeline — individual parcellation EM.

Step 1 has no top-level re-export because it has no ``StepNPipeline``
class — the four subgraphs (generate_profiles, avg_profiles, ini_params,
radius_mask) are called directly from
:mod:`arealmshbm.pipeline.step1_runners`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .step0_pipeline import (
    Step0Config,
    Step0Pipeline,
    Step0Inputs,
    Step0Result,
)
from .step2_pipeline import (
    Step2Config,
    Step2Pipeline,
    Step2Inputs,
    Step2Result,
)
from .step3_pipeline import (
    Step3Config,
    Step3Pipeline,
    Step3Inputs,
    Step3Result,
)

__all__ = [
    "Step0Config",
    "Step0Pipeline",
    "Step0Inputs",
    "Step0Result",
    "Step2Config",
    "Step2Pipeline",
    "Step2Inputs",
    "Step2Result",
    "Step3Config",
    "Step3Pipeline",
    "Step3Inputs",
    "Step3Result",
]
