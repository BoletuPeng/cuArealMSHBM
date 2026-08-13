"""step3_pipeline

End-to-end Python step-3 super-call (Mode A).

Public API:
    Step3Config         — pipeline-config dataclass (every knob).
    Step3Pipeline       — single-subject load → setup → EM → save lifecycle.
    Step3Inputs         — bundled outputs of load + setup stages.
    Step3Result         — bundled outputs of one ``run()``.

For multi-subject batch runs, use the unified pipeline driver at
``arealmshbm.pipeline.Pipeline`` with ``mode='modeA_batch'``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .config import Step3Config
from .variant import VariantSpec, VALID_PIPELINE_TYPES
from .pipeline import (
    Step3Pipeline,
    Step3Inputs,
    Step3Result,
)

__all__ = [
    "Step3Config",
    "VariantSpec",
    "VALID_PIPELINE_TYPES",
    "Step3Pipeline",
    "Step3Inputs",
    "Step3Result",
]
