"""step0_pipeline

End-to-end Python step-0 super-call (Mode A, fsaverage6).

Public API:
    Step0Config         — pipeline-config dataclass (every knob).
    Step0Pipeline       — single-subject load → run → save lifecycle.
    Step0Inputs         — bundled subject-independent loaded inputs.
    Step0Result         — bundled outputs of one ``run()``.

For multi-subject batch runs, use the unified pipeline driver at
``arealmshbm.pipeline.Pipeline`` with ``mode='modeA_batch'``.

Production path (gpu backend; cpu also available)::

    from arealmshbm.step0_pipeline import Step0Config, Step0Pipeline

    cfg = Step0Config(
        project_dir="projects/demo/sub-001",
        sub_id="sub-001",
        sess_list=("ses-01", "ses-02", "ses-03",
                   "ses-04", "ses-05", "ses-06"),
        backend="gpu",
    )
    with Step0Pipeline(cfg) as pipe:
        result = pipe.run_and_save()

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .config import Step0Config
from .pipeline import (
    Step0Pipeline,
    Step0Inputs,
    Step0Result,
)


__all__ = [
    "Step0Config",
    "Step0Pipeline",
    "Step0Inputs",
    "Step0Result",
]
