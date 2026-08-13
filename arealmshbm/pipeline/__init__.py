"""arealmshbm.pipeline — unified project-driven pipeline driver.

A *project* is a directory under ``projects/<name>/`` representing one
parcellation campaign on one dataset. The only user-provided input is
``bold_inputs.json`` (BOLD time-series manifest) + ``pipeline_config.json``
(mode + hyperparams). Everything else under the project dir is **cache**
— produced, owned, and overwritten by the driver.

The driver dispatches across three modes:

  * ``modeA_single`` — one subject, requires a pre-existing group prior.
  * ``modeA_batch``  — K subjects, requires a pre-existing group prior.
  * ``modeB_train_prior`` — K subjects, runs step2 to train a new prior.

Public surface:
    BoldInputs        — parsed bold_inputs.json
    PipelineConfig    — parsed pipeline_config.json
    ProjectLayout     — path conventions inside a project
    Pipeline          — the orchestrator (Pipeline(project_dir).run())

CLI: ``python -m arealmshbm.pipeline projects/<name>``

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from .config import PipelineConfig, PipelineMode, PipelineVariant
from .driver import Pipeline
from .inputs import BoldInputs, BoldInputSession, BoldInputSubject
from .layout import ProjectLayout

__all__ = [
    "BoldInputs",
    "BoldInputSession",
    "BoldInputSubject",
    "Pipeline",
    "PipelineConfig",
    "PipelineMode",
    "PipelineVariant",
    "ProjectLayout",
]
