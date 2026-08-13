"""step2_io

Data-I/O leaves for the Python port of CBIG MSHBM step-2
(group prior estimation).

Public API — used by Step2Pipeline.load_inputs:
    SubjectProfileLoader   — per-subject BOLD profile loader. Reads
                             paths from ``cohort.json`` at construction;
                             exposes ``load_into(s, out)`` for streaming
                             one subject's (N, T, D) slab into a re-used
                             scratch slot. Per-session demean +
                             L2-row-norm runs in a fused numba kernel.
    SubjectGradientLoader  — per-subject diffusion-embedding loader.
                             Same cohort.json contract; sniffs .npy /
                             .mat by suffix on the cohort path.
    load_group_mtc         — read step-1 ``group.mat``.

Test / bench adapters:
    InMemoryProfileLoader  — wraps an in-RAM (S, N, T, D) array as a
                             SubjectProfileLoader (for synthetic
                             fixtures).
    InMemoryGradientLoader — same, for (S, T, N, D_grad).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .load_group_mtc import load_group_mtc
from .subject_loaders import (
    InMemoryGradientLoader,
    InMemoryProfileLoader,
    SubjectGradientLoader,
    SubjectProfileLoader,
)

__all__ = [
    "load_group_mtc",
    "SubjectProfileLoader",
    "SubjectGradientLoader",
    "InMemoryProfileLoader",
    "InMemoryGradientLoader",
]
