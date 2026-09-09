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

Public API — used by Step2Pipeline.load_inputs_sparse (backend='gpu'):
    Step2Layout            — the static P-layout over the boundary
                             mask's nonzero cells (CSR + CSC views).
    build_step2_layout     — build it from the two per-hemi masks.
    build_step2_layout_dense — reference builder from a dense (N, L)
                             mask (tests).
    layouts_equal          — structural equality of two layouts.
    layout_to_device       — upload the six index arrays to cupy.
    Step2SparseInputs      — the sparse backend's input bundle.
    load_step2_sparse_inputs — read one from a project dir.

Test / bench adapters:
    InMemoryProfileLoader  — wraps an in-RAM (S, N, T, D) array as a
                             SubjectProfileLoader (for synthetic
                             fixtures).
    InMemoryGradientLoader — same, for (S, T, N, D_grad).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .load_group_mtc import load_group_mtc
from .sparse_inputs import Step2SparseInputs, load_step2_sparse_inputs
from .sparse_layout import (
    Step2Layout,
    build_step2_layout,
    build_step2_layout_dense,
    layout_to_device,
    layouts_equal,
)
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
    "Step2Layout",
    "build_step2_layout",
    "build_step2_layout_dense",
    "layouts_equal",
    "layout_to_device",
    "Step2SparseInputs",
    "load_step2_sparse_inputs",
]
