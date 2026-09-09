"""test_cpu_leaf_guards.py — backend-mismatch guards on the CPU leaf.

``compute_profile_arrays`` takes ``precomputed_bold_runs`` as a public
kwarg, so a caller can hand the CPU leaf device buffers (e.g. straight
out of ``read_subject_bold_gpu``). The guard sits ahead of every file
read, so the check below needs no project on disk -- and it is a
module-name test (``type(x).__module__`` under ``cupy``, so the CPU
leaf never imports cupy), so a stand-in with that module name
exercises it on the machines it exists for: the ones without cupy.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import pytest


class _DeviceArrayStandIn:
    """What the guard sees of a ``cupy.ndarray``: its module name."""
    __module__ = "cupy._core.core"


def test_compute_profile_arrays_cpu_rejects_cupy_precomputed():
    """compute_profile_arrays with cupy-resident
    precomputed_bold_runs must raise a named TypeError, not fail deep
    inside numba. Without the guard the failure surfaces inside the
    ``s_norm.T @ lh_norm`` BLAS dispatch with no diagnostic value."""
    from arealmshbm.generate_profiles import compute_profile_arrays

    cupy_runs = [(_DeviceArrayStandIn(), _DeviceArrayStandIn())]
    with pytest.raises(TypeError, match=r"compute_profile_arrays: "):
        compute_profile_arrays(
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            out_dir="/tmp/does-not-matter",
            sub="1", sess="1",
            precomputed_bold_runs=cupy_runs,
        )
