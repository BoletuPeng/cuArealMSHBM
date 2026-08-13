"""test_icosphere_nverts.py — pin the closed-form seed vertex count.

``_icosphere_nverts`` replaced a ``load_avg_mesh`` call as the SOLE source
of the RSFC seed vertex count in ``profiles.py``. Its failure mode is
SILENT: a wrong-but-smaller count sails past the ``MARS_label`` length
guard in ``compute_profile_arrays`` (which only rejects too-LARGE) and
yields a wrong seed mask → wrong K → wrong parcellation, with no
exception. Unlike the asset-loader changes it rides with (a missing
bundle is a loud ``FileNotFoundError``), this one cannot lean on the
deferred E2E gate — so it is pinned directly here, both against known
constants and against real staged geometry.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import pytest

from arealmshbm.generate_profiles.profiles import _icosphere_nverts


def test_icosphere_nverts_matches_known_constants():
    """10·4^order + 2 for the fsaverage orders the pipeline uses."""
    for mesh, n in [("fsaverage3", 642), ("fsaverage4", 2562),
                    ("fsaverage5", 10242), ("fsaverage6", 40962)]:
        assert _icosphere_nverts(mesh) == n


def test_icosphere_nverts_equals_bundle_vertex_count():
    """Pin the formula against the actual staged mesh, so a
    self-consistent typo (e.g. ``order**4`` vs ``4**order``) that still
    matches a hand-typed constant can't slip through. Skips cleanly when
    the asset bundle isn't staged (e.g. a bare CI checkout).
    """
    from arealmshbm.data_io import load_avg_mesh
    try:
        mesh = load_avg_mesh("lh", "fsaverage6", "inflated")
    except FileNotFoundError:
        pytest.skip("fsaverage6 bundle not staged")
    assert _icosphere_nverts("fsaverage6") == mesh["MARS_label"].shape[0]


def test_icosphere_nverts_rejects_non_icosphere_names():
    """A ``fsaverage``-prefixed non-icosphere name (fs_LR-style) must hit
    the clear guard, not a cryptic ``int()`` ValueError; a non-fsaverage
    name hits the first guard.
    """
    with pytest.raises(ValueError, match="icosphere"):
        _icosphere_nverts("fsaverage_LR32k")
    with pytest.raises(ValueError, match="not an fsaverage mesh"):
        _icosphere_nverts("fs_LR_32k")
