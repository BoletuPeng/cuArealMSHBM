"""test_seed_vertices.py — pin the RSFC seed set on the staged mesh.

``N_SEED_VERTS`` is the sole source of the seed count, and a wrong
value fails SILENTLY: the seed mask selects another set of vertices,
Mode B trains a prior of another D, and nothing downstream objects. So
it is pinned against two things that do not come from the constant —
the geometry of the sphere bundle and the profile dimension of CBIG's
HCP prior (checked when one is staged; this repository does not ship
it).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import arealmshbm
from arealmshbm.generate_profiles.profiles import N_SEED_VERTS


def _mesh(hemi: str, surface: str):
    from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
    try:
        return load_avg_mesh(hemi, surface)
    except FileNotFoundError as e:
        pytest.skip(f"avg_mesh bundle not staged: {e}")


def _nearest_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle from each unit vector of ``a`` to its nearest other one in
    ``b``, in degrees."""
    c = a @ b.T
    c[c > 1.0 - 1e-12] = -1.0          # a vertex is not its own neighbour
    return np.degrees(np.arccos(c.max(axis=1)))


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_seed_prefix_is_exactly_the_fsaverage3_icosphere(hemi):
    """The first ``N_SEED_VERTS`` sphere vertices are one icosphere
    level: evenly spaced at ~8 degrees. The next vertex belongs to the
    level below and sits at half that spacing, so a prefix one vertex
    longer or shorter is not a level."""
    v = np.asarray(_mesh(hemi, "sphere")["vertices"], dtype=np.float64)
    v = v.T if v.shape[0] == 3 else v
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    seeds, after = u[:N_SEED_VERTS], u[N_SEED_VERTS:N_SEED_VERTS + 1]

    assert _nearest_deg(seeds, seeds).min() > 6.0
    assert _nearest_deg(after, seeds)[0] < 6.0


def test_seed_count_is_the_hcp_prior_dimension():
    """D — the seed vertices off the medial wall, both hemispheres — is
    the row count of ``mu`` in every group prior trained on this
    scheme. The prior half needs CBIG's HCP prior staged under
    ``arealmshbm/data/group_priors/`` (see ``arealmshbm/data/README.md``)
    and is skipped without it."""
    from arealmshbm.data_io.load_group_prior import load_group_prior

    d = sum(int((np.asarray(_mesh(h, "inflated")["MARS_label"]).ravel()
                 [:N_SEED_VERTS] == 2).sum()) for h in ("lh", "rh"))
    assert d == 1175
    prior_path = (
        Path(arealmshbm.__file__).parent / "data" / "group_priors"
        / "HCP_40sub" / "100" / "gMSHBM" / "beta5" / "Params_Final.mat")
    if not prior_path.exists():
        pytest.skip(f"HCP group prior not staged at {prior_path}")
    assert load_group_prior(prior_path)["mu"].shape[0] == d
