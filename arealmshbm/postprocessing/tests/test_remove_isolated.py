"""test_remove_isolated.py

``remove_isolated_surface_components`` against an independent brute-force
reference (BFS components, explicit vote counting) on synthetic grid
meshes, plus the rules one at a time: size cutoff, own-label and
medial-wall exclusion, smallest-label tie-break, votes read from the
entry-state labels, no-candidate keeps the label, input not mutated.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.check_connectedness.tests._grid_fixture import grid_mesh
from arealmshbm.postprocessing import remove_isolated_surface_components


def _reference(labels: np.ndarray, nbors: np.ndarray, thr: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    adj = [[int(u) - 1 for u in nbors[:, v] if u != 0] for v in range(n)]
    comp = np.full(n, -1, dtype=np.int64)
    comps = []
    for v in range(n):
        if comp[v] >= 0:
            continue
        comp[v] = len(comps)
        stack, members = [v], []
        while stack:
            x = stack.pop()
            members.append(x)
            for u in adj[x]:
                if comp[u] < 0 and labels[u] == labels[v]:
                    comp[u] = comp[v]
                    stack.append(u)
        comps.append(members)
    out = labels.copy()
    for members in comps:
        if len(members) >= thr:
            continue
        own = labels[members[0]]
        votes = [int(labels[u]) for x in members for u in adj[x]
                 if labels[u] != own and labels[u] != 0]
        if not votes:
            continue
        counts: dict = {}
        for c in votes:
            counts[c] = counts.get(c, 0) + 1
        out[members] = min(counts, key=lambda c: (-counts[c], c))
    return out


@pytest.mark.parametrize("rows,cols,n_labels,thr,seed", [
    (6, 9, 4, 5, 0), (12, 15, 6, 5, 1), (12, 15, 6, 2, 2), (12, 15, 3, 8, 3),
    (20, 20, 10, 3, 4), (20, 20, 2, 5, 5), (7, 30, 5, 4, 6),
])
def test_random_grids_match_reference(rows, cols, n_labels, thr, seed):
    _, nbors = grid_mesh(rows, cols)
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_labels + 1, size=rows * cols)   # 0 = medial wall
    labels[:cols] = 1                                          # one solid stripe
    before = labels.copy()
    got = remove_isolated_surface_components(labels, nbors, thr)
    exp = _reference(labels, nbors, thr)
    assert got.dtype == np.int64 and got.shape == labels.shape
    assert np.array_equal(got, exp)
    assert np.array_equal(labels, before), "input mutated"
    assert np.any(got != labels), "case exercises nothing"


def test_votes_read_entry_state_labels():
    """Two tiny components next to each other: each votes on the ORIGINAL
    labels, so relabelling one must not feed the other's vote."""
    rows, cols = 5, 5
    _, nbors = grid_mesh(rows, cols)
    lab = np.full(rows * cols, 7, dtype=np.int64)              # sea of 7 (21 vertices)
    A, B, C, Dv = 0, 1, cols, 2                                # (0,0) (0,1) (1,0) (0,2)
    lab[A] = 9
    lab[B] = 5
    lab[C] = 5
    lab[Dv] = 9
    got = remove_isolated_surface_components(lab, nbors, 5)
    # A sees B=5 and C=5 -> 5. B sees A=9, D=9, (1,1)=7 -> 9 (entry state!);
    # an in-place sweep that relabelled A first would see A=5 (B's own
    # label, excluded) and give B {9, 7} -> tie -> 7.
    assert got[A] == 5 and got[B] == 9
    assert got[C] == 7 and got[Dv] == 7
    assert np.all(got[[i for i in range(rows * cols) if i not in (A, B, C, Dv)]] == 7)


def test_size_cutoff_is_strict():
    """size == thr stays; size == thr - 1 goes."""
    rows, cols = 4, 8
    _, nbors = grid_mesh(rows, cols)
    lab = np.full(rows * cols, 1, dtype=np.int64)
    lab[0:5] = 2                                               # 5-vertex run on row 0
    lab[cols:cols + 4] = 3                                     # 4-vertex run on row 1
    got = remove_isolated_surface_components(lab, nbors, 5)
    assert np.all(got[0:5] == 2)
    assert np.all(got[cols:cols + 4] == 1)


def test_tie_breaks_on_smallest_label():
    rows, cols = 3, 3
    _, nbors = grid_mesh(rows, cols)
    lab = np.zeros(rows * cols, dtype=np.int64)
    lab[4] = 9                                                 # centre, size 1
    lab[1], lab[7] = 6, 6                                      # up / down -> two votes for 6
    lab[3], lab[5] = 4, 4                                      # left / right -> two votes for 4
    got = remove_isolated_surface_components(lab, nbors, 2)
    assert got[4] == 4


def test_medial_wall_and_own_label_are_not_candidates():
    rows, cols = 3, 3
    _, nbors = grid_mesh(rows, cols)
    lab = np.zeros(rows * cols, dtype=np.int64)               # everything medial wall...
    lab[4] = 9                                                 # ...but the centre
    assert np.array_equal(remove_isolated_surface_components(lab, nbors, 2), lab)
    lab[1] = 9                                                 # own label above: still no candidate
    lab[7] = 3                                                 # one real candidate below
    got = remove_isolated_surface_components(lab, nbors, 3)    # {4, 1} is a size-2 component
    assert got[4] == 3 and got[1] == 3


def test_validation():
    _, nbors = grid_mesh(3, 3)
    with pytest.raises(ValueError, match="labels must be 1D"):
        remove_isolated_surface_components(np.zeros((3, 3)), nbors, 5)
    with pytest.raises(ValueError, match="second axis"):
        remove_isolated_surface_components(np.zeros(8), nbors, 5)
    with pytest.raises(ValueError, match="abs_threshold must be positive"):
        remove_isolated_surface_components(np.zeros(9), nbors, 0)
