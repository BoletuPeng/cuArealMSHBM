"""test_mesh_csr.py — the mesh-CSR and mask-to-sparse helpers.

What is pinned here:

  * ``_build_mesh_csr`` reading adjacency out of the bundle's
    ``vertexNbors`` table is bit-identical (``np.array_equal`` on
    indptr / indices / weights) to the previous implementation, which
    deduplicated the 6*F directed face edges through ``np.unique``.
    Checked on a synthetic icosphere and on both shipped fsaverage6
    bundles;
  * every documented precondition of the new argument raises a named
    ``ValueError`` -- including the three graph invariants the retired
    faces+``np.unique`` route used to guarantee by construction and a
    raw bundle table cannot be assumed to have: no self-loop, no
    duplicate neighbour, symmetry;
  * ``_mask_to_csc`` equals ``scipy.sparse.csc_matrix(mask.astype(f8))``
    down to indptr / indices / data.

Run::

    python -m pytest arealmshbm/radius_mask/tests/test_mesh_csr.py -v

The fsaverage6 cases skip when the shipped ``avg_mesh`` bundle is not
staged in this checkout (see ``arealmshbm/data/README.md``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from arealmshbm.radius_mask._common import _build_mesh_csr, _mask_to_csc


# ─────────────────────────────────────────────────────────────────────
# Reference: the pre-vertexNbors builder, verbatim.
# ─────────────────────────────────────────────────────────────────────
def _build_mesh_csr_from_faces(vertices, faces):
    V_xyz = np.asarray(vertices)
    if V_xyz.shape[0] == 3 and V_xyz.shape[1] != 3:
        V_xyz = V_xyz.T
    V_xyz = np.ascontiguousarray(V_xyz, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    e1 = V_xyz[F[:, 1]] - V_xyz[F[:, 0]]
    e2 = V_xyz[F[:, 2]] - V_xyz[F[:, 0]]
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1).sum()
    factor = float(np.sqrt(4.0 * np.pi * 100.0 * 100.0 / area))
    V_fp32 = (V_xyz * factor).astype(np.float32)
    n = V_fp32.shape[0]

    e_a = F[:, [0, 1, 2]].ravel()
    e_b = F[:, [1, 2, 0]].ravel()
    rows = np.concatenate([e_a, e_b])
    cols = np.concatenate([e_b, e_a])
    diff = V_fp32[rows] - V_fp32[cols]
    w = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64)).astype(np.float32)

    key = rows.astype(np.int64) * np.int64(n) + cols.astype(np.int64)
    _, uidx = np.unique(key, return_index=True)
    rows = rows[uidx]; cols = cols[uidx]; w = w[uidx]

    csr = sp.csr_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32)
    csr.sort_indices()
    return (csr.indptr.astype(np.int64),
            csr.indices.astype(np.int64),
            np.ascontiguousarray(csr.data, dtype=np.float32))


def _nbors_from_faces(faces, n):
    """(max_neigh, V) 1-indexed neighbour table, 0 = absent slot — the
    layout ``load_avg_mesh`` bundles ship."""
    F = np.asarray(faces, dtype=np.int64)
    a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    adj = [set() for _ in range(n)]
    for u, v in zip(a.tolist(), b.tolist()):
        adj[u].add(v)
        adj[v].add(u)
    m = max(len(s) for s in adj)
    tab = np.zeros((m, n), dtype=np.int64)
    for v, s in enumerate(adj):
        # deliberately NOT sorted: the builder must sort per column.
        for k, u in enumerate(sorted(s, reverse=True)):
            tab[k, v] = u + 1
    return tab


@pytest.fixture(scope="module")
def icosphere_mesh():
    from arealmshbm.icosphere import make_icosphere
    verts, faces = make_icosphere(200, radius=100.0)
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    return verts, faces, _nbors_from_faces(faces, verts.shape[0])


def _load_bundle(hemi):
    from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
    try:
        return load_avg_mesh(hemi, "fsaverage6", "inflated")
    except FileNotFoundError as e:  # pragma: no cover - staging dependent
        pytest.skip(f"avg_mesh bundle not staged: {e}")


def test_csr_matches_face_builder_on_icosphere(icosphere_mesh):
    verts, faces, nbors = icosphere_mesh
    ref = _build_mesh_csr_from_faces(verts, faces)
    got = _build_mesh_csr(verts, faces, nbors)
    for name, a, b in zip(("indptr", "indices", "weights"), ref, got):
        assert a.dtype == b.dtype, name
        assert np.array_equal(a, b), name


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_csr_matches_face_builder_on_fsaverage6(hemi):
    mesh = _load_bundle(hemi)
    ref = _build_mesh_csr_from_faces(mesh["vertices"], mesh["faces"])
    got = _build_mesh_csr(mesh["vertices"], mesh["faces"], mesh["vertexNbors"])
    for name, a, b in zip(("indptr", "indices", "weights"), ref, got):
        assert a.dtype == b.dtype, name
        assert np.array_equal(a, b), name
    # ascending per row, both directions present.
    indptr, indices, _ = got
    for v in range(0, indptr.shape[0] - 1, 997):
        row = indices[indptr[v]:indptr[v + 1]]
        assert np.all(np.diff(row) > 0)


def test_csr_rejects_bad_nbors(icosphere_mesh):
    verts, faces, nbors = icosphere_mesh
    with pytest.raises(ValueError, match=r"vertex_nbors must be \(max_neigh"):
        _build_mesh_csr(verts, faces, nbors.T)
    bad = nbors.copy()
    bad[0, 0] = nbors.shape[1] + 5
    with pytest.raises(ValueError, match="1-indexed vertex ids"):
        _build_mesh_csr(verts, faces, bad)
    bad = nbors.copy()
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="1-indexed vertex ids"):
        _build_mesh_csr(verts, faces, bad)


@pytest.mark.parametrize("shape,density", [((1, 1), 1.0), ((7, 3), 0.5),
                                           ((4096, 37), 0.2), ((512, 5), 0.0)])
def test_mask_to_csc_matches_scipy(shape, density):
    rng = np.random.default_rng(0xC0FFEE)
    mask = (rng.random(shape) < density).astype(np.uint8)
    got = _mask_to_csc(mask)
    ref = sp.csc_matrix(mask.astype(np.float64))
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert np.array_equal(got.indptr.astype(np.int64),
                          ref.indptr.astype(np.int64))
    assert np.array_equal(got.indices.astype(np.int64),
                          ref.indices.astype(np.int64))
    assert np.array_equal(got.data, ref.data)
    assert (got != ref).nnz == 0


def test_mask_to_csc_rejects_bad_input():
    with pytest.raises(ValueError, match="must be 2-D"):
        _mask_to_csc(np.zeros(4, dtype=np.uint8))
    with pytest.raises(ValueError, match="must be uint8"):
        _mask_to_csc(np.zeros((4, 2), dtype=np.float64))


def test_csr_rejects_asymmetric_nbors(icosphere_mesh):
    """The retired faces route built the reverse edge by construction;
    the table route has to check for it, or a corrupt bundle silently
    yields a directed graph and wrong geodesics."""
    verts, faces, nbors = icosphere_mesh
    bad = nbors.copy()
    # drop vertex 0 from its first neighbour's list, keeping the
    # forward edge 0 -> u. counts change, so the graph goes directed.
    u = int(bad[0, 0]) - 1
    col = bad[:, u]
    col[col == 1] = 0
    bad[:, u] = col
    with pytest.raises(ValueError, match="not symmetric"):
        _build_mesh_csr(verts, faces, bad)


def test_csr_rejects_duplicate_and_self_loop(icosphere_mesh):
    verts, faces, nbors = icosphere_mesh
    dup = nbors.copy()
    dup[1, 0] = dup[0, 0]                      # same neighbour twice
    with pytest.raises(ValueError, match="same neighbour twice"):
        _build_mesh_csr(verts, faces, dup)
    loop = nbors.copy()
    empty = int(np.argmin(loop[:, 0] > 0))     # first free slot of col 0
    loop[empty, 0] = 1                         # vertex 0 lists itself
    with pytest.raises(ValueError, match="lists itself"):
        _build_mesh_csr(verts, faces, loop)


def test_csr_accepts_the_real_bundles():
    """The shipped fsaverage6 tables must pass all three checks (they
    are what production runs on)."""
    for hemi in ("lh", "rh"):
        mesh = _load_bundle(hemi)
        _build_mesh_csr(mesh["vertices"], mesh["faces"], mesh["vertexNbors"])
