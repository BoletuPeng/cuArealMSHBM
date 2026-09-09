"""test_sssp_gpu.py — the batched delta-stepping SSSP behind central_sulcus.

What is pinned here, on a synthetic icosphere:

  * ``central_sulcus_distances_cupy`` reproduces the pull-based
    Bellman-Ford (:func:`bellman_ford_bounded_cupy`, the reference
    solver this replaced) **bit for bit** at every (source, relevant
    vertex) pair -- for a source count that is and is not a multiple of
    the 32-source block, and across bucket widths, which is the whole
    delta-stepping correctness claim: the schedule reorders
    relaxations, the fixed point does not move;
  * the per-parcel ``avg_dis`` built from those distances matches the
    numba ``central_sulcus_kernel`` (which runs a per-source Dijkstra
    with early exit) to fp64 round-off;
  * the kernel is run-to-run deterministic;
  * every documented precondition raises a named ``ValueError`` --
    including an out-of-range vertex id (which would otherwise be a
    silent out-of-bounds device write, not an error) and a mesh too
    large for the O(V) shared-memory frontier (which would otherwise
    be a raw ptxas trace).

And on real data, when the fsaverage6 bench baseline is staged:

  * the GPU supercall's ``.mat`` decodes equal to the baseline masks.

Run::

    python -m pytest arealmshbm/radius_mask/tests/test_sssp_gpu.py -v

Skips without cupy; the real-data case additionally skips when the
baseline ``.mat``, the atlas dir or the avg_mesh bundles are absent.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.radius_mask._common import _build_mesh_csr  # noqa: E402
from arealmshbm.radius_mask._kernels import (  # noqa: E402
    build_parcel_csr_kernel, central_sulcus_kernel)
from arealmshbm.radius_mask import _kernels_gpu as K  # noqa: E402

BASELINE_MAT = Path(os.environ.get("MSHBM_STEP1_BENCH_DIR",
                                   "testdata/step1_bench"),
                    "baseline_gpu/spatial_mask/"
                    "spatial_mask_fsaverage6.mat")


def _nbors_from_faces(faces, n):
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
        for k, u in enumerate(sorted(s)):
            tab[k, v] = u + 1
    return tab


@pytest.fixture(scope="module")
def sphere():
    from arealmshbm.icosphere import make_icosphere
    verts, faces = make_icosphere(400, radius=100.0)
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    n = verts.shape[0]
    indptr, indices, w = _build_mesh_csr(verts, faces,
                                         _nbors_from_faces(faces, n))
    return dict(V=n, indptr=indptr, indices=indices, weights=w,
                mean_w=float(w.mean(dtype=np.float64)))


def _bf_reference(sphere, src, rel):
    """Distances at ``rel`` from every source in ``src``, via the
    pull-based Bellman-Ford — the solver the SSSP kernel replaced."""
    V = sphere["V"]
    K_ = int(src.shape[0])
    dist = cp.full((V, K_), cp.float32(np.inf), dtype=cp.float32)
    dist[cp.asarray(src.astype(np.int64)), cp.arange(K_)] = 0.0
    scratch = cp.empty_like(dist)
    K.bellman_ford_bounded_cupy(
        cp.asarray(sphere["indptr"]), cp.asarray(sphere["indices"]),
        cp.asarray(sphere["weights"]), dist, scratch,
        radius=float("inf"), max_iters=5000)
    return cp.asnumpy(dist[cp.asarray(rel.astype(np.int64))])


@pytest.mark.parametrize("n_src", [32, 40, 96])
@pytest.mark.parametrize("delta_mult", [1.0, 10.0])
def test_sssp_matches_bellman_ford(sphere, n_src, delta_mult):
    V = sphere["V"]
    rng = np.random.default_rng(11)
    src = rng.choice(V, size=n_src, replace=False).astype(np.int32)
    rel = np.sort(rng.choice(V, size=37, replace=False)).astype(np.int32)

    tab = K.build_edge_table(sphere["indptr"], sphere["indices"],
                             sphere["weights"])
    got = cp.asnumpy(K.central_sulcus_distances_cupy(
        tab, src, rel, V, delta_mult * sphere["mean_w"]))
    assert got.shape[1] % 32 == 0 and got.shape[1] >= n_src
    ref = _bf_reference(sphere, src, rel)
    assert np.array_equal(got[:, :n_src], ref)


def test_sssp_is_deterministic(sphere):
    V = sphere["V"]
    rng = np.random.default_rng(3)
    src = rng.choice(V, size=64, replace=False).astype(np.int32)
    rel = np.sort(rng.choice(V, size=20, replace=False)).astype(np.int32)
    tab = K.build_edge_table(sphere["indptr"], sphere["indices"],
                             sphere["weights"])
    runs = [cp.asnumpy(K.central_sulcus_distances_cupy(
        tab, src, rel, V, 10.0 * sphere["mean_w"])) for _ in range(3)]
    assert np.array_equal(runs[0], runs[1])
    assert np.array_equal(runs[0], runs[2])


def test_avg_dis_matches_cpu_dijkstra_kernel(sphere):
    """Same per-parcel mean the CPU kernel accumulates inline."""
    V = sphere["V"]
    L = 4
    labels = (np.arange(V, dtype=np.int64) % L) + 1
    rng = np.random.default_rng(5)
    pre = np.sort(rng.choice(V, size=33, replace=False)).astype(np.int64)
    post = np.sort(rng.choice(np.setdiff1d(np.arange(V), pre), size=29,
                              replace=False)).astype(np.int64)
    relevant_parcels = np.array([1, 0, 1, 0], dtype=np.uint8)
    relevant_verts = relevant_parcels[labels - 1].astype(np.uint8)
    parcel_offs, parcel_inds = build_parcel_csr_kernel(labels, L)

    cpu = np.empty((2, L), dtype=np.float64)
    central_sulcus_kernel(sphere["indptr"], sphere["indices"],
                          sphere["weights"], pre, post,
                          parcel_offs, parcel_inds, L,
                          relevant_parcels, relevant_verts, cpu)

    rel = np.flatnonzero(relevant_verts).astype(np.int32)
    src = np.concatenate([pre, post]).astype(np.int32)
    tab = K.build_edge_table(sphere["indptr"], sphere["indices"],
                             sphere["weights"])
    g = K.central_sulcus_distances_cupy(
        tab, src, rel, V, K._SSSP_DELTA_MULT * sphere["mean_w"])
    s_pre = cp.asnumpy(g[:, :pre.size].astype(cp.float64).sum(axis=1))
    s_post = cp.asnumpy(
        g[:, pre.size:pre.size + post.size].astype(cp.float64).sum(axis=1))
    row_of = np.full(V, -1, dtype=np.int64)
    row_of[rel] = np.arange(rel.shape[0], dtype=np.int64)

    gpu = np.zeros((2, L), dtype=np.float64)
    for l in range(L):
        if relevant_parcels[l] == 0:
            continue
        vs = parcel_inds[parcel_offs[l]:parcel_offs[l + 1]]
        n_v = float(vs.shape[0])
        rows = row_of[vs]
        gpu[0, l] = float(s_pre[rows].sum()) / (n_v * float(pre.size))
        gpu[1, l] = float(s_post[rows].sum()) / (n_v * float(post.size))
    np.testing.assert_allclose(gpu, cpu, rtol=1e-12, atol=0.0)


def test_edge_table_preconditions(sphere):
    ip, ix, w = sphere["indptr"], sphere["indices"], sphere["weights"]
    with pytest.raises(ValueError, match="must be 1-D"):
        K.build_edge_table(ip, ix.reshape(-1, 1), w)
    with pytest.raises(ValueError, match="must have equal length"):
        K.build_edge_table(ip, ix[:-1], w)
    with pytest.raises(ValueError, match="weights must be float32"):
        K.build_edge_table(ip, ix, w.astype(np.float64))
    # A vertex with more neighbours than the packed table has slots.
    dense_ip = np.array([0, K._EDGE_SLOTS + 1], dtype=np.int64)
    dense_ix = np.zeros(K._EDGE_SLOTS + 1, dtype=np.int64)
    dense_w = np.ones(K._EDGE_SLOTS + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds the packed table"):
        K.build_edge_table(dense_ip, dense_ix, dense_w)


def test_sssp_preconditions(sphere):
    V = sphere["V"]
    tab = K.build_edge_table(sphere["indptr"], sphere["indices"],
                             sphere["weights"])
    src = np.array([0, 1], dtype=np.int32)
    rel = np.array([2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="tab must be"):
        K.central_sulcus_distances_cupy(tab[:, :4], src, rel, V, 5.0)
    with pytest.raises(ValueError, match="delta must be finite"):
        K.central_sulcus_distances_cupy(tab, src, rel, V, 0.0)
    with pytest.raises(ValueError, match="delta must be finite"):
        K.central_sulcus_distances_cupy(tab, src, rel, V, float("inf"))
    with pytest.raises(ValueError, match="empty source or relevant set"):
        K.central_sulcus_distances_cupy(tab, src[:0], rel, V, 5.0)
    # Vertex ids index the (V, 32) work buffer and the shared frontier
    # bitmask directly: out of range must raise, not write out of bounds.
    for bad in (np.array([V], dtype=np.int32),
                np.array([-1], dtype=np.int32)):
        with pytest.raises(ValueError, match=r"src_verts must hold vertex"):
            K.central_sulcus_distances_cupy(tab, bad, rel, V, 5.0)
        with pytest.raises(ValueError, match=r"rel_verts must hold vertex"):
            K.central_sulcus_distances_cupy(tab, src, bad, V, 5.0)


def test_sssp_refuses_a_mesh_too_large_for_shared_memory():
    """The frontier bitmask is RING x ceil(V/32) words of STATIC shared
    memory, so V has a hard ceiling (~90 k) well below fsaverage's
    163842. Refuse it by name here rather than at ptxas."""
    assert K._sssp_smem_bytes(40962, K._SSSP_CAP) <= K._SMEM_LIMIT
    assert K._sssp_smem_bytes(10242, K._SSSP_CAP) <= K._SMEM_LIMIT
    with pytest.raises(ValueError, match="mesh too large"):
        K._sssp_module(163842, K._SSSP_CAP, K._SSSP_TPB)


@pytest.mark.skipif(not BASELINE_MAT.exists(),
                    reason="fsaverage6 bench baseline not staged")
@pytest.mark.skipif(not os.environ.get("MSHBM_ATLAS_DIR"),
                    reason="MSHBM_ATLAS_DIR unset")
def test_fsaverage6_masks_match_baseline(tmp_path):
    import scipy.io as sio

    from arealmshbm.data_io.annot_io import read_annot_labels
    from arealmshbm.radius_mask import generate_radius_mask

    atlas = Path(os.environ["MSHBM_ATLAS_DIR"]) / "fsaverage6" / "label"
    stem = "Schaefer2018_300Parcels_Kong2022_17Networks_order.annot"
    if not (atlas / f"lh.{stem}").exists():
        pytest.skip(f"Schaefer-300 annot not staged under {atlas}")
    try:
        from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
        load_avg_mesh("lh", "fsaverage6", "inflated")
    except FileNotFoundError as e:  # pragma: no cover - staging dependent
        pytest.skip(f"avg_mesh bundle not staged: {e}")

    def lab(h):
        a = read_annot_labels(atlas / f"{h}.{stem}")
        return np.where(a < 0, 0, a).astype(np.int64)

    res = generate_radius_mask(lh_labels=lab("lh"), rh_labels=lab("rh"),
                               mesh="fsaverage6", radius=30,
                               out_dir=tmp_path, verbose=False, backend="gpu")
    got = sio.loadmat(res["mat_path"])
    ref = sio.loadmat(str(BASELINE_MAT))
    for key in ("lh_boundary", "rh_boundary"):
        assert (got[key] != ref[key]).nnz == 0, key
