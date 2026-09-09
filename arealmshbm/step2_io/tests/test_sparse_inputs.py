"""test_sparse_inputs.py — ``load_step2_sparse_inputs`` (step-2 ``gpu`` path).

Two layers:

* a **synthetic project** built the way ``test_subject_loaders.py`` does
  (real ``.b2nd`` + ``cohort.json`` + a hand-written ``group.mat`` /
  spatial mask / gradients), with ``load_avg_mesh`` monkeypatched to a
  tiny fake mesh — it exercises both gradient formats, the host-cache
  and stream branches, ``overlap`` on and off, and the medial-wall
  contract;
* a **bench-backed** layer (``MSHBM_STEP2_BENCH_DIR``, default
  ``testdata/step2_bench``) asserting that every field equals what the
  dense ``Step2Pipeline.load_inputs`` path would have produced —
  layout, packed bytes, gradient, ``mtc``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from arealmshbm.data_io.cohort import backfill_cohort_json
from arealmshbm.data_io.profile_io import profile_path, write_subject_profile_tnd
from arealmshbm.step2_io import (
    Step2SparseInputs,
    build_step2_layout_dense,
    layouts_equal,
    load_step2_sparse_inputs,
)
import arealmshbm.step2_io.sparse_inputs as si
from arealmshbm.vmf_clustering.tests._sub001_fixture import missing_avg_mesh


BENCH_DIR = Path(os.environ.get("MSHBM_STEP2_BENCH_DIR", "testdata/step2_bench"))

TARG, SEED = "fsaverage6", "fsaverage3"


# ─────────────────────────────────────────────────────────────────────
# Synthetic project fixture
# ─────────────────────────────────────────────────────────────────────
class _Cfg:
    """The subset of ``Step2Config`` the loader reads."""

    def __init__(self, project_dir, S, T, L, mode, n_grad):
        self.project_dir = project_dir
        self.num_sub = S
        self.num_session = T
        self.num_clusters = L
        self.mode = mode
        self.mesh = TARG
        self.seed_mesh = SEED
        self.n_grad_components = n_grad


def _fake_mesh(n_h: int, mw: np.ndarray):
    """``load_avg_mesh`` stand-in — only ``MARS_label`` is read."""
    lab = np.ones((1, n_h), dtype=np.int32) * 2
    lab[0, mw] = 1                      # 1 == medial wall
    return lambda hemi, mesh, surf: {"MARS_label": lab}


def _make_project(tmp_path: Path, *, S=2, T=2, n_h=9, L_h=3, D=11,
                  n_grad=4, grad_fmt="npy", mw=(0, 4), break_mw=False,
                  seed=0):
    """Write a complete synthetic step-2 project; return ``(cfg, refs)``."""
    rng = np.random.default_rng(seed)
    N, L = 2 * n_h, 2 * L_h
    mw_idx = np.asarray(mw, dtype=np.int64)

    raws = []
    for s in range(1, S + 1):
        raw = (rng.random((T, N, D)) < 0.4).astype(np.float32)
        if not break_mw:
            raw[:, mw_idx, :] = 0.0             # LH medial wall
            raw[:, n_h + mw_idx, :] = 0.0       # RH medial wall
        write_subject_profile_tnd(profile_path(tmp_path, str(s), TARG, SEED),
                                  np.ascontiguousarray(raw))
        raws.append(raw)

    grads = []
    for s in range(1, S + 1):
        gdir = tmp_path / "gradients" / f"sub{s}"
        gdir.mkdir(parents=True, exist_ok=True)
        lh = rng.standard_normal((n_h, n_grad + 2)).astype(np.float32)
        rh = rng.standard_normal((n_h, n_grad + 2)).astype(np.float32)
        if grad_fmt == "npy":
            np.save(gdir / f"lh_emb_{n_grad}_distance_matrix.npy", lh)
            np.save(gdir / f"rh_emb_{n_grad}_distance_matrix.npy", rh)
        else:
            sio.savemat(gdir / f"lh_emb_{n_grad}_distance_matrix.mat",
                        {"emb": lh})
            sio.savemat(gdir / f"rh_emb_{n_grad}_distance_matrix.mat",
                        {"emb": rh})
        grads.append(np.concatenate([lh[:, :n_grad], rh[:, :n_grad]], axis=0))

    (tmp_path / "group").mkdir(parents=True, exist_ok=True)
    mtc = rng.standard_normal((D, L))
    sio.savemat(tmp_path / "group" / "group.mat",
                {"mtc": mtc,
                 "epsil": np.array([[3.0]]),
                 "lh_labels": np.arange(n_h).reshape(-1, 1),
                 "rh_labels": np.arange(n_h).reshape(-1, 1),
                 "lambda": rng.standard_normal((D, 4 * L))},
                do_compression=True, format="5")

    (tmp_path / "spatial_mask").mkdir(parents=True, exist_ok=True)
    lh_b = (rng.random((n_h, L_h)) < 0.5).astype(np.float64)
    rh_b = (rng.random((n_h, L_h)) < 0.5).astype(np.float64)
    for m in (lh_b, rh_b):
        m[:, 0] = 0.0
        m[3, 0] = 1.0                            # no empty column
    sio.savemat(tmp_path / "spatial_mask" / f"spatial_mask_{TARG}.mat",
                {"lh_boundary": sp.csc_matrix(lh_b),
                 "rh_boundary": sp.csc_matrix(rh_b)},
                do_compression=True, format="5")

    backfill_cohort_json(tmp_path,
                         subjects=[str(s) for s in range(1, S + 1)],
                         sessions=[str(t) for t in range(1, T + 1)],
                         targ_mesh=TARG, seed_mesh=SEED,
                         n_grad_components=n_grad)

    cfg = _Cfg(tmp_path, S, T, L, "gMSHBM", n_grad)
    refs = {"raws": raws, "grads": grads, "mtc": mtc,
            "lh_b": lh_b, "rh_b": rh_b, "n_h": n_h, "N": N, "L": L, "D": D,
            "mw": mw_idx}
    return cfg, refs


@pytest.fixture()
def patched_mesh(monkeypatch):
    # ``arealmshbm.data_io.load_avg_mesh`` names the re-exported function
    # in the package namespace, so the dotted-string form of setattr
    # resolves to the function, not the module. Grab the module itself.
    import importlib
    mod = importlib.import_module("arealmshbm.data_io.load_avg_mesh")

    def _install(n_h, mw):
        monkeypatch.setattr(mod, "load_avg_mesh",
                            _fake_mesh(n_h, np.asarray(mw, dtype=np.int64)))
    return _install


def _dense_bm(lh_b, rh_b):
    n_h, L_h = lh_b.shape
    top = np.concatenate([lh_b, np.zeros((n_h, L_h))], axis=1)
    bot = np.concatenate([np.zeros((n_h, L_h)), rh_b], axis=1)
    return np.concatenate([top, bot], axis=0)


# ─────────────────────────────────────────────────────────────────────
# Synthetic tests
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("grad_fmt", ["npy", "mat"])
@pytest.mark.parametrize("overlap", [True, False])
def test_synthetic_project_round_trip(tmp_path: Path, patched_mesh,
                                      grad_fmt: str, overlap: bool) -> None:
    cfg, refs = _make_project(tmp_path, grad_fmt=grad_fmt)
    patched_mesh(refs["n_h"], refs["mw"])

    inp = load_step2_sparse_inputs(cfg, overlap=overlap)

    assert (inp.S, inp.T, inp.N, inp.D) == (2, 2, refs["N"], refs["D"])
    assert inp.dim == refs["D"] - 1
    assert inp.D_bytes == (refs["D"] + 7) // 8
    assert inp.n_lh == inp.n_rh == refs["n_h"]
    assert inp.D_grad == 4
    assert inp.L == refs["L"]

    # mtc verbatim (fp64), lambda skipped
    assert inp.mtc.dtype == np.float64
    assert np.array_equal(inp.mtc, refs["mtc"])

    # layout == the dense pipeline's boundary mask
    ref_lay = build_step2_layout_dense(_dense_bm(refs["lh_b"], refs["rh_b"]))
    assert layouts_equal(inp.layout, ref_lay)

    # packed BOLD: bit-identical to what the writer packed
    assert inp.packed_host is not None
    out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
    for s in (1, 2):
        inp.bold_reader(s, out)
        exp = np.packbits(refs["raws"][s - 1].astype(np.uint8), axis=-1,
                          bitorder="little")
        assert np.array_equal(out, exp), s
        assert np.array_equal(out, inp.packed_host[s - 1]), s

    # gradient: hemispheres stacked, sliced to D_grad, no T axis
    g = np.empty((inp.N, inp.D_grad), dtype=np.float32)
    for s in (1, 2):
        inp.grad_reader(s, g)
        assert np.array_equal(g, refs["grads"][s - 1]), s

    assert set(inp.timings) >= {"cohort", "mesh", "profiles", "gradients",
                                "group_mtc", "boundary", "total"}


def test_stream_mode_when_the_host_cache_does_not_fit(
        tmp_path: Path, patched_mesh, monkeypatch) -> None:
    cfg, refs = _make_project(tmp_path)
    patched_mesh(refs["n_h"], refs["mw"])
    monkeypatch.setattr(si, "_HOST_CACHE_FRACTION", 1e-15)

    inp = load_step2_sparse_inputs(cfg)
    assert inp.packed_host is None
    out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
    inp.bold_reader(2, out)
    exp = np.packbits(refs["raws"][1].astype(np.uint8), axis=-1,
                      bitorder="little")
    assert np.array_equal(out, exp)


@pytest.mark.parametrize("stream", [False, True])
def test_a_second_subject_with_another_D_unpacked_raises(
        tmp_path: Path, patched_mesh, monkeypatch, stream: bool) -> None:
    """D=11 and D=12 share ceil(D/8)=2, so the shape check cannot see it.

    Only ``vlmeta.D_unpacked`` distinguishes them, and it is read off
    subject 1; without a per-subject check subject 2 would be normalised
    and unpacked with subject 1's D on both the eager and the streaming
    path.
    """
    cfg, refs = _make_project(tmp_path, D=11, seed=17)
    patched_mesh(refs["n_h"], refs["mw"])
    if stream:
        monkeypatch.setattr(si, "_HOST_CACHE_FRACTION", 1e-15)

    rng = np.random.default_rng(5)
    N, T = refs["N"], 2
    raw2 = (rng.random((T, N, 12)) < 0.4).astype(np.float32)
    raw2[:, refs["mw"], :] = 0.0
    raw2[:, refs["n_h"] + refs["mw"], :] = 0.0
    write_subject_profile_tnd(profile_path(tmp_path, "2", TARG, SEED),
                              np.ascontiguousarray(raw2))
    assert (11 + 7) // 8 == (12 + 7) // 8      # the shapes really do agree

    if stream:
        # Streaming defers the read to ``bold_reader``.
        inp = load_step2_sparse_inputs(cfg)
        out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
        with pytest.raises(ValueError, match="D_unpacked"):
            inp.bold_reader(2, out)
    else:
        with pytest.raises(ValueError, match="D_unpacked"):
            load_step2_sparse_inputs(cfg)


@pytest.mark.parametrize("stream", [False, True])
def test_a_second_subject_with_fewer_sessions_raises(
        tmp_path: Path, patched_mesh, monkeypatch, stream: bool) -> None:
    """A T=1 subject must raise, not be broadcast into all T slots.

    ``np.copyto`` broadcasts a leading axis of length 1, so without the
    packed-shape check subject 2 would silently become T identical
    copies of its single session.
    """
    cfg, refs = _make_project(tmp_path, T=2, seed=23)
    patched_mesh(refs["n_h"], refs["mw"])
    if stream:
        monkeypatch.setattr(si, "_HOST_CACHE_FRACTION", 1e-15)

    rng = np.random.default_rng(9)
    raw2 = (rng.random((1, refs["N"], refs["D"])) < 0.4).astype(np.float32)
    raw2[:, refs["mw"], :] = 0.0
    raw2[:, refs["n_h"] + refs["mw"], :] = 0.0
    write_subject_profile_tnd(profile_path(tmp_path, "2", TARG, SEED),
                              np.ascontiguousarray(raw2))

    if stream:
        inp = load_step2_sparse_inputs(cfg)
        out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
        with pytest.raises(ValueError, match="packed shape"):
            inp.bold_reader(2, out)
    else:
        with pytest.raises(ValueError, match="packed shape"):
            load_step2_sparse_inputs(cfg)


def test_medial_wall_contract_violation_raises(tmp_path: Path,
                                               patched_mesh) -> None:
    cfg, refs = _make_project(tmp_path, break_mw=True, seed=3)
    patched_mesh(refs["n_h"], refs["mw"])
    with pytest.raises(ValueError, match="MW vertices"):
        load_step2_sparse_inputs(cfg)


def test_dmshbm_needs_no_gradient(tmp_path: Path, patched_mesh) -> None:
    cfg, refs = _make_project(tmp_path)
    patched_mesh(refs["n_h"], refs["mw"])
    cfg.mode = "dMSHBM"
    inp = load_step2_sparse_inputs(cfg)
    assert inp.grad_reader is None and inp.D_grad == 0


def test_cohort_disagreement_raises(tmp_path: Path, patched_mesh) -> None:
    cfg, refs = _make_project(tmp_path)
    patched_mesh(refs["n_h"], refs["mw"])
    cfg.num_sub = 3
    with pytest.raises(ValueError, match="num_sub"):
        load_step2_sparse_inputs(cfg)
    cfg.num_sub = 2
    cfg.num_session = 5
    with pytest.raises(ValueError, match="num_session"):
        load_step2_sparse_inputs(cfg)
    cfg.num_session = 2
    cfg.seed_mesh = "fsaverage4"
    with pytest.raises(ValueError, match="seed_mesh"):
        load_step2_sparse_inputs(cfg)


def test_num_clusters_disagreement_raises(tmp_path: Path,
                                          patched_mesh) -> None:
    cfg, refs = _make_project(tmp_path)
    patched_mesh(refs["n_h"], refs["mw"])
    cfg.num_clusters = 8
    with pytest.raises(ValueError, match="num_clusters"):
        load_step2_sparse_inputs(cfg)


def test_bold_reader_rejects_a_wrong_buffer(tmp_path: Path,
                                            patched_mesh) -> None:
    cfg, refs = _make_project(tmp_path)
    patched_mesh(refs["n_h"], refs["mw"])
    inp = load_step2_sparse_inputs(cfg)
    with pytest.raises(ValueError):
        inp.bold_reader(1, np.empty((inp.T, inp.N, inp.D_bytes), np.float32))
    with pytest.raises(ValueError):
        inp.grad_reader(1, np.empty((inp.N, inp.D_grad + 1), np.float32))


def test_from_arrays_matches_a_loaded_bundle(tmp_path: Path,
                                             patched_mesh) -> None:
    cfg, refs = _make_project(tmp_path)
    patched_mesh(refs["n_h"], refs["mw"])
    inp = load_step2_sparse_inputs(cfg)

    grads = np.stack(refs["grads"])
    mem = Step2SparseInputs.from_arrays(inp.layout, inp.packed_host, inp.mtc,
                                        grad_SNDg=grads)
    assert (mem.S, mem.T, mem.N, mem.D, mem.dim) == \
           (inp.S, inp.T, inp.N, inp.D, inp.dim)
    a = np.empty((inp.T, inp.N, inp.D_bytes), np.uint8)
    b = np.empty_like(a)
    inp.bold_reader(1, a)
    mem.bold_reader(1, b)
    assert np.array_equal(a, b)


# ─────────────────────────────────────────────────────────────────────
# Bench-backed — vs the dense Step2Pipeline.load_inputs path
# ─────────────────────────────────────────────────────────────────────
_BENCH_PROJ = BENCH_DIR / "proj"
# The bench path builds the real layout, which reads the shipped (and
# gitignored) avg_mesh bundles — a fresh worktree has to skip, not error.
_NO_MESH = missing_avg_mesh("inflated")
_HAVE_BENCH = (_BENCH_PROJ / "cohort.json").exists() and _NO_MESH is None
_HAVE_BENCH2 = ((BENCH_DIR / "proj2" / "cohort.json").exists()
                and _NO_MESH is None)


def _bench_cfg(proj: Path, S: int):
    from arealmshbm.step2_pipeline.config import Step2Config
    return Step2Config(project_dir=proj, num_sub=S, num_session=6,
                       num_clusters=300, mode="gMSHBM", beta_scalar=5,
                       backend="cpu", out_dir=proj / "_t2_scratch",
                       verbose=False)


@pytest.mark.skipif(not _HAVE_BENCH, reason="step-2 bench project not present")
def test_bench_inputs_match_the_dense_path() -> None:
    from arealmshbm.data_io.cohort import read_cohort, resolve_path
    from arealmshbm.data_io.load_spatial_mask import load_spatial_mask
    from arealmshbm.data_io.profile_io import read_subject_profile_packed_tnd
    from arealmshbm.step2_init.boundary_mask import build_step2_boundary_mask
    from arealmshbm.step2_io import SubjectGradientLoader, load_group_mtc

    cfg = _bench_cfg(_BENCH_PROJ, 1)
    inp = load_step2_sparse_inputs(cfg)

    assert (inp.S, inp.T, inp.N, inp.D, inp.dim) == (1, 6, 81924, 1175, 1174)
    assert (inp.n_lh, inp.n_rh, inp.D_grad, inp.D_bytes) == \
           (40962, 40962, 100, 147)
    assert inp.layout.L == 300 and inp.layout.P == 586522

    lh_b, rh_b = load_spatial_mask(
        _BENCH_PROJ / "spatial_mask" / "spatial_mask_fsaverage6.mat")
    bm = build_step2_boundary_mask(lh_boundary=lh_b, rh_boundary=rh_b,
                                   mode="gMSHBM")
    assert layouts_equal(inp.layout, build_step2_layout_dense(bm))

    assert np.array_equal(
        inp.mtc, load_group_mtc(_BENCH_PROJ / "group" / "group.mat")["mtc"])

    cohort = read_cohort(_BENCH_PROJ)
    ref_packed, D_unpacked = read_subject_profile_packed_tnd(
        resolve_path(_BENCH_PROJ, cohort.subjects[0].profile_b2nd))
    assert D_unpacked == inp.D
    out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
    inp.bold_reader(1, out)
    assert np.array_equal(out, ref_packed)

    gl = SubjectGradientLoader(cohort=cohort, project_dir=_BENCH_PROJ,
                               n_components=100)
    g = np.empty((inp.N, inp.D_grad), dtype=np.float32)
    inp.grad_reader(1, g)
    assert np.array_equal(g, gl.load(1)[0])


@pytest.mark.skipif(not _HAVE_BENCH2,
                    reason="step-2 S=2 bench project not present")
def test_bench_two_subject_cache() -> None:
    cfg = _bench_cfg(BENCH_DIR / "proj2", 2)
    a = load_step2_sparse_inputs(cfg, overlap=True)
    b = load_step2_sparse_inputs(cfg, overlap=False)
    assert a.packed_host is not None and b.packed_host is not None
    assert np.array_equal(a.packed_host, b.packed_host)
    assert layouts_equal(a.layout, b.layout)
    assert np.array_equal(a.mtc, b.mtc)
