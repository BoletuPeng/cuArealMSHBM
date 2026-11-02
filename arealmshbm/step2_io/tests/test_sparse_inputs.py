"""test_sparse_inputs.py — ``load_step2_sparse_inputs`` (step-2 ``gpu`` path).

Two layers:

* a **synthetic project** (``_synthetic_project.make_project``: real
  ``.b2nd`` + ``cohort.json`` + a hand-written ``group.mat`` / spatial
  mask / gradients; ``load_project`` loads it with a tiny fake mesh
  standing in for ``load_avg_mesh`` — the Session tests get their
  cohort through the same pair) — it exercises both gradient formats,
  the per-subject reader, ``overlap`` on and off, and the medial-wall
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

from arealmshbm.data_io.cohort import backfill_cohort_json
from arealmshbm.data_io.profile_io import profile_path, write_subject_profile_tnd
from arealmshbm.step2_init.boundary_mask import build_step2_boundary_mask
from arealmshbm.step2_io import load_step2_sparse_inputs
from arealmshbm.step2_io.tests._layout_oracle import (
    build_step2_layout_dense,
    layouts_equal,
)
from arealmshbm.step2_io.tests._synthetic_project import (
    SEED,
    TARG,
    load_project,
    make_project,
)
from arealmshbm.vmf_clustering.tests._sub001_fixture import missing_avg_mesh


BENCH_DIR = Path(os.environ.get("MSHBM_STEP2_BENCH_DIR", "testdata/step2_bench"))


# ─────────────────────────────────────────────────────────────────────
# Synthetic tests
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("grad_fmt", ["npy", "mat"])
@pytest.mark.parametrize("overlap", [True, False])
def test_synthetic_project_round_trip(tmp_path: Path, grad_fmt: str,
                                      overlap: bool) -> None:
    cfg, refs = make_project(tmp_path, grad_fmt=grad_fmt)

    inp = load_project(cfg, refs, overlap=overlap)

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
    ref_lay = build_step2_layout_dense(build_step2_boundary_mask(
        refs["lh_b"], refs["rh_b"], "gMSHBM"))
    assert layouts_equal(inp.layout, ref_lay)

    # packed BOLD: bit-identical to what the writer packed
    out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
    for s in (1, 2):
        inp.bold_reader(s, out)
        exp = np.packbits(refs["raws"][s - 1].astype(np.uint8), axis=-1,
                          bitorder="little")
        assert np.array_equal(out, exp), s

    # gradient: hemispheres stacked, sliced to D_grad, no T axis
    g = np.empty((inp.N, inp.D_grad), dtype=np.float32)
    for s in (1, 2):
        inp.grad_reader(s, g)
        assert np.array_equal(g, refs["grads"][s - 1]), s

    assert set(inp.timings) >= {"cohort", "mesh", "profiles", "gradients",
                                "group_mtc", "boundary", "total"}


def test_a_second_subject_with_another_D_unpacked_raises(
        tmp_path: Path) -> None:
    """D=11 and D=12 share ceil(D/8)=2, so the shape check cannot see it.

    Only ``vlmeta.D_unpacked`` distinguishes them, and it is read off
    subject 1; without a per-call check subject 2 would be normalised
    and unpacked with subject 1's D.
    """
    cfg, refs = make_project(tmp_path, D=11, seed=17)

    rng = np.random.default_rng(5)
    N, T = refs["N"], 2
    raw2 = (rng.random((T, N, 12)) < 0.4).astype(np.float32)
    raw2[:, refs["mw"], :] = 0.0
    raw2[:, refs["n_h"] + refs["mw"], :] = 0.0
    write_subject_profile_tnd(profile_path(tmp_path, "2", TARG, SEED),
                              np.ascontiguousarray(raw2))
    assert (11 + 7) // 8 == (12 + 7) // 8      # the shapes really do agree

    # Only subject 1 is read at load time; the reader checks subject 2.
    inp = load_project(cfg, refs)
    out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
    with pytest.raises(ValueError, match="D_unpacked"):
        inp.bold_reader(2, out)


def test_a_second_subject_with_fewer_sessions_raises(tmp_path: Path) -> None:
    """A T=1 subject raises from the reader, naming the subject and its
    file; the loader, which reads subject 1's header only, cannot see
    it."""
    cfg, refs = make_project(tmp_path, T=2, seed=23)

    rng = np.random.default_rng(9)
    raw2 = (rng.random((1, refs["N"], refs["D"])) < 0.4).astype(np.float32)
    raw2[:, refs["mw"], :] = 0.0
    raw2[:, refs["n_h"] + refs["mw"], :] = 0.0
    write_subject_profile_tnd(profile_path(tmp_path, "2", TARG, SEED),
                              np.ascontiguousarray(raw2))

    inp = load_project(cfg, refs)
    out = np.empty((inp.T, inp.N, inp.D_bytes), dtype=np.uint8)
    with pytest.raises(ValueError, match="packed shape"):
        inp.bold_reader(2, out)


def test_medial_wall_contract_violation_raises(tmp_path: Path) -> None:
    cfg, refs = make_project(tmp_path, break_mw=True, seed=3)
    with pytest.raises(ValueError, match="MW vertices"):
        load_project(cfg, refs)


def test_dmshbm_needs_no_gradient(tmp_path: Path) -> None:
    cfg, refs = make_project(tmp_path)
    cfg.mode = "dMSHBM"
    inp = load_project(cfg, refs)
    assert inp.grad_reader is None and inp.D_grad == 0


def test_mixed_gradient_suffix_raises(tmp_path: Path) -> None:
    """Subject 1 with an lh ``.npy`` and an rh ``.mat`` is rejected."""
    cfg, refs = make_project(tmp_path)
    rh = tmp_path / "gradients" / "sub1" / "rh_emb_4_distance_matrix.npy"
    sio.savemat(rh.with_suffix(".mat"), {"emb": np.load(rh)})
    rh.unlink()
    backfill_cohort_json(tmp_path, subjects=["1", "2"], sessions=["1", "2"],
                         targ_mesh=TARG, seed_mesh=SEED, n_grad_components=4)
    with pytest.raises(ValueError, match="mixed gradient suffix"):
        load_project(cfg, refs)


def test_cohort_disagreement_raises(tmp_path: Path) -> None:
    cfg, refs = make_project(tmp_path)
    cfg.num_sub = 3
    with pytest.raises(ValueError, match="num_sub"):
        load_project(cfg, refs)
    cfg.num_sub = 2
    cfg.num_session = 5
    with pytest.raises(ValueError, match="num_session"):
        load_project(cfg, refs)
    cfg.num_session = 2
    cfg.seed_mesh = "fsaverage4"
    with pytest.raises(ValueError, match="seed_mesh"):
        load_project(cfg, refs)


def test_num_clusters_disagreement_raises(tmp_path: Path) -> None:
    cfg, refs = make_project(tmp_path)
    cfg.num_clusters = 8
    with pytest.raises(ValueError, match="num_clusters"):
        load_project(cfg, refs)


def test_bold_reader_rejects_a_wrong_buffer(tmp_path: Path) -> None:
    cfg, refs = make_project(tmp_path)
    inp = load_project(cfg, refs)
    with pytest.raises(ValueError):
        inp.bold_reader(1, np.empty((inp.T, inp.N, inp.D_bytes), np.float32))
    with pytest.raises(ValueError):
        inp.grad_reader(1, np.empty((inp.N, inp.D_grad + 1), np.float32))


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
def test_bench_two_subject_overlap_matches_serial() -> None:
    cfg = _bench_cfg(BENCH_DIR / "proj2", 2)
    a = load_step2_sparse_inputs(cfg, overlap=True)
    b = load_step2_sparse_inputs(cfg, overlap=False)
    assert layouts_equal(a.layout, b.layout)
    assert np.array_equal(a.mtc, b.mtc)
    x = np.empty((a.T, a.N, a.D_bytes), dtype=np.uint8)
    y = np.empty_like(x)
    for s in (1, 2):
        a.bold_reader(s, x)
        b.bold_reader(s, y)
        assert np.array_equal(x, y), s
