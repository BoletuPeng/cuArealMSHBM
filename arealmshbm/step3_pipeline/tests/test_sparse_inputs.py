"""test_sparse_inputs.py — exactness + timing for the sparse input path.

Three families:

  * synthetic MAT round-trips (``scipy.io.savemat``) that need no data
    store — they pin the fast MAT parser against the dense loaders for
    compressed / uncompressed, fp32 / fp64 θ, and the v7.3 branch when
    ``h5py`` is importable, plus what routes to scipy (an unsupported
    layout) and what does not (a corrupt payload, a missing ``isal``);
  * sub-001 exactness — prior, mask, layout (``layouts_equal`` against
    ``build_candidate_layout_dense``), packed BOLD, gradient,
    ``setting_params`` — skipped when the profile store is absent;
  * :class:`Step3SparseCohort` — the identity check, and that sharing
    one cohort across subjects moves no bytes.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arealmshbm.step3_pipeline import sparse_inputs as si
from arealmshbm.vmf_clustering.sparse_layout import (
    build_candidate_layout_dense, layouts_equal,
)
from arealmshbm.vmf_clustering.tests._sub001_fixture import (
    SUB001_DIR, sub001_available,
)


# ─────────────────────────────────────────────────────────────────────
# Synthetic MAT fixtures
# ─────────────────────────────────────────────────────────────────────
def _make_prior(rng, N=64, L=8, Dp=11, dtype=np.float32):
    theta = np.zeros((N, L), dtype=dtype)
    for n in range(N):
        for l in rng.choice(L, size=2, replace=False):
            theta[n, l] = dtype(rng.uniform(0.01, 1.0))
    theta[3, :] = 0                                  # an inactive row
    return {
        "mu": rng.standard_normal((Dp, L)).astype(np.float32),
        "theta": theta,
        "epsil": rng.uniform(1, 50, size=(1, L)).astype(np.float32),
        "sigma": rng.uniform(1, 50, size=(1, L)).astype(np.float32),
    }


def _write_prior(tmp_path, params, compress, name="Params_Final.mat"):
    from scipy.io import savemat
    p = tmp_path / name
    savemat(str(p), {"Params": params}, do_compression=compress)
    return p


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("theta_dtype", [np.float32, np.float64])
def test_prior_csr_matches_dense_synthetic(tmp_path, compress, theta_dtype):
    from arealmshbm.data_io import load_group_prior

    rng = np.random.default_rng(7)
    params = _make_prior(rng, dtype=theta_dtype)
    p = _write_prior(tmp_path, params, compress)

    ref = load_group_prior(p)
    got = si.load_group_prior_csr(p)

    assert np.array_equal(got["mu"], ref["mu"])
    assert np.array_equal(got["epsil"], ref["epsil"].ravel())
    assert np.array_equal(got["sigma"], ref["sigma"].ravel())
    assert (got["N"], got["L"]) == ref["theta"].shape
    dense = si.theta_csr_to_dense(got["theta_csr"], got["N"], got["L"])
    assert np.array_equal(dense, ref["theta"])
    # ascending columns inside each row
    row_ptr, col, _ = got["theta_csr"]
    for r in range(got["N"]):
        seg = col[row_ptr[r]:row_ptr[r + 1]]
        assert np.all(np.diff(seg) > 0)


def test_prior_csr_fast_path_used_on_compressed(tmp_path):
    rng = np.random.default_rng(11)
    p = _write_prior(tmp_path, _make_prior(rng), True)
    si.load_group_prior_csr(p)
    assert si.LAST_PRIOR_PATH == "fast"


def test_prior_csr_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        si.load_group_prior_csr(tmp_path / "nope.mat")


def test_prior_csr_fast_path_without_isal(tmp_path, monkeypatch):
    """The shared reader's stdlib-zlib fallback keeps this path exact."""
    import zlib
    from arealmshbm.data_io import mat5_stream as ms

    p = _write_prior(tmp_path, _make_prior(np.random.default_rng(23)), True)
    ref = si.load_group_prior_csr(p)
    assert si.LAST_PRIOR_PATH == "fast"

    monkeypatch.setattr(ms, "_zlib", zlib)
    got = si.load_group_prior_csr(p)
    assert si.LAST_PRIOR_PATH == "fast"      # not the 6x-slower scipy path
    for k in ("mu", "epsil", "sigma"):
        assert got[k].tobytes() == ref[k].tobytes(), k
    for a, b in zip(got["theta_csr"], ref["theta_csr"]):
        assert np.array_equal(a, b)


def test_prior_csr_corrupt_payload_propagates(tmp_path):
    """A decode error is not a "layout I do not model" verdict.

    Only ``_Unsupported`` routes to scipy; a malformed element must
    reach the caller instead of being re-read on the dense fallback
    (which materialises the 33 MB θ this module exists to avoid).
    Patches ``epsil``'s payload byte count to a non-multiple of the
    fp32 itemsize, so ``np.frombuffer`` raises ``ValueError`` — the
    class the walker's ``except`` clause used to swallow.
    """
    import struct
    import zlib

    p = _write_prior(tmp_path, _make_prior(np.random.default_rng(29)), True)
    raw = bytearray(p.read_bytes())
    typ, nb = struct.unpack("<II", raw[128:136])
    assert typ == 15                                  # miCOMPRESSED
    body = bytearray(zlib.decompress(bytes(raw[136:136 + nb])))
    i = body.find(struct.pack("<II", 7, 32))          # miSINGLE, (1, 8) epsil
    assert i > 0
    body[i:i + 8] = struct.pack("<II", 7, 30)         # 30 // 4 is not exact
    comp = zlib.compress(bytes(body), 6)
    p.write_bytes(bytes(raw[:128]) + struct.pack("<II", 15, len(comp)) + comp)

    si.LAST_PRIOR_PATH = "sentinel"
    with pytest.raises(ValueError, match="multiple of element size"):
        si.load_group_prior_csr(p)
    assert si.LAST_PRIOR_PATH == "sentinel"  # never re-routed to scipy


def test_prior_csr_v73(tmp_path):
    h5py = pytest.importorskip("h5py")
    from arealmshbm.data_io import load_group_prior

    rng = np.random.default_rng(13)
    params = _make_prior(rng)
    p = tmp_path / "Params_Final_v73.mat"
    with h5py.File(p, "w") as f:
        # MATLAB v7.3 stores transposed.
        for k, v in params.items():
            f.create_dataset(f"Params/{k}", data=np.asarray(v).T)
    ref = load_group_prior(p)
    got = si.load_group_prior_csr(p)
    assert si.LAST_PRIOR_PATH == "scipy"
    dense = si.theta_csr_to_dense(got["theta_csr"], got["N"], got["L"])
    assert np.array_equal(dense, ref["theta"])
    assert np.array_equal(got["mu"], ref["mu"])


@pytest.mark.parametrize("compress", [True, False])
def test_spatial_mask_csr_matches_dense_synthetic(tmp_path, compress):
    import scipy.sparse as sp
    from scipy.io import savemat
    from arealmshbm.data_io import load_spatial_mask

    rng = np.random.default_rng(3)
    lh = sp.random(40, 6, density=0.25, random_state=rng, format="csr")
    rh = sp.random(40, 6, density=0.25, random_state=rng, format="csr")
    p = tmp_path / "spatial_mask_test.mat"
    savemat(str(p), {"lh_boundary": lh, "rh_boundary": rh},
            do_compression=compress)

    lh_ref, rh_ref = load_spatial_mask(p)
    lh_got, rh_got = si.load_spatial_mask_csr(p)
    assert np.array_equal(lh_got.toarray(), lh_ref)
    assert np.array_equal(rh_got.toarray(), rh_ref)
    assert lh_got.dtype == np.float64


def test_spatial_mask_csr_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        si.load_spatial_mask_csr(tmp_path / "nope.mat")


def test_layout_fast_matches_dense_synthetic(tmp_path):
    """Small synthetic bilateral problem: fast layout ≡ dense layout."""
    import scipy.sparse as sp
    from arealmshbm.pipeline_setup import build_boundary_mask

    rng = np.random.default_rng(17)
    n_h, l_h, m1 = 24, 5, 6
    N, L = 2 * n_h, 2 * l_h

    # masks: dense-ish support per hemi
    lh_m = (rng.random((n_h, l_h)) < 0.5).astype(np.float64)
    rh_m = (rng.random((n_h, l_h)) < 0.5).astype(np.float64)
    lh_m[:, 0] = 1.0                       # guarantee a support per row
    rh_m[:, 0] = 1.0

    # theta inside the mask support, block-diagonal
    theta = np.zeros((N, L), dtype=np.float32)
    for n in range(N):
        blk = lh_m[n] if n < n_h else rh_m[n - n_h]
        cand = np.flatnonzero(blk)
        pick = rng.choice(cand, size=min(2, cand.size), replace=False)
        off = 0 if n < n_h else l_h
        if n % 9:                          # leave some rows inactive
            theta[n, pick + off] = rng.uniform(0.1, 1.0, size=pick.size)

    nbors = rng.integers(0, n_h + 1, size=(m1, n_h)).astype(np.int64)
    lh_nb, rh_nb = nbors, nbors[::-1].copy()

    bm = build_boundary_mask(lh_m, rh_m)
    ref = build_candidate_layout_dense(theta, bm, lh_nb, rh_nb)
    csr = si._theta_dense_to_csr(theta)
    fast = si.build_candidate_layout_fast(
        csr, sp.csr_matrix(lh_m), sp.csr_matrix(rh_m), lh_nb, rh_nb)
    assert layouts_equal(fast, ref)


def test_layout_fast_rejects_theta_outside_mask():
    import scipy.sparse as sp

    n_h, l_h = 4, 2
    N, L = 2 * n_h, 2 * l_h
    theta = np.zeros((N, L), dtype=np.float32)
    theta[0, 1] = 0.5
    lh_m = np.zeros((n_h, l_h)); rh_m = np.zeros((n_h, l_h))
    lh_m[0, 0] = 1.0
    nb = np.zeros((3, n_h), dtype=np.int64)
    with pytest.raises(ValueError, match="boundary_mask"):
        si.build_candidate_layout_fast(
            si._theta_dense_to_csr(theta),
            sp.csr_matrix(lh_m), sp.csr_matrix(rh_m), nb, nb)


def test_layout_fast_rejects_cross_hemisphere():
    import scipy.sparse as sp

    n_h, l_h = 4, 2
    theta = np.zeros((2 * n_h, 2 * l_h), dtype=np.float32)
    theta[0, l_h] = 0.5                     # LH vertex, RH parcel
    lh_m = np.ones((n_h, l_h)); rh_m = np.ones((n_h, l_h))
    nb = np.zeros((3, n_h), dtype=np.int64)
    with pytest.raises(ValueError, match="cross-hemisphere"):
        si.build_candidate_layout_fast(
            si._theta_dense_to_csr(theta),
            sp.csr_matrix(lh_m), sp.csr_matrix(rh_m), nb, nb)


# ─────────────────────────────────────────────────────────────────────
# sub-001 exactness
# ─────────────────────────────────────────────────────────────────────
pytestmark_sub001 = pytest.mark.skipif(
    not sub001_available(),
    reason=f"sub-001 profile store not found at {SUB001_DIR}",
)


def _cfg():
    from arealmshbm.step3_pipeline import Step3Config
    return Step3Config(
        project_dir=SUB001_DIR, num_session=6, num_clusters=300, subid=1,
        mesh="fsaverage6", w=50.0, c=10.0, beta_scalar=5.0, backend="cpu",
    )


@pytestmark_sub001
def test_sub001_prior_csr_exact():
    from arealmshbm.data_io import load_group_prior

    cfg = _cfg()
    ref = load_group_prior(cfg.group_prior_path)
    got = si.load_group_prior_csr(cfg.group_prior_path)
    assert si.LAST_PRIOR_PATH == "fast"
    assert np.array_equal(got["mu"], ref["mu"])
    assert np.array_equal(got["epsil"], ref["epsil"].ravel())
    assert np.array_equal(got["sigma"], ref["sigma"].ravel())
    dense = si.theta_csr_to_dense(got["theta_csr"], got["N"], got["L"])
    assert np.array_equal(dense, ref["theta"])


@pytestmark_sub001
def test_sub001_spatial_mask_csr_exact():
    from arealmshbm.data_io import load_spatial_mask

    cfg = _cfg()
    lh_ref, rh_ref = load_spatial_mask(cfg.spatial_mask_path)
    lh_got, rh_got = si.load_spatial_mask_csr(cfg.spatial_mask_path)
    assert np.array_equal(lh_got.toarray(), lh_ref)
    assert np.array_equal(rh_got.toarray(), rh_ref)


@pytestmark_sub001
def test_sub001_layout_equals_dense():
    from arealmshbm.vmf_clustering.tests._sub001_fixture import load_sub001

    fx = load_sub001()
    cfg = _cfg()
    prior = si.load_group_prior_csr(cfg.group_prior_path)
    lh_m, rh_m = si.load_spatial_mask_csr(cfg.spatial_mask_path)
    fast = si.build_candidate_layout_fast(
        prior["theta_csr"], lh_m, rh_m,
        fx.lh_inflated["vertexNbors"], fx.rh_inflated["vertexNbors"])
    assert layouts_equal(fast, fx.layout)


@pytestmark_sub001
def test_sub001_packed_bold_and_gradient():
    from arealmshbm.data_io import fetch_data, load_avg_mesh

    cfg = _cfg()
    lh_inf = load_avg_mesh("lh", cfg.mesh, "inflated")
    rh_inf = load_avg_mesh("rh", cfg.mesh, "inflated")
    ref = fetch_data(project_dir=cfg.project_dir, num_session=cfg.num_session,
                     subid=cfg.subid, mesh=cfg.mesh,
                     lh_mesh=lh_inf, rh_mesh=rh_inf,
                     n_grad_components=cfg.n_grad_components,
                     with_gradient=True)
    packed, D, mw_lh, mw_rh = si.fetch_packed_bold_TND(
        cfg.project_dir, cfg.num_session, cfg.subid, cfg.mesh, lh_inf, rh_inf)
    assert D == int(ref["D_unpacked"])
    assert packed.flags.c_contiguous
    # ``fetch_data`` zeroes the MW rows on the host; the sparse path
    # defers that to the device backend, which uses the returned index
    # lists. Apply it here to compare like for like — everything else
    # (the (T, N, D_b) bytes themselves) must match exactly.
    n_lh = int(lh_inf["MARS_label"].shape[0])
    zeroed = packed.copy()
    zeroed[:, mw_lh, :] = 0
    zeroed[:, n_lh + mw_rh, :] = 0
    assert np.array_equal(zeroed, np.transpose(ref["series"], (1, 0, 2)))
    assert np.array_equal(mw_lh, np.flatnonzero(lh_inf["MARS_label"] == 1))
    assert np.array_equal(mw_rh, np.flatnonzero(rh_inf["MARS_label"] == 1))

    grad = si.fetch_gradient(cfg.project_dir, cfg.subid, cfg.mesh,
                             packed.shape[1], cfg.n_grad_components)
    assert np.array_equal(grad, ref["gradient_mat"])

    # precomputed passthrough
    g2 = si.fetch_gradient(cfg.project_dir, cfg.subid, cfg.mesh,
                           packed.shape[1], cfg.n_grad_components,
                           precomputed_gradient_mat=grad)
    assert np.array_equal(g2, grad)


@pytestmark_sub001
def test_sub001_sparse_inputs_bundle():
    from arealmshbm.step3_pipeline import Step3Pipeline

    cfg = _cfg()
    inp = si.load_step3_sparse_inputs(cfg)
    pipe = Step3Pipeline(_cfg())
    ref = pipe.load_inputs()

    assert layouts_equal(inp.layout, build_candidate_layout_dense(
        ref.Params["theta"], ref.boundary_mask,
        ref.lh_inflated["vertexNbors"], ref.rh_inflated["vertexNbors"]))
    zeroed = inp.packed_TND.copy()
    n_lh = int(ref.lh_inflated["MARS_label"].shape[0])
    zeroed[:, inp.mw_lh_idx, :] = 0
    zeroed[:, n_lh + inp.mw_rh_idx, :] = 0
    assert np.array_equal(zeroed,
                          np.transpose(ref.data["series"], (1, 0, 2)))
    assert inp.D == int(ref.data["D_unpacked"])
    assert np.array_equal(inp.mu, ref.Params["mu"])
    assert np.array_equal(inp.epsil, ref.Params["epsil"])
    assert np.array_equal(inp.sigma, ref.Params["sigma"])
    assert np.array_equal(inp.gradient_mat, ref.data["gradient_mat"])
    assert np.array_equal(inp.sphere_xyz_bilateral, ref.sphere_xyz_bilateral)
    assert inp.ini_val == ref.ini_val
    for k in ("mesh", "num_session", "num_clusters", "subid", "w", "c",
              "epsilon", "connect_th", "dim", "num_verts"):
        assert inp.setting_params[k] == ref.setting_params[k], k
    assert np.array_equal(inp.setting_params["beta"],
                          ref.setting_params["beta"])
    pipe.close()


@pytestmark_sub001
def test_sub001_timing_report(capsys):
    """Not an assertion — prints the per-piece wall times."""
    cfg = _cfg()
    si.load_step3_sparse_inputs(cfg, overlap=False)     # warm caches / JIT

    t0 = time.perf_counter()
    serial = si.load_step3_sparse_inputs(cfg, overlap=False)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    par = si.load_step3_sparse_inputs(cfg, overlap=True)
    t_par = time.perf_counter() - t0

    lines = ["", "sub-001 sparse load_inputs timings (s):"]
    for k, v in serial.timings.items():
        lines.append(f"  serial {k:26s} {v:.4f}")
    lines.append(f"  serial total (wall)          {t_serial:.4f}")
    lines.append(f"  overlapped total (wall)      {t_par:.4f}")
    lines.append(f"  prior path: {si.LAST_PRIOR_PATH}")
    with capsys.disabled():
        print("\n".join(lines))
    assert par.layout.P == serial.layout.P


# ─────────────────────────────────────────────────────────────────────
# Padding-bit guard on the packed BOLD
#
# The device ``acc_bits`` kernel folds all 8 bits of every byte with no
# ``d < D`` clamp, so a set padding bit in the trailing byte would be
# counted — and on the last row would read past the allocation. Both
# in-tree writers zero the padding; ``fetch_packed_bold_TND`` re-checks
# it because the step-3 read path is the only thing between an
# arbitrary .b2nd on disk and those kernels.
# ─────────────────────────────────────────────────────────────────────
def _tiny_packed_project(tmp_path, *, D, dirty):
    """Project with a cohort + one 2-session, N=4-vertex packed .b2nd."""
    from arealmshbm.data_io.cohort import (
        CohortSubject, compute_run_id, write_cohort_partial,
    )
    from arealmshbm.data_io.profile_io import SubjectProfileStreamWriter

    T, N = 2, 4
    Db = (D + 7) // 8
    packed = np.full((T, N, Db), 0b0000_0001, dtype=np.uint8)
    if dirty:
        packed[1, 2, -1] |= np.uint8(0b1000_0000)      # top padding bit

    b2nd = tmp_path / "profile.b2nd"
    # The packed writer (the GPU leaf's): it takes the bytes verbatim,
    # which is what lets the ``dirty`` padding bit reach the file.
    with SubjectProfileStreamWriter(b2nd, T=T, N=N, D_unpacked=D) as w:
        for t in range(T):
            w.write_session(t, packed[t])

    sub = CohortSubject(id="1", sessions=["1", "2"],
                        profile_b2nd=str(b2nd))
    write_cohort_partial(
        tmp_path,
        subjects=[sub],
        mesh={"targ": "fsaverage6", "seed": "fsaverage3"},
        n_grad_components=100,
        run_id=compute_run_id(subjects=["1"], sessions=["1", "2"],
                              targ_mesh="fsaverage6",
                              seed_mesh="fsaverage3",
                              n_grad_components=100),
    )
    half = {"MARS_label": np.array([2, 1, 2, 2], dtype=np.int32)[: N // 2]}
    return tmp_path, half, half


@pytest.mark.parametrize("D", [11, 20])          # pad = 5 and 4 bits
def test_packed_bold_rejects_dirty_padding_bits(tmp_path, D):
    project, lh, rh = _tiny_packed_project(tmp_path, D=D, dirty=True)
    with pytest.raises(ValueError, match="padding bits"):
        si.fetch_packed_bold_TND(project, 2, 1, "fsaverage6", lh, rh)


@pytest.mark.parametrize("D", [11, 16])          # pad = 5 and 0 bits
def test_packed_bold_accepts_clean_padding(tmp_path, D):
    project, lh, rh = _tiny_packed_project(tmp_path, D=D, dirty=False)
    packed, D_out, _, _ = si.fetch_packed_bold_TND(
        project, 2, 1, "fsaverage6", lh, rh)
    assert D_out == D
    assert packed.shape == (2, 4, (D + 7) // 8)
    assert (packed == 0b0000_0001).all()


def test_packed_bold_rejects_profile_wider_than_the_kernel_limit(tmp_path):
    """``acc_bits`` caps ``ceil(D/8)`` at ``MAX_D_BYTES``.

    The loader raises here so the per-step API fails before it loads
    anything; the driver's own seed-mesh guard is the other half.
    """
    from arealmshbm.vmf_clustering.sparse_layout import MAX_D_BYTES

    D = 8 * MAX_D_BYTES + 1
    project, lh, rh = _tiny_packed_project(tmp_path, D=D, dirty=False)
    with pytest.raises(ValueError, match=r"exceeds the gpu_sparse"):
        si.fetch_packed_bold_TND(project, 2, 1, "fsaverage6", lh, rh)


def test_packed_bold_accepts_the_widest_supported_profile(tmp_path):
    from arealmshbm.vmf_clustering.sparse_layout import MAX_D_BYTES

    D = 8 * MAX_D_BYTES
    project, lh, rh = _tiny_packed_project(tmp_path, D=D, dirty=False)
    packed, D_out, _, _ = si.fetch_packed_bold_TND(
        project, 2, 1, "fsaverage6", lh, rh)
    assert D_out == D and packed.shape[2] == MAX_D_BYTES


# ─────────────────────────────────────────────────────────────────────
# Step3SparseCohort — the prior / layout shared across subjects
# ─────────────────────────────────────────────────────────────────────
def _dummy_cohort(**overrides):
    """A cohort object with the sub-001 identity and no payload.

    ``check`` only reads the four identity fields, so the payload can be
    empty — this keeps the mismatch tests free of the data store.
    """
    d = dict(
        mesh="fsaverage6",
        group_prior_path=str(_cfg().group_prior_path),
        spatial_mask_path=str(_cfg().spatial_mask_path),
        num_clusters=300,
        prior={}, layout=None, timings={},
    )
    d.update(overrides)
    return si.Step3SparseCohort(**d)


def test_cohort_check_accepts_matching_cfg():
    _dummy_cohort().check(_cfg())


# (cfg override, value, the identity field ``check`` reports). The
# reported field is not always the one overridden: ``check`` walks
# mesh → group_prior_path → spatial_mask_path → num_clusters and stops
# at the first disagreement, and both paths are *derived* from several
# knobs (config.py's ``group_prior_path`` / ``spatial_mask_path``).
@pytest.mark.parametrize("field, value, reported", [
    ("num_clusters", 299, "num_clusters"),
    ("mesh", "fsaverage5", "mesh"),
    ("beta_scalar", 6.0, "group_prior_path"),
    ("pipeline_type", "dMSHBM", "group_prior_path"),
    ("project_dir", "/tmp/other_project", "group_prior_path"),
])
def test_cohort_check_rejects_mismatching_cfg(field, value, reported):
    from arealmshbm.step3_pipeline import Step3Config

    kwargs = dict(
        project_dir=SUB001_DIR, num_session=6, num_clusters=300, subid=1,
        mesh="fsaverage6", w=50.0, c=10.0, beta_scalar=5.0, backend="cpu",
    )
    kwargs[field] = value
    cfg = Step3Config(**kwargs)
    with pytest.raises(ValueError, match=reported):
        _dummy_cohort().check(cfg)
    # The loader guards on the same check, before it touches any disk.
    with pytest.raises(ValueError, match=reported):
        si.load_step3_sparse_inputs(cfg, cohort=_dummy_cohort())


def test_cohort_check_rejects_mismatching_spatial_mask_path():
    """The only field reachable with mesh and prior path both matching.

    A different ``project_dir`` moves the mask; pinning the prior with
    ``group_prior_path_override`` keeps the two earlier fields equal.
    """
    from arealmshbm.step3_pipeline import Step3Config

    cfg = Step3Config(
        project_dir="/tmp/other_project", num_session=6, num_clusters=300,
        subid=1, mesh="fsaverage6", w=50.0, c=10.0, beta_scalar=5.0,
        backend="cpu",
        group_prior_path_override=_cfg().group_prior_path,
    )
    with pytest.raises(ValueError, match="spatial_mask_path"):
        _dummy_cohort().check(cfg)
    with pytest.raises(ValueError, match="spatial_mask_path"):
        si.load_step3_sparse_inputs(cfg, cohort=_dummy_cohort())


def test_pipeline_rejects_sparse_cohort_on_dense_backend():
    from arealmshbm.step3_pipeline import Step3Config, Step3Pipeline

    cfg = Step3Config(project_dir="/tmp/nonexistent", num_session=6,
                      num_clusters=300, backend="cpu")
    with pytest.raises(ValueError, match="gpu_sparse"):
        Step3Pipeline(cfg, sparse_cohort=_dummy_cohort())


@pytestmark_sub001
def test_sub001_cohort_shared_inputs_bit_identical():
    """Sharing a cohort object must not move a single byte."""
    cfg = _cfg()
    standalone = si.load_step3_sparse_inputs(cfg)
    cohort = si.load_step3_sparse_cohort(cfg)
    shared = si.load_step3_sparse_inputs(_cfg(), cohort=cohort)

    assert layouts_equal(shared.layout, standalone.layout)
    assert np.array_equal(shared.packed_TND, standalone.packed_TND)
    assert shared.D == standalone.D
    assert np.array_equal(shared.mw_lh_idx, standalone.mw_lh_idx)
    assert np.array_equal(shared.mw_rh_idx, standalone.mw_rh_idx)
    for name in ("mu", "epsil", "sigma", "gradient_mat",
                 "sphere_xyz_bilateral"):
        assert np.array_equal(getattr(shared, name),
                              getattr(standalone, name)), name
    for a, b in zip(shared.theta_csr, standalone.theta_csr):
        assert np.array_equal(a, b)
    assert shared.ini_val == standalone.ini_val
    assert shared.setting_params.keys() == standalone.setting_params.keys()
    for k, v in standalone.setting_params.items():
        got = shared.setting_params[k]
        if isinstance(v, np.ndarray):
            assert np.array_equal(got, v), k
        else:
            assert got == v, k

    # Shared, not copied: the bundle hands out the cohort's own objects.
    assert shared.layout is cohort.layout
    assert shared.mu is cohort.prior["mu"]
    assert shared.epsil is cohort.prior["epsil"]
    assert shared.sigma is cohort.prior["sigma"]
    assert shared.theta_csr is cohort.prior["theta_csr"]

    # The cohort's timings belong to whoever built it.
    assert "build_candidate_layout" in standalone.timings
    assert "build_candidate_layout" not in shared.timings
