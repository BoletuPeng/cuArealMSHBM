"""test_connectedness_gpu.py

Parity tests for :class:`ConnectednessGPU` against the CPU reference chain
(``compute_components_general`` + ``component_distance``) and against the
decision logic of ``vmf_clustering._check_connectedness_step``.

Skips cleanly when cupy or the sub-001 profile store is missing.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arealmshbm.check_connectedness import (
    compute_components_general,
    component_distance,
)
from arealmshbm.check_connectedness.component_distance import clear_mesh_caches
from arealmshbm.check_connectedness.tests._grid_fixture import grid_mesh, mesh
from arealmshbm.vmf_clustering.tests._sub001_fixture import (
    skip_unless_cupy,
    skip_unless_sub001,
)


# ─────────────────────────────────────────────────────────────────────────
# CPU reference helpers
# ─────────────────────────────────────────────────────────────────────────
def _cpu_ref(lh_labels, rh_labels, lh_mesh, rh_mesh, L):
    """(parcel_components fp64, eucli fp32) exactly as the EM body computes."""
    pc, lh_ci, rh_ci = compute_components_general(
        lh_labels, rh_labels,
        lh_mesh["vertexNbors"], rh_mesh["vertexNbors"], L, return_ci=True,
    )
    eucli = component_distance(
        lh_labels, rh_labels, lh_mesh, rh_mesh, L,
        parcel_components=pc, lh_ci_full=lh_ci, rh_ci_full=rh_ci,
    )
    return pc, eucli


def _cpu_step(pc, eucli, xyz_gamma, connect_th, components_threshold):
    """Lines 182-198 of ``_check_connectedness_step``, verbatim."""
    distrib = (eucli > connect_th) | (pc > components_threshold)
    if distrib.any():
        max_conn = float(eucli[distrib].max())
        max_comp = float(pc[distrib & np.isfinite(pc)].max())
        xyz_new = xyz_gamma.copy()
        xyz_new[distrib] += 1000.0
    else:
        max_conn = 0.0
        max_comp = float(components_threshold)
        xyz_new = xyz_gamma.copy()
    return xyz_new, max_conn, max_comp


def _compare(name, gpu, lh_labels, rh_labels, lh_mesh, rh_mesh, L,
             connect_th, comp_th, rng, report):
    import cupy as cp

    pc_c, eu_c = _cpu_ref(lh_labels, rh_labels, lh_mesh, rh_mesh, L)

    labels = np.concatenate([lh_labels, rh_labels]).astype(np.int32)
    labels_d = cp.asarray(labels)
    pc_d, eu_d = gpu.components_and_distance(labels_d)
    pc_g = pc_d.get()
    eu_g = eu_d.get()

    assert np.array_equal(np.isnan(pc_c), np.isnan(pc_g)), f"{name}: NaN mask"
    fin = ~np.isnan(pc_c)
    assert np.array_equal(pc_c[fin], pc_g[fin]), f"{name}: parcel_components"

    diff = np.abs(eu_c.astype(np.float64) - eu_g.astype(np.float64))
    report.append(float(diff.max()) if diff.size else 0.0)
    assert np.allclose(eu_c, eu_g, rtol=1e-6, atol=1e-6), (
        f"{name}: eucli max|diff|={diff.max()}")
    _check_step(name, gpu, labels_d, pc_c, eu_c, L, connect_th, comp_th, rng)
    return eu_g


def _check_step(name, gpu, labels_d, pc_c, eu_c, L, connect_th, comp_th, rng):
    """``step()`` decisions vs ``_cpu_step`` on the CPU ``(pc, eucli)``."""
    import cupy as cp

    xyz0 = rng.random(L) * 10.0
    xyz_c, mc_c, mk_c = _cpu_step(pc_c, eu_c, xyz0, connect_th, comp_th)
    xyz_d = cp.asarray(xyz0.copy())
    mc_g, mk_g = gpu.step(labels_d, xyz_d)
    assert mc_c == mc_g, f"{name}: max_connectedness {mc_c} vs {mc_g}"
    assert mk_c == mk_g, f"{name}: max_components {mk_c} vs {mk_g}"
    assert np.array_equal(xyz_c, xyz_d.get()), f"{name}: xyz_gamma"


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — sub-001 derive_labels
# ─────────────────────────────────────────────────────────────────────────
def _sub001_setup(connect_th=15.0, comp_th=3, **kw):
    """(fixture, lh0, rh0, L, ConnectednessGPU on the sphere mesh)."""
    fx = skip_unless_sub001()
    from arealmshbm.data_io import derive_labels
    from arealmshbm.check_connectedness.connectedness_gpu import ConnectednessGPU

    lh0, rh0 = derive_labels(fx.theta)
    L = fx.theta.shape[1]
    gpu = ConnectednessGPU(
        fx.lh_sphere["vertexNbors"], fx.rh_sphere["vertexNbors"],
        fx.lh_sphere["vertices"], fx.rh_sphere["vertices"],
        L, connect_th, comp_th, **kw,
    )
    return fx, lh0, rh0, L, gpu


def test_sub001_baseline_labels():
    skip_unless_cupy()
    fx, lh0, rh0, L, gpu = _sub001_setup()
    rng = np.random.default_rng(0)
    report: list[float] = []
    _compare("sub001-base", gpu, lh0, rh0, fx.lh_sphere, fx.rh_sphere,
             L, 15.0, 3, rng, report)
    print(f"\n[sub001 base] max |eucli diff| = {report[0]:.3e}")


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — 20 seeded perturbations
# ─────────────────────────────────────────────────────────────────────────
def _perturb(lh, rh, nbors_lh, nbors_rh, n_lh, rng):
    """~2 % scattered relabels + a handful of whole-blob flips per hemi."""
    lh = lh.copy()
    rh = rh.copy()
    for lab, nb, lo, hi in ((lh, nbors_lh, 1, n_lh),
                            (rh, nbors_rh, n_lh + 1, 2 * n_lh)):
        active = np.flatnonzero(lab != 0)
        k = max(1, int(0.02 * active.size))
        idx = rng.choice(active, size=k, replace=False)
        lab[idx] = rng.integers(lo, hi + 1, size=k)
        # blob flips: BFS ball of ~25 verts around 8 random seeds
        for _ in range(8):
            seed = int(rng.choice(active))
            ball = {seed}
            frontier = [seed]
            while len(ball) < 25 and frontier:
                v = frontier.pop(0)
                for kk in range(nb.shape[0]):
                    u = nb[kk, v]
                    if u == 0:
                        continue
                    u -= 1
                    if u not in ball and lab[u] != 0:
                        ball.add(int(u))
                        frontier.append(int(u))
            new = int(rng.integers(lo, hi + 1))
            lab[list(ball)] = new
    return lh, rh


def test_sub001_random_perturbations():
    skip_unless_cupy()
    fx, lh0, rh0, L, gpu = _sub001_setup()
    report: list[float] = []
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        lh, rh = _perturb(lh0, rh0, fx.lh_sphere["vertexNbors"],
                          fx.rh_sphere["vertexNbors"], L // 2, rng)
        _compare(f"perturb-{seed}", gpu, lh, rh, fx.lh_sphere, fx.rh_sphere,
                 L, 15.0, 3, rng, report)
    print(f"\n[sub001 perturbed x20] max |eucli diff| = {max(report):.3e}")


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — small synthetic meshes
# ─────────────────────────────────────────────────────────────────────────
def _synthetic_cases():
    """Yield (name, lh_labels, rh_labels, lh_mesh, rh_mesh, L)."""
    rows, cols = 6, 9
    n = rows * cols
    v_lh, nb = grid_mesh(rows, cols)
    v_rh, _ = grid_mesh(rows, cols, x0=1000.0)
    L = 8            # 4 parcels per hemisphere
    lh_mesh, rh_mesh = mesh(v_lh, nb), mesh(v_rh, nb.copy())

    def blank():
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)

    # (a) all-zero labels -> every parcel empty (NaN), eucli 0
    lh, rh = blank()
    yield "all-zero", lh, rh, lh_mesh, rh_mesh, L

    # (b) parcel 1 single component; parcel 2 empty; RH parcel 5 single
    lh, rh = blank()
    lh[:cols] = 1
    rh[:cols] = 5
    yield "single-comp", lh, rh, lh_mesh, rh_mesh, L

    # (c) parcel 1 with 2 components far apart, LH + RH mirror
    lh, rh = blank()
    lh[0:2] = 1
    lh[cols - 2:cols] = 1
    rh[0:2] = 5
    rh[cols - 2:cols] = 5
    yield "two-comp", lh, rh, lh_mesh, rh_mesh, L

    # (d) parcel 1 with 3 components on one row
    lh, rh = blank()
    lh[0] = 1
    lh[4] = 1
    lh[8] = 1
    yield "three-comp", lh, rh, lh_mesh, rh_mesh, L

    # (e) 2-component parcel touching a single-component parcel (the `fake`
    #     sentinel path: verts adjacent to a zeroed label must count as
    #     boundary)
    lh, rh = blank()
    lh[0:3] = 1                     # comp A of parcel 1
    lh[6:9] = 1                     # comp B of parcel 1
    lh[cols:cols + cols] = 2        # whole second row = parcel 2 (1 comp)
    rh[:] = 0
    yield "touch-single", lh, rh, lh_mesh, rh_mesh, L

    # (f) both hemispheres multi-component with different geometry
    lh, rh = blank()
    lh[0] = 1
    lh[cols - 1] = 1
    lh[2 * cols] = 2
    lh[2 * cols + 3] = 2
    rh[0] = 5
    rh[4] = 5
    rh[cols] = 6
    rh[cols + 8] = 6
    rh[3 * cols:3 * cols + 3] = 7
    yield "both-hemis", lh, rh, lh_mesh, rh_mesh, L

    # (g) medial-wall-adjacent 2-comp parcel spanning rows
    lh, rh = blank()
    lh[0:cols] = 1
    lh[(rows - 1) * cols:rows * cols] = 1
    lh[cols + 1] = 3
    rh[0:cols] = 6
    yield "rows-split", lh, rh, lh_mesh, rh_mesh, L


@pytest.mark.parametrize("case_idx", range(7))
def test_synthetic(case_idx):
    skip_unless_cupy()
    from arealmshbm.check_connectedness.connectedness_gpu import ConnectednessGPU

    cases = list(_synthetic_cases())
    name, lh, rh, lh_mesh, rh_mesh, L = cases[case_idx]
    clear_mesh_caches()
    gpu = ConnectednessGPU(
        lh_mesh["vertexNbors"], rh_mesh["vertexNbors"],
        lh_mesh["vertices"], rh_mesh["vertices"], L, 15.0, 3,
    )
    rng = np.random.default_rng(7)
    report: list[float] = []
    _compare(name, gpu, lh, rh, lh_mesh, rh_mesh, L, 15.0, 3, rng, report)
    # also exercise the cMSHBM-ish thresholds (connect_th=0, comp_th=1)
    gpu0 = ConnectednessGPU(
        lh_mesh["vertexNbors"], rh_mesh["vertexNbors"],
        lh_mesh["vertices"], rh_mesh["vertices"], L, 0.0, 1,
    )
    _compare(name + "-th0", gpu0, lh, rh, lh_mesh, rh_mesh, L, 0.0, 1,
             rng, report)
    clear_mesh_caches()


# ─────────────────────────────────────────────────────────────────────────
# Test 4 — determinism
# ─────────────────────────────────────────────────────────────────────────
def test_determinism():
    skip_unless_cupy()
    import cupy as cp

    fx, lh0, rh0, L, gpu = _sub001_setup()
    rng = np.random.default_rng(1000)
    lh, rh = _perturb(lh0, rh0, fx.lh_sphere["vertexNbors"],
                      fx.rh_sphere["vertexNbors"], L // 2, rng)
    labels_d = cp.asarray(np.concatenate([lh, rh]).astype(np.int32))

    pc1, eu1 = gpu.components_and_distance(labels_d)
    pc1, eu1 = pc1.get(), eu1.get()
    pc2, eu2 = gpu.components_and_distance(labels_d)
    pc2, eu2 = pc2.get(), eu2.get()
    assert np.array_equal(np.isnan(pc1), np.isnan(pc2))
    f = ~np.isnan(pc1)
    assert np.array_equal(pc1[f], pc2[f])
    assert np.array_equal(eu1.view(np.uint32), eu2.view(np.uint32))

    xyz = cp.zeros(L, dtype=cp.float64)
    a = gpu.step(labels_d, xyz)
    xyz2 = cp.zeros(L, dtype=cp.float64)
    b = gpu.step(labels_d, xyz2)
    assert a == b
    assert np.array_equal(xyz.get(), xyz2.get())


# ─────────────────────────────────────────────────────────────────────────
# Test 5 — timing (informational, no assertion)
# ─────────────────────────────────────────────────────────────────────────
def test_timing_sub001():
    skip_unless_cupy()
    import cupy as cp

    fx, lh0, rh0, L, gpu = _sub001_setup()
    labels_d = cp.asarray(np.concatenate([lh0, rh0]).astype(np.int32))
    xyz = cp.zeros(L, dtype=cp.float64)

    for _ in range(10):
        gpu.step(labels_d, xyz)
    cp.cuda.Device().synchronize()

    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        gpu.step(labels_d, xyz)
    cp.cuda.Device().synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"\n[timing] ConnectednessGPU.step() = {dt * 1e3:.3f} ms/call; "
          f"CPU reference is 4.3 ms")


# ─────────────────────────────────────────────────────────────────────────
# Test 6 — the NVRTC compiles are per PROCESS
#
# The step-3 stage pipeline builds one ConnectednessGPU per subject on
# the LOAD thread while EM workers run kernels on other streams; the
# compiles happen once per process, not per instance.
# ─────────────────────────────────────────────────────────────────────────
def test_modules_compiled_once_per_process():
    skip_unless_cupy()
    from arealmshbm.check_connectedness import connectedness_gpu as cg

    lh_v, lh_nb = grid_mesh(3, 3)
    rh_v, rh_nb = grid_mesh(3, 3, x0=100.0)
    args = (lh_nb, rh_nb, lh_v, rh_v, 4, 15.0, 3)

    a = cg.ConnectednessGPU(*args)
    b = cg.ConnectednessGPU(*args)

    # Same compiled modules, and both are the process-wide cache entries.
    assert a._mod is b._mod
    assert a._k_cc_coop is b._k_cc_coop
    mod, k_coop = cg._compiled_modules()
    assert (a._mod, a._k_cc_coop) == (mod, k_coop)

    # Per-instance device buffers stay per instance.
    assert a._comp is not b._comp
    assert a._changed is not b._changed
    assert a._out2 is not b._out2


def test_modules_require_cooperative_launch(monkeypatch):
    """A device without cooperative-launch support fails at compile time
    with a message naming the device and the ``cpu`` backend."""
    skip_unless_cupy()
    from arealmshbm.check_connectedness import connectedness_gpu as cg

    class _NoCoopDevice:
        id = 0
        attributes = {'CooperativeLaunch': 0}

    saved = cg._MODULES
    cg._MODULES = None
    monkeypatch.setattr(cg.cp.cuda, 'Device', _NoCoopDevice)
    try:
        with pytest.raises(RuntimeError, match='cooperative launch'):
            cg.prewarm_connectedness_gpu()
        assert cg._MODULES is None
    finally:
        cg._MODULES = saved


def test_modules_require_a_cuda_toolkit(monkeypatch):
    """Without a CUDA toolkit root the cooperative kernel cannot link;
    the error names ``CUDA_PATH`` and the ``cpu`` backend."""
    skip_unless_cupy()
    from arealmshbm.check_connectedness import connectedness_gpu as cg

    saved = cg._MODULES
    cg._MODULES = None
    monkeypatch.setattr(cg.cp.cuda, 'get_cuda_path', lambda: None)
    try:
        with pytest.raises(RuntimeError, match='CUDA_PATH'):
            cg.prewarm_connectedness_gpu()
        assert cg._MODULES is None
    finally:
        cg._MODULES = saved


# ─────────────────────────────────────────────────────────────────────────
# Test 7 — cMSHBM pre-predicate: remove_isolated on device
# ─────────────────────────────────────────────────────────────────────────
def _host_clean(lh, rh, lh_mesh, rh_mesh, thr):
    from arealmshbm.postprocessing import remove_isolated_surface_components
    a = remove_isolated_surface_components(lh, lh_mesh["vertexNbors"], thr)
    b = remove_isolated_surface_components(rh, rh_mesh["vertexNbors"], thr)
    return np.concatenate([a, b]).astype(np.int32)


def _check_pre_predicate(name, gpu, lh, rh, lh_mesh, rh_mesh, L, thr,
                         connect_th, comp_th, rng):
    """``remove_isolated`` == the host function per hemisphere; ``step``
    == the CPU chain on the host-cleaned labels; neither touches the
    input. Returns the number of relabelled vertices."""
    import cupy as cp

    raw = np.concatenate([lh, rh]).astype(np.int32)
    raw_d = cp.asarray(raw)
    got = gpu.remove_isolated(raw_d).get()
    exp = _host_clean(lh, rh, lh_mesh, rh_mesh, thr)
    assert np.array_equal(got, exp), (
        f"{name}: remove_isolated differs at {int((got != exp).sum())} vertices")
    assert np.array_equal(raw_d.get(), raw), f"{name}: input labels modified"

    n = lh.size
    pc_c, eu_c = _cpu_ref(exp[:n].astype(np.int64), exp[n:].astype(np.int64),
                          lh_mesh, rh_mesh, L)
    _check_step(name, gpu, raw_d, pc_c, eu_c, L, connect_th, comp_th, rng)
    assert np.array_equal(raw_d.get(), raw), f"{name}: step modified the input"
    return int((got != raw).sum())


@pytest.mark.parametrize("thr", [2, 3, 5, 8])
def test_remove_isolated_synthetic(thr):
    """Random labels on two 12x15 grids (almost every vertex is its own
    component) plus solid stripes, at the cMSHBM thresholds."""
    skip_unless_cupy()
    from arealmshbm.check_connectedness.connectedness_gpu import ConnectednessGPU

    rows, cols = 12, 15
    v_lh, nb = grid_mesh(rows, cols)
    v_rh, _ = grid_mesh(rows, cols, x0=1000.0)
    lh_mesh, rh_mesh = mesh(v_lh, nb), mesh(v_rh, nb.copy())
    L = 8
    clear_mesh_caches()
    gpu = ConnectednessGPU(nb, nb.copy(), v_lh, v_rh, L, 0.0, 1,
                           isolated_component_min_size=thr)
    n = rows * cols
    changed = 0
    for seed in range(6):
        rng = np.random.default_rng(300 + seed)
        lh = rng.integers(0, 5, size=n).astype(np.int64)        # 0 = medial wall, 1..4
        rh = rng.integers(4, 9, size=n).astype(np.int64)        # 5..8, 4 -> 0
        rh[rh == 4] = 0
        lh[:2 * cols] = 1
        rh[:2 * cols] = 5
        changed += _check_pre_predicate(f"grid-thr{thr}-s{seed}", gpu, lh, rh,
                                        lh_mesh, rh_mesh, L, thr, 0.0, 1, rng)
    assert changed > 0
    clear_mesh_caches()


def test_remove_isolated_sub001():
    """Base labels + 8 perturbations (scattered islands), on both CC paths."""
    skip_unless_cupy()
    fx, lh0, rh0, L, gpu = _sub001_setup(0.0, 1, isolated_component_min_size=5)
    n_changed = [_check_pre_predicate("sub001-ri-base", gpu, lh0, rh0,
                                      fx.lh_sphere, fx.rh_sphere, L, 5, 0.0, 1,
                                      np.random.default_rng(0))]
    for seed in range(8):
        rng = np.random.default_rng(2000 + seed)
        lh, rh = _perturb(lh0, rh0, fx.lh_sphere["vertexNbors"],
                          fx.rh_sphere["vertexNbors"], L // 2, rng)
        n_changed.append(_check_pre_predicate(
            f"sub001-ri-{seed}", gpu, lh, rh, fx.lh_sphere, fx.rh_sphere,
            L, 5, 0.0, 1, rng))
    print(f"\n[sub001 remove_isolated] relabelled vertices per case: {n_changed}")
    assert max(n_changed) > 0


def test_remove_isolated_contract():
    """The threshold is required and positive; labels are a C-contiguous
    (2n,) int32 device array (``components_and_distance`` shares the
    check) and never the buffer ``remove_isolated`` itself returns."""
    skip_unless_cupy()
    import cupy as cp
    from arealmshbm.check_connectedness.connectedness_gpu import ConnectednessGPU

    v, nb = grid_mesh(3, 3)
    lab = cp.zeros(18, dtype=cp.int32)
    plain = ConnectednessGPU(nb, nb.copy(), v, v, 2, 0.0, 1)
    with pytest.raises(ValueError, match="isolated_component_min_size"):
        plain.remove_isolated(lab)
    with pytest.raises(ValueError, match="must be positive"):
        ConnectednessGPU(nb, nb.copy(), v, v, 2, 0.0, 1,
                         isolated_component_min_size=0)
    gpu = ConnectednessGPU(nb, nb.copy(), v, v, 2, 0.0, 1,
                           isolated_component_min_size=2)
    for bad in (cp.zeros(18, dtype=cp.int64),            # dtype
                cp.zeros(9, dtype=cp.int32),             # one hemisphere
                cp.zeros(36, dtype=cp.int32)[::2]):      # strided
        with pytest.raises(ValueError, match="int32"):
            gpu.remove_isolated(bad)
        with pytest.raises(ValueError, match="int32"):
            plain.components_and_distance(bad)
    with pytest.raises(ValueError, match="returns"):
        gpu.remove_isolated(gpu.remove_isolated(lab))
