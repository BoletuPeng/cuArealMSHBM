"""test_step1_runners_gpu_flow.py — the GPU in-memory step-1 chain
through ``step1_runners`` on a tiny synthetic cohort.

Pins the runner-level contract the driver relies on:

  * ``run_generate_profiles(backend='gpu', avg_accumulator=…,
    write_handles=…)`` returns one .b2nd path per subject, folds every
    session into the accumulator on device and — single subject —
    defers the .b2nd join to ``join_step1_writers``.
  * The average built on device while the leaf packed equals the
    disk-mediated ``avg_profiles`` that re-reads the .b2nd files.
  * ``run_avg_profiles(accumulator=…)`` and
    ``run_ini_params(precomputed_*_avg_dev=…, save_async=True)``
    produce the same artifacts as the disk-mediated path, and
    ``join_step1_writers`` leaves every file on disk.
  * The multi-subject case joins per subject (no pending handles).
  * ``verbose=True`` prints one line per subject on the fused path.

Synthetic GIFTI generation is the same minimal writer the prefetcher
tests use. The end-to-end tests carry ``_needs_gpu`` (cupy + nvcomp +
staged fsaverage6 bundles); the argument-level contracts below them
run everywhere off stubs.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

from arealmshbm.data_io.profile_io import read_subject_profile_packed_tnd
from arealmshbm.pipeline import step1_runners
from arealmshbm.pipeline.step1_runners import (
    join_step1_writers, make_avg_accumulator, run_avg_profiles,
    run_generate_profiles, run_ini_params,
)

_MESH = "fsaverage6"
_SEED = "fsaverage3"


def _synth_gifti_bytes(N: int, T: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>\n',
             b'<GIFTI Version="1.0" NumberOfDataArrays="',
             str(T).encode(), b'">\n']
    for _ in range(T):
        arr = rng.standard_normal(N).astype(np.float32)
        payload = base64.b64encode(zlib.compress(arr.tobytes()))
        parts.append(
            f'<DataArray DataType="NIFTI_TYPE_FLOAT32" '
            f'ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" '
            f'Encoding="GZipBase64Binary" Endian="LittleEndian" '
            f'ExternalFileName="" ExternalFileOffset="0" '
            f'Dim0="{N}"><Data>'.encode())
        parts.append(payload)
        parts.append(b'</Data></DataArray>\n')
    parts.append(b'</GIFTI>\n')
    return b"".join(parts)


def _stage_project(tmp_path: Path, subjects, sessions, T=5):
    """Write a project dir with fsaverage6-sized synthetic BOLD lists."""
    from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
    V = int(load_avg_mesh("lh", _MESH, "inflated")["MARS_label"].shape[0])
    proj = tmp_path / "proj"
    lists = proj / "data_list" / "fMRI_list"
    lists.mkdir(parents=True)
    bold = tmp_path / "bold"
    bold.mkdir()
    k = 0
    for sub in subjects:
        for sess in sessions:
            lh = bold / f"sub{sub}_sess{sess}_lh.func.gii"
            rh = bold / f"sub{sub}_sess{sess}_rh.func.gii"
            lh.write_bytes(_synth_gifti_bytes(V, T, seed=1000 + k))
            rh.write_bytes(_synth_gifti_bytes(V, T, seed=2000 + k))
            k += 1
            (lists / f"lh_sub{sub}_sess{sess}.txt").write_text(str(lh) + "\n")
            (lists / f"rh_sub{sub}_sess{sess}.txt").write_text(str(rh) + "\n")
    return proj


def _nvcomp_installed() -> bool:
    """Package presence only, without importing the binding.

    ``importlib.util.find_spec`` imports the *parent* first, so on a
    box with no ``nvidia`` namespace package at all it raises instead
    of returning ``None`` -- that is "not installed" too.
    """
    import importlib.util
    try:
        return importlib.util.find_spec("nvidia.nvcomp") is not None
    except ModuleNotFoundError:
        return False


def _gpu_skip_reason() -> str:
    """Empty when the end-to-end tests can run. Only the nvCOMP
    *package* is probed: an installed binding that fails to load must
    fail these tests, not skip them."""
    try:
        import cupy  # noqa: F401
    except ImportError as e:
        return f"cupy unavailable ({e})"
    if not _nvcomp_installed():
        return "nvidia-nvcomp-cu12 not installed"
    try:
        from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
        load_avg_mesh("lh", _MESH, "inflated")
        load_avg_mesh("rh", _MESH, "inflated")
    except Exception as e:      # noqa: BLE001 — absent, stale or corrupt
        return f"fsaverage6 avg_mesh bundles not loadable ({e})"
    return ""


_GPU_SKIP = _gpu_skip_reason()
_needs_gpu = pytest.mark.skipif(bool(_GPU_SKIP), reason=_GPU_SKIP)


@_needs_gpu
def test_single_subject_in_memory_chain_matches_disk(tmp_path, monkeypatch):
    subjects, sessions = ["1"], ["1", "2"]
    proj = _stage_project(tmp_path, subjects, sessions)

    acc = make_avg_accumulator(targ_mesh=_MESH, seed_mesh=_SEED,
                               num_sess=len(sessions))
    handles: list = []
    paths = run_generate_profiles(
        project_dir=proj, subjects=subjects, sessions=sessions,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
        avg_accumulator=acc, write_handles=handles,
    )
    assert [p[0] for p in paths] == ["1"]
    assert len(handles) == 1, "single subject defers its .b2nd join"
    assert acc.n_pairs == len(sessions)

    avg_mem = run_avg_profiles(
        project_dir=proj, num_sub=1, num_sess=len(sessions),
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
        accumulator=acc,
    )
    assert avg_mem.lh_avg_dev is acc.lh and avg_mem.writer is not None

    # Labels: a synthetic 4-parcel partition of the cortex per hemi.
    from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
    lh_m = load_avg_mesh("lh", _MESH, "inflated")["MARS_label"]
    rh_m = load_avg_mesh("rh", _MESH, "inflated")["MARS_label"]
    V = lh_m.shape[0]
    lh_labels = np.where(lh_m == 2, (np.arange(V) % 4) + 1, 0).astype(np.int64)
    rh_labels = np.where(rh_m == 2, (np.arange(V) % 4) + 1, 0).astype(np.int64)

    ini_mem = run_ini_params(
        project_dir=proj, seed_mesh=_SEED, targ_mesh=_MESH,
        lh_labels=lh_labels, rh_labels=rh_labels, backend="gpu",
        precomputed_lh_avg=avg_mem.lh_avg, precomputed_rh_avg=avg_mem.rh_avg,
        precomputed_lh_avg_dev=avg_mem.lh_avg_dev,
        precomputed_rh_avg_dev=avg_mem.rh_avg_dev,
        save_async=True,
    )
    assert ini_mem.writer is not None
    path, epsil = ini_mem[:2]
    join_step1_writers(handles, avg_mem, ini_mem)

    # Everything landed on disk.
    assert Path(paths[0][1]).exists()
    assert avg_mem.lh_path.exists() and avg_mem.rh_path.exists()
    assert path.exists()

    # The .b2nd carries the packed layout the accumulator was sized for.
    on_disk, D_disk = read_subject_profile_packed_tnd(paths[0][1])
    assert D_disk == acc.D
    assert on_disk.shape == (len(sessions), acc.V_lh + acc.V_rh, acc.D_bytes)

    # Disk-mediated avg (re-reads the .b2nd) equals the on-device avg.
    import scipy.io as sio
    ref_lh = np.load(avg_mem.lh_path)
    avg_disk = run_avg_profiles(
        project_dir=proj, num_sub=1, num_sess=len(sessions),
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
    )
    join_step1_writers(None, avg_disk, None)
    assert np.array_equal(avg_disk.lh_avg, ref_lh)
    assert np.array_equal(avg_disk.lh_avg, avg_mem.lh_avg)

    # Disk-mediated ini_params (host arrays, sync write) gives the same
    # mtc / epsil as the device hand-off.
    m_mem = sio.loadmat(str(path))
    ini_disk = run_ini_params(
        project_dir=proj, seed_mesh=_SEED, targ_mesh=_MESH,
        lh_labels=lh_labels, rh_labels=rh_labels, backend="gpu",
        precomputed_lh_avg=avg_disk.lh_avg, precomputed_rh_avg=avg_disk.rh_avg,
    )
    assert ini_disk.writer is None
    m_disk = sio.loadmat(str(ini_disk.group_mat_path))
    assert np.array_equal(m_mem["mtc"], m_disk["mtc"])
    assert float(m_mem["epsil"].ravel()[0]) == float(m_disk["epsil"].ravel()[0]) == epsil
    assert np.array_equal(m_mem["lambda"], m_disk["lambda"])


@_needs_gpu
def test_multi_subject_joins_per_subject(tmp_path):
    subjects, sessions = ["1", "2"], ["1"]
    proj = _stage_project(tmp_path, subjects, sessions, T=4)
    acc = make_avg_accumulator(targ_mesh=_MESH, seed_mesh=_SEED,
                               num_sess=len(sessions))
    handles: list = []
    paths = run_generate_profiles(
        project_dir=proj, subjects=subjects, sessions=sessions,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
        avg_accumulator=acc, write_handles=handles,
    )
    assert [p[0] for p in paths] == ["1", "2"]
    assert handles == [], "multi-subject cohorts join each write inline"
    assert acc.n_pairs == 2
    for _sub_id, p in paths:
        on_disk, D = read_subject_profile_packed_tnd(p)
        assert D == acc.D
        assert on_disk.shape == (1, acc.V_lh + acc.V_rh, acc.D_bytes)

    # The sums built on device while the leaf packed equal the
    # disk-mediated average of the two .b2nd files it wrote.
    avg_mem = run_avg_profiles(
        project_dir=proj, num_sub=2, num_sess=1,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
        accumulator=acc,
    )
    join_step1_writers(None, avg_mem, None)
    avg_disk = run_avg_profiles(
        project_dir=proj, num_sub=2, num_sess=1,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
    )
    join_step1_writers(None, avg_disk, None)
    assert np.array_equal(avg_mem.lh_avg, avg_disk.lh_avg)
    assert np.array_equal(avg_mem.rh_avg, avg_disk.rh_avg)


class _FakeAccumulator:
    def __init__(self, num_sess: int) -> None:
        self.num_sess = num_sess
        self.n_pairs = 0

    def add(self, sess_index, packed_dev):        # the leaf's hook
        self.n_pairs += 1


def test_cpu_backend_rejects_the_accumulator_handoff(tmp_path):
    """The CPU avg leaf reads the .b2nd files back; it has no in-memory
    entry point, and the CPU profile stage cannot feed one."""
    with pytest.raises(ValueError, match="GPU-only"):
        run_avg_profiles(
            project_dir=tmp_path, num_sub=1, num_sess=1,
            seed_mesh=_SEED, targ_mesh=_MESH, backend="cpu",
            accumulator=_FakeAccumulator(1),
        )
    lists = tmp_path / "data_list" / "fMRI_list"
    lists.mkdir(parents=True)
    (lists / "lh_sub1_sess1.txt").write_text("lh.func.gii\n")
    (lists / "rh_sub1_sess1.txt").write_text("rh.func.gii\n")
    with pytest.raises(ValueError, match="GPU-only"):
        run_generate_profiles(
            project_dir=tmp_path, subjects=["1"], sessions=["1"],
            seed_mesh=_SEED, targ_mesh=_MESH, backend="cpu",
            avg_accumulator=_FakeAccumulator(1),
        )


def _stub_accumulator_avg(monkeypatch, calls: list):
    """Replace the GPU avg leaf so the accumulator branch's argument
    checks can run without a device."""
    mod = types.ModuleType("arealmshbm.avg_profiles.avg_profiles_gpu")
    mod.avg_profiles_from_accumulator = (
        lambda *a, **k: calls.append((a, k)) or "avg"
    )
    monkeypatch.setitem(
        sys.modules, "arealmshbm.avg_profiles.avg_profiles_gpu", mod)


def test_accumulator_handoff_rejects_a_num_sess_mismatch(
        tmp_path, monkeypatch):
    """The disk branch checks ``T == num_sess`` on every .b2nd; the
    accumulator counted each subject's sessions against its own
    ``num_sess``, which must be this cohort's. ``num_sub`` is the disk
    branch's discovery bound and is not consulted."""
    calls: list = []
    _stub_accumulator_avg(monkeypatch, calls)
    acc = _FakeAccumulator(2)

    with pytest.raises(ValueError,
                       match=r"built for num_sess=2 but num_sess=6"):
        run_avg_profiles(
            project_dir=tmp_path, num_sub=1, num_sess=6,
            seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu",
            accumulator=acc,
        )
    assert calls == [], "the leaf must not be reached"

    assert run_avg_profiles(
        project_dir=tmp_path, num_sub=7, num_sess=2,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu",
        accumulator=acc,
    ) == "avg"
    assert len(calls) == 1 and calls[0][0][0] is acc


def test_cpu_backend_forwards_save_async(tmp_path, monkeypatch):
    """``group.mat``'s background writer is device-agnostic, so the CPU
    backend takes ``save_async`` too and its handle joins like the GPU
    one."""
    from arealmshbm.ini_params import IniParamsResult
    import arealmshbm.ini_params as ini_pkg

    class _Writer:
        def __init__(self):
            self.joined = False

        def wait(self):
            self.joined = True

    seen: dict = {}

    def _fake_generate(**kw):
        seen.update(kw)
        out = IniParamsResult({"epsil": np.array([[0.25]])})
        out.writer = _Writer()
        return out

    monkeypatch.setattr(ini_pkg, "generate_ini_params", _fake_generate)

    res = run_ini_params(
        project_dir=tmp_path, seed_mesh=_SEED, targ_mesh=_MESH,
        lh_labels=np.zeros(4, np.int64), rh_labels=np.zeros(4, np.int64),
        backend="cpu", save_async=True,
    )
    assert seen["backend"] == "cpu" and seen["save_async"] is True
    assert res.writer is not None and res.epsil == 0.25
    join_step1_writers(None, None, res)
    assert res.writer.joined


def test_fused_gpu_path_reports_each_subject_when_verbose(
        tmp_path, monkeypatch, capsys):
    """The fused leaf is the longest step-1 stage; ``verbose=True`` must
    print one line per subject the way the stage pipeline does."""
    lists = tmp_path / "data_list" / "fMRI_list"
    lists.mkdir(parents=True)
    for sess in ("1", "2"):
        (lists / f"lh_sub1_sess{sess}.txt").write_text("lh.func.gii\n")
        (lists / f"rh_sub1_sess{sess}.txt").write_text("rh.func.gii\n")

    class _Res:
        packed = np.zeros((2, 4, 7), np.uint8)
        K = 55
        out_path = tmp_path / "sub1.b2nd"

        def wait(self):
            return self.out_path

    leaf = types.ModuleType(
        "arealmshbm.generate_profiles.profiles_subject_gpu")
    leaf.generate_subject_profiles_gpu = lambda *a, **k: _Res()
    leaf.prewarm_generate_profiles_gpu = lambda *a, **k: None
    monkeypatch.setitem(
        sys.modules,
        "arealmshbm.generate_profiles.profiles_subject_gpu", leaf)

    def _call(verbose: bool):
        capsys.readouterr()
        step1_runners.run_generate_profiles(
            project_dir=tmp_path, subjects=["1"], sessions=["1", "2"],
            seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=verbose,
        )
        return capsys.readouterr().out

    out = _call(True)
    assert "sub=1" in out and "sessions=2" in out and "K=55" in out
    assert _call(False) == ""
