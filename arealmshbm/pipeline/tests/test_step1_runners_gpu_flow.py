"""test_step1_runners_gpu_flow.py — the GPU in-memory step-1 chain
through ``step1_runners`` on a tiny synthetic cohort.

Pins the runner-level contract the driver relies on:

  * ``run_generate_profiles(backend='gpu', packed_sink=…,
    write_handles=…)`` returns one .b2nd path per subject, fills the
    sink with ``(packed, K)`` and — single subject — defers the .b2nd
    join to ``join_step1_writers``.
  * The packed slab handed over in memory is byte-identical to what
    the .b2nd on disk decodes to (so ``avg_profiles`` from the sink
    equals ``avg_profiles`` from disk).
  * ``run_avg_profiles(packed_subjects=…)`` and
    ``run_ini_params(precomputed_*_avg_dev=…, save_async=True)``
    produce the same artifacts as the disk-mediated path, and
    ``join_step1_writers`` leaves every file on disk.
  * The multi-subject case joins per subject (no pending handles).
  * The verbose per-subject line names the ingest the leaf used.

The nvCOMP-vs-CPU-reader ingest equality is the leaf's contract and is
pinned once, in ``generate_profiles/tests/test_generate_subject_profiles_gpu.py``;
the runner has no ingest branch of its own.

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
    join_step1_writers, run_avg_profiles, run_generate_profiles,
    run_ini_params,
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


def _gpu_stack_present() -> bool:
    try:
        import cupy  # noqa: F401
        from arealmshbm.data_io import _nvcomp_batched
        from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
        if not _nvcomp_batched.nvcomp_available():
            return False
        load_avg_mesh("lh", _MESH, "inflated")
        load_avg_mesh("rh", _MESH, "inflated")
        return True
    except Exception:      # noqa: BLE001 — no cupy / nvcomp / bundles
        return False


_needs_gpu = pytest.mark.skipif(
    not _gpu_stack_present(),
    reason="cupy / nvcomp / fsaverage6 avg_mesh bundles not staged")


@_needs_gpu
def test_single_subject_in_memory_chain_matches_disk(tmp_path, monkeypatch):
    subjects, sessions = ["1"], ["1", "2"]
    proj = _stage_project(tmp_path, subjects, sessions)

    sink: dict = {}
    handles: list = []
    paths = run_generate_profiles(
        project_dir=proj, subjects=subjects, sessions=sessions,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
        packed_sink=sink, write_handles=handles,
    )
    assert [p[0] for p in paths] == ["1"]
    assert len(handles) == 1, "single subject defers its .b2nd join"
    packed, K = sink["1"]
    assert packed.dtype == np.uint8 and packed.shape[0] == len(sessions)

    avg_mem = run_avg_profiles(
        project_dir=proj, num_sub=1, num_sess=len(sessions),
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
        packed_subjects=[packed], D=K,
    )
    assert avg_mem.lh_avg_dev is not None and avg_mem.writer is not None

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

    # The in-memory slab equals what the .b2nd decodes to.
    on_disk, D_disk = read_subject_profile_packed_tnd(paths[0][1])
    assert D_disk == K
    assert np.array_equal(on_disk, packed)

    # Disk-mediated avg (re-reads the .b2nd) equals the in-memory avg.
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
    sink: dict = {}
    handles: list = []
    paths = run_generate_profiles(
        project_dir=proj, subjects=subjects, sessions=sessions,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=False,
        packed_sink=sink, write_handles=handles,
    )
    assert [p[0] for p in paths] == ["1", "2"]
    assert handles == [], "multi-subject cohorts join each write inline"
    for sub_id, p in paths:
        on_disk, D = read_subject_profile_packed_tnd(p)
        packed, K = sink[sub_id]
        assert D == K and np.array_equal(on_disk, packed)
    # Pageable copies: mutating one must not alias the other.
    assert not np.shares_memory(sink["1"][0], sink["2"][0])


def test_cpu_backend_rejects_the_packed_handoff(tmp_path):
    """The CPU avg leaf reads the .b2nd files back; it has no packed
    in-memory entry point."""
    with pytest.raises(ValueError, match="GPU-only"):
        run_avg_profiles(
            project_dir=tmp_path, num_sub=1, num_sess=1,
            seed_mesh=_SEED, targ_mesh=_MESH, backend="cpu",
            packed_subjects=[np.zeros((1, 2, 1), np.uint8)], D=3,
        )


def _stub_packed_avg(monkeypatch, calls: list):
    """Replace the GPU avg leaf so the packed branch's argument checks
    can run without a device."""
    mod = types.ModuleType("arealmshbm.avg_profiles.avg_profiles_gpu")
    mod.avg_profiles_from_packed_gpu = (
        lambda *a, **k: calls.append((a, k)) or "avg"
    )
    monkeypatch.setitem(
        sys.modules, "arealmshbm.avg_profiles.avg_profiles_gpu", mod)


def test_packed_handoff_rejects_a_session_count_mismatch(tmp_path, monkeypatch):
    """Same invariant the disk branch enforces on the decoded .b2nd:
    the slab's T must be ``num_sess``."""
    calls: list = []
    _stub_packed_avg(monkeypatch, calls)
    packed = np.zeros((2, 4, 1), np.uint8)          # T=2

    with pytest.raises(ValueError, match=r"T=2 but num_sess=6"):
        run_avg_profiles(
            project_dir=tmp_path, num_sub=1, num_sess=6,
            seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu",
            packed_subjects=[packed], D=3,
        )
    assert calls == [], "the leaf must not be reached"

    assert run_avg_profiles(
        project_dir=tmp_path, num_sub=1, num_sess=2,
        seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu",
        packed_subjects=[packed], D=3,
    ) == "avg"
    assert len(calls) == 1


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
        ingest = "cpu-reader"
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

    import arealmshbm.data_io._nvcomp_batched as nvc
    monkeypatch.setattr(nvc, "nvcomp_available", lambda: True)

    def _call(verbose: bool):
        capsys.readouterr()
        step1_runners.run_generate_profiles(
            project_dir=tmp_path, subjects=["1"], sessions=["1", "2"],
            seed_mesh=_SEED, targ_mesh=_MESH, backend="gpu", verbose=verbose,
        )
        return capsys.readouterr().out

    out = _call(True)
    assert "sub=1" in out and "sessions=2" in out and "K=55" in out
    assert "ingest=cpu-reader" in out
    assert _call(False) == ""
