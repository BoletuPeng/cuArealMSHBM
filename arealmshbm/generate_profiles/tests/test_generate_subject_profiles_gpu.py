"""test_generate_subject_profiles_gpu.py

The subject orchestration: ingest -> fused compute -> background .b2nd
write, plus the result handle's write-failure contract.

The write is incremental -- one .b2nd chunk per session, handed to the
writer thread as soon as that session's D2H is enqueued -- so the tests
below also pin the per-session callback contract (``on_session``), that
a failure on either side leaves no half-written profile, and that
``write=False`` still skips the lot.

The end-to-end cases need the shipped ``fsaverage6`` mesh bundle (it
resolves the real MARS_label seed / medial-wall vectors; see
``arealmshbm/data/README.md``) and the ``nvidia-nvcomp-cu12`` package,
so they skip when either is absent. An installed nvCOMP that does not
load is a failure, not a skip.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.generate_profiles import profiles_subject_gpu as M  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Result handle
# ─────────────────────────────────────────────────────────────────────
def test_result_wait_is_idempotent():
    res = M.SubjectProfilesResult(np.zeros(64, np.uint8), 8,
                                  Path("nowhere.b2nd"), None)
    assert res.wait() == Path("nowhere.b2nd")
    assert res.wait() == Path("nowhere.b2nd")


# ─────────────────────────────────────────────────────────────────────
# End to end on synthetic fsaverage6 BOLD
# ─────────────────────────────────────────────────────────────────────
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


def _e2e_skip_reason() -> str:
    """Empty when the end-to-end tests can run. Only the nvCOMP
    *package* is probed: an installed binding that fails to load must
    fail these tests, not skip them."""
    try:
        from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
        load_avg_mesh("lh", "fsaverage6", "inflated")
    except Exception as e:      # noqa: BLE001 — absent, stale or corrupt
        return f"fsaverage6 mesh bundle not loadable ({e})"
    if not _nvcomp_installed():
        return "nvidia-nvcomp-cu12 not installed"
    return ""


_E2E_SKIP = _e2e_skip_reason()
_needs_e2e = pytest.mark.skipif(bool(_E2E_SKIP), reason=_E2E_SKIP)


def _gifti(values: np.ndarray) -> bytes:
    """``(T, N) fp32`` -> a minimal contract-compliant ``.func.gii``."""
    T, N = values.shape
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>\n',
             b'<GIFTI Version="1.0" NumberOfDataArrays="',
             str(T).encode(), b'">\n']
    for t in range(T):
        payload = base64.b64encode(
            zlib.compress(np.ascontiguousarray(values[t],
                                               dtype=np.float32).tobytes()))
        parts.append(
            f'<DataArray Intent="NIFTI_INTENT_TIME_SERIES" '
            f'DataType="NIFTI_TYPE_FLOAT32" '
            f'ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" '
            f'Encoding="GZipBase64Binary" Endian="LittleEndian" '
            f'ExternalFileName="" ExternalFileOffset="0" '
            f'Dim0="{N}">'.encode())
        parts.append(b"<Data>" + payload + b"</Data></DataArray>\n")
    parts.append(b"</GIFTI>\n")
    return b"".join(parts)


@_needs_e2e
def test_end_to_end_matches_the_compute_leaf_and_writes_the_b2nd(
        tmp_path: Path):
    from arealmshbm.data_io.profile_io import read_subject_profile_packed_tnd

    seed_idx, mw, n_lh, n_rh = M.subject_seed_and_mw("fsaverage3",
                                                      "fsaverage6")
    n_full = n_lh + n_rh
    assert mw.shape == (n_full,)
    # A seed vertex is never a medial-wall vertex (MARS_label 2 vs 1).
    assert not mw[seed_idx].any()

    T, n_sess = 4, 2
    rng = np.random.default_rng(0)
    fl = tmp_path / "data_list" / "fMRI_list"
    fl.mkdir(parents=True)
    raw = []
    for s in range(1, n_sess + 1):
        lh = rng.standard_normal((T, n_lh)).astype(np.float32)
        rh = rng.standard_normal((T, n_rh)).astype(np.float32)
        lh_p = tmp_path / f"s{s}_L.func.gii"
        rh_p = tmp_path / f"s{s}_R.func.gii"
        lh_p.write_bytes(_gifti(lh))
        rh_p.write_bytes(_gifti(rh))
        (fl / f"lh_sub1_sess{s}.txt").write_text(str(lh_p) + "\n")
        (fl / f"rh_sub1_sess{s}.txt").write_text(str(rh_p) + "\n")
        raw.append(np.concatenate([lh, rh], axis=1))

    out = tmp_path / "p.b2nd"
    res = M.generate_subject_profiles_gpu(
        tmp_path, "1", [str(i) for i in range(1, n_sess + 1)],
        out_path=out, seed_mesh="fsaverage3", targ_mesh="fsaverage6",
        threshold=0.1)
    packed = np.array(res.packed, copy=True)
    assert res.wait() == out
    assert res.K == int(seed_idx.size)
    assert packed.shape == (n_sess, n_full, (res.K + 7) // 8)

    # Same bytes as feeding the compute leaf the host-decoded buffers.
    want, K2 = M.compute_subject_profiles_gpu(
        [(i, cp.asarray(a)) for i, a in enumerate(raw)],
        n_sess=n_sess, n_full=n_full, seed_idx=seed_idx, mw_mask=mw,
        threshold=0.1)
    assert K2 == res.K
    assert np.array_equal(packed, want)

    on_disk, D = read_subject_profile_packed_tnd(out)
    assert D == res.K
    assert np.array_equal(on_disk, packed)
    cp.get_default_memory_pool().free_all_blocks()


@_needs_e2e
def test_prewarm_is_idempotent_and_thread_safe():
    import threading

    errs = []

    def _go():
        try:
            M.prewarm_generate_profiles_gpu()
        except BaseException as e:          # pragma: no cover
            errs.append(e)

    ths = [threading.Thread(target=_go, daemon=True) for _ in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=120)
    assert not errs, errs
    M.prewarm_generate_profiles_gpu()       # a fifth call is a no-op


# --------------------------------------------------------------------
# Abandoning the ingest generator must not deadlock
# --------------------------------------------------------------------
def _slow_source(n=40):
    def make_iter():
        for i in range(n):
            yield (i, cp.zeros(4, dtype=cp.float32))
    return make_iter


def test_threaded_generator_close_returns():
    """The producer parks in ``q.put`` on the bounded queue; closing the
    generator must drain it, not join a blocked thread."""
    import threading

    gen = M._threaded(_slow_source(), int(cp.cuda.runtime.getDevice()))
    next(gen)
    next(gen)
    done = []
    th = threading.Thread(target=lambda: (gen.close(), done.append(True)))
    th.start()
    th.join(timeout=30)
    assert done, "gen.close() deadlocked"
    assert not any(t.name == "profile-ingest" for t in threading.enumerate())


def test_threaded_generator_survives_a_consumer_raise():
    import gc
    import threading

    gen = M._threaded(_slow_source(), int(cp.cuda.runtime.getDevice()))
    with pytest.raises(RuntimeError, match="boom"):
        for i, _item in enumerate(gen):
            if i == 2:
                raise RuntimeError("boom")
    del gen
    done = []
    th = threading.Thread(target=lambda: (gc.collect(), done.append(True)))
    th.start()
    th.join(timeout=30)
    assert done, "gc.collect() deadlocked on the abandoned generator"
    assert not any(t.name == "profile-ingest" for t in threading.enumerate())


@_needs_e2e
def test_a_compute_side_error_propagates_without_hanging(tmp_path: Path):
    """End to end: a bad censor vector makes the COMPUTE loop raise
    while the ingest producer is still mid-flight."""
    import gc
    import threading

    seed_idx, mw, n_lh, n_rh = M.subject_seed_and_mw("fsaverage3",
                                                     "fsaverage6")
    T, n_sess = 4, 3
    rng = np.random.default_rng(3)
    fl = tmp_path / "data_list" / "fMRI_list"
    fl.mkdir(parents=True)
    cl = tmp_path / "data_list" / "censor_list"
    cl.mkdir(parents=True)
    for s in range(1, n_sess + 1):
        lh_p = tmp_path / f"s{s}_L.func.gii"
        rh_p = tmp_path / f"s{s}_R.func.gii"
        lh_p.write_bytes(_gifti(
            rng.standard_normal((T, n_lh)).astype(np.float32)))
        rh_p.write_bytes(_gifti(
            rng.standard_normal((T, n_rh)).astype(np.float32)))
        (fl / f"lh_sub1_sess{s}.txt").write_text(str(lh_p) + "\n")
        (fl / f"rh_sub1_sess{s}.txt").write_text(str(rh_p) + "\n")
    bad = tmp_path / "censor.txt"
    bad.write_text("\n".join(["1"] * (T - 1)) + "\n")
    (cl / "sub1_sess1.txt").write_text(str(bad) + "\n")

    with pytest.raises(ValueError, match="censor length"):
        M.generate_subject_profiles_gpu(
            tmp_path, "1", [str(i) for i in range(1, n_sess + 1)],
            out_path=tmp_path / "p.b2nd", seed_mesh="fsaverage3",
            targ_mesh="fsaverage6", threshold=0.1)
    done = []
    th = threading.Thread(target=lambda: (gc.collect(), done.append(True)))
    th.start()
    th.join(timeout=60)
    assert done, "gc.collect() deadlocked after the compute-side error"


# --------------------------------------------------------------------
# Sessions of differing length
# --------------------------------------------------------------------
def test_t_blocks_splits_on_the_header_hint(tmp_path: Path):
    def w(name, T):
        p = tmp_path / name
        p.write_bytes(_gifti(np.zeros((T, 3), dtype=np.float32)))
        return str(p)

    pairs = [(w("a_L.gii", 5), w("a_R.gii", 5)),
             (w("b_L.gii", 5), w("b_R.gii", 5)),
             (w("c_L.gii", 7), w("c_R.gii", 7))]
    assert M._t_blocks(pairs) == [(0, 2), (2, 3)]
    # A uniform subject stays ONE block -- one reader call, no bubble.
    assert M._t_blocks(pairs[:2]) == [(0, 2)]
    # An unprobeable pair is isolated rather than guessed at.
    pairs2 = pairs[:2] + [("no_such_L.gii", "no_such_R.gii")]
    assert M._t_blocks(pairs2) == [(0, 2), (2, 3)]
    # Hemispheres that disagree are isolated too.
    pairs3 = [(w("d_L.gii", 5), w("d_R.gii", 6)), pairs[0]]
    assert M._t_blocks(pairs3) == [(0, 1), (1, 2)]


@_needs_e2e
def test_sessions_may_differ_in_T_and_run_count(tmp_path: Path):
    """Sessions of different length and a session with two runs, through
    the real ingest: ``_t_blocks`` splits the reader calls on ``T`` and
    ``_grouped`` folds the flat per-run stream back into sessions
    (``run_counts=[2, 1, 1]`` across three reader calls). The oracle is
    the compute leaf fed the same arrays directly; the .b2nd read back
    from disk must equal the in-memory slab."""
    from arealmshbm.data_io.profile_io import read_subject_profile_packed_tnd

    seed_idx, mw, n_lh, n_rh = M.subject_seed_and_mw("fsaverage3",
                                                     "fsaverage6")
    n_full = n_lh + n_rh
    # sess 1: two runs of T=4; sess 2: one run of T=6; sess 3: one of T=4.
    layout = [(4, 4), (6,), (4,)]
    rng = np.random.default_rng(11)
    fl = tmp_path / "data_list" / "fMRI_list"
    fl.mkdir(parents=True)
    raw = []
    for si, Ts in enumerate(layout, start=1):
        lh_paths, rh_paths, runs = [], [], []
        for r, T in enumerate(Ts):
            lh = rng.standard_normal((T, n_lh)).astype(np.float32)
            rh = rng.standard_normal((T, n_rh)).astype(np.float32)
            lh_p = tmp_path / f"s{si}_r{r}_L.func.gii"
            rh_p = tmp_path / f"s{si}_r{r}_R.func.gii"
            lh_p.write_bytes(_gifti(lh))
            rh_p.write_bytes(_gifti(rh))
            lh_paths.append(str(lh_p))
            rh_paths.append(str(rh_p))
            runs.append(cp.asarray(np.concatenate([lh, rh], axis=1)))
        (fl / f"lh_sub1_sess{si}.txt").write_text("\n".join(lh_paths) + "\n")
        (fl / f"rh_sub1_sess{si}.txt").write_text("\n".join(rh_paths) + "\n")
        raw.append((si - 1, runs[0] if len(runs) == 1 else runs))
    sess = [str(i) for i in range(1, len(layout) + 1)]

    out = tmp_path / "p.b2nd"
    with M.generate_subject_profiles_gpu(
            tmp_path, "1", sess, out_path=out, seed_mesh="fsaverage3",
            targ_mesh="fsaverage6", threshold=0.1) as res:
        got = np.array(res.packed, copy=True)
        K = res.K
    want, K2 = M.compute_subject_profiles_gpu(
        raw, n_sess=len(layout), n_full=n_full, seed_idx=seed_idx,
        mw_mask=mw, threshold=0.1)
    assert K2 == K
    assert np.array_equal(got, want)
    on_disk, D = read_subject_profile_packed_tnd(out)
    assert D == K
    assert np.array_equal(on_disk, got)
    cp.get_default_memory_pool().free_all_blocks()


def _wrap_first_payload(gifti: bytes) -> bytes:
    """Splice four newlines into the first ``<Data>`` payload -- a
    multiple of 4 so the b64 length invariant still holds and the
    alphabet check is what fires, as in ``test_gifti_readers.py``."""
    at = gifti.index(b"<Data>") + len(b"<Data>") + 8
    return gifti[:at] + b"\n\n\n\n" + gifti[at:]


@_needs_e2e
def test_a_reader_error_on_the_producer_thread_reaches_the_caller(
        tmp_path: Path):
    """The ingest runs on the ``profile-ingest`` producer thread. A file
    the device reader refuses (a line-wrapped base64 payload) must
    surface as the reader's own ValueError in the caller's stack, with
    no .b2nd left behind and no producer thread alive. The refusal
    itself is pinned at the reader boundary in ``test_gifti_readers.py``;
    this pins the hand-off through the leaf."""
    import threading

    _seed_idx, _mw, n_lh, n_rh = M.subject_seed_and_mw("fsaverage3",
                                                        "fsaverage6")
    rng = np.random.default_rng(77)
    lh_p = tmp_path / "w_L.func.gii"
    rh_p = tmp_path / "w_R.func.gii"
    lh_p.write_bytes(_wrap_first_payload(_gifti(
        rng.standard_normal((3, n_lh)).astype(np.float32))))
    rh_p.write_bytes(_gifti(rng.standard_normal((3, n_rh)).astype(np.float32)))
    out = tmp_path / "w.b2nd"
    with pytest.raises(ValueError, match="outside the base64 alphabet"):
        M.generate_subject_profiles_gpu(
            tmp_path, "1", ["1"],
            bold_paths={"1": ([str(lh_p)], [str(rh_p)])},
            out_path=out, seed_mesh="fsaverage3", targ_mesh="fsaverage6",
            threshold=0.1)
    assert not out.exists()
    assert not any(t.name == "profile-ingest" for t in threading.enumerate())
    cp.get_default_memory_pool().free_all_blocks()


# --------------------------------------------------------------------
# Device memory is O(1) in the session count
# --------------------------------------------------------------------
def test_device_memory_does_not_grow_with_session_count():
    pool = cp.get_default_memory_pool()
    n_lh = n_rh = 2048
    n_full = n_lh + n_rh
    T = 64
    K = 64
    seed_idx = np.arange(K, dtype=np.int64)
    mw = np.zeros(n_full, dtype=np.uint8)
    sess_bytes = T * n_full * 4

    def peak(n_sess):
        pool.free_all_blocks()
        base = pool.used_bytes()
        hi = [0]

        def gen():
            for i in range(n_sess):
                yield i, cp.zeros((T, n_full), dtype=cp.float32)
                hi[0] = max(hi[0], pool.used_bytes() - base)

        M.compute_subject_profiles_gpu(
            gen(), n_sess=n_sess, n_full=n_full, seed_idx=seed_idx,
            mw_mask=mw, threshold=0.1)
        return hi[0]

    small = peak(3)
    large = peak(24)
    pool.free_all_blocks()
    # Peak must not grow per session beyond the packed output itself.
    packed_growth = (24 - 3) * n_full * ((K + 7) // 8)
    assert large - small < packed_growth + 3 * sess_bytes, (
        f"peak grew {large - small} B over 21 extra sessions "
        f"(one raw session buffer is {sess_bytes} B)")


# --------------------------------------------------------------------
# A failing background write is loud even without .wait()
# --------------------------------------------------------------------
def test_a_failed_background_write_warns_immediately():
    import warnings as _w
    from concurrent.futures import Future

    fut: Future = Future()
    res = M.SubjectProfilesResult(np.zeros(64, np.uint8), 8,
                                  Path("nowhere.b2nd"), fut)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        fut.set_exception(OSError("disk full"))
    assert any(issubclass(c.category, RuntimeWarning)
               and "disk full" in str(c.message) for c in caught), caught
    with pytest.raises(OSError, match="disk full"):
        res.wait()
    # The failure is latched: a second joiner must not read the missing
    # file as a finished write.
    with pytest.raises(OSError, match="disk full"):
        res.wait()


# --------------------------------------------------------------------
# Per-session D2H + the incremental .b2nd write
# --------------------------------------------------------------------
def test_on_session_needs_an_out_buffer():
    """The callback hands out slab VIEWS of ``out``; without one there
    is nothing to view."""
    with pytest.raises(ValueError, match="on_session needs out="):
        M.compute_subject_profiles_gpu(
            iter([]), n_sess=1, n_full=8,
            seed_idx=np.arange(4, dtype=np.int64),
            mw_mask=np.zeros(8, dtype=np.uint8), threshold=0.1,
            on_session=lambda *a: None)


def test_on_session_fires_per_session_with_a_finished_slab_view():
    """One call per session, in order, each carrying the slab of ``out``
    that session's D2H was issued into plus an event that covers it --
    which is the whole contract the .b2nd writer thread relies on."""
    n_lh = n_rh = 512
    n_full = n_lh + n_rh
    T, n_sess, K = 16, 4, 40
    D_bytes = (K + 7) // 8
    seed_idx = np.arange(K, dtype=np.int64)
    mw = np.zeros(n_full, dtype=np.uint8)
    rng = np.random.default_rng(5)
    raw = [rng.standard_normal((T, n_full)).astype(np.float32)
           for _ in range(n_sess)]

    out = np.empty((n_sess, n_full, D_bytes), dtype=np.uint8)
    seen = []

    def cb(si, slab, evt):
        # The slab must BE the destination row, not a copy of it.
        assert slab.base is out or slab.base is out.base
        evt.synchronize()
        seen.append((si, np.array(slab, copy=True)))

    got, K2 = M.compute_subject_profiles_gpu(
        [(i, cp.asarray(a)) for i, a in enumerate(raw)],
        n_sess=n_sess, n_full=n_full, seed_idx=seed_idx, mw_mask=mw,
        threshold=0.1, out=out, on_session=cb)
    assert K2 == K
    assert got is out
    assert [si for si, _ in seen] == list(range(n_sess))
    # What the callback saw == what the finished array holds, and the
    # finished array == the plain whole-subject-D2H path.
    for si, slab in seen:
        assert np.array_equal(slab, out[si])
    want, _ = M.compute_subject_profiles_gpu(
        [(i, cp.asarray(a)) for i, a in enumerate(raw)],
        n_sess=n_sess, n_full=n_full, seed_idx=seed_idx, mw_mask=mw,
        threshold=0.1)
    assert np.array_equal(out, want)
    cp.get_default_memory_pool().free_all_blocks()


def test_on_packed_fires_per_session_with_the_device_slab():
    """One call per session, in order, carrying the DEVICE slab the
    pack kernel just wrote -- equal to the finished host array's row --
    and it composes with ``on_session`` (the .b2nd writer's hook)."""
    n_lh = n_rh = 512
    n_full = n_lh + n_rh
    T, n_sess, K = 16, 4, 40
    D_bytes = (K + 7) // 8
    seed_idx = np.arange(K, dtype=np.int64)
    mw = np.zeros(n_full, dtype=np.uint8)
    rng = np.random.default_rng(6)
    raw = [rng.standard_normal((T, n_full)).astype(np.float32)
           for _ in range(n_sess)]

    seen = []

    def on_packed(si, slab_dev):
        assert isinstance(slab_dev, cp.ndarray)
        assert slab_dev.shape == (n_full, D_bytes)
        assert slab_dev.dtype == cp.uint8 and slab_dev.flags["C_CONTIGUOUS"]
        # Same stream as the pack: the copy is ordered after it.
        seen.append((si, slab_dev.copy()))

    got, K2 = M.compute_subject_profiles_gpu(
        [(i, cp.asarray(a)) for i, a in enumerate(raw)],
        n_sess=n_sess, n_full=n_full, seed_idx=seed_idx, mw_mask=mw,
        threshold=0.1, on_packed=on_packed)
    assert K2 == K
    assert [si for si, _ in seen] == list(range(n_sess))
    for si, slab in seen:
        assert np.array_equal(cp.asnumpy(slab), got[si])

    # With the per-session D2H hook as well (the production shape).
    out = np.empty((n_sess, n_full, D_bytes), dtype=np.uint8)
    seen2 = []
    got2, _ = M.compute_subject_profiles_gpu(
        [(i, cp.asarray(a)) for i, a in enumerate(raw)],
        n_sess=n_sess, n_full=n_full, seed_idx=seed_idx, mw_mask=mw,
        threshold=0.1, out=out,
        on_session=lambda si, slab, evt: evt.synchronize(),
        on_packed=lambda si, slab_dev: seen2.append(cp.asnumpy(slab_dev)))
    assert got2 is out
    assert np.array_equal(np.stack(seen2), out)
    cp.get_default_memory_pool().free_all_blocks()


def _write_subject(tmp_path: Path, n_sess: int = 3, T: int = 4,
                   seed: int = 21):
    """Stage a synthetic fsaverage6 subject; return its session ids."""
    seed_idx, mw, n_lh, n_rh = M.subject_seed_and_mw("fsaverage3",
                                                     "fsaverage6")
    rng = np.random.default_rng(seed)
    fl = tmp_path / "data_list" / "fMRI_list"
    fl.mkdir(parents=True, exist_ok=True)
    for s in range(1, n_sess + 1):
        lh_p = tmp_path / f"s{s}_L.func.gii"
        rh_p = tmp_path / f"s{s}_R.func.gii"
        lh_p.write_bytes(_gifti(
            rng.standard_normal((T, n_lh)).astype(np.float32)))
        rh_p.write_bytes(_gifti(
            rng.standard_normal((T, n_rh)).astype(np.float32)))
        (fl / f"lh_sub1_sess{s}.txt").write_text(str(lh_p) + "\n")
        (fl / f"rh_sub1_sess{s}.txt").write_text(str(rh_p) + "\n")
    return [str(i) for i in range(1, n_sess + 1)]


@_needs_e2e
def test_write_false_fills_packed_and_writes_nothing(tmp_path: Path):
    """``write=False`` must skip the whole streaming apparatus -- no
    file, no writer thread -- and still hand back the same bytes."""
    import threading

    sess = _write_subject(tmp_path)
    out = tmp_path / "skipped.b2nd"
    with M.generate_subject_profiles_gpu(
            tmp_path, "1", sess, out_path=out, seed_mesh="fsaverage3",
            targ_mesh="fsaverage6", threshold=0.1, write=False) as res:
        assert res.out_path == out
        no_write = np.array(res.packed, copy=True)
    assert not out.exists()

    with M.generate_subject_profiles_gpu(
            tmp_path, "1", sess, out_path=tmp_path / "written.b2nd",
            seed_mesh="fsaverage3", targ_mesh="fsaverage6",
            threshold=0.1) as res:
        written = np.array(res.packed, copy=True)
    assert np.array_equal(no_write, written)
    from arealmshbm.data_io.profile_io import read_subject_profile_packed_tnd
    on_disk, D = read_subject_profile_packed_tnd(tmp_path / "written.b2nd")
    assert D == res.K
    assert np.array_equal(on_disk, no_write)
    assert not any(t.name.startswith("profile-ingest")
                   for t in threading.enumerate())
    cp.get_default_memory_pool().free_all_blocks()


@_needs_e2e
def test_a_compute_side_error_leaves_no_partial_b2nd(tmp_path: Path):
    """The streaming writer creates the frame BEFORE the first session
    lands, so a mid-subject failure must take the file with it -- the
    one-shot writer could never leave a truncated profile behind and
    neither may this one."""
    sess = _write_subject(tmp_path, n_sess=3, seed=22)
    cl = tmp_path / "data_list" / "censor_list"
    cl.mkdir(parents=True, exist_ok=True)
    bad = tmp_path / "censor.txt"
    bad.write_text("1\n1\n1\n")            # T is 4, so this is short
    (cl / "sub1_sess2.txt").write_text(str(bad) + "\n")

    out = tmp_path / "partial.b2nd"
    with pytest.raises(ValueError, match="censor length"):
        M.generate_subject_profiles_gpu(
            tmp_path, "1", sess, out_path=out, seed_mesh="fsaverage3",
            targ_mesh="fsaverage6", threshold=0.1)
    assert not out.exists()
    cp.get_default_memory_pool().free_all_blocks()


@_needs_e2e
def test_a_write_side_error_reaches_wait_and_warns(tmp_path: Path):
    """A .b2nd the writer cannot even create still has to surface: as a
    RuntimeWarning when it happens, and as a real traceback out of
    .wait(). ``.packed`` is unaffected -- the compute never touched
    the writer."""
    import warnings as _w

    sess = _write_subject(tmp_path, n_sess=2, seed=23)
    blocked = tmp_path / "iam_a_directory.b2nd"
    blocked.mkdir()

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        res = M.generate_subject_profiles_gpu(
            tmp_path, "1", sess, out_path=blocked, seed_mesh="fsaverage3",
            targ_mesh="fsaverage6", threshold=0.1)
        packed = np.array(res.packed, copy=True)
        with pytest.raises(OSError):
            res.wait()
    assert packed.shape[0] == len(sess)
    assert any(issubclass(c.category, RuntimeWarning)
               and "background .b2nd write failed" in str(c.message)
               for c in caught), [str(c.message) for c in caught]
    assert blocked.is_dir()             # abort() must not eat the dir
    cp.get_default_memory_pool().free_all_blocks()


@_needs_e2e
def test_a_failed_submit_still_terminates_the_writer(tmp_path: Path,
                                                     monkeypatch):
    """``submit`` queues the work item BEFORE it starts the pool thread,
    so an interrupt inside it leaves the job running with no future to
    steer it. Without an abort sentinel on that path the writer blocks
    in ``q.get()`` forever and the interpreter's exit hook joins it."""
    import threading

    sess = _write_subject(tmp_path, n_sess=2, seed=24)
    done = threading.Event()

    class _RacingPool:
        """Enqueues the job, then fails the way an interrupt would."""

        def submit(self, fn, *args):
            def _run():
                try:
                    fn(*args)
                finally:
                    done.set()
            threading.Thread(target=_run, daemon=True).start()
            raise KeyboardInterrupt("interrupted inside submit")

    monkeypatch.setattr(M, "_write_pool", lambda: _RacingPool())
    out = tmp_path / "racing.b2nd"
    with pytest.raises(KeyboardInterrupt):
        M.generate_subject_profiles_gpu(
            tmp_path, "1", sess, out_path=out, seed_mesh="fsaverage3",
            targ_mesh="fsaverage6", threshold=0.1)
    assert done.wait(20), "the writer never saw a sentinel"
    assert not out.exists()
    cp.get_default_memory_pool().free_all_blocks()
