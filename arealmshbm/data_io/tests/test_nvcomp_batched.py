"""The ctypes binding to nvCOMP batched Deflate: round-trip equality
with ``zlib``, the input-alignment planner, and the exact per-chunk
size / status the reader's ``Dim0 * 4`` check is built on.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from arealmshbm.data_io._nvcomp_batched import nvcomp_available

try:
    import cupy as _cp
    import nvidia.nvcomp as _nvcomp
    _GPU = _cp is not None and _nvcomp is not None
    _GPU_WHY = ""
except ImportError as _e:  # pragma: no cover — GPU stack is in the env
    _GPU = False
    _GPU_WHY = f"GPU stack unavailable ({_e})"


def _batch():
    from arealmshbm.data_io._nvcomp_batched import get_deflate_batch
    return get_deflate_batch()


def test_nvcomp_available_returns_a_bool():
    """The driver calls this to pick the ingest backend; it must not raise."""
    assert isinstance(nvcomp_available(), bool)
    assert nvcomp_available() is nvcomp_available()


@pytest.mark.parametrize("exc", [OSError, AttributeError, RuntimeError])
def test_nvcomp_available_false_when_the_binding_cannot_be_built(exc):
    """Locating the library is not the same as being able to bind it.

    ``ctypes.CDLL`` raises ``OSError`` and a missing symbol raises
    ``AttributeError`` — neither is the ``ImportError`` the step-1
    callers catch, so probing only ``_find_library()`` promised a
    backend that then failed hard instead of falling back.
    """
    from arealmshbm.data_io import _nvcomp_batched as nb

    def _boom():
        raise exc("nvCOMP binding is broken")

    saved = dict(nb._CACHE)
    nb._CACHE.clear()
    real = nb.DeflateBatch
    try:
        nb.DeflateBatch = _boom
        assert nb.nvcomp_available() is False
        assert nb._CACHE["avail"] is False, "the verdict must be cached"
    finally:
        nb.DeflateBatch = real
        nb._CACHE.clear()
        nb._CACHE.update(saved)


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_nvcomp_available_probes_the_real_constructor():
    """True must mean ``get_deflate_batch()`` actually works."""
    from arealmshbm.data_io import _nvcomp_batched as nb

    saved = dict(nb._CACHE)
    nb._CACHE.clear()
    try:
        assert nb.nvcomp_available() is True
        assert nb._CACHE.get("b") is not None
    finally:
        nb._CACHE.clear()
        nb._CACHE.update(saved)


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_plan_aligns_every_chunk_and_never_overlaps():
    nb = _batch()
    assert nb.input_align >= 1
    lens = np.array([3, 6, 9, 12, 1, 4096, 163851], dtype=np.int64)
    for skip in (0, 2):
        off, total = nb.plan(lens, header_skip=skip)
        assert np.all((off + skip) % nb.input_align == 0)
        # slots hold their payload and do not run into the next one
        assert np.all(off[1:] >= off[:-1] + lens[:-1])
        assert total >= int(off[-1] + lens[-1])
    off, total = nb.plan(np.zeros(0, dtype=np.int64))
    assert off.shape == (0,) and total == 0


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_decompress_async_matches_zlib_and_reports_exact_sizes():
    import cupy as cp

    nb = _batch()
    rng = np.random.default_rng(7)
    payloads = [rng.standard_normal(n).astype(np.float32).tobytes()
                for n in (1024, 4096, 40962, 17)]
    # raw DEFLATE: strip zlib's 2-byte header and 4-byte adler32
    streams = [zlib.compress(p, 6)[2:-4] for p in payloads]
    lens = np.array([len(s) for s in streams], dtype=np.int64)
    off, total = nb.plan(lens, header_skip=0)

    src = cp.zeros(total, dtype=cp.uint8)
    for o, s in zip(off, streams):
        src[int(o):int(o) + len(s)] = cp.asarray(
            np.frombuffer(s, dtype=np.uint8))
    caps = np.array([len(p) for p in payloads], dtype=np.uint64)
    dst_off = np.zeros(len(payloads), dtype=np.int64)
    np.cumsum(caps[:-1].astype(np.int64), out=dst_off[1:])
    dst = cp.empty(int(caps.sum()), dtype=cp.uint8)

    stream = cp.cuda.Stream(non_blocking=True)
    actual = cp.zeros(len(payloads), dtype=cp.uint64)
    statuses = cp.full(len(payloads), -1, dtype=cp.int32)
    tb = nb.temp_bytes(len(payloads), int(caps.max()), int(caps.sum()))
    temp = cp.empty(tb, dtype=cp.uint8) if tb else None
    # The staging above ran on the current stream; a non-blocking stream
    # has no implicit dependency on it, so hand over through an event —
    # the same prep-event handshake ``gifti_bold_gpu`` uses.
    stream.wait_event(cp.cuda.get_current_stream().record())
    with stream:
        nb.decompress_async(
            src_ptrs=cp.asarray((off + int(src.data.ptr)).astype(np.uint64)),
            src_bytes=cp.asarray(lens.astype(np.uint64)),
            dst_ptrs=cp.asarray((dst_off + int(dst.data.ptr)).astype(np.uint64)),
            dst_capacity=cp.asarray(caps),
            actual_bytes=actual, statuses=statuses, temp=temp, stream=stream)
    stream.synchronize()

    assert np.array_equal(cp.asnumpy(statuses),
                          np.zeros(len(payloads), np.int32))
    assert np.array_equal(cp.asnumpy(actual), caps)
    got = cp.asnumpy(dst)
    for o, p in zip(dst_off, payloads):
        assert got[int(o):int(o) + len(p)].tobytes() == p


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_decompress_async_reports_short_and_overlong_chunks():
    """The exact length contract, at the binding level: neither a short
    nor an overlong inflate may be reported as the reserved size."""
    import cupy as cp

    nb = _batch()
    want = 4096
    short = zlib.compress(np.zeros(want // 4 - 1, np.float32).tobytes(),
                          6)[2:-4]
    over = zlib.compress(np.zeros(want // 4 + 8, np.float32).tobytes(),
                         6)[2:-4]
    for stream_bytes, label in ((short, "short"), (over, "long")):
        lens = np.array([len(stream_bytes)], dtype=np.int64)
        off, total = nb.plan(lens, header_skip=0)
        src = cp.zeros(total, dtype=cp.uint8)
        src[int(off[0]):int(off[0]) + len(stream_bytes)] = cp.asarray(
            np.frombuffer(stream_bytes, dtype=np.uint8))
        dst = cp.zeros(want, dtype=cp.uint8)
        actual = cp.zeros(1, dtype=cp.uint64)
        statuses = cp.full(1, -1, dtype=cp.int32)
        st = cp.cuda.Stream(non_blocking=True)
        st.wait_event(cp.cuda.get_current_stream().record())
        with st:
            nb.decompress_async(
                src_ptrs=cp.asarray(np.array(
                    [int(src.data.ptr) + int(off[0])], dtype=np.uint64)),
                src_bytes=cp.asarray(lens.astype(np.uint64)),
                dst_ptrs=cp.asarray(np.array([int(dst.data.ptr)],
                                             dtype=np.uint64)),
                dst_capacity=cp.asarray(np.array([want], dtype=np.uint64)),
                actual_bytes=actual, statuses=statuses, temp=None, stream=st)
        st.synchronize()
        ok = (int(cp.asnumpy(actual)[0]) == want
              and int(cp.asnumpy(statuses)[0]) == 0)
        assert not ok, f"{label} chunk was accepted as exactly {want} bytes"
