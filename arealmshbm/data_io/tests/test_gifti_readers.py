"""test_gifti_readers.py — the GIL-free CPU reader and the fused
device-stay subject reader.

What is pinned here
-------------------
1. **Bit-equality of the fast CPU route** — ``read_surface_gifti``
   (numba base64 + isal + staged transpose) against the reference
   route it replaced (``_parse_gifti_chunks`` + ``_decode_one_chunk``
   + strided column write), byte-for-byte on the ``uint32`` view, for
   the serial and the chunk-parallel entry points and for a payload of
   NaN / ±inf / -0.0 / subnormals.
   The oracle is ``_head_parse`` / ``_reference_read``, frozen copies
   of the shipped reader — NOT the live ``_scan_gifti_spans``, which
   the rewrite replaced and would otherwise be compared with itself.
2. **The base64 contract is not narrowed** — the default still accepts
   line-wrapped / pretty-printed payloads (via a per-chunk fallback)
   and a length that is not a multiple of 4, ``allow_wrapped_b64=False``
   opts into the strict GPU contract, and ``True`` forces the
   permissive route.
3. **``read_subject_bold_gpu`` semantics** — output equals
   ``concat_hemis_drop_medial(read_surface_bold(lh),
   read_surface_bold(rh), mask)`` bit-for-bit, including the NaN→0 and
   ±inf clamping the fused epilogue does inline, for both the batched
   and the pipelined (generator) entry points; a payload that does not
   inflate to ``Dim0 * 4`` bytes raises instead of leaking recycled
   device memory; no fixed timepoint ceiling; and the pipelined mode's
   device memory does not grow with the session count.
4. **``raw_sessions=True``** — the pre-epilogue ``(T, lh|rh)`` buffer
   equals the CPU reader's transpose, and the mask stays mandatory on
   the epilogue path.
5. **No nested-pool deadlock** — ``pool=<the caller's own pool>`` from
   inside one of its workers demotes to the serial path.
6. **The GPU base64 contract** — a ``'='`` outside the payload's final
   quad is an error (the decode table maps it to 0, so only the kernel's
   placement guard catches it) while every legal tail pad still decodes.
7. **The tag scan emits int64 offsets** — matching a host bytescan
   exactly, so there is no byte-span ceiling on a pipeline group.
8. **The shared pinned staging block is not duplicated** — a regrown
   buffer replaces the old one instead of holding both, a request
   inside the pool's rounded capacity reuses it, and a failure to pin
   the whole subject is reported as a host-staging limit, not as a
   bare CUDA OOM.

GPU tests skip cleanly when cupy / nvcomp are absent — same pattern as
``arealmshbm/local_minima/tests/test_gpu_correctness.py``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from arealmshbm.bold_io import concat_hemis_drop_medial, read_surface_bold
from arealmshbm.data_io.gifti_io import (
    _scan_gifti_spans,
    read_surface_gifti,
)

try:
    import cupy as _cp  # noqa: F401
    import nvidia.nvcomp as _nvcomp  # noqa: F401
    _GPU = True
    _GPU_WHY = ""
except ImportError as _e:  # pragma: no cover — GPU stack is in the env
    _GPU = False
    _GPU_WHY = f"GPU stack unavailable ({_e})"


def _make_gifti(dim0: int, T: int, values=None, wrap: bool = False) -> bytes:
    """Minimal single-hemi ``.func.gii`` with the production contract.

    ``values[t]`` (if given) is the ``(dim0,) fp32`` payload of darray
    ``t``; otherwise a deterministic ramp is used. ``wrap=True`` splices
    four newlines into every payload — a multiple of 4 so the b64 length
    invariant still holds and the *whitespace* branch is what fires.
    """
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>\n',
             b'<GIFTI Version="1.0" NumberOfDataArrays="',
             str(T).encode(), b'">\n']
    for t in range(T):
        arr = (values[t] if values is not None
               else (np.arange(dim0, dtype=np.float32) + t))
        payload = base64.b64encode(
            zlib.compress(np.asarray(arr, dtype=np.float32).tobytes()))
        if wrap:
            payload = payload[:8] + b"\n\n\n\n" + payload[8:]
        parts.append(
            f'<DataArray Intent="NIFTI_INTENT_TIME_SERIES" '
            f'DataType="NIFTI_TYPE_FLOAT32" '
            f'ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" '
            f'Encoding="GZipBase64Binary" Endian="LittleEndian" '
            f'ExternalFileName="" ExternalFileOffset="0" '
            f'Dim0="{dim0}">'.encode())
        parts.append(b"<Data>" + payload + b"</Data></DataArray>\n")
    parts.append(b"</GIFTI>\n")
    return b"".join(parts)


def _head_parse(path):
    """The pre-optimization ``_parse_gifti_chunks``, frozen verbatim.

    The optimized parser rewrote ``_parse_gifti_chunks`` into a wrapper
    over the new one-pass ``_scan_gifti_spans``, so comparing against
    the live function would compare the new scanner with itself. This
    is the three-sweep original (two unbounded ``bytes.find`` passes
    plus a bounded ``</Data>`` search per darray) copied out of the
    shipped reader, so the span logic actually has an oracle.
    """
    with open(path, "rb") as f:
        data = f.read()
    headers = []
    pos = 0
    while True:
        s = data.find(b"<DataArray", pos)
        if s < 0:
            break
        e = data.find(b">", s)
        if e < 0:
            raise ValueError(f"GIFTI {path}: unterminated <DataArray>")
        headers.append((s, e + 1))
        pos = e + 1
    if not headers:
        raise ValueError(f"GIFTI {path}: no <DataArray> element found")
    first = data[headers[0][0]:headers[0][1]]
    N = int(re.search(rb'Dim0="(\d+)"', first).group(1))
    chunks = []
    for i, (s, e) in enumerate(headers):
        boundary = headers[i + 1][0] if i + 1 < len(headers) else len(data)
        s_data = data.find(b"<Data>", e, boundary)
        if s_data < 0:
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: missing <Data> element")
        s_data += len(b"<Data>")
        e_data = data.find(b"</Data>", s_data, boundary)
        if e_data < 0:
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: unterminated <Data>")
        chunks.append(data[s_data:e_data])
    return N, len(chunks), chunks


def _scan_chunks(path):
    """``(N, T, chunks)`` off the live span scanner -- the ``bytes``-
    materialising wrapper the reader itself never needed, kept with its
    only user."""
    data = Path(path).read_bytes()
    N, T, _hs, _he, ds, de = _scan_gifti_spans(data, path)
    return N, T, [data[ds[i]:de[i]] for i in range(T)]


def _reference_read(path) -> np.ndarray:
    """The pre-optimization decode route, kept here as the oracle."""
    N, T, chunks = _head_parse(path)
    out = np.empty((N, T), dtype=np.float32)
    for t, c in enumerate(chunks):
        raw = zlib.decompress(base64.b64decode(c))
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.shape[0] != N:
            raise ValueError(
                f"GIFTI darray length {arr.shape[0]} != expected {N}; "
                f"Dim0 disagrees with decoded byte count")
        out[:, t] = arr
    return out


def test_span_scanner_matches_frozen_head_parser(tmp_path: Path):
    """``_scan_gifti_spans`` replaced a three-sweep scan with a memchr
    pass plus a fallback; this is the oracle that change never had."""
    for dim0, T in [(8, 3), (255, 17), (1024, 5), (37, 41)]:
        p = tmp_path / f"scan{dim0}_{T}.func.gii"
        p.write_bytes(_make_gifti(dim0, T))
        n_ref, t_ref, chunks_ref = _head_parse(p)
        n_new, t_new, chunks_new = _scan_chunks(p)
        assert (n_ref, t_ref) == (n_new, t_new)
        assert chunks_ref == chunks_new


def _bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality including NaN payloads and signed zeros."""
    return (a.shape == b.shape
            and np.array_equal(a.view(np.uint32), b.view(np.uint32)))


# ── CPU reader ───────────────────────────────────────────────────────
@pytest.mark.parametrize("dim0,T", [(8, 3), (255, 17), (1024, 5)])
def test_cpu_reader_bit_equal_to_reference(tmp_path: Path, dim0, T):
    p = tmp_path / "a.func.gii"
    p.write_bytes(_make_gifti(dim0, T))
    assert _bit_equal(read_surface_gifti(p), _reference_read(p))


def test_cpu_reader_bit_equal_on_exotic_floats(tmp_path: Path):
    """NaN (both quiet payloads), ±inf, -0.0 and subnormals.

    The reader never inspects a value — it moves bytes — so the point
    is that nothing in the decode/transpose path normalises a bit
    pattern. Compared on the ``uint32`` view, where -0.0 and a signalling
    NaN are distinguishable from their look-alikes.
    """
    special = np.array(
        [np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0,
         np.float32(1e-45), np.float32(-1e-45), np.float32(1.1754944e-38),
         np.float32(5.877472e-39), 1.0, -1.0], dtype=np.float32)
    # A signalling NaN payload numpy will not produce arithmetically.
    snan = np.array([0x7F800001, 0xFF800001], dtype=np.uint32).view(np.float32)
    vals = [np.concatenate([special, snan]) for _ in range(3)]
    p = tmp_path / "exotic.func.gii"
    p.write_bytes(_make_gifti(vals[0].shape[0], 3, values=vals))
    got = read_surface_gifti(p)
    assert _bit_equal(got, _reference_read(p))
    assert np.array_equal(got[:, 0].view(np.uint32),
                          vals[0].view(np.uint32))


def test_cpu_reader_chunk_parallel_matches_serial(tmp_path: Path):
    """The chunk-range split must not perturb a single byte — the
    ranges are disjoint writes into a shared staging buffer, so this
    is the test that would catch an off-by-one in the split bounds."""
    p = tmp_path / "b.func.gii"
    p.write_bytes(_make_gifti(300, 37))
    ref = read_surface_gifti(p)
    for nw in (2, 3, 8, 64):
        assert _bit_equal(read_surface_gifti(p, n_workers=nw), ref), nw
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert _bit_equal(read_surface_gifti(p, pool=pool, n_workers=4), ref)


def test_cpu_reader_padding_lengths(tmp_path: Path):
    """``dim0`` chosen so the compressed payload hits each of the three
    base64 padding cases across timepoints — the strict kernel's ``=``
    branches are the easiest thing to get subtly wrong."""
    rng = np.random.default_rng(0)
    for dim0 in range(1, 40):
        vals = [rng.standard_normal(dim0).astype(np.float32)
                for _ in range(3)]
        p = tmp_path / f"pad{dim0}.func.gii"
        p.write_bytes(_make_gifti(dim0, 3, values=vals))
        assert _bit_equal(read_surface_gifti(p), _reference_read(p)), dim0


def test_cpu_reader_time_major_is_the_transpose(tmp_path: Path):
    """``time_major=True`` hands back the decode's own ``(T, N)`` layout:
    the default output transposed, bit for bit, on every route (serial,
    chunk-parallel, and the wrapped-b64 fallback). Step 1 reads this
    way, so the host transpose the reader used to do and the one the
    consumer undid are both gone."""
    rng = np.random.default_rng(9)
    vals = rng.standard_normal((5, 96)).astype(np.float32)
    p = tmp_path / "tm.func.gii"
    p.write_bytes(_make_gifti(96, 5, values=vals))
    want = np.ascontiguousarray(read_surface_gifti(p).T)
    got = read_surface_gifti(p, time_major=True)
    assert got.shape == (5, 96) and got.flags["C_CONTIGUOUS"]
    assert _bit_equal(got, want)
    assert _bit_equal(read_surface_gifti(p, time_major=True, n_workers=3),
                      want)
    w = tmp_path / "tmw.func.gii"
    w.write_bytes(_make_gifti(96, 5, values=vals, wrap=True))
    assert _bit_equal(read_surface_gifti(w, time_major=True), want)


def test_cpu_reader_accepts_wrapped_b64_by_default(tmp_path: Path):
    """No narrowing of the shipped reader's accepted-input set.

    The strict numba kernel refuses a whitespace-wrapped payload; the
    default ``allow_wrapped_b64=None`` must fall back to
    ``base64.b64decode`` for that chunk and produce exactly what the
    pre-optimization reader produced.
    """
    vals = [np.linspace(-1, 1, 64, dtype=np.float32) * (t + 1)
            for t in range(3)]
    wrapped = tmp_path / "wrapped.func.gii"
    wrapped.write_bytes(_make_gifti(64, 3, values=vals, wrap=True))
    assert _bit_equal(read_surface_gifti(wrapped), _reference_read(wrapped))

    # A lone spliced newline breaks the length-multiple-of-4 invariant,
    # i.e. the other refusal code the auto fallback must cover.
    odd = tmp_path / "odd.func.gii"
    odd.write_bytes(_make_gifti(64, 3, values=vals).replace(
        b"<Data>", b"<Data>\n"))
    assert _bit_equal(read_surface_gifti(odd), _reference_read(odd))

    # RFC-2045 / pretty-printer shapes, the two real-world cases.
    for name, splice in (("wrap76", b"\n"), ("pretty", b"\n  ")):
        raw = _make_gifti(500, 2, values=[
            np.arange(500, dtype=np.float32) + t for t in range(2)])
        out, i = [], 0
        while True:
            j = raw.find(b"<Data>", i)
            if j < 0:
                out.append(raw[i:])
                break
            k = raw.index(b"</Data>", j)
            body = raw[j + 6:k]
            body = splice.join(body[m:m + 76]
                               for m in range(0, len(body), 76))
            out.append(raw[i:j + 6] + splice + body + splice)
            i = k
        p = tmp_path / f"{name}.func.gii"
        p.write_bytes(b"".join(out))
        assert _bit_equal(read_surface_gifti(p), _reference_read(p)), name


def test_cpu_reader_strict_mode_rejects_wrapped_b64(tmp_path: Path):
    """``allow_wrapped_b64=False`` opts into the GPU readers' contract."""
    p = tmp_path / "wrapped2.func.gii"
    p.write_bytes(_make_gifti(64, 3, wrap=True))
    with pytest.raises(ValueError, match="outside the base64 alphabet"):
        read_surface_gifti(p, allow_wrapped_b64=False)


def test_cpu_reader_wrapped_b64_forced_permissive(tmp_path: Path):
    """``allow_wrapped_b64=True`` routes every chunk through
    ``base64.b64decode`` and still yields the reference values."""
    vals = [np.linspace(-1, 1, 64, dtype=np.float32) * (t + 1)
            for t in range(3)]
    straight = tmp_path / "s.func.gii"
    straight.write_bytes(_make_gifti(64, 3, values=vals))
    wrapped = tmp_path / "w.func.gii"
    wrapped.write_bytes(_make_gifti(64, 3, values=vals, wrap=True))
    assert _bit_equal(read_surface_gifti(wrapped, allow_wrapped_b64=True),
                      read_surface_gifti(straight))


def test_cpu_reader_nested_pool_does_not_deadlock(tmp_path: Path):
    """Passing the caller's OWN pool from inside one of its workers is
    the obvious integration mistake (the prefetchers own exactly such a
    pool). It must demote to the serial path, not hang."""
    import concurrent.futures as cf

    p = tmp_path / "nest.func.gii"
    p.write_bytes(_make_gifti(128, 9))
    ref = read_surface_gifti(p)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(read_surface_gifti, p, pool=pool, n_workers=2)
                for _ in range(2)]
        done, pending = cf.wait(futs, timeout=30)
        assert not pending, "read_surface_gifti(pool=own pool) deadlocked"
        for f in done:
            assert _bit_equal(f.result(), ref)


def test_cpu_reader_still_rejects_off_contract(tmp_path: Path):
    """The span scanner is new; the contract errors must survive it."""
    p = tmp_path / "bad.func.gii"
    p.write_bytes(_make_gifti(8, 2).replace(
        b'DataType="NIFTI_TYPE_FLOAT32"', b'DataType="NIFTI_TYPE_INT32"', 1))
    with pytest.raises(ValueError, match="DataType"):
        read_surface_gifti(p)


def test_cpu_reader_missing_data_element(tmp_path: Path):
    p = tmp_path / "nodata.func.gii"
    raw = _make_gifti(8, 2)
    # Strip the FIRST darray's <Data>...</Data> entirely.
    i = raw.index(b"<Data>")
    j = raw.index(b"</Data>") + len(b"</Data>")
    p.write_bytes(raw[:i] + raw[j:])
    with pytest.raises(ValueError, match="missing <Data> element"):
        read_surface_gifti(p)


# ── GPU subject reader ───────────────────────────────────────────────
def _host_reference(lh_path, rh_path, mask):
    return concat_hemis_drop_medial(read_surface_bold(lh_path),
                                    read_surface_bold(rh_path), mask)


def _synth_subject(tmp_path, n_lh, n_rh, T, n_sess, seed=0):
    """Write ``n_sess`` (lh, rh) pairs and return (pairs, mask)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for s in range(n_sess):
        lh_v = [rng.standard_normal(n_lh).astype(np.float32)
                for _ in range(T)]
        rh_v = [rng.standard_normal(n_rh).astype(np.float32)
                for _ in range(T)]
        # Exercise the epilogue's cleaning branches.
        lh_v[0][3] = np.nan
        rh_v[1][5] = np.inf
        rh_v[2][7] = -np.inf
        lh = tmp_path / f"s{s}_L.func.gii"
        rh = tmp_path / f"s{s}_R.func.gii"
        lh.write_bytes(_make_gifti(n_lh, T, values=lh_v))
        rh.write_bytes(_make_gifti(n_rh, T, values=rh_v))
        pairs.append((lh, rh))
    mask = np.zeros(n_lh + n_rh, dtype=bool)
    mask[rng.choice(n_lh + n_rh, size=(n_lh + n_rh) // 8,
                    replace=False)] = True
    return pairs, mask


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
@pytest.mark.parametrize("group_sessions", [None, 1, 2])
def test_subject_gpu_matches_host(tmp_path: Path, group_sessions):
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    pairs, mask = _synth_subject(tmp_path, 97, 89, 6, 4)
    got = read_subject_bold_gpu(pairs, medial_mask_d=cp.asarray(mask),
                                n_lh=97, n_rh=89,
                                group_sessions=group_sessions)
    assert len(got) == len(pairs)
    for (lh, rh), g in zip(pairs, got):
        assert _bit_equal(cp.asnumpy(g), _host_reference(lh, rh, mask))


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_pipelined_order_and_values(tmp_path: Path):
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import iter_subject_bold_gpu

    pairs, mask = _synth_subject(tmp_path, 64, 64, 5, 5, seed=1)
    seen = []
    for idx, arr in iter_subject_bold_gpu(
            pairs, medial_mask_d=cp.asarray(mask), n_lh=64, n_rh=64,
            group_sessions=1):
        seen.append(idx)
        lh, rh = pairs[idx]
        assert _bit_equal(cp.asnumpy(arr), _host_reference(lh, rh, mask))
    assert seen == list(range(len(pairs)))


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_rejects_wrapped_b64(tmp_path: Path):
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    lh = tmp_path / "w_L.func.gii"
    rh = tmp_path / "w_R.func.gii"
    lh.write_bytes(_make_gifti(32, 3, wrap=True))
    rh.write_bytes(_make_gifti(32, 3))
    with pytest.raises(ValueError, match="outside the base64 alphabet"):
        read_subject_bold_gpu([(lh, rh)],
                              medial_mask_d=cp.zeros(64, dtype=cp.bool_),
                              n_lh=32, n_rh=32)


def _with_fdict(gifti: bytes) -> bytes:
    """Set FDICT (FLG bit 5) on the first payload's zlib header, refit
    FCHECK and splice in a 4-byte DICTID the way a preset-dictionary
    writer would; the DEFLATE stream itself is untouched."""
    s = gifti.index(b"<Data>") + len(b"<Data>")
    e = gifti.index(b"</Data>", s)
    raw = base64.b64decode(gifti[s:e])
    cmf, flg_hi = raw[0], (raw[1] | 0x20) & 0xE0
    flg = flg_hi | ((31 - (cmf * 256 + flg_hi) % 31) % 31)
    assert (cmf * 256 + flg) % 31 == 0 and flg & 0x20
    fixed = bytes([cmf, flg]) + b"\x00\x00\x00\x01" + raw[2:]
    return gifti[:s] + base64.b64encode(fixed) + gifti[e:]


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_rejects_fdict(tmp_path: Path):
    """A zlib stream with FDICT set carries a DICTID after the header;
    the device reader strips exactly 2 bytes and would hand those 4
    bytes to nvCOMP as DEFLATE data. Refused by name instead -- the CPU
    reader refuses such a stream as well (isal has no dictionary to
    offer), so neither backend reads it."""
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    lh = tmp_path / "fd_L.func.gii"
    rh = tmp_path / "fd_R.func.gii"
    lh.write_bytes(_with_fdict(_make_gifti(32, 3)))
    rh.write_bytes(_make_gifti(32, 3))
    with pytest.raises(ValueError, match="FDICT"):
        read_subject_bold_gpu([(lh, rh)],
                              medial_mask_d=cp.zeros(64, dtype=cp.bool_),
                              n_lh=32, n_rh=32)
    with pytest.raises(Exception):
        read_surface_gifti(lh)


def _splice_eq(raw: bytes, quad: int, pos: int) -> bytes:
    """Put a ``'='`` at ``pos`` of ``quad`` in the FIRST ``<Data>`` payload.

    Length-preserving, so every host-side length invariant (multiple of
    4, ``Dim0 * 4`` after inflate) still holds and the *placement* check
    is the only thing that can fire.
    """
    j = raw.index(b"<Data>") + len(b"<Data>")
    k = raw.index(b"</Data>", j)
    off = j + quad * 4 + pos
    assert off < k - 4, "the target quad must not be the payload's last"
    return raw[:off] + b"=" + raw[off + 1:]


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
@pytest.mark.parametrize("pos", [0, 1, 2, 3])
def test_subject_gpu_rejects_misplaced_b64_pad(tmp_path: Path, pos):
    """A ``'='`` anywhere but the payload's tail must be an error.

    ``'='`` is in the decode table (legal tail padding needs it), so the
    kernel's alphabet guard alone accepted it mid-payload and decoded it
    to a 0 sextet. The bytes then go to nvCOMP as raw DEFLATE with the
    adler32 trailer dropped, so a stored block came back as *wrong
    floats of the right length* — silently. The CPU reader refuses all
    four positions (binascii padding / truncated stream).
    """
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    lh = tmp_path / f"eq{pos}_L.func.gii"
    rh = tmp_path / f"eq{pos}_R.func.gii"
    lh.write_bytes(_splice_eq(_make_gifti(64, 3), quad=2, pos=pos))
    rh.write_bytes(_make_gifti(64, 3))
    with pytest.raises(ValueError, match="base64"):
        read_subject_bold_gpu([(lh, rh)],
                              medial_mask_d=cp.zeros(128, dtype=cp.bool_),
                              n_lh=64, n_rh=64)


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_accepts_every_legal_tail_pad(tmp_path: Path):
    """The placement guard must not cost the legal ``'='`` padding.

    One ``dim0`` per padding class (0, 1 and 2 ``'='``), picked from the
    deterministic ramp payload, decoded and compared with the host
    route bit for bit.
    """
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    by_pad = {}
    for dim0 in range(1, 64):
        raw = _make_gifti(dim0, 2)
        body = raw[raw.index(b"<Data>") + 6:raw.index(b"</Data>")]
        by_pad.setdefault(body[-2:].count(b"="), dim0)
    assert set(by_pad) == {0, 1, 2}, by_pad

    for n_pad, dim0 in sorted(by_pad.items()):
        lh = tmp_path / f"pad{n_pad}_L.func.gii"
        rh = tmp_path / f"pad{n_pad}_R.func.gii"
        lh.write_bytes(_make_gifti(dim0, 2))
        rh.write_bytes(_make_gifti(dim0, 2))
        mask = np.zeros(2 * dim0, dtype=bool)
        got = read_subject_bold_gpu([(lh, rh)],
                                    medial_mask_d=cp.asarray(mask),
                                    n_lh=dim0, n_rh=dim0)
        assert _bit_equal(cp.asnumpy(got[0]),
                          _host_reference(lh, rh, mask)), n_pad


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_rejects_wrong_dim0(tmp_path: Path):
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    lh = tmp_path / "d_L.func.gii"
    rh = tmp_path / "d_R.func.gii"
    lh.write_bytes(_make_gifti(32, 3))
    rh.write_bytes(_make_gifti(32, 3))
    with pytest.raises(ValueError, match="Dim0"):
        read_subject_bold_gpu([(lh, rh)],
                              medial_mask_d=cp.zeros(80, dtype=cp.bool_),
                              n_lh=48, n_rh=32)


def _mismatched_dim0_gifti(dim0_claimed: int, dim0_actual: int,
                           T: int) -> bytes:
    """A GIFTI whose ``Dim0`` lies about the payload length."""
    raw = _make_gifti(dim0_actual, T)
    return raw.replace(f'Dim0="{dim0_actual}"'.encode(),
                       f'Dim0="{dim0_claimed}"'.encode())


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
@pytest.mark.parametrize("claimed,actual", [(16, 8), (8, 16)])
def test_subject_gpu_rejects_short_or_long_payload(tmp_path: Path,
                                                   claimed, actual):
    """A payload that does not inflate to ``Dim0 * 4`` bytes must be a
    named error, never uninitialised device memory.

    nvCOMP is handed pre-sized views into the session buffer, so a
    chunk that decompresses SHORT leaves the tail of its row unwritten
    and whatever the CuPy pool last held there would otherwise flow
    into the result. The CPU reader has always refused this input
    ('Dim0 disagrees with decoded byte count'); the GPU reader checks
    the same thing against nvCOMP's reported output sizes.
    """
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    lh = tmp_path / f"sz{claimed}_L.func.gii"
    rh = tmp_path / f"sz{claimed}_R.func.gii"
    lh.write_bytes(_mismatched_dim0_gifti(claimed, actual, 3))
    rh.write_bytes(_make_gifti(claimed, 3))
    # Warm the pool so a second call would see recycled, dirty blocks —
    # the exact condition under which the missing check returned
    # garbage instead of raising.
    read_subject_bold_gpu([(rh, rh)],
                          medial_mask_d=cp.zeros(2 * claimed,
                                                 dtype=cp.bool_),
                          n_lh=claimed, n_rh=claimed)
    with pytest.raises(ValueError, match="Dim0|decompress"):
        read_subject_bold_gpu([(lh, rh)],
                              medial_mask_d=cp.zeros(2 * claimed,
                                                     dtype=cp.bool_),
                              n_lh=claimed, n_rh=claimed)


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_handles_more_than_4096_tags(tmp_path: Path):
    """No fixed timepoint ceiling.

    The tag scan budgets 4096 slots per file and a GIFTI emits three
    tags per ``<DataArray>``, so T > 1365 overflowed the slot array and
    raised. It now retries once at the exact count the kernel's
    counter reports.
    """
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    T = 1400          # 4200 tags/file, above the 4096 first estimate
    lh = tmp_path / "big_L.func.gii"
    rh = tmp_path / "big_R.func.gii"
    lh.write_bytes(_make_gifti(4, T))
    rh.write_bytes(_make_gifti(4, T))
    mask = np.zeros(8, dtype=bool)
    got = read_subject_bold_gpu([(lh, rh)], medial_mask_d=cp.asarray(mask),
                                n_lh=4, n_rh=4)
    assert _bit_equal(cp.asnumpy(got[0]), _host_reference(lh, rh, mask))


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_iter_device_memory_is_bounded(tmp_path: Path):
    """Per-group transients must not accumulate over the subject.

    The first revision retained every group's base64 buffer for the
    whole generator, so the pool high-water grew linearly with the
    session count — the opposite of what the pipelined mode is for.
    """
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import iter_subject_bold_gpu

    pairs, mask = _synth_subject(tmp_path, 4096, 4096, 6, 8, seed=3)
    mask_d = cp.asarray(mask)
    pool = cp.get_default_memory_pool()
    # One warm pass so the pool is already sized for the working set.
    for _i, a in iter_subject_bold_gpu(pairs, medial_mask_d=mask_d,
                                       n_lh=4096, n_rh=4096,
                                       group_sessions=1):
        del a
    pool.free_all_blocks()
    highs = []
    for _i, a in iter_subject_bold_gpu(pairs, medial_mask_d=mask_d,
                                       n_lh=4096, n_rh=4096,
                                       group_sessions=1):
        del a
        highs.append(pool.used_bytes())
    # Flat, not monotone: allow one group's worth of slack, never eight.
    assert max(highs) <= 2 * min(highs), highs


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_rejects_host_mask(tmp_path: Path):
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    lh = tmp_path / "m_L.func.gii"
    rh = tmp_path / "m_R.func.gii"
    lh.write_bytes(_make_gifti(16, 2))
    rh.write_bytes(_make_gifti(16, 2))
    with pytest.raises(TypeError, match="cupy.ndarray"):
        read_subject_bold_gpu([(lh, rh)],
                              medial_mask_d=np.zeros(32, dtype=bool),
                              n_lh=16, n_rh=16)


# ─────────────────────────────────────────────────────────────────────
# ``raw_sessions=True`` — the pre-epilogue (T, lh|rh) session buffer
# that step 1's fused subject leaf consumes.
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
@pytest.mark.parametrize("group_sessions", [None, 1, 2, 3])
def test_subject_gpu_raw_sessions_match_cpu_reader(tmp_path: Path,
                                                   group_sessions):
    """``buf[:, :n_lh] == read_surface_gifti(lh).T`` bit for bit.

    No medial drop, no transpose, no NaN cleaning — the synthetic
    subject injects NaN/±inf and every one of them must survive, so
    the comparison is on the ``uint32`` view.
    """
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    n_lh, n_rh, T = 97, 89, 6
    pairs, _mask = _synth_subject(tmp_path, n_lh, n_rh, T, 6, seed=11)
    got = read_subject_bold_gpu(pairs, n_lh=n_lh, n_rh=n_rh,
                                group_sessions=group_sessions,
                                raw_sessions=True)
    assert len(got) == len(pairs)
    for (lh, rh), g in zip(pairs, got):
        assert g.shape == (T, n_lh + n_rh)
        assert g.dtype == cp.float32
        assert g.flags["C_CONTIGUOUS"]
        h = cp.asnumpy(g)
        assert _bit_equal(h[:, :n_lh], read_surface_gifti(lh).T)
        assert _bit_equal(h[:, n_lh:], read_surface_gifti(rh).T)


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_raw_sessions_pipelined(tmp_path: Path):
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import iter_subject_bold_gpu

    n_lh, n_rh, T = 64, 48, 5
    pairs, _mask = _synth_subject(tmp_path, n_lh, n_rh, T, 5, seed=12)
    seen = []
    for idx, arr in iter_subject_bold_gpu(pairs, n_lh=n_lh, n_rh=n_rh,
                                          group_sessions=2,
                                          raw_sessions=True):
        seen.append(idx)
        lh, rh = pairs[idx]
        h = cp.asnumpy(arr)
        assert h.shape == (T, n_lh + n_rh)
        assert _bit_equal(h[:, :n_lh], read_surface_gifti(lh).T)
        assert _bit_equal(h[:, n_lh:], read_surface_gifti(rh).T)
    assert seen == list(range(len(pairs)))


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_epilogue_path_still_needs_a_device_mask(tmp_path: Path):
    """``medial_mask_d`` only became optional for ``raw_sessions``."""
    from arealmshbm.data_io.gifti_bold_gpu import read_subject_bold_gpu

    lh = tmp_path / "n_L.func.gii"
    rh = tmp_path / "n_R.func.gii"
    lh.write_bytes(_make_gifti(16, 2))
    rh.write_bytes(_make_gifti(16, 2))
    with pytest.raises(TypeError, match="cupy.ndarray"):
        read_subject_bold_gpu([(lh, rh)], n_lh=16, n_rh=16)


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_subject_gpu_abandoned_generator_is_safe(tmp_path: Path):
    """Closing the generator early parks the in-flight keeps.

    An abandoned ``iter_subject_bold_gpu`` must not hand a buffer the
    decode stream is still reading back to the CuPy pool; the next
    ingest drains it against its own event.
    """
    import cupy as cp
    from arealmshbm.data_io.gifti_bold_gpu import (
        _PENDING_KEEPS, iter_subject_bold_gpu, read_subject_bold_gpu,
    )

    pairs, mask = _synth_subject(tmp_path, 512, 512, 4, 6, seed=13)
    mask_d = cp.asarray(mask)
    gen = iter_subject_bold_gpu(pairs, medial_mask_d=mask_d,
                                n_lh=512, n_rh=512, group_sessions=1)
    idx, arr = next(gen)
    assert idx == 0
    gen.close()
    assert _PENDING_KEEPS, "abandoned generator parked nothing"
    got = read_subject_bold_gpu(pairs, medial_mask_d=mask_d,
                                n_lh=512, n_rh=512)
    assert not _PENDING_KEEPS, "the next ingest did not drain the keeps"
    for (lh, rh), g in zip(pairs, got):
        assert _bit_equal(cp.asnumpy(g), _host_reference(lh, rh, mask))


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_tag_scan_emits_int64_positions():
    """``find_gifti_tags_batch`` positions match a host bytescan.

    The kernel writes ``long long`` offsets into the group's device
    buffer, so a group's byte span is bounded only by VRAM.
    """
    import cupy as cp
    from arealmshbm.data_io._gifti_kernels_gpu import get_gifti_gpu_kernels

    find_tags, _b64, _epi, _table = get_gifti_gpu_kernels()

    buf = bytearray(b"." * (1 << 20))
    want = []
    for k, tag in enumerate((b"<DataArray", b"<Data>", b"</Data>")):
        for j in range(200):
            off = 137 * (k * 200 + j) + 11
            buf[off:off + len(tag)] = tag
            want.append((off, k))
    want.sort()

    dev = cp.asarray(np.frombuffer(bytes(buf), dtype=np.uint8))
    span = int(dev.shape[0])
    max_found = 4096
    pos_d = cp.empty(max_found, dtype=cp.int64)
    kind_d = cp.empty(max_found, dtype=cp.int32)
    n_found_d = cp.zeros(1, dtype=cp.int32)
    block = 256
    find_tags(((span + block - 1) // block,), (block,),
              (dev, np.int64(span), pos_d, kind_d, n_found_d,
               np.int32(max_found)))
    n_found = int(cp.asnumpy(n_found_d)[0])
    assert n_found == len(want)

    assert pos_d.dtype == cp.int64
    positions = cp.asnumpy(pos_d[:n_found])
    kinds = cp.asnumpy(kind_d[:n_found])
    order = np.argsort(positions, kind="stable")
    got = list(zip(positions[order].tolist(), kinds[order].tolist()))
    assert got == want


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_staging_regrow_does_not_hold_two_blocks(monkeypatch):
    """Growing the shared pinned buffer frees the old block first, and
    the recorded capacity is what the pool actually handed back — so a
    slightly larger next request reuses the same allocation."""
    import cupy as cp
    from arealmshbm.data_io import gifti_bold_gpu as gbg

    pool = cp.get_default_pinned_memory_pool()
    gbg.release_staging_buffer()
    pool.free_all_blocks()

    calls: list = []
    free_at_alloc: list = []
    real_alloc = cp.cuda.alloc_pinned_memory

    def _probing_alloc(nbytes):
        calls.append(int(nbytes))
        free_at_alloc.append(pool.n_free_blocks())
        return real_alloc(nbytes)

    monkeypatch.setattr(cp.cuda, "alloc_pinned_memory", _probing_alloc)
    try:
        mem_a, cached_a = gbg._acquire_staging(3 << 20)
        assert cached_a
        cap = int(mem_a.mem.size)
        ptr_a = int(mem_a.ptr)
        gbg._release_staging()
        assert calls == [3 << 20] and free_at_alloc == [0]

        # Inside the pool's rounded capacity: no new allocation, and the
        # very same block comes back.
        mem_b, cached_b = gbg._acquire_staging(cap - 4096)
        assert cached_b and int(mem_b.ptr) == ptr_a
        assert len(calls) == 1
        gbg._release_staging()

        # Past it: the old block must be back in the pool BEFORE the
        # larger allocation runs, or both stay resident.
        del mem_a, mem_b
        mem_c, cached_c = gbg._acquire_staging(cap + 4096)
        assert cached_c
        assert len(calls) == 2
        assert free_at_alloc[1] == 1, "the old block was still held"
        gbg._release_staging()
        del mem_c
    finally:
        monkeypatch.setattr(cp.cuda, "alloc_pinned_memory", real_alloc)
        gbg._release_staging()          # a failed assert must not leave it busy
        gbg.release_staging_buffer()


@pytest.mark.skipif(not _GPU, reason=_GPU_WHY)
def test_staging_oom_names_the_subject_not_the_gpu(tmp_path: Path,
                                                   monkeypatch):
    """The whole subject is pinned at once, so a host-pinned OOM must
    say so — ``cudaHostAlloc``'s own error reads like a VRAM OOM."""
    import cupy as cp
    from arealmshbm.data_io import gifti_bold_gpu as gbg

    pairs, mask = _synth_subject(tmp_path, 64, 64, 4, 2, seed=5)

    def _oom(nbytes):
        raise cp.cuda.runtime.CUDARuntimeError(2)   # cudaErrorMemoryAllocation

    monkeypatch.setattr(gbg, "_acquire_staging", _oom)
    with pytest.raises(RuntimeError, match="host staging"):
        gbg.read_subject_bold_gpu(pairs, medial_mask_d=cp.asarray(mask),
                                  n_lh=64, n_rh=64)
