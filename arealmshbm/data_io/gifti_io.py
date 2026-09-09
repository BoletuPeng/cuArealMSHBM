"""gifti_io.py — direct .func.gii surface BOLD reader (CPU).

Reads DeepPrep / fmriprep GIFTI ``.func.gii`` (one ``<DataArray>`` per
timepoint, gzipped-then-base64'd fp32) directly. The contract is
enforced on **every** darray, so a same-byte-width type switch cannot
slip through: ``NIFTI_TYPE_FLOAT32`` / ``GZipBase64Binary`` /
``LittleEndian`` and a ``Dim0`` equal to the first darray's.

:func:`read_surface_gifti` is the CPU reader;
:mod:`arealmshbm.data_io.gifti_bold_gpu` has the whole-subject
device-stay one.

Two rules: handing :func:`read_surface_gifti` the very pool whose
worker is calling it would deadlock (detected, demoted to serial), and
``allow_wrapped_b64=None`` (default) falls back to ``base64.b64decode``
per refused chunk, so line-wrapped payloads still read. The decode's
own layout is ``(T, N)`` (one darray per row); ``time_major=True``
returns it as is, the default transposes to ``(N, T)``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from arealmshbm.data_io._gifti_kernels import (
    B64_ERR_LEN, B64_TABLE, b64_decode_strict, copy_row_f32, transpose_f32,
)

# Optional Intel ISA-L zlib. Same gating pattern as fetch_data.py.
# isal_zlib.decompress is a drop-in for stdlib zlib.decompress with
# AVX/AVX-512 acceleration on the GIFTI chunks (~165 KB each, ~242
# chunks per file).
try:
    from isal import isal_zlib as _zlib
except ImportError:  # pragma: no cover — isal is in the env
    import zlib as _zlib  # type: ignore[no-redef]

PathLike = Union[str, Path]

# Header attribute scan. The first <DataArray>'s header is validated
# fully via the regexes below so the error message can name the actual
# offending attribute. Every subsequent darray header is then
# validated by cheap substring presence (see ``_scan_gifti_spans``)
# — full regex re-runs only when a needle is missing. This catches
# heterogeneous-DataType GIFTIs (e.g. a darray that switches to
# NIFTI_TYPE_INT32 at the same N*4 bytes/chunk as FLOAT32) that a
# per-chunk byte-length check alone cannot detect.
_RE_DIM0 = re.compile(rb'Dim0="(\d+)"')
_RE_DATATYPE = re.compile(rb'DataType="([^"]+)"')
_RE_ENCODING = re.compile(rb'Encoding="([^"]+)"')
_RE_ENDIAN = re.compile(rb'Endian="([^"]+)"')

# Permitted contract values. ``LittleEndian`` is checked but the
# resulting fp32 buffer is read with ``np.frombuffer(dtype=np.float32)``
# which is native-endian — that matches LittleEndian on every platform
# this codebase targets (x86_64 Windows / Linux). Extending to
# BigEndian would require an explicit ``.byteswap()`` pass.
_EXPECTED_DTYPE = b"NIFTI_TYPE_FLOAT32"
_EXPECTED_ENC = b"GZipBase64Binary"
_EXPECTED_END = b"LittleEndian"

# Tag literals — kept as module constants so the per-file ``bytes.find``
# loop doesn't re-build the small ``bytes`` objects on each call.
_DA_OPEN = b"<DataArray"
_DATA_OPEN = b"<Data>"
_DATA_CLOSE = b"</Data>"
_LEN_DATA_CLOSE = len(_DATA_CLOSE)


def _decode_one_chunk(b64gz: bytes, n_expected: int) -> np.ndarray:
    """base64 → isal-zlib decompress → (N,) fp32 view of the result.

    Both decode steps release the GIL on inputs of this size (~165 KB),
    so concurrent calls from a thread pool actually run in parallel.
    """
    raw = base64.b64decode(b64gz)
    buf = _zlib.decompress(raw)
    arr = np.frombuffer(buf, dtype=np.float32)
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"GIFTI darray length {arr.shape[0]} != expected {n_expected}; "
            f"Dim0 disagrees with decoded byte count"
        )
    return arr


def _scan_gifti_spans(data: bytes, path):
    """Byte-scan a ``.func.gii`` buffer → span tables + contract check.

    Returns ``(N, T, hdr_starts, hdr_ends, data_starts, data_ends)``,
    four ``int`` lists of length ``T``, so a caller decoding from a
    ``np.frombuffer`` view of ``data`` never materialises the chunks.

    One forward pass, exploiting the fact that **a GZipBase64Binary
    payload cannot contain ``<``** so the terminator is a single-byte
    ``memchr``. Both fallbacks below exist to keep the accept/reject
    verdict identical to a plain ``find(b"</Data>")`` scan.
    """
    # Single interleaved pass. ``pos`` walks strictly forward.
    hdr_starts: List[int] = []
    hdr_ends: List[int] = []
    data_starts: List[int] = []
    data_ends: List[int] = []
    n_data = len(data)
    pos = 0
    while True:
        s = data.find(_DA_OPEN, pos)
        if s < 0:
            break
        e = data.find(b">", s)
        if e < 0:
            raise ValueError(
                f"GIFTI {path}: unterminated <DataArray> header at byte {s}"
            )
        hdr_starts.append(s)
        hdr_ends.append(e + 1)
        i = len(hdr_starts) - 1

        s_data = data.find(_DATA_OPEN, e + 1)
        # Old rule: the <Data> for darray i must precede darray i+1.
        # Equivalent bounded check without knowing darray i+1 yet.
        if s_data < 0 or data.find(_DA_OPEN, e + 1, s_data) >= 0:
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: missing <Data> element"
            )
        s_data += len(_DATA_OPEN)
        # memchr for the payload terminator (payload is pure base64).
        e_data = data.find(b"<", s_data)
        # Slice-compare, not ``data.startswith(needle, off)``: the
        # identical test, but it also works on an ``mmap`` object.
        if e_data < 0 or data[e_data:e_data + _LEN_DATA_CLOSE] != _DATA_CLOSE:
            # Payload contained a stray '<' (or EOF hit): fall back to
            # the bounded search so the verdict is unchanged.
            boundary = data.find(_DA_OPEN, s_data)
            if boundary < 0:
                boundary = n_data
            e_data = data.find(_DATA_CLOSE, s_data, boundary)
            if e_data < 0:
                raise ValueError(
                    f"GIFTI {path}: DataArray {i}: unterminated <Data> "
                    f"at byte {s_data}"
                )
        data_starts.append(s_data)
        data_ends.append(e_data)
        pos = e_data + len(_DATA_CLOSE)

    if not hdr_starts:
        raise ValueError(f"GIFTI {path}: no <DataArray> element found")

    # Validate the first DataArray header fully — full regex so the
    # error message names the actual offending attribute.
    first = data[hdr_starts[0]:hdr_ends[0]]
    dt = _RE_DATATYPE.search(first)
    if not dt or dt.group(1) != _EXPECTED_DTYPE:
        raise ValueError(
            f"GIFTI {path}: DataType="
            f"{dt.group(1).decode() if dt else 'MISSING'!r}; "
            f"this reader requires {_EXPECTED_DTYPE.decode()!r}"
        )
    enc = _RE_ENCODING.search(first)
    if not enc or enc.group(1) != _EXPECTED_ENC:
        raise ValueError(
            f"GIFTI {path}: Encoding="
            f"{enc.group(1).decode() if enc else 'MISSING'!r}; "
            f"this reader requires {_EXPECTED_ENC.decode()!r}"
        )
    end_attr = _RE_ENDIAN.search(first)
    if not end_attr or end_attr.group(1) != _EXPECTED_END:
        raise ValueError(
            f"GIFTI {path}: Endian="
            f"{end_attr.group(1).decode() if end_attr else 'MISSING'!r}; "
            f"this reader requires {_EXPECTED_END.decode()!r}"
        )
    m_dim0 = _RE_DIM0.search(first)
    if not m_dim0:
        raise ValueError(f"GIFTI {path}: missing Dim0 attribute")
    N = int(m_dim0.group(1))

    # Pre-build needles for the per-darray substring check. Built
    # once so the per-darray loop is two bytes.find calls per attr.
    _DTYPE_NEEDLE = b'DataType="' + _EXPECTED_DTYPE + b'"'
    _ENC_NEEDLE = b'Encoding="' + _EXPECTED_ENC + b'"'
    _END_NEEDLE = b'Endian="' + _EXPECTED_END + b'"'
    _DIM0_NEEDLE = b'Dim0="' + str(N).encode() + b'"'

    # Validate every subsequent DataArray header inherits the
    # contract via cheap substring presence; fall back to the full
    # regex only when a needle is missing so the error names the
    # actual offending attribute.
    for i in range(1, len(hdr_starts)):
        h = data[hdr_starts[i]:hdr_ends[i]]
        if _DTYPE_NEEDLE not in h:
            actual = _RE_DATATYPE.search(h)
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: heterogeneous DataType="
                f"{actual.group(1).decode() if actual else 'MISSING'!r}; "
                f"this reader requires every darray's DataType to be "
                f"{_EXPECTED_DTYPE.decode()!r}"
            )
        if _ENC_NEEDLE not in h:
            actual = _RE_ENCODING.search(h)
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: heterogeneous Encoding="
                f"{actual.group(1).decode() if actual else 'MISSING'!r}; "
                f"this reader requires every darray's Encoding to be "
                f"{_EXPECTED_ENC.decode()!r}"
            )
        if _END_NEEDLE not in h:
            actual = _RE_ENDIAN.search(h)
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: heterogeneous Endian="
                f"{actual.group(1).decode() if actual else 'MISSING'!r}; "
                f"this reader requires every darray's Endian to be "
                f"{_EXPECTED_END.decode()!r}"
            )
        if _DIM0_NEEDLE not in h:
            actual = _RE_DIM0.search(h)
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: heterogeneous Dim0="
                f"{actual.group(1).decode() if actual else 'MISSING'!r}; "
                f"this reader requires every darray's Dim0 to match "
                f"the first darray ({N})"
            )

    return (N, len(data_starts), hdr_starts, hdr_ends,
            data_starts, data_ends)


# ─────────────────────────────────────────────────────────────────────
# CPU reader — GIL-free decode (numba base64 + isal + numba transpose)
# ─────────────────────────────────────────────────────────────────────
def _decode_chunk_range(
    buf, starts, lens, stage, N, path,
    t0, t1, allow_wrapped_b64,
):
    """Decode timepoints ``[t0, t1)`` into rows of the ``(T, N)`` stage.

    One scratch buffer per call, so concurrent tasks share no state.
    """
    if t1 <= t0:
        return
    scratch = np.empty(int(lens[t0:t1].max()) // 4 * 3 + 3, dtype=np.uint8)
    expected_bytes = N * 4
    for t in range(t0, t1):
        if allow_wrapped_b64 is True:
            # Permissive route: binascii silently strips whitespace.
            raw = _zlib.decompress(
                base64.b64decode(bytes(buf[starts[t]:starts[t] + lens[t]])),
                15, expected_bytes)
        else:
            n_raw = b64_decode_strict(
                B64_TABLE, buf, int(starts[t]), int(lens[t]), scratch, 0)
            if n_raw < 0:
                if allow_wrapped_b64 is None:
                    # Auto (the default): a non-canonical chunk is
                    # decoded the way the pre-kernel reader did, which
                    # keeps the accepted-input set identical.
                    raw = _zlib.decompress(
                        base64.b64decode(
                            bytes(buf[starts[t]:starts[t] + lens[t]])),
                        15, expected_bytes)
                    arr = np.frombuffer(raw, dtype=np.float32)
                    if arr.shape[0] != N:
                        raise ValueError(
                            f"GIFTI darray length {arr.shape[0]} != expected "
                            f"{N}; Dim0 disagrees with decoded byte count"
                        )
                    copy_row_f32(arr, stage, t)
                    continue
                if n_raw == B64_ERR_LEN:
                    raise ValueError(
                        f"GIFTI {path}: DataArray {t} <Data> payload length "
                        f"{int(lens[t])} is not a multiple of 4 (invalid "
                        f"base64)"
                    )
                raise ValueError(
                    f"GIFTI {path}: DataArray {t} <Data> payload contains a "
                    f"byte outside the base64 alphabet (line-wrapped or "
                    f"otherwise non-canonical b64). ``allow_wrapped_b64`` "
                    f"was set to False, which opts into the strict contract "
                    f"the GPU reader enforces; the default (``None``) falls "
                    f"back to ``base64.b64decode`` for such a chunk."
                )
            raw = _zlib.decompress(memoryview(scratch)[:n_raw], 15,
                                   expected_bytes)
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.shape[0] != N:
            raise ValueError(
                f"GIFTI darray length {arr.shape[0]} != expected {N}; "
                f"Dim0 disagrees with decoded byte count"
            )
        copy_row_f32(arr, stage, t)


#: How long a submitted chunk task may sit unscheduled before the
#: calling thread assumes the pool is starved and runs it itself.
_POOL_STARVE_TIMEOUT = 1.0


def _current_thread_belongs_to(pool) -> bool:
    """Is the calling thread a worker of ``pool``?

    Submitting into a pool and blocking on the result from one of its
    own workers deadlocks, so :func:`read_surface_gifti` checks and
    takes the serial path. Two probes, because ``_threads`` is
    populated only *after* ``Thread.start()``: membership in
    ``pool._threads``, and the weakref to the owning executor in the
    worker's ``_args``. Best-effort reads of CPython internals; the
    authoritative guard is :func:`_await_or_run`.
    """
    cur = threading.current_thread()
    threads = getattr(pool, "_threads", None)
    if threads:
        try:
            if cur in threads:
                return True
        except TypeError:      # pragma: no cover — exotic executor
            pass
    args = getattr(cur, "_args", None)
    if args:
        ref = args[0]
        try:
            if callable(ref) and ref() is pool:
                return True
        except Exception:      # pragma: no cover — not a weakref
            pass
    return False


def _await_or_run(futs, tasks):
    """Collect ``futs``, executing any the pool never scheduled.

    A future still *pending* after :data:`_POOL_STARVE_TIMEOUT` is
    cancelled and run on the calling thread, so a starved pool cannot
    hang. ``tasks[i]`` is the ``(fn, args)`` pair behind ``futs[i]``.
    """
    from concurrent.futures import TimeoutError as _FTimeout

    for f, (fn, args) in zip(futs, tasks):
        try:
            f.result(timeout=_POOL_STARVE_TIMEOUT)
        except _FTimeout:
            if f.cancel():
                fn(*args)
            else:
                f.result()


def read_surface_gifti(path: PathLike, *,
                       pool=None,
                       n_workers: Optional[int] = None,
                       allow_wrapped_b64: Optional[bool] = None,
                       time_major: bool = False,
                       ) -> np.ndarray:
    """Read a surface BOLD ``.func.gii`` → ``(N, T) fp32`` host array
    (``(T, N)`` with ``time_major``).

    Parameters
    ----------
    path : str | Path
        A ``.func.gii`` meeting the module docstring's BOLD contract.
    pool : concurrent.futures.Executor, optional
        Caller-owned pool to split this **one** file's chunks across;
        use it only when there is no outer file-level parallelism.
        **Passing the pool whose worker is making this very call would
        deadlock**; that is detected and demoted to the serial path.
    n_workers : int, optional
        Chunk sub-ranges to split the file into. ``None`` or ``1`` is
        the serial path (the default, and what the prefetchers use);
        with ``pool=None`` a private pool is created for the call.
    allow_wrapped_b64 : bool or None, optional
        ``None`` (default) uses the strict numba kernel and falls back
        to ``base64.b64decode`` per refused chunk, so line-wrapped
        payloads still read; ``True`` always uses ``base64.b64decode``;
        ``False`` opts into the strict contract the GPU reader
        enforces, raising on a non-alphabet byte.
    time_major : bool, optional
        Return ``(T, N)`` -- the decode's own layout, one darray per
        row -- and skip the transpose pass. Step 1 consumes BOLD
        time-major; step 0 takes the default ``(N, T)``.

    Returns
    -------
    (N, T) np.ndarray, dtype float32, C-contiguous
        ``N`` = first ``<DataArray>``'s ``Dim0``, ``T`` = darray count;
        ``(T, N)`` when ``time_major``.

    Notes
    -----
    No arithmetic is performed on the payload — base64 and DEFLATE are
    byte-exact — so every route through this function returns the same
    bytes.
    """
    with open(path, "rb") as f:
        data = f.read()
    N, T, _hs, _he, d_starts, d_ends = _scan_gifti_spans(data, path)

    buf = np.frombuffer(data, dtype=np.uint8)
    starts = np.asarray(d_starts, dtype=np.int64)
    lens = np.asarray(d_ends, dtype=np.int64) - starts

    stage = np.empty((T, N), dtype=np.float32)
    out = None if time_major else np.empty((N, T), dtype=np.float32)

    if (pool is None and (n_workers is None or int(n_workers) <= 1)) or (
            pool is not None and _current_thread_belongs_to(pool)):
        _decode_chunk_range(buf, starts, lens, stage, N, path,
                            0, T, allow_wrapped_b64)
        if time_major:
            return stage
        transpose_f32(stage, out, 0, N)
        return out

    from concurrent.futures import ThreadPoolExecutor

    nw = int(n_workers) if n_workers else int(
        getattr(pool, "_max_workers", 4))
    nw = max(1, min(nw, T))
    own_pool = None
    if pool is None:
        own_pool = ThreadPoolExecutor(
            max_workers=nw, thread_name_prefix="gifti-chunk")
        pool = own_pool
    try:
        bounds = [int(round(k * T / nw)) for k in range(nw + 1)]
        tasks = [(_decode_chunk_range,
                  (buf, starts, lens, stage, N, path, bounds[k],
                   bounds[k + 1], allow_wrapped_b64))
                 for k in range(nw)]
        _await_or_run([pool.submit(fn, *a) for fn, a in tasks], tasks)
        if time_major:
            return stage
        nbounds = [int(round(k * N / nw)) for k in range(nw + 1)]
        ttasks = [(transpose_f32, (stage, out, nbounds[k], nbounds[k + 1]))
                  for k in range(nw)]
        _await_or_run([pool.submit(fn, *a) for fn, a in ttasks], ttasks)
    finally:
        if own_pool is not None:
            own_pool.shutdown(wait=True)
    return out
