"""gifti_io.py — direct .func.gii surface BOLD reader (CPU).

Reads DeepPrep / fmriprep GIFTI ``.func.gii`` (one ``<DataArray>`` per
timepoint, gzipped-then-base64'd fp32) directly. The contract is
enforced on **every** darray, so a same-byte-width type switch cannot
slip through: ``NIFTI_TYPE_FLOAT32`` / ``GZipBase64Binary`` /
``LittleEndian`` and a ``Dim0`` equal to the first darray's.

:func:`read_surface_gifti` is the CPU reader;
:mod:`arealmshbm.data_io.gifti_bold_gpu` has the whole-subject
device-stay one.

A chunk the strict base64 kernel refuses falls back to
``base64.b64decode``, so line-wrapped payloads still read. The decode's
own layout is ``(T, N)`` (one darray per row); ``time_major=True``
returns it as is, the default transposes to ``(N, T)``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import List, Union

import numpy as np

from arealmshbm.data_io._gifti_kernels import (
    B64_TABLE, b64_decode_strict, copy_row_f32, transpose_f32,
)

# Intel ISA-L zlib: a drop-in for stdlib zlib.decompress with AVX /
# AVX-512 inflate on the GIFTI chunks (~165 KB each, ~242 per file). A
# hard dependency -- there is no stdlib fallback to a slower inflate.
from isal import isal_zlib as _zlib

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
def _decode_chunk_range(buf, starts, lens, stage, N, path, t0, t1):
    """Decode timepoints ``[t0, t1)`` into rows of the ``(T, N)`` stage.

    One scratch buffer per call, so concurrent calls share no state.
    The strict numba base64 kernel decodes a canonical single-line
    payload in place; a chunk it refuses (line-wrapped, pretty-printed,
    odd length) goes through ``base64.b64decode`` instead, so the
    accepted-input set is ``base64.b64decode``'s.
    """
    if t1 <= t0:
        return
    scratch = np.empty(int(lens[t0:t1].max()) // 4 * 3 + 3, dtype=np.uint8)
    expected_bytes = N * 4
    for t in range(t0, t1):
        n_raw = b64_decode_strict(
            B64_TABLE, buf, int(starts[t]), int(lens[t]), scratch, 0)
        if n_raw < 0:
            raw = _zlib.decompress(
                base64.b64decode(bytes(buf[starts[t]:starts[t] + lens[t]])),
                15, expected_bytes)
        else:
            raw = _zlib.decompress(memoryview(scratch)[:n_raw], 15,
                                   expected_bytes)
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.shape[0] != N:
            raise ValueError(
                f"GIFTI darray length {arr.shape[0]} != expected {N}; "
                f"Dim0 disagrees with decoded byte count"
            )
        copy_row_f32(arr, stage, t)


def read_surface_gifti(path: PathLike, *,
                       time_major: bool = False,
                       ) -> np.ndarray:
    """Read a surface BOLD ``.func.gii`` → ``(N, T) fp32`` host array
    (``(T, N)`` with ``time_major``).

    Parameters
    ----------
    path : str | Path
        A ``.func.gii`` meeting the module docstring's BOLD contract.
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
    byte-exact.
    """
    with open(path, "rb") as f:
        data = f.read()
    N, T, _hs, _he, d_starts, d_ends = _scan_gifti_spans(data, path)

    buf = np.frombuffer(data, dtype=np.uint8)
    starts = np.asarray(d_starts, dtype=np.int64)
    lens = np.asarray(d_ends, dtype=np.int64) - starts

    stage = np.empty((T, N), dtype=np.float32)
    out = None if time_major else np.empty((N, T), dtype=np.float32)

    _decode_chunk_range(buf, starts, lens, stage, N, path, 0, T)
    if time_major:
        return stage
    transpose_f32(stage, out, 0, N)
    return out
