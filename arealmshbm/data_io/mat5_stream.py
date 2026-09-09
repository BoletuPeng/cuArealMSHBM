"""mat5_stream.py — a small streaming reader for MATLAB Level-5 ``.mat`` files.

``scipy.io.loadmat`` is correct but eager: it inflates and materialises
**every** variable in the container. Two step-2 inputs make that
expensive:

* ``group/group.mat`` holds a 22 MB ``lambda`` next to the 2.7 MB ``mtc``
  the pipeline actually wants;
* ``spatial_mask/spatial_mask_<mesh>.mat`` holds two MATLAB-*sparse*
  arrays that ``loadmat`` hands back as scipy sparse, which the dense
  step-2 path then densifies to 98 MB of fp64 only for the sparse
  layout builder to scan the nonzeros back out.

This module walks the container itself and **never inflates an unwanted
variable's body**: the container cursor advances by the raw element size,
and a skipped compressed element is inflated only far enough to read its
name (one source chunk). Three entry points:

    walk_names(path)          → the top-level variable names, in file order
    read_fields(path, names)  → {name: ndarray} for numeric (non-sparse) vars
    read_sparse(path, name)   → scipy ``csc_matrix`` from MATLAB's ir/jc/pr

Every case the walker does not handle raises :class:`_Unsupported`
internally and the public helpers fall back to ``scipy.io``. That is the
only exception they catch: a corrupt zlib payload, a ``MemoryError`` or
a walker bug propagates to the caller rather than being re-read by
``loadmat`` under a different error (or, worse, silently succeeding at
~40x the cost). A short or truncated container is the one grey case —
the walker cannot tell a truncation from a layout it mis-parsed, so it
calls it unsupported and lets ``loadmat``'s own error name the problem.

:mod:`arealmshbm.step3_pipeline.sparse_inputs` reuses the reader and the
element primitives below to stream the step-3 group prior's ``theta``
column-by-column straight into a CSR build, so it never materialises the
matrix ``read_sparse`` returns here.

Dtype contract, and it is NOT ``loadmat``'s default: a dense variable
comes back in its **mx class** dtype, i.e. what
``loadmat(..., mat_dtype=True)`` returns. MATLAB stores a double array
with a narrower payload when that is lossless (an mxDOUBLE_CLASS array
whose ``pr`` element is ``miINT16``), and default ``loadmat``
(``mat_dtype=False``) hands back the *payload* dtype — int16 there,
where this reader widens to float64. Values are equal either way; only
the dtype differs, and only for a narrowed payload. The scipy fallback
route is therefore not dtype-identical to the fast route in that one
case; both are used only where the caller casts anyway. Sparse
variables keep the payload dtype on both routes.

``LAST_PATH`` records which route the last public call took
(``'fast'`` / ``'scipy'``) for tests and timing reports.

Container layout (MAT v5, little-endian only): a 128-byte header, then
a sequence of data elements. A top-level variable is either an
``miMATRIX`` element written in the clear or an ``miCOMPRESSED``
element whose zlib payload inflates to one ``miMATRIX``. An
``miMATRIX`` is itself a sequence of sub-elements: array flags, dims,
the variable name, then the payload (``pr`` — plus ``ir``/``jc``
ahead of it for ``mxSPARSE_CLASS``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional Intel ISA-L zlib. Same gating pattern as gifti_io.py;
# stdlib ``zlib.decompressobj`` is a drop-in for the walker's use.
try:
    from isal import isal_zlib as _zlib
except ImportError:  # pragma: no cover — isal is in the env
    import zlib as _zlib  # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────
# MAT Level-5 constants
# ─────────────────────────────────────────────────────────────────────
_miINT8, _miUINT8 = 1, 2
_miINT16, _miUINT16 = 3, 4
_miINT32, _miUINT32 = 5, 6
_miSINGLE, _miDOUBLE = 7, 9
_miINT64, _miUINT64 = 12, 13
_miMATRIX, _miCOMPRESSED = 14, 15
_miUTF8 = 16

_mxCELL_CLASS = 1
_mxSTRUCT_CLASS = 2
_mxOBJECT_CLASS = 3
_mxCHAR_CLASS = 4
_mxSPARSE_CLASS = 5
_mxDOUBLE_CLASS, _mxSINGLE_CLASS = 6, 7
_mxINT8_CLASS, _mxUINT8_CLASS = 8, 9
_mxINT16_CLASS, _mxUINT16_CLASS = 10, 11
_mxINT32_CLASS, _mxUINT32_CLASS = 12, 13
_mxINT64_CLASS, _mxUINT64_CLASS = 14, 15

# Numeric (non-sparse, non-char) classes this reader will materialise.
_NUMERIC_CLASSES = frozenset({
    _mxDOUBLE_CLASS, _mxSINGLE_CLASS,
    _mxINT8_CLASS, _mxUINT8_CLASS, _mxINT16_CLASS, _mxUINT16_CLASS,
    _mxINT32_CLASS, _mxUINT32_CLASS, _mxINT64_CLASS, _mxUINT64_CLASS,
})

_MI_DTYPE = {
    _miINT8: np.int8, _miUINT8: np.uint8,
    _miINT16: np.int16, _miUINT16: np.uint16,
    _miINT32: np.int32, _miUINT32: np.uint32,
    _miSINGLE: np.float32, _miDOUBLE: np.float64,
    _miINT64: np.int64, _miUINT64: np.uint64,
    _miUTF8: np.uint8,
}

# The dtype each numeric mx class is presented as — the CLASS dtype,
# i.e. ``loadmat(mat_dtype=True)`` semantics, not the default. MATLAB
# may store a double array with a narrower payload (its own lossless
# compression); this reader widens back to the class dtype, while
# default ``loadmat`` would return the narrow payload dtype.
_MX_DTYPE = {
    _mxDOUBLE_CLASS: np.float64, _mxSINGLE_CLASS: np.float32,
    _mxINT8_CLASS: np.int8, _mxUINT8_CLASS: np.uint8,
    _mxINT16_CLASS: np.int16, _mxUINT16_CLASS: np.uint16,
    _mxINT32_CLASS: np.int32, _mxUINT32_CLASS: np.uint32,
    _mxINT64_CLASS: np.int64, _mxUINT64_CLASS: np.uint64,
}

#: Which route the last public call took — ``'fast'`` or ``'scipy'``.
LAST_PATH: str = ""


class _Unsupported(Exception):
    """The fast walker hit something it does not handle → use scipy."""


# ─────────────────────────────────────────────────────────────────────
# Forward-only readers (inflating / in-memory) sharing read/skip
# ─────────────────────────────────────────────────────────────────────
class _InflateReader:
    """Sequential reader over a zlib stream, inflated in small pieces.

    Only forward ``read``/``skip`` are needed to walk a MAT container,
    and streaming means a multi-megabyte payload we are skipping never
    exists as one buffer.
    """

    __slots__ = ("_src", "_pos", "_csz", "_do", "_buf", "_off", "_done")

    # 4 KiB of compressed input inflates to ~100 KiB here, which keeps
    # the freshly written bytes in L2 for the consumer. Measured on the
    # step-3 prior: 4 KiB → 59 ms vs 103 ms at 256 KiB — the payload is
    # memory-bandwidth bound, not inflate-bound.
    def __init__(self, comp, src_chunk: int = 4096):
        self._src = comp
        self._pos = 0
        self._csz = int(src_chunk)
        self._do = _zlib.decompressobj()
        self._buf = b""
        self._off = 0
        self._done = False

    def _next_chunk(self) -> bytes:
        while True:
            if self._done:
                return b""
            if self._pos >= len(self._src):
                self._done = True
                return self._do.flush()
            src = self._src[self._pos:self._pos + self._csz]
            self._pos += len(src)
            out = self._do.decompress(src)
            if out:
                return out

    def read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        avail = len(self._buf) - self._off
        if n <= avail:
            out = self._buf[self._off:self._off + n]
            self._off += n
            return out
        parts: List[bytes] = []
        if avail:
            parts.append(self._buf[self._off:])
        got = avail
        self._buf, self._off = b"", 0
        while got < n:
            c = self._next_chunk()
            if not c:
                raise _Unsupported("truncated MAT stream")
            if got + len(c) > n:
                need = n - got
                parts.append(c[:need])
                self._buf, self._off = c, need
                got = n
            else:
                parts.append(c)
                got += len(c)
        return b"".join(parts) if len(parts) != 1 else parts[0]

    def iter_raw(self, n: int):
        """Yield memoryviews covering the next ``n`` bytes — no copy.

        Each view borrows the inflate output chunk it came from; the
        chunk stays alive as long as the view does.
        """
        while n > 0:
            avail = len(self._buf) - self._off
            if avail == 0:
                c = self._next_chunk()
                if not c:
                    raise _Unsupported("truncated MAT stream")
                self._buf, self._off = c, 0
                avail = len(c)
            take = avail if avail < n else n
            yield memoryview(self._buf)[self._off:self._off + take]
            self._off += take
            n -= take

    def skip(self, n: int) -> None:
        while n > 0:
            avail = len(self._buf) - self._off
            if avail >= n:
                self._off += n
                return
            n -= avail
            self._buf, self._off = b"", 0
            c = self._next_chunk()
            if not c:
                raise _Unsupported("truncated MAT stream")
            self._buf, self._off = c, 0


class _BytesReader:
    """The same forward-only interface over an already-resident buffer.

    Used for uncompressed (``miMATRIX``) top-level elements, which
    ``scipy.io.savemat(do_compression=False)`` writes and MATLAB's
    ``-v6`` produces.
    """

    __slots__ = ("_b", "_off", "_end")

    def __init__(self, buf: bytes, off: int, end: int):
        self._b = buf
        self._off = int(off)
        self._end = int(end)

    def read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        if self._off + n > self._end:
            raise _Unsupported("truncated MAT element")
        out = self._b[self._off:self._off + n]
        self._off += n
        return out

    def skip(self, n: int) -> None:
        if self._off + n > self._end:
            raise _Unsupported("truncated MAT element")
        self._off += n


# ─────────────────────────────────────────────────────────────────────
# Element / tag primitives
# ─────────────────────────────────────────────────────────────────────
def _pad8(n: int) -> int:
    return (n + 7) & ~7


def _parse_tag(hdr: bytes) -> Tuple[int, int, Optional[bytes]]:
    """``(mi_type, n_bytes, small_payload_or_None)`` from an 8-byte tag."""
    if len(hdr) != 8:
        raise _Unsupported("short MAT tag")
    w0, w1 = struct.unpack("<II", hdr)
    if w0 >> 16:                       # small-data-element form
        return (w0 & 0xFFFF), (w0 >> 16), hdr[4:4 + (w0 >> 16)]
    return w0, w1, None


def _read_element(rd) -> Tuple[int, bytes, int]:
    """Read one whole non-``miMATRIX`` element.

    Returns ``(mi_type, payload, bytes_consumed)`` — the consumed count
    includes the tag and the 8-byte alignment padding.
    """
    typ, nb, small = _parse_tag(rd.read(8))
    if small is not None:
        return typ, small, 8
    data = rd.read(nb)
    rd.skip(_pad8(nb) - nb)
    return typ, data, 8 + _pad8(nb)


def _as_array(mi_type: int, payload: bytes) -> np.ndarray:
    dt = _MI_DTYPE.get(mi_type)
    if dt is None:
        raise _Unsupported(f"unsupported mi type {mi_type}")
    return np.frombuffer(payload, dtype=dt)


# ─────────────────────────────────────────────────────────────────────
# miMATRIX walking
# ─────────────────────────────────────────────────────────────────────
class _MatrixHeader:
    __slots__ = ("cls", "complex", "logical", "nzmax", "dims", "name",
                 "consumed")

    def __init__(self, cls, complex_, logical, nzmax, dims, name, consumed):
        self.cls = cls
        self.complex = complex_
        self.logical = logical
        self.nzmax = nzmax
        self.dims = dims
        self.name = name
        self.consumed = consumed


def _read_matrix_header(rd) -> _MatrixHeader:
    """Read array flags + dims + name of an ``miMATRIX`` body."""
    ftyp, fdata, used = _read_element(rd)
    fa = _as_array(ftyp, fdata).astype(np.uint32)
    if fa.size < 2:
        raise _Unsupported("short array-flags element")
    cls = int(fa[0] & 0xFF)
    complex_ = bool(fa[0] & 0x0800)
    logical = bool(fa[0] & 0x0200)
    nzmax = int(fa[1])
    consumed = used

    dtyp, ddata, used = _read_element(rd)
    consumed += used
    dims = _as_array(dtyp, ddata).astype(np.int64)

    ntyp, ndata, used = _read_element(rd)
    consumed += used
    name = bytes(ndata).split(b"\0")[0].decode("ascii", "replace")

    return _MatrixHeader(cls, complex_, logical, nzmax, dims, name, consumed)


def _read_numeric_body(rd, hdr: _MatrixHeader) -> np.ndarray:
    """Payload of a numeric ``miMATRIX`` → ndarray in MATLAB (F) order."""
    if hdr.complex:
        raise _Unsupported(f"{hdr.name}: complex arrays unsupported")
    if hdr.cls not in _NUMERIC_CLASSES:
        raise _Unsupported(f"{hdr.name}: class {hdr.cls} unsupported")
    if hdr.dims.size != 2:
        raise _Unsupported(f"{hdr.name}: {hdr.dims.size}-D array unsupported")
    ptyp, payload, _ = _read_element(rd)
    dt = _MI_DTYPE.get(ptyp)
    if dt is None:
        raise _Unsupported(f"{hdr.name}: payload type {ptyp} unsupported")
    R, C = int(hdr.dims[0]), int(hdr.dims[1])
    arr = np.frombuffer(payload, dtype=dt)
    if arr.size != R * C:
        raise _Unsupported(f"{hdr.name}: payload size mismatch")
    out = arr.reshape(R, C, order="F")
    want = _MX_DTYPE[hdr.cls]
    if out.dtype != want:
        # MATLAB stores a double array with a narrower payload when it
        # is lossless; widen to the class dtype (``mat_dtype=True``
        # semantics — see the module docstring's dtype contract).
        out = out.astype(want)
    return np.asfortranarray(out)


def _read_sparse_body(rd, hdr: _MatrixHeader):
    """Payload of an ``mxSPARSE_CLASS`` ``miMATRIX`` → ``csc_matrix``."""
    import scipy.sparse as sp

    if hdr.complex:
        raise _Unsupported(f"{hdr.name}: complex sparse unsupported")
    if hdr.dims.size != 2:
        raise _Unsupported(f"{hdr.name}: {hdr.dims.size}-D sparse unsupported")
    R, C = int(hdr.dims[0]), int(hdr.dims[1])

    ityp, idata, _ = _read_element(rd)
    ir = _as_array(ityp, idata)
    jtyp, jdata, _ = _read_element(rd)
    jc = _as_array(jtyp, jdata)
    ptyp, pdata, _ = _read_element(rd)
    dt = _MI_DTYPE.get(ptyp)
    if dt is None:
        raise _Unsupported(f"{hdr.name}: sparse payload type {ptyp}")
    pr = np.frombuffer(pdata, dtype=dt)

    if jc.size != C + 1:
        raise _Unsupported(f"{hdr.name}: jc has {jc.size} entries, want {C + 1}")
    nnz = int(jc[-1])
    if ir.size < nnz or pr.size < nnz:
        raise _Unsupported(f"{hdr.name}: ir/pr shorter than jc[-1]={nnz}")
    # ``nzmax`` may exceed nnz (MATLAB preallocates); only the first
    # ``jc[-1]`` entries are live.
    indices = np.ascontiguousarray(ir[:nnz], dtype=np.int32)
    indptr = np.ascontiguousarray(jc, dtype=np.int32)
    # Keep the payload's own dtype, which is what ``scipy.io.loadmat``
    # returns for a sparse variable: it does NOT widen to the mx class
    # the way the dense reader does. Measured against savemat fixtures,
    # compressed and uncompressed: fp64 -> float64, single -> float32,
    # logical -> uint8 (the miUINT8 payload, NOT bool), int32 -> int32.
    # Forcing float64 / bool here would silently widen (and copy)
    # relative to scipy, breaking the module docstring's "always what
    # scipy would have returned" contract.
    data = np.ascontiguousarray(pr[:nnz])
    return sp.csc_matrix((data, indices, indptr), shape=(R, C))


# ─────────────────────────────────────────────────────────────────────
# Container walk
# ─────────────────────────────────────────────────────────────────────
def _check_header(raw: bytes) -> None:
    if len(raw) < 136:
        raise _Unsupported("file too small to be a MAT v5 container")
    if raw[124:126] != b"\x00\x01" or raw[126:128] != b"IM":
        # version word + endian indicator; 'IM' == little-endian.
        raise _Unsupported("not a little-endian MAT v5 container")


def _iter_top_elements(raw: bytes):
    """Yield one forward-only reader per top-level variable.

    Compressed elements are handed back as an ``_InflateReader`` over
    the zlib payload; uncompressed ones as a ``_BytesReader`` over the
    element body. Advancing the container cursor never inflates.
    """
    off = 128
    n = len(raw)
    mv = memoryview(raw)
    while off + 8 <= n:
        typ, nb = struct.unpack("<II", raw[off:off + 8])
        if typ >> 16:
            raise _Unsupported("small-data element at container level")
        end = off + 8 + nb
        if end > n:
            raise _Unsupported("top-level element runs past EOF")
        if typ == _miCOMPRESSED:
            yield _InflateReader(mv[off + 8:end])
            # miCOMPRESSED elements are NOT padded to 8 bytes at the
            # container level (scipy's own reader advances by the raw
            # byte count); miMATRIX ones already are, but pad anyway.
            off = end
        elif typ == _miMATRIX:
            yield _BytesReader(raw, off + 8, end)
            off = off + 8 + _pad8(nb)
        else:
            raise _Unsupported(f"top-level element type {typ} unsupported")


def _enter_matrix(rd):
    """Consume the ``miMATRIX`` tag of a top-level element body.

    Uncompressed elements have already had their tag consumed by the
    container walk (the reader starts at the body); compressed ones
    still carry it inside the zlib stream.
    """
    if isinstance(rd, _BytesReader):
        return rd
    typ, _nb, small = _parse_tag(rd.read(8))
    if small is not None or typ != _miMATRIX:
        raise _Unsupported("top-level element is not an miMATRIX")
    return rd


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
def walk_names(path: str | Path) -> List[str]:
    """Top-level variable names, in file order.

    Falls back to ``scipy.io.whosmat`` when the container is anything
    this walker does not understand (v7.3 / HDF5 included).
    """
    global LAST_PATH
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MAT file not found: {p}")
    try:
        raw = p.read_bytes()
        _check_header(raw)
        names: List[str] = []
        for rd in _iter_top_elements(raw):
            hdr = _read_matrix_header(_enter_matrix(rd))
            names.append(hdr.name)
        LAST_PATH = "fast"
        return names
    except _Unsupported:
        LAST_PATH = "scipy"
    from scipy.io import whosmat
    return [str(nm) for nm, _shape, _dt in whosmat(str(p))]


def read_fields(path: str | Path,
                names: Iterable[str]) -> Dict[str, np.ndarray]:
    """Read the named top-level **numeric** variables.

    An unwanted variable's body is never inflated — a 22 MB ``lambda``
    next to a 2.7 MB ``mtc`` costs the inflate of its header chunk (~4
    KiB of source) and a cursor bump. Values equal
    ``scipy.io.loadmat(path)[name]``; the dtype is the mx CLASS dtype
    (``mat_dtype=True`` semantics), which differs from default
    ``loadmat`` only when MATLAB narrowed the payload — see the module
    docstring's dtype contract.

    Raises ``KeyError`` if a requested name is absent from the file.
    """
    global LAST_PATH
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MAT file not found: {p}")
    want = {str(n) for n in names}
    if not want:
        LAST_PATH = "fast"
        return {}
    out: Dict[str, np.ndarray] = {}
    try:
        raw = p.read_bytes()
        _check_header(raw)
        for rd in _iter_top_elements(raw):
            if not (want - set(out)):
                break
            body = _enter_matrix(rd)
            hdr = _read_matrix_header(body)
            if hdr.name not in want or hdr.name in out:
                continue                    # the container walk skips the bytes
            out[hdr.name] = _read_numeric_body(body, hdr)
        missing = want - set(out)
        if missing:
            raise _Unsupported(f"variables not found by the fast walk: "
                               f"{sorted(missing)}")
        LAST_PATH = "fast"
        return out
    except _Unsupported:
        LAST_PATH = "scipy"

    from scipy.io import loadmat
    m = loadmat(str(p), squeeze_me=False)
    got: Dict[str, np.ndarray] = {}
    for nm in sorted(want):
        if nm not in m:
            raise KeyError(f"{p}: variable {nm!r} not in the file")
        got[nm] = np.asarray(m[nm])
    return got


def read_sparse(path: str | Path, name: str):
    """Read one top-level variable as a scipy ``csc_matrix``.

    MATLAB sparse (``mxSPARSE_CLASS``) is parsed straight out of its
    ``ir`` / ``jc`` / ``pr`` triplet — no dense intermediate, no nonzero
    scan. A dense-class variable is accepted too and returned as the csc
    of its nonzeros, so hand-made mask files still work.
    """
    global LAST_PATH
    import scipy.sparse as sp

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MAT file not found: {p}")
    nm_want = str(name)
    try:
        raw = p.read_bytes()
        _check_header(raw)
        for rd in _iter_top_elements(raw):
            body = _enter_matrix(rd)
            hdr = _read_matrix_header(body)
            if hdr.name != nm_want:
                continue
            if hdr.cls == _mxSPARSE_CLASS:
                m = _read_sparse_body(body, hdr)
            else:
                m = sp.csc_matrix(_read_numeric_body(body, hdr))
            LAST_PATH = "fast"
            return m
        raise _Unsupported(f"variable {nm_want!r} not found by the fast walk")
    except _Unsupported:
        LAST_PATH = "scipy"

    from scipy.io import loadmat
    m = loadmat(str(p), squeeze_me=False)
    if nm_want not in m:
        raise KeyError(f"{p}: variable {nm_want!r} not in the file")
    v = m[nm_want]
    return sp.csc_matrix(v)
