"""gifti_io.py — direct .func.gii surface BOLD reader (CPU + GPU).

DeepPrep / fmriprep surface fMRI outputs are GIFTI ``.func.gii``: an
XML envelope with one ``<DataArray>`` per timepoint, each carrying a
gzipped-then-base64'd binary fp32 vertex vector. The pipeline now
reads that source directly — no offline ``.nii.gz`` conversion needed.

Three readers share one parser:

  * :func:`read_surface_gifti` — CPU path. bytescan + base64 +
    isal_zlib + per-column write into a host ``(N, T) fp32`` buffer.
    Used by the step0/step1 BOLD prefetchers' ``backend='cpu'``
    worker pool.
  * :func:`read_surface_gifti_gpu` — single-file GPU path. CPU
    bytescan + base64 pre-pass on host, then nvCOMP batched Deflate
    decompression on device. Used by :mod:`step0`'s GPU prefetcher;
    not used by step1, which has the faster batched reader below.
  * :func:`read_surface_giftis_gpu_full_pipeline` — production GPU
    path used by step1's ``backend='gpu'`` prefetcher. H2Ds each
    file's 45 MB buffer once, then runs ``<DataArray>`` / ``<Data>``
    bytescan AND base64 decode as on-device CuPy RawKernels, and
    issues one batched nvCOMP Deflate decode across all files'
    chunks. Per-darray header validation happens host-side against
    D2H'd header spans (small, sub-ms). End-to-end on the YS
    reference: ~25 ms / file at batch=8 (= ~50 ms / sess), vs
    ~135 ms / file for ``read_surface_gifti_gpu``.

Why bypass nibabel
------------------
``nib.load(path).darrays`` works correctly but pays ~3× the wall of
the NIFTI fast path: stdlib expat XML parse + stdlib ``zlib`` decode
per-darray, all serial under the GIL. The 242 per-file darrays mean
242 small GIL-held Python callbacks, which makes outer-pool
parallelism (the step1 prefetcher's 8 workers) scale poorly.

The CPU reader takes a flatter route:

1. ``open(path, 'rb').read()`` — one big buffer.
2. Byte-scan ``<DataArray ...>`` header attrs once with ``re`` (assert
   the GIFTI shape contract: FLOAT32 / GZipBase64Binary / LittleEndian).
3. Byte-scan all ``<Data>...</Data>`` spans into a list of ``bytes``
   slices — pure C-level ``bytes.find`` over the buffer, no XML tree.
4. ``base64.b64decode`` + ``isal_zlib.decompress`` per chunk into the
   pre-allocated ``(N, T) fp32`` output. Both decode steps release the
   GIL, so outer-pool parallelism actually works.

On the YS reference (sub-001, fsa6, T=242) the CPU reader is ~165 ms
per file warm. Benchmark harness scripts that produced this number
were retired with the NIFTI strip; the figure is captured in the
PR #54 description and is reproducible by timing
``read_surface_gifti`` on any fsa6 BOLD file.

The GPU reader replaces step (4) with a batched nvCOMP Deflate
decode after stripping each chunk's 2-byte zlib header + 4-byte
adler32 trailer (raw deflate is what nvCOMP wants). Per-file warm
~135 ms (device-stay) on a 5090 Laptop; batched 8-files-at-once
drops the GPU decode itself to ~17 ms/file_eff, so a GPU-aware
prefetcher can run faster than NIFTI's CPU fast path while staying
on device. (Same retirement caveat — numbers captured in PR #54;
re-derive by timing ``read_surface_gifti_gpu`` on a batch of fsa6
files.)

Contract
--------
Both readers are strict on the GIFTI shape they accept:

  * Top-level ``<DataArray>`` per timepoint, ``Dim0=<N>`` matching
    the per-hemi vertex count.
  * ``DataType="NIFTI_TYPE_FLOAT32"``, ``Encoding="GZipBase64Binary"``,
    ``Endian="LittleEndian"``.

A GIFTI that violates any of those is refused with an explicit
``ValueError`` naming the offending attribute. Extending to other
encodings is straightforward (one branch per encoding in the chunk
decode), but the BOLD contract has been FLOAT32 / B64GZ / LE
end-to-end so we don't speculatively support what hasn't been
exercised.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Forward-only — cupy is imported lazily inside the GPU reader so
    # CPU-only environments can import this module. The annotation
    # exists purely so a type-checker can verify GPU callers.
    import cupy as _cupy_type_check  # noqa: F401
    CupyArray = "_cupy_type_check.ndarray"
else:
    CupyArray = "object"  # runtime placeholder; never inspected.

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
# validated by cheap substring presence (see ``_parse_gifti_chunks``)
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


def _parse_gifti_chunks(path: PathLike) -> Tuple[int, int, List[bytes]]:
    """Open + bytescan a ``.func.gii``; return ``(N, T, chunks)``.

    Shared by the CPU (:func:`read_surface_gifti`) and GPU
    (:func:`read_surface_gifti_gpu`) readers — only the decode-route
    after parsing differs. ``chunks[t]`` is the raw base64'd zlib
    bytes from the ``t``-th ``<DataArray>``'s ``<Data>`` child.

    Per-DataArray contract validation
    ---------------------------------
    The first ``<DataArray>``'s header is validated fully
    (DataType / Encoding / Endian / Dim0) via regex so the error
    message can name the offending attribute. Every subsequent
    header is validated via cheap substring presence of the same
    four contract values — a heterogeneous-attribute GIFTI is
    refused at parse time with an error that names the offending
    darray index. This guards against same-byte-width type
    mismatches (e.g. ``NIFTI_TYPE_INT32`` at the same ``N*4`` bytes
    per chunk as ``NIFTI_TYPE_FLOAT32``) that the per-chunk length
    check in :func:`_decode_one_chunk` cannot detect.

    ``<Data>`` extraction is bounded per-darray
    -------------------------------------------
    The ``<Data>`` scan for darray ``i`` is bounded by the start of
    darray ``i + 1`` (or EOF). This prevents a stray ``<Data>``
    substring inside a *neighbouring* darray's ``<MetaData>`` /
    ``<Value>`` block from being mistaken for the current chunk. A
    ``<MetaData>`` block *inside* the current darray that contains
    the literal substring ``<Data>`` would still mis-parse — we
    have not observed this in production GIFTI sources
    (DeepPrep / fmriprep have single, well-formed writer paths) and
    don't speculatively defend against it.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Pass 1: locate every <DataArray ...> header span via two
    # C-level bytes.find calls per darray. No XML tree, no per-
    # element expat callback.
    headers: List[Tuple[int, int]] = []  # (open_start, header_end_excl)
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
        headers.append((s, e + 1))
        pos = e + 1

    if not headers:
        raise ValueError(f"GIFTI {path}: no <DataArray> element found")

    # Validate the first DataArray header fully — full regex so the
    # error message names the actual offending attribute.
    first = data[headers[0][0]:headers[0][1]]
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
    for i in range(1, len(headers)):
        s, e = headers[i]
        h = data[s:e]
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

    # Pass 2: extract exactly one <Data>...</Data> per DataArray,
    # bounded by the start of the next DataArray (or EOF).
    chunks: List[bytes] = []
    for i, (s, e) in enumerate(headers):
        boundary = headers[i + 1][0] if i + 1 < len(headers) else len(data)
        s_data = data.find(_DATA_OPEN, e, boundary)
        if s_data < 0:
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: missing <Data> element"
            )
        s_data += len(_DATA_OPEN)
        e_data = data.find(_DATA_CLOSE, s_data, boundary)
        if e_data < 0:
            raise ValueError(
                f"GIFTI {path}: DataArray {i}: unterminated <Data> "
                f"at byte {s_data}"
            )
        chunks.append(data[s_data:e_data])

    return N, len(chunks), chunks


def read_surface_gifti(path: PathLike) -> np.ndarray:
    """Read a surface BOLD ``.func.gii`` → ``(N, T) fp32`` host array.

    Parameters
    ----------
    path : str | Path
        Path to a ``.func.gii`` with the BOLD contract listed in the
        module docstring (FLOAT32 / GZipBase64Binary / LittleEndian).

    Returns
    -------
    (N, T) np.ndarray, dtype float32, C-contiguous
        ``N`` = ``Dim0`` of the first ``<DataArray>`` (per-hemi vertex
        count for surface BOLD); ``T`` = number of ``<DataArray>``
        elements in the file.

    Notes
    -----
    Strict on shape: a GIFTI with non-FLOAT32 datatype, non-GZip-Base64
    encoding, or non-little-endian byte order is refused with an
    explicit ``ValueError``. The pipeline's whole step0/1/3 surface
    BOLD contract is FLOAT32-LE — and we only validate what we've
    actually exercised end-to-end.

    Per-file decode is serial — the step0/step1 BOLD prefetchers
    drive an outer 8-worker pool, so a nested per-darray pool would
    oversubscribe.
    """
    N, T, chunks = _parse_gifti_chunks(path)
    out = np.empty((N, T), dtype=np.float32)
    for t, chunk in enumerate(chunks):
        out[:, t] = _decode_one_chunk(chunk, N)
    return out


# ─────────────────────────────────────────────────────────────────────
# GPU reader — nvCOMP batched Deflate decompression
# ─────────────────────────────────────────────────────────────────────
# Module-level lazy cache for the nvCOMP codec. The first
# ``read_surface_gifti_gpu`` call pays codec init (~10 ms); subsequent
# calls reuse it. We don't import cupy / nvcomp at module load so a
# CPU-only environment can ``import arealmshbm.data_io.gifti_io``
# without a GPU stack.
#
# The cache is guarded by a ``threading.Lock`` so two concurrent
# first-callers (e.g. a worker pool that calls the GPU reader before
# the codec has been built) cannot double-init the codec. Steady-state
# reads pay only one dict lookup — the lock is taken only on the cold
# path (double-checked locking).
#
# Device-id binding: the cached codec is NOT tagged with a CUDA device
# id. nvCOMP's ``Codec`` object is device-agnostic at construction; the
# actual decode runs on whichever CUDA device is current at
# ``decode(srcs)`` call-time, with the input buffers' device. So a
# multi-GPU process that switches ``cp.cuda.Device(...)`` between
# reads can safely reuse the same codec — the cache key would only
# matter if the codec held device-bound state, which it doesn't.
_GPU_CODEC_CACHE: dict = {}
_GPU_CODEC_LOCK = threading.Lock()


def _get_gpu_codec():
    """Lazily import nvCOMP + cache the Deflate codec instance.

    nvCOMP's Deflate decoder under ``BitstreamKind.RAW`` consumes raw
    deflate streams (RFC 1951). GIFTI's ``<Data>`` payload is a stock
    ``zlib.compress`` output — 2-byte zlib header + raw deflate +
    4-byte adler32 trailer — so the caller strips ``[2:-4]`` before
    handing each chunk to the codec.

    Thread-safe: a double-checked-lock keeps the hot path lock-free
    (one dict lookup) while serialising the one-time codec init.
    """
    if "codec" in _GPU_CODEC_CACHE:
        return _GPU_CODEC_CACHE["codec"], _GPU_CODEC_CACHE["nvcomp"]
    with _GPU_CODEC_LOCK:
        if "codec" not in _GPU_CODEC_CACHE:
            try:
                import nvidia.nvcomp as _nvcomp
            except ImportError as e:
                raise ImportError(
                    "read_surface_gifti_gpu requires nvidia-nvcomp-cu12. "
                    "Install: pip install nvidia-nvcomp-cu12"
                ) from e
            _GPU_CODEC_CACHE["nvcomp"] = _nvcomp
            _GPU_CODEC_CACHE["codec"] = _nvcomp.Codec(
                algorithm="Deflate",
                bitstream_kind=_nvcomp.BitstreamKind.RAW,
            )
        return _GPU_CODEC_CACHE["codec"], _GPU_CODEC_CACHE["nvcomp"]


def read_surface_gifti_gpu(path: PathLike,
                           *,
                           to_host: bool = False,
                           out: "Optional[CupyArray]" = None):
    """Read a ``.func.gii`` with nvCOMP batched Deflate on GPU.

    Pipeline (one file):

      1. CPU: bytescan the file via :func:`_parse_gifti_chunks` →
         ``(N, T, chunks)`` where ``chunks[t]`` is base64'd zlib bytes.
      2. CPU: per-chunk ``base64.b64decode + bytes[2:-4]`` strip →
         raw deflate. ~42 ms per file warm on the YS reference.
      3. GPU: nvCOMP batched Deflate decode of all ``T`` chunks in
         one ``Codec.decode(srcs)`` call. Returns a list of ``(N*4,)
         uint8`` device arrays. ~32 ms per file at batch=1, ~17 ms
         per file at batch=8 (multi-file co-decode).
      4. GPU: reinterpret each chunk as fp32 and column-write into a
         pre-allocated ``(N, T) cupy.float32`` array.

    Parameters
    ----------
    path : str | Path
        Path to a ``.func.gii`` with the contract documented at the
        module top.
    to_host : bool, optional
        If True, copy the device output back to a host ``(N, T)
        np.ndarray fp32`` before returning. Default False — keep the
        output on device for direct hand-off to GPU compute. The H2D
        round-trip costs ~10 ms on a (40962, 242) fp32 buffer.
    out : cupy.ndarray, optional
        Pre-allocated ``(N, T) cupy.float32`` destination. When set,
        the reader writes into it directly (saves one allocation per
        file). Caller must ensure shape and dtype match.

        ``out`` and ``to_host`` compose, they are not alternatives:
        when both are passed, the caller's ``out`` buffer is populated
        in-place AND a host copy is returned. That's useful for callers
        that want to keep a persistent device-side ring buffer while
        still handing a fresh host array back to a CPU consumer.

    Returns
    -------
    cupy.ndarray, dtype float32, on device
        ``(N, T)`` BOLD. Bit-equal to :func:`read_surface_gifti` —
        the deflate decompress is byte-deterministic regardless of
        whether stock zlib (CPU) or nvCOMP (GPU) runs it.
    -or-
    np.ndarray, dtype float32, on host
        Same buffer materialized on host, if ``to_host=True``.

    Raises
    ------
    ImportError
        If ``nvidia-nvcomp-cu12`` or ``cupy`` is not installed. CPU-only
        environments should call :func:`read_surface_gifti` instead.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            "read_surface_gifti_gpu requires cupy. "
            "Install: pip install cupy-cuda12x"
        ) from e

    codec, nvcomp = _get_gpu_codec()

    N, T, chunks = _parse_gifti_chunks(path)

    # base64 + strip zlib wrapper (2-byte CMF/FLG header, 4-byte
    # adler32 trailer) → list of raw deflate bytes that nvCOMP can
    # ingest. ``[2:-4]`` is a 6-byte trim per chunk; no actual byte
    # copy on slicing a bytes object.
    #
    # FDICT check on the first chunk: zlib's FLG byte (byte 1) carries
    # a "preset dictionary" flag at bit 5 (mask 0x20). If set, a 4-byte
    # adler32 of the dictionary follows FLG before the deflate stream
    # — the ``[2:-4]`` strip would feed nvCOMP the wrong bytes and
    # silently corrupt the decoded payload. Stock ``zlib.compress``
    # (DeepPrep / fmriprep's GIFTI writer path) NEVER sets this flag,
    # so the check is cheap insurance: one byte read on the first
    # chunk only. If a future writer enables preset dictionaries,
    # the reader fails loudly instead of producing garbage.
    decoded_chunks = [base64.b64decode(c) for c in chunks]
    if decoded_chunks and (decoded_chunks[0][1] & 0x20):
        raise ValueError(
            f"GIFTI {path}: zlib FLG byte has FDICT bit set (preset "
            f"dictionary). This reader's [2:-4] strip assumes a stock "
            f"zlib.compress stream (no preset dict). Re-encode the "
            f"GIFTI with stock zlib or implement the FDICT skip in "
            f"read_surface_gifti_gpu."
        )
    raw_deflates = [d[2:-4] for d in decoded_chunks]

    # Wrap each as an nvCOMP Array (host uint8 buffer). nvCOMP H2Ds
    # the compressed payload internally during ``decode``.
    srcs = [nvcomp.as_array(np.frombuffer(rd, dtype=np.uint8))
            for rd in raw_deflates]

    # Batched Deflate decompress on device. Output is a list of T
    # ``(N*4,) uint8`` device arrays — one per timepoint.
    outs = codec.decode(srcs)

    if out is None:
        out = cp.empty((N, T), dtype=cp.float32)
    else:
        # Type check first — a numpy ndarray would pass the dtype
        # equality (cupy reuses numpy fp32) but fail with an opaque
        # AttributeError on ``out.device.id``. Surface a named error
        # instead.
        if not isinstance(out, cp.ndarray):
            raise ValueError(
                f"read_surface_gifti_gpu: out must be a cupy.ndarray; "
                f"got {type(out).__name__}. To return a host buffer "
                f"instead, omit ``out=`` and pass ``to_host=True``."
            )
        if out.shape != (N, T) or out.dtype != cp.float32:
            raise ValueError(
                f"read_surface_gifti_gpu: out shape/dtype "
                f"{out.shape}/{out.dtype} != expected "
                f"({N}, {T})/float32"
            )
        # Cross-device assignment from the nvCOMP-allocated device
        # buffer to a foreign-device ``out`` would raise deep inside
        # cupy with an opaque message. Catch it here with an explicit
        # error that names both devices.
        cur_dev = int(cp.cuda.runtime.getDevice())
        out_dev = int(out.device.id)
        if out_dev != cur_dev:
            raise ValueError(
                f"read_surface_gifti_gpu: out buffer is on cupy "
                f"device {out_dev} but the current cupy context is "
                f"on device {cur_dev}; switch into the buffer's "
                f"device (e.g. ``with out.device:``) before calling."
            )

    # Reinterpret each uint8 chunk as (N,) fp32 and write into its
    # column. ``cp.asarray(o)`` is a zero-copy view of the device
    # buffer nvCOMP owns; ``.view(cp.float32)`` is also zero-copy;
    # the column-write is one cudaMemcpyDtoD per timepoint (~16 µs
    # launch latency × T = ~4 ms total at T=242). Could be folded
    # into one transpose-style kernel, but is small in absolute
    # terms.
    #
    # Per-chunk byte-length validation: nvCOMP's decode of a
    # heterogeneous-Dim0 darray that slipped past the parser
    # contract check (e.g. the parser's check fires only when Dim0
    # disagrees with the first darray; same-byte-width corruption
    # by a hostile writer is mostly closed off by the per-darray
    # DataType / Encoding / Endian substring check in
    # ``_parse_gifti_chunks``) would otherwise fail deep inside the
    # column write with an opaque slice-mismatch error. This branch
    # surfaces the same explicit "expected N*4 bytes" message as
    # the CPU path's ``_decode_one_chunk``.
    expected_bytes = N * 4
    for t, o in enumerate(outs):
        o_cp = cp.asarray(o)
        if int(o_cp.nbytes) != expected_bytes:
            raise ValueError(
                f"GIFTI {path}: GPU decode of DataArray {t} produced "
                f"{int(o_cp.nbytes)} bytes, expected {expected_bytes} "
                f"(= N={N} * 4 bytes/fp32); Dim0 disagrees with the "
                f"decoded payload size."
            )
        out[:, t] = o_cp.view(cp.float32)

    if to_host:
        return cp.asnumpy(out)
    return out


# ─────────────────────────────────────────────────────────────────────
# Full device-stay GPU pipeline (bytescan + base64 on GPU, then nvCOMP)
# ─────────────────────────────────────────────────────────────────────
_GPU_PIPE_CACHE: dict = {}
_GPU_PIPE_LOCK = threading.Lock()


def _get_gpu_pipe_kernels():
    """Lazy-build the bytescan + base64 RawKernels + lookup table.

    The kernels are pure CuPy RawKernel — same convention as
    ``arealmshbm/generate_profiles/_kernels_gpu.py``. Cached so subsequent
    file reads don't pay the JIT cost.
    """
    if "kernels" in _GPU_PIPE_CACHE:
        return _GPU_PIPE_CACHE["kernels"]
    with _GPU_PIPE_LOCK:
        if "kernels" in _GPU_PIPE_CACHE:
            return _GPU_PIPE_CACHE["kernels"]
        import cupy as cp

        bytescan = cp.RawKernel(r"""
extern "C" __global__
void find_gifti_tags(
    const unsigned char* __restrict__ buf,
    long long              buf_len,
    int*                  __restrict__ out_positions,
    int*                  __restrict__ out_kinds,    // 0=DA_open, 1=Data_open, 2=Data_close
    int*                  __restrict__ n_found,
    int                    max_found
){
    const long long i = (long long)blockIdx.x * (long long)blockDim.x
                      + (long long)threadIdx.x;
    if (i >= buf_len) return;
    if (buf[i] != (unsigned char)'<') return;
    int kind = -1;
    if (i + 10 <= buf_len &&
        buf[i+1]==(unsigned char)'D' && buf[i+2]==(unsigned char)'a' &&
        buf[i+3]==(unsigned char)'t' && buf[i+4]==(unsigned char)'a' &&
        buf[i+5]==(unsigned char)'A' && buf[i+6]==(unsigned char)'r' &&
        buf[i+7]==(unsigned char)'r' && buf[i+8]==(unsigned char)'a' &&
        buf[i+9]==(unsigned char)'y') {
        kind = 0;
    } else if (i + 6 <= buf_len &&
        buf[i+1]==(unsigned char)'D' && buf[i+2]==(unsigned char)'a' &&
        buf[i+3]==(unsigned char)'t' && buf[i+4]==(unsigned char)'a' &&
        buf[i+5]==(unsigned char)'>') {
        kind = 1;
    } else if (i + 7 <= buf_len &&
        buf[i+1]==(unsigned char)'/' && buf[i+2]==(unsigned char)'D' &&
        buf[i+3]==(unsigned char)'a' && buf[i+4]==(unsigned char)'t' &&
        buf[i+5]==(unsigned char)'a' && buf[i+6]==(unsigned char)'>') {
        kind = 2;
    } else {
        return;
    }
    int idx = atomicAdd(n_found, 1);
    if (idx < max_found) {
        out_positions[idx] = (int)i;
        out_kinds[idx] = kind;
    }
}
""", "find_gifti_tags")

        b64 = cp.RawKernel(r"""
extern "C" __global__
void base64_decode_chunks(
    const unsigned char* __restrict__ b64_table,
    const unsigned char* __restrict__ in,
    const int*           __restrict__ in_offsets,
    const int*           __restrict__ in_lens,
    unsigned char*       __restrict__ out,
    const int*           __restrict__ out_offsets
){
    const int chunk = blockIdx.x;
    const int q     = blockIdx.y * blockDim.x + threadIdx.x;
    const int len_in = in_lens[chunk];
    const int n_quads = len_in >> 2;
    if (q >= n_quads) return;
    const unsigned char* src = in  + in_offsets[chunk]  + (q << 2);
    unsigned char*       dst = out + out_offsets[chunk] + q * 3;
    unsigned int a = b64_table[src[0]];
    unsigned int b = b64_table[src[1]];
    unsigned int c = b64_table[src[2]];
    unsigned int d = b64_table[src[3]];
    unsigned int triple = (a << 18) | (b << 12) | (c << 6) | d;
    dst[0] = (unsigned char)((triple >> 16) & 0xffu);
    dst[1] = (unsigned char)((triple >>  8) & 0xffu);
    dst[2] = (unsigned char)( triple        & 0xffu);
}
""", "base64_decode_chunks")

        # Standard base64 lookup table.
        table = np.zeros(256, dtype=np.uint8)
        for i, c in enumerate(
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        ):
            table[c] = i
        table_dev = cp.asarray(table)

        _GPU_PIPE_CACHE["kernels"] = (bytescan, b64, table_dev)
        return _GPU_PIPE_CACHE["kernels"]


def _read_file_to_host_bytes(path: PathLike) -> bytes:
    """Single open + read pass. Released GIL during the syscall —
    useful as a thread-pool task."""
    with open(path, "rb") as f:
        return f.read()


def read_surface_giftis_gpu_full_pipeline(
    paths,
    *,
    parse_pool=None,
    n_workers: int = 8,
    to_host: bool = False,
    layout: str = "NxT",
):
    """End-to-end device-stay decode of N ``.func.gii`` files.

    Vs :func:`read_surface_gifti_gpu` (single-file path with CPU-side
    bytescan + base64), this version moves the parse + base64 stages
    onto the GPU and batches the nvCOMP decode across all input files:

      1. host (parallel n_workers): file ``open + read`` (GIL-released).
      2. GPU: H2D one ~45 MB buffer per file.
      3. GPU: ``find_gifti_tags`` kernel locates every ``<DataArray``,
         ``<Data>``, ``</Data>`` start in the file. One thread per byte;
         atomicAdd to collect positions. ~0.13 ms / file.
      4. host: small D2H of position+kind arrays (~3 KB), sort, pair
         (DA, Data, /Data) triples → per-chunk (b64_offset, b64_length).
         Per-darray header validation runs against the D2H'd header
         spans (sub-ms total).
      5. GPU: ``base64_decode_chunks`` kernel decodes all 242 chunks /
         file in one launch. ~0.4 ms / file.
      6. GPU: wrap each chunk's raw-deflate slice (skip 2 zlib header
         bytes, drop trailing 4 adler32 bytes) as ``nvcomp.Array`` views
         — zero-copy on top of the device output buffer.
      7. GPU: drain the worker stream, then ONE batched
         ``codec.decode`` across all files' chunks. The drain is
         required because nvCOMP's ``Codec`` runs on its own internal
         CUDA stream (per ``Codec(cuda_stream=...)`` docs); without an
         explicit sync, the last file's b64 kernel write to
         ``decoded_dev`` is not guaranteed to be visible to the codec
         (worker stream is non_blocking; legacy-default-stream implicit
         sync does not apply). The decode call is host-synchronous on
         return (verified empirically: ``cudaDeviceSynchronize`` after
         ``codec.decode`` returns in <10 µs), so step-8 reads of
         ``decoded_outs`` are safe without a second sync.
      8. GPU: per-file ``cp.stack`` of T chunks → ``(N, T) fp32`` (or
         ``(T, N)`` when ``layout='TxN'`` — saves one transpose vs
         building NxT then re-transposing in the caller).

    All output buffers stay on device through step 8; ``to_host=True``
    triggers a final ``cp.asnumpy`` per result.

    Whitespace constraint
    ---------------------
    Each chunk's b64 payload (the bytes between ``<Data>`` and
    ``</Data>``) MUST contain only b64 alphabet bytes (``A-Z``,
    ``a-z``, ``0-9``, ``+``, ``/``, ``=``). Wrapped b64 with embedded
    ``\\n`` / ``\\r`` / space is rejected with a named ``ValueError``.
    The host-side CPU reader (:func:`read_surface_gifti`) uses
    ``base64.b64decode`` which silently strips whitespace; the GPU
    kernel's table maps every non-alphabet byte to 0 and would
    therefore produce wrong bytes. Diverging-on-input is worse than
    failing-loudly, so we refuse instead of silently mis-decoding.
    DeepPrep / fmriprep don't emit wrapped b64; if a future cohort
    source does, re-encode to unwrapped (single-line) b64.

    Parameters
    ----------
    paths : sequence of PathLike
        ``.func.gii`` paths. Must share the FLOAT32 / GZipBase64Binary
        / LittleEndian contract.
    parse_pool : ThreadPoolExecutor, optional
        Caller-supplied pool for the file read step (step 1). Falls
        back to a per-call pool of size ``n_workers``.
    n_workers : int, optional
        File-read pool size when ``parse_pool`` is None. Default 8.
    to_host : bool, optional
        Final D2H of each output. Default False.
    layout : {'NxT', 'TxN'}, optional
        Output axis order. ``'NxT'`` (default) returns ``(N, T) fp32``
        and matches :func:`read_surface_gifti`'s shape. ``'TxN'``
        returns ``(T, N) fp32`` directly and saves one transpose for
        callers (e.g. the step1 GPU prefetcher) that immediately want
        T-major.

    Returns
    -------
    list of cupy.ndarray (or np.ndarray when to_host)
        ``[(N_i, T_i) fp32]`` per input path in order (or
        ``[(T_i, N_i) fp32]`` when ``layout='TxN'``).
    """
    if layout not in ("NxT", "TxN"):
        raise ValueError(
            f"read_surface_giftis_gpu_full_pipeline: layout must be "
            f"'NxT' or 'TxN'; got {layout!r}"
        )
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            "read_surface_giftis_gpu_full_pipeline requires cupy."
        ) from e
    from concurrent.futures import ThreadPoolExecutor

    paths = [Path(p) for p in paths]
    if not paths:
        return []

    bytescan, b64_kernel, b64_table_dev = _get_gpu_pipe_kernels()
    codec, nvcomp = _get_gpu_codec()

    # ── Step 1: parallel file open + read on host ──
    if parse_pool is None:
        with ThreadPoolExecutor(max_workers=n_workers,
                                 thread_name_prefix="gifti-read") as pool:
            host_bufs = list(pool.map(_read_file_to_host_bytes, paths))
    else:
        host_bufs = list(parse_pool.map(_read_file_to_host_bytes, paths))

    # ── Step 2: H2D each buffer ──
    # ``cp.asarray(np.frombuffer(host_buf, uint8))`` H2Ds the whole 45 MB
    # in one transfer. We allocate per-file because each file is
    # variable-size and we want zero-copy nvcomp.Array views into each.
    file_bufs_dev = [
        cp.asarray(np.frombuffer(b, dtype=np.uint8)) for b in host_bufs
    ]

    # ── Steps 3-5: bytescan + chunk extraction + base64 per file ──
    # We keep host-side bookkeeping minimal: positions are sorted on host
    # (~us for 726 ints), per-chunk b64 (start, len) is computed and
    # H2D'd back as small int32 arrays for the base64 kernel.
    #
    # The decoded device buffer for each file becomes the source for
    # nvcomp.Array wraps in step 6.

    per_file = []   # (N, T, decoded_dev, chunk_offsets_dev, chunk_real_lens)
    for fi, (buf_dev, host_buf) in enumerate(zip(file_bufs_dev, host_bufs)):
        # 3a) GPU bytescan
        buf_len = int(buf_dev.size)
        MAX_FOUND = 4096   # 242*3 = 726 expected on YS; cap allows up to
                            # ~1300 darrays before overflow.
        positions_dev = cp.empty(MAX_FOUND, dtype=cp.int32)
        kinds_dev = cp.empty(MAX_FOUND, dtype=cp.int32)
        n_found_dev = cp.zeros(1, dtype=cp.int32)
        block = 256
        grid = (buf_len + block - 1) // block
        bytescan(
            (grid,), (block,),
            (buf_dev, np.int64(buf_len),
             positions_dev, kinds_dev, n_found_dev, np.int32(MAX_FOUND)),
        )
        n_found = int(n_found_dev.get().item())
        if n_found > MAX_FOUND:
            raise ValueError(
                f"GIFTI {paths[fi]}: bytescan found {n_found} tags > "
                f"max_found {MAX_FOUND}; raise MAX_FOUND in gifti_io.py."
            )

        # 4) D2H positions+kinds (~ {n_found} ints = sub-KB)
        positions = cp.asnumpy(positions_dev[:n_found])
        kinds = cp.asnumpy(kinds_dev[:n_found])
        order = np.argsort(positions)
        positions = positions[order]
        kinds = kinds[order]

        # Walk the position stream and pair (DA, Data, /Data) triples
        # per darray. Production GIFTI files have a strict
        # DA -> Data -> /Data -> DA -> ... sequence.
        da_starts = positions[kinds == 0]
        data_starts = positions[kinds == 1]
        data_closes = positions[kinds == 2]
        T = int(len(data_starts))
        if not (len(da_starts) == T and len(data_closes) == T):
            raise ValueError(
                f"GIFTI {paths[fi]}: tag counts mismatch — "
                f"DA={len(da_starts)} Data={T} /Data={len(data_closes)}"
            )

        # Per-chunk b64 byte range: from end of <Data> to start of </Data>.
        DATA_OPEN_LEN = len(b"<Data>")
        chunk_b64_starts = data_starts + DATA_OPEN_LEN  # exclusive of '<Data>'
        chunk_b64_ends = data_closes
        chunk_b64_lens = chunk_b64_ends - chunk_b64_starts
        # Sanity 1: every chunk length must be non-negative. A malformed
        # GIFTI where <Data>/</Data> pairing slips could produce a
        # negative length that the modulo-4 check below would not catch
        # — numpy's ``%`` returns a non-negative remainder, so e.g.
        # ``(-4) % 4 == 0`` passes the modulo test but trips
        # ``cp.empty(total_out)`` later with an opaque "negative
        # dimensions" error. Surface the misparse here instead.
        if np.any(chunk_b64_lens < 0):
            bad = np.where(chunk_b64_lens < 0)[0]
            raise ValueError(
                f"GIFTI {paths[fi]}: chunk {int(bad[0])} has negative "
                f"b64 length {int(chunk_b64_lens[bad[0]])} — malformed "
                f"<Data>/</Data> pairing"
            )
        # Sanity 2: every chunk length must be a multiple of 4 (b64).
        if not np.all(chunk_b64_lens % 4 == 0):
            bad = np.where(chunk_b64_lens % 4 != 0)[0]
            raise ValueError(
                f"GIFTI {paths[fi]}: chunk {int(bad[0])} b64 length "
                f"{int(chunk_b64_lens[bad[0]])} not a multiple of 4"
            )

        # First-DataArray full validation + N extraction (host).
        # Read the header span [da_starts[0], data_starts[0]) and pull
        # Dim0/DataType/Encoding/Endian out via the same regexes as
        # _parse_gifti_chunks. We re-use existing module-level needles.
        first_da_start = int(da_starts[0])
        first_da_header_end = int(data_starts[0])
        first_da_bytes = host_buf[first_da_start:first_da_header_end]
        dt = _RE_DATATYPE.search(first_da_bytes)
        if not dt or dt.group(1) != _EXPECTED_DTYPE:
            raise ValueError(
                f"GIFTI {paths[fi]}: DataType="
                f"{dt.group(1).decode() if dt else 'MISSING'!r}; "
                f"requires {_EXPECTED_DTYPE.decode()!r}"
            )
        enc = _RE_ENCODING.search(first_da_bytes)
        if not enc or enc.group(1) != _EXPECTED_ENC:
            raise ValueError(
                f"GIFTI {paths[fi]}: Encoding="
                f"{enc.group(1).decode() if enc else 'MISSING'!r}; "
                f"requires {_EXPECTED_ENC.decode()!r}"
            )
        end_attr = _RE_ENDIAN.search(first_da_bytes)
        if not end_attr or end_attr.group(1) != _EXPECTED_END:
            raise ValueError(
                f"GIFTI {paths[fi]}: Endian="
                f"{end_attr.group(1).decode() if end_attr else 'MISSING'!r}; "
                f"requires {_EXPECTED_END.decode()!r}"
            )
        m_dim0 = _RE_DIM0.search(first_da_bytes)
        if not m_dim0:
            raise ValueError(
                f"GIFTI {paths[fi]}: missing Dim0 attribute"
            )
        N = int(m_dim0.group(1))

        # Subsequent darrays: cheap substring presence checks against
        # the host buffer. Bounded per darray (header ends at the
        # corresponding <Data> start), so each check is bytes.find over
        # ~100 bytes — sub-us total at T=242 / file, well inside the
        # noise floor of the GPU pipeline. Always on (no opt-out knob)
        # since the cost is negligible and skipping it would leak
        # heterogeneous-attr GIFTIs into nvCOMP with opaque errors.
        DTYPE_NEEDLE = b'DataType="' + _EXPECTED_DTYPE + b'"'
        ENC_NEEDLE = b'Encoding="' + _EXPECTED_ENC + b'"'
        END_NEEDLE = b'Endian="' + _EXPECTED_END + b'"'
        DIM0_NEEDLE = b'Dim0="' + str(N).encode() + b'"'
        for j in range(1, T):
            hs = int(da_starts[j])
            he = int(data_starts[j])
            hdr = host_buf[hs:he]
            if (DTYPE_NEEDLE not in hdr or ENC_NEEDLE not in hdr or
                END_NEEDLE not in hdr or DIM0_NEEDLE not in hdr):
                raise ValueError(
                    f"GIFTI {paths[fi]}: DataArray {j} contract "
                    f"violation (DataType/Encoding/Endian/Dim0)"
                )

        # Whitespace contract: the GPU base64 kernel's lookup table
        # maps every non-alphabet byte to 0, so wrapped b64 with
        # embedded \n / \r / space / \t would silently produce wrong
        # bytes (and the deflate decode that follows would fail with
        # an opaque "invalid stored block lengths" or similar). The
        # CPU reader's ``base64.b64decode`` strips whitespace, so
        # accepting it here would also drift the two backends out of
        # sync. Surface the divergence as an explicit ValueError
        # instead. ~0.12 µs / chunk on a 165 KB span (measured), so
        # ~120 µs / file at T=242 — negligible.
        for j in range(T):
            cs = int(chunk_b64_starts[j])
            ce = int(chunk_b64_ends[j])
            chunk_bytes = host_buf[cs:ce]
            if (b'\n' in chunk_bytes or b'\r' in chunk_bytes or
                b' ' in chunk_bytes or b'\t' in chunk_bytes):
                raise ValueError(
                    f"GIFTI {paths[fi]}: DataArray {j} <Data> payload "
                    f"contains whitespace. This GPU reader requires "
                    f"unwrapped (single-line) b64; the writer must not "
                    f"line-wrap the GZipBase64Binary payload."
                )

        # 5) GPU base64 decode of all T chunks of this file.
        chunk_b64_starts_dev = cp.asarray(chunk_b64_starts.astype(np.int32))
        chunk_b64_lens_dev = cp.asarray(chunk_b64_lens.astype(np.int32))
        # Output buffer: per-chunk (len_in/4)*3 bytes; over-allocates by
        # ≤2 bytes per chunk for '=' padding handling.
        out_aligned_lens = (chunk_b64_lens >> 2) * 3
        out_offsets = np.zeros(T, dtype=np.int32)
        out_offsets[1:] = np.cumsum(out_aligned_lens[:-1])
        total_out = int(out_aligned_lens.sum())
        out_offsets_dev = cp.asarray(out_offsets)
        decoded_dev = cp.empty(total_out, dtype=cp.uint8)

        bblock = 256
        max_quads = int((chunk_b64_lens.max() + 3) >> 2)
        grid_y = (max_quads + bblock - 1) // bblock
        b64_kernel(
            (T, grid_y), (bblock,),
            (b64_table_dev, buf_dev,
             chunk_b64_starts_dev, chunk_b64_lens_dev,
             decoded_dev, out_offsets_dev),
        )

        # Real per-chunk decoded length (accounting for '=' padding).
        # decoded_real_len = (b64_len / 4) * 3 - pad_count where
        # pad_count ∈ {0, 1, 2}. We compute on host from the b64 tail.
        real_lens = out_aligned_lens.copy()
        for ti in range(T):
            tail_start = int(chunk_b64_starts[ti] + chunk_b64_lens[ti])
            # Look at last 2 bytes of b64 input for '='
            pad = 0
            if host_buf[tail_start - 1:tail_start] == b"=":
                pad += 1
                if host_buf[tail_start - 2:tail_start - 1] == b"=":
                    pad += 1
            real_lens[ti] -= pad

        per_file.append((N, T, decoded_dev, out_offsets, real_lens))

    # ── Step 6: wrap raw-deflate slices as nvcomp.Array views ──
    # zlib framing: skip first 2 bytes (header), drop last 4 (adler32).
    all_srcs = []
    file_chunk_ranges = []   # (start_idx_in_all_srcs, n_chunks)
    for (N, T, decoded_dev, offsets, real_lens) in per_file:
        start_idx = len(all_srcs)
        for ti in range(T):
            chunk_view = decoded_dev[
                int(offsets[ti]) + 2 : int(offsets[ti]) + int(real_lens[ti]) - 4
            ]
            all_srcs.append(nvcomp.as_array(chunk_view))
        file_chunk_ranges.append((start_idx, T))

    # ── Step 7: one batched nvCOMP decode for all files' chunks ──
    # Drain the worker stream first. ``Codec.decode`` runs on nvCOMP's
    # own internal stream (per ``nvidia.nvcomp.Codec(cuda_stream=...)``
    # docs); the b64 kernel for the last file was launched on the
    # worker's non_blocking stream and is not yet guaranteed to be
    # visible to the codec. Earlier files were implicitly drained by
    # the next file's ``n_found_dev.get()`` host sync, but the last
    # file has no following sync. ``synchronize()`` here is a single
    # host stall per call (~tens of µs); negligible vs the per-call
    # decode cost.
    cp.cuda.get_current_stream().synchronize()
    decoded_outs = codec.decode(all_srcs)

    # ── Step 8: per-file stack to (N, T) or (T, N) fp32 ──
    # ``cp.stack`` of T per-chunk uint8 buffers yields (T, N*4) C-contig.
    # The view+reshape is zero-copy and lands at (T, N) — that is the
    # ``layout='TxN'`` exit directly. ``layout='NxT'`` pays one
    # ascontiguousarray-after-transpose (~66 µs / file at fsa6 fp32).
    # The step1 GPU prefetcher requests ``layout='TxN'`` to skip BOTH
    # this transpose AND the immediate re-transpose it would otherwise
    # do on the consumer side.
    results = []
    for ((N, T, _, _, _), (start, n_chunks)) in zip(per_file, file_chunk_ranges):
        chunks_dev = [cp.asarray(decoded_outs[start + ti]) for ti in range(n_chunks)]
        stacked = cp.stack(chunks_dev, axis=0)             # (T, N*4) uint8
        as_fp32 = stacked.view(cp.float32).reshape(n_chunks, N)  # (T, N)
        if layout == "TxN":
            out = cp.ascontiguousarray(as_fp32)             # (T, N) C-contig
        else:
            out = cp.ascontiguousarray(as_fp32.T)           # (N, T) C-contig
        if to_host:
            results.append(cp.asnumpy(out))
        else:
            results.append(out)

    return results
