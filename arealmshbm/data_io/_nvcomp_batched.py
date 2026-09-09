"""_nvcomp_batched.py — direct ctypes binding to nvCOMP's batched Deflate.

The BOLD GIFTI ingest decompresses one raw-DEFLATE stream per timepoint
per hemisphere, so it needs the batched C entry point: one async launch
taking **device** arrays of pointers and sizes, reporting per chunk the
true decompressed size and an ``nvcompStatus_t``. Alignment is the
caller's job — nvCOMP wants 4-byte-aligned chunk *start addresses* while
base64 output is 3-byte-strided, so :meth:`DeflateBatch.plan` picks the
offsets. Output is bit-identical to ``zlib.decompress``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import ctypes
import glob
import os
import threading

import numpy as np

#: ``nvcompStatus_t`` values named in error messages.
_STATUS_NAMES = {
    0: "nvcompSuccess",
    10: "nvcompErrorInvalidValue",
    11: "nvcompErrorNotSupported",
    12: "nvcompErrorCannotDecompress",
    13: "nvcompErrorBadChecksum",
    14: "nvcompErrorCannotVerifyChecksums",
    15: "nvcompErrorOutputBufferTooSmall",
    16: "nvcompErrorWrongHeaderLength",
    17: "nvcompErrorAlignment",
    18: "nvcompErrorChunkSizeTooLarge",
    19: "nvcompErrorCannotCompress",
}


class _DecompressOpts(ctypes.Structure):
    """``nvcompBatchedDeflateDecompressOpts_t``: 64 B by value, ``reserved`` zeroed."""

    _fields_ = [
        ("backend", ctypes.c_int),               # nvcompDecompressBackend_t
        ("sort_before_hw_decompress", ctypes.c_int),
        ("reserved", ctypes.c_char * 56),
    ]


class _AlignmentRequirements(ctypes.Structure):
    """``nvcompAlignmentRequirements_t`` — input / output / temp."""

    _fields_ = [
        ("input", ctypes.c_size_t),
        ("output", ctypes.c_size_t),
        ("temp", ctypes.c_size_t),
    ]


_CACHE: dict = {}
_LOCK = threading.Lock()


def _find_library() -> str:
    """Absolute path of the nvCOMP shared library.

    Found next to the installed ``nvidia.nvcomp`` package rather than by
    name on ``PATH``: importing that package is what maps the library.
    """
    try:
        import nvidia.nvcomp as _nvcomp
    except ImportError as e:            # pragma: no cover — nvcomp is in the env
        raise ImportError(
            "the GPU BOLD reader requires nvidia-nvcomp-cu12. "
            "Install: pip install nvidia-nvcomp-cu12"
        ) from e
    root = os.path.dirname(os.path.dirname(os.path.abspath(_nvcomp.__file__)))
    pats = [os.path.join(root, "libnvcomp", "bin", "nvcomp64_*.dll"),
            os.path.join(root, "libnvcomp", "lib", "libnvcomp.so*"),
            os.path.join(root, "libnvcomp", "lib64", "libnvcomp.so*")]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise ImportError(
        f"nvidia.nvcomp is installed at {root} but no nvCOMP shared library "
        f"was found next to it (looked for {pats}); the batched Deflate "
        f"binding cannot be loaded.")


class DeflateBatch:
    """Stateless wrapper over nvCOMP's batched raw-Deflate decompressor."""

    def __init__(self):
        path = _find_library()
        dirname = os.path.dirname(path)
        if hasattr(os, "add_dll_directory") and os.path.isdir(dirname):
            # Keeps the handle alive for the process; the directory also
            # holds the library's own dependencies.
            self._dll_dir = os.add_dll_directory(dirname)
        self.path = path
        lib = ctypes.CDLL(path)
        c_sz, c_vp = ctypes.c_size_t, ctypes.c_void_p

        lib.nvcompBatchedDeflateDecompressGetRequiredAlignments.argtypes = [
            _DecompressOpts, ctypes.POINTER(_AlignmentRequirements)]
        lib.nvcompBatchedDeflateDecompressGetRequiredAlignments.restype = \
            ctypes.c_int
        lib.nvcompBatchedDeflateDecompressGetTempSizeAsync.argtypes = [
            c_sz, c_sz, _DecompressOpts, ctypes.POINTER(c_sz), c_sz]
        lib.nvcompBatchedDeflateDecompressGetTempSizeAsync.restype = \
            ctypes.c_int
        lib.nvcompBatchedDeflateDecompressAsync.argtypes = [
            c_vp,        # const void* const* device_compressed_chunk_ptrs
            c_vp,        # const size_t*      device_compressed_chunk_bytes
            c_vp,        # const size_t*      device_uncompressed_buffer_bytes
            c_vp,        # size_t*            device_uncompressed_chunk_bytes
            c_sz,        # size_t             num_chunks
            c_vp,        # void*              device_temp_ptr
            c_sz,        # size_t             temp_bytes
            c_vp,        # void* const*       device_uncompressed_chunk_ptrs
            _DecompressOpts,
            c_vp,        # nvcompStatus_t*    device_statuses
            c_vp,        # cudaStream_t       stream
        ]
        lib.nvcompBatchedDeflateDecompressAsync.restype = ctypes.c_int
        self.lib = lib
        # NVCOMP_DECOMPRESS_BACKEND_DEFAULT, no HW sort, reserved zeroed.
        self.opts = _DecompressOpts(0, 0, b"\0" * 56)

        al = _AlignmentRequirements()
        rc = lib.nvcompBatchedDeflateDecompressGetRequiredAlignments(
            self.opts, ctypes.byref(al))
        self._check(rc, "nvcompBatchedDeflateDecompressGetRequiredAlignments")
        #: Alignment nvCOMP wants for each chunk's start ADDRESS.
        self.input_align = int(al.input) or 1
        self.output_align = int(al.output) or 1
        self.temp_align = int(al.temp) or 1

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(
                f"{what} failed: nvcompStatus_t="
                f"{_STATUS_NAMES.get(rc, rc)} ({rc})")

    def temp_bytes(self, num_chunks: int, max_uncompressed_chunk_bytes: int,
                   max_total_uncompressed_bytes: int) -> int:
        """Scratch bytes the decompressor needs; 0 for the GIFTI shapes."""
        out = ctypes.c_size_t(0)
        rc = self.lib.nvcompBatchedDeflateDecompressGetTempSizeAsync(
            ctypes.c_size_t(int(num_chunks)),
            ctypes.c_size_t(int(max_uncompressed_chunk_bytes)),
            self.opts, ctypes.byref(out),
            ctypes.c_size_t(int(max_total_uncompressed_bytes)))
        self._check(rc, "nvcompBatchedDeflateDecompressGetTempSizeAsync")
        return int(out.value)

    def plan(self, payload_bytes, header_skip: int = 0):
        """Byte offsets to lay out ``payload_bytes`` at, input-aligned.

        ``payload_bytes`` is the decoded-base64 size of each chunk;
        ``header_skip`` is the leading bytes NOT handed to nvCOMP (the
        2-byte zlib header), so it is ``offset + header_skip`` that comes
        out :attr:`input_align`-aligned. Returns ``(offsets, total)``,
        the second being the buffer size to allocate.
        """
        a = int(self.input_align)
        lens = np.asarray(payload_bytes, dtype=np.int64)
        n = int(lens.shape[0])
        offsets = np.empty(n, dtype=np.int64)
        if n == 0:
            return offsets, 0
        # Slot sizes rounded up to the alignment keep every subsequent
        # offset congruent to the first one modulo ``a``.
        slots = (lens + (a - 1)) // a * a
        first = (-int(header_skip)) % a       # so first + header_skip ≡ 0
        offsets[0] = first
        np.cumsum(slots[:-1], out=offsets[1:])
        offsets[1:] += first
        return offsets, int(first + slots.sum())

    def decompress_async(self, *, src_ptrs, src_bytes, dst_ptrs, dst_capacity,
                         actual_bytes, statuses, temp, stream):
        """Launch the batched decode; returns as soon as it is queued.

        Every argument except ``stream`` is a CuPy device array: ``(n,)``
        chunk source addresses (:attr:`input_align`-aligned, see
        :meth:`plan`) and lengths, destination addresses and capacities
        (nvCOMP never writes past one), and the output ``actual_bytes`` /
        ``statuses`` pair. Nothing here synchronises — record an event on
        ``stream`` and wait on it before reading any of the outputs.
        """
        n = int(src_ptrs.shape[0])
        if n == 0:
            return
        tptr = 0 if temp is None else int(temp.data.ptr)
        tbytes = 0 if temp is None else int(temp.nbytes)
        rc = self.lib.nvcompBatchedDeflateDecompressAsync(
            ctypes.c_void_p(int(src_ptrs.data.ptr)),
            ctypes.c_void_p(int(src_bytes.data.ptr)),
            ctypes.c_void_p(int(dst_capacity.data.ptr)),
            ctypes.c_void_p(int(actual_bytes.data.ptr)),
            ctypes.c_size_t(n),
            ctypes.c_void_p(tptr),
            ctypes.c_size_t(tbytes),
            ctypes.c_void_p(int(dst_ptrs.data.ptr)),
            self.opts,
            ctypes.c_void_p(int(statuses.data.ptr)),
            ctypes.c_void_p(int(stream.ptr)),
        )
        self._check(rc, "nvcompBatchedDeflateDecompressAsync")

    @staticmethod
    def status_name(code: int) -> str:
        return _STATUS_NAMES.get(int(code), f"nvcompStatus_t={int(code)}")


def nvcomp_available() -> bool:
    """Can the batched binding actually be built? Cached, never raises.

    Locating the library is not enough: ``ctypes.CDLL`` can still fail
    with ``OSError`` and symbol binding with ``AttributeError``, neither
    of which is an ``ImportError`` the callers catch. So the probe is
    the real constructor (whose result :func:`get_deflate_batch` caches,
    making this free for the caller that then uses it).
    """
    hit = _CACHE.get("avail")
    if hit is None:
        try:
            get_deflate_batch()
            hit = True
        except Exception:
            hit = False
        _CACHE["avail"] = hit
    return hit


def get_deflate_batch() -> DeflateBatch:
    """Process-wide :class:`DeflateBatch` (double-checked locking)."""
    hit = _CACHE.get("b")
    if hit is not None:
        return hit
    with _LOCK:
        hit = _CACHE.get("b")
        if hit is None:
            hit = _CACHE["b"] = DeflateBatch()
        return hit
