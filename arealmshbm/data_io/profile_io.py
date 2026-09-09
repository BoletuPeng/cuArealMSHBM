"""profile_io.py

Per-subject RSFC profile IO. One on-disk format: bitpacked uint8.
Only bit-packed readers — no fp32 view or full-load fp32 helper is
exposed. Consumers fuse bit-unpack + per-row demean + L2-norm into a
single numba/RawKernel pass (the fp32 round-trip would otherwise
materialize a ~2.3 GB (T, N, D) intermediate at fsa6 T=6).

The binary 0/1 profile is stored as ``(T, N, ⌈D/8⌉)`` uint8 with LSB-first
packing along D (cell ``d`` ↔ bit ``(d & 7)`` of byte ``(d >> 3)``,
matching ``numpy.packbits(bitorder='little')``). Padding bits past D in
the last byte are zero.

The b2nd ``vlmeta`` carries ``format_version='bitpacked_uint8_v1'`` plus
``D_unpacked`` so readers can recover the exact D given only ``⌈D/8⌉``
bytes per row.

Layout:
    project_dir/profiles_raw/sub<S>/sub<S>_<targ>_roi<seed>.profile.b2nd

Public API
----------
    profile_path(out_dir, sub, targ_mesh, seed_mesh) -> Path
        Canonical per-subject .b2nd path.

    write_subject_profile_tnd(path, arr_tnd) -> Path
        Pack ``arr_tnd`` (binary {0, 1} fp32 or uint8) into a bitpacked
        .b2nd at ``path``.

    SubjectProfileStreamWriter(path, T=, N=, D_unpacked=)
        Incremental variant of the pre-packed branch of the above: one
        ``write_session(t, slab)`` per session, ``close()`` at the end.
        Same payload, same chunk/block grid, same cparams and vlmeta —
        indistinguishable to the readers.

    open_subject_profile_packed_tnd(path)
        Lazy packed-bytes handle: ``arr[t] -> (N, ⌈D/8⌉) uint8`` (zero
        unpack cost). Carries ``.D_unpacked`` (int).

    read_subject_profile_packed_tnd(path) -> (ndarray, int)
        Full-load packed bytes + ``D_unpacked``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


# ``blosc2`` is imported lazily: ``import blosc2`` pulls ``requests``
# (a remote-proxy feature this project never uses) and cost half a
# second on the import of *any* entry point. The codec/filter constants
# moved into ``_cparams()`` for that reason; their values are unchanged.
_BLOSC2 = None


def _blosc2():
    """Import (once) and return the ``blosc2`` module."""
    global _BLOSC2
    if _BLOSC2 is None:
        import blosc2 as _b2
        _BLOSC2 = _b2
    return _BLOSC2


PROFILE_NAME_FMT = "sub{sub}_{targ_mesh}_roi{seed_mesh}.profile.b2nd"

# A plain int, so it needs no blosc2 attribute access at import time.
_CLEVEL = 5


def _cparams():
    """Codec + filter pipeline for the profile writer.

    Bitshuffle on packed bytes is roughly neutral, but LZ4 still picks
    up the runs that survive packing (medial wall, etc.).
    """
    b2 = _blosc2()
    return b2.CParams(
        codec=b2.Codec.LZ4,
        clevel=_CLEVEL,
        nthreads=os.cpu_count() or 1,
        filters=[
            b2.Filter.BITSHUFFLE,
            b2.Filter.NOFILTER,
            b2.Filter.NOFILTER,
            b2.Filter.NOFILTER,
            b2.Filter.NOFILTER,
            b2.Filter.NOFILTER,
        ],
    )

# vlmeta header keys. format_version pins this reader to the exact
# packing convention; D_unpacked recovers the unpacked length; bit_order
# documents the LSB-first convention.
_VLMETA_FORMAT_KEY = "format_version"
_VLMETA_FORMAT_VAL = "bitpacked_uint8_v1"
_VLMETA_D_UNPACKED_KEY = "D_unpacked"
_VLMETA_BITORDER_KEY = "bit_order"
_VLMETA_BITORDER_VAL = "little"


# ─────────────────────────────────────────────────────────────────────
# Path helper
# ─────────────────────────────────────────────────────────────────────
def profile_path(out_dir: str | Path, sub: str | int,
                 targ_mesh: str, seed_mesh: str) -> Path:
    """Canonical per-subject .b2nd path under ``profiles_raw/sub<S>/``."""
    return (Path(out_dir) / "profiles_raw" / f"sub{sub}" /
            PROFILE_NAME_FMT.format(sub=sub, targ_mesh=targ_mesh,
                                     seed_mesh=seed_mesh))


# ─────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────
def write_subject_profile_tnd(
    path: str | Path,
    arr_tnd: np.ndarray,
) -> Path:
    """Write a per-subject ``(T, N, D)`` binary profile as bitpacked .b2nd.

    ``arr_tnd`` is ``(T, N, D)``, dtype fp32 or uint8, C-contiguous, every
    value in {0, 1} (scanned on fp32 input — a non-binary cell would
    silently become 1 under astype(uint8)+packbits, so it is caught
    here). The writer packs with ``np.packbits(axis=-1,
    bitorder='little')`` before handing the bytes to blosc2 and records
    ``D`` in the vlmeta. This is the CPU stage pipeline's writer; the GPU
    leaf packs on device and streams its slabs through
    :class:`SubjectProfileStreamWriter` instead.

    Per-chunk size is fixed at ``(1, N, ceil(D/8))`` so ``arr[t]``
    triggers a single chunk decode.
    """
    arr = np.asarray(arr_tnd)
    if arr.ndim != 3:
        raise ValueError(
            f"write_subject_profile_tnd: arr must be 3-D (T, N, D); "
            f"got shape {arr.shape}"
        )
    if not arr.flags.c_contiguous:
        raise ValueError(
            "write_subject_profile_tnd: arr must be C-contiguous "
            "(call np.ascontiguousarray on the caller side)"
        )

    T, N, D = arr.shape
    if arr.dtype == np.float32:
        bad_mask = np.logical_and(arr != 0.0, arr != 1.0)
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            t_bad, _, _ = np.nonzero(bad_mask)
            t0 = int(t_bad[0])
            sample = float(arr[bad_mask][:1].item())
            raise ValueError(
                f"write_subject_profile_tnd: {n_bad} non-binary cell(s) "
                f"across {T} sessions (first hit in session {t0}, "
                f"e.g. value {sample}). The bitpacked .b2nd format only "
                f"accepts step1's threshold+binarize output."
            )
        arr_u8 = arr.astype(np.uint8, copy=False)
    elif arr.dtype == np.uint8:
        arr_u8 = arr
    else:
        raise ValueError(
            f"write_subject_profile_tnd: dtype must be float32 (binary) "
            f"or uint8 (got {arr.dtype})"
        )

    # numpy.packbits with bitorder='little' packs LSB-first along the
    # specified axis. Cell d -> bit (d & 7) of byte (d >> 3). Padding
    # bits (d >= D within the last byte) are zero.
    packed = np.ascontiguousarray(
        np.packbits(arr_u8, axis=-1, bitorder="little")
    )

    T_p, N_p, D_bytes = packed.shape
    assert (T_p, N_p) == (T, N)
    assert D_bytes == (D + 7) // 8

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()

    cparams = _cparams()
    a = _blosc2().asarray(
        packed,
        chunks=(1, N, D_bytes),
        urlpath=str(p),
        mode="w",
        cparams=cparams,
    )
    a.schunk.vlmeta[_VLMETA_FORMAT_KEY] = _VLMETA_FORMAT_VAL
    a.schunk.vlmeta[_VLMETA_D_UNPACKED_KEY] = int(D)
    a.schunk.vlmeta[_VLMETA_BITORDER_KEY] = _VLMETA_BITORDER_VAL
    del a
    return p


# ─────────────────────────────────────────────────────────────────────
# Streaming (per-session) writer
# ─────────────────────────────────────────────────────────────────────
class SubjectProfileStreamWriter:
    """Write a bitpacked profile .b2nd one **session** at a time.

    The on-disk chunk shape has always been ``(1, N, ceil(D/8))`` — one
    chunk per session — so nothing forces the writer to hold the whole
    subject. ``write_session(t, slab)`` compresses exactly chunk ``t``,
    so a producer that finishes session ``t`` while still computing
    ``t + 1`` overlaps the compression with its own work instead of
    paying for the whole subject after the last session.

    The result carries the same payload, the same chunk/block grid, the
    same cparams and the same three vlmeta keys as the one-shot
    ``write_subject_profile_tnd`` — the readers cannot tell the two
    apart. This is the packed writer: slabs are stored verbatim.

    Sessions may arrive in any order — the b2nd chunk index IS the
    session index, not an append counter — but the production caller
    writes ``0 .. T-1`` in order.

    ``format_version`` is stamped by :meth:`close`, after the last
    session chunk, and it is the key :func:`_open_and_verify` gates on.
    A frame whose ``close()`` never ran (the process died mid-subject)
    is therefore *rejected* by the readers rather than served with
    zero-filled sessions.

    Usable as a context manager: a clean exit calls :meth:`close`, an
    exception calls :meth:`abort` (releases the handle and removes the
    partial file, so a failed subject leaves no half-profile behind).

    Parameters
    ----------
    path : path
        Target .b2nd. Truncated if it exists.
    T, N : int
        Session count and vertex count — the frame's first two dims.
    D_unpacked : int
        Original (unpacked) cell count. The last dim is ``ceil(D/8)``
        and the value is recorded in vlmeta.
    """

    __slots__ = ("path", "T", "N", "D_unpacked", "D_bytes", "written",
                 "_arr", "_closed")

    def __init__(self, path, *, T: int, N: int, D_unpacked: int):
        T = int(T)
        N = int(N)
        D = int(D_unpacked)
        if T <= 0 or N <= 0:
            raise ValueError(
                f"SubjectProfileStreamWriter: T={T}, N={N} must be > 0")
        if D <= 0:
            raise ValueError(
                f"SubjectProfileStreamWriter: D_unpacked must be > 0; "
                f"got {D}")
        self.T = T
        self.N = N
        self.D_unpacked = D
        self.D_bytes = (D + 7) // 8
        self.written: set = set()
        self._closed = False

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
        self.path = p

        b2 = _blosc2()
        # ``empty`` lays the frame out on the same chunk grid the
        # one-shot writer's ``asarray`` picks, and ``blocks`` is
        # auto-computed from ``chunks`` + itemsize, so it matches too.
        # The chunks it pre-creates are placeholders that
        # ``arr[t] = slab`` replaces one at a time — no whole-file
        # rewrite.
        self._arr = b2.empty(
            (T, self.N, self.D_bytes),
            dtype=np.uint8,
            chunks=(1, self.N, self.D_bytes),
            urlpath=str(p),
            mode="w",
            cparams=_cparams(),
        )
        # These two describe the layout, not the completeness of the
        # file, so they may be stamped up front; ``format_version`` may
        # not — see the class docstring.
        vlm = self._arr.schunk.vlmeta
        vlm[_VLMETA_D_UNPACKED_KEY] = D
        vlm[_VLMETA_BITORDER_KEY] = _VLMETA_BITORDER_VAL

    # -- writing ------------------------------------------------------
    def write_session(self, t: int, slab) -> None:
        """Compress + store session ``t``'s ``(N, ceil(D/8))`` uint8 slab.

        The slab is fully consumed before this returns, so the caller
        may recycle the buffer afterwards — but not before.
        """
        if self._closed:
            raise ValueError(
                "SubjectProfileStreamWriter: write_session after close()")
        t = int(t)
        if not (0 <= t < self.T):
            raise ValueError(
                f"SubjectProfileStreamWriter: session index {t} outside "
                f"[0, {self.T})")
        if t in self.written:
            raise ValueError(
                f"SubjectProfileStreamWriter: session {t} written twice")
        a = np.asarray(slab)
        want = (self.N, self.D_bytes)
        if a.dtype != np.uint8:
            raise ValueError(
                f"SubjectProfileStreamWriter: slab must be uint8 "
                f"(got {a.dtype})")
        if a.shape != want:
            raise ValueError(
                f"SubjectProfileStreamWriter: session {t} slab shape "
                f"{a.shape} != {want}")
        self._arr[t] = a
        self.written.add(t)

    # -- teardown -----------------------------------------------------
    def _release(self) -> None:
        arr, self._arr = self._arr, None
        self._closed = True
        del arr

    def close(self):
        """Finalise the frame. Raises if a session was never written."""
        if self._closed:
            return self.path
        missing = sorted(set(range(self.T)) - self.written)
        if not missing:
            # LAST, and only now: the file is complete, so it may
            # advertise the format the readers accept.
            self._arr.schunk.vlmeta[_VLMETA_FORMAT_KEY] = _VLMETA_FORMAT_VAL
            self._release()
            return self.path
        self._release()
        tail = "..." if len(missing) > 8 else ""
        try:
            self.path.unlink()
        except OSError as exc:
            fate = (f"{self.path} could not be removed ({exc}) — delete "
                    f"it by hand")
        else:
            fate = f"{self.path.name} was removed"
        raise ValueError(
            f"SubjectProfileStreamWriter: close() with {len(missing)} "
            f"session(s) never written: {missing[:8]}{tail}. Those chunks "
            f"would read back as zeros, so {fate}.")

    def abort(self) -> None:
        """Release the handle and delete the partial file.

        Never raises — an abort is already on an error path — but a
        file it could not remove is warned about rather than dropped
        silently.
        """
        if not self._closed:
            self._release()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            import warnings
            warnings.warn(
                f"SubjectProfileStreamWriter.abort(): could not remove the "
                f"partial profile {self.path} ({exc}). It carries no "
                f"'{_VLMETA_FORMAT_KEY}' vlmeta, so the readers reject it.",
                RuntimeWarning, stacklevel=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


# ─────────────────────────────────────────────────────────────────────
# vlmeta helpers
# ─────────────────────────────────────────────────────────────────────
def _get_vlmeta_D_unpacked(a) -> int:
    """Read the unpacked D from a bitpacked b2nd handle's vlmeta."""
    vlm = a.schunk.vlmeta
    try:
        D = vlm[_VLMETA_D_UNPACKED_KEY]
    except KeyError:
        D = None
    if D is None:
        raise ValueError(
            f"bitpacked profile is missing '{_VLMETA_D_UNPACKED_KEY}' "
            f"vlmeta — cannot recover the unpacked length"
        )
    return int(D)


def _open_and_verify(path: str | Path):
    """Open a b2nd and verify it carries the expected bitpacked header.

    Returns the open ``blosc2.NDArray``. Raises ``ValueError`` if the
    file isn't a uint8 bitpacked profile produced by this writer.
    """
    a = _blosc2().open(str(Path(path)), mode="r")
    try:
        if a.dtype != np.uint8:
            raise ValueError(
                f"profile_io: {path} has dtype={a.dtype}; bitpacked .b2nd "
                f"must be uint8. Regenerate via step1's generate_profiles."
            )
        vlm = a.schunk.vlmeta
        try:
            fmt_tag = vlm[_VLMETA_FORMAT_KEY]
        except KeyError:
            fmt_tag = None
        if fmt_tag != _VLMETA_FORMAT_VAL:
            raise ValueError(
                f"profile_io: {path} has format_version={fmt_tag!r}; "
                f"this reader only knows {_VLMETA_FORMAT_VAL!r}."
            )
    except Exception:
        del a
        raise
    return a


# ─────────────────────────────────────────────────────────────────────
# Packed-bytes readers (only readers — no fp32-materialize variant)
# ─────────────────────────────────────────────────────────────────────
def open_subject_profile_packed_tnd(path: str | Path):
    """Lazy-decode handle for the bitpacked bytes.

    Returns the underlying ``blosc2.NDArray`` (or a thin wrapper) with
    ``shape = (T, N, ⌈D/8⌉)``, ``dtype = uint8``. Each ``arr[t]`` returns
    a fresh ``(N, ⌈D/8⌉) uint8`` chunk decode — no fp32 cost.

    The returned object carries ``.D_unpacked`` (int) so callers can
    recover the exact cell count.
    """
    a = _open_and_verify(path)
    try:
        a.D_unpacked = _get_vlmeta_D_unpacked(a)  # type: ignore[attr-defined]
    except AttributeError:
        # Cython class that prohibits dynamic attribute assignment —
        # wrap in a thin pass-through.
        a = _PackedHandle(a, _get_vlmeta_D_unpacked(a))
    return a


def read_subject_profile_packed_tnd(
    path: str | Path,
) -> Tuple[np.ndarray, int]:
    """Full-load packed bytes from a bitpacked profile.

    Returns ``(packed, D_unpacked)`` where ``packed`` is a fresh
    ``(T, N, ⌈D/8⌉) uint8`` ndarray and ``D_unpacked`` is the original
    cell count.
    """
    a = _open_and_verify(path)
    try:
        D = _get_vlmeta_D_unpacked(a)
        packed = a[:]
    finally:
        del a
    return packed, D


class _PackedHandle:
    """Minimal wrapper carrying ``D_unpacked`` if ``blosc2.NDArray`` bans
    dynamic attribute assignment. Mirrors ``shape``, ``dtype``,
    ``__getitem__`` so callers can use it just like the raw NDArray.
    """

    __slots__ = ("_a", "D_unpacked")

    def __init__(self, a, D_unpacked: int):
        self._a = a
        self.D_unpacked = D_unpacked

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    def __getitem__(self, t):
        return self._a[t]
