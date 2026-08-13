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

import blosc2
import numpy as np


PROFILE_NAME_FMT = "sub{sub}_{targ_mesh}_roi{seed_mesh}.profile.b2nd"

# Codec + filter pipeline. bitshuffle on packed bytes is roughly neutral
# but LZ4 still picks up the structure that survives packing (long runs
# at medial wall, etc.). LZ4 clevel 5 — IO bench can re-tune without
# touching reader compat.
_CODEC = blosc2.Codec.LZ4
_CLEVEL = 5
_FILTERS = [
    blosc2.Filter.BITSHUFFLE,
    blosc2.Filter.NOFILTER,
    blosc2.Filter.NOFILTER,
    blosc2.Filter.NOFILTER,
    blosc2.Filter.NOFILTER,
    blosc2.Filter.NOFILTER,
]

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
    *,
    D_unpacked: int | None = None,
) -> Path:
    """Write a per-subject ``(T, N, D)`` binary profile as bitpacked .b2nd.

    Two input modes — selected by the ``D_unpacked`` keyword:

    **Unpacked** (``D_unpacked is None``, default):
      * ``arr_tnd.shape == (T, N, D)``, dtype fp32 or uint8
      * values MUST be in {0, 1} (scanned on fp32 input — a non-binary
        cell silently becomes 1 under astype(uint8)+packbits, so we
        catch it here)
      * writer calls ``np.packbits(axis=-1, bitorder='little')``
        internally before handing the bytes to blosc2.

    **Pre-packed** (``D_unpacked`` is an int):
      * ``arr_tnd.shape == (T, N, ⌈D_unpacked/8⌉)``, dtype uint8 — the
        bytes are already LSB-first packed along the last axis with
        ``D_unpacked`` the original unpacked length.
      * writer skips validation + packbits and forwards the bytes
        straight to blosc2. The vlmeta still records the original
        ``D_unpacked``.
      * This is the GPU-fused-kernel path — the step1 GPU leaf emits
        ``(V_h, ceil(K/8)) uint8`` directly via
        ``binarize_mwzero_pack_cupy``, saving 32x on D2H + the host
        transpose + the writer's astype+packbits.

    Caller-side invariants in both modes:
      * ``arr_tnd.ndim == 3``
      * ``arr_tnd.flags.c_contiguous``

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

    if D_unpacked is None:
        # Legacy path: unpacked binary input; this writer packs it.
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
    else:
        # Pre-packed path: input is already (T, N, ⌈D_unpacked/8⌉) uint8.
        # Skips validation + packbits — caller (the GPU leaf via
        # binarize_mwzero_pack_cupy) guarantees the bytes obey the
        # LSB-first packing convention and that padding bits past
        # D_unpacked in the last byte are zero. Pinned at unit-test
        # level by tests/test_binarize_mwzero_pack_gpu.py.
        if arr.dtype != np.uint8:
            raise ValueError(
                f"write_subject_profile_tnd(D_unpacked={D_unpacked}): "
                f"pre-packed input must be uint8 (got {arr.dtype})"
            )
        D = int(D_unpacked)
        if D <= 0:
            raise ValueError(
                f"write_subject_profile_tnd: D_unpacked must be > 0; got {D}"
            )
        expected_D_bytes = (D + 7) // 8
        T, N, D_bytes_in = arr.shape
        if D_bytes_in != expected_D_bytes:
            raise ValueError(
                f"write_subject_profile_tnd(D_unpacked={D}): input last-axis "
                f"size ({D_bytes_in}) != ceil(D/8) ({expected_D_bytes})"
            )
        packed = arr

    T_p, N_p, D_bytes = packed.shape
    assert (T_p, N_p) == (T, N)
    assert D_bytes == (D + 7) // 8

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()

    cparams = blosc2.CParams(
        codec=_CODEC,
        clevel=_CLEVEL,
        nthreads=os.cpu_count() or 1,
        filters=_FILTERS,
    )
    a = blosc2.asarray(
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
    a = blosc2.open(str(Path(path)), mode="r")
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
