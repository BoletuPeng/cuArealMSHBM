"""test_profile_io_packed.py — focused unit tests for the pre-packed
input path of ``write_subject_profile_tnd(D_unpacked=...)``.

The unpacked path is exercised indirectly by ``step2_io`` tests via the
public writer entry; the packed path was added with the step1 GPU
fused kernel (PR #64) and is only exercised end-to-end through the
40-sub YS reference. Add direct round-trip + rejection coverage here
so a future refactor of the writer's packed branch fails locally
instead of at full pipeline run.

Contract pinned by this file:

  * **Bit-equality** — the packed-bytes path produces a .b2nd that is
    bit-equal to feeding the same data through the unpacked path
    (writer-side ``np.packbits`` of the same uint8 input).
  * **Round-trip** — packed write + packed read returns the original
    packed bytes verbatim, with ``D_unpacked`` preserved in vlmeta.
  * **Padding-bit-zero contract** — when ``D`` is not a multiple of 8,
    padding bits in the last byte must be zero. The writer accepts
    pre-packed input where the caller (the GPU fused kernel) already
    enforces this; the round-trip read preserves it.
  * **Three rejection paths** — non-uint8 dtype, ``D_unpacked <= 0``,
    last-axis size disagrees with ``ceil(D/8)``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arealmshbm.data_io.profile_io import (
    write_subject_profile_tnd,
    read_subject_profile_packed_tnd,
)


def _make_binary_tnd(T: int, N: int, D: int, seed: int = 0) -> np.ndarray:
    """Random binary (T, N, D) fp32 in {0, 1}."""
    rng = np.random.default_rng(seed)
    return (rng.random((T, N, D)) < 0.3).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Bit-equality between unpacked and pre-packed write paths
# ─────────────────────────────────────────────────────────────────────
def test_packed_path_matches_unpacked_path_bit_equal(tmp_path: Path) -> None:
    """Writing via D_unpacked=D vs the unpacked path on the same binary
    input produces bit-equal packed bytes on disk."""
    T, N, D = 3, 17, 19  # D not a multiple of 8 to exercise padding.
    arr_unpacked = _make_binary_tnd(T, N, D, seed=1)
    arr_u8 = arr_unpacked.astype(np.uint8)
    pre_packed = np.ascontiguousarray(
        np.packbits(arr_u8, axis=-1, bitorder="little")
    )
    assert pre_packed.shape == (T, N, (D + 7) // 8)
    assert pre_packed.dtype == np.uint8

    p_unpacked = tmp_path / "via_unpacked.b2nd"
    p_packed = tmp_path / "via_packed.b2nd"
    write_subject_profile_tnd(p_unpacked, np.ascontiguousarray(arr_unpacked))
    write_subject_profile_tnd(p_packed, pre_packed, D_unpacked=D)

    out_unpacked, D_unpacked_read = read_subject_profile_packed_tnd(p_unpacked)
    out_packed, D_packed_read = read_subject_profile_packed_tnd(p_packed)
    assert D_unpacked_read == D
    assert D_packed_read == D
    assert out_unpacked.dtype == np.uint8
    assert out_packed.dtype == np.uint8
    assert out_unpacked.shape == ((T, N, (D + 7) // 8))
    assert np.array_equal(out_unpacked, out_packed), (
        "Pre-packed write path produced bytes that differ from the "
        "unpacked-then-packbits path on identical binary input."
    )


def test_packed_path_round_trip_byte_equal(tmp_path: Path) -> None:
    """Packed write + packed read returns the original packed bytes."""
    T, N, D = 2, 11, 17
    rng = np.random.default_rng(7)
    pre_packed = rng.integers(0, 256, size=(T, N, (D + 7) // 8),
                              dtype=np.uint8)
    # Zero padding bits past D in the last byte (caller's contract).
    last_byte_keep = (1 << (D % 8)) - 1 if D % 8 else 0xff
    if D % 8:
        pre_packed[..., -1] &= last_byte_keep
    pre_packed = np.ascontiguousarray(pre_packed)

    p = tmp_path / "round_trip.b2nd"
    write_subject_profile_tnd(p, pre_packed, D_unpacked=D)
    out, D_read = read_subject_profile_packed_tnd(p)
    assert D_read == D
    assert np.array_equal(out, pre_packed)


# ─────────────────────────────────────────────────────────────────────
# Padding-bit-zero contract preservation
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("D", [1, 7, 8, 9, 15, 16, 17, 24, 1175])
def test_packed_path_padding_zero_preserved(tmp_path: Path, D: int) -> None:
    """When D is not a multiple of 8, padding bits in the last byte are
    zero on output (writer doesn't accidentally touch them)."""
    T, N = 2, 7
    arr_unpacked = _make_binary_tnd(T, N, D, seed=D)
    arr_u8 = arr_unpacked.astype(np.uint8)
    pre_packed = np.ascontiguousarray(
        np.packbits(arr_u8, axis=-1, bitorder="little")
    )

    p = tmp_path / f"D{D}.b2nd"
    write_subject_profile_tnd(p, pre_packed, D_unpacked=D)
    out, D_read = read_subject_profile_packed_tnd(p)
    assert D_read == D

    # Padding bits past D in the last byte must be zero. ``D % 8 == 0``
    # is the no-padding case — no assertion needed.
    if D % 8:
        used_mask = np.uint8((1 << (D % 8)) - 1)
        padding = out[..., -1] & np.uint8(~used_mask)
        assert int(padding.max(initial=0)) == 0, (
            f"D={D}: padding bits past D in last byte are non-zero "
            f"on the round-trip read."
        )


# ─────────────────────────────────────────────────────────────────────
# Rejection paths
# ─────────────────────────────────────────────────────────────────────
def test_packed_path_rejects_non_uint8(tmp_path: Path) -> None:
    T, N, D = 2, 5, 9
    bad = np.zeros((T, N, (D + 7) // 8), dtype=np.float32)
    p = tmp_path / "bad_dtype.b2nd"
    with pytest.raises(ValueError, match="pre-packed input must be uint8"):
        write_subject_profile_tnd(p, bad, D_unpacked=D)


@pytest.mark.parametrize("D_bad", [0, -1, -8])
def test_packed_path_rejects_nonpositive_D(tmp_path: Path, D_bad: int) -> None:
    T, N = 2, 5
    pre_packed = np.zeros((T, N, 1), dtype=np.uint8)
    p = tmp_path / f"bad_D{D_bad}.b2nd"
    with pytest.raises(ValueError, match="D_unpacked must be > 0"):
        write_subject_profile_tnd(p, pre_packed, D_unpacked=D_bad)


def test_packed_path_rejects_dim_size_mismatch(tmp_path: Path) -> None:
    """Caller-provided last-axis size must match ceil(D/8)."""
    T, N, D = 2, 5, 17  # ceil(17/8) = 3
    pre_packed = np.zeros((T, N, 4), dtype=np.uint8)  # 4 != 3
    p = tmp_path / "bad_dim.b2nd"
    with pytest.raises(
        ValueError, match=r"input last-axis size \(4\) != ceil\(D/8\) \(3\)"
    ):
        write_subject_profile_tnd(p, pre_packed, D_unpacked=D)


def test_packed_path_rejects_non_contiguous(tmp_path: Path) -> None:
    """Generic C-contiguity guard fires for both packed and unpacked
    paths; pin it for the packed path here so a future refactor that
    moves the check inside the unpacked branch by accident is caught."""
    T, N, D = 2, 5, 17
    base = np.zeros((T, N, 6), dtype=np.uint8)
    not_contig = base[:, :, ::2]  # strided view, last axis = 3 = ceil(17/8)
    assert not not_contig.flags.c_contiguous
    p = tmp_path / "bad_contig.b2nd"
    with pytest.raises(ValueError, match="C-contiguous"):
        write_subject_profile_tnd(p, not_contig, D_unpacked=D)
