"""test_bitpacked_norm.py — direct tests for the host bit-unpack +
demean + L2-norm kernel used by the CPU :class:`VmfClusteringSession`.

The kernel is the host counterpart of the device-side
``_normalize_bold_NTD_from_packed`` in vmf_clustering_gpu.py: both
ingest the same on-disk packed bytes and produce numerically
equivalent ``(N, T, D)`` fp32 BOLD.

Three contract pins live here:

* **MW-zero contract** — packed rows that are all-zero on input (the
  upstream invariant after ``fetch_data._read_b2nd_series_packed``)
  produce all-zero fp32 output. No explicit ``mw_full`` kwarg is
  needed because the kernel's ``has_zero`` gate skips the divide on
  zero-norm rows; this test pins that gate.
* **Non-zero correctness** — for a non-zero binary row, output =
  fp32 cast of ``(bit - mean) / sqrt(post_sumsq)``. This is the
  algebraic identity the kernel claims; pinning it here means a
  future refactor that breaks the math fails this test, not the
  E2E.
* **Shape / dtype validation** — the input contract for the public
  ``unpack_normalize_packed_NTD_host`` wrapper.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.data_io.bitpacked_norm import (
    unpack_normalize_packed_NTD_host,
)


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────
def _pack_binary_row(row: np.ndarray) -> np.ndarray:
    """Pack a 1-D 0/1 fp32/uint8 row into uint8 bit-packed (LSB-first).
    Mirrors what step1's binarize writes to .b2nd."""
    row_u8 = row.astype(np.uint8)
    return np.packbits(row_u8, bitorder="little")


def _reference_normalize(row: np.ndarray, D: int) -> np.ndarray:
    """Reference fp32 demean + L2-norm of a binary row, written as
    plainly as possible. Used as the ground truth for the kernel's
    output on non-zero rows."""
    row_f = row.astype(np.float32)[:D]
    if np.all(row_f == 0):
        return row_f
    mean = np.float32(np.float64(row_f.sum()) / np.float64(D))
    v = row_f - mean
    if np.any(v == 0.0):
        # Match the kernel's all_nonzero gate.
        return v
    sumsq = float((v.astype(np.float64) ** 2).sum())
    inv = np.float32(1.0) / np.float32(np.sqrt(sumsq))
    return (v * inv).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# contract pin: MW-zero rows
# ─────────────────────────────────────────────────────────────────────
def test_all_zero_packed_row_stays_zero() -> None:
    """Pins the upstream MW-zero contract: a packed row that's all
    zero (the MW pattern after fetch_data._read_b2nd_series_packed)
    must produce an all-zero fp32 output row.

    This is what lets the kernel skip taking ``mw_full`` as a kwarg —
    the upstream pass guarantees MW rows are zero on input, and the
    kernel's has_zero gate produces zero on zero.
    """
    N, T, D = 4, 2, 17                    # D < 24 → 3 bytes/row
    D_bytes = (D + 7) // 8
    packed = np.zeros((N, T, D_bytes), dtype=np.uint8)
    # Make one non-MW row actually have data so the test isn't trivial
    # — only rows [1] and [3] stay all-zero.
    for n in (0, 2):
        for t in range(T):
            row = np.zeros(D, dtype=np.uint8)
            row[: D // 2] = 1            # arbitrary non-zero pattern
            packed[n, t] = _pack_binary_row(row)
    out = unpack_normalize_packed_NTD_host(packed, D)
    assert out.shape == (N, T, D)
    assert out.dtype == np.float32
    # MW rows (1 and 3) stay all-zero.
    assert np.all(out[1] == 0.0), "MW row 1 leaked non-zero output"
    assert np.all(out[3] == 0.0), "MW row 3 leaked non-zero output"
    # Non-MW rows do something.
    assert not np.all(out[0] == 0.0)
    assert not np.all(out[2] == 0.0)


# ─────────────────────────────────────────────────────────────────────
# correctness: non-zero rows match the algebraic identity
# ─────────────────────────────────────────────────────────────────────
def test_non_zero_row_matches_reference() -> None:
    """A packed row with a non-trivial 0/1 pattern produces the same
    fp32 output as the plain reference implementation."""
    N, T, D = 3, 1, 23                   # uneven D → padding bits matter
    D_bytes = (D + 7) // 8
    rng = np.random.default_rng(seed=20260605)
    bits = rng.integers(0, 2, size=(N, D), dtype=np.uint8)
    # Ensure at least one all-zero row (handled by the MW test above)
    # and at least one fully-1 row (degenerate post-mean zero case).
    bits[0] = 0
    bits[1] = 1
    packed = np.zeros((N, T, D_bytes), dtype=np.uint8)
    for n in range(N):
        packed[n, 0] = _pack_binary_row(bits[n])

    out = unpack_normalize_packed_NTD_host(packed, D)
    for n in range(N):
        ref = _reference_normalize(bits[n], D)
        assert np.allclose(out[n, 0], ref, rtol=0, atol=1e-6), (
            f"row {n} kernel output deviates from reference: "
            f"max |Δ| = {np.max(np.abs(out[n, 0] - ref))}"
        )


def test_all_ones_row_produces_zero() -> None:
    """All-ones row: after demean every cell becomes 0; the kernel's
    has_zero gate fires and the divide is skipped. Output is exactly
    zero (NOT an L2-normalized vector — this matches MATLAB's
    ``all_nonzero`` gate behavior for the degenerate case)."""
    N, T, D = 1, 1, 24
    D_bytes = (D + 7) // 8
    packed = np.zeros((N, T, D_bytes), dtype=np.uint8)
    packed[0, 0] = _pack_binary_row(np.ones(D, dtype=np.uint8))
    out = unpack_normalize_packed_NTD_host(packed, D)
    assert np.all(out == 0.0), (
        f"all-ones row should produce all-zero output via has_zero gate; "
        f"got max |out| = {np.max(np.abs(out))}"
    )


# ─────────────────────────────────────────────────────────────────────
# input-contract validation (the public wrapper raises)
# ─────────────────────────────────────────────────────────────────────
def test_wrong_dtype_raises() -> None:
    arr = np.zeros((2, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        unpack_normalize_packed_NTD_host(arr, D=17)


def test_wrong_ndim_raises() -> None:
    arr = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="3D"):
        unpack_normalize_packed_NTD_host(arr, D=17)


def test_wrong_d_bytes_raises() -> None:
    # D=17 needs 3 bytes/row; pass 4 to fail the ceil(D/8) check.
    arr = np.zeros((2, 1, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"D_bytes"):
        unpack_normalize_packed_NTD_host(arr, D=17)
