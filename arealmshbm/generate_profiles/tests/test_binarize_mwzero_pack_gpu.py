"""test_binarize_mwzero_pack_gpu.py — bit-exact regression gate for the
fused binarize + MW-zero + transpose-pack RawKernel.

Reference: a pure-numpy chain that mirrors the current production path
on host:

    bin_KxV  = (corr_KxV >= threshold).astype(uint8)
    bin_KxV[:, mw_v == 1] = 0                   # MW zero (columns)
    bin_VxK  = np.ascontiguousarray(bin_KxV.T)  # transpose (POST stage)
    packed   = np.packbits(bin_VxK, axis=-1, bitorder='little')

The kernel collapses all four into one device-side launch. This test
checks that the byte-level output is **identical** (np.array_equal) to
the reference for:

  * production-sized shape (K=1175, V=40962) matching fsa6 + Schaefer 300
  * small synthetic shapes including K not a multiple of 8
  * boundary MW masks (all-zero, all-one, sparse, dense)
  * threshold edges (all-above, all-below, mixed)
  * the contract that padding bits past K in the last byte are zero
  * the bitpacked rows at MW vertices are all-zero (writer-side
    contract that step2 SubjectProfileLoader will assert on load).

Run::

    python -m pytest arealmshbm/generate_profiles/tests/test_binarize_mwzero_pack_gpu.py -v

Skips if cupy is unavailable.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.generate_profiles._kernels_gpu import binarize_mwzero_pack_cupy


# ─────────────────────────────────────────────────────────────────────
# Pure-numpy reference. Mirrors the current production chain on host
# (binarize → MW zero → transpose → packbits), end-to-end byte output.
# ─────────────────────────────────────────────────────────────────────
def _reference_pack(corr_KxV: np.ndarray, mw_v: np.ndarray,
                     threshold: float) -> np.ndarray:
    """Reference implementation: bit-for-bit what host code does today."""
    bin_KxV = (corr_KxV >= np.float32(threshold)).astype(np.uint8)
    # Apply MW zero (zero out columns at MW positions).
    bin_KxV[:, mw_v.astype(bool)] = 0
    bin_VxK = np.ascontiguousarray(bin_KxV.T)
    return np.packbits(bin_VxK, axis=-1, bitorder="little")


# ─────────────────────────────────────────────────────────────────────
# Kernel launcher wrapper for the tests.
# ─────────────────────────────────────────────────────────────────────
def _run_kernel(corr_KxV: np.ndarray, mw_v: np.ndarray,
                 threshold: float) -> np.ndarray:
    K, V = corr_KxV.shape
    D_bytes = (K + 7) // 8
    corr_dev = cp.asarray(corr_KxV)
    mw_dev = cp.asarray(mw_v)
    packed_dev = cp.zeros((V, D_bytes), dtype=cp.uint8)
    binarize_mwzero_pack_cupy(corr_dev, mw_dev, float(threshold), packed_dev)
    return cp.asnumpy(packed_dev)


# ─────────────────────────────────────────────────────────────────────
# 1. Production-sized test (fsa6 + Schaefer-300 / K=1175, V=40962).
#    Catches any layout / coalescing / boundary bug at the actual size
#    the GPU stage runs on in production.
# ─────────────────────────────────────────────────────────────────────
def test_bit_exact_production_shape():
    rng = np.random.default_rng(0xCAFE)
    K, V = 1175, 40962
    corr = rng.standard_normal((K, V), dtype=np.float32)
    # ~30% of vertices MW (overpopulated vs YS, exercises the MW path).
    mw = (rng.random(V) < 0.3).astype(np.uint8)
    # Threshold picked so ~10% of cells pass (matches step1 top-fraction=0.1).
    threshold = float(np.quantile(corr, 0.9))

    got = _run_kernel(corr, mw, threshold)
    want = _reference_pack(corr, mw, threshold)
    assert got.shape == want.shape
    assert got.dtype == np.uint8
    assert np.array_equal(got, want), (
        f"bit-exact mismatch on prod shape: "
        f"{int(np.sum(got != want))} differing bytes"
    )


# ─────────────────────────────────────────────────────────────────────
# 2. Small synthetic shapes — K not divisible by 8 stresses the last-
#    byte handling (padding bits must be zero).
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("K,V", [
    (1, 4),         # K=1: single-bit byte
    (7, 16),        # K=7: full first byte minus one
    (8, 32),        # K=8: exactly one byte
    (9, 32),        # K=9: spills to second byte
    (10, 64),       # K=10: 2 bytes, 6 padding bits in last byte
    (17, 100),      # K=17: 3 bytes, 7 padding bits
    (300, 1024),    # K=300: 38 bytes, 4 padding bits
])
def test_bit_exact_small_shapes(K, V):
    rng = np.random.default_rng(0x1234 + K * 31 + V)
    corr = rng.standard_normal((K, V), dtype=np.float32)
    mw = (rng.random(V) < 0.2).astype(np.uint8)
    threshold = float(np.median(corr))

    got = _run_kernel(corr, mw, threshold)
    want = _reference_pack(corr, mw, threshold)
    assert np.array_equal(got, want), (
        f"K={K}, V={V}: {int(np.sum(got != want))} differing bytes"
    )


# ─────────────────────────────────────────────────────────────────────
# 3. MW mask boundary cases.
# ─────────────────────────────────────────────────────────────────────
def test_all_cortex_mw_zero_is_noop():
    """mw_v all zero ⇒ output identical to a pure binarize+transpose+pack."""
    rng = np.random.default_rng(7)
    K, V = 100, 256
    corr = rng.standard_normal((K, V), dtype=np.float32)
    mw = np.zeros(V, dtype=np.uint8)
    threshold = 0.0

    got = _run_kernel(corr, mw, threshold)
    bin_KxV = (corr >= np.float32(threshold)).astype(np.uint8)
    want = np.packbits(np.ascontiguousarray(bin_KxV.T),
                        axis=-1, bitorder="little")
    assert np.array_equal(got, want)


def test_all_mw_yields_all_zero():
    """mw_v all one ⇒ every output byte is zero."""
    rng = np.random.default_rng(11)
    K, V = 50, 64
    corr = rng.standard_normal((K, V), dtype=np.float32)
    mw = np.ones(V, dtype=np.uint8)
    threshold = -10.0  # would normally produce many 1s

    got = _run_kernel(corr, mw, threshold)
    assert got.shape == (V, (K + 7) // 8)
    assert np.array_equal(got, np.zeros_like(got))


# ─────────────────────────────────────────────────────────────────────
# 4. Threshold extremes.
# ─────────────────────────────────────────────────────────────────────
def test_threshold_above_all():
    """threshold > max(corr) ⇒ no bit set (all output bytes are zero)."""
    rng = np.random.default_rng(13)
    K, V = 80, 128
    corr = rng.standard_normal((K, V), dtype=np.float32)
    mw = (rng.random(V) < 0.2).astype(np.uint8)
    threshold = float(corr.max() + 1.0)

    got = _run_kernel(corr, mw, threshold)
    assert np.array_equal(got, np.zeros_like(got))


def test_threshold_below_all():
    """threshold ≤ min(corr) ⇒ cortex bits all set; MW rows stay zero;
    padding bits in the last byte stay zero."""
    rng = np.random.default_rng(17)
    K, V = 13, 64
    corr = rng.standard_normal((K, V), dtype=np.float32)
    mw = (rng.random(V) < 0.4).astype(np.uint8)
    threshold = float(corr.min() - 1.0)

    got = _run_kernel(corr, mw, threshold)
    # First (full) byte: covers k=0..7 ⇒ all 1s on cortex.
    cortex = (mw == 0)
    assert np.all(got[cortex, 0] == 0xFF), "cortex byte0 should be all-1"
    # Second byte covers k=8..12 (5 bits): should be 0b00011111 = 0x1F.
    expected_last = 0x1F
    assert np.all(got[cortex, 1] == expected_last), (
        "cortex last byte should have low 5 bits set, padding zero"
    )
    # MW rows must be zero everywhere.
    assert np.all(got[~cortex] == 0)


# ─────────────────────────────────────────────────────────────────────
# 5. Padding-bit contract — explicit assertion at K=10 (8 + 2 valid bits
#    in the last byte; high 6 bits MUST be 0).
# ─────────────────────────────────────────────────────────────────────
def test_padding_bits_zero_when_K_not_multiple_of_8():
    rng = np.random.default_rng(19)
    K, V = 10, 32
    corr = np.ones((K, V), dtype=np.float32)  # everything above threshold
    mw = np.zeros(V, dtype=np.uint8)
    threshold = 0.0

    got = _run_kernel(corr, mw, threshold)
    # Last byte covers k=8..9 ⇒ valid bits are b0, b1. Padding bits b2..b7
    # must be 0. Expected pattern: 0b00000011 = 0x03.
    assert np.all(got[:, 1] == 0x03), (
        f"padding bits not zero — last bytes: {got[:, 1]}"
    )


# ─────────────────────────────────────────────────────────────────────
# 6. Bit convention sanity — explicit mapping check.
# ─────────────────────────────────────────────────────────────────────
def test_lsb_first_bit_convention():
    """Cell k ↔ bit (k & 7) of byte (k >> 3), LSB-first. Build a profile
    where only k=3 is above threshold, verify byte0 == 0b00001000 = 0x08.
    """
    K, V = 8, 4
    corr = np.full((K, V), -1.0, dtype=np.float32)
    corr[3, :] = 1.0  # only k=3 above threshold
    mw = np.zeros(V, dtype=np.uint8)

    got = _run_kernel(corr, mw, 0.0)
    assert np.all(got[:, 0] == 0x08), (
        f"bit convention wrong — got {got[:, 0]} expected 0x08 (bit 3 set)"
    )
