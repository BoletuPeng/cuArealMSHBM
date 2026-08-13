"""test_bold_io_gpu.py — bit-equality CPU↔GPU on
:func:`concat_hemis_drop_medial`.

Skipped if cupy is unavailable. The CPU implementation
(``concat_hemis_drop_medial``) is the source of truth; this file pins
that the GPU mirror (``concat_hemis_drop_medial_gpu``) returns the
exact same fp32 byte sequence on synthetic inputs that exercise the
two MATLAB-parity-relevant branches:

  * per-hemi NaN→0 happens BEFORE vstack (so a NaN in lh vs in rh both
    map to the same 0.0 cell after the drop)
  * mask reshape accepts (N_full,) and (N_full, 1) shapes

The kernel itself is small; the bit-equality is the contract worth
pinning so a future cupy version upgrade can't silently drift the
device-side NaN handling.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.bold_io import concat_hemis_drop_medial
from arealmshbm.bold_io.bold_io_gpu import concat_hemis_drop_medial_gpu


def _rand_hemi_with_nans(rng: np.random.Generator, n: int, T: int):
    """``(n, T) fp32`` host array sprinkled with NaNs at deterministic
    cells so a bit-equality test under NaN→0 is meaningful."""
    arr = rng.standard_normal((n, T)).astype(np.float32)
    # ~5% NaN density, deterministic indices via the seeded rng.
    n_nan = max(1, (n * T) // 20)
    idx = rng.choice(n * T, size=n_nan, replace=False)
    flat = arr.reshape(-1)
    flat[idx] = np.nan
    return arr


def test_concat_hemis_drop_medial_gpu_bit_equal_basic():
    """Synthetic random lh/rh + alternating medial mask → CPU and GPU
    paths produce bit-equal (N_cortex, T) fp32 output."""
    rng = np.random.default_rng(2026)
    n_h = 64
    T = 7
    lh = _rand_hemi_with_nans(rng, n_h, T)
    rh = _rand_hemi_with_nans(rng, n_h, T)
    mask = np.zeros(2 * n_h, dtype=np.uint8)
    # Make medial-mask non-trivial: alternate cells. ~half of rows drop.
    mask[::3] = 1

    # CPU reference. ``concat_hemis_drop_medial`` mutates lh / rh
    # in-place via ``np.nan_to_num(copy=False)``, so feed it a copy.
    expected = concat_hemis_drop_medial(lh.copy(), rh.copy(), mask.copy())

    lh_dev = cp.asarray(lh)
    rh_dev = cp.asarray(rh)
    mask_dev = cp.asarray(mask)
    out_dev = concat_hemis_drop_medial_gpu(lh_dev, rh_dev, mask_dev)
    out_host = cp.asnumpy(out_dev)

    np.testing.assert_array_equal(out_host, expected)
    assert out_host.dtype == np.float32
    assert out_host.shape == expected.shape


def test_concat_hemis_drop_medial_gpu_mask_2d_shape():
    """``(N_full, 1)`` mask is accepted (matches CPU reshape-safe
    behavior). Bit-equal to CPU on the same input."""
    rng = np.random.default_rng(7)
    n_h = 32
    T = 4
    lh = _rand_hemi_with_nans(rng, n_h, T)
    rh = _rand_hemi_with_nans(rng, n_h, T)
    mask = (rng.integers(0, 2, size=2 * n_h)).astype(np.uint8)
    mask2d = mask.reshape(-1, 1)

    expected = concat_hemis_drop_medial(lh.copy(), rh.copy(), mask2d.copy())
    out_dev = concat_hemis_drop_medial_gpu(
        cp.asarray(lh), cp.asarray(rh), cp.asarray(mask2d))
    np.testing.assert_array_equal(cp.asnumpy(out_dev), expected)


def test_concat_hemis_drop_medial_gpu_all_medial_drops_all_rows():
    """All-1 mask → empty (0, T) output. Pins the trivial-edge case
    so a future refactor can't accidentally segfault on an empty
    boolean gather."""
    rng = np.random.default_rng(3)
    n_h = 16
    T = 5
    lh = rng.standard_normal((n_h, T)).astype(np.float32)
    rh = rng.standard_normal((n_h, T)).astype(np.float32)
    mask = np.ones(2 * n_h, dtype=np.uint8)
    out_dev = concat_hemis_drop_medial_gpu(
        cp.asarray(lh), cp.asarray(rh), cp.asarray(mask))
    assert out_dev.shape == (0, T)
    assert out_dev.dtype == cp.float32


def test_concat_hemis_drop_medial_gpu_rejects_host_inputs():
    """Host numpy inputs must raise — the function is for device
    arrays only. The named TypeError keeps the failure mode visible
    instead of letting it surface deep inside cupy."""
    n_h = 8
    T = 2
    lh = np.zeros((n_h, T), dtype=np.float32)
    rh = np.zeros((n_h, T), dtype=np.float32)
    mask = np.zeros(2 * n_h, dtype=np.uint8)
    with pytest.raises(TypeError, match="cupy.ndarray"):
        concat_hemis_drop_medial_gpu(lh, cp.asarray(rh), cp.asarray(mask))
    with pytest.raises(TypeError, match="cupy.ndarray"):
        concat_hemis_drop_medial_gpu(cp.asarray(lh), rh, cp.asarray(mask))


def test_concat_hemis_drop_medial_gpu_rejects_host_mask():
    """Host numpy mask must raise — closes the silent-H2D-per-call
    footgun where ``cp.asarray(host_mask)`` would copy the mask on
    every leaf call. The prefetcher always uploads the mask at
    ``__init__``, so there's no legitimate caller passing host mask
    at the leaf boundary."""
    n_h = 8
    T = 2
    lh = cp.zeros((n_h, T), dtype=cp.float32)
    rh = cp.zeros((n_h, T), dtype=cp.float32)
    host_mask = np.zeros(2 * n_h, dtype=np.uint8)
    with pytest.raises(TypeError, match="medial_mask_dev must be"):
        concat_hemis_drop_medial_gpu(lh, rh, host_mask)


def test_concat_hemis_drop_medial_gpu_coerces_non_fp32_input():
    """Mirror CPU's ``np.asarray(lh, dtype=np.float32)`` — non-fp32
    cupy input must be coerced to fp32 to match CPU semantics, so a
    future non-fp32 caller can't silently diverge between backends.

    fp64 input → fp32 output, same numerical content (within fp32
    precision; this test picks values that round-trip exactly)."""
    n_h = 16
    T = 3
    # Use values exactly representable in both fp64 and fp32 so the
    # bit-equality check below has a well-defined expected result.
    lh_f64 = (cp.arange(n_h * T, dtype=cp.float64).reshape(n_h, T) * 0.25)
    rh_f64 = (cp.arange(n_h * T, dtype=cp.float64).reshape(n_h, T) * -0.5)
    mask = cp.zeros(2 * n_h, dtype=cp.bool_)
    out = concat_hemis_drop_medial_gpu(lh_f64, rh_f64, mask)
    assert out.dtype == cp.float32
    expected = cp.concatenate([lh_f64, rh_f64], axis=0).astype(cp.float32)
    cp.testing.assert_array_equal(out, expected)


def test_concat_hemis_drop_medial_gpu_mask_length_mismatch():
    n_h = 8
    T = 2
    lh = cp.zeros((n_h, T), dtype=cp.float32)
    rh = cp.zeros((n_h, T), dtype=cp.float32)
    bad_mask = cp.zeros(2 * n_h + 3, dtype=cp.uint8)
    with pytest.raises(ValueError, match="medial_mask length"):
        concat_hemis_drop_medial_gpu(lh, rh, bad_mask)
