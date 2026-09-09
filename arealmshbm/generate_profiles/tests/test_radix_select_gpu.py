"""test_radix_select_gpu.py

The three-pass radix histogram selector must return **exactly** what
``np.partition(flat, kth)[kth]`` returns -- it feeds a ``>=`` compare
that decides every bit of the packed profile, so "close" is wrong.

Covered: random normal, heavy ties, an all-equal array, extremes
(``+-inf``, denormals, ``+-FLT_MAX``), ``kth`` at both ends, a size
sweep from 1 to 1e7, and the production ``(1175, 81924)`` shape once.
The ``+-0.0`` convention (this selector orders ``-0.0`` below ``+0.0``,
numpy treats them equal) is pinned as *numerically* equal, which is
all the downstream ``>=`` can observe.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.generate_profiles._kernels_gpu import (  # noqa: E402
    exact_kth_smallest_cupy,
    threshold_top_fraction_exact_cupy,
)
from arealmshbm.generate_profiles.profiles import (  # noqa: E402
    _threshold_top_fraction,
)


def _check(a: np.ndarray, kth: int) -> None:
    got = exact_kth_smallest_cupy(cp.asarray(a), kth)
    want = float(np.partition(a, kth)[kth])
    if np.isnan(want):
        assert np.isnan(got)
    else:
        # ``==`` and not array_equal: -0.0 == 0.0 is the documented
        # latitude, and it is invisible to the ``>=`` this feeds.
        assert got == want, (kth, got, want)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 31, 32, 33, 255, 1000, 65537,
                               1_000_003])
def test_matches_np_partition_on_random_normal(n):
    rng = np.random.default_rng(n)
    a = rng.standard_normal(n).astype(np.float32)
    for kth in {0, n // 3, n // 2, n - 1}:
        _check(a, kth)


def test_ten_million_elements():
    rng = np.random.default_rng(7)
    n = 10_000_000
    a = rng.standard_normal(n).astype(np.float32)
    for kth in (0, 1, n // 10, n - 1):
        _check(a, kth)


def test_heavy_ties():
    """Quantized data: 8 distinct values over 2 M elements."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 8, size=2_000_000).astype(np.float32) * np.float32(0.125)
    for kth in (0, 1, 999_999, 1_999_999):
        _check(a, kth)


@pytest.mark.parametrize("v", [0.0, -0.0, 1.0, -3.5, np.inf, -np.inf])
def test_all_equal(v):
    a = np.full(100_000, v, dtype=np.float32)
    for kth in (0, 50_000, 99_999):
        got = exact_kth_smallest_cupy(cp.asarray(a), kth)
        assert got == float(v) or (np.isinf(got) and np.isinf(float(v))
                                    and np.sign(got) == np.sign(float(v)))


def test_extremes_are_ordered_like_numpy():
    a = np.array([np.inf, -np.inf, 0.0, -0.0,
                  np.float32(1e-45),                 # denormal
                  np.finfo(np.float32).max,
                  -np.finfo(np.float32).max,
                  np.finfo(np.float32).tiny], dtype=np.float32)
    for kth in range(a.size):
        _check(a, kth)


def test_mixed_signed_zeros_are_numerically_equal():
    """-0.0 sorts below +0.0 here and numpy calls them equal; both
    answers are the same number, which is all ``>=`` can see."""
    a = np.array([-0.0, 0.0, -0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    for kth in range(a.size):
        got = exact_kth_smallest_cupy(cp.asarray(a), kth)
        want = float(np.partition(a, kth)[kth])
        assert got == want


def test_rejects_bad_input():
    a = cp.zeros(16, dtype=cp.float32)
    with pytest.raises(ValueError, match="fp32"):
        exact_kth_smallest_cupy(cp.zeros(16, dtype=cp.float64), 0)
    with pytest.raises(ValueError, match="out of range"):
        exact_kth_smallest_cupy(a, 16)
    with pytest.raises(ValueError, match="out of range"):
        exact_kth_smallest_cupy(a, -1)
    with pytest.raises(ValueError, match="C-contiguous"):
        exact_kth_smallest_cupy(cp.zeros((4, 8), dtype=cp.float32)[:, ::2], 0)


def test_threshold_wrapper_matches_the_host_rank_arithmetic():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((37, 211)).astype(np.float32)
    for frac in (0.1, 0.05, 0.5, 1.0, 1e-6):
        got = threshold_top_fraction_exact_cupy(cp.asarray(a), frac)
        want = float(_threshold_top_fraction(a, frac))
        assert got == want, frac


def test_production_shape_matches_cp_partition():
    """The real (K, n_full) = (1175, 81924) block, 96.3 M elements."""
    K, N = 1175, 81924
    cp.random.seed(0)
    corr = (cp.random.standard_normal((K, N), dtype=cp.float32)
            * cp.float32(0.3))
    numel = K * N
    kth = numel - int(np.floor(numel * 0.1 + 0.5))
    got = threshold_top_fraction_exact_cupy(corr, 0.1)
    want = float(cp.partition(corr.reshape(-1), kth)[kth].get())
    assert got == want
