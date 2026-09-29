"""test_cdln_d3_oracle.py — ``cdln_general_to_f32(k, d=3)`` against the
S² closed form, transcribed here as the oracle.

``spatial_xyz_prior`` evaluates the log-vMF normaliser on S² (d = 3),
where ``I_{1/2}(k) = sqrt(2/(πk)) · sinh(k)`` gives

    Cdln(k, 3) = log k - k + log(2π)/2 - log1p(-exp(-2k))

with the last term dropped for k ≥ 30 (below fp32 eps) and NaN at
k ≤ 0 (MATLAB convention). The CPU path runs the general Bessel route
in fp64 and casts; ``vmf_clustering/_kernels_gpu.py`` evaluates the
closed form on device. This test pins bit-equality of the fp32 outputs
on the whole domain ``xyz_gamma`` can take.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import math

import numpy as np

from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32

_LOG_2PI = 1.8378770664093454835606594728112


def _closed_form_d3(k_arr: np.ndarray) -> np.ndarray:
    out = np.empty(k_arr.shape[0], dtype=np.float32)
    for j, k in enumerate(k_arr):
        if k <= 0.0:
            out[j] = np.float32(np.nan)
            continue
        lom = math.log1p(-math.exp(-2.0 * k)) if k < 30.0 else 0.0
        out[j] = np.float32(math.log(k) - k + _LOG_2PI * 0.5 - lom)
    return out


def _general_d3(k_arr: np.ndarray) -> np.ndarray:
    out = np.empty(k_arr.shape[0], dtype=np.float32)
    cdln_general_to_f32(k_arr, 3, out)
    return out


def _assert_bit_equal(k_arr: np.ndarray) -> None:
    got = _general_d3(k_arr)
    ref = _closed_form_d3(k_arr)
    finite = np.isfinite(ref)
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    assert np.array_equal(got[finite].view(np.uint32),
                          ref[finite].view(np.uint32))


def test_integer_gammas_to_1e5():
    _assert_bit_equal(np.arange(0, 100_001, dtype=np.float64))


def test_log_spaced_1e_3_to_1e6():
    _assert_bit_equal(np.logspace(-3.0, 6.0, 20_000))


def test_fine_grid_across_the_k_30_switch():
    _assert_bit_equal(np.linspace(1e-3, 60.0, 20_000))


def test_nonpositive_is_nan():
    got = _general_d3(np.array([0.0, -1.0, -1e6]))
    assert np.all(np.isnan(got))
