"""_cdln.py

``Cdln(k, d)`` — log partition function of the von Mises-Fisher
distribution for general dimension ``d``. 5-term Debye asymptotic
expansion (DLMF 10.41) with a series fallback at small ν / x.

Precision: ~1e-15 fp64 throughout. Materially more accurate than
MATLAB's ``CBIG_ArealMSHBM_Cdln``, whose AB-integration path drifts
~1e-4 absolute past κ=650 — and that drift is the dominant source of
the small (<1%) end-to-end vertex disagreement between this pipeline
and the MATLAB reference.

Public API:
    cdln_general_to_f32(k, d, out, scratch)
        — entrywise log partition; in-place fp32 output.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, float64, int64, void


_LOG_2PI = 1.8378770664093454835606594728112


@njit(float64(float64), cache=True, fastmath=False, inline='always')
def _lgamma(x):
    return math.lgamma(x)


@njit(float64(float64, float64), cache=True, fastmath=False, inline='always')
def _log_bessel_i_series(v, x):
    """log(I_v(x)) via series + log-sum-exp. Best for small / moderate x."""
    if x <= 0.0:
        return 0.0 if v == 0.0 else -math.inf

    log_x_2 = math.log(x * 0.5)
    log_term = v * log_x_2 - _lgamma(v + 1.0)
    log_max = log_term

    x2_4 = x * x * 0.25
    log_x2_4 = math.log(x2_4) if x2_4 > 0.0 else -math.inf

    n_max = min(300, int(x + v + 100))
    terms = np.empty(n_max + 1, dtype=np.float64)
    terms[0] = log_term
    count = 1

    for k in range(1, n_max + 1):
        log_term += log_x2_4 - math.log(float(k)) - math.log(v + float(k))
        terms[count] = log_term
        count += 1
        if log_term > log_max:
            log_max = log_term
        if log_term < log_max - 50.0:
            break

    s = 0.0
    for i in range(count):
        s += math.exp(terms[i] - log_max)

    return log_max + math.log(s)


@njit(float64(float64, float64), cache=True, fastmath=False, inline='always')
def _log_bessel_i_debye(nu, x):
    """log(I_nu(x)) via Debye expansion (DLMF 10.41), 5 terms.

    ~1e-15 precision for nu>=25 (any x), nu>=1 (any x), nu<1 (x>=50).
    """
    if x <= 0.0:
        return 0.0 if nu == 0.0 else -math.inf

    if nu < 1e-10:
        nu = 1e-10

    z = x / nu
    z2 = z * z
    sqrt_1pz2 = math.sqrt(1.0 + z2)

    eta = sqrt_1pz2 + math.log(z / (1.0 + sqrt_1pz2))

    p = 1.0 / sqrt_1pz2
    p2, p3, p4 = p*p, p*p*p, p*p*p*p
    p5, p6, p7 = p4*p, p4*p2, p4*p3
    p9, p10, p11 = p4*p5, p5*p5, p5*p6
    p12, p13, p15 = p6*p6, p6*p7, p7*p7*p

    u1 = (3.0*p - 5.0*p3) / 24.0
    u2 = (81.0*p2 - 462.0*p4 + 385.0*p6) / 1152.0
    u3 = (30375.0*p3 - 369603.0*p5 + 765765.0*p7 - 425425.0*p9) / 414720.0
    u4 = (4465125.0*p4 - 94121676.0*p6 + 349922430.0*p4*p4
          - 446185740.0*p10 + 185910725.0*p12) / 39813120.0
    u5 = (1519035525.0*p5 - 49286948607.0*p7 + 284499769554.0*p9
          - 614135872350.0*p11 + 566098157625.0*p13
          - 188699385875.0*p15) / 6688604160.0

    inv_nu = 1.0 / nu
    correction = (1.0 + u1*inv_nu + u2*inv_nu*inv_nu + u3*inv_nu*inv_nu*inv_nu
                  + u4*inv_nu*inv_nu*inv_nu*inv_nu + u5*inv_nu*inv_nu*inv_nu*inv_nu*inv_nu)

    log_iv = nu*eta - 0.5*(_LOG_2PI + math.log(nu)) - 0.25*math.log(1.0 + z2)
    if correction > 0.0:
        log_iv += math.log(correction)

    return log_iv


@njit(float64(float64, float64), cache=True, fastmath=False, inline='always')
def _log_bessel_i(v, x):
    """log(I_v(x)) — Debye for v>=25, series for v<25 & x<=50, else Debye."""
    if v >= 25.0:
        return _log_bessel_i_debye(v, x)
    elif x <= 50.0:
        return _log_bessel_i_series(v, x)
    else:
        return _log_bessel_i_debye(v, x)


@njit(float64(float64, float64), cache=True, fastmath=False, inline='always')
def _cdln_single(k, v):
    return v * math.log(k) - _log_bessel_i(v, k)


@njit(void(float64[:], float64, float64[:]),
      cache=True, fastmath=False, boundscheck=False)
def cdln_into(k, v, out):
    """``out[i] = v * ln(k[i]) - ln(I_v(k[i]))`` where v = d/2 - 1.

    Returns -Inf at k <= 0; the fp32 wrapper rewrites these as NaN to
    match MATLAB's ``CBIG_ArealMSHBM_Cdln``.
    """
    n = k.shape[0]
    for i in range(n):
        out[i] = _cdln_single(k[i], v)


def cdln_general_to_f32(k_arr: np.ndarray, d: int, out: np.ndarray,
                        scratch_f64: np.ndarray | None = None) -> None:
    """Run ``cdln_into`` in fp64 then cast to fp32 in ``out``. ``k <= 0``
    is rewritten to NaN (MATLAB convention).

    ``scratch_f64`` is a caller-provided (L,) fp64 buffer; allocated per
    call if None.
    """
    L = k_arr.shape[0]
    if scratch_f64 is None:
        scratch_f64 = np.empty(L, dtype=np.float64)
    v = float(d) * 0.5 - 1.0
    cdln_into(k_arr, v, scratch_f64)
    np.copyto(out, scratch_f64)
    bad = (k_arr <= 0.0)
    if bad.any():
        out[bad] = np.float32(np.nan)


def warmup() -> None:
    k = np.array([0.0, 500.0, 1000.0], dtype=np.float64)
    out = np.empty(3, dtype=np.float32)
    scratch = np.empty(3, dtype=np.float64)
    cdln_general_to_f32(k, 1174, out, scratch)
