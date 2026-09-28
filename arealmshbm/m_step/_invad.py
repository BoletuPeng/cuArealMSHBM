"""_invad.py

Inverse of ``A_d`` for the M-step kappa update — given mean resultant
length ``rbar`` and dimension ``D``, return the vMF concentration κ:

    A_d(κ) = I_{D/2}(κ) / I_{D/2-1}(κ)  =  rbar
    invAd(D, rbar) = the κ solving the above

Algorithm (the value stock CBIG's local ``invAd`` computes into
``out``):

    1. ``κ0 = (D-1)·rbar/(1-rbar²) + (D/(D-1))·rbar``  (Banerjee-2005)
    2. If ``besseli(D/2-1, κ0)`` is Inf / NaN / 0:
           ``κ = κ0 - D/(D-1) · rbar/2``  (asymptotic correction)
    3. Otherwise:
           ``κ = fzero(λx. A_d(x) - rbar, κ0)``, the asymptotic
           correction if fzero does not converge

Stock CBIG declares it ``function [outu] = invAd(D, rbar)`` (a local
function in ``generate_ini_params.m`` and in each step-2 / step-3
estimation file): it computes ``out`` above and
returns ``outu``, the warm start κ0. So MATLAB's M-step uses κ0; this
port returns the polished root (or the asymptotic value where the
secant reports non-convergence), and neither this module nor
``ini_params._invad.invAd`` returns the MATLAB value.

At D ≈ 1175 the besseli probe is finite for κ0 below about 880, so the
root branch runs whenever κ0 is below that; above it the probe
overflows and the asymptotic correction runs. The step-1 ``group.mat``
``epsil`` values measured on disk (151 files) span 1126-4841, all in
the asymptotic branch, where :func:`invad` and ``ini_params``' brentq
``invAd`` are bit-equal. The step-3 M-step takes the root branch on
almost every call (124/126 on reference-cohort sub-001, rbar 0.25-0.54).

Called ~3 times per M-step inner iter, ~9 times per M-step call,
~27-30 times per parcellation. Sub-millisecond and not a hotspot.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy import optimize, special

from arealmshbm.em_stop_criterion._cdln import _log_bessel_i


def invad(D: float, rbar: float) -> float:
    """Inverse of ``A_d`` (Bessel ratio) at ``rbar``.

    Parameters
    ----------
    D    : float — dimension parameter of the vMF distribution.
    rbar : float — mean resultant length, typically in (0, 1).

    Returns
    -------
    kappa : float (fp64) — concentration parameter.

    Notes
    -----
    Computes what MATLAB's local ``invAd`` computes into ``out`` (it
    returns the warm start ``outu`` instead):
      * Same initial guess.
      * Same fall-through to asymptotic correction when besseli probe is
        non-finite.
      * Root polish by ``scipy.optimize.root_scalar(method="secant")``
        seeded at κ0 in place of ``fzero``; when the secant does not
        converge it returns the asymptotic value. The secant stalls at
        the root (``f(x1) == f(x0)``, non-converged) on about half the
        root-branch calls, so this is not ULP-equal to fzero or to
        :func:`invad_numba`.
    """
    D = float(D)
    rbar = float(rbar)

    # Banerjee-2005 truncated Newton initial guess.
    kappa0 = (D - 1.0) * rbar / (1.0 - rbar * rbar) + (D / (D - 1.0)) * rbar

    # Probe besseli(D/2 - 1, kappa0). If it overflows / underflows / NaNs,
    # fall through to the asymptotic correction (matches MATLAB's branch).
    p = D / 2.0 - 1.0
    bessel_probe = special.iv(p, kappa0)
    if (not np.isfinite(bessel_probe)) or bessel_probe == 0.0:
        return kappa0 - (D / (D - 1.0)) * rbar / 2.0

    # Root polish in place of MATLAB's ``fzero`` (a Brent-style
    # rootfinder seeded with a single point); scipy's secant is the
    # closest single-seed match. On non-convergence MATLAB's ``out`` is
    # the asymptotic correction, and so is this return value.
    def F(x: float) -> float:
        # A_d(x, D) = besseli(D/2, x) / besseli(D/2 - 1, x)
        return special.iv(D / 2.0, x) / special.iv(p, x) - rbar

    try:
        sol = optimize.root_scalar(
            F,
            x0=kappa0,
            x1=kappa0 * 1.001,
            method="secant",
            maxiter=100,
            xtol=1e-12,
        )
        if sol.converged and np.isfinite(sol.root) and sol.root > 0.0:
            return float(sol.root)
    except (ValueError, RuntimeError, FloatingPointError):
        pass

    # Non-convergence → asymptotic correction.
    return kappa0 - (D / (D - 1.0)) * rbar / 2.0


# ─────────────────────────────────────────────────────────────────────
# numba-compatible variant for use inside @njit master kernels.
#
# Same algorithm as ``invad`` above (Bessel probe → secant polish on
# A_d(x)=rbar, else asymptotic fallback) but built on the numba-friendly
# ``_log_bessel_i`` (5-term Debye / series, fp64 throughout) instead of
# scipy. The overflow probe uses ``log(I_v(k0)) > 709`` (the fp64-exp
# overflow threshold) since computing the raw besseli would itself
# overflow at the same threshold.
#
# In the production regime (D = setting_params.dim = 1174 → v = 586),
# log(I_v(k0)) crosses 709 at k0 ≈ 880, so:
#   * iter 1 m=1 (κ ~ 553):  probe finite → secant polish runs
#   * iter 1 m=2+  (κ ≳ 1200): probe overflows → asymptotic correction
# Where scipy's secant in :func:`invad` stalls (f(x1) == f(x0)) and
# reports non-convergence, ``invad`` returns the asymptotic value while
# this one returns the current iterate, so the two differ by up to
# ~4e-4 relative in the root branch.
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False, inline='always',
      error_model='numpy')
def invad_numba(D, rbar):
    """invAd root (MATLAB ``invAd``'s ``out``): Bessel-probe + secant
    polish, asymptotic fallback. ``@njit`` callable.

    Mathematically: returns κ s.t. ``I_{D/2}(κ) / I_{D/2-1}(κ) = rbar``.

    Algorithm:
      1. ``κ0 = (D-1)·rbar/(1-rbar²) + (D/(D-1))·rbar``   (Banerjee init)
      2. Probe ``log(I_{D/2-1}(κ0))``. If > 709 (Bessel overflows fp64),
         or NaN, return ``κ0 - (D/(D-1))·rbar/2``                (asymp).
      3. Else secant polish on ``f(x) = A_d(x) - rbar`` from κ0, with
         ``A_d`` computed in log-space (``exp(log_I_v - log_I_{v-1})``)
         for overflow safety. Up to 50 iters, xtol 1e-12.
      4. If secant diverges / hits ≤0, fall back to asymptotic.
    """
    if rbar <= 0.0:
        return 0.0
    if rbar >= 1.0:
        return math.inf

    v = D / 2.0 - 1.0
    kappa0 = (D - 1.0) * rbar / (1.0 - rbar * rbar) + (D / (D - 1.0)) * rbar
    asymp = kappa0 - (D / (D - 1.0)) * rbar * 0.5

    # Probe: would besseli overflow / underflow / be NaN?
    log_iv = _log_bessel_i(v, kappa0)
    if (log_iv > 709.0) or (log_iv < -708.0) or not (log_iv == log_iv):
        return asymp

    # Secant root-find on f(x) = A_d(x) - rbar.
    #   A_d(x) = I_{v+1}(x) / I_v(x), computed as exp(log diff).
    x0 = kappa0
    x1 = kappa0 * 1.001
    # f0
    lA0 = _log_bessel_i(v + 1.0, x0) - _log_bessel_i(v, x0)
    f0 = math.exp(lA0) - rbar
    # f1
    lA1 = _log_bessel_i(v + 1.0, x1) - _log_bessel_i(v, x1)
    f1 = math.exp(lA1) - rbar

    for _ in range(50):
        df = f1 - f0
        if abs(df) < 1e-300:
            return x1
        x_new = x1 - f1 * (x1 - x0) / df
        if x_new <= 0.0 or not (x_new == x_new):
            return asymp
        if abs(x_new - x1) < 1e-12:
            return x_new
        x0 = x1
        x1 = x_new
        f0 = f1
        lA1 = _log_bessel_i(v + 1.0, x1) - _log_bessel_i(v, x1)
        f1 = math.exp(lA1) - rbar
    # Did not converge → asymptotic correction.
    return asymp
