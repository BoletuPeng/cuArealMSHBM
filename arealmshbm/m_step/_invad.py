"""_invad.py

Inverse of ``A_d`` for the M-step kappa update — given mean resultant
length ``rbar`` and dimension ``D``, return the vMF concentration κ:

    A_d(κ) = I_{D/2}(κ) / I_{D/2-1}(κ)  =  rbar
    invAd(D, rbar) = the κ solving the above

Algorithm (mirrors the MATLAB ``CBIG_ArealMSHBM_invAd``):

    1. ``κ0 = (D-1)·rbar/(1-rbar²) + (D/(D-1))·rbar``  (Banerjee-2005)
    2. If ``besseli(D/2-1, κ0)`` is Inf / NaN / 0:
           ``κ = κ0 - D/(D-1) · rbar/2``  (asymptotic correction)
    3. Otherwise:
           ``κ = fzero(λx. A_d(x) - rbar, κ0)``

For our Mode A pipeline (D = setting_params.dim = 1174), ``I_{586}(...)``
overflows fp64 for any κ in the EM-realistic range, so step 2's
asymptotic correction is what runs. We mirror MATLAB exactly so the
fall-through path is bit-comparable when the besseli probe is finite.

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
    Mirrors MATLAB ``CBIG_ArealMSHBM_invAd``:
      * Same initial guess.
      * Same fall-through to asymptotic correction when besseli probe is
        non-finite.
      * Same fzero polish (scipy.optimize.root_scalar / Brent vs MATLAB's
        fzero produce results within fp64 ULP of each other for the
        regimes this function is exercised in).
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

    # fzero polish. MATLAB's ``fzero`` is a Brent-style rootfinder seeded
    # with a single point; scipy's secant-with-derivative-free fallback
    # is the closest match. If fzero fails to converge, MATLAB returns
    # the asymptotic correction — we do the same.
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

    # Failed convergence → MATLAB's fallback.
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
# Matches MATLAB's branch selection bit-for-bit; secant root matches
# scipy's secant within fp64 ULP on the regimes this is exercised in.
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False, inline='always',
      error_model='numpy')
def invad_numba(D, rbar):
    """invAd matching MATLAB ``CBIG_ArealMSHBM_invAd``: Bessel-probe +
    secant polish, asymptotic fallback. ``@njit`` callable.

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
    # Did not converge — MATLAB falls back to asymptotic on bad exitflag.
    return asymp
