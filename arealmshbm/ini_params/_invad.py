"""invAd Bessel root finder (fp64, scipy).

``A_d`` is the ratio of modified Bessel functions of the first kind:

    A_d(kappa) = I_{D/2}(kappa) / I_{D/2-1}(kappa)

Given a feature dimension ``D`` and a mean cosine ``rbar`` in (0, 1),
``invAd`` returns the vMF concentration estimate ``kappa`` solving
``A_d(kappa) - rbar == 0`` (Banerjee et al. 2005).

Numerical notes:
    * MATLAB ``besseli(nu, x)`` overflows fp64 around ``x ~ 700`` for
      ``nu ~ 587`` (D=1175 case). MATLAB falls back to a closed-form
      approximation when ``besseli`` returns ``Inf``/``NaN``/``0``.
    * We use ``scipy.special.ive`` (exponentially scaled
      ``exp(-|x|) * I_nu(x)``) and form the ratio
      ``ive(D/2, k) / ive(D/2-1, k)`` so the ``exp(-x)`` factors cancel
      and the ratio stays well-conditioned in the range we see
      (``kappa ~ 100..2000``).
    * The "is besseli pathological?" probe before fzero in MATLAB is
      reproduced via the unscaled ``iv`` (it overflows to ``Inf`` —
      that's the same trigger MATLAB uses) so we take the same fallback
      branch in the same regime. This is necessary for behaviour parity
      because the saved ``epsil`` depends on whether MATLAB went down
      the fzero or the closed-form path.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize
from scipy import special as _spec


def _ad(kappa: float, D: float) -> float:
    """``A_d(kappa) = I_{D/2}(k) / I_{D/2-1}(k)`` via ``ive`` (stable).

    The exponentially scaled Bessels share the same ``exp(-|x|)`` factor,
    which cancels in the ratio, so we get a finite result for any
    ``kappa > 0`` while ``iv`` itself overflows past ``kappa ~ 700``.
    """
    nu_top = D / 2.0
    nu_bot = D / 2.0 - 1.0
    top = _spec.ive(nu_top, kappa)
    bot = _spec.ive(nu_bot, kappa)
    if bot == 0.0 or not np.isfinite(top) or not np.isfinite(bot):
        # Both overflow / underflow regimes: fall back to large-kappa
        # asymptotic A_d(k) ~ 1 - (D-1)/(2k) (DLMF 10.41 / Banerjee 2005
        # eq. (4.4)) so the residual still has the right sign for the
        # bracket search.
        return 1.0 - (D - 1.0) / (2.0 * kappa)
    return float(top / bot)


def invAd(D: float, rbar: float) -> float:
    """Inverse of ``A_d`` — vMF concentration estimate for mean cosine ``rbar``.

    Mirrors MATLAB ``CBIG_ArealMSHBM_invAd``:

        outu = (D-1)*rbar/(1-rbar^2) + D/(D-1)*rbar      # warm start
        if besseli(D/2-1, outu) is Inf/NaN/0:
            return outu - D/(D-1)*rbar/2                  # closed form
        else:
            try fzero on (A_d(k) - rbar) starting at outu
            return fzero result if converged else closed form

    Parameters
    ----------
    D    : feature dimension (positive int or float >= 2).
    rbar : scalar mean cosine in (0, 1).

    Returns
    -------
    kappa : float — vMF concentration. fp64 throughout.
    """
    D = float(D)
    rbar = float(rbar)

    # MATLAB warm start.
    outu = (D - 1.0) * rbar / (1.0 - rbar * rbar) + D / (D - 1.0) * rbar
    closed_form = outu - D / (D - 1.0) * rbar / 2.0

    # MATLAB's pathology probe: unscaled besseli at the warm start.
    # ive is stable so to trigger MATLAB's fallback path we explicitly
    # query the unscaled iv and check for the same Inf/NaN/0 conditions.
    nu_probe = D / 2.0 - 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        iprobe = _spec.iv(nu_probe, outu)
    if (not np.isfinite(iprobe)) or (iprobe == 0.0):
        return float(closed_form)

    # fzero counterpart: bracket the root (A_d(k) - rbar) around outu and
    # polish with brentq. A_d is monotone increasing in k, so we expand
    # outward from outu until we straddle zero.
    def f(k: float) -> float:
        return _ad(k, D) - rbar

    f_outu = f(outu)
    if f_outu == 0.0 or math.isnan(f_outu):
        # Either exact (degenerate) or unrecoverable — match MATLAB
        # fallback for the NaN case (MATLAB's fzero would set
        # exitflag != 1).
        return float(outu) if f_outu == 0.0 else float(closed_form)

    # Walk a bracket out by doubling. f is monotone in k, so signs flip
    # at most once. Caps mirror MATLAB's silent failure → closed form.
    lo: float
    hi: float
    if f_outu > 0.0:
        # Need lower endpoint with f<0.
        hi = outu
        lo = max(outu * 0.5, 1e-6)
        for _ in range(200):
            if f(lo) < 0.0:
                break
            lo *= 0.5
            if lo < 1e-12:
                return float(closed_form)
        else:
            return float(closed_form)
    else:
        lo = outu
        hi = max(outu * 2.0, outu + 1.0)
        for _ in range(200):
            if f(hi) > 0.0:
                break
            hi *= 2.0
            if hi > 1e15:
                return float(closed_form)
        else:
            return float(closed_form)

    try:
        sol = optimize.root_scalar(
            f, bracket=(lo, hi), method="brentq", xtol=1e-9, rtol=1e-12,
            maxiter=200,
        )
    except (ValueError, RuntimeError):
        return float(closed_form)
    if not sol.converged:
        return float(closed_form)
    return float(sol.root)


__all__ = ["invAd"]
