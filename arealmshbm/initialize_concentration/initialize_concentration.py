"""initialize_concentration.py

vMF concentration parameter initial value ``ini_val``. Solves
``|I_{D/2-1}(κ)| == 1e10`` for a given feature dimension D; the result
seeds ``Params.kappa`` at the start of every ``intra_em`` iteration.

Implementation note:
    ``scipy.special.iv(nu, x)`` overflows fp64 around ``x ≈ 700`` for
    the typical D=1175 (ν ≈ 587), and ``scipy.special.ive`` underflows
    for small x at large ν. We compute ``log(I_ν(x))`` directly via the
    Debye asymptotic expansion (DLMF 10.41) — same closed form as
    :mod:`arealmshbm.em_stop_criterion._cdln`, kept dependency-free
    here for clarity.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

from scipy import optimize


def _log_besseli_debye(nu: float, x: float) -> float:
    """``log(I_nu(x))`` via the 5-term Debye asymptotic expansion (DLMF 10.41).

    ~1e-15 fp64 precision at nu >= 25 for any positive x. Equivalent to
    the implementation in ``em_stop_criterion/_cdln.py`` —
    re-implemented here without the numba decorators so this module is
    importable without compiling a kernel.
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
    correction = (1.0 + u1*inv_nu + u2*inv_nu*inv_nu
                  + u3*inv_nu**3 + u4*inv_nu**4 + u5*inv_nu**5)

    log_iv = nu * eta - 0.5 * (math.log(2.0 * math.pi) + math.log(nu)) \
             - 0.25 * math.log(1.0 + z2)
    if correction > 0.0:
        log_iv += math.log(correction)
    return log_iv


def initialize_concentration(D: float) -> float:
    """Find ``kappa`` solving ``|I_{D/2 - 1}(kappa)| = 1e10``.

    Mirrors MATLAB ``CBIG_ArealMSHBM_initialize_concentration``:
    bracket-search + Brent polish, but in log-space using the Debye
    expansion so it's stable where MATLAB's direct ``besseli`` form
    overflows.

    Parameters
    ----------
    D : float — feature dimension. ``setting_params.dim + 1`` in step3
                (e.g. 1175 for fsaverage6 RSFC profiles).

    Returns
    -------
    kappa : float — initial concentration value.
    """
    nu = float(D) / 2.0 - 1.0
    log_target = math.log(1e10)

    def f(x: float) -> float:
        return _log_besseli_debye(nu, x) - log_target

    # log(I_nu(x)) is monotone increasing for x > 0; bracket = (1, 2*nu);
    # double hi until f(hi) > 0.
    lo = 1.0
    hi = max(50.0, 2.0 * nu)
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError(
                f"Cannot bracket initialize_concentration root for D={D}"
            )

    sol = optimize.root_scalar(
        f, bracket=(lo, hi), method="brentq", xtol=1e-9, rtol=1e-12,
    )
    if not sol.converged:
        raise RuntimeError(
            f"initialize_concentration failed to converge for D={D}"
        )
    return float(sol.root)
