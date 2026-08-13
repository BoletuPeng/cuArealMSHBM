"""intra_em_cost.py

Outer-loop convergence cost — runs once per outer ``intra_em`` iteration,
after ``intra_subject_var``. Drives the test
``abs(abs(update_cost - cost) / cost) <= 1e-4``.

Math (single-subject, no S axis):

    term1 = Σ_{d, l, t} σ[l] · s_psi[d, l] · s_t_nu[d, l, t]
            + T · Σ_l Cdln(σ_l, dim)
    term2 = Σ_{d, l} ε[l] · μ[d, l] · s_psi[d, l]
            + Σ_l Cdln(ε_l, dim)

    update_cost = term1 + term2

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np

from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32


def intra_em_cost(s_psi: np.ndarray,
                  s_t_nu: np.ndarray,
                  mu: np.ndarray,
                  sigma: np.ndarray,
                  epsil: np.ndarray,
                  cost_em: float | np.ndarray,
                  dim: int) -> float:
    """Outer-loop cost for the ``intra_em`` convergence test.

    Parameters
    ----------
    s_psi   : (D, L) — subject cluster centroids.
    s_t_nu  : (D, L, T) — per-session cluster centroids.
    mu      : (D, L) — group cluster centroids.
    sigma   : (L,) or (1, L) — intra-subject concentration.
    epsil   : (L,) or (1, L) — inter-subject concentration.
    cost_em : last EM-stop cost (scalar or (1, 1)). MATLAB adds
              ``sum(Params.cost_em)`` into update_cost.
    dim     : int — vMF dimension parameter (= D - 1).

    Returns
    -------
    update_cost : float — outer-loop convergence metric.
    """
    psi = np.ascontiguousarray(s_psi, dtype=np.float32)
    nu = np.ascontiguousarray(s_t_nu, dtype=np.float32)
    mu_f = np.ascontiguousarray(mu, dtype=np.float32)
    sig = np.ascontiguousarray(np.asarray(sigma).ravel(), dtype=np.float32)
    eps = np.ascontiguousarray(np.asarray(epsil).ravel(), dtype=np.float32)
    ce = np.asarray(cost_em).ravel()

    if psi.ndim != 2:
        raise ValueError(f"s_psi must be 2D (D, L); got {s_psi.shape}")
    D, L = psi.shape
    if nu.ndim != 3 or nu.shape[:2] != (D, L):
        raise ValueError(f"s_t_nu must be 3D (D, L, T); got {s_t_nu.shape}")
    T = nu.shape[2]
    if mu_f.shape != (D, L) or sig.shape[0] != L or eps.shape[0] != L:
        raise ValueError(
            "shape mismatch among s_psi, s_t_nu, mu, sigma, epsil"
        )

    # Cdln(sigma, dim) and Cdln(epsil, dim) — fp64 internal, cast to fp32.
    cdln_sigma = np.empty(L, dtype=np.float32)
    cdln_epsil = np.empty(L, dtype=np.float32)
    cdln_general_to_f32(sig.astype(np.float64), int(dim), cdln_sigma)
    cdln_general_to_f32(eps.astype(np.float64), int(dim), cdln_epsil)

    # term1 = sigma · Σ_d s_psi[d,l] · Σ_t s_t_nu[d,l,t]  +  T · Σ_l Cdln(sigma_l)
    #
    # Algebraic collapse of the MATLAB ``bsxfun + sum-over-(d,l,t)`` reduction:
    # the summand factors as ``sigma[l] · s_psi[d,l] · s_t_nu[d,l,t]``, so we
    # can pre-sum over t (giving ``summed_nu[d,l] = Σ_t s_t_nu[d,l,t]``) and
    # then per-column-dot ``s_psi[:,l] · summed_nu[:,l]``. Avoids materializing
    # the (D, L, T) intermediate (~16 MB fp64) that the literal MATLAB
    # translation produced. Math is identical; production inputs (s_psi /
    # s_t_nu unit-norm columns) don't change the algebra.
    summed_nu = nu.sum(axis=2)                              # (D, L) fp32
    per_col1 = (psi.astype(np.float64) * summed_nu.astype(np.float64)
                ).sum(axis=0)                               # (L,) fp64
    term1_data = float((sig.astype(np.float64) * per_col1).sum())
    term1_cdln = float(T) * float(cdln_sigma.astype(np.float64).sum())
    term1 = term1_data + term1_cdln

    # term2 = epsil · Σ_d mu[d,l] · s_psi[d,l]  +  Σ_l Cdln(epsil_l)
    per_col2 = (mu_f.astype(np.float64) * psi.astype(np.float64)
                ).sum(axis=0)                               # (L,) fp64
    term2_data = float((eps.astype(np.float64) * per_col2).sum())
    term2_cdln = float(cdln_epsil.astype(np.float64).sum())
    term2 = term2_data + term2_cdln

    update_cost = term1 + term2 + float(ce.sum())
    return update_cost
