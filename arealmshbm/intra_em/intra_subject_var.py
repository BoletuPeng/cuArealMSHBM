"""intra_subject_var.py

Update ``s_psi`` from ``s_t_nu``, ``sigma``, ``epsil``, ``mu``. Runs
once per outer ``intra_em`` iteration, between the EM body and
``intra_em_cost``.

Math (single-subject, no S axis):

    s_psi_update[d, l] = Σ_t (σ[l] · s_t_nu[d, l, t]) + ε[l] · μ[d, l]
    s_psi_update      /= sqrt(Σ_d s_psi_update²)        # column-normalize

The MATLAB code also computes a per-subject convergence flag
``flag_psi(s, 1)``; the outer driver does not act on it, so we drop it.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np


def intra_subject_var(s_t_nu: np.ndarray,
                      sigma: np.ndarray,
                      epsil: np.ndarray,
                      mu: np.ndarray) -> np.ndarray:
    """Update ``s_psi`` from ``s_t_nu``, ``sigma``, ``epsil``, ``mu``.

    Parameters
    ----------
    s_t_nu : (D, L, T) float — per-session vMF mean directions.
    sigma  : (L,) or (1, L) float — intra-subject concentration.
    epsil  : (L,) or (1, L) float — inter-subject concentration.
    mu     : (D, L) float — group-level cluster centroids.

    Returns
    -------
    s_psi  : (D, L) fp32 — updated subject-level cluster centroids.
    """
    nu = np.ascontiguousarray(s_t_nu, dtype=np.float32)
    sig = np.ascontiguousarray(np.asarray(sigma).ravel(), dtype=np.float32)
    eps = np.ascontiguousarray(np.asarray(epsil).ravel(), dtype=np.float32)
    mu_f = np.ascontiguousarray(mu, dtype=np.float32)

    if nu.ndim != 3:
        raise ValueError(f"s_t_nu must be 3D (D, L, T); got {s_t_nu.shape}")
    D, L, T = nu.shape
    if sig.shape[0] != L:
        raise ValueError(f"sigma length {sig.shape[0]} != L={L}")
    if eps.shape[0] != L:
        raise ValueError(f"epsil length {eps.shape[0]} != L={L}")
    if mu_f.shape != (D, L):
        raise ValueError(f"mu shape {mu_f.shape} != ({D}, {L})")

    # Σ_t (sigma[l] * s_t_nu[d, l, t])  =  sigma[l] * Σ_t s_t_nu[..., t].
    summed_nu = nu.sum(axis=2)             # (D, L) fp32
    s_psi_update = summed_nu * sig[None, :] + mu_f * eps[None, :]   # (D, L)

    # Column-wise unit-normalize.
    col_norms = np.sqrt((s_psi_update * s_psi_update).sum(axis=0, keepdims=True))
    # Avoid divide-by-zero — match MATLAB which would emit Inf/NaN; keep zeros
    # in zero-norm columns instead so caller sees deterministic output.
    safe = np.where(col_norms > 0, col_norms, 1.0)
    s_psi_new = s_psi_update / safe
    s_psi_new[:, col_norms[0] == 0] = 0.0
    return s_psi_new
