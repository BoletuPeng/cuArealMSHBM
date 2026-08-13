"""intra_subject_var_loop.py — L17.

Multi-subject while-loop wrapping the step-3 single-subject ``s_psi``
update with an additional ``sigma`` (intra-subject concentration)
update at the end of each iteration.

Per-iteration math (internal (S, L, D) / (S, T, L, D) / (L, D) layout):

    s_psi_update[s, l, d] = sigma[l] · Σ_t s_t_nu[s, t, l, d]
                            + ε[l] · μ[l, d]
    s_psi_update          /= sqrt(Σ_d s_psi_update²)         (column-norm)

    sigma_input[l] = Σ_{s, t, d} s_psi[s, l, d] · s_t_nu[s, t, l, d]
                     / (S · T)
    sigma_input    = min(sigma_input, 1.0)
    sigma_new[l]   = invAd(dim, sigma_input[l])  with the ini_val / inf
                     fallbacks from MATLAB.

Convergence — both must hold:

    1. ``flag_psi[s] == 1 for all s`` — per-subject diag(s_psi_newᵀ ·
       s_psi_prev) close-to-1 on every parcel.
    2. ``mean_l |sigma_prev[l] - sigma_new[l]| / sigma_prev[l] < epsilon``.

``flag_psi`` accumulates across iterations (MATLAB never zeros it on a
new iter — once a subject converges its bit stays set). We mirror that.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np

from ._kernels import _intra_subject_var_loop_kernel


def intra_subject_var_loop(
    s_t_nu: np.ndarray,
    s_psi_in: np.ndarray,
    sigma_in: np.ndarray,
    epsil: np.ndarray,
    mu: np.ndarray,
    dim: int,
    ini_val: float,
    epsilon: float = 1e-4,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loop the (s_psi, sigma) update until convergence or ``max_iter``.

    Parameters
    ----------
    s_t_nu  : (S, T, L, D) fp32 — per-session vMF mean directions.
    s_psi_in: (S, L, D) fp32 — initial s_psi (used as ``prev`` term in
              iter-1's flag check).
    sigma_in: (1, L) or (L,) fp32 — initial intra-subject concentration.
    epsil   : (1, L) or (L,) fp32 — current inter-subject concentration.
    mu      : (L, D) fp32 — current group-level cluster centroids.
    dim     : int — vMF dimension (= setting_params.dim, e.g. 1174).
    ini_val : float — concentration floor (Params.ini_val).
    epsilon : float — both s_psi and sigma convergence tolerances.
    max_iter: int — hard iteration cap.

    Returns
    -------
    s_psi_new : (S, L, D) fp32
    sigma_new : (1, L) fp32
    flag_psi  : (S,) int32  — 1 where subject ``s`` satisfied the s_psi
                 convergence in at least one iter.
    """
    nu = np.ascontiguousarray(s_t_nu, dtype=np.float32)
    if nu.ndim != 4:
        raise ValueError(f"s_t_nu must be (S, T, L, D); got {s_t_nu.shape}")
    S, T, L, D = nu.shape

    psi_in = np.ascontiguousarray(s_psi_in, dtype=np.float32)
    if psi_in.shape != (S, L, D):
        raise ValueError(
            f"s_psi_in shape {psi_in.shape} != (S={S}, L={L}, D={D})"
        )

    sigma_in_L = np.ascontiguousarray(
        np.asarray(sigma_in).ravel(), dtype=np.float32,
    )
    if sigma_in_L.shape[0] != L:
        raise ValueError(f"sigma_in length {sigma_in_L.shape[0]} != L={L}")

    epsil_L = np.ascontiguousarray(np.asarray(epsil).ravel(), dtype=np.float32)
    if epsil_L.shape[0] != L:
        raise ValueError(f"epsil length {epsil_L.shape[0]} != L={L}")

    mu_f = np.ascontiguousarray(mu, dtype=np.float32)
    if mu_f.shape != (L, D):
        raise ValueError(f"mu shape {mu_f.shape} != (L={L}, D={D})")

    # Outputs.
    psi_new = np.empty((S, L, D), dtype=np.float32)
    sigma_new_L = np.empty(L, dtype=np.float32)
    flag_psi = np.zeros(S, dtype=np.int32)

    # Scratch.
    psi_prev = np.empty((S, L, D), dtype=np.float32)
    eps_mu = np.empty((L, D), dtype=np.float32)
    sigma_cur = np.empty(L, dtype=np.float32)
    sigma_tmp = np.empty(L, dtype=np.float32)
    cos_per_LS = np.empty((L, S), dtype=np.float32)

    _intra_subject_var_loop_kernel(
        nu, psi_in, sigma_in_L, epsil_L, mu_f,
        psi_new, sigma_new_L, flag_psi,
        psi_prev, eps_mu, sigma_cur, sigma_tmp, cos_per_LS,
        float(dim), float(ini_val), np.float32(epsilon), int(max_iter),
    )

    return psi_new, sigma_new_L.reshape(1, L), flag_psi
