"""inter_subject_var.py — L18.

Closed-form ``(mu, epsil)`` update from the per-subject ``s_psi``.

Math (internal (S, L, D) / (L, D) layout):

    mu_update[l, d]   = Σ_s s_psi[s, l, d]                          (L, D)
    mu_new[l, :]      = mu_update[l, :] / ||mu_update[l, :]||₂

    epsil_input[l]    = Σ_{s, d} s_psi[s, l, d] · mu_new[l, d] / S  (L,)
    epsil_input       = min(epsil_input, 1.0)
    epsil_new[l]      = invAd(dim, epsil_input[l])  with the ini_val /
                        inf fallbacks from MATLAB.

Inf fallback uses the *previous* mu / epsil at that slot, mirroring the
MATLAB ``Params.epsil(i)`` reference.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np

from ._kernels import (
    _inter_subject_var_epsil_update,
    _inter_subject_var_mu_update,
)


def inter_subject_var(
    s_psi: np.ndarray,
    prev_mu: np.ndarray,
    prev_epsil: np.ndarray,
    dim: int,
    ini_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form group ``(mu, epsil)`` update.

    Parameters
    ----------
    s_psi      : (S, L, D) fp32 — converged per-subject cluster centroids.
    prev_mu    : (L, D)    fp32 — fall-back for zero-norm μ columns.
    prev_epsil : (1, L) or (L,) fp32 — inf-fallback for ``invAd`` failures.
    dim        : int — vMF dimension (= setting_params.dim).
    ini_val    : float — concentration floor (Params.ini_val).

    Returns
    -------
    mu_new    : (L, D) fp32
    epsil_new : (1, L) fp32
    """
    psi = np.ascontiguousarray(s_psi, dtype=np.float32)
    if psi.ndim != 3:
        raise ValueError(f"s_psi must be (S, L, D); got {s_psi.shape}")
    S, L, D = psi.shape

    prev_mu_f = np.ascontiguousarray(prev_mu, dtype=np.float32)
    if prev_mu_f.shape != (L, D):
        raise ValueError(f"prev_mu shape {prev_mu_f.shape} != (L={L}, D={D})")
    prev_eps_L = np.ascontiguousarray(
        np.asarray(prev_epsil).ravel(), dtype=np.float32,
    )
    if prev_eps_L.shape[0] != L:
        raise ValueError(f"prev_epsil length {prev_eps_L.shape[0]} != L={L}")

    mu_new = np.empty((L, D), dtype=np.float32)
    mu_norms = np.empty(L, dtype=np.float32)
    _inter_subject_var_mu_update(psi, prev_mu_f, mu_new, mu_norms)

    epsil_new_L = np.empty(L, dtype=np.float32)
    _inter_subject_var_epsil_update(
        psi, mu_new, prev_eps_L, epsil_new_L,
        float(dim), float(ini_val),
    )

    return mu_new, epsil_new_L.reshape(1, L)
