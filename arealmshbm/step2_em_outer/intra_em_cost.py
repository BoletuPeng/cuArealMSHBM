"""intra_em_cost.py — L16.

Outer-loop convergence cost for step-2's intra-EM loop. Multi-subject
form of step-3's ``intra_em.intra_em_cost``.

Math (S subjects, L parcels, D ambient, T sessions):

    term1_data = Σ_l σ[l] · (Σ_{s,t,d} ψ[s,l,d] · ν[s,t,l,d])
    term1_cdln = S · T · Σ_l Cdln(σ_l, dim)
    term2_data = Σ_l ε[l] · (Σ_{s,d} ψ[s,l,d] · μ[l,d])
    term2_cdln = S · Σ_l Cdln(ε_l, dim)
    update_cost = term1 + term2 + Σ_s cost_em[s]

**Internal layout invariant**: this leaf consumes Params in the
internal (S, L, D) / (S, T, L, D) / (L, D) layout that the rest of
the intra-EM scope uses. There is no per-call layout conversion
inside this leaf.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ._kernels import _intra_em_cost_step2_kernel


def intra_em_cost_step2(params: Mapping[str, Any], dim: int) -> float:
    """Outer-loop cost for the step-2 intra-EM convergence test.

    Parameters
    ----------
    params : mapping with these keys (numpy arrays, fp32 unless noted):
        * ``s_psi``   — (S, L, D)        fp32
        * ``s_t_nu``  — (S, T, L, D)     fp32
        * ``mu``      — (L, D)           fp32
        * ``sigma``   — (1, L) or (L,)   fp32
        * ``epsil``   — (1, L) or (L,)   fp32
        * ``cost_em`` — (S,) or (1, S)   numeric (cast to fp64)
    dim    : int — vMF dimension parameter (e.g. 1174).

    Returns
    -------
    update_cost : float — the value MATLAB computes for the outer-EM
                          convergence check.
    """
    psi = np.ascontiguousarray(params["s_psi"], dtype=np.float32)
    nu = np.ascontiguousarray(params["s_t_nu"], dtype=np.float32)
    mu_f = np.ascontiguousarray(params["mu"], dtype=np.float32)
    sig = np.ascontiguousarray(np.asarray(params["sigma"]).ravel(), dtype=np.float32)
    eps = np.ascontiguousarray(np.asarray(params["epsil"]).ravel(), dtype=np.float32)
    cost_em = np.ascontiguousarray(np.asarray(params["cost_em"]).ravel(), dtype=np.float64)

    if psi.ndim != 3:
        raise ValueError(f"s_psi must be (S, L, D); got {psi.shape}")
    S, L, D = psi.shape
    if nu.ndim != 4 or nu.shape[0] != S or nu.shape[2] != L or nu.shape[3] != D:
        raise ValueError(
            f"s_t_nu must be (S, T, L, D); got {nu.shape} vs s_psi {psi.shape}"
        )
    if mu_f.shape != (L, D):
        raise ValueError(f"mu shape {mu_f.shape} != ({L}, {D})")
    if sig.shape[0] != L or eps.shape[0] != L:
        raise ValueError(
            f"sigma / epsil length mismatch: sigma={sig.shape[0]}, "
            f"epsil={eps.shape[0]}, L={L}"
        )
    if cost_em.shape[0] != S:
        raise ValueError(f"cost_em length {cost_em.shape[0]} != S={S}")

    return float(_intra_em_cost_step2_kernel(
        psi, nu, mu_f, sig, eps, cost_em, float(dim),
    ))
