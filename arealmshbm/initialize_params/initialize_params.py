"""initialize_params.py

Build the per-subject ``Params`` dict that step 3's ``intra_em`` loop
consumes (single-subject — no S axis):

    mu                : from group prior, (D, L)
    epsil             : from group prior, (L,) — inter-subject vMF κ
    sigma             : from group prior, (L,) — intra-subject vMF κ
    theta             : from group prior, (N, L) — spatial prior P(parcel | vertex)
    s_psi             : init = mu, (D, L)
    kappa             : init = ini_val, (L,) — inter-region vMF κ
    s_t_nu            : init = mu broadcast over T, (D, L, T)
    s_lambda          : init = theta, (N, L)
    xyz_gamma         : init = zeros, (L,) — distributed-parcel accumulator
    max_connectedness, max_components = 0
    spatial_xyz_vmf   : init = zeros, (N, L)

1-D vectors are stored flat as ``(L,)`` rather than MATLAB's ``(1, L)``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def initialize_params(group_prior: Dict[str, np.ndarray],
                      ini_val: float,
                      num_session: int,
                      num_clusters: int,
                      num_verts: int) -> Dict[str, Any]:
    """Build the initial ``Params`` dict (Mode A).

    Parameters
    ----------
    group_prior : dict — output of :func:`data_io.load_group_prior`. Keys
                  ``mu`` (D, L), ``theta`` (N, L), ``epsil`` (1, L) or (L,),
                  ``sigma`` (1, L) or (L,).
    ini_val     : float — initial concentration value (from
                  :func:`initialize_concentration.initialize_concentration`).
    num_session : int — number of sessions T.
    num_clusters: int — number of clusters / parcels L.
    num_verts   : int — number of vertices N (bilateral).

    Returns
    -------
    Params : dict — keys listed above. Mode A — no S axis.
    """
    mu = np.ascontiguousarray(group_prior["mu"], dtype=np.float32)
    theta = np.ascontiguousarray(group_prior["theta"], dtype=np.float32)
    epsil = np.ascontiguousarray(np.asarray(group_prior["epsil"]).ravel(),
                                 dtype=np.float32)
    sigma = np.ascontiguousarray(np.asarray(group_prior["sigma"]).ravel(),
                                 dtype=np.float32)

    L = int(num_clusters)
    T = int(num_session)
    N = int(num_verts)
    if mu.shape[1] != L:
        raise ValueError(f"mu cols {mu.shape[1]} != num_clusters={L}")
    if theta.shape != (N, L):
        raise ValueError(f"theta shape {theta.shape} != ({N}, {L})")
    if epsil.shape != (L,) or sigma.shape != (L,):
        raise ValueError(
            f"epsil/sigma shape {epsil.shape}/{sigma.shape} != ({L},)"
        )
    D = mu.shape[0]

    Params: Dict[str, Any] = {}
    Params["mu"] = mu
    Params["epsil"] = epsil
    Params["sigma"] = sigma
    Params["theta"] = theta

    # s_psi: init = mu.
    Params["s_psi"] = mu.copy()

    # kappa: init = ini_val * ones(L,) — fp64 (matches MATLAB precision).
    Params["kappa"] = float(ini_val) * np.ones(L, dtype=np.float64)

    # s_t_nu: init = repmat(mu, 1, 1, T). Shape (D, L, T).
    Params["s_t_nu"] = np.broadcast_to(mu[..., None], (D, L, T)).copy()

    # s_lambda: init = theta. Shape (N, L).
    Params["s_lambda"] = theta.copy()

    # xyz_gamma: init = zeros(L,) — fp64 (accumulator gets +1000 per call).
    Params["xyz_gamma"] = np.zeros(L, dtype=np.float64)

    Params["max_connectedness"] = 0.0
    Params["max_components"] = 0.0

    # spatial_xyz_vmf: init = zeros(N, L).
    Params["spatial_xyz_vmf"] = np.zeros((N, L), dtype=np.float32)

    Params["iter_inter"] = 1.0   # MATLAB sets this before EM; carried for parity.

    return Params
