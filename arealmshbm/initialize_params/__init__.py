"""initialize_params

Build the per-subject ``Params`` struct fed into the EM body.

Public API:
    initialize_params(group_prior, ini_val, num_session, num_clusters,
                      num_verts) -> dict
        — Params struct (mu, theta, epsil, sigma, kappa, s_psi, s_t_nu,
          s_lambda, ...) for one subject.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .initialize_params import initialize_params

__all__ = ["initialize_params"]
