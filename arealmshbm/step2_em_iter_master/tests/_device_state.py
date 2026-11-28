"""_device_state.py — host copies of a Step2SparseSession's device state.

The kernel and session tests compare the Session's device buffers against
the CPU reference; :func:`sync_to_host` reads them by name.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from arealmshbm.step2_io.tests._layout_oracle import scatter

# name -> Session attribute copied verbatim.
_DEVICE_FIELDS = {
    "s_t_nu": "_s_t_nu",
    "s_psi": "_s_psi",
    "sigma": "_sigma",
    "epsil": "_epsil",
    "mu": "_mu",
    "s_lambda_P": "_s_lambda",
    "theta_P": "_theta",
    "log_theta": "_log_theta",
    "lv_sum": "_lv_sum",
    "log_vmf": "_log_vmf",
    "log_connect": "_log_connect",
    "scr": "_scr",
    "rmax": "_rmax",
    "rmax_dense": "_rmax_dense",
    "u": "_u_LD",
    "u_sq": "_u_sq",
    "grad_sq": "_grad_sq",
    "n_alive": "_n_alive",
    "row_mean": "_row_mean",
    "row_inv": "_row_inv",
    "active": "_active",
}


def sync_to_host(sess, fields: Iterable[str]) -> Dict[str, np.ndarray]:
    """Copy the named device arrays of ``sess`` to fresh host arrays.

    ``theta`` / ``s_lambda`` come back scattered to dense ``(N, L)``, their
    ``*_P`` twins in P layout; ``cost_em`` is the last ``run_iter``'s
    ``(S,)`` cost and ``kappa`` the last M-step's concentration. An unknown
    name raises ``ValueError``.
    """
    import cupy as cp

    lay = sess._inputs.layout
    out: Dict[str, np.ndarray] = {}
    for f in fields:
        if f in _DEVICE_FIELDS:
            out[f] = cp.asnumpy(getattr(sess, _DEVICE_FIELDS[f]))
        elif f == "s_lambda":
            out[f] = scatter(lay, cp.asnumpy(sess._s_lambda))
        elif f == "theta":
            out[f] = scatter(lay, cp.asnumpy(sess._theta))
        elif f == "cost_em":
            out[f] = sess._cost_host[:sess.S].copy()
        elif f == "kappa":
            out[f] = np.float64(sess._kappa_final)
        else:
            raise ValueError(f"sync_to_host: unknown field {f!r}")
    return out
