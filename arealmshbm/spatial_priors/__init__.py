"""spatial_priors

Two spatial-prior terms added to ``log_vmf`` in the gMSHBM E-step:

  * ``spatial_xyz_prior``     — sphere-vMF compactness regularizer,
                                activated by ``check_connectedness`` via
                                ``Params.xyz_gamma``.
  * ``spatial_connect_prior`` — gradient-embedding squared-distance prior
                                (the β term, Gordon-2016 edge-detection style).

Both produce an ``(N, L)`` array.

Public API:
    XyzSession              — Session for spatial_xyz_prior.
    ConnectSession          — Session for spatial_connect_prior.
    compute_unit_sphere_xyz — load + row-normalize bilateral sphere coords;
                              shared with the GPU full-device super-call.
    warmup                  — pre-compile every numba kernel. Idempotent.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .spatial_xyz import (
    XyzSession,
    compute_unit_sphere_xyz,
)
from .spatial_connect import (
    ConnectSession,
)


def warmup() -> None:
    """Pre-compile every numba kernel in this module with realistic
    dtypes / shapes. ~50 ms one-shot."""
    from . import _cdln, _kernels
    _cdln.warmup()
    _kernels.warmup()


__all__ = [
    "XyzSession",
    "ConnectSession",
    "compute_unit_sphere_xyz",
    "warmup",
]
