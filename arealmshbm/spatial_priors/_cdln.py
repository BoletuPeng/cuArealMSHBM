"""_cdln.py

Closed-form ``Cdln(k, d=3)`` — the log-vMF normalization specialized to
3D vectors on S². ``spatial_xyz_prior`` is the only consumer; it always
calls with d=3, so the generic Bessel route the MATLAB original takes
is unnecessary.

For d=3, ``I_{1/2}(k) = sqrt(2/(πk)) · sinh(k)`` gives the closed form:

    Cdln(k, 3) = log k - k + 0.5·log(2π) - log(1 - exp(-2k))

For k > ~30 the ``log(1 - exp(-2k))`` term is below fp32 epsilon and is
dropped. For 0 < k ≤ 30 we keep it via ``log1p(-exp(-2k))`` for full
precision.

NaN handling at k=0:
    MATLAB's Cdln returns NaN at k=0; we match that NaN here so
    ``Params.spatial_xyz_vmf[isnan(...)] = 0`` cleanup downstream
    behaves identically.

Output dtype is fp32.

Precision (audited against mpmath at 50 dps over the EM-realistic range
k ∈ {1000, 2000, 5000, 10000, 15000}):

    closed-form fp64                  : 5.6e-17  (0.25 × fp64 eps)
    closed-form fp32 (this output)    : 3.7e-08  (0.31 × fp32 eps)
    MATLAB CBIG_ArealMSHBM_Cdln       : 5.3e-07  (4.4  × fp32 ULP)

This Python output is up to 19× more accurate than MATLAB's at k=1000.
Any residual on ``spatial_xyz_vmf`` between Python and MATLAB is
dominated by MATLAB's drift, not ours — together with the general-d
Cdln in :mod:`arealmshbm.em_stop_criterion._cdln`, this is the
dominant source of the small (<1%) end-to-end vertex disagreement.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
import numba as nb


_LOG_2PI = 1.8378770664093454835606594728112  # log(2*pi), fp64 literal


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def cdln_d3_to_f32(k_arr, out):
    """Compute Cdln(k_arr[j], d=3) for each j; write fp32 to ``out``.

    Closed form for d=3 (uses ``I_{1/2}(k) = sqrt(2/(pi k)) * sinh(k)``):

        Cdln(k, 3) = log k - k + 0.5 log(2 pi) - log(1 - e^{-2k})

    For k > 30 the trailing term is below fp32 ULP and is skipped. For
    k = 0 we emit NaN to match MATLAB's behavior (downstream code does
    ``vmf(isnan(vmf)) = 0``).

    Inputs:
        k_arr : (L,) float64 read-only
    Outputs:
        out   : (L,) float32 rewritten
    """
    L = k_arr.shape[0]
    for j in range(L):
        k = k_arr[j]
        if k <= 0.0:
            out[j] = np.float32(np.nan)
        else:
            # log(1 - exp(-2k)) for numerical stability via log1p.
            # At k=30, exp(-60) ≈ 8.8e-27, so log1p(-tiny) ≈ -tiny ≈ 0
            # well below fp32 eps ≈ 1.2e-7. Skip past k=30.
            if k < 30.0:
                log_one_minus = math.log1p(-math.exp(-2.0 * k))
            else:
                log_one_minus = 0.0
            val = math.log(k) - k + _LOG_2PI * 0.5 - log_one_minus
            out[j] = np.float32(val)


def warmup() -> None:
    """Compile ``cdln_d3_to_f32`` with realistic dtypes/shapes once."""
    k = np.array([0.0, 1000.0], dtype=np.float64)
    out = np.empty(2, dtype=np.float32)
    cdln_d3_to_f32(k, out)
