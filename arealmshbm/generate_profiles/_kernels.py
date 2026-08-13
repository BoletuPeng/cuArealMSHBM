"""_kernels.py

Numba kernels for the per-session FC-profile leaf. Three kernels —
each appears exactly once.

    _zscore_unit_norm_columns_kernel(x_TxN, out_TxN)
        Per-column zero-mean + L2-unit-norm. Reductions in fp64,
        storage in fp32. Equivalent to
        ``(x - x.mean(0, fp64)) / sqrt(((x-mean)**2).sum(0, fp64))``.

    _nan_to_zero_kernel(x_2d)
        Replace NaN / ±Inf with 0.0 in place. (Constant-vertex BOLD
        produces a zero-variance column, which propagates to NaN in
        the correlation matrix; we want it to score zero, not blow up
        the threshold step.)

    _threshold_kernel(corr_KxV, threshold, out_KxV)
        Binarize ``(corr >= threshold).astype(fp32)`` with output in
        (K, V) C-contig layout — bit-identical to NIFTI's natural
        on-disk byte order, so the writer dumps the bytes directly
        without a transpose.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _zscore_unit_norm_columns_kernel(x_TxN, out_TxN):
    T = x_TxN.shape[0]
    N = x_TxN.shape[1]
    inv_T = 1.0 / T
    for j in prange(N):
        s = 0.0
        for i in range(T):
            s += x_TxN[i, j]
        m = s * inv_T
        ss = 0.0
        for i in range(T):
            d = float(x_TxN[i, j]) - m
            ss += d * d
        if ss > 0.0:
            inv_norm = np.float32(1.0 / np.sqrt(ss))
        else:
            inv_norm = np.float32(np.inf)
        m32 = np.float32(m)
        for i in range(T):
            out_TxN[i, j] = (x_TxN[i, j] - m32) * inv_norm


@njit(cache=True, parallel=True)
def _nan_to_zero_kernel(x_2d):
    M = x_2d.shape[0]
    N = x_2d.shape[1]
    for i in prange(M):
        for j in range(N):
            v = x_2d[i, j]
            if v != v or v == np.float32(np.inf) or v == np.float32(-np.inf):
                x_2d[i, j] = np.float32(0.0)


@njit(cache=True, parallel=True)
def _threshold_kernel(corr_KxV, threshold, out_KxV):
    K = corr_KxV.shape[0]
    V = corr_KxV.shape[1]
    thr = np.float32(threshold)
    for k in prange(K):
        for v in range(V):
            out_KxV[k, v] = np.float32(1.0) if corr_KxV[k, v] >= thr else np.float32(0.0)
