"""_kernels.py

Numba kernels for the vMF init-params supercall. Four kernels — each
appears exactly once.

    _zero_mw_and_detect_nonzero_kernel(profile, medial_mask, OUT keep)
        Per-row pass: if medial, write zeros and clear keep[i]; else
        scan the row and set keep[i] = (any entry != 0). Folds the
        ``profile[medial] = 0`` write and ``np.abs(profile).sum(1) != 0``
        detection into one traversal.

    _demean_l2norm_inplace_kernel(profile, keep)
        Per-row demean + L2 normalize, in place. Reductions in fp64,
        storage at the input dtype. Skips rows where keep is False.

    _groupsum_kernel(profile, labels, L, OUT mtc)
        ``mtc = profile.T @ one_hot(labels)`` exploiting the one-hot
        structure: per parcel l, sum the rows where ``labels==l``.
        Same value as the dense GEMM, O(N·D) instead of O(N·L·D).

    _epsil_input_kernel(inner, labels) -> scalar
        ``Σ_i inner[i, labels[i]-1]`` for ``labels[i] > 0`` — the
        one-hot collapse of ``Σ_i,l one_hot[i,l] * inner[i,l]``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import numba
from numba import njit, prange


@njit(cache=True, parallel=True)
def _zero_mw_and_detect_nonzero_kernel(profile_NxD, medial_mask_N, keep_mask_N):
    N = profile_NxD.shape[0]
    D = profile_NxD.shape[1]
    for i in prange(N):
        if medial_mask_N[i]:
            for j in range(D):
                profile_NxD[i, j] = 0.0
            keep_mask_N[i] = False
        else:
            any_nz = False
            for j in range(D):
                if profile_NxD[i, j] != 0.0:
                    any_nz = True
                    break
            keep_mask_N[i] = any_nz


@njit(cache=True, parallel=True)
def _demean_l2norm_inplace_kernel(profile_NxD, keep_mask_N):
    N = profile_NxD.shape[0]
    D = profile_NxD.shape[1]
    inv_D = 1.0 / D
    for i in prange(N):
        if not keep_mask_N[i]:
            continue
        s = 0.0
        for j in range(D):
            s += profile_NxD[i, j]
        m = s * inv_D
        ss = 0.0
        for j in range(D):
            d = profile_NxD[i, j] - m
            ss += d * d
        inv_norm = 1.0 / np.sqrt(ss) if ss > 0.0 else 1.0
        for j in range(D):
            profile_NxD[i, j] = (profile_NxD[i, j] - m) * inv_norm


@njit(cache=True, parallel=True)
def _groupsum_kernel(profile_NxD, labels_N, L, mtc_DxL):
    D = profile_NxD.shape[1]
    N = profile_NxD.shape[0]
    for d in range(D):
        for l in range(L):
            mtc_DxL[d, l] = 0.0
    # prange over D — each thread owns one feature column, no race.
    for d in prange(D):
        for i in range(N):
            l = labels_N[i]
            if l > 0:
                mtc_DxL[d, l - 1] += profile_NxD[i, d]


@njit(cache=True, parallel=True)
def _epsil_input_kernel(inner_NxL, labels_N):
    N = inner_NxL.shape[0]
    T = numba.get_num_threads()
    parts = np.zeros(T, dtype=np.float64)
    for i in prange(N):
        tid = numba.get_thread_id()
        l = labels_N[i]
        if l > 0:
            parts[tid] += inner_NxL[i, l - 1]
    s = 0.0
    for t in range(T):
        s += parts[t]
    return s
