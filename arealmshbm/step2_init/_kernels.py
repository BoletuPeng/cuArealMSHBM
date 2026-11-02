"""_kernels.py — step-2 init leaves numba kernels.

Two kernels backing the production ``compose_init_state`` super-call:
the port of the stock CBIG initialisation, i.e. the ``ini_s_lambda``
block of ``CBIG_ArealMSHBM_{g,d}MSHBM_estimate_group_priors_child.m``
and the mask / normalise / ``theta_num`` block of ``..._parent.m``.

* :func:`profile_to_soft_init_kernel` — per subject, the soft masked
  posterior under the group centroids::

      v[n, l]  = kappa_T * (mean_t X[n, t]) · g_mu[:, l]   for every l
      e[n, l]  = exp(v[n, l] - max_l v[n, :])              at in-mask cells
      s_lambda[n, l] = e[n, l] / sum_{l in mask} e[n, l]

  (stock: ``kappa_ini * sum_t X_t * g_mu``, then ``.* boundary_mask``
  and ``./ sum(., 2)``; stock also adds ``Cdln(kappa_ini, dim)`` to
  every logit once per session, a per-row constant that cancels in
  ``exp(v - rowmax)`` — the port leaves it out, and adding it back in
  fp32 leaves the final prior unchanged at the reported precision:
  94.07 % / 98.16 % vs stock MATLAB either way on the 8-subject gMSHBM
  cohort). Stock evaluates
  all of it in single precision,
  so the shifted logits are fp32, ``exp`` is rounded to fp32 with
  subnormals kept, and the row sum / divide are fp32: cells more than
  ~104 nats below the row max underflow to exactly 0, and a row whose
  in-mask cells all underflow is left all-zero (stock's ``0/0 -> NaN``
  rows, zeroed by ``mask_nan``). Medial rows (``||mean_t X|| == 0``)
  are all-zero (stock's ``mask``).

* :func:`compose_theta_kernel` — stock ``theta = mean(s_lambda, 3)``,
  ``theta(theta ~= 0) ./= theta_num`` (the number of subjects whose
  cell is nonzero), ``theta = theta + eps``: per cell, the fp32 sum
  over subjects divided by S and then AGAIN by the number of subjects
  nonzero there (stock's double division, kept as is), plus eps; cells
  no subject touches are ``f32(eps)``.

Reduction dtype: the ``mean_t X · g_mu`` accumulator is a private serial
ascending-d fp32 sum (no FMA, fp32 g_mu cast), the same evaluation order
as the GPU ``init_soft`` kernel, which is what keeps the two backends'
init bit-identical. Every later op is a native fp32 op or an fp64 op
rounded to fp32, mirrored one-for-one on the GPU (``f32_rn`` helpers).

Storage: profile is fp32 (matches caller). g_mu is fp64 input to the
public API (from group.mat); the wrapper casts to fp32 once on the
caller side. T-mean accumulator stays fp32 (matches MATLAB
``mean(data_series, 3)``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False, cache=True, boundscheck=False)
def profile_to_soft_init_kernel(profile: np.ndarray,
                                g_mu_f32: np.ndarray,
                                boundary_mask: np.ndarray,
                                kappa_T: np.float32,
                                s_lambda_out: np.ndarray) -> None:
    """Per-subject stock soft init row (single-precision semantics).

    Parameters
    ----------
    profile       : (N, T, D) fp32 C-contig — per-subject BOLD,
                    **NTD layout** (matches the unified step-2 SNTD
                    stack ``bold_SNTD[s]`` slice).
    g_mu_f32      : (D, L)   fp32 C-contig — group centroids
                    (cast from fp64 source by caller).
    boundary_mask : (N, L)   fp32 — bilateral block-diagonal radius mask,
                    a 0/1 indicator (the contract ``step2_io.sparse_layout``
                    enforces for the GPU backend): a cell is in-mask iff
                    nonzero, which is stock's ``.* boundary_mask`` on such
                    a mask.
    kappa_T       : fp32 scalar — ``f32(ini_val) * f32(T)`` (see
                    :func:`compose_init_state.init_kappa_T`); the stock
                    logits ``kappa_ini * sum_t`` evaluated as
                    ``mean_t * kappa_T``.
    s_lambda_out  : (N, L) fp32 — fully written: the masked, row-
                    normalised soft posterior; medial rows and rows
                    whose in-mask cells all underflow are all-zero.
    """
    N, T, D = profile.shape
    L = g_mu_f32.shape[1]
    inv_T = np.float32(1.0) / np.float32(T)
    zero = np.float32(0.0)

    for n in prange(N):
        # Per-n mean over t into a thread-local (D,) scratch.
        # For (N, T, D) layout: profile[n] is a (T, D) C-contig slab
        # (~9 KB at T=2/D=1175) that fits in L1 — per-thread access
        # of profile[n, t, :] for t in range(T) is fully cache-local.
        mean_d = np.zeros(D, dtype=np.float32)
        for t in range(T):
            row = profile[n, t]   # (D,) unit-stride view
            for d in range(D):
                mean_d[d] += row[d]

        acc = np.zeros(L, dtype=np.float32)
        norm_sq = zero
        for d in range(D):
            a = mean_d[d] * inv_T
            norm_sq += a * a
            for l in range(L):
                acc[l] += a * g_mu_f32[d, l]

        if norm_sq == zero:                       # medial wall: stock ``mask``
            for l in range(L):
                s_lambda_out[n, l] = zero
            continue

        # Shifted logits over ALL L (both hemispheres), like stock's
        # ``max(ini_log_vmf, [], 2)`` before the mask is applied.
        rowmax = acc[0] * kappa_T
        for l in range(L):
            v = acc[l] * kappa_T
            acc[l] = v
            if v > rowmax:
                rowmax = v

        # exp rounded to fp32 (subnormals kept), fp32 serial row sum in
        # ascending l over the in-mask cells.
        rs = zero
        for l in range(L):
            if boundary_mask[n, l] != zero:
                e = np.float32(np.exp(np.float64(acc[l] - rowmax)))
                s_lambda_out[n, l] = e
                rs += e
            else:
                s_lambda_out[n, l] = zero

        # rs == 0: every in-mask cell underflowed -> stock's 0/0 row, left 0.
        if rs > zero:
            rs_f64 = np.float64(rs)
            for l in range(L):
                if boundary_mask[n, l] != zero:
                    s_lambda_out[n, l] = np.float32(
                        np.float64(s_lambda_out[n, l]) / rs_f64)


@njit(parallel=True, fastmath=False, cache=True, boundscheck=False)
def compose_theta_kernel(s_lambda_SNL: np.ndarray,
                         theta: np.ndarray,
                         eps_f64: np.float64) -> None:
    """Stock ``Params.theta`` from the composed ``s_lambda``.

    Per (n, l), in single precision like the stock parent (note the two
    divisions: by S, then by the nonzero-subject count)::

        theta = sum_s s_lambda[s, n, l] / S            (mean(., 3))
        theta = theta / #{s : s_lambda[s, n, l] != 0}  (./ theta_num)
        theta = theta + eps

    and ``f32(eps)`` where no subject is nonzero (stock: ``0 + eps``).

    Parameters
    ----------
    s_lambda_SNL : (S, N, L) fp32 — output of the soft-init kernel.
    theta        : (N, L) fp32 — caller-allocated; fully written.
    eps_f64      : ``np.finfo(np.float64).eps``.
    """
    S, N, L = s_lambda_SNL.shape
    S_f64 = np.float64(S)
    zero = np.float32(0.0)

    for n in prange(N):
        for l in range(L):
            acc = zero
            cnt = 0
            for s in range(S):
                v = s_lambda_SNL[s, n, l]
                if v != zero:
                    acc += v
                    cnt += 1
            if cnt > 0:
                th = np.float32(np.float64(acc) / S_f64)
                th = np.float32(np.float64(th) / np.float64(cnt))
                theta[n, l] = np.float32(np.float64(th) + eps_f64)
            else:
                theta[n, l] = np.float32(eps_f64)
