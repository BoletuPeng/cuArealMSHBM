"""_kernels.py — step-2 init leaves numba kernels.

Two kernels backing the production ``compose_init_state`` super-call:

* :func:`profile_to_hard_labels_kernel` — per-subject argmax of
  ``mean(profile, axis=T) @ g_mu`` + medial-mask norm-zero check.
  Writes ``hard_label (N,) int32`` instead of an (N, L) fp32 one-hot
  (Params["s_lambda"] is sparse one-hot after init; the (N, L, S)
  intermediate buffer at fsa6 / S=3 / L=400 is 393 MB of mostly zeros
  with one nonzero per (n, s) slice — entirely skipped).

* :func:`compose_init_state_kernel` — walks the per-subject
  hard_labels + medial flags + boundary_mask and sparse-writes the
  final ``(S, N, L) fp64`` Params["s_lambda"] buffer + computes
  ``Params["theta"] (N, L) fp32`` in two parallel passes. Replaces
  the 9 dense (N, L, S) numpy operations of the old
  ``init_theta_from_s_lambda`` leaf (1.58s wall) + the final
  ``(N, L, S) → (S, N, L)`` fp64 transpose (380ms wall).

Reduction dtype: ``profile_to_hard_labels_kernel`` uses fp32 acc +
fp32 g_mu cast. At fsa6 / S=3 the argmax disagreement vs an fp64-acc
variant was 1 vertex out of 224k non-medial (0.0004%) — within the
BLAS-vendor noise floor already in the pipeline. fp32 was 2.0×
faster per sub (445 → 218 ms) so the cast lives in the wrapper.

Storage: profile is fp32 (matches caller). g_mu is fp64 input
to the public API (from group.mat); the wrapper casts to fp32 once
on the caller side. T-mean accumulator stays fp32 (matches
MATLAB ``mean(data_series, 3)``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False, cache=True, boundscheck=False)
def profile_to_hard_labels_kernel(profile: np.ndarray,
                                  g_mu_f32: np.ndarray,
                                  hard_label_out: np.ndarray,
                                  medial_out: np.ndarray) -> None:
    """Per-subject argmax-label + medial flag (no (N, L) materialization).

    Variant of :func:`init_s_lambda_from_profile_kernel` that writes
    just the int32 argmax index per row instead of the (N, L) fp32
    one-hot. Downstream :func:`compose_init_state_kernel` consumes
    hard_label directly, so we skip the 131 MB one-hot buffer.

    **fp32 reduction** — see module docstring. avg row values are unit-
    normalized (||avg|| = 1) and g_mu entries are O(0.01-0.1); the
    inner-loop sum stays well within fp32 mantissa precision. Tested
    on fsa6 / S=3: 1 argmax flip vs fp64 (out of 224k non-medial)
    and 0 medial-mask flips — within the BLAS-vendor noise floor
    already in the pipeline. 2.0× faster per sub.

    Parameters
    ----------
    profile        : (N, T, D) fp32 C-contig — per-subject BOLD,
                     **NTD layout** (matches the unified step-2 SNTD
                     stack ``bold_SNTD[s]`` slice). Each per-(n, t)
                     ``(D,)`` row is unit-stride contig in memory; the
                     T inner axis is also contig per n (sub-T axis).
    g_mu_f32       : (D, L)   fp32 C-contig — group centroids
                     (cast from fp64 source by caller).
    hard_label_out : (N,)     int32 — column index of row-argmax of
                     ``mean(profile, 1) @ g_mu``.
    medial_out     : (N,)     bool  — ``||mean(profile, 1)||_2 == 0``.
    """
    N, T, D = profile.shape
    L = g_mu_f32.shape[1]
    inv_T = np.float32(1.0) / np.float32(T)

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
        norm_sq = np.float32(0.0)
        for d in range(D):
            a = mean_d[d] * inv_T
            norm_sq += a * a
            for l in range(L):
                acc[l] += a * g_mu_f32[d, l]

        best_val = acc[0]
        best_l = 0
        for l in range(1, L):
            if acc[l] > best_val:
                best_val = acc[l]
                best_l = l

        hard_label_out[n] = best_l
        medial_out[n] = (norm_sq == np.float32(0.0))


@njit(parallel=True, fastmath=False, cache=True, boundscheck=False)
def compose_init_state_kernel(hard_label_SN: np.ndarray,
                              medial_SN: np.ndarray,
                              boundary_mask: np.ndarray,
                              s_lambda_SNL_f32: np.ndarray,
                              theta: np.ndarray,
                              inv_S_f32: np.float32,
                              eps_f64: np.float64) -> None:
    """Compose the post-init Params state from per-sub hard labels.

    Replaces the 9-pass dense (N, L, S) numpy chain in
    :func:`init_theta_from_s_lambda` PLUS the final
    ``(N, L, S) → (S, N, L)`` fp64 transpose. Exploits the fact that
    ``init_s_lambda`` output is one-hot — each (n, s) slice has at
    most one nonzero cell, so the (N, L, S) intermediate buffer is
    99.75% zeros and can be skipped entirely.

    Algorithm (semantic mirror of MATLAB lines 172-180):

      Per (n, s):
        l_act = hard_label[s, n]
        if medial[s, n]:                 continue
        if boundary_mask[n, l_act] == 0: continue       (NaN-row after /)
        s_lambda_SNL[s, n, l_act] = 1.0
        theta_num_row[l_act] += 1.0

      Per (n, l):
        active   ⇒ theta[n, l] = float32(1/S + eps_f64)
        inactive ⇒ theta[n, l] = float32(eps_f64)

    Race-free: outer loop is ``prange(N)``; per n, the s-loop is
    sequential and the (n, *) cells in s_lambda are only touched by
    the thread owning n. ``theta_num_row`` is a per-iter local.

    Parameters
    ----------
    hard_label_SN    : (S, N) int32 — per-sub argmax cluster index.
    medial_SN        : (S, N) bool  — per-sub medial-wall flag.
    boundary_mask    : (N, L) fp32  — bilateral block-diagonal mask.
    s_lambda_SNL_f32 : (S, N, L) fp32 — caller-zeroed; sparse-written.
                       The only stored values are 0.0 / 1.0 (hard-label
                       one-hot), both exactly representable in fp32 — no
                       precision loss vs an fp64 scratch.
    theta            : (N, L) fp32 — caller-allocated; fully written.
    inv_S_f32        : 1.0 / S in fp32 (precomputed by Python wrapper).
    eps_f64          : ``np.finfo(np.float64).eps`` (passed in to keep
                       numba's compile-time bindings simple).
    """
    S, N = hard_label_SN.shape
    L = boundary_mask.shape[1]
    active_theta_f64 = np.float64(inv_S_f32) + eps_f64
    inactive_theta_f64 = eps_f64

    for n in prange(N):
        theta_num_row = np.zeros(L, dtype=np.float32)

        for s in range(S):
            if medial_SN[s, n]:
                continue
            l_act = hard_label_SN[s, n]
            if boundary_mask[n, l_act] == np.float32(0.0):
                continue
            s_lambda_SNL_f32[s, n, l_act] = np.float32(1.0)
            theta_num_row[l_act] += np.float32(1.0)

        for l in range(L):
            if theta_num_row[l] > np.float32(0.0):
                theta[n, l] = np.float32(active_theta_f64)
            else:
                theta[n, l] = np.float32(inactive_theta_f64)
