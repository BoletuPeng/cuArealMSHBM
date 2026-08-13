"""fc_similarity.py

Block-wise FC-similarity matrix computation.

Direct port of the inner scan loop in
``CBIG_SPGrad_RSFC_gradients.m`` (lines 285-402). Matmuls go through
numpy ``@`` (OpenBLAS sgemm); the demean / L2-norm /
outer-product-divide steps around each matmul are fused into a single
in-place numba @njit pass per matrix.

MATLAB block partition (lines 304-307):
    iter_a       = a + (mod(num_sample_S, a) ~= 0)
    iter_b       = b + (mod(num_vertices, b) ~= 0)
    block_size_a = fix(num_sample_S / a)
    block_size_b = fix(num_vertices / b)

Precision: MATLAB ``mean(single_array, dim)`` accumulates in fp64 and
``bsxfun(@minus, X, mean(X, dim))`` happens in fp64 before casting
back to fp32. The fused kernels below replicate this: read fp32,
accumulate mean + sum-of-squares in fp64, write fp32, emit fp32 sqrt
of fp64 sum.

FC_A is built once per call and reused across the j (block-B) loop.
The MATLAB reference re-derives it every j-iter; in exact arithmetic
that re-derivation would be a no-op (the second demean is already
zero-mean), but the fp32 in-place version is NOT idempotent under
repeated application, so doing it once is both faster and more
faithful to the algorithm's intent.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numba as nb


# ─────────────────────────────────────────────────────────────────────
# Fused numba kernels: fp32 in, fp64-accumulated mean/L2, fp32 out.
# ─────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def _gather_T_demean_and_norm(curr_data, row_idx, out_x, mag_out):
    """Gather rows ``curr_data[row_idx[j], :]`` into ``out_x[:, j]``
    (transpose), then demean each column + emit column L2 norms.
    Two-pass: pass 1 gathers + accumulates fp64 sum; pass 2 subtracts
    mean + accumulates fp64 sum-of-squares. fp32 storage throughout.
    """
    T, K = out_x.shape
    for j in nb.prange(K):
        r = row_idx[j]
        s = 0.0
        for t in range(T):
            v = curr_data[r, t]
            out_x[t, j] = v
            s += v
        m = s / T
        ss = 0.0
        for t in range(T):
            v = out_x[t, j] - m
            out_x[t, j] = np.float32(v)
            ss += v * v
        mag_out[j] = np.float32(np.sqrt(ss))


@nb.njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def _slice_T_demean_and_norm(curr_data, b_start, b_end, out_x, mag_out):
    """Like ``_gather_T_demean_and_norm`` but row source is the
    contiguous slice ``curr_data[b_start:b_end, :]``.
    """
    T, K = out_x.shape
    for j in nb.prange(K):
        r = b_start + j
        s = 0.0
        for t in range(T):
            v = curr_data[r, t]
            out_x[t, j] = v
            s += v
        m = s / T
        ss = 0.0
        for t in range(T):
            v = out_x[t, j] - m
            out_x[t, j] = np.float32(v)
            ss += v * v
        mag_out[j] = np.float32(np.sqrt(ss))


@nb.njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def _scale_demean_norm_inplace(x, row_scale, col_scale, mag_out):
    """Fused per-column op on (M, K) fp32 ``x``, in place:
        (1) x[i,j] /= row_scale[i] * col_scale[j]
        (2) x[i,j] -= mean(x[:,j])    (mean accumulated in fp64)
        (3) mag_out[j] = ||x[:,j]||_2 fp32 (sum in fp64)
    Eliminates the two (M, K) outer-product temporaries (mt @ col_scale)
    that the prior numpy code allocated per j-iter.
    """
    M, K = x.shape
    for j in nb.prange(K):
        cs = col_scale[j]
        s = 0.0
        for i in range(M):
            v = x[i, j] / (row_scale[i] * cs)
            x[i, j] = np.float32(v)
            s += v
        m = s / M
        ss = 0.0
        for i in range(M):
            v = x[i, j] - m
            x[i, j] = np.float32(v)
            ss += v * v
        mag_out[j] = np.float32(np.sqrt(ss))


@nb.njit(cache=True, fastmath=False, boundscheck=False, parallel=True)
def _rows_demean_T_and_norm(curr_data, row_idx, out_t, mag_out):
    """Gather rows ``curr_data[row_idx[i], :]`` into ``out_t[i, :]``,
    then row-demean (along axis=1) + emit per-row L2 norms. Used by
    ``compute_t_series`` to mirror MATLAB ``mean(t_series, 2)``.
    """
    K, T = out_t.shape
    for i in nb.prange(K):
        r = row_idx[i]
        s = 0.0
        for t in range(T):
            v = curr_data[r, t]
            out_t[i, t] = v
            s += v
        m = s / T
        ss = 0.0
        for t in range(T):
            v = out_t[i, t] - m
            out_t[i, t] = np.float32(v)
            ss += v * v
        mag_out[i] = np.float32(np.sqrt(ss))


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
def compute_t_series(
    curr_data: np.ndarray,
    randinds_FC: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(t_series, mag_t)`` for one scan. MATLAB lines 317-319:
        t_series = curr_data(randinds_FC, :);
        t_series = bsxfun(@minus, t_series, mean(t_series, 2));
        mag_t = sqrt(sum(t_series.^2, 2));

    curr_data   : (N_cortex, T) fp32 — fp32 contract from bold_io.
    randinds_FC : (N2,) int — 0-indexed row indices.
    Returns (t_series (N2, T) fp32, mag_t (N2, 1) fp32).
    """
    assert curr_data.dtype == np.float32, "curr_data must be fp32"
    idx = np.ascontiguousarray(np.asarray(randinds_FC).reshape(-1),
                               dtype=np.int64)
    n2 = idx.shape[0]
    t_series = np.empty((n2, curr_data.shape[1]), dtype=np.float32)
    mag_flat = np.empty(n2, dtype=np.float32)
    _rows_demean_T_and_norm(curr_data, idx, t_series, mag_flat)
    return t_series, mag_flat.reshape(-1, 1)


def compute_FC_simi_block(
    curr_data: np.ndarray,
    t_series: np.ndarray,
    mag_t: np.ndarray,
    randinds_verts: np.ndarray,
    block_a_index: int,
    *,
    num_blocks_a: int = 3,
    num_blocks_b: int = 10,
) -> np.ndarray:
    """Compute one ``FC_simi_block`` (N_cortex, block_a_size) fp32.
    All array inputs are fp32 by upstream contract (asserted).
    """
    assert curr_data.dtype == np.float32, "curr_data must be fp32"
    assert t_series.dtype == np.float32, "t_series must be fp32"
    assert mag_t.dtype == np.float32, "mag_t must be fp32"

    n_cortex, T = curr_data.shape
    rv = np.ascontiguousarray(np.asarray(randinds_verts).reshape(-1),
                              dtype=np.int64)
    n1 = rv.shape[0]
    a, b = int(num_blocks_a), int(num_blocks_b)
    iter_a = a + (1 if (n1 % a) != 0 else 0)
    iter_b = b + (1 if (n_cortex % b) != 0 else 0)
    block_size_a = n1 // a
    block_size_b = n_cortex // b
    if not (0 <= block_a_index < iter_a):
        raise ValueError(
            f"block_a_index {block_a_index} out of range [0, {iter_a})")

    t = t_series                                  # (N2, T) fp32
    mt_flat = np.ascontiguousarray(mag_t.ravel(),
                                   dtype=np.float32)  # (N2,) fp32

    # ---- FC_A for block_a_index (built once, reused across j) -------
    a_start = block_a_index * block_size_a
    a_end = n1 if block_a_index == iter_a - 1 else (block_a_index + 1) * block_size_a
    a_idx = np.ascontiguousarray(rv[a_start:a_end], dtype=np.int64)
    block_a_size = a_idx.shape[0]
    s_series_A = np.empty((T, block_a_size), dtype=np.float32)
    mag_s_A = np.empty(block_a_size, dtype=np.float32)
    _gather_T_demean_and_norm(curr_data, a_idx, s_series_A, mag_s_A)

    # MATLAB sequence per j-iter (we hoist out of the j-loop — see
    # the module docstring for why this is faithful, not just faster):
    #     FC_A = (t @ s_series_A) / (mt @ mag_s_A)
    #     FC_A = FC_A - mean(FC_A, 1)
    #     mag_a = sqrt(sum(FC_A.^2, 1))
    FC_A = t @ s_series_A
    if FC_A.dtype != np.float32:
        FC_A = FC_A.astype(np.float32, copy=False)
    mag_a = np.empty(block_a_size, dtype=np.float32)
    _scale_demean_norm_inplace(FC_A, mt_flat, mag_s_A, mag_a)
    mag_a_row = mag_a.reshape(1, -1)

    # ---- block-B loop (reuse buffers when sizes match) --------------
    FC_simi_block = np.zeros((n_cortex, block_a_size), dtype=np.float32)
    s_series_B_buf = np.empty((T, block_size_b), dtype=np.float32)
    mag_s_B_buf = np.empty(block_size_b, dtype=np.float32)

    for j in range(iter_b):
        b_start = j * block_size_b
        b_end = n_cortex if j == iter_b - 1 else (j + 1) * block_size_b
        kb = b_end - b_start
        if kb == block_size_b:
            s_series_B, mag_s_B = s_series_B_buf, mag_s_B_buf
        else:
            s_series_B = np.empty((T, kb), dtype=np.float32)
            mag_s_B = np.empty(kb, dtype=np.float32)
        _slice_T_demean_and_norm(curr_data, b_start, b_end,
                                 s_series_B, mag_s_B)

        FC_B = t @ s_series_B
        if FC_B.dtype != np.float32:
            FC_B = FC_B.astype(np.float32, copy=False)
        # Fused: divide by mt @ mag_s_B (was a 22 MB temp), then column
        # demean + L2 — single in-place sweep over FC_B.
        mag_b = np.empty(kb, dtype=np.float32)
        _scale_demean_norm_inplace(FC_B, mt_flat, mag_s_B, mag_b)

        block = (FC_B.T @ FC_A) / (mag_b.reshape(-1, 1) @ mag_a_row)
        FC_simi_block[b_start:b_end, :] = block.astype(np.float32, copy=False)

    # Mimic wb_command -cifti-gradient's input-time NaN→0 substitution.
    # Stuck (var=0) vertices in the BOLD input produce mag==0 → 0/0=NaN
    # in the correlation divides above. The MATLAB CBIG reference path
    # writes FC_simi_block to a dtseries and runs wb_command; wb_command
    # silently replaces NaN inputs with 0 before computing the gradient.
    # Our self-implemented cifti_gradient is NaN-passthrough, so without
    # this clamp a single stuck seed pollutes the entire (N_cortex, K)
    # output and the downstream avg_grads → edge_density → diffmap chain
    # collapses to zero. Verified on YS sub-005 (session 1 has 1 stuck
    # seed → without clamp, edge_density is all-zero; with clamp,
    # edge_density matches MATLAB scale, min≈1e-3, mean≈0.16).
    #
    # ``posinf=0.0, neginf=0.0`` are kept defensive but mathematically
    # unreachable here: whenever ``mag_t[i]==0`` (or ``mag_s[j]==0``),
    # the corresponding row (col) of t_series (s_series) is identically
    # zero after demean, so the dot-product numerator is also zero —
    # only 0/0 = NaN can arise, never ±finite/0 = ±inf.
    np.nan_to_num(FC_simi_block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return FC_simi_block
