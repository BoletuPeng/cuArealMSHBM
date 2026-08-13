"""_kernels.py

Numba kernels for the M-step fast path. Single-core, out-buffer style.
Every kernel takes pre-allocated input + output buffers, allocates
nothing inside the JIT'd region, and returns nothing or a single fp64
scalar reduction.

Non-numba pieces of the M-step:
    * ``X_dot_sl`` — BLAS sgemm via ``np.matmul``.
    * ``invad``    — fp64 scalar root-find on a Bessel ratio
                     (``scipy.special.iv`` + secant).
    * Layout transposes at Session I/O — NumPy ``ascontiguousarray``
      copies. ``data_series`` is held by reference (single (N, T, D)
      copy shared across the super-call's sub-Sessions); per-t access
      uses the strided view ``data_series_NTD[:, t, :]`` so there is no
      BOLD-sized transpose. Per-call transposes for ``s_t_nu``,
      ``s_lambda``, ``s_psi`` run on every ``run()`` (~7 ms total on
      sub-001).

Cores belong to the outer subject-level multiprocessing harness; no
``parallel=True`` / ``prange`` here.

Buffer reference (Mode A, sub-001, fsaverage6, 6 sessions, 300 ROIs):
    N = 81924  D = 1175  L = 300  T = 6.

dtype contract:
    * Hot-path arithmetic is fp32 throughout.
    * Reduction accumulators are fp64, populated by implicit promotion
      (``acc = 0.0`` + fp32 operand arithmetic). Explicit
      ``np.float64(...)`` wrappers around fp32 element loads disable
      numba's SIMD auto-vectorization and slow the kernel ~100×. With
      ~2.1M-term sums, fp32 acc would inject ~1.7e-4 relative noise
      into the kappa scalar (above the M-step's ε = 1e-4 convergence
      threshold) and prevent termination.
    * The vMF kappa scalar (``invad`` output) is fp64; cast to fp32
      exactly once before the per-iter fused kernel.

Kernel-by-kernel I/O contract:
+----------------------------------+------------------------------+-----------------+----------------+
| Kernel                           | Reads (+ scalars)            | Writes          | Returns        |
+----------------------------------+------------------------------+-----------------+----------------+
| compute_sigma_psi_DL_f32         | sigma (L,) fp32,             | sigma_psi (D,L) | None           |
|                                  |   s_psi (D,L) fp32           |   fp32          |                |
| compute_denom_NL_f64             | num_session (i64),           | -               | scalar fp64    |
|                                  |   s_lambda (N,L) fp32        |                 |                |
| kappa_sum_reduce_TDL_f32         | s_t_nu (T,D,L) fp32,         | -               | scalar fp64    |
|                                  |   X_dot_sl (T,D,L) fp32      |                 |                |
| fused_lambda_X_normalize_TDL_f32 | kappa_f32,                   | s_t_nu_new      | None           |
|                                  |   X_dot_sl (T,D,L) fp32,     |   (T,D,L) fp32, |                |
|                                  |   sigma_psi (D,L) fp32,      |   cos (T,L)     |                |
|                                  |   s_t_nu_old (T,D,L) fp32    |   fp32          |                |
| convergence_flag_accumulate_T    | cos (T,L) fp32, eps_f32      | flag_acc (T,)   | None           |
|                                  |                              |   i8 (OR-acc;   |                |
|                                  |                              |   caller zeros) |                |
| kappa_drift_mean_f64             | kappa_old (L,) fp64,         | -               | scalar fp64    |
|                                  |   kappa_new (L,) fp64        |                 |                |
+----------------------------------+------------------------------+-----------------+----------------+

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import numba as nb


# ─────────────────────────────────────────────────────────────────────────
# Per-call setup kernels (called once per M-step, before the iter_m loop)
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False)
def compute_sigma_psi_DL_f32(sigma_L, s_psi_DL, sigma_psi_DL):
    """sigma_psi[d, l] = sigma[l] * s_psi[d, l].

    Mirrors MATLAB ``bsxfun(@times, Params.sigma, Params.s_psi(:, :, s))``.
    fp32 output (matches MATLAB's bsxfun downcast). Mode A only — single
    subject, so the (S, D, L) → (D, L) drop is exact.

    Inputs:
        sigma_L        : (L,)   fp32 read-only
        s_psi_DL       : (D, L) fp32 read-only C-contig
    Output (caller-provided):
        sigma_psi_DL   : (D, L) fp32 — overwritten
    """
    D, L = s_psi_DL.shape
    for d in range(D):
        for l in range(L):
            sigma_psi_DL[d, l] = sigma_L[l] * s_psi_DL[d, l]


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def compute_denom_NL_f64(num_session, s_lambda_NL):
    """denom = num_session * sum_{n, l} s_lambda[n, l].

    Mirrors MATLAB ``sum(num_session .* sum(sum(s_lambda, 1), 3))`` for
    Mode A (single subject), which expands to
    ``num_session * sum(s_lambda)`` (associativity holds at fp64). fp64
    result so the subsequent ``rbar = kappa_sum / denom`` divide inherits
    double precision; the divide feeds invAd which is fp64.

    Implicit promotion: ``acc = 0.0`` is fp64; ``acc += s_lambda[n, l]``
    promotes the fp32 operand to fp64 and accumulates in fp64.

    Inputs:
        num_session    : int (passed as i64, multiplied as fp64 below)
        s_lambda_NL    : (N, L) fp32 read-only C-contig
    Returns:
        scalar fp64.
    """
    N, L = s_lambda_NL.shape
    acc = 0.0
    for n in range(N):
        for l in range(L):
            acc += s_lambda_NL[n, l]
    return float(num_session) * acc


# ─────────────────────────────────────────────────────────────────────────
# Per-iter kernels (called once per iter_m of the M-step)
# ─────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=False, boundscheck=False)
def kappa_sum_reduce_TDL_f32(s_t_nu_TDL, X_dot_sl_TDL):
    """sum_{t, d, l} s_t_nu[t, d, l] * X_dot_sl[t, d, l].

    Mathematically equivalent to MATLAB's
        kappa_update = mtimesx(data.series, permute(s_t_nu, [1,2,4,3]));
        kappa_update = bsxfun(@times, s_lambda, kappa_update);
        kappa_sum    = sum(sum(sum(sum(kappa_update, 1), 4), 3));
    Mode A (S=1) collapse: the (T, S, D, L) reduction degenerates to
    (T, D, L). Precomputed via the gemm trick (``X_dot_sl`` cached at
    M-step entry) so this kernel just sums the pointwise product.

    Reduction order: ascending t, d, l (C-order natural). Differs from
    MATLAB's mtimesx + 4D-sum tree; per-element drift is fp32 ULP scale,
    per-call kappa_scalar agreement is within a few fp32 ULP.

    Inputs:
        s_t_nu_TDL    : (T, D, L) fp32 read-only C-contig
        X_dot_sl_TDL  : (T, D, L) fp32 read-only C-contig
    Returns:
        scalar fp64.
    """
    T, D, L = s_t_nu_TDL.shape
    acc = 0.0  # fp64 (Python literal -> fp64 in numba; implicit promotion)
    for t in range(T):
        for d in range(D):
            for l in range(L):
                acc += s_t_nu_TDL[t, d, l] * X_dot_sl_TDL[t, d, l]
    return acc


@nb.njit(cache=True, fastmath=False, boundscheck=False, error_model="numpy")
def fused_lambda_X_normalize_TDL_f32(
    kappa_f32,
    X_dot_sl_TDL,
    sigma_psi_DL,
    s_t_nu_old_TDL,
    s_t_nu_new_TDL,
    cos_TL,
    eps_f32,
    flag_T_acc,
):
    """Per-t M-step body — TDL-batch shape (Mode A: S=1 dropped).

    For each (t, l):
        col[d]            = kappa * X_dot_sl[t, d, l] + sigma_psi[d, l]
        col_norm          = sqrt(sum_d col[d]^2)                # fp64 acc
        s_t_nu_new[t,d,l] = col[d] / col_norm
        cos[t, l]         = sum_d s_t_nu_new[t,d,l] * s_t_nu_old[t,d,l]   # fp64 acc

    After all L columns of row ``t`` finish, the per-t MONOTONE
    convergence flag is OR-accumulated into ``flag_T_acc`` in the same
    nopython region — no follow-up Python wrapper / second numba kernel.
    Per-t flag semantics mirror step3 lines 482 + 520-522 of the MATLAB
    driver (S=1 dropped):

        for t: if all L (1 - cos) < eps this iter, set flag_T_acc[t] = 1;
        else leave as-is (monotone OR across iter_m).

    Caller responsibilities:
      * ``s_t_nu_old_TDL`` and ``s_t_nu_new_TDL`` MUST be different
        buffers (the kernel reads old while writing new).
      * ``flag_T_acc`` zeroed ONCE at top of M-step (before iter_m loop).
        Per-iter calls only set bits to 1, never clear.

    Empty parcels (col-norm == 0): the chain
        acc_n == 0  =>  cn = 0  =>  inv_cn = inf
        new_v = v * inv_cn = NaN  (error_model="numpy")
    poisons cos and the per-t flag check sees ``1 - NaN`` = NaN < eps =
    False — empty columns count as NOT converged. Mirrors MATLAB.

    Inputs:
        kappa_f32       : fp32 scalar
        X_dot_sl_TDL    : (T, D, L) fp32 read-only C-contig
        sigma_psi_DL    : (D, L)    fp32 read-only C-contig
        s_t_nu_old_TDL  : (T, D, L) fp32 read-only C-contig
        eps_f32         : fp32 scalar — convergence tolerance
    Outputs (caller-provided):
        s_t_nu_new_TDL  : (T, D, L) fp32 — overwritten
        cos_TL          : (T, L)    fp32 — overwritten (kept for diagnostics)
    Input/output:
        flag_T_acc      : (T,) int8 — bits set to 1 for converged-this-iter
                          rows; never cleared inside this kernel.
    """
    T, D, L = X_dot_sl_TDL.shape
    for t in range(T):
        for l in range(L):
            # Pass 1: lambda_X -> scratch (s_t_nu_new), acc col-norm.
            acc_n = 0.0  # fp64
            for d in range(D):
                v = (kappa_f32 * X_dot_sl_TDL[t, d, l]
                     + sigma_psi_DL[d, l])
                s_t_nu_new_TDL[t, d, l] = v
                acc_n += v * v
            cn = np.sqrt(acc_n)
            inv_cn = np.float32(1.0 / cn)
            # Pass 2: divide in place + cosine vs old.
            acc_c = 0.0  # fp64
            for d in range(D):
                new_v = s_t_nu_new_TDL[t, d, l] * inv_cn
                s_t_nu_new_TDL[t, d, l] = new_v
                acc_c += new_v * s_t_nu_old_TDL[t, d, l]
            cos_TL[t, l] = np.float32(acc_c)
        # Per-t monotone convergence flag (no second kernel — inlined).
        # ``cos_TL[t, *]`` was just written above; cells that became NaN
        # via 0*inf fail ``(1-NaN) < eps`` (NaN comparison is False), so
        # NaN entries correctly count as NOT converged.
        n_converged = 0
        for l in range(L):
            if (np.float32(1.0) - cos_TL[t, l]) < eps_f32:
                n_converged += 1
        if n_converged >= L:
            flag_T_acc[t] = np.int8(1)


# ─────────────────────────────────────────────────────────────────────────
# Warmup — compile every kernel once at process startup
# ─────────────────────────────────────────────────────────────────────────
def warmup() -> None:
    """Pre-compile every kernel with realistic dtypes/shapes. ~50 ms one-shot.

    Recommended at process startup so the first hot-path call doesn't
    pay the JIT cost. Idempotent (numba caches the compiled artifacts).
    """
    T, D, L, N = 2, 4, 3, 5

    sigma_L = np.full(L, 0.5, dtype=np.float32)
    s_psi_DL = np.full((D, L), 0.1, dtype=np.float32)
    sigma_psi_DL = np.empty((D, L), dtype=np.float32)
    compute_sigma_psi_DL_f32(sigma_L, s_psi_DL, sigma_psi_DL)

    s_lambda_NL = np.full((N, L), 0.7, dtype=np.float32)
    compute_denom_NL_f64(np.int64(T), s_lambda_NL)

    s_t_nu_TDL = np.full((T, D, L), 0.3, dtype=np.float32)
    X_dot_sl_TDL = np.full((T, D, L), 0.4, dtype=np.float32)
    kappa_sum_reduce_TDL_f32(s_t_nu_TDL, X_dot_sl_TDL)

    s_t_nu_old_TDL = np.full((T, D, L), 0.2, dtype=np.float32)
    s_t_nu_new_TDL = np.empty((T, D, L), dtype=np.float32)
    cos_TL = np.empty((T, L), dtype=np.float32)
    flag_T_acc = np.zeros(T, dtype=np.int8)
    fused_lambda_X_normalize_TDL_f32(
        np.float32(2.0), X_dot_sl_TDL, sigma_psi_DL,
        s_t_nu_old_TDL, s_t_nu_new_TDL, cos_TL,
        np.float32(1e-4), flag_T_acc,
    )
