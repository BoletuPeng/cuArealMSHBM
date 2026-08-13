"""m_step.py

M-step inner-while loop body. Single-subject (Mode A, S=1).

The M-step does two things per inner iter:
    1. Update the kappa scalar (vMF concentration) via a 3-D mean-
       resultant reduction → ``invad``.
    2. Update the per-session vMF mean direction ``s_t_nu`` by
       combining ``X' * s_lambda`` with the ``sigma * s_psi`` prior pull,
       then column-normalizing.

Public API:
    MStepSession — caches per-call scratch.

External shapes:
    data_series_NTD : (N, T, D) fp32 C-contig — the only accepted BOLD
                      layout. Per-t access uses strided slicing
                      (``data_series_NTD[:, t, :]``, lda = T·D); MKL
                      sgemm handles the lda with <1% overhead vs a
                      dedicated TND copy.
    s_t_nu          : (D, L, T) fp32 — input read, output freshly allocated.
    s_lambda        : (N, L)    fp32.
    s_psi           : (D, L)    fp32.
    sigma           : (L,)      any float.
    kappa_init      : (L,)      fp64.

dtype contract:
    s_t_nu, s_lambda, s_psi, sigma, X_dot_sl, sigma_psi — fp32.
    kappa — fp64 (cast to fp32 once per iter for the fused kernel).
    rbar, invad output — fp64 (Bessel territory).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import _kernels
from ._invad import invad


# ─────────────────────────────────────────────────────────────────────────
# Layout helpers — MATLAB shape <-> internal shape, at the I/O boundary
# (Session construction or per-call entry). Not in the per-iter hot path.
# ─────────────────────────────────────────────────────────────────────────
def _stage_data_series_NTD(data_series_NTD: np.ndarray) -> np.ndarray:
    """Validate caller's ``(N, T, D)`` BOLD; zero-copy passthrough.

    The super-call (:class:`VmfClusteringSession`) owns the single
    (N, T, D) fp32 C-contig buffer (materialized once by
    ``unpack_normalize_packed_NTD_host`` in its ``__init__``) and
    passes the same reference to every sub-Session — so the
    ``np.ascontiguousarray`` below is always a no-op in production.
    """
    arr = np.asarray(data_series_NTD)
    if arr.ndim != 3:
        raise ValueError(
            f"data_series_NTD must be 3D (N, T, D); got {arr.shape}"
        )
    return np.ascontiguousarray(arr, dtype=np.float32)


def _s_t_nu_to_internal(s_t_nu: np.ndarray) -> np.ndarray:
    """``(D, L, T) external`` -> ``(T, D, L) C-contig fp32``."""
    arr = np.asarray(s_t_nu)
    if arr.ndim != 3:
        raise ValueError(f"s_t_nu must be 3D (D, L, T); got {arr.shape}")
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def _s_t_nu_to_external(s_t_nu_TDL: np.ndarray) -> np.ndarray:
    """``(T, D, L) internal`` -> ``(D, L, T) external C-contig``.

    Returns a FRESH allocation (not a view of the Session's internal
    buffer) so the caller can hold the result across the next ``run()``
    call without it being overwritten by the next ping-pong write.
    """
    return np.ascontiguousarray(s_t_nu_TDL.transpose(1, 2, 0))


def _s_lambda_to_internal(s_lambda: np.ndarray) -> np.ndarray:
    """``(N, L) external`` -> ``(N, L) C-contig fp32`` (zero-copy if already)."""
    arr = np.asarray(s_lambda)
    if arr.ndim != 2:
        raise ValueError(f"s_lambda must be 2D (N, L); got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _s_psi_to_internal(s_psi: np.ndarray) -> np.ndarray:
    """``(D, L) external`` -> ``(D, L) C-contig fp32`` (zero-copy if already)."""
    arr = np.asarray(s_psi)
    if arr.ndim != 2:
        raise ValueError(f"s_psi must be 2D (D, L); got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


# X_dot_sl computation — the only non-numba compute in the M-step.
# ``np.matmul`` dispatches to MKL multi-threaded sgemm.
def _compute_X_dot_sl_TDL(
    data_series_NTD: np.ndarray,    # (N, T, D) fp32 C-contig
    s_lambda_NL: np.ndarray,        # (N, L)    fp32 C-contig
    out_TDL: np.ndarray,            # (T, D, L) fp32 C-contig — output
) -> None:
    """Per-t gemm: ``out_TDL[t] = X[:, t, :].T @ s_lambda``.

    Mirrors the per-t ``X' * s_lambda`` in step3 line 514 (Mode A:
    S=1). Pre-computed once per M-step call so the inner iter_m loop
    doesn't recompute it (MATLAB recomputes every iter_m).

    The per-t slice ``data_series_NTD[:, t, :]`` is a (N, D) strided
    view (lda = T·D). MKL sgemm handles strided lda with ≤1% overhead
    vs a dedicated TND C-contig copy.
    """
    T = data_series_NTD.shape[1]
    for t in range(T):
        X = data_series_NTD[:, t, :]  # (N, D) strided view (lda = T·D)
        # X.T (D, N) @ s_lambda (N, L) -> (D, L). BLAS sgemm.
        np.matmul(X.T, s_lambda_NL, out=out_TDL[t])


# ─────────────────────────────────────────────────────────────────────────
# Production hot path — Session with cached static state + ping-pong buffers
# ─────────────────────────────────────────────────────────────────────────
class MStepSession:
    """Pre-allocated state for the M-step fast path.

    Owns all per-call and per-iter scratch — every numba kernel inside
    ``run`` writes into a buffer on this object. No allocations inside
    the loop.

    Ping-pong buffers
    -----------------
    ``s_t_nu`` is double-buffered as ``_s_t_nu_TSDL_A`` / ``_s_t_nu_TSDL_B``.
    Each iter_m, the fused kernel reads from ``buffers_st[old_idx]`` and
    writes to ``buffers_st[new_idx]``; old/new are swapped at the end of
    the iter (no memcpy — just swap which buffer is "current"). Same
    pattern for ``kappa`` via ``_kappa_old_L`` / ``_kappa_new_L``. This
    is the natural CUDA pattern: two device arrays, the "current" one
    rotates each step.

    The two ``s_t_nu`` TSDL buffers are MANDATORY (not just an
    optimization): the fused kernel's cosine convergence test reads the
    old value while writing the new. Aliasing them would corrupt the
    test.

    Layout
    ------
    Externally the caller speaks in **MATLAB shape** (matches the GT
    files); internally the Session uses a **Python-natural** layout
    with batch axes (T, S, ...) leading so per-(t, s) slices are
    C-contig. See module docstring for the full layout map.

    The Session contract assumes ``data_series`` does NOT change between
    calls within a parcellation — it's the BOLD time series, which is
    fixed. The per-call inputs ``s_t_nu``, ``s_lambda``, ``s_psi``,
    ``sigma``, ``kappa_init`` are all re-read each call.
    """

    __slots__ = (
        # Shape constants.
        "N", "D", "L", "T",
        "dim", "epsilon", "max_iter",
        # Static cache (held by reference; caller-owned).
        "data_series_NTD",
        # Per-call inputs in internal layout (re-populated each run()).
        "_s_lambda_NL",
        "_s_psi_DL",
        "_sigma_L",
        # Per-call setup outputs (reused across the iter_m loop).
        "_X_dot_sl_TDL",
        "_sigma_psi_DL",
        # Ping-pong buffers for s_t_nu (alternated between iters).
        # Kappa is single-scalar throughout the M-step (always built via
        # ``invad`` into a scalar; original L-vector buffer materialized
        # the same value L times — eliminated as redundant). Tracked as
        # Python floats inside ``run()``; the caller-facing (L,) is built
        # once at exit time.
        "_s_t_nu_TDL_A",
        "_s_t_nu_TDL_B",
        # Per-iter scratch.
        "_cos_TL",
        "_flag_T_acc",   # monotone OR-accumulator (zeroed at top of run())
    )

    def __init__(self,
                 data_series_NTD: np.ndarray,
                 dim: float,
                 num_clusters: int,
                 num_session: int,
                 epsilon: float = 1e-4,
                 max_iter: int = 50,
                 ):
        ds = _stage_data_series_NTD(data_series_NTD)
        N, T, D = ds.shape
        if num_session != T:
            raise ValueError(
                f"num_session={num_session} != data_series_NTD.T={T}"
            )

        self.N = int(N)
        self.D = int(D)
        self.L = int(num_clusters)
        self.T = int(T)
        self.dim = float(dim)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)

        L = self.L

        # data_series held by reference. The super-call (vmf_clustering)
        # owns the single (N, T, D) C-contig BOLD and shares it across
        # ELambda, EMStop, and this Session. No per-Session BOLD copy.
        self.data_series_NTD = ds

        # Per-call inputs.
        self._s_lambda_NL = np.empty((N, L), dtype=np.float32)
        self._s_psi_DL = np.empty((D, L), dtype=np.float32)
        self._sigma_L = np.empty(L, dtype=np.float32)

        # Per-call setup outputs.
        self._X_dot_sl_TDL = np.empty((T, D, L), dtype=np.float32)
        self._sigma_psi_DL = np.empty((D, L), dtype=np.float32)

        # Ping-pong buffers for s_t_nu (TWO independent allocations — required
        # by the cosine convergence test that reads old while writing new).
        self._s_t_nu_TDL_A = np.empty((T, D, L), dtype=np.float32)
        self._s_t_nu_TDL_B = np.empty((T, D, L), dtype=np.float32)

        # Per-iter scratch.
        self._cos_TL = np.empty((T, L), dtype=np.float32)
        # Monotone OR-accumulator — zeroed at run() entry; allocation only.
        self._flag_T_acc = np.empty(T, dtype=np.int8)

    # ────────── Per-call API ──────────
    def run(self,
            s_t_nu: np.ndarray,
            s_lambda: np.ndarray,
            s_psi: np.ndarray,
            sigma: np.ndarray,
            kappa_init: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Execute one full M-step inner-while loop.

        Single-subject pipeline — no S axis anywhere.

        Parameters
        ----------
        s_t_nu     : (D, L, T) float32 — input is read; output is a
                     freshly-allocated (D, L, T) array.
        s_lambda   : (N, L)    float32 (or fp64; cast to fp32 internally).
        s_psi      : (D, L)    float32 (or fp64; cast to fp32 internally).
        sigma      : (L,) any float; cast to fp32 internally. Accepts
                     (1, L) at the boundary and flattens.
        kappa_init : (L,) float64. Accepts (1, L) at the boundary.
                     **Must be uniform across L** — every entry equal.
                     The M-step's kappa-as-scalar invariant means only
                     ``kappa_init[0]`` would drive the iteration; a
                     non-uniform input is rejected at the boundary
                     instead of silently using the first entry. The
                     production caller (vmf_clustering) seeds with
                     ``ini_val * np.ones(L)`` and the M-step itself only
                     ever produces a scalar via ``invad``, so the
                     invariant is preserved on the natural call chain.

        Returns
        -------
        s_t_nu : (D, L, T) float32 — fresh allocation.
        kappa  : (L,)     float64 — converged kappa (replicated scalar).
        iter_m : int — number of inner iterations.
        """
        # Stage inputs into internal layout.
        if s_t_nu.shape != (self.D, self.L, self.T):
            raise ValueError(
                f"s_t_nu shape mismatch: expected "
                f"({self.D}, {self.L}, {self.T}), got {s_t_nu.shape}"
            )
        if s_lambda.shape != (self.N, self.L):
            raise ValueError(
                f"s_lambda shape mismatch: expected "
                f"({self.N}, {self.L}), got {s_lambda.shape}"
            )
        if s_psi.shape != (self.D, self.L):
            raise ValueError(
                f"s_psi shape mismatch: expected "
                f"({self.D}, {self.L}), got {s_psi.shape}"
            )

        # Buffer 0 (A) holds the input s_t_nu; buffer 1 (B) gets written iter 1.
        np.copyto(self._s_t_nu_TDL_A, _s_t_nu_to_internal(s_t_nu))
        np.copyto(self._s_lambda_NL, _s_lambda_to_internal(s_lambda))
        np.copyto(self._s_psi_DL, _s_psi_to_internal(s_psi))
        np.copyto(self._sigma_L,
                  np.ascontiguousarray(np.asarray(sigma).ravel(),
                                        dtype=np.float32))
        # Kappa is a scalar throughout the M-step — every iter calls
        # ``invad`` to produce ONE fp64 value; the (L,) buffer the prior
        # implementation used to materialize was always uniform. Track
        # the scalar in a Python local; rebuild the (L,) at exit.
        # Precondition: kappa_init is uniform across L (the production
        # caller — vmf_clustering — always seeds with
        # ``ini_val * np.ones(L)`` and the M-step itself only ever
        # produces a scalar via ``invad``). A non-uniform kappa_init
        # would silently use only entry [0] without this check.
        kappa_init_arr = np.asarray(kappa_init).ravel()
        if kappa_init_arr.size != self.L:
            raise ValueError(
                f"kappa_init length {kappa_init_arr.size} != L={self.L}"
            )
        if not np.all(kappa_init_arr == kappa_init_arr[0]):
            raise ValueError(
                "kappa_init must be uniform across L (M-step's kappa-as-"
                "scalar invariant); got non-uniform input. The kappa "
                "scalar invariant is documented in the class docstring."
            )
        kappa_prev_scalar = float(kappa_init_arr[0])

        # Per-call setup (numba kernels + BLAS).
        _compute_X_dot_sl_TDL(self.data_series_NTD, self._s_lambda_NL,
                              self._X_dot_sl_TDL)
        _kernels.compute_sigma_psi_DL_f32(
            self._sigma_L, self._s_psi_DL, self._sigma_psi_DL,
        )
        denom = _kernels.compute_denom_NL_f64(
            np.int64(self.T), self._s_lambda_NL,
        )

        # Zero the monotone convergence-flag accumulator. MATLAB initializes
        # ``flag_nu`` ONCE outside the while loop (step3 line 482) and only
        # ever sets entries to 1; this matches that semantics.
        self._flag_T_acc.fill(0)

        # Ping-pong index lists.
        st_buffers = (self._s_t_nu_TDL_A, self._s_t_nu_TDL_B)

        eps_f32 = np.float32(self.epsilon)

        iter_m = 0
        final_idx = 0
        kappa_new_scalar = kappa_prev_scalar     # init for first-iter drift

        while True:
            iter_m += 1
            old_idx = (iter_m - 1) % 2   # iter 1 -> read 0 (A)
            new_idx = iter_m % 2         # iter 1 -> write 1 (B)
            final_idx = new_idx

            # Step 1: scalar reduction -> rbar -> kappa_new.
            kappa_sum_f64 = _kernels.kappa_sum_reduce_TDL_f32(
                st_buffers[old_idx], self._X_dot_sl_TDL,
            )
            rbar = float(kappa_sum_f64 / denom)
            kappa_new_scalar = invad(self.dim, rbar)
            kappa_f32 = np.float32(kappa_new_scalar)

            # Step 2: per-t update — TDL-batch fused kernel; the kernel
            # also OR-accumulates the per-t convergence flag into
            # ``self._flag_T_acc`` at the end of each ``t`` outer loop
            # (no separate kernel needed).
            _kernels.fused_lambda_X_normalize_TDL_f32(
                kappa_f32,
                self._X_dot_sl_TDL,
                self._sigma_psi_DL,
                st_buffers[old_idx],     # read
                st_buffers[new_idx],     # write
                self._cos_TL,
                eps_f32,
                self._flag_T_acc,
            )

            # Step 3: convergence — scalar kappa drift + per-t flag check.
            # ``kappa_drift_mean`` reduced to a scalar Python op since
            # kappa is uniform across L throughout the M-step.
            all_flag = (int(self._flag_T_acc.sum()) == self.T)
            kappa_drift = (abs(kappa_prev_scalar - kappa_new_scalar)
                           / kappa_prev_scalar)

            kappa_prev_scalar = kappa_new_scalar

            if all_flag and kappa_drift < self.epsilon:
                break
            # Safety cap: matches MATLAB step3 line 529 ``if(iter_m > 50)``.
            if iter_m > self.max_iter:
                break

        s_t_nu_out = _s_t_nu_to_external(st_buffers[final_idx])
        # Materialize the converged scalar into a (L,) fp64 array at the
        # caller-visible boundary — preserves the public output contract.
        kappa_out = np.full(self.L, kappa_new_scalar, dtype=np.float64)
        return s_t_nu_out, kappa_out, iter_m


