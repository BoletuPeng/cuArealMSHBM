"""e_step_lambda.py

E-step λ-loop body — numba-kernel-backed Session, single-core serial,
out-buffer style, fp32 throughout (fp64 only for the convergence-drift
accumulator, where fp32 reduction over (N, L) drifts ~7e-5 absolute,
just above the ε=1e-4 threshold).

Public API:
    ELambdaSession — caches per-Session scratch (theta, boundary_mask,
        cached log_theta, V_lambda Session) once; recomputes the (N, L)
        log-vMF + softmax + V_lambda close-form each ``compute()`` call.

Per-comp_iter inputs (re-staged each call):
    s_t_nu              : (D, L, T) fp32 — from M-step.
    kappa               : (L,) fp64 — from M-step.
    s_lambda            : (N, L) fp32 — read then overwritten.
    spatial_connect_vmf : (N, L) fp32 — from spatial_connect_prior.
    spatial_xyz_vmf     : (N, L) fp32 — from spatial_xyz_prior, or
                          zeros at the first EM iter.

Static across the whole vmf_clustering call (cached on __init__):
    data_series_NTD : (N, T, D) fp32 C-contig (caller-owned); the fused
                      sgemm uses ``.reshape(N, T*D)``, zero-copy.
    theta           : (N, L) fp32.
    boundary_mask   : (N, L) fp32.
    w, c, beta      : scalars / (L,) fp32.
    v_lambda_session : pre-built MRF Potts close-form Session.
    row_idx_active  : int64 indices of theta-active rows (where
                      ``sum(theta, axis=1) != 0``). Used for the
                      V_lambda candidate-row slice + V_temp scatter.
    log_theta_f32   : fp32 (N, L) cached log(θ); -Inf at θ==0.

Per-EM-iter (cached at top of each ``run`` call):
    cdln_per_k_f32  : (L,) fp32 = ``Cdln(kappa, dim)``; multiplied by
                      T inside the assemble kernel.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from arealmshbm.V_lambda import Session as VLambdaSession
from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32

from . import _kernels


def _bump(d: Optional[dict], key: str, t0: float) -> float:
    """Add (perf_counter() - t0) to d[key] if d is provided. Returns now."""
    now = time.perf_counter()
    if d is not None:
        d[key] = d.get(key, 0.0) + (now - t0)
    return now


# ─────────────────────────────────────────────────────────────────────────
# ELambdaSession.
# ─────────────────────────────────────────────────────────────────────────
class ELambdaSession:
    """Pre-allocated state for the E-step lambda loop fast path.

    Construction caches:
      * data_series in (N, T*D) C-contig fp32 (the fused-sgemm view).
      * log(θ), boundary_mask in fp32.
      * theta-active row indices for the V_lambda candidate set.
      * V_lambda Session (caller-passed).
      * All per-iter scratch.

    Per call (``run``): caller hands in the current per-comp_iter
    Params slices (s_t_nu, kappa, s_lambda, spatial_connect_vmf,
    spatial_xyz_vmf); Session computes Cdln(kappa, dim) once, then
    iterates the lambda loop. Returns ``(s_lambda_new, V_temp,
    lambda_iter)``.

    Mode A only (S=1).
    """

    __slots__ = (
        # Shape constants.
        "N", "D", "T", "L", "dim",
        "epsilon", "max_iter",
        "M_active",
        # Static cache.
        "data_series_NTD",         # (N, T, D) fp32 C-contig — caller-owned ref
        "data_series_NxTD_view",   # (N, T*D) fp32 — zero-copy reshape
        "log_theta_f32",           # (N, L) fp32 — computed once
        "row_idx_active",          # (M_active,) int64 — np.where(sum(theta, 1) != 0)
        "inv_active_idx",          # (N,) int64 — m_active if active, -1 otherwise.
                                    # Used by the mega-fused kernel to gather
                                    # V_lambda_active rows directly without
                                    # materializing the full (N, L) V_temp.
        "boundary_mask_f32",       # (N, L) fp32
        "beta_f32",                # (L,) fp32
        "w_f32", "c_f32",
        "v_lambda_session",
        # Per-call scratch (re-populated each run()).
        "_s_t_nu_TDxL",            # (T*D, L) fp32 — perm + reshape buffer
        "_acc_NL",                 # (N, L) fp32 — fused sgemm out
        "_kappa_f32",              # (L,) fp32
        "_kappa_f64",              # (L,) fp64
        "_cdln_per_k_f32",         # (L,) fp32 — Cdln(kappa, dim)
        "_cdln_per_k_f64",         # (L,) fp64 — scratch for cdln_general_to_f32
        "_cdln_T_f32",             # (L,) fp32 — T * Cdln(kappa, dim)
        "_col_zero_mask",          # (L,) bool
        "_V_temp_full",            # (N, L) fp32 — V_lambda splat target
        "_row_buf",                # (L,) fp32 — mega-fused kernel row
                                    # scratch. Lives in L1 across the 3
                                    # passes of the per-row body.
        # ── s_lambda ping-pong buffers (memcpy-elimination) ──
        # Two stable underlying allocations. ``_s_lambda_curr`` /
        # ``_s_lambda_new`` are Python attributes that point to one or
        # the other; they swap by reassignment (no memcpy) at the end of
        # each λ iter. The output of ``run_lambda_loop`` is a VIEW of
        # whichever buffer is "current" at exit; the caller is expected
        # to pass that view back on the next call. We detect "caller's
        # input is one of our buffers" via ``ctypes.data`` and skip the
        # entry-side copyto, eliminating ~14 GB of memcpy per parcellation.
        "_buf_A",                  # (N, L) fp32 — owns memory
        "_buf_B",                  # (N, L) fp32 — owns memory
        "_buf_A_data",             # int — data pointer of _buf_A (cached)
        "_buf_B_data",             # int — data pointer of _buf_B (cached)
        "_s_lambda_curr",          # alias → _buf_A or _buf_B
        "_s_lambda_new",           # alias → the other of the pair
    )

    def __init__(self,
                 data_series_NTD: np.ndarray,    # (N, T, D) fp32 C-contig
                 theta: np.ndarray,              # (N, L) fp32
                 boundary_mask: np.ndarray,      # (N, L) any float
                 v_lambda_session: VLambdaSession,
                 dim: int,
                 num_clusters: int,
                 num_session: int,
                 w: float,
                 c: float,
                 beta: np.ndarray,
                 epsilon: float = 1e-4,
                 max_iter: int = 50):
        ds = np.ascontiguousarray(data_series_NTD, dtype=np.float32)
        if ds.ndim != 3:
            raise ValueError(
                f"data_series_NTD must be 3D (N, T, D); got {ds.shape}"
            )
        N, T, D = ds.shape
        L = int(num_clusters)
        if T != int(num_session):
            raise ValueError(
                f"data_series_NTD T={T} != num_session={num_session}"
            )

        self.N = int(N)
        self.D = int(D)
        self.T = int(T)
        self.L = L
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.w_f32 = np.float32(w)
        self.c_f32 = np.float32(c)
        self.beta_f32 = np.ascontiguousarray(np.asarray(beta).ravel(), dtype=np.float32)
        if self.beta_f32.shape[0] != L:
            raise ValueError(f"beta length {self.beta_f32.shape[0]} != L={L}")

        # ── data_series held by reference ──
        # The super-call (vmf_clustering) materializes the (N, T, D) fp32
        # C-contig BOLD once (via ``unpack_normalize_packed_NTD_host``)
        # and shares it with both this Session and EMStopSession — so
        # the ``ascontiguousarray`` above is always a no-op in production.
        # No per-Session transpose copy. ``.reshape(N, T*D)`` is a
        # zero-copy view used by the fused sgemm.
        self.data_series_NTD = ds
        self.data_series_NxTD_view = self.data_series_NTD.reshape(N, T * D)

        # ── log(θ) once ──
        th_f32 = np.ascontiguousarray(theta, dtype=np.float32)
        if th_f32.shape != (N, L):
            raise ValueError(f"theta shape {th_f32.shape} != ({N}, {L})")
        self.log_theta_f32 = np.empty((N, L), dtype=np.float32)
        _kernels.compute_log_theta_with_neginf_cap_f32(th_f32, self.log_theta_f32)

        # ── theta-active row indices ──
        theta_row_sum = th_f32.sum(axis=1)            # (N,) fp32
        self.row_idx_active = np.ascontiguousarray(
            np.where(theta_row_sum != 0)[0], dtype=np.int64,
        )
        self.M_active = int(self.row_idx_active.shape[0])
        # Inverse map: full-row index n → m_active (or -1 if inactive).
        # The mega-fused kernel reads this to gather V_lambda_active rows
        # directly, without the splat materialization.
        self.inv_active_idx = np.full(N, -1, dtype=np.int64)
        self.inv_active_idx[self.row_idx_active] = np.arange(
            self.M_active, dtype=np.int64
        )

        # ── boundary_mask in fp32 ──
        self.boundary_mask_f32 = np.ascontiguousarray(boundary_mask, dtype=np.float32)
        if self.boundary_mask_f32.shape != (N, L):
            raise ValueError(
                f"boundary_mask shape {self.boundary_mask_f32.shape} != ({N}, {L})"
            )

        # ── V_lambda Session — caller-built; cached by reference ──
        if v_lambda_session.N != self.M_active:
            raise ValueError(
                f"V_lambda Session N={v_lambda_session.N} != M_active={self.M_active}"
            )
        if v_lambda_session.K != L:
            raise ValueError(
                f"V_lambda Session K={v_lambda_session.K} != L={L}"
            )
        self.v_lambda_session = v_lambda_session

        # ── Per-call scratch ──
        self._s_t_nu_TDxL = np.empty((T * D, L), dtype=np.float32)
        self._acc_NL = np.empty((N, L), dtype=np.float32)
        self._kappa_f32 = np.empty(L, dtype=np.float32)
        self._kappa_f64 = np.empty(L, dtype=np.float64)
        self._cdln_per_k_f32 = np.empty(L, dtype=np.float32)
        self._cdln_per_k_f64 = np.empty(L, dtype=np.float64)
        self._cdln_T_f32 = np.empty(L, dtype=np.float32)
        self._col_zero_mask = np.empty(L, dtype=np.bool_)
        self._V_temp_full = np.empty((N, L), dtype=np.float32)
        # Mega-fused kernel scratch — L1-resident across the 3-pass
        # per-row body (assemble / softmax / normalize+drift).
        self._row_buf = np.empty(L, dtype=np.float32)
        # s_lambda ping-pong: two stable underlying allocations.
        self._buf_A = np.empty((N, L), dtype=np.float32)
        self._buf_B = np.empty((N, L), dtype=np.float32)
        # Cache the data pointers once so the per-call detect is integer
        # comparison (cheap) instead of repeated ctypes attribute access.
        self._buf_A_data = int(self._buf_A.ctypes.data)
        self._buf_B_data = int(self._buf_B.ctypes.data)
        # Initial alias assignment — first call's external input copies into curr.
        self._s_lambda_curr = self._buf_A
        self._s_lambda_new = self._buf_B

    # ─────────────────────────────────────────────────────────────────
    # Two-stage API — separates the heavy per-em-iter work (one matmul +
    # one Cdln) from the per-comp-iter lambda loop. The two pieces of
    # state that the lambda loop reads but does NOT modify are
    # ``s_t_nu`` and ``kappa``; both come from the M-step at the top
    # of an em_iter and are invariant across the entire comp_iter loop.
    # By caching their derivative ``acc = X @ s_t_nu`` we avoid 65+
    # redundant 6.6 GFLOP sgemms per parcellation (sub-001 baseline).
    # ─────────────────────────────────────────────────────────────────

    def prepare_em_iter(self,
                        s_t_nu: np.ndarray,    # (D, L, T) fp32
                        kappa: np.ndarray,     # (L,) fp64 or (1, L)
                        sub_timings: Optional[dict] = None,
                        ) -> None:
        """Compute the em-iter-invariant cache: ``acc = X @ s_t_nu`` and
        ``Cdln(kappa, dim)``. Run once after each M-step.

        These two outputs are NOT read by the M-step or
        spatial_*_prior calls — only by the E-step lambda loop body.
        Caching them on the Session means subsequent comp_iter calls
        (typically 11-13 per em_iter on sub-001) reuse one matmul
        instead of redoing it each time.
        """
        t0 = time.perf_counter()

        # s_t_nu (D, L, T) -> (T, D, L) C-contig view of pre-allocated buffer.
        st = np.ascontiguousarray(s_t_nu, dtype=np.float32)
        if st.ndim != 3 or st.shape != (self.D, self.L, self.T):
            raise ValueError(
                f"s_t_nu shape {st.shape} != ({self.D}, {self.L}, {self.T})"
            )
        view_TDL = self._s_t_nu_TDxL.reshape(self.T, self.D, self.L)
        np.copyto(view_TDL, np.transpose(st, (2, 0, 1)))

        # kappa → fp32 + fp64 staging.
        kap = np.asarray(kappa).ravel()
        if kap.shape[0] != self.L:
            raise ValueError(f"kappa length {kap.shape[0]} != L={self.L}")
        np.copyto(self._kappa_f64, kap, casting="safe")
        np.copyto(self._kappa_f32, kap.astype(np.float32))

        t0 = _bump(sub_timings, "stage_setup", t0)

        # ── acc = data_series_NxTD @ s_t_nu_TDxL (one-time per em_iter) ──
        # The dominant cost at this seam: ~6.6 GFLOP, ~370 ms via MKL sgemm.
        # Move it OUT of the lambda loop body so 79 lambda iters (with
        # invariant s_t_nu) do 6 GEMMs instead of 79.
        np.matmul(self.data_series_NxTD_view, self._s_t_nu_TDxL,
                   out=self._acc_NL)
        t0 = _bump(sub_timings, "mtimesx_acc", t0)

        # ── Column-zero mask on acc (acc is em-iter-invariant) ──
        _kernels.compute_col_zero_mask_f32(self._acc_NL, self._col_zero_mask)

        # ── Cdln(kappa, dim) (em-iter-invariant) ──
        cdln_general_to_f32(self._kappa_f64, self.dim, self._cdln_per_k_f32,
                             scratch_f64=self._cdln_per_k_f64)
        np.multiply(self._cdln_per_k_f32, np.float32(self.T),
                     out=self._cdln_T_f32)

        _bump(sub_timings, "kappa_cdln_combine", t0)

    def run_lambda_loop(self,
                        s_lambda: np.ndarray,                # (N, L) fp32
                        spatial_connect_vmf: np.ndarray,     # (N, L) fp32
                        spatial_xyz_vmf: np.ndarray,         # (N, L) fp32
                        sub_timings: Optional[dict] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run the inner λ-loop until convergence, using the cached acc /
        Cdln from the most recent ``prepare_em_iter`` call. Mode A only —
        S axis dropped from all inputs.

        Returns ``(s_lambda_out, V_temp_full, lambda_iter)``.

        ``s_lambda_out`` is an (N, L) **VIEW** of the Session's
        internal ping-pong buffer — DO NOT mutate it; pass it back on
        the next call to skip the entry-side copyto.

        ``V_temp_full`` is a (N, L) **VIEW** of ``self._V_temp_full`` —
        valid until the next ``run_lambda_loop`` call.
        """
        # ── Stage per-comp_iter inputs ──
        sl = np.ascontiguousarray(s_lambda, dtype=np.float32)
        if sl.ndim != 2 or sl.shape != (self.N, self.L):
            raise ValueError(
                f"s_lambda shape {sl.shape} != ({self.N}, {self.L})"
            )
        sl_2d = sl

        # ── Buffer-shared fast path ──
        # If the caller passed back a view of one of our two ping-pong
        # buffers, alias _s_lambda_curr to it and skip the 94 MB copyto.
        # Otherwise fall through to the copy.
        sl_data_ptr = int(sl_2d.ctypes.data)
        if sl_data_ptr == self._buf_A_data:
            self._s_lambda_curr = self._buf_A
            self._s_lambda_new = self._buf_B
        elif sl_data_ptr == self._buf_B_data:
            self._s_lambda_curr = self._buf_B
            self._s_lambda_new = self._buf_A
        else:
            # External buffer (first call, or caller bypassed the view).
            np.copyto(self._s_lambda_curr, sl_2d)

        scv_2d = np.ascontiguousarray(spatial_connect_vmf, dtype=np.float32)
        if scv_2d.ndim != 2 or scv_2d.shape != (self.N, self.L):
            raise ValueError(
                f"spatial_connect_vmf shape {scv_2d.shape} != ({self.N}, {self.L})"
            )

        sxv_2d = np.ascontiguousarray(spatial_xyz_vmf, dtype=np.float32)
        if sxv_2d.shape != (self.N, self.L):
            raise ValueError(
                f"spatial_xyz_vmf shape {sxv_2d.shape} != ({self.N}, {self.L})"
            )

        # ── Inner λ-loop ──
        checklam = 0.0
        lambda_iter = 0

        while True:
            lambda_iter += 1

            # ── Step 1: V_lambda close-form (no caller-side gather) ──
            # ``compute_full`` consumes the FULL (N, L) s_lambda directly
            # via its row_idx_active table — eliminates the ~90 MB fancy-
            # index allocation that the prior ``compute(curr[mask, :])``
            # path paid every λ-iter.
            t0 = time.perf_counter()
            V_lambda_active = self.v_lambda_session.compute_full(
                self._s_lambda_curr,
            )
            t0 = _bump(sub_timings, "v_lambda_compute", t0)

            # ── Steps 2+3+4+5: mega-fused (V_lambda gather + assemble +
            # softmax + drift). ``inv_active_idx[n]`` substitutes for the
            # splat → V_temp_full → assemble round-trip. ``V_temp_out`` is
            # still written so em_stop_criterion can consume it on the
            # final converged iter (see vmf_clustering.py V_temp_last
            # view propagation).
            checklam_update = float(_kernels.fused_v_lambda_assemble_softmax_drift_f32(
                self._acc_NL,
                self._kappa_f32,
                self._cdln_T_f32,
                self._col_zero_mask,
                self.log_theta_f32, self.w_f32,
                V_lambda_active, self.inv_active_idx, self.c_f32,
                self.beta_f32,
                scv_2d,
                sxv_2d,
                self.boundary_mask_f32,
                self._s_lambda_curr,
                self._row_buf,
                self._V_temp_full,    # written by kernel for em_stop's last consumption
                self._s_lambda_new,
            ))
            self._s_lambda_curr, self._s_lambda_new = (
                self._s_lambda_new, self._s_lambda_curr,
            )
            converged = abs(checklam_update - checklam) <= self.epsilon
            checklam = checklam_update
            _bump(sub_timings, "fused_v_lambda_assemble_softmax_drift", t0)

            if converged:
                break
            if lambda_iter > self.max_iter:
                break

        # Return a 2D (N, L) view of the current ping-pong buffer.
        # Caller passes this view back unchanged on the next call; the
        # entry-side ``ctypes.data`` check then short-circuits the 94 MB
        # copyto. Caller contract: do NOT wrap the returned view in
        # ``np.ascontiguousarray`` / ``np.asarray(..., dtype=...)`` /
        # ``.copy()`` before re-passing — those may silently allocate a
        # new buffer with a different ``ctypes.data``, which defeats the
        # elision. The view is already C-contig fp32.
        return self._s_lambda_curr, self._V_temp_full, lambda_iter


# ─────────────────────────────────────────────────────────────────────────
# Module-level warmup
# ─────────────────────────────────────────────────────────────────────────
def warmup() -> None:
    """Pre-compile the local numba kernels. Idempotent."""
    _kernels.warmup()
