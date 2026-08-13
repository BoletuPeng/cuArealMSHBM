"""e_step_lambda_gpu.py

Partial-GPU port of :class:`ELambdaSession` — only the λ-loop runs on
device. Static state (BOLD, theta, log_theta, boundary_mask,
neighborhood, V_lambda candidate set) is pinned on GPU at construction;
the per-call API takes / returns numpy arrays, with H2D / D2H hidden
inside.

The inner λ-loop runs entirely on device: V_lambda close-form fused
kernel plus the assemble+softmax+drift kernel from
:mod:`._kernels_gpu`. M-step / EMStop / spatial_priors /
check_connectedness stay on CPU, so per comp_iter (N, L) buffers
ping-pong across PCIe. The zero-ping-pong path is
:class:`VmfClusteringSessionCUDA` in :mod:`.vmf_clustering_gpu`.

Cdln(kappa, dim) is computed CPU-side (scipy Bessel, L=300, ~1 ms) then
H2D'd alongside ``s_t_nu`` / ``kappa``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cupy as cp
import numpy as np

from arealmshbm.V_lambda import Session as VLambdaSession
from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32

from . import _kernels       # CPU kernels — for log_theta one-shot
from . import _kernels_gpu


def _bump(d: Optional[dict], key: str, t0: float) -> float:
    """Add (perf_counter() - t0) to d[key] if d is provided. Returns now."""
    now = time.perf_counter()
    if d is not None:
        d[key] = d.get(key, 0.0) + (now - t0)
    return now


class ELambdaSessionCUDA:
    """GPU port of :class:`ELambdaSession`.

    External API matches the CPU class — same constructor signature,
    same ``prepare_em_iter`` + ``run_lambda_loop`` shape contract — so
    the super-call wires it up via env-var dispatch with no other code
    changes.

    Internally:
      * BOLD, log(θ), boundary_mask, neighborhood, candidate (row, col)
        indices, beta — all GPU-resident from construction.
      * Per-em_iter: H2D ``s_t_nu`` (8 MB) + ``kappa`` (2.4 KB), compute
        Cdln on CPU (small), one cuBLAS sgemm for ``acc_NL``.
      * Per-comp_iter: H2D ``s_lambda`` + ``scv`` + ``sxv`` (3 × 94 MB),
        run the inner λ-loop on GPU until convergence, D2H ``s_lambda``
        + ``V_temp`` (2 × 94 MB).
      * V_lambda Session is **embedded** — its 3 buffers (neighborhood
        transposed, row_idx, col_idx, pre-zeroed V_lam_active) are also
        GPU-resident. No CPU V_lambda call inside the loop.

    Mode A only.
    """

    __slots__ = (
        # Shape constants (mirrors CPU class).
        "N", "D", "T", "L", "dim", "epsilon", "max_iter", "M_active",
        # GPU-resident static state.
        "data_series_NTD_dev",         # (N, T, D) fp32 on device
        "data_series_NxTD_view_dev",   # (N, T*D) fp32 — zero-copy reshape
        "log_theta_dev",               # (N, L) fp32
        "boundary_mask_dev",           # (N, L) fp32
        "beta_dev",                    # (L,) fp32
        "w_f32", "c_f32",
        "row_idx_active_dev",          # (M_active,) int64 on device
        "inv_active_idx_dev",          # (N,) int64 on device
        # V_lambda session state on GPU.
        "vlambda_neighborhood_NM_dev", # (M_active, M1) int64
        "vlambda_row_idx_dev",         # (P,) int64
        "vlambda_col_idx_dev",         # (P,) int64
        "vlambda_V_lam_dev",           # (M_active, L) fp32 — pre-zeroed
        # Per-em_iter scratch on GPU.
        "_s_t_nu_TDxL_dev",            # (T*D, L) fp32 — buffer for the sgemm
        "_acc_NL_dev",                 # (N, L) fp32 — sgemm output
        "_kappa_dev",                  # (L,) fp32
        "_cdln_per_k_dev",             # (L,) fp32
        "_cdln_T_dev",                 # (L,) fp32
        "_col_zero_mask_dev",          # (L,) bool
        # Per-comp_iter scratch on GPU.
        "_V_temp_full_dev",            # (N, L) fp32
        "_buf_A_dev", "_buf_B_dev",    # (N, L) fp32 — s_lambda ping-pong
        "_s_lambda_curr_dev",          # alias
        "_s_lambda_new_dev",           # alias
        "_scv_dev",                    # (N, L) fp32 — H2D scv
        "_sxv_dev",                    # (N, L) fp32 — H2D sxv
        # CPU staging buffers (pinned for fast PCIe transfer; numpy for now).
        "_s_lambda_host_buf",          # (N, L) fp32 — D2H staging
        "_V_temp_host_buf",            # (N, L) fp32 — D2H staging
        # Backref to a CPU V_lambda Session — needed for compatibility with
        # the super-call's constructor signature; not used internally.
        "_cpu_v_lambda_session",
    )

    def __init__(self,
                 data_series_NTD: np.ndarray,
                 theta: np.ndarray,
                 boundary_mask: np.ndarray,
                 v_lambda_session: VLambdaSession,
                 dim: int,
                 num_clusters: int,
                 num_session: int,
                 w: float,
                 c: float,
                 beta: np.ndarray,
                 epsilon: float = 1e-4,
                 max_iter: int = 50):
        # Super-call materializes (N, T, D) fp32 C-contig once via
        # ``unpack_normalize_packed_NTD_host`` and shares it across all
        # sub-Sessions — so this is always a no-op in production.
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

        self.N = int(N); self.D = int(D); self.T = int(T); self.L = L
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.w_f32 = np.float32(w)
        self.c_f32 = np.float32(c)

        # ── H2D the static state ──
        self.data_series_NTD_dev = cp.asarray(ds)
        self.data_series_NxTD_view_dev = self.data_series_NTD_dev.reshape(N, T * D)

        th_f32 = np.ascontiguousarray(theta, dtype=np.float32)
        if th_f32.shape != (N, L):
            raise ValueError(f"theta shape {th_f32.shape} != ({N}, {L})")
        # Compute log(θ) on CPU (one-shot), then H2D — saves writing a CUDA
        # log-with-floor kernel for a single call.
        log_theta_cpu = np.empty((N, L), dtype=np.float32)
        _kernels.compute_log_theta_with_neginf_cap_f32(th_f32, log_theta_cpu)
        self.log_theta_dev = cp.asarray(log_theta_cpu)

        bm = np.ascontiguousarray(boundary_mask, dtype=np.float32)
        if bm.shape != (N, L):
            raise ValueError(f"boundary_mask shape {bm.shape} != ({N}, {L})")
        self.boundary_mask_dev = cp.asarray(bm)

        self.beta_dev = cp.asarray(np.ascontiguousarray(np.asarray(beta).ravel(),
                                                          dtype=np.float32))
        if self.beta_dev.shape[0] != L:
            raise ValueError(f"beta length {self.beta_dev.shape[0]} != L={L}")

        # Active row indices (device). The boolean mask itself is unused
        # after construction — kept as a local.
        theta_row_active_mask = (th_f32.sum(axis=1) != 0)
        row_idx_active = np.ascontiguousarray(
            np.where(theta_row_active_mask)[0], dtype=np.int64
        )
        self.M_active = int(row_idx_active.shape[0])
        self.row_idx_active_dev = cp.asarray(row_idx_active)
        inv_active_idx = np.full(N, -1, dtype=np.int64)
        inv_active_idx[row_idx_active] = np.arange(self.M_active, dtype=np.int64)
        self.inv_active_idx_dev = cp.asarray(inv_active_idx)

        # ── V_lambda state on GPU ──
        # The CPU Session holds neighborhood_NM (transposed for cache-friendly
        # CPU access). On GPU we use the same layout — fancy-indexing pattern
        # is the same.
        if v_lambda_session.N != self.M_active:
            raise ValueError(
                f"V_lambda Session N={v_lambda_session.N} != M_active={self.M_active}"
            )
        if v_lambda_session.K != L:
            raise ValueError(
                f"V_lambda Session K={v_lambda_session.K} != L={L}"
            )
        self._cpu_v_lambda_session = v_lambda_session  # kept for sanity / debug
        self.vlambda_neighborhood_NM_dev = cp.asarray(v_lambda_session.neighborhood_NM)
        self.vlambda_row_idx_dev = cp.asarray(v_lambda_session.row_idx)
        self.vlambda_col_idx_dev = cp.asarray(v_lambda_session.col_idx)
        # Pre-zeroed V_lam — non-candidate entries stay 0 forever.
        self.vlambda_V_lam_dev = cp.zeros((self.M_active, L), dtype=cp.float32)

        # ── Per-em_iter scratch ──
        self._s_t_nu_TDxL_dev = cp.empty((T * D, L), dtype=cp.float32)
        self._acc_NL_dev = cp.empty((N, L), dtype=cp.float32)
        self._kappa_dev = cp.empty(L, dtype=cp.float32)
        self._cdln_per_k_dev = cp.empty(L, dtype=cp.float32)
        self._cdln_T_dev = cp.empty(L, dtype=cp.float32)
        self._col_zero_mask_dev = cp.empty(L, dtype=cp.bool_)

        # ── Per-comp_iter scratch ──
        self._V_temp_full_dev = cp.empty((N, L), dtype=cp.float32)
        self._buf_A_dev = cp.empty((N, L), dtype=cp.float32)
        self._buf_B_dev = cp.empty((N, L), dtype=cp.float32)
        self._s_lambda_curr_dev = self._buf_A_dev
        self._s_lambda_new_dev = self._buf_B_dev
        self._scv_dev = cp.empty((N, L), dtype=cp.float32)
        self._sxv_dev = cp.empty((N, L), dtype=cp.float32)

        # ── Host staging buffers ──
        # Used as the D2H landing pad. Keeping them as Session attributes
        # avoids per-call numpy allocation. Pinned-host allocation would be
        # marginally faster but adds a dependency; revisit if H2D dominates.
        self._s_lambda_host_buf = np.empty((N, L), dtype=np.float32)
        self._V_temp_host_buf = np.empty((N, L), dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────
    # API — mirrors the CPU class
    # ─────────────────────────────────────────────────────────────────
    def prepare_em_iter(self,
                        s_t_nu: np.ndarray,
                        kappa: np.ndarray,
                        sub_timings: Optional[dict] = None,
                        ) -> None:
        """H2D s_t_nu/kappa, run one cuBLAS sgemm for ``acc = X @ s_t_nu``,
        compute Cdln on CPU then H2D it.
        """
        t0 = time.perf_counter()

        st = np.ascontiguousarray(s_t_nu, dtype=np.float32)
        if st.shape != (self.D, self.L, self.T):
            raise ValueError(
                f"s_t_nu shape {st.shape} != ({self.D}, {self.L}, {self.T})"
            )
        # Permute to (T, D, L) C-contig on host, then H2D into the (T*D, L)
        # buffer view. Could also do the permute on device but the host
        # permute is ~5 ms vs an extra H2D + on-device transpose.
        st_TDL = np.ascontiguousarray(np.transpose(st, (2, 0, 1)))
        self._s_t_nu_TDxL_dev.set(st_TDL.reshape(self.T * self.D, self.L))

        kap = np.ascontiguousarray(np.asarray(kappa).ravel(), dtype=np.float32)
        if kap.shape[0] != self.L:
            raise ValueError(f"kappa length {kap.shape[0]} != L={self.L}")
        self._kappa_dev.set(kap)

        t0 = _bump(sub_timings, "stage_setup", t0)

        # ── Fused sgemm: (N, T·D) @ (T·D, L) → (N, L) via cuBLAS ──
        cp.matmul(self.data_series_NxTD_view_dev,
                   self._s_t_nu_TDxL_dev,
                   out=self._acc_NL_dev)
        # Column-zero mask on acc.
        cp.all(self._acc_NL_dev == 0, axis=0, out=self._col_zero_mask_dev)
        cp.cuda.get_current_stream().synchronize()
        t0 = _bump(sub_timings, "mtimesx_acc", t0)

        # ── Cdln on CPU then H2D ──
        kap_f64 = kap.astype(np.float64)
        cdln_per_k = np.empty(self.L, dtype=np.float32)
        cdln_general_to_f32(kap_f64, self.dim, cdln_per_k)
        self._cdln_per_k_dev.set(cdln_per_k)
        cp.multiply(self._cdln_per_k_dev, np.float32(self.T),
                     out=self._cdln_T_dev)
        cp.cuda.get_current_stream().synchronize()
        _bump(sub_timings, "kappa_cdln_combine", t0)

    def run_lambda_loop(self,
                        s_lambda: np.ndarray,
                        spatial_connect_vmf: np.ndarray,
                        spatial_xyz_vmf: np.ndarray,
                        sub_timings: Optional[dict] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run the inner λ-loop on GPU. Returns numpy arrays.

        Per call:
          1. H2D s_lambda, scv, sxv (3 × 94 MB ≈ 24 ms at 12 GB/s).
          2. GPU loop: V_lambda fused + mega-mega-fused, ~10 ms / λ-iter.
          3. D2H s_lambda + V_temp (2 × 94 MB ≈ 50 ms at 4 GB/s).

        Returns ``(s_lambda_np, V_temp_np, lambda_iter)`` — both arrays
        are FRESH numpy buffers (not views) so the caller can hold them
        across subsequent ``run_lambda_loop`` calls without aliasing
        worries; on the next call we overwrite the device-side buffers
        but the returned host arrays survive.
        """
        # ── H2D inputs ──
        t0 = time.perf_counter()

        sl = np.ascontiguousarray(s_lambda, dtype=np.float32)
        if sl.shape != (self.N, self.L):
            raise ValueError(
                f"s_lambda shape {sl.shape} != ({self.N}, {self.L})"
            )
        scv = np.ascontiguousarray(spatial_connect_vmf, dtype=np.float32)
        if scv.shape != (self.N, self.L):
            raise ValueError(
                f"scv shape {scv.shape} != ({self.N}, {self.L})"
            )
        sxv = np.ascontiguousarray(spatial_xyz_vmf, dtype=np.float32)
        if sxv.shape != (self.N, self.L):
            raise ValueError(
                f"sxv shape {sxv.shape} != ({self.N}, {self.L})"
            )

        # H2D — load s_lambda into A, scv / sxv into their dedicated buffers.
        # Reset the ping-pong alias so curr is always A at the start of a call.
        self._s_lambda_curr_dev = self._buf_A_dev
        self._s_lambda_new_dev = self._buf_B_dev
        self._buf_A_dev.set(sl)
        self._scv_dev.set(scv)
        self._sxv_dev.set(sxv)
        cp.cuda.get_current_stream().synchronize()
        t0 = _bump(sub_timings, "h2d_per_comp_iter", t0)

        # ── Inner λ-loop ──
        checklam = 0.0
        lambda_iter = 0
        while True:
            lambda_iter += 1

            # V_lambda fused (3-phase close-form Potts) — reads from
            # s_lambda_curr_dev sliced at active rows.
            t_v0 = time.perf_counter()
            tmp_lambda_dev = self._s_lambda_curr_dev[self.row_idx_active_dev]
            _kernels_gpu.vlambda_potts_closeform_fused_cupy(
                self.vlambda_neighborhood_NM_dev,
                tmp_lambda_dev,
                self.vlambda_row_idx_dev,
                self.vlambda_col_idx_dev,
                self.vlambda_V_lam_dev,
            )
            cp.cuda.get_current_stream().synchronize()
            t_v0 = _bump(sub_timings, "v_lambda_compute", t_v0)

            # Mega-mega-fused: V_lambda gather + assemble + softmax + drift.
            checklam_update = _kernels_gpu.fused_v_lambda_assemble_softmax_drift_cupy(
                self._acc_NL_dev,
                self._kappa_dev,
                self._cdln_T_dev,
                self._col_zero_mask_dev,
                self.log_theta_dev,
                float(self.w_f32),
                self.vlambda_V_lam_dev,
                self.inv_active_idx_dev,
                float(self.c_f32),
                self.beta_dev,
                self._scv_dev,
                self._sxv_dev,
                self.boundary_mask_dev,
                self._s_lambda_curr_dev,
                self._V_temp_full_dev,
                self._s_lambda_new_dev,
            )
            cp.cuda.get_current_stream().synchronize()
            _bump(sub_timings, "fused_v_lambda_assemble_softmax_drift", t_v0)

            # Ping-pong swap.
            self._s_lambda_curr_dev, self._s_lambda_new_dev = (
                self._s_lambda_new_dev, self._s_lambda_curr_dev,
            )
            converged = abs(checklam_update - checklam) <= self.epsilon
            checklam = checklam_update
            if converged:
                break
            if lambda_iter > self.max_iter:
                break

        # ── D2H outputs ──
        t1 = time.perf_counter()
        self._s_lambda_curr_dev.get(out=self._s_lambda_host_buf)
        self._V_temp_full_dev.get(out=self._V_temp_host_buf)
        cp.cuda.get_current_stream().synchronize()
        _bump(sub_timings, "d2h_per_comp_iter", t1)

        # Return fresh copies so downstream code holds independent storage.
        return (self._s_lambda_host_buf.copy(),
                self._V_temp_host_buf.copy(),
                lambda_iter)


def warmup() -> None:
    """Pre-compile / pre-warm the GPU kernels."""
    _kernels_gpu.warmup()
