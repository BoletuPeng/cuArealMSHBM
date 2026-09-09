"""session_gpu.py — device-resident Session of the step-2 ``gpu`` backend.

:class:`Step2SparseSession` owns every device buffer of the P-layout step-2
EM and drives the kernels in ``_kernels_gpu``. Nothing bulk crosses the
PCIe bus after the constructor: one EM iteration costs ``m_iters + 1`` blocking
syncs, all of them tiny pinned reads.

Contract: ``docs/step2_sparse_design.md`` §2 (device state), §3 (kernel
catalog) and §4 (this API). The numerical reference is the CPU numba backend.

Layout reminder (design §2.1): every per-cell quantity lives on a ``(P,)``
vector in CSR order, ``P = nnz(boundary_mask)``; ``bm == 1`` on P so the E.1
multiply-by-mask is an exact no-op and the cross-hemisphere ``-inf`` half of
the dense ``log_connect`` does not exist.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised only on CPU-only hosts
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore

from . import _kernels_gpu as K


def _alloc_pinned(shape: Tuple[int, ...], dtype) -> np.ndarray:
    """Page-locked host buffer (cheap, blocking D2H target)."""
    n = int(np.prod(shape)) if shape else 1
    nbytes = n * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(mem, dtype, n).reshape(shape)


class Step2SparseSession:
    """Device-resident step-2 EM session (Mode-B group prior).

    The ctor and :meth:`_stream_subject` refill one pinned staging buffer
    between H2D copies and fence it on the **current** stream, so every
    kernel and copy this Session issues must run on the stream that is
    current at the call.
    """

    def __init__(
        self,
        inputs,
        *,
        mode: str,
        num_clusters: int,
        dim: int,
        ini_val: float,
        beta_internal: float,
        eps_m_step: float,
        max_iter_m: int,
        eps_intra_var: float,
        max_iter_intra_var: int,
        bold_cache_mode: str = "auto",
        bold_cache_safety_margin_gb: float = 4.0,
    ) -> None:
        if cp is None:  # pragma: no cover
            raise RuntimeError("cupy is required for backend='gpu'")
        t_ctor = time.perf_counter()

        lay = inputs.layout
        if int(num_clusters) != lay.L:
            raise ValueError(
                f"num_clusters={num_clusters} != layout.L={lay.L}")
        if mode not in ("gMSHBM", "dMSHBM"):
            raise ValueError(f"unsupported mode {mode!r}")

        self.mode = mode
        self.has_spatial = 1 if mode == "gMSHBM" else 0
        self.S = int(inputs.S)
        self.T = int(inputs.T)
        self.N = int(inputs.N)
        self.D = int(inputs.D)
        self.Db = int(inputs.D_bytes)
        self.L = int(lay.L)
        self.P = int(lay.P)
        self.n_lh = int(lay.n_lh)
        self.L_lh = int(lay.L_lh)
        self.D_grad = int(inputs.D_grad)
        self.dim = int(dim)
        self.ini_val = float(ini_val)
        self.beta_internal = float(beta_internal)
        self.eps_m_step = float(eps_m_step)
        self.max_iter_m = int(max_iter_m)
        self.eps_intra_var = float(eps_intra_var)
        self.max_iter_intra_var = int(max_iter_intra_var)
        self.timings: Dict[str, float] = {}

        K.check_dims(self.D, self.Db, self.L, self.dim,
                     self.D_grad if self.has_spatial else 0)
        if self.has_spatial and inputs.grad_reader is None:
            raise ValueError("gMSHBM needs a gradient reader")

        self._inputs = inputs
        self._mod = K.module()
        self._fn: Dict[str, Any] = {}

        S, T, N, D, Db, L, P = (self.S, self.T, self.N, self.D, self.Db,
                                self.L, self.P)
        Dg = self.D_grad

        # ── static layout ──
        from arealmshbm.step2_io.sparse_layout import layout_to_device
        self._lay = layout_to_device(lay)

        # ── BOLD cache sizing (design §2.2) ──
        self._bold_cache_mode = self._resolve_cache_mode(
            bold_cache_mode, bold_cache_safety_margin_gb)

        pinned_bold = _alloc_pinned((T, N, Db), np.uint8)
        self._pinned_bold = pinned_bold
        if self._bold_cache_mode == "eager_bitpacked":
            self._packed = cp.empty((S, T, N, Db), dtype=cp.uint8)
        else:
            self._packed = cp.empty((1, T, N, Db), dtype=cp.uint8)
        # ``row_mean`` / ``row_inv`` / ``n_alive`` stay device-resident for ALL
        # S even in stream mode: they are 2*S*T*N*4 + S*N*4 bytes (786 MB at
        # S=200) against the 14.5 GB packed cache stream mode exists to avoid,
        # and recomputing them per subject-visit costs a 750 us
        # ``row_stats_exact`` launch twice per subject per EM iteration.
        self._row_mean = cp.empty((S, T, N), dtype=cp.float32)
        self._row_inv = cp.empty((S, T, N), dtype=cp.float32)
        self._resident_s = -1
        self._n_alive = cp.zeros((S, N), dtype=cp.int32)

        inv_D = np.float32(1.0) / np.float32(D)
        self._inv_D = inv_D
        if self._bold_cache_mode == "eager_bitpacked":
            for s in range(S):
                inputs.bold_reader(s + 1, pinned_bold)
                # ``set`` is asynchronous on the current stream and the source
                # is page-locked, so the host must not refill the slot before
                # the copy has landed.
                self._packed[s].set(pinned_bold)
                cp.cuda.get_current_stream().synchronize()
                self._row_stats_into(s, s)
        else:
            for s in range(S):
                self._stream_subject(s)
                self._row_stats_into(0, s)
        self._call("count_alive", (K.grid(S * N, 256),), (256,),
                   (self._row_inv, self._n_alive,
                    np.int32(S), np.int32(T), np.int32(N)))

        # ── gradient (device resident) ──
        if self.has_spatial:
            self._grad = cp.empty((S, N, Dg), dtype=cp.float32)
            self._grad_sq = cp.empty((S, N), dtype=cp.float32)
            pinned_grad = _alloc_pinned((N, Dg), np.float32)
            for s in range(S):
                inputs.grad_reader(s + 1, pinned_grad)
                self._grad[s].set(pinned_grad)
                cp.cuda.get_current_stream().synchronize()
                self._call("grad_sq_rows", (K.grid(N, 256),), (256,),
                           (self._grad[s], self._grad_sq[s],
                            np.int32(N), np.int32(Dg)))
            del pinned_grad
        else:
            self._grad = cp.zeros((1, 1, 1), dtype=cp.float32)
            self._grad_sq = cp.zeros((S, N), dtype=cp.float32)

        # ── group centroids ──
        mtc = np.ascontiguousarray(np.asarray(inputs.mtc, dtype=np.float64))
        self._mtc_DL = cp.asarray(np.ascontiguousarray(mtc.astype(np.float32)))
        self._mtc_LD = cp.asarray(np.ascontiguousarray(mtc.astype(np.float32).T))

        # ── EM state ──
        self._mu = cp.empty((L, D), dtype=cp.float32)
        self._s_psi = cp.empty((S, L, D), dtype=cp.float32)
        self._sigma_psi = cp.empty((S, L, D), dtype=cp.float32)
        self._s_t_nu = cp.empty((S, T, L, D), dtype=cp.float32)
        self._X_dot_sl = cp.empty((S, T, L, D), dtype=cp.float32)
        self._s_lambda = cp.zeros((S, P), dtype=cp.float32)
        self._theta = cp.zeros(P, dtype=cp.float32)
        self._log_theta = cp.zeros(P, dtype=cp.float32)
        self._sigma = cp.full(L, np.float32(ini_val), dtype=cp.float32)
        self._epsil = cp.full(L, np.float32(ini_val), dtype=cp.float32)

        # active support
        self._active = cp.zeros(P, dtype=cp.int32)
        self._flag_csc = cp.zeros(P, dtype=cp.int32)
        self._incl_p = cp.zeros(P + 1, dtype=cp.int32)
        self._incl_csc = cp.zeros(P + 1, dtype=cp.int32)
        self._act_p = cp.zeros(P, dtype=cp.int32)
        self._act_col_ptr = cp.zeros(L + 1, dtype=cp.int32)
        self._act_csc_row = cp.zeros(P, dtype=cp.int32)
        self._act_csc_pidx = cp.zeros(P, dtype=cp.int32)
        self._n_act = P

        # per-subject scratch
        self._scr = cp.zeros(P, dtype=cp.float64)
        self._lv_sum = cp.zeros(P, dtype=cp.float32)
        self._log_vmf = cp.zeros(P, dtype=cp.float32)
        self._log_connect = cp.zeros(P, dtype=cp.float32)
        self._rmax = cp.zeros(N, dtype=cp.float32)
        self._rmax_dense = cp.zeros(N, dtype=cp.float32)
        self._sum_lambda = cp.zeros(L, dtype=cp.float32)
        self._u_LD = cp.zeros((L, max(Dg, 1)), dtype=cp.float32)
        self._u_sq = cp.zeros(L, dtype=cp.float32)
        self._S_TL = cp.zeros((S, T, L), dtype=cp.float64)

        # reductions / scalars
        self._part = cp.zeros(K.REDUCE_GRID, dtype=cp.float64)
        self._cost_nblocks = K.grid(N, K.ROW_BLOCK)
        self._cost_part = cp.zeros(self._cost_nblocks, dtype=cp.float64)
        self._costbuf = cp.zeros(S + 1, dtype=cp.float64)
        self._cost_host = _alloc_pinned((S + 1,), np.float64)
        self._mstate = cp.zeros(4, dtype=cp.float64)
        self._kappa_f32 = cp.zeros(1, dtype=cp.float32)
        self._cdln = cp.zeros(1, dtype=cp.float32)
        self._result = cp.zeros(2 + S * T, dtype=cp.float64)
        self._result_host = _alloc_pinned((2 + S * T,), np.float64)
        self._cos_STL = cp.zeros((S, T, L), dtype=cp.float32)

        # outer-leaf scratch
        self._psiA = cp.empty((S, L, D), dtype=cp.float32)
        self._psiB = cp.empty((S, L, D), dtype=cp.float32)
        self._cos_LS = cp.zeros((L, S), dtype=cp.float32)
        self._eps_mu = cp.empty((L, D), dtype=cp.float32)
        self._sigma_cur = cp.empty(L, dtype=cp.float32)
        self._sigma_new = cp.empty(L, dtype=cp.float32)
        self._flag_psi = cp.zeros(S, dtype=cp.int32)
        self._accL = cp.zeros(L, dtype=cp.float64)
        self._term1 = cp.zeros(L, dtype=cp.float64)
        self._term2 = cp.zeros(L, dtype=cp.float64)
        self._cdln_s = cp.zeros(L, dtype=cp.float64)
        self._cdln_e = cp.zeros(L, dtype=cp.float64)
        self._mu_new = cp.empty((L, D), dtype=cp.float32)
        self._outres = cp.zeros(2, dtype=cp.float64)
        self._outres_host = _alloc_pinned((2,), np.float64)
        self._scalar = cp.zeros(1, dtype=cp.float64)
        self._scalar_host = _alloc_pinned((1,), np.float64)

        # iteration-1 dense scratch (allocated on first use, freed after)
        self._k7_tile = 8192
        self._k7_X = None
        self._k7_lv = None
        self._k7_cross = None

        # scalars carried on the host
        self._theta_out = 0.0
        self._kappa_seed = float(np.float32(ini_val))
        self._kappa_final = float(ini_val)
        self._m_iters = 0
        self._warned_kappa = False
        self._log_eps20 = float(K.LOG_EPS20_F32)
        self._init_dt = K.init_dt()

        cp.cuda.get_current_stream().synchronize()
        self.timings["session_ctor"] = time.perf_counter() - t_ctor

    # ─────────────────────────────────────────────────────────────
    # plumbing
    # ─────────────────────────────────────────────────────────────
    def _f(self, name: str):
        fn = self._fn.get(name)
        if fn is None:
            fn = self._mod.get_function(name)
            self._fn[name] = fn
        return fn

    def _call(self, name, grid, block, args, shared_mem=0):
        self._f(name)(grid, block, args, shared_mem=shared_mem)

    def _resolve_cache_mode(self, requested: str, margin_gb: float) -> str:
        S, T, N, Db, L, D = (self.S, self.T, self.N, self.Db, self.L, self.D)
        P = self.P
        Dg = self.D_grad
        if requested not in ("auto", "eager_bitpacked", "stream"):
            raise ValueError(f"unknown bold_cache_mode {requested!r}")
        packed_b = S * T * N * Db
        grad_b = (S * N * Dg * 4 + S * N * 4) if self.has_spatial else 0
        # Resident per-subject state.  The first two terms are s_t_nu +
        # X_dot_sl and s_psi + sigma_psi; the rest are the S-scaled buffers the
        # design's sizing rule used to omit (1.10 GiB at S=200 / L=300 /
        # D=1175 / N=81924 / P=586522): s_lambda, the intra-closure psi
        # ping-pong, the per-subject row stats and n_alive.
        state_b = (2 * S * T * L * D * 4 + 2 * S * L * D * 4
                   + S * P * 4                      # s_lambda (S,P) f32
                   + 2 * S * L * D * 4              # _psiA / _psiB
                   + 2 * S * T * N * 4              # row_mean / row_inv
                   + S * N * 4)                     # n_alive (S,N) i32
        # iteration-1 scratch: one row tile of the dense pass.
        k7_b = 8192 * (T * D + 2 * L) * 4 + T * D * L * 4
        margin = int(margin_gb * (1 << 30))
        need = packed_b + grad_b + state_b + k7_b + margin
        # ``mem_info`` reports DRIVER-free memory, which does not see blocks the
        # cupy pools are holding but not using (step 1 runs on GPU and nothing
        # drains between step 1 and step 2), so a pooled step-1 arena would make
        # the sizing rule read a free_b that is a whole pool too low.  Reclaim
        # only when the undrained reading is already short: draining can only
        # ADD free memory, so "it fits before the drain" implies "it fits after
        # it", and the drain is not cheap -- it costs ~8 ms here AND throws away
        # the cached PINNED blocks, which makes the ctor's 104 MB of
        # ``alloc_pinned`` a fresh cudaHostAlloc (~18 ms) on every Session.
        free_b = int(cp.cuda.Device().mem_info[0])
        if need > free_b:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            free_b = int(cp.cuda.Device().mem_info[0])
        resident_b = need - packed_b
        if requested == "eager_bitpacked":
            if need > free_b:
                raise ValueError(
                    f"bold_cache_mode='eager_bitpacked' needs "
                    f"{need / 2**30:.2f} GiB (packed {packed_b / 2**30:.2f} + "
                    f"grad {grad_b / 2**30:.2f} + state {state_b / 2**30:.2f} + "
                    f"iter-1 scratch {k7_b / 2**30:.2f} + margin "
                    f"{margin / 2**30:.2f}) but only {free_b / 2**30:.2f} GiB "
                    f"is free; use 'auto' or 'stream'.")
            return "eager_bitpacked"
        # An explicit 'stream' still has to fit the resident state -- otherwise
        # the ctor OOMs on ``self._grad`` with a raw cudaError instead of this
        # named error (design 2.2: "if grad alone does not fit, raise").
        if resident_b > free_b:
            raise ValueError(
                f"backend='gpu' cannot fit the resident state "
                f"({resident_b / 2**30:.2f} GiB without the BOLD cache: "
                f"grad {grad_b / 2**30:.2f} + state {state_b / 2**30:.2f} + "
                f"iter-1 scratch {k7_b / 2**30:.2f} + margin "
                f"{margin / 2**30:.2f}) in {free_b / 2**30:.2f} GiB of free "
                f"device memory.")
        if requested == "stream":
            return "stream"
        if need <= free_b:
            return "eager_bitpacked"
        warnings.warn(
            f"step2 gpu: BOLD cache does not fit ({need / 2**30:.2f} GiB "
            f"needed, {free_b / 2**30:.2f} GiB free) — falling back to "
            f"bold_cache_mode='stream' (2 host->device visits per subject "
            f"per EM iteration).", RuntimeWarning, stacklevel=3)
        return "stream"

    def _row_stats_into(self, pslot: int, sslot: int) -> None:
        """K0 on packed slot ``pslot`` into the stats row of subject ``sslot``."""
        T, N, D, Db = self.T, self.N, self.D, self.Db
        n_rows = T * N
        self._call("row_stats_exact", (K.grid(n_rows, 256),), (256,),
                   (self._packed[pslot], self._row_mean[sslot],
                    self._row_inv[sslot], np.int64(n_rows), np.int32(D),
                    np.int32(Db), np.float32(self._inv_D)))

    def _stream_subject(self, s: int) -> None:
        """Make subject ``s``'s packed bytes resident (stream mode).

        Pure H2D: the row statistics of every subject were computed once in the
        constructor and stay device-resident, so a residency change costs one
        72 MB copy and nothing else.
        """
        if self._bold_cache_mode == "eager_bitpacked":
            return
        if self._resident_s == s:
            return
        cache = self._inputs.packed_host
        if cache is not None:
            np.copyto(self._pinned_bold, cache[s])
        else:
            self._inputs.bold_reader(s + 1, self._pinned_bold)
        self._packed[0].set(self._pinned_bold)
        cp.cuda.get_current_stream().synchronize()   # pinned slot is reused next visit
        self._resident_s = s

    def _slot(self, s: int) -> int:
        return s if self._bold_cache_mode == "eager_bitpacked" else 0

    # ─────────────────────────────────────────────────────────────
    # state setup / resets
    # ─────────────────────────────────────────────────────────────
    def initialize_state(self) -> None:
        """K1 (init) + log_theta + active support; sets every buffer §1.2 zeroes."""
        S, T, N, D, Db, L, P = (self.S, self.T, self.N, self.D, self.Db,
                                self.L, self.P)
        t0 = time.perf_counter()

        self._s_lambda.fill(0)
        self._lv_sum.fill(0)
        self._log_vmf.fill(0)
        self._log_connect.fill(0)
        self._scr.fill(0)
        self._rmax.fill(0)
        self._rmax_dense.fill(0)

        hard = cp.empty(S * N, dtype=cp.int32)
        med = cp.empty(S * N, dtype=cp.uint8)
        dt = self._init_dt
        smem = K.INIT_ROWS * dt * 4      # the ``at`` stage only
        inv_T = np.float32(1.0) / np.float32(T)
        if self._bold_cache_mode == "eager_bitpacked":
            self._call("init_hard_labels",
                       (K.grid(N, K.INIT_ROWS), S), (K.INIT_BLOCK,),
                       (self._packed, self._row_mean, self._row_inv,
                        self._mtc_DL, hard, med,
                        np.int32(S), np.int32(T), np.int32(N), np.int32(D),
                        np.int32(Db), np.int32(L), np.int32(dt),
                        np.float32(inv_T)),
                       shared_mem=smem)
        else:
            for s in range(S):
                self._stream_subject(s)
                self._call("init_hard_labels",
                           (K.grid(N, K.INIT_ROWS), 1), (K.INIT_BLOCK,),
                           (self._packed, self._row_mean[s], self._row_inv[s],
                            self._mtc_DL, hard[s * N:], med[s * N:],
                            np.int32(1), np.int32(T), np.int32(N), np.int32(D),
                            np.int32(Db), np.int32(L), np.int32(dt),
                            np.float32(inv_T)),
                           shared_mem=smem)

        self._call("init_slambda", (K.grid(S * N, 256),), (256,),
                   (hard, med, self._lay["row_ptr"], self._lay["col"],
                    self._s_lambda, np.int32(S), np.int32(N), np.int64(P)))

        eps_f64 = K.EPS_F64
        inv_S_f32 = np.float32(1.0) / np.float32(S)
        theta_on = np.float32(np.float64(inv_S_f32) + eps_f64)
        theta_off = np.float32(eps_f64)
        self._call("init_theta", (K.grid(P, 256),), (256,),
                   (self._s_lambda, self._theta, self._log_theta,
                    self._active, np.int32(S), np.int64(P),
                    np.float32(theta_on), np.float32(theta_off)))
        self._theta_out = float(theta_off)
        self._rebuild_active(fold_slot=None)

        self._sigma.fill(np.float32(self.ini_val))
        self._epsil.fill(np.float32(self.ini_val))
        cp.copyto(self._mu, self._mtc_LD)
        self._broadcast_LD(self._mtc_LD, self._s_psi)
        self._broadcast_LD(self._mtc_LD, self._s_t_nu)
        self._kappa_seed = float(np.float32(self.ini_val))
        self._kappa_final = float(self.ini_val)

        del hard, med
        cp.cuda.get_current_stream().synchronize()
        self.timings["initialize_state"] = time.perf_counter() - t0

    def _broadcast_LD(self, src_LD, dst) -> None:
        n = int(dst.size)
        self._call("broadcast_LD", (K.grid(n, 256),), (256,),
                   (src_LD, dst, np.int32(self.L), np.int32(self.D),
                    np.int64(n)))

    def reset_inter(self) -> None:
        """Reset the inter-subject device state (``sigma``, ``s_psi``)
        to its start-of-inter values. Idempotent; no host transfer."""
        self._sigma.fill(np.float32(self.ini_val))
        self._broadcast_LD(self._mtc_LD, self._s_psi)

    def reset_intra(self) -> None:
        """Reset the intra-subject device state (``kappa`` seed,
        ``s_t_nu``) to its start-of-intra values. Idempotent."""
        self._kappa_seed = float(np.float32(self.ini_val))
        self._broadcast_LD(self._mtc_LD, self._s_t_nu)

    # ─────────────────────────────────────────────────────────────
    # active support (design §2.3)
    # ─────────────────────────────────────────────────────────────
    def _rebuild_active(self, fold_slot: Optional[int]) -> None:
        P, L = self.P, self.L
        self._incl_p[0] = 0
        cp.cumsum(self._active, dtype=cp.int32, out=self._incl_p[1:])
        self._call("compact_scatter", (K.grid(P, 256),), (256,),
                   (self._active, self._incl_p[1:], self._act_p, np.int64(P)))
        self._call("gather_i32", (K.grid(P, 256),), (256,),
                   (self._active, self._lay["csc_pidx"], self._flag_csc,
                    np.int64(P)))
        self._incl_csc[0] = 0
        cp.cumsum(self._flag_csc, dtype=cp.int32, out=self._incl_csc[1:])
        cp.take(self._incl_csc, self._lay["col_ptr"], out=self._act_col_ptr)
        self._call("compact_csc", (K.grid(P, 256),), (256,),
                   (self._flag_csc, self._incl_csc[1:], self._lay["csc_row"],
                    self._lay["csc_pidx"], self._act_csc_row,
                    self._act_csc_pidx, np.int64(P)))
        if fold_slot is None:
            self._n_act = int(self._incl_p[P])
        else:
            self._call("write_scalar_d", (1,), (32,),
                       (self._incl_p[P:], self._costbuf, np.int32(fold_slot)))

    # ─────────────────────────────────────────────────────────────
    # one outer EM iteration
    # ─────────────────────────────────────────────────────────────
    def run_iter(self) -> Tuple[int, np.ndarray]:
        """Run one outer EM iteration in place on the device state.

        Returns ``(m_iters, cost_S)`` — the number of inner M-step
        iterations taken and a fresh host ``(S,)`` fp64 per-subject cost
        array. Call repeatedly; the caller owns the convergence test.
        """
        S, T, N, D, Db, L, P = (self.S, self.T, self.N, self.D, self.Db,
                                self.L, self.P)

        # ── Phase A.1 ──
        n_sp = S * L * D
        self._call("sigma_psi_SLD", (K.grid(n_sp, 256),), (256,),
                   (self._sigma, self._s_psi, self._sigma_psi,
                    np.int32(L), np.int32(D), np.int64(n_sp)))

        # ── Phase A.3 ──
        smem_x = K.XDOT_CHUNK * 8          # member weight + row id (no bytes)
        for s in range(S):
            self._stream_subject(s)
            sl = self._slot(s)
            self._call("x_dot_sl_bits", (T * L,), (K.XDOT_BLOCK,),
                       (self._packed[sl], self._row_mean[s], self._row_inv[s],
                        self._act_col_ptr, self._act_csc_row,
                        self._act_csc_pidx, self._s_lambda[s],
                        self._X_dot_sl[s], np.int32(N), np.int32(L),
                        np.int32(D), np.int32(Db), np.int32(K.XDOT_CHUNK)),
                       shared_mem=smem_x)

        # ── denom ──
        self._call("reduce_sum_f32_stage1", (K.REDUCE_GRID,), (K.REDUCE_BLOCK,),
                   (self._s_lambda, self._part, np.int64(S * P)))
        self._call("reduce_stage2", (1,), (K.REDUCE_BLOCK,),
                   (self._part, self._mstate, np.int32(3),
                    np.int32(K.REDUCE_GRID)))

        # ── Phase B — M-step ──
        self._mstate[0] = self._kappa_seed
        m_iters = self._mstep_loop()
        self._m_iters = m_iters
        if self._kappa_final >= 2308.0 and not self._warned_kappa:
            self._warned_kappa = True
            warnings.warn(
                f"step2 gpu: kappa={self._kappa_final:.1f} is above the "
                f"Cdln(kappa)==kappa crossover (~2308 at dim=1174), so "
                f"tmp_idx := (n_alive == 0) is no longer provably equivalent "
                f"to the CPU's 'log_vmf == 0' row scan.",
                RuntimeWarning, stacklevel=2)

        cdln_v = float(self.dim) * 0.5 - 1.0
        self._call("cdln_after_loop", (1,), (32,),
                   (self._mstate, self._cdln, self._kappa_f32,
                    np.float64(cdln_v)))
        self._call("stnu_row_sum", (S * T * L,), (256,),
                   (self._s_t_nu, self._S_TL, np.int32(D)))

        # ── Phases C + D + E.1, per subject ──
        beta_f32 = np.float32(self.beta_internal)
        use_dense = 1 if self._theta_out != 0.0 else 0
        log_theta_out = (np.float32(math.log(float(self._theta_out)))
                         if use_dense else np.float32(0.0))
        for s in range(S):
            self._stream_subject(s)
            sl = self._slot(s)

            if self.has_spatial:
                self._call("connect_u", (L,), (K.CONNECT_BLOCK,),
                           (self._act_col_ptr, self._act_csc_row,
                            self._act_csc_pidx, self._s_lambda[s],
                            self._grad[s], self._sum_lambda, self._u_LD,
                            self._u_sq, np.int32(self.D_grad)),
                           shared_mem=self.D_grad * 4)
                self._call("connect_scv_P", (K.grid(self._n_act, 256),), (256,),
                           (self._act_p, self._lay["p_row"], self._lay["col"],
                            self._grad[s], self._grad_sq[s], self._u_LD,
                            self._u_sq, self._log_connect,
                            np.int32(self._n_act), np.int32(self.D_grad),
                            np.int32(T)))

            self._call("acc_P", (L,), (K.ACC_BLOCK,),
                       (self._packed[sl], self._row_mean[s], self._row_inv[s],
                        self._s_t_nu[s], self._S_TL[s], self._act_col_ptr,
                        self._act_csc_row, self._act_csc_pidx, self._lv_sum,
                        np.int32(T), np.int32(N), np.int32(D), np.int32(Db),
                        np.int32(L)),
                       shared_mem=Db * 8 * 4)

            if use_dense:
                self._k7_dense_pass(s, beta_f32, log_theta_out)

            self._call("estep_row1", (K.grid(N, K.ROW_BLOCK),), (K.ROW_BLOCK,),
                       (self._lay["row_ptr"], self._lay["col"], self._lv_sum,
                        self._theta, self._log_theta, self._log_connect,
                        self._n_alive[s], self._kappa_f32, self._cdln,
                        self._rmax_dense, np.float32(beta_f32),
                        np.int32(self.has_spatial), np.int32(use_dense),
                        np.int32(N), self._log_vmf, self._scr, self._rmax))
            self._call("dead_col", (L,), (256,),
                       (self._act_col_ptr, self._act_csc_pidx, self._scr))
            self._call("estep_row2", (self._cost_nblocks,), (K.ROW_BLOCK,),
                       (self._lay["row_ptr"], self._scr, self._theta,
                        self._log_vmf, self._log_connect, self._n_alive[s],
                        np.float32(beta_f32), np.int32(self.has_spatial),
                        np.float32(self._log_eps20), np.int32(N),
                        self._s_lambda[s], self._cost_part))
            self._call("reduce_stage2", (1,), (K.REDUCE_BLOCK,),
                       (self._cost_part, self._costbuf, np.int32(s),
                        np.int32(self._cost_nblocks)))

        # ── Phase E.2 + active-support rebuild ──
        inv_S_f32 = np.float32(1.0) / np.float32(S)
        self._call("theta_mean_logtheta", (K.grid(P, 256),), (256,),
                   (self._s_lambda, self._theta, self._log_theta, self._active,
                    np.int32(S), np.int64(P), np.float32(inv_S_f32)))
        self._theta_out = 0.0
        self._rebuild_active(fold_slot=S)

        self._costbuf.get(out=self._cost_host)
        self._n_act = int(self._cost_host[S])
        self._kappa_seed = float(np.float32(self._kappa_final))
        return m_iters, self._cost_host[:S].copy()

    def _mstep_loop(self) -> int:
        S, T, L, D = self.S, self.T, self.L, self.D
        n_stnu = S * T * L * D
        eps_f32 = np.float32(self.eps_m_step)
        eps_f64 = float(np.float64(eps_f32))
        dim_f64 = np.float64(self.dim)
        ini_f64 = np.float64(self.ini_val)
        denom_scale = np.float64(T)
        smem_fused = D * 4
        iter_m = 0
        while True:
            iter_m += 1
            self._call("kappa_sum_stage1", (K.REDUCE_GRID,), (K.REDUCE_BLOCK,),
                       (self._s_t_nu, self._X_dot_sl, self._part,
                        np.int64(n_stnu)))
            self._call("mstep_kappa", (1,), (K.REDUCE_BLOCK,),
                       (self._part, np.int32(K.REDUCE_GRID), self._mstate,
                        self._kappa_f32, self._result, dim_f64, ini_f64,
                        denom_scale))
            self._call("mstep_fused_body", (S * T * L,), (K.FUSED_BLOCK,),
                       (self._kappa_f32, self._X_dot_sl, self._sigma_psi,
                        self._s_t_nu, self._s_t_nu, self._cos_STL,
                        np.int32(T), np.int32(L), np.int32(D)),
                       shared_mem=smem_fused)
            self._call("mstep_flags", (S * T,), (256,),
                       (self._cos_STL, self._result, np.float32(eps_f32),
                        np.int32(L), np.int32(2)))
            self._result.get(out=self._result_host)
            drift = float(self._result_host[0])
            self._kappa_final = float(self._result_host[1])
            all_flag = bool(np.all(self._result_host[2:] == 1.0))
            if all_flag and drift < eps_f64:
                break
            if iter_m > self.max_iter_m:
                break
        return iter_m

    def _k7_dense_pass(self, s: int, beta_f32, log_theta_out) -> None:
        """Iteration-1 out-of-P row max (design §3 K7). Runs once per subject."""
        S, T, N, D, Db, L = (self.S, self.T, self.N, self.D, self.Db, self.L)
        tile = min(self._k7_tile, N)
        if self._k7_X is None:
            self._k7_X = cp.empty((tile, T * D), dtype=cp.float32)
            self._k7_lv = cp.empty((tile, L), dtype=cp.float32)
            self._k7_cross = cp.empty((tile, L), dtype=cp.float32)
        nu_TDL = cp.ascontiguousarray(
            self._s_t_nu[s].transpose(0, 2, 1)).reshape(T * D, L)
        u_T = cp.ascontiguousarray(self._u_LD.T) if self.has_spatial else None
        sl = self._slot(s)
        for off in range(0, N, tile):
            nt = min(tile, N - off)
            self._call("widen_exact", (K.grid(nt * T * 32, 256),), (256,),
                       (self._packed[sl], self._row_mean[s],
                        self._row_inv[s], self._k7_X, np.int32(T),
                        np.int32(N), np.int32(D), np.int32(Db),
                        np.int32(off), np.int32(nt)))
            cp.matmul(self._k7_X[:nt], nu_TDL, out=self._k7_lv[:nt])
            if self.has_spatial:
                cp.matmul(self._grad[s][off:off + nt], u_T,
                          out=self._k7_cross[:nt])
            self._call("rmax_out_k", (K.grid(nt * 32, 256),), (256,),
                       (self._lay["row_ptr"], self._lay["col"], self._k7_lv,
                        self._k7_cross, self._grad_sq[s], self._u_sq,
                        self._n_alive[s], self._kappa_f32, self._cdln,
                        np.float32(log_theta_out), np.float32(beta_f32),
                        np.int32(self.has_spatial), np.int32(self.n_lh),
                        np.int32(self.L_lh), np.int32(L), np.int32(T),
                        np.int32(off), np.int32(nt), self._rmax_dense))
        del nu_TDL, u_T
        if s == S - 1:
            self._k7_X = None
            self._k7_lv = None
            self._k7_cross = None
            cp.get_default_memory_pool().free_all_blocks()

    # ─────────────────────────────────────────────────────────────
    # outer-EM closures
    # ─────────────────────────────────────────────────────────────
    def intra_closure(self) -> float:
        """L17 (``intra_subject_var_loop``) then L16 (``intra_em_cost_step2``)."""
        S, T, L, D = self.S, self.T, self.L, self.D
        t0 = time.perf_counter()
        n_lm = L * D
        self._call("eps_mu_kernel", (K.grid(n_lm, 256),), (256,),
                   (self._epsil, self._mu, self._eps_mu, np.int32(D),
                    np.int64(n_lm)))
        cp.copyto(self._psiA, self._s_psi)
        cp.copyto(self._sigma_cur, self._sigma)
        self._flag_psi.fill(0)

        prev, new = self._psiA, self._psiB
        eps_f32 = np.float32(self.eps_intra_var)
        eps_f64 = float(np.float64(eps_f32))
        smem = D * 4
        for _ in range(self.max_iter_intra_var):
            self._call("intra_psi_iter", (S * L,), (256,),
                       (self._s_t_nu, prev, new, self._sigma_cur, self._eps_mu,
                        self._cos_LS, np.int32(S), np.int32(T), np.int32(L),
                        np.int32(D)),
                       shared_mem=smem)
            prev, new = new, prev
            self._call("intra_sigma_dot", (L,), (256,),
                       (prev, self._s_t_nu, self._accL, np.int32(S),
                        np.int32(T), np.int32(L), np.int32(D)))
            self._call("invad_L", (L,), (32,),
                       (self._accL, self._sigma_cur, self._sigma_new,
                        np.float64(1.0 / float(S * T)), np.float64(self.dim),
                        np.float64(self.ini_val)))
            self._call("intra_flags_rel", (1,), (K.FLAG_BLOCK,),
                       (self._cos_LS, self._flag_psi, self._sigma_cur,
                        self._sigma_new, self._outres, np.float32(eps_f32),
                        np.int32(S), np.int32(L)))
            self._outres.get(out=self._outres_host)
            if (int(self._outres_host[0]) == S
                    and float(self._outres_host[1]) < eps_f64):
                break
        cp.copyto(self._s_psi, prev)
        cp.copyto(self._sigma, self._sigma_cur)

        # L16 — intra_em_cost_step2
        cdln_v = float(self.dim) * 0.5 - 1.0
        self._call("cost_terms_perl", (L,), (256,),
                   (self._s_psi, self._s_t_nu, self._mu, self._term1,
                    self._term2, np.int32(S), np.int32(T), np.int32(L),
                    np.int32(D)))
        self._call("cdln_L", (L,), (32,),
                   (self._sigma, self._cdln_s, np.float64(cdln_v)))
        self._call("cdln_L", (L,), (32,),
                   (self._epsil, self._cdln_e, np.float64(cdln_v)))
        self._call("cost_epilogue", (1,), (32,),
                   (self._term1, self._term2, self._cdln_s, self._cdln_e,
                    self._sigma, self._epsil, self._costbuf, self._scalar,
                    np.int32(S), np.int32(T), np.int32(L)))
        self._scalar.get(out=self._scalar_host)
        self.timings["intra_closure"] = (self.timings.get("intra_closure", 0.0)
                                         + time.perf_counter() - t0)
        return float(self._scalar_host[0])

    def inter_closure(self) -> None:
        """L18 (``inter_subject_var``): mu then epsil."""
        S, L, D = self.S, self.L, self.D
        t0 = time.perf_counter()
        self._call("inter_mu", (L,), (256,),
                   (self._s_psi, self._mu, self._mu_new, np.int32(S),
                    np.int32(L), np.int32(D)),
                   shared_mem=D * 8)
        cp.copyto(self._mu, self._mu_new)
        self._call("inter_eps_dot", (L,), (32,),
                   (self._s_psi, self._mu, self._accL, np.int32(S),
                    np.int32(L), np.int32(D)))
        self._call("invad_L", (L,), (32,),
                   (self._accL, self._epsil, self._sigma_new,
                    np.float64(1.0 / float(S)), np.float64(self.dim),
                    np.float64(self.ini_val)))
        cp.copyto(self._epsil, self._sigma_new)
        self.timings["inter_closure"] = (self.timings.get("inter_closure", 0.0)
                                         + time.perf_counter() - t0)

    # ─────────────────────────────────────────────────────────────
    # host views
    # ─────────────────────────────────────────────────────────────
    @property
    def kappa(self) -> float:
        """The last M-step's converged concentration (host scalar)."""
        return float(self._kappa_final)

    @property
    def n_active(self) -> int:
        """Number of live cells in the compacted theta support."""
        return int(self._n_act)

    @property
    def bold_cache_mode(self) -> str:
        """``'eager_bitpacked'`` or ``'stream'`` — how the session
        holds the per-subject BOLD, chosen at ctor from device free
        memory."""
        return self._bold_cache_mode

    def export_params(self) -> Dict[str, Any]:
        """Host snapshot of the saved ``Params`` fields, in the shapes
        :func:`step2_pipeline._save.save_params_final` writes: ``sigma``
        / ``epsil`` / ``kappa`` ``(1, L)``, ``mu`` ``(L, D)``,
        ``cost_em`` ``(S,)``, ``theta`` an ``(N, L)`` fp64 scipy
        ``csc_matrix``, ``ini_val`` a float. Pure read — callable more
        than once, and it does not disturb the device state."""
        import scipy.sparse as sp
        lay = self._inputs.layout
        L = self.L
        theta = cp.asnumpy(self._theta).astype(np.float64)
        keep = theta != 0.0
        csc = sp.csc_matrix(
            (theta[keep], (lay.p_row[keep], lay.col[keep])),
            shape=(self.N, L))
        return {
            "sigma": cp.asnumpy(self._sigma).reshape(1, L),
            "epsil": cp.asnumpy(self._epsil).reshape(1, L),
            "kappa": np.full((1, L), np.float32(self._kappa_final),
                             dtype=np.float32),
            "mu": cp.asnumpy(self._mu),
            "cost_em": self._cost_host[:self.S].copy(),
            "theta": csc,
            "ini_val": float(self.ini_val),
        }

    def sync_to_host(self, fields: Iterable[str]) -> Dict[str, np.ndarray]:
        """Copy the named device arrays to fresh host arrays.

        Recognised names are the device state fields (see the
        dispatch below); ``theta`` / ``s_lambda`` come back scattered to
        dense ``(N, L)``, their ``*_P`` twins in P layout. An unknown
        name raises ``ValueError``. Diagnostics only — the pipeline reads
        ``export_params`` instead."""
        lay = self._inputs.layout
        out: Dict[str, np.ndarray] = {}
        for f in fields:
            if f == "s_t_nu":
                out[f] = cp.asnumpy(self._s_t_nu)
            elif f == "s_psi":
                out[f] = cp.asnumpy(self._s_psi)
            elif f == "sigma_psi":
                out[f] = cp.asnumpy(self._sigma_psi)
            elif f == "X_dot_sl":
                out[f] = cp.asnumpy(self._X_dot_sl)
            elif f == "sigma":
                out[f] = cp.asnumpy(self._sigma)
            elif f == "epsil":
                out[f] = cp.asnumpy(self._epsil)
            elif f == "mu":
                out[f] = cp.asnumpy(self._mu)
            elif f == "s_lambda":
                out[f] = lay.scatter(cp.asnumpy(self._s_lambda))
            elif f == "s_lambda_P":
                out[f] = cp.asnumpy(self._s_lambda)
            elif f == "theta":
                out[f] = lay.scatter(cp.asnumpy(self._theta))
            elif f == "theta_P":
                out[f] = cp.asnumpy(self._theta)
            elif f == "log_theta":
                out[f] = cp.asnumpy(self._log_theta)
            elif f == "lv_sum":
                out[f] = cp.asnumpy(self._lv_sum)
            elif f == "log_vmf":
                out[f] = cp.asnumpy(self._log_vmf)
            elif f == "log_connect":
                out[f] = cp.asnumpy(self._log_connect)
            elif f == "scr":
                out[f] = cp.asnumpy(self._scr)
            elif f == "rmax":
                out[f] = cp.asnumpy(self._rmax)
            elif f == "rmax_dense":
                out[f] = cp.asnumpy(self._rmax_dense)
            elif f == "sum_lambda":
                out[f] = cp.asnumpy(self._sum_lambda)
            elif f == "u":
                out[f] = cp.asnumpy(self._u_LD)
            elif f == "u_sq":
                out[f] = cp.asnumpy(self._u_sq)
            elif f == "grad_sq":
                out[f] = cp.asnumpy(self._grad_sq)
            elif f == "n_alive":
                out[f] = cp.asnumpy(self._n_alive)
            elif f == "row_mean":
                out[f] = cp.asnumpy(self._row_mean)
            elif f == "row_inv":
                out[f] = cp.asnumpy(self._row_inv)
            elif f == "active":
                out[f] = cp.asnumpy(self._active)
            elif f == "cost_em":
                out[f] = self._cost_host[:self.S].copy()
            elif f == "kappa":
                out[f] = np.float64(self._kappa_final)
            else:
                raise ValueError(f"sync_to_host: unknown field {f!r}")
        return out


__all__ = ["Step2SparseSession"]
