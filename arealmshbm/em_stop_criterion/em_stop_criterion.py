"""em_stop_criterion.py

EM stop-criterion block: cost reduction + convergence test that closes
the outer EM loop in :class:`vmf_clustering.VmfClusteringSession`.

Public API:
    EMStopSession         — pre-allocated workspace + ``compute()`` hot
                            path; single-pass fused kernel that assembles,
                            cleans, and reduces in one (N, L) sweep.
    matlab_ratio_converged — shared MATLAB-faithful ratio convergence
                            helper. Also called by ``intra_em``.
    convergence_test      — small Python helper for the convergence
                            threshold (also used by the GPU path).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import _kernels
from ._cdln import cdln_general_to_f32


def matlab_ratio_converged(update_cost: float, cost: float,
                            threshold: float = 1e-4) -> bool:
    """``not (|update_cost - cost| / |cost| > threshold)``, with MATLAB's
    NaN/Inf branch semantics.

    * ``0 / 0``      -> NaN — treated as converged because
                        ``not (NaN > threshold)`` is True.
    * ``nonzero / 0`` -> Inf — not converged.
    * else            -> standard ratio.

    Hardcoded threshold mirrors MATLAB's epsilon (1e-4). Used by both
    the inner EM convergence test and the outer intra_em loop —
    extracted here so the two sites can't drift if MATLAB's rule ever
    changes.

    The CPU path uses plain Python arithmetic (no numpy division) so no
    ``np.errstate`` is needed; the GPU path (vmf_clustering_gpu.py) uses
    numpy float64 division and keeps its own errstate guard.
    """
    if cost == 0.0:
        ratio = float("nan") if update_cost == 0.0 else float("inf")
    else:
        ratio = abs((update_cost - cost) / cost)
    return not (ratio > threshold)   # NaN > threshold is False -> True


def convergence_test(update_cost,
                     cost,
                     iter_em: int,
                     ) -> Tuple[int, Optional[np.ndarray]]:
    """``stop_em = converged or iter_em > 100``. MATLAB lines 747-755.

    Single-subject pipeline: ``update_cost`` and ``cost`` are scalars
    (Python ``float`` / 0-d / shape ``(1,)`` / ``(1, 1)``). Multi-element
    arrays are an error — passing one (e.g. an old multi-subject GT
    replay) used to silently take ``arr[0]``.

    NaN-as-converged: ``NaN > 1e-4`` is False, so NaN ratios are treated
    as converged (matches MATLAB).

    Returns:
        (stop_em, cost_em):
            stop_em : 0 or 1.
            cost_em : ``cost`` if converged via threshold, ``update_cost``
                      if forced via iter_em > 100, else None.
    """
    uc = _scalarize(update_cost, "update_cost")
    co = _scalarize(cost, "cost")
    converged = matlab_ratio_converged(uc, co)
    stop_em = 0
    cost_em: Optional[np.ndarray] = None
    if converged:
        stop_em = 1
        cost_em = np.asarray(cost).copy()
    # Hard ceiling at 100 — mirrors MATLAB step3 line 757. The outer
    # ``VmfClusteringSession.run`` loop also tests its own configurable
    # ``max_iter_em``, which acts as a soft cap for callers that want to
    # stop earlier (max_iter_em < 100). Setting max_iter_em > 100 has no
    # effect — this hard cap fires first.
    if iter_em > 100:
        stop_em = 1
        cost_em = np.asarray(update_cost).copy()
    return stop_em, cost_em


def _scalarize(x, name: str) -> float:
    """Accept Python scalar / 0-d / (1,) / (1, 1) and return a fp64 float.

    Multi-element arrays raise — single-subject pipeline.
    """
    if isinstance(x, (int, float)):
        return float(x)
    arr = np.asarray(x).ravel()
    if arr.size != 1:
        raise ValueError(
            f"convergence_test: single-subject expects scalar {name}; "
            f"got size={arr.size}"
        )
    return float(arr[0])


class EMStopSession:
    """Pre-allocated workspace + ``compute(...)`` hot path.

    Caches ``data.series``, ``log_theta_cost``, and per-call scratch
    buffers; ``compute()`` allocates nothing. The cached
    ``log_lambda_prop`` and cleaned ``spatial_connect_vmf`` are returned
    as VIEWS into session buffers — copy if you need to keep them across
    calls.
    """

    __slots__ = (
        "N", "D", "T", "L",
        "dim", "num_session", "num_clusters",
        "w", "c", "beta_f32",
        "data_series_NTD",        # (N, T, D) fp32 C-contig — caller-owned ref
        "data_series_NxTD_view",  # (N, T*D) fp32 — zero-copy reshape
        "log_theta_cost_f32",     # (N, L) fp32 — computed once
        "_acc",                   # (N, L) fp32 — fused-sgemm out
        "_spatial_connect_vmf",   # (N, L) fp32 — cleaned scv (out of fused kernel)
        "_kappa_f64_buf",         # (L,) fp64 — cdln_general_to_f32 input
        "_cdln_per_k_f64",        # (L,) fp64 — cdln scratch
        "_cdln_per_k",            # (L,) fp32 — cdln output (kernel input)
        "_s_t_nu_per_t",          # (T, D, L) fp32 — permuted s_t_nu
        "_update_cost_buf",       # (1,) fp64 — kernel scalar out
    )

    def __init__(self,
                 data_series_NTD: np.ndarray,
                 theta: np.ndarray,
                 dim: int,
                 num_session: int,
                 num_clusters: int,
                 w: float,
                 c: float,
                 beta: np.ndarray):
        # Mode A only — S axis dropped throughout. Caller (vmf_clustering
        # super-call) materializes (N, T, D) fp32 C-contig once (via
        # ``unpack_normalize_packed_NTD_host``) and shares it with us
        # and ELambdaSession — so this is always a no-op in production.
        ds = np.ascontiguousarray(data_series_NTD, dtype=np.float32)
        if ds.ndim != 3:
            raise ValueError(
                f"data_series_NTD must be 3D (N, T, D); got {ds.shape}"
            )
        N, T, D = ds.shape
        if T != int(num_session):
            raise ValueError(
                f"data_series_NTD T={T} != num_session={num_session}"
            )
        L = int(num_clusters)
        if L <= 0:
            raise ValueError(f"num_clusters must be positive; got {L}")

        self.N = int(N)
        self.D = int(D)
        self.T = int(T)
        self.L = L
        self.dim = int(dim)
        self.num_session = int(num_session)
        self.num_clusters = int(num_clusters)
        self.w = float(w)
        self.c = float(c)
        self.beta_f32 = np.ascontiguousarray(np.asarray(beta).ravel(),
                                              dtype=np.float32)
        if self.beta_f32.shape[0] != L:
            raise ValueError(
                f"beta length {self.beta_f32.shape[0]} != num_clusters={L}"
            )

        # data_series held by reference. Caller already owns (N, T, D)
        # C-contig fp32; no per-Session transpose. ``.reshape(N, T*D)``
        # is a zero-copy view used by the fused sgemm:
        #     X_flat @ nu_flat = sum_t sum_d X[n,t,d] * nu[t,d,l]
        self.data_series_NTD = ds
        self.data_series_NxTD_view = self.data_series_NTD.reshape(N, T * D)

        # log_theta_cost is session-invariant — compute once.
        th = np.ascontiguousarray(theta, dtype=np.float32)
        if th.shape != (N, L):
            raise ValueError(f"theta shape {th.shape} != ({N}, {L})")
        self.log_theta_cost_f32 = np.empty_like(th)
        _kernels.log_with_neginf_floor(th, self.log_theta_cost_f32)

        self._acc = np.empty((N, L), dtype=np.float32)
        self._spatial_connect_vmf = np.empty((N, L), dtype=np.float32)
        self._kappa_f64_buf = np.empty(L, dtype=np.float64)
        self._cdln_per_k_f64 = np.empty(L, dtype=np.float64)
        self._cdln_per_k = np.empty(L, dtype=np.float32)
        self._s_t_nu_per_t = np.empty((T, D, L), dtype=np.float32)
        self._update_cost_buf = np.empty(1, dtype=np.float64)

    def compute(self,
                s_t_nu: np.ndarray,
                kappa: np.ndarray,
                s_lambda: np.ndarray,
                spatial_connect_vmf: np.ndarray,
                V_temp: np.ndarray,
                cost: np.ndarray,
                iter_em: int,
                ) -> Tuple[float, int, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Run the full block. Mode A only — S axis dropped from inputs.

        Inputs:
            s_t_nu              : (D, L, T) any float dtype.
            kappa               : (L,) or (1, L) any float dtype.
            s_lambda            : (N, L) any float dtype.
            spatial_connect_vmf : (N, L) any float dtype.
            V_temp              : (N, L) any float dtype. **Read-only.**
                                  In the super-call this is a view of
                                  ELambdaSession's ``_V_temp_full`` —
                                  the caller must not write it (or any
                                  buffer aliasing it) between the
                                  preceding ``run_lambda_loop`` and this
                                  call, or the cost integration sees
                                  wrong V_temp.
            cost                : (1, 1) or (1,) any float dtype.
            iter_em             : int.
        Returns:
            (update_cost, stop_em, cost_em, log_lambda_prop_view, scv_view).
            ``update_cost`` is a Python float (fp64). ``cost_em`` is
            ``None`` unless ``stop_em == 1``. ``log_lambda_prop_view`` is
            always ``None`` (the fused kernel never materializes the
            (N, L) llp intermediate). ``scv_view`` is a view of the
            session's cached cleaned-scv buffer (overwritten on the
            next call).
        """
        s_lam = self._stage_s_lambda(s_lambda)
        scv = self._stage_spatial_connect_vmf(spatial_connect_vmf)
        Vt = self._stage_V_temp(V_temp)
        kap = self._stage_kappa(kappa)
        nu_per_t = self._stage_s_t_nu_per_t(s_t_nu)

        # Step 1 (BLAS): acc = data_series_NxTD @ nu_TDxL via one fused
        # sgemm (X (N, T*D) @ nu (T*D, L) = sum_t (X_t @ nu_t)).
        np.copyto(self._kappa_f64_buf, kap, casting="safe")
        cdln_general_to_f32(self._kappa_f64_buf, self.dim, self._cdln_per_k,
                            scratch_f64=self._cdln_per_k_f64)
        np.matmul(self.data_series_NxTD_view,
                  nu_per_t.reshape(self.T * self.D, self.L),
                  out=self._acc)

        # Steps 2 + 3 + 4 fused into one nopython kernel:
        #   * inline assemble log_lambda_prop = T*cdln + kappa*acc
        #     (no (N, L) intermediate — value computed and consumed in
        #     the same inner loop iteration)
        #   * inline cleanup_spatial_connect_vmf (NaN/Inf -> floor),
        #     written into ``self._spatial_connect_vmf`` for the caller
        #   * inline 5-term cost reduction with inline log(s_lambda)
        # Saves one (N, L) fp32 streaming pass (~94 MB) per call.
        _kernels.fused_em_stop_assemble_cleanup_cost_f32(
            self._acc, kap, self._cdln_per_k, self.T,
            s_lam, self.log_theta_cost_f32, Vt,
            scv, self._spatial_connect_vmf,
            self.w, self.c, self.beta_f32,
            self._update_cost_buf,
        )
        update_cost_scalar = float(self._update_cost_buf[0])

        # Step 5: convergence test (single-subject). Pass the fp64
        # scalar directly — the prior implementation wrapped it in a
        # fp32 array which dropped ~7 fp32 ULPs of precision for no
        # reason. ``convergence_test`` accepts Python scalars.
        stop_em, cost_em = convergence_test(update_cost_scalar, cost,
                                            iter_em=int(iter_em))

        # ``_log_lambda_prop`` is no longer materialized in the fused path.
        # The super-call only consumes the cleaned spatial_connect_vmf
        # (it propagates the NaN/Inf cleanup back into outer state); the
        # llp view returned by the prior implementation had no consumer
        # outside diagnostic / validate scripts. Return None to signal
        # absence; super-call's existing ``_llp`` capture is unused.
        return (update_cost_scalar, stop_em, cost_em,
                None, self._spatial_connect_vmf)

    def _stage_s_lambda(self, s_lambda: np.ndarray) -> np.ndarray:
        a = np.asarray(s_lambda)
        if a.shape != (self.N, self.L):
            raise ValueError(f"s_lambda shape {a.shape} != ({self.N}, {self.L})")
        return np.ascontiguousarray(a, dtype=np.float32)

    def _stage_spatial_connect_vmf(self, vmf: np.ndarray) -> np.ndarray:
        a = np.asarray(vmf)
        if a.shape != (self.N, self.L):
            raise ValueError(
                f"spatial_connect_vmf shape {a.shape} != ({self.N}, {self.L})"
            )
        return np.ascontiguousarray(a, dtype=np.float32)

    def _stage_V_temp(self, V_temp: np.ndarray) -> np.ndarray:
        a = np.asarray(V_temp)
        if a.shape != (self.N, self.L):
            raise ValueError(f"V_temp shape {a.shape} != ({self.N}, {self.L})")
        return np.ascontiguousarray(a, dtype=np.float32)

    def _stage_kappa(self, kappa: np.ndarray) -> np.ndarray:
        a = np.asarray(kappa).ravel()
        if a.shape[0] != self.L:
            raise ValueError(f"kappa length {a.shape[0]} != L={self.L}")
        return np.ascontiguousarray(a, dtype=np.float32)

    def _stage_s_t_nu_per_t(self, s_t_nu: np.ndarray) -> np.ndarray:
        """Permute (D, L, T) -> (T, D, L) C-contig. Mode A only — S axis
        dropped from the external API.
        """
        a = np.asarray(s_t_nu)
        if a.ndim != 3 or a.shape != (self.D, self.L, self.T):
            raise ValueError(
                f"s_t_nu shape {a.shape} != ({self.D}, {self.L}, {self.T})"
            )
        permuted = np.transpose(a, (2, 0, 1))   # (T, D, L) view
        np.copyto(self._s_t_nu_per_t, permuted)
        return self._s_t_nu_per_t
