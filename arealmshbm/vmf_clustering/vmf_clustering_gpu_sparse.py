"""vmf_clustering_gpu_sparse.py

``backend='gpu_sparse'`` — the candidate-set (P-layout) step-3 EM.
Everything the EM touches lives on the device for the **whole
intra_em outer loop** (not just one EM body): ``s_lambda``, the spatial
priors, ``s_t_nu``, ``s_psi``, ``xyz_gamma``. The host only sees
scalars (convergence tests, ``invad``) until the final outputs are
gathered.

Contract: ``docs/step3_sparse_design.md``. Reference semantics: the CPU
:class:`VmfClusteringSession` + :class:`Step3Pipeline.run` outer loop.
Per-stage math lives in :mod:`._kernels_gpu_sparse`,
:mod:`arealmshbm.m_step.m_step_gpu_sparse` and
:mod:`arealmshbm.check_connectedness.connectedness_gpu`.

Variant support: gMSHBM (production) and dMSHBM. cMSHBM's
``remove_isolated_surface_components`` pre-predicate has no device
port yet — constructing this Session with a cMSHBM spec raises.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import numpy as np

from arealmshbm.em_stop_criterion import matlab_ratio_converged
from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32
from arealmshbm.spatial_priors.spatial_xyz import compute_unit_sphere_xyz
from arealmshbm.step3_pipeline.variant import VariantSpec

from . import _kernels_gpu_sparse as K
from .sparse_layout import MAX_D_BYTES, CandidateLayout, layout_to_device


class VmfClusteringSessionSparseCUDA:
    """Device-resident single-subject step-3 EM on the candidate set."""

    def __init__(self, *,
                 layout: CandidateLayout,
                 packed_TND: np.ndarray,          # (T, N, ⌈D/8⌉) uint8, on-disk layout
                 D_unpacked: int,
                 mw_rows: np.ndarray,             # full vertex ids to zero (both hemis)
                 mu: np.ndarray,                  # (D, L) fp32
                 epsil: np.ndarray,               # (L,)
                 sigma: np.ndarray,               # (L,)
                 grad_data: Optional[np.ndarray], # (N, Dg) fp32 — gMSHBM
                 sphere_xyz: Optional[np.ndarray],
                 lh_sphere_mesh: Optional[Dict[str, np.ndarray]],
                 rh_sphere_mesh: Optional[Dict[str, np.ndarray]],
                 dim: int, num_clusters: int, num_session: int,
                 w: float, c: float, beta: np.ndarray,
                 ini_val: float,
                 connect_th: float = 15.0, epsilon: float = 1e-4,
                 max_iter_em: int = 100, max_iter_lambda: int = 50,
                 max_iter_m: int = 50, max_iter_comp: int = 15,
                 variant_spec: Optional[VariantSpec] = None):
        from arealmshbm.m_step.m_step_gpu_sparse import compute_row_stats, MStepGPU

        if variant_spec is None:
            variant_spec = VariantSpec.from_pipeline_type("gMSHBM")
        if variant_spec.pre_predicate_remove_isolated:
            raise NotImplementedError(
                "gpu_sparse: cMSHBM's remove_isolated pre-predicate is not ported; "
                "use backend='gpu_full' or 'cpu' for cMSHBM")
        self.variant = variant_spec

        packed_TND = np.asarray(packed_TND)
        if packed_TND.dtype != np.uint8 or packed_TND.ndim != 3:
            raise ValueError("packed_TND must be (T, N, D_bytes) uint8")
        T, N, Db = packed_TND.shape
        D = int(D_unpacked)
        L = int(num_clusters)
        if T != int(num_session):
            raise ValueError(f"packed_TND T={T} != num_session={num_session}")
        if N != layout.N or L != layout.L:
            raise ValueError("layout / data shape mismatch")
        if Db != (D + 7) // 8:
            raise ValueError("packed_TND last axis != ceil(D/8)")
        if Db > MAX_D_BYTES:
            raise ValueError(
                f"gpu_sparse: profile dimension D={D} exceeds the backend's "
                f"limit of {8 * MAX_D_BYTES} (ceil(D/8)={Db} > {MAX_D_BYTES} "
                f"bytes); use backend='gpu_full' or 'cpu'")
        if variant_spec.use_connect_prior and grad_data is None:
            raise ValueError("gMSHBM needs grad_data")
        if variant_spec.use_check_connectedness and (
                sphere_xyz is None or lh_sphere_mesh is None or rh_sphere_mesh is None):
            raise ValueError("variant needs sphere_xyz + LH/RH sphere meshes")

        self.N, self.T, self.D, self.L, self.Db = N, T, D, L, Db
        self.M = int(layout.M_active); self.P = int(layout.P)
        self.n_lh, self.L_lh = N // 2, L // 2
        self.dim = int(dim)
        self.w, self.c = float(w), float(c)
        self.ini_val = float(ini_val)
        self.connect_th = float(connect_th)
        self.epsilon = float(epsilon)
        self.max_iter_em = int(max_iter_em)
        self.max_iter_lambda = int(max_iter_lambda)
        self.max_iter_m = int(max_iter_m)
        self.max_iter_comp = int(max_iter_comp)

        # ── static device state ──
        self.lay = layout_to_device(layout)
        self.layout_host = layout
        self.p_row_m = K.p_row_m_of(self.lay)
        self.packed = cp.asarray(packed_TND)
        mw = np.ascontiguousarray(np.asarray(mw_rows).ravel(), dtype=np.int32)
        if mw.size:
            K.zero_rows_packed(self.packed, cp.asarray(mw))
        self.row_mean, self.row_inv = compute_row_stats(self.packed, D)

        # log(θ) on P, computed on the HOST in fp64 and cast, like numba's
        # ``float32(math.log(v))``. cupy's elementwise kernels (``log`` and
        # even ``astype``) flush fp32 denormals to zero; θ carries a few
        # 1e-45..1e-39 cells, which would become -Inf on device while the
        # CPU backend has a finite ~-100 there.
        self.log_theta = cp.asarray(
            np.log(np.asarray(layout.theta, dtype=np.float64)).astype(np.float32))
        self.log_theta_cost = self.log_theta                # same values on P (θ > 0)
        self.beta = cp.asarray(np.ascontiguousarray(np.asarray(beta).ravel(), dtype=np.float32))
        if self.beta.shape[0] != L:
            raise ValueError("beta length != L")
        mu32 = np.ascontiguousarray(mu, dtype=np.float32)
        if mu32.shape != (D, L):
            raise ValueError(f"mu shape {mu32.shape} != ({D}, {L})")
        # mu is never updated by the EM; the output Params hand back this
        # host array, as the CPU / gpu_full backends hand back theirs.
        self.mu_host = mu32
        self.mu_LD = cp.asarray(np.ascontiguousarray(mu32.T))
        self.epsil = cp.asarray(np.ascontiguousarray(np.asarray(epsil).ravel(), dtype=np.float32))
        self.sigma = cp.asarray(np.ascontiguousarray(np.asarray(sigma).ravel(), dtype=np.float32))
        # cdln(σ), cdln(ε) for the intra_em cost — constants.
        cs = np.empty(L, np.float32); ce = np.empty(L, np.float32)
        cdln_general_to_f32(np.asarray(sigma, np.float64).ravel(), self.dim, cs)
        cdln_general_to_f32(np.asarray(epsil, np.float64).ravel(), self.dim, ce)
        self._term1_cdln = float(T) * float(cs.astype(np.float64).sum())
        self._term2_cdln = float(ce.astype(np.float64).sum())
        self._sigma_h = np.asarray(sigma, np.float64).ravel()
        self._epsil_h = np.asarray(epsil, np.float64).ravel()

        if variant_spec.use_connect_prior:
            gd = np.ascontiguousarray(grad_data, dtype=np.float32)
            if gd.shape[0] != N:
                raise ValueError("grad_data rows != N")
            self.grad = cp.asarray(gd)
            self.grad_sq = cp.power(self.grad, 2, dtype=cp.float64).sum(axis=1).astype(cp.float32)
            self.Dg = int(gd.shape[1])
            self.u = cp.empty((L, self.Dg), cp.float32)
            self.u_sq = cp.empty(L, cp.float32)
        else:
            self.grad = None; self.Dg = 0

        if variant_spec.use_check_connectedness:
            from arealmshbm.check_connectedness.connectedness_gpu import ConnectednessGPU
            self.sphere = cp.asarray(compute_unit_sphere_xyz(sphere_xyz))
            self.conn = ConnectednessGPU(
                lh_sphere_mesh["vertexNbors"], rh_sphere_mesh["vertexNbors"],
                lh_sphere_mesh["vertices"], rh_sphere_mesh["vertices"],
                L, self.connect_th, int(variant_spec.components_threshold))
            self.labels = cp.empty(N, cp.int32)
        else:
            self.sphere = None; self.conn = None

        # ── EM state on P ──
        z = lambda: cp.zeros(self.P, cp.float32)
        self.s_lambda_A = cp.asarray(layout.theta)     # initial s_lambda = θ
        self.s_lambda_B = z()
        self.V_temp = z()
        self.acc = z()
        self.scv = z()
        self.sxv = z()                                  # initial spatial_xyz_vmf = 0
        self.zero_P = z()
        self.s_psi_LD = self.mu_LD.copy()          # s_psi init = mu, mutable
        self.xyz_gamma = cp.zeros(L, cp.float64)
        self.s_muc = cp.zeros((3, L), cp.float32)
        self.cdln3 = cp.empty(L, cp.float32)
        self.gamma_f32 = cp.empty(L, cp.float32)
        # per-em_iter scalars/vectors
        self.kappa_f32 = cp.empty(L, cp.float32)
        self.cdln = cp.empty(L, cp.float32)
        self.cdln_T = cp.empty(L, cp.float32)
        self.S_TL = cp.empty((T, L), cp.float64)
        self.allzero_TL = cp.empty((T, L), cp.uint8)
        self.anynan_TL = cp.empty((T, L), cp.uint8)
        self.col_zero = cp.empty(L, cp.uint8)
        self.col_nan = cp.empty(L, cp.uint8)
        self.row_poison = cp.zeros(self.M, cp.uint8)
        self.per_col1 = cp.empty(L, cp.float64)
        self.per_col2 = cp.empty(L, cp.float64)
        self.s_psi_tmp = cp.empty((L, D), cp.float32)
        self.summed_LD = cp.empty((L, D), cp.float32)
        self._dense_host: Optional[np.ndarray] = None   # pinned (N, L) staging
        self.ws_e = K.EStepWorkspace(self.M)
        self.ws_stop = K.EMStopWorkspace(self.P)
        self.mstep = MStepGPU(T, D, L, self.dim, self.epsilon, self.max_iter_m,
                              self.packed, self.row_mean, self.row_inv, self.lay)
        self.s_t_nu = None          # (T, L, D) view into the M-step ping-pong
        # Reset value of s_t_nu at the top of every intra_em round: mu
        # broadcast over t. mu is fixed, so build it once.
        self.s_t_nu_init = cp.empty((T, L, D), cp.float32)
        K.broadcast_mu(self.mu_LD, self.s_t_nu_init)
        self._cdln_scratch = np.empty(L, np.float64)
        self._cdln_host = np.empty(L, np.float32)
        self._kappa_host = np.empty(L, np.float64)
        self.last_timings: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────
    # EM body (one vmf_clustering super-call) on device state
    # ─────────────────────────────────────────────────────────────
    def _prepare_em_iter(self, kappa: float) -> bool:
        """acc / Cdln / column flags. Returns any(col_nan)."""
        L, T = self.L, self.T
        self._kappa_host.fill(kappa)
        cdln_general_to_f32(self._kappa_host, self.dim, self._cdln_host, self._cdln_scratch)
        self.cdln.set(self._cdln_host)
        cp.multiply(self.cdln, np.float32(T), out=self.cdln_T)
        self.kappa_f32.fill(np.float32(kappa))
        K.stnu_col_stats(self.s_t_nu, self.S_TL, self.allzero_TL, self.anynan_TL,
                         self.col_zero, self.col_nan)
        K.acc_bits(self.lay, self.packed, self.row_mean, self.row_inv, self.s_t_nu,
                   self.S_TL, self.D, self.acc)
        any_nan = bool(self.col_nan.any())
        if any_nan:
            K.row_poison(self.lay, self.col_nan, self.row_poison)
        else:
            self.row_poison.fill(0)
        return any_nan

    def _lambda_loop(self, curr, new, scv, sxv) -> Tuple[Any, Any, int]:
        eps = self.epsilon
        checklam = 0.0
        it = 0
        while True:
            it += 1
            upd = K.estep_iteration(
                self.lay, self.ws_e, curr, self.acc, self.kappa_f32, self.cdln_T,
                self.col_zero, self.log_theta, self.w, self.c, self.beta, scv, sxv,
                self.lay["bm"], self.row_poison, self.V_temp, new)
            curr, new = new, curr
            converged = abs(upd - checklam) <= eps
            checklam = upd
            if converged or it > self.max_iter_lambda:
                break
        return curr, new, it

    def _em_body(self, timings: Dict[str, Any]) -> Dict[str, Any]:
        v = self.variant
        sync = cp.cuda.get_current_stream().synchronize
        kappa = self.ini_val
        self.s_t_nu = self.s_t_nu_init
        curr, new = self.s_lambda_A, self.s_lambda_B
        cost_prev = 0.0
        max_conn = 0.0; max_comp = 0.0
        cost_em_out: Optional[np.ndarray] = None
        update_cost = 0.0
        iter_em = 0
        scv_estep = self.scv if v.use_connect_prior else (self.sxv if v.use_xyz_prior else self.zero_P)
        sxv_estep = self.sxv if v.use_connect_prior else self.zero_P
        while True:
            iter_em += 1
            timings["iter_count_em_total"] = iter_em

            t0 = time.perf_counter()
            self.s_t_nu, kappa, iter_m = self.mstep.run(
                self.s_t_nu, curr, self.s_psi_LD, self.sigma, kappa)
            sync()
            timings["m_step"] += time.perf_counter() - t0
            timings["iter_count_m_total"] += iter_m

            t0 = time.perf_counter()
            any_nan = self._prepare_em_iter(kappa)
            sync()
            timings["e_step_lambda_loop"] += time.perf_counter() - t0

            if v.use_connect_prior:
                t0 = time.perf_counter()
                K.connect_prior(self.lay, self.p_row_m, curr, self.grad, self.grad_sq,
                                self.u, self.u_sq, self.scv)
                sync()
                timings["spatial_connect_prior"] += time.perf_counter() - t0
            elif v.use_xyz_prior:
                t0 = time.perf_counter()
                K.xyz_prior(self.lay, self.p_row_m, curr, self.sphere, self.xyz_gamma,
                            self.s_muc, self.cdln3, self.gamma_f32, self.sxv)
                sync()
                timings["spatial_xyz_prior"] += time.perf_counter() - t0
                timings["iter_count_spatial_xyz"] += 1

            if not v.wrap_comp_iter:
                t0 = time.perf_counter()
                curr, new, n_lam = self._lambda_loop(curr, new, scv_estep, sxv_estep)
                timings["e_step_lambda_loop"] += time.perf_counter() - t0
                timings["iter_count_lambda_total"] += n_lam
                timings["iter_count_comp_total"] += 1
            else:
                comp_iter = 0
                while True:
                    comp_iter += 1
                    timings["iter_count_comp_total"] += 1
                    t0 = time.perf_counter()
                    curr, new, n_lam = self._lambda_loop(curr, new, scv_estep, sxv_estep)
                    timings["e_step_lambda_loop"] += time.perf_counter() - t0
                    timings["iter_count_lambda_total"] += n_lam
                    run_conn = (v.use_check_connectedness
                                and iter_em >= v.first_em_iter_with_conn_xyz)
                    stop_comp = False
                    if run_conn:
                        t0 = time.perf_counter()
                        K.argmax_labels(self.lay, curr, self.labels)
                        max_conn, max_comp = self.conn.step(self.labels, self.xyz_gamma)
                        timings["check_connectedness"] += time.perf_counter() - t0
                        timings["iter_count_check_conn"] += 1
                        if v.use_xyz_prior:
                            t0 = time.perf_counter()
                            K.xyz_prior(self.lay, self.p_row_m, curr, self.sphere,
                                        self.xyz_gamma, self.s_muc, self.cdln3,
                                        self.gamma_f32, self.sxv)
                            sync()
                            timings["spatial_xyz_prior"] += time.perf_counter() - t0
                            timings["iter_count_spatial_xyz"] += 1
                    else:
                        stop_comp = True
                    if max_conn <= self.connect_th and max_comp <= float(v.components_threshold):
                        stop_comp = True
                    if comp_iter >= self.max_iter_comp:
                        stop_comp = True
                    if stop_comp:
                        break

            t0 = time.perf_counter()
            scv_stop = self.scv if v.em_stop_uses_beta_scv else self.zero_P
            update_cost = K.em_stop_cost(
                self.lay, self.ws_stop, self.acc, self.kappa_f32, self.cdln, self.T, curr,
                self.log_theta_cost, self.V_temp, scv_stop, self.w, self.c, self.beta)
            if any_nan:
                update_cost = float("nan")
            timings["em_stop_criterion"] += time.perf_counter() - t0
            stop_em = 0
            if matlab_ratio_converged(update_cost, cost_prev):
                stop_em = 1
                cost_em_out = np.array([[cost_prev]], dtype=np.float32)
            if iter_em > 100:
                stop_em = 1
                cost_em_out = np.array([[update_cost]], dtype=np.float32)
            if stop_em:
                break
            if iter_em > self.max_iter_em:
                cost_em_out = np.array([[update_cost]], dtype=np.float32)
                break
            cost_prev = float(np.float32(update_cost))     # MATLAB keeps cost in fp32

        # Publish ping-pong state.
        self.s_lambda_A, self.s_lambda_B = curr, new
        return {"kappa": kappa, "iter_em": iter_em, "cost_em": cost_em_out,
                "max_connectedness": max_conn, "max_components": max_comp,
                "update_cost": update_cost}

    # ─────────────────────────────────────────────────────────────
    # intra_em outer loop (Step3Pipeline.run semantics)
    # ─────────────────────────────────────────────────────────────
    def run_intra_em(self, max_iter_intra_em: int, time_stages: bool = True,
                     emit_dense_priors: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the whole outer loop on device. Returns ``(Params, info)``
        where ``Params`` follows the CPU pipeline's output dict (dense
        ``s_lambda``; ``s_t_nu`` as (D, L, T); flat (L,) vectors) and
        ``info`` carries ``record``, ``iter_intra_em``, ``per_stage_em``,
        ``per_iter_intra_em``.
        """
        L, T, D = self.L, self.T, self.D
        per_stage = {
            "m_step": 0.0, "spatial_connect_prior": 0.0, "e_step_lambda_loop": 0.0,
            "check_connectedness": 0.0, "spatial_xyz_prior": 0.0, "em_stop_criterion": 0.0,
            "iter_count_em_total": 0, "iter_count_m_total": 0, "iter_count_lambda_total": 0,
            "iter_count_comp_total": 0, "iter_count_check_conn": 0, "iter_count_spatial_xyz": 0,
        }
        per_iter: list = []
        record: list = []
        cost = 0.0
        iter_intra = 0
        em: Dict[str, Any] = {}
        cost_intra = 0.0
        while True:
            iter_intra += 1
            t_round = time.perf_counter()
            body_t = {k: 0.0 if isinstance(v, float) else 0 for k, v in per_stage.items()}
            em = self._em_body(body_t)
            for k, vv in body_t.items():
                per_stage[k] += vv
            # intra_subject_var + intra_em_cost on device
            K.intra_psi(self.s_t_nu, self.sigma, self.epsil, self.mu_LD, self.s_psi_tmp,
                        self.summed_LD, self.per_col1, self.per_col2)
            self.s_psi_LD, self.s_psi_tmp = self.s_psi_tmp, self.s_psi_LD
            pc1 = self.per_col1.get(); pc2 = self.per_col2.get()
            term1 = float((self._sigma_h * pc1).sum()) + self._term1_cdln
            term2 = float((self._epsil_h * pc2).sum()) + self._term2_cdln
            ce = em["cost_em"]
            update_cost = term1 + term2 + (float(np.asarray(ce).ravel().sum()) if ce is not None else 0.0)
            record.append(update_cost)
            converged = matlab_ratio_converged(update_cost, cost)
            stop = converged or iter_intra >= int(max_iter_intra_em)
            cost = update_cost
            per_iter.append(time.perf_counter() - t_round)
            if stop:
                cost_intra = update_cost
                break

        params = self._collect_params(em, emit_dense_priors)
        params["cost_intra"] = cost_intra
        params["Record"] = np.asarray(record, dtype=np.float64)
        info = {"record": record, "iter_intra_em": iter_intra,
                "per_stage_em": per_stage, "per_iter_intra_em": per_iter}
        if time_stages:
            self.last_timings = dict(per_stage)
        return params, info

    def labels_host(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lh_labels, rh_labels) int64 from the current s_lambda (device argmax)."""
        lab = cp.empty(self.N, cp.int32)
        K.argmax_labels(self.lay, self.s_lambda_A, lab)
        lab_h = lab.get().astype(np.int64)
        return lab_h[:self.n_lh], lab_h[self.n_lh:]

    def dense_P(self, x_P: cp.ndarray) -> np.ndarray:
        """Dense (N, L) host copy of a P-vector (fresh numpy array)."""
        dense = cp.zeros((self.N, self.L), cp.float32)
        K.scatter_dense(self.lay, self.p_row_m, x_P, dense)
        return dense.get()

    def dense_P_pinned(self, x_P: cp.ndarray) -> np.ndarray:
        """Like :meth:`dense_P` but lands in a session-owned pinned host
        buffer (PCIe-rate D2H, ~3x faster than pageable). The returned
        array is overwritten by the next call; the pinned block stays
        alive as long as any returned view does."""
        if self._dense_host is None:
            mem = cp.cuda.alloc_pinned_memory(self.N * self.L * 4)
            self._dense_host = np.frombuffer(mem, dtype=np.float32,
                                             count=self.N * self.L).reshape(self.N, self.L)
        dense = cp.zeros((self.N, self.L), cp.float32)
        K.scatter_dense(self.lay, self.p_row_m, x_P, dense)
        dense.get(out=self._dense_host)
        return self._dense_host

    def _collect_params(self, em: Dict[str, Any], emit_dense_priors: bool) -> Dict[str, Any]:
        L, T, D = self.L, self.T, self.D
        p: Dict[str, Any] = {}
        p["s_lambda"] = self.dense_P_pinned(self.s_lambda_A)
        st = self.s_t_nu.get()                                  # (T, L, D)
        p["s_t_nu"] = np.ascontiguousarray(np.transpose(st, (2, 1, 0)))   # (D, L, T)
        p["s_psi"] = np.ascontiguousarray(self.s_psi_LD.get().T)
        p["kappa"] = np.full(L, float(em["kappa"]), dtype=np.float64)
        p["sigma"] = self.sigma.get(); p["epsil"] = self.epsil.get()
        p["mu"] = self.mu_host
        p["xyz_gamma"] = self.xyz_gamma.get()
        p["max_connectedness"] = em["max_connectedness"]
        p["max_components"] = em["max_components"]
        if em["cost_em"] is not None:
            p["cost_em"] = em["cost_em"]
        p["iter_em"] = em["iter_em"]
        p["iter_inter"] = 1.0
        p["candidate_layout"] = self.layout_host
        p["spatial_xyz_vmf_P"] = self.sxv.get()
        if self.variant.use_xyz_prior:
            p["s_muc"] = self.s_muc.get()
        if self.variant.use_connect_prior:
            p["spatial_connect_vmf_P"] = self.scv.get()
            p["u"] = self.u.get()
        if emit_dense_priors:
            p["spatial_xyz_vmf"] = self.dense_P(self.sxv)
            p["spatial_connect_vmf"] = (self.dense_P(self.scv) if self.variant.use_connect_prior
                                        else np.zeros((self.N, L), np.float32))
        return p
