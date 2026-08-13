"""vmf_clustering.py

The full step-3 EM body, single-subject (Mode A). One ``run()`` call
corresponds to one ``intra_em`` outer-loop iteration.

Public API:
    VmfClusteringSession — caches every sub-Session and per-call scratch.
        ``backend='cpu' | 'gpu_elambda' | 'gpu_full'``.

Caller-facing array shapes (all single-subject; no ``S`` axis):
    data_series_NTD       : (N, T, ⌈D/8⌉) uint8 — bit-packed caller BOLD;
                            ``__init__`` widens to a single (N, T, D) fp32
                            buffer via
                            :func:`arealmshbm.data_io.bitpacked_norm.unpack_normalize_packed_NTD_host`
                            and shares that buffer by reference with the
                            M-step / ELambda / EMStop sub-Sessions. M-step's
                            per-t sgemm uses the strided view ``[:, t, :]``
                            (lda = T·D).
    D_unpacked            : int — original cell count along axis -1
                            (required; pairs with the packed buffer).
    data.gradient_mat     : (N, D_grad) fp32
    Params.s_lambda       : (N, L)    fp32
    Params.s_t_nu         : (D, L, T) fp32
    Params.s_psi          : (D, L)    fp32
    Params.theta          : (N, L)    fp32
    Params.mu             : (D, L)    fp32 (carried through, not read inside)
    Params.kappa          : (L,)      fp64
    Params.sigma          : (L,)      fp32
    Params.epsil          : (L,)      fp32
    Params.xyz_gamma      : (L,)      fp64 — accumulator across calls
    Params.spatial_xyz_vmf : (N, L)   fp32 — zeros at first iter
    boundary_mask         : (N, L)    fp64/fp32

Mesh inputs (consumed once per Session):
    lh_sphere_mesh, rh_sphere_mesh : dicts with keys ``vertices``
        (3, n_lh) and ``vertexNbors`` (max_neigh, n_lh).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from arealmshbm.m_step import MStepSession
from arealmshbm.spatial_priors import ConnectSession, XyzSession
from arealmshbm.V_lambda import Session as VLambdaSession
from arealmshbm.check_connectedness.component_distance import (
    component_distance,
    compute_components_general,
)
from arealmshbm.em_stop_criterion import EMStopSession
from arealmshbm.postprocessing import remove_isolated_surface_components
from arealmshbm.step3_pipeline.variant import VariantSpec

from arealmshbm.data_io.bitpacked_norm import unpack_normalize_packed_NTD_host

from ._session_common import (
    validate_packed_bold_shape,
    validate_variant_requirements,
)
from .e_step_lambda import ELambdaSession


# Backend selection (constructor parameter — see VmfClusteringSession docstring).
# Validated values for ``backend=``:
_VALID_BACKENDS = ("cpu", "gpu_elambda", "gpu_full")


def _validate_backend(backend: str) -> str:
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_VALID_BACKENDS}"
        )
    return backend


# ─────────────────────────────────────────────────────────────────────────
# A lightweight Params container — mirrors the MATLAB struct's fields we
# actually use inside vmf_clustering_subject_session.
# Caller hands in a dict-like (e.g. h5py-backed dict) at the entry boundary;
# the Session updates the relevant fields and returns an updated dict.
# ─────────────────────────────────────────────────────────────────────────
def _get(params: Dict[str, Any], key: str) -> Any:
    """Read a Params field, raising a clear error if missing."""
    if key not in params:
        raise KeyError(f"Params is missing required field '{key}'")
    return params[key]


# _stage_{1,2,3}d moved to ._session_common (cross-backend shared
# helpers; the GPU module also uses them). Re-exported here so any
# external script that imported them from this module before
# 2026-06-XX continues to work.
from ._session_common import _stage_1d, _stage_2d, _stage_3d  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────
# check_connectedness (Python equivalent of step3 lines 368-407)
#
# This is a thin glue function that mirrors what the MATLAB local sub-
# function does: take Params.s_lambda, derive labels, call the
# check_connectedness library to get parcel_components + eucli_dist,
# then update xyz_gamma + max_connectedness + max_components.
# ─────────────────────────────────────────────────────────────────────────
def _check_connectedness_step(
    s_lambda: np.ndarray,           # (N, L) fp32
    xyz_gamma: np.ndarray,          # (L,) or (1, L) fp64
    lh_mesh: Dict[str, np.ndarray],
    rh_mesh: Dict[str, np.ndarray],
    num_clusters: int,
    connect_th: float,
    components_threshold: int = 3,
    pre_remove_isolated: bool = False,
    remove_isolated_threshold: int = 5,
) -> Tuple[np.ndarray, float, float]:
    """Mirror of MATLAB ``check_connectedness`` for both gMSHBM and cMSHBM.

    Inputs:
        s_lambda    : (N, L) fp32 — current soft posterior.
        xyz_gamma   : (L,)    fp64 — current accumulator (will be updated).
        lh_mesh, rh_mesh : dicts with 'vertices' (3, n_lh) and
                           'vertexNbors' (max_neigh, n_lh).
        num_clusters : int — L.
        connect_th   : float — euclidean-distance threshold for the
                       "distributed parcel" decision (gMSHBM=15, cMSHBM=0).
        components_threshold : parcel_components > this counts as
                       distributed. gMSHBM=3, cMSHBM=1.
        pre_remove_isolated : if True (cMSHBM), apply
                       :func:`remove_isolated_surface_components` per
                       hemisphere on the argmax labels BEFORE the
                       components / distance test. Mirrors MATLAB cMSHBM
                       lines 357-358.
        remove_isolated_threshold : passed through to RemoveIsolated.
                       Default 5 (the only value used by cMSHBM).
    Returns:
        (xyz_gamma_new, max_connectedness, max_components)
    """
    N, L = s_lambda.shape
    if L != num_clusters:
        raise ValueError(f"s_lambda has {L} cols, expected num_clusters={num_clusters}")
    n_per_hemi = N // 2

    # MATLAB:
    #   labels = zeros(1, N)
    #   labels(sum(s_lambda, 2) ~= 0) = argmax(s_lambda(active, :), [], 2)
    # i.e., 1-indexed parcel id at each active vertex; 0 at inactive (medial wall).
    s_lambda_2d = s_lambda
    row_active = (s_lambda_2d.sum(axis=1) != 0)
    labels = np.zeros(N, dtype=np.int64)
    if row_active.any():
        # argmax over axis 1, 1-indexed.
        labels[row_active] = s_lambda_2d[row_active, :].argmax(axis=1) + 1
    lh_labels = labels[:n_per_hemi]
    rh_labels = labels[n_per_hemi:]

    # cMSHBM: clean isolated <5-vert components per hemisphere BEFORE the
    # components / distance test (MATLAB cMSHBM lines 357-358).
    if pre_remove_isolated:
        lh_labels = remove_isolated_surface_components(
            lh_labels, lh_mesh["vertexNbors"], remove_isolated_threshold,
        )
        rh_labels = remove_isolated_surface_components(
            rh_labels, rh_mesh["vertexNbors"], remove_isolated_threshold,
        )

    # Run compute_components_general + component_distance.
    parcel_components, lh_ci, rh_ci = compute_components_general(
        lh_labels, rh_labels,
        lh_mesh["vertexNbors"], rh_mesh["vertexNbors"],
        num_clusters, return_ci=True,
    )
    eucli_dist = component_distance(
        lh_labels, rh_labels, lh_mesh, rh_mesh, num_clusters,
        parcel_components=parcel_components,
        lh_ci_full=lh_ci, rh_ci_full=rh_ci,
    )

    # Distributed parcels: eucli_dist > connect_th OR parcel_components > threshold.
    # NB: MATLAB ``find`` returns 1-indexed; here we keep 0-indexed since we
    # only use the indices to update xyz_gamma in-place.
    distrib_mask = (eucli_dist > connect_th) | (parcel_components > components_threshold)
    if distrib_mask.any():
        max_connectedness = float(eucli_dist[distrib_mask].max())
        max_components = float(parcel_components[distrib_mask & np.isfinite(parcel_components)].max())
        # Update xyz_gamma in the distributed-parcel slots.
        xyz_gamma_new = xyz_gamma.copy()
        xyz_gamma_new[distrib_mask] += 1000.0
    else:
        max_connectedness = 0.0
        # MATLAB clamps the all-OK ``max_components`` to the distributed-
        # threshold value (gMSHBM line 304: ``Params.max_components = 3``;
        # cMSHBM line 376: ``Params.max_components = 1``).
        max_components = float(components_threshold)
        xyz_gamma_new = xyz_gamma.copy()

    return xyz_gamma_new, max_connectedness, max_components


# ─────────────────────────────────────────────────────────────────────────
# Production hot path — Session
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class _Inputs:
    """Static inputs cached on the Session (don't change across run() calls
    within a single sub-task)."""
    boundary_mask: np.ndarray
    lh_sphere_mesh: Dict[str, np.ndarray]
    rh_sphere_mesh: Dict[str, np.ndarray]


class VmfClusteringSession:
    """Pre-allocated state for a vmf_clustering super-call.

    Construction caches:
      * Sub-sessions: MStepSession, ConnectSession, XyzSession,
        VLambdaSession, EMStopSession.
      * Mesh data for check_connectedness (LH + RH sphere mesh).
      * boundary_mask (NxL).

    Per call (``run``): consumes a Params dict with the entry-boundary
    fields, runs the EM loop, returns a fresh Params dict with the
    converged fields.

    Single-subject by design. Not thread-safe — ``run()`` mutates
    Session-owned scratch and ping-pong buffers; concurrent calls on the
    same Session would race. Build one Session per worker.

    Backend dispatch
    ----------------
    The ``backend=`` constructor parameter selects the compute path:

      * ``'cpu'`` (default) — pure CPU numba kernels.
      * ``'gpu_elambda'``   — only the E-step lambda loop body runs on GPU
        (CuPy); per-comp_iter ping-pongs (N, L) arrays. Useful baseline.
      * ``'gpu_full'``      — full-GPU super-call (M-step + ELambda +
        EMStop + spatial_priors all device-resident; only check_connectedness
        BFS stays CPU). ``__new__`` redirects construction to
        :class:`VmfClusteringSessionCUDA`, drop-in same external API.

    Promote backend to constructor param (rather than env var) so it's
    settable from a config file or GUI.
    """

    def __new__(cls, *args, backend: str = "cpu", **kwargs):
        _validate_backend(backend)
        # Full-GPU dispatch: redirect construction to the GPU class.
        # If __new__ returns an instance of a *different* class than ``cls``,
        # Python skips this class's ``__init__`` — the GPU class's own
        # ``__init__`` (already invoked inside the call below) is the only
        # initializer that runs. Same external API, drop-in replacement.
        if cls is VmfClusteringSession and backend == "gpu_full":
            from .vmf_clustering_gpu import VmfClusteringSessionCUDA
            # The GPU class doesn't accept ``backend`` itself.
            return VmfClusteringSessionCUDA(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self,
                 data_series_NTD: np.ndarray,              # (N, T, ⌈D/8⌉) uint8 packed
                 grad_data: Optional[np.ndarray],          # (N, D_grad) fp32 or None
                 sphere_xyz: Optional[np.ndarray],         # (3, N) or (N, 3) or None
                 lh_sphere_mesh: Optional[Dict[str, np.ndarray]],
                 rh_sphere_mesh: Optional[Dict[str, np.ndarray]],
                 theta: np.ndarray,                        # (N, L) fp32
                 boundary_mask: np.ndarray,                # (N, L) fp32/fp64
                 neighborhood: np.ndarray,                 # (M1, N) int — for V_lambda
                 row_idx: np.ndarray,                      # (P,) int — 0-indexed
                 col_idx: np.ndarray,                      # (P,) int — 0-indexed
                 dim: int,
                 num_clusters: int,
                 num_session: int,
                 w: float,
                 c: float,
                 beta: np.ndarray,                         # (L,) or (1, L) fp64
                 connect_th: float = 15.0,
                 epsilon: float = 1e-4,
                 max_iter_em: int = 100,
                 max_iter_lambda: int = 50,
                 max_iter_m: int = 50,
                 max_iter_comp: int = 15,
                 cMSHBM_isolated_component_min_size: int = 5,
                 *,
                 D_unpacked: int,
                 backend: str = "cpu",
                 variant_spec: Optional[VariantSpec] = None):
        # Default variant for back-compat: gMSHBM. Existing call sites
        # that don't pass ``variant_spec`` see the same behaviour as
        # before this refactor.
        if variant_spec is None:
            variant_spec = VariantSpec.from_pipeline_type("gMSHBM")
        self.variant = variant_spec
        # Strong invariants per variant — fail loudly at construction
        # if the caller fed mismatched inputs. Shared with the GPU
        # Session via ``_session_common`` so the two backends can't
        # drift on what input shapes they accept.
        validate_variant_requirements(
            variant_spec, grad_data, sphere_xyz,
            lh_sphere_mesh, rh_sphere_mesh,
        )
        # Single-subject pipeline — no S axis anywhere in the super-call.
        # ``backend`` is already validated by ``__new__`` (which is the
        # only construction path; instantiating bypassing it would be a
        # subclass concern). Just store the validated value.
        self.backend = backend

        self.dim = int(dim)
        self.num_clusters = int(num_clusters)
        self.num_session = int(num_session)
        self.w = float(w)
        self.c = float(c)
        self.beta_f32 = np.ascontiguousarray(np.asarray(beta).ravel(),
                                              dtype=np.float32)
        self.connect_th = float(connect_th)
        self.epsilon = float(epsilon)
        self.max_iter_em = int(max_iter_em)
        self.max_iter_lambda = int(max_iter_lambda)
        self.max_iter_m = int(max_iter_m)
        self.max_iter_comp = int(max_iter_comp)
        self.cMSHBM_isolated_component_min_size = int(
            cMSHBM_isolated_component_min_size
        )

        # ── BOLD: host-side bit-unpack + demean + L2-norm ─────────────
        # Packed (N, T, ⌈D/8⌉) uint8 in; the host numba kernel
        # (``arealmshbm.data_io.bitpacked_norm``) unpacks + normalizes
        # per session into a single (N, T, D) fp32 C-contig buffer
        # shared (by-reference) with all sub-Sessions. ELambda + EMStop
        # consume it via ``.reshape(N, T*D)`` (stride-only view);
        # M-step uses the strided view ``ds_NTD[:, t, :]`` (lda = T·D)
        # — MKL handles the lda (≤1% slower than a dedicated TND
        # C-contig copy), in trade for halving the BOLD footprint per
        # pipeline (4.6 GB → 2.3 GB).
        N, T, _D_bytes = validate_packed_bold_shape(
            data_series_NTD, num_session, D_unpacked,
        )
        D = int(D_unpacked)
        ds_NTD = unpack_normalize_packed_NTD_host(
            np.asarray(data_series_NTD), D,
        )

        self.N = int(N)
        self.D = int(D)
        self.T = int(T)
        self.L = int(num_clusters)

        # Inputs (cached static).
        self.boundary_mask_f32 = np.ascontiguousarray(boundary_mask, dtype=np.float32)
        self.theta_f32 = np.ascontiguousarray(theta, dtype=np.float32)
        # Active row mask (theta sum > 0). Same as the V_lambda candidate-row mask.
        # Demoted to a local — only the index list (``row_idx_active``) is read
        # later; the boolean mask itself is never used after construction.
        theta_row_active = (self.theta_f32.sum(axis=1) != 0)
        self.row_idx_active = np.where(theta_row_active)[0].astype(np.int64)
        # Convenience: number of active rows.
        self.M_active = int(self.row_idx_active.shape[0])

        # Mesh dicts for check_connectedness (sphere coords + vertex nbors).
        self.lh_sphere_mesh = lh_sphere_mesh
        self.rh_sphere_mesh = rh_sphere_mesh

        # Sub-sessions. All three (M-step, ELambda, EMStop) share the SAME
        # (N, T, D) caller-owned buffer — single BOLD copy per pipeline.
        self.m_step_session = MStepSession(
            data_series_NTD=ds_NTD,
            dim=self.dim,
            num_clusters=self.num_clusters,
            num_session=self.num_session,
            epsilon=self.epsilon,
            max_iter=self.max_iter_m,
        )

        # spatial_connect_prior session — gMSHBM only. Skipped for
        # cMSHBM/dMSHBM (they don't read the gradient embedding at all).
        if variant_spec.use_connect_prior:
            gd = np.ascontiguousarray(grad_data, dtype=np.float32)
            if gd.shape[0] != N:
                raise ValueError(
                    f"grad_data has {gd.shape[0]} rows, expected {N}"
                )
            self.connect_session: Optional[ConnectSession] = ConnectSession(
                grad_data=gd,
                num_verts=N,
                num_clusters=self.num_clusters,
            )
        else:
            self.connect_session = None

        # spatial_xyz_prior session — gMSHBM (gated by check_connectedness)
        # and cMSHBM (always-on). dMSHBM has no spatial term at all.
        if variant_spec.use_xyz_prior:
            self.xyz_session: Optional[XyzSession] = XyzSession(
                sphere_xyz=sphere_xyz,
                L=self.num_clusters,
            )
        else:
            self.xyz_session = None

        # V_lambda session — needs neighborhood + candidate idx.
        # Potts weights (V_same=0, V_diff=1) are baked into the kernel.
        # Pass ``row_idx_active`` so V_lambda's ``compute_full`` path can
        # consume the full (N, L) s_lambda directly without a Python-level
        # fancy-index gather (saves ~M_active*L*4-byte alloc per λ-iter).
        self.v_lambda_session = VLambdaSession(
            neighborhood=neighborhood,
            row_idx=row_idx,
            col_idx=col_idx,
            K=self.num_clusters,
            row_idx_active=self.row_idx_active,
        )

        # E-step lambda Session — picked by ``self.backend``.
        #   'gpu_elambda' → CUDA / CuPy implementation
        #   'cpu'         → CPU numba implementation
        # The GPU backend H2D's BOLD + theta + log(θ) + boundary_mask + the
        # V_lambda neighborhood/(row,col) once at construction; from then on
        # each ``run_lambda_loop`` call ping-pongs s_lambda / scv / sxv
        # between host and device. The CPU backend shares the (N, T, D)
        # BOLD reference with the super-call (no per-Session copy).
        if self.backend == "gpu_elambda":
            from .e_step_lambda_gpu import ELambdaSessionCUDA
            self.e_lambda_session = ELambdaSessionCUDA(
                data_series_NTD=ds_NTD,
                theta=self.theta_f32,
                boundary_mask=self.boundary_mask_f32,
                v_lambda_session=self.v_lambda_session,
                dim=self.dim,
                num_clusters=self.num_clusters,
                num_session=self.num_session,
                w=self.w,
                c=self.c,
                beta=self.beta_f32,
                epsilon=self.epsilon,
                max_iter=self.max_iter_lambda,
            )
        else:
            self.e_lambda_session = ELambdaSession(
                data_series_NTD=ds_NTD,
                theta=self.theta_f32,
                boundary_mask=self.boundary_mask_f32,
                v_lambda_session=self.v_lambda_session,
                dim=self.dim,
                num_clusters=self.num_clusters,
                num_session=self.num_session,
                w=self.w,
                c=self.c,
                beta=self.beta_f32,
                epsilon=self.epsilon,
                max_iter=self.max_iter_lambda,
            )

        # EM-stop session — shares the same (N, T, D) BOLD reference as
        # ELambda. The pre-S-drop refactor allocated 2.31 GB twice (once
        # per Session); now both reference the same buffer.
        self.em_stop_session = EMStopSession(
            data_series_NTD=ds_NTD,
            theta=self.theta_f32,
            dim=self.dim,
            num_session=self.num_session,
            num_clusters=self.num_clusters,
            w=self.w,
            c=self.c,
            beta=self.beta_f32,
        )

        # Keep a reference to the single (N, T, D) BOLD buffer. Same memory
        # is referenced by all three sub-Sessions (verifiable via
        # ``ctypes.data`` pointer match).
        self.data_series_NTD = ds_NTD

        # Variant-routing scratch: a single zeroed (N, L) buffer used as
        # the kernel input for any prior the variant doesn't compute
        # (cMSHBM gets zero scv into both E-step and EM-stop; dMSHBM gets
        # zero scv + zero sxv). Allocated once; never written. The
        # E-step kernel only READS it, so safe to share across all
        # absent-prior slots.
        self._zero_NL = np.zeros((self.N, self.L), dtype=np.float32)

        # Instrumentation: per-stage timings, populated in run() if requested.
        self.last_timings: Dict[str, float] = {}

    def run(self,
            params_in: Dict[str, Any],
            time_stages: bool = False,
            ) -> Dict[str, Any]:
        """Run one full vmf_clustering_subject_session.

        Parameters
        ----------
        params_in : dict — Params at function entry (post-kappa/s_t_nu reset).
            Must contain: ``s_lambda`` (N, L), ``s_t_nu`` (D, L, T),
            ``s_psi`` (D, L), ``sigma`` (L,), ``kappa`` (L,),
            ``epsil`` (L,), ``mu`` (D, L), ``theta`` (N, L) [optional —
            uses Session-cached if missing], ``xyz_gamma`` (L,),
            ``spatial_xyz_vmf`` (N, L) [pre-EM zeros are typical].
            All 1-D vectors are flat (L,); (1, L) is also accepted at
            the entry boundary and flattened.
        time_stages : if True, populate ``self.last_timings`` with per-stage
            wall times.

        Returns
        -------
        params_out : dict — Params at function exit. Same shapes as input:
            ``s_lambda`` (N, L), ``s_t_nu`` (D, L, T), ``s_psi`` (D, L),
            1-D vectors flat (L,) — kappa/sigma/epsil/xyz_gamma all
            ravelled. Also includes ``max_connectedness``,
            ``max_components``, ``s_muc``, ``u``, ``spatial_connect_vmf``,
            ``spatial_xyz_vmf``, ``cost_em``.
        """
        import time as _time

        # ── Stage Params into local Python state ──
        # Strict shapes — single-subject pipeline; the (..., 1) MATLAB
        # leftovers are no longer accepted.
        s_lambda = _stage_2d(_get(params_in, "s_lambda"), self.N, self.L, "s_lambda")
        s_t_nu   = _stage_3d(_get(params_in, "s_t_nu"),
                              self.D, self.L, self.T, "s_t_nu")
        s_psi    = _stage_2d(_get(params_in, "s_psi"), self.D, self.L, "s_psi")
        sigma    = _stage_1d(_get(params_in, "sigma"), self.L, "sigma", np.float32)
        kappa    = _stage_1d(_get(params_in, "kappa"), self.L, "kappa", np.float64)
        # epsil + mu are not directly read inside vmf_clustering; carried for output.
        epsil    = _stage_1d(_get(params_in, "epsil"), self.L, "epsil", np.float32)
        mu       = np.asarray(_get(params_in, "mu"))
        theta    = self.theta_f32   # session-cached
        xyz_gamma = _stage_1d(_get(params_in, "xyz_gamma"), self.L,
                               "xyz_gamma", np.float64)
        spatial_xyz_vmf = np.ascontiguousarray(_get(params_in, "spatial_xyz_vmf"),
                                                dtype=np.float32)

        # spatial_connect_vmf is set inside the loop (first call to
        # spatial_connect_prior); not read from params_in.
        spatial_connect_vmf = np.zeros((self.N, self.L), dtype=np.float32)
        u_centroid = None    # set by spatial_connect_prior
        s_muc = None         # set by spatial_xyz_prior

        # Track the EM stop block's last V_temp (needed to build cost_em).
        V_temp_last = np.zeros((self.N, self.L), dtype=np.float32)

        # Convergence state.
        cost = np.zeros((1, 1), dtype=np.float32)
        max_connectedness = 0.0
        max_components = 0.0
        cost_em_out = None

        # Per-stage timing accumulators (one dict per stage, summed across iters).
        timings = {
            "m_step": 0.0,
            "spatial_connect_prior": 0.0,
            "e_step_lambda_loop": 0.0,
            "check_connectedness": 0.0,
            "spatial_xyz_prior": 0.0,
            "em_stop_criterion": 0.0,
            "iter_count_em_total": 0,
            "iter_count_m_total": 0,
            "iter_count_lambda_total": 0,
            "iter_count_comp_total": 0,
            "iter_count_check_conn": 0,
            "iter_count_spatial_xyz": 0,
        }
        # Per-substep timing for the E-step lambda loop body. Populated only
        # when time_stages=True; the inner loop accumulates each substep's
        # wall time across all (comp_iter, lambda_iter) pairs.
        sub_timings_lambda: Optional[Dict[str, float]] = (
            {} if time_stages else None
        )

        # Variant-routing slot semantics:
        #   The E-step kernel does ``v += β[l]·scv[n,l] + sxv[n,l]``. We
        #   reuse the same kernel for all three variants by changing what
        #   buffers feed the (scv, sxv) slots — see :class:`VariantSpec`
        #   and ``docs/pipeline_variants.md`` §3.2 "Variant dispatch".
        #     gMSHBM: scv = spatial_connect_vmf (β-scaled gradient prior)
        #             sxv = spatial_xyz_vmf      (unscaled sphere prior)
        #     cMSHBM: scv = spatial_xyz_vmf      (β-scaled sphere prior;
        #                                         MATLAB ``+β·sxv``)
        #             sxv = zero                 (no second prior)
        #     dMSHBM: scv = sxv = zero           (no priors at all)
        # ``spatial_connect_vmf`` and ``spatial_xyz_vmf`` track the
        # underlying Session buffers; the routing decision happens at
        # E-step / EM-stop call sites.

        iter_em = 0

        while True:
            iter_em += 1
            timings["iter_count_em_total"] = iter_em

            # ── M-step ──
            t0 = _time.perf_counter()
            s_t_nu, kappa, iter_m = self.m_step_session.run(
                s_t_nu, s_lambda, s_psi, sigma, kappa,
            )
            timings["m_step"] += _time.perf_counter() - t0
            timings["iter_count_m_total"] += iter_m

            # ── ELambdaSession.prepare_em_iter ──
            # M-step refreshed s_t_nu and kappa. Recompute the em-iter-
            # invariant cache (acc = X @ s_t_nu, Cdln(kappa)) ONCE here
            # so the inner lambda loop reuses it across all comp_iter
            # passes within this em_iter.
            t0 = _time.perf_counter()
            self.e_lambda_session.prepare_em_iter(
                s_t_nu=s_t_nu,
                kappa=kappa,
                sub_timings=sub_timings_lambda,
            )
            timings["e_step_lambda_loop"] += _time.perf_counter() - t0

            # ── Pre-comp_iter prior refresh (variant-specific) ──
            # gMSHBM: spatial_connect_prior — once per EM iter, gradient
            #         centroid recomputed from current s_lambda.
            # cMSHBM: spatial_xyz_prior — once per EM iter, sphere vMF
            #         from current s_lambda + xyz_gamma. (MATLAB cMSHBM
            #         line 485: ``Params = spatial_xyz_prior(...)`` runs
            #         BEFORE the comp_iter loop on every EM iter — even
            #         iter_em==1, which gives ``Cdln(0,3)·1 + 0·cos =
            #         NaN→0`` after cleanup, i.e. effectively zero on
            #         the first ever call.)
            # dMSHBM: no spatial prior at all.
            if self.variant.use_connect_prior:
                t0 = _time.perf_counter()
                u_centroid_v, scv_v = self.connect_session.compute(s_lambda)
                u_centroid = u_centroid_v
                spatial_connect_vmf = scv_v
                timings["spatial_connect_prior"] += _time.perf_counter() - t0
            elif self.variant.use_xyz_prior:
                t0 = _time.perf_counter()
                s_muc_v, sxv_v = self.xyz_session.compute(s_lambda, xyz_gamma)
                s_muc = s_muc_v
                spatial_xyz_vmf = sxv_v
                timings["spatial_xyz_prior"] += _time.perf_counter() - t0
                timings["iter_count_spatial_xyz"] += 1

            # ── E-step + (optional) check_connectedness + xyz_prior loop ──
            if not self.variant.wrap_comp_iter:
                # dMSHBM: single λ-pass per EM iter, no comp_iter wrap, no
                # connectedness step. scv = sxv = zero.
                t0 = _time.perf_counter()
                s_lambda, V_temp_last, lambda_iter = self.e_lambda_session.run_lambda_loop(
                    s_lambda=s_lambda,
                    spatial_connect_vmf=self._zero_NL,
                    spatial_xyz_vmf=self._zero_NL,
                    sub_timings=sub_timings_lambda,
                )
                timings["e_step_lambda_loop"] += _time.perf_counter() - t0
                timings["iter_count_lambda_total"] += lambda_iter
                timings["iter_count_comp_total"] += 1
            else:
                # gMSHBM / cMSHBM: comp_iter wrap.
                stop_comp = False
                comp_iter = 0
                while not stop_comp:
                    comp_iter += 1
                    timings["iter_count_comp_total"] += 1

                    # ── E-step routing: variant-specific (scv, sxv) plug ──
                    if self.variant.use_connect_prior:
                        scv_estep = spatial_connect_vmf      # gMSHBM
                        sxv_estep = spatial_xyz_vmf
                    else:
                        scv_estep = spatial_xyz_vmf          # cMSHBM
                        sxv_estep = self._zero_NL

                    # E-step lambda loop body — ELambdaSession (numba kernels).
                    # Uses the prepare_em_iter cache; only the lambda loop runs.
                    # Returns:
                    #   * s_lambda — (N, L) VIEW of the Session's ping-pong
                    #     buffer. Passing it back unchanged on the next call
                    #     short-circuits the entry-side copyto via ctypes.data
                    #     match.
                    #   * V_temp_last — (N, L) VIEW of Session's _V_temp_full.
                    #     em_stop_criterion below READS it; nothing writes it
                    #     until the NEXT comp_iter's run_lambda_loop.
                    t0 = _time.perf_counter()
                    s_lambda, V_temp_last, lambda_iter = self.e_lambda_session.run_lambda_loop(
                        s_lambda=s_lambda,
                        spatial_connect_vmf=scv_estep,
                        spatial_xyz_vmf=sxv_estep,
                        sub_timings=sub_timings_lambda,
                    )
                    timings["e_step_lambda_loop"] += _time.perf_counter() - t0
                    timings["iter_count_lambda_total"] += lambda_iter

                    # ── check_connectedness + xyz update ──
                    run_conn = (
                        self.variant.use_check_connectedness
                        and iter_em >= self.variant.first_em_iter_with_conn_xyz
                    )
                    if run_conn:
                        t0 = _time.perf_counter()
                        xyz_gamma, max_connectedness, max_components = _check_connectedness_step(
                            s_lambda, xyz_gamma,
                            self.lh_sphere_mesh, self.rh_sphere_mesh,
                            self.num_clusters, self.connect_th,
                            components_threshold=self.variant.components_threshold,
                            pre_remove_isolated=self.variant.pre_predicate_remove_isolated,
                            remove_isolated_threshold=self.cMSHBM_isolated_component_min_size,
                        )
                        timings["check_connectedness"] += _time.perf_counter() - t0
                        timings["iter_count_check_conn"] += 1

                        if self.variant.use_xyz_prior:
                            t0 = _time.perf_counter()
                            s_muc_v, sxv_v = self.xyz_session.compute(s_lambda, xyz_gamma)
                            s_muc = s_muc_v
                            spatial_xyz_vmf = sxv_v
                            timings["spatial_xyz_prior"] += _time.perf_counter() - t0
                            timings["iter_count_spatial_xyz"] += 1
                    else:
                        # gMSHBM iter_em==1 (first_em_iter_with_conn_xyz==2):
                        # MATLAB sets ``stop_comp2 = 1`` to exit the wrap
                        # immediately after the first λ-pass.
                        stop_comp = True

                    # comp_iter convergence test: matches MATLAB's
                    # ``(max_connectedness <= connect_th) && (max_components
                    # <= components_threshold)``.
                    if (max_connectedness <= self.connect_th) and (
                        max_components <= float(self.variant.components_threshold)
                    ):
                        stop_comp = True
                    if comp_iter >= self.max_iter_comp:
                        stop_comp = True

            # ── EM stop criterion ──
            # gMSHBM: cost integration includes the ``+ Σ β·s_lambda·scv``
            # term, with scv = the gradient prior (ConnectSession output).
            # cMSHBM/dMSHBM: no spatial term in cost — scv slot fed zero.
            scv_em_stop = (
                spatial_connect_vmf
                if self.variant.em_stop_uses_beta_scv
                else self._zero_NL
            )
            t0 = _time.perf_counter()
            update_cost_scalar, stop_em, cost_em_view, _llp, scv_cleaned_view = self.em_stop_session.compute(
                s_t_nu=s_t_nu,
                kappa=kappa,
                s_lambda=s_lambda,
                spatial_connect_vmf=scv_em_stop,
                V_temp=V_temp_last,
                cost=cost,
                iter_em=iter_em,
            )
            # gMSHBM only: the em_stop session cleans NaN/Inf inside
            # spatial_connect_vmf; propagate that cleanup into our local
            # state so the next iter sees the cleaned version (mirrors
            # MATLAB's in-place mutation of Params.spatial_connect_vmf
            # inside the EM stop block).
            if self.variant.em_stop_uses_beta_scv:
                np.copyto(spatial_connect_vmf, scv_cleaned_view)
            update_cost = np.array([[update_cost_scalar]], dtype=np.float32)
            timings["em_stop_criterion"] += _time.perf_counter() - t0

            if stop_em:
                cost_em_out = cost_em_view.copy() if cost_em_view is not None else cost.copy()
                break
            if iter_em > self.max_iter_em:
                # Forced stop — cost_em is the *update_cost* in MATLAB's branch.
                cost_em_out = update_cost.copy()
                break

            cost = update_cost

        if time_stages:
            self.last_timings = dict(timings)
            if sub_timings_lambda is not None:
                self.last_timings["e_step_lambda_substeps"] = dict(sub_timings_lambda)

        # ── Build output Params dict ──
        # Several locals (s_lambda, spatial_xyz_vmf, spatial_connect_vmf,
        # u_centroid, s_muc) are VIEWS into Sub-session-internal buffers
        # to skip per-iter memcpy. At the boundary we copy them ONCE so
        # the caller holds independent buffers safe across future
        # ``run()`` calls (which would overwrite the underlying state).
        # These end-of-call copies cost ~94 MB × 4 ≈ 380 MB (one-shot,
        # not per-iter).
        params_out = dict(params_in)   # shallow copy — preserves any extras
        # s_lambda is a (N, L) view of e_lambda_session's ping-pong buffer;
        # the next ``run()`` call would overwrite it, so copy at the boundary.
        params_out["s_lambda"] = np.ascontiguousarray(s_lambda)
        params_out["s_t_nu"] = s_t_nu
        params_out["s_psi"] = s_psi
        # 1-D vectors emitted as (L,) flat — single-subject pipeline.
        params_out["sigma"] = sigma
        params_out["epsil"] = epsil
        params_out["mu"] = mu
        params_out["kappa"] = kappa
        params_out["theta"] = theta
        params_out["xyz_gamma"] = xyz_gamma
        # spatial_*_vmf are views of XyzSession / ConnectSession internal
        # buffers; copy out so the caller holds independent storage.
        params_out["spatial_xyz_vmf"] = np.ascontiguousarray(spatial_xyz_vmf).copy()
        params_out["spatial_connect_vmf"] = np.ascontiguousarray(spatial_connect_vmf).copy()
        if u_centroid is not None:
            params_out["u"] = np.ascontiguousarray(u_centroid).copy()
        if s_muc is not None:
            params_out["s_muc"] = np.ascontiguousarray(s_muc).copy()
        params_out["max_connectedness"] = max_connectedness
        params_out["max_components"] = max_components
        if cost_em_out is not None:
            params_out["cost_em"] = cost_em_out
        params_out["iter_em"] = iter_em
        return params_out
