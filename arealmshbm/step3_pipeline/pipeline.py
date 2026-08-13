"""pipeline.py

End-to-end Python step-3 super-call. One entry point that consumes a
CBIG-style ``project_dir``, runs the full single-subject Mode-A
parcellation EM, and writes the ``Ind_parcellation_*.mat`` artifact.

Composes (in order):

    1. ``data_io.load_group_prior``           — Params_Final.mat (mu, θ, ε, σ).
    2. ``data_io.load_spatial_mask``          — spatial_mask_<mesh>.mat.
    3. ``data_io.load_avg_mesh``              — fsaverage* inflated + sphere.
    4. ``data_io.fetch_data``                 — BOLD profile + diffusion gradient.
    5. ``initialize_concentration``           — vMF κ init.
    6. ``initialize_params``                  — Params dict (s_psi/s_t_nu/…).
    7. ``pipeline_setup.build_pipeline_setup`` — boundary_mask + neighborhood + …
    8. ``vmf_clustering.VmfClusteringSession`` — the EM body super-call.
    9. ``intra_em.intra_subject_var`` + ``intra_em_cost`` — outer-loop math.
   10. ``data_io.save_parcellation``          — argmax + savemat.

Public API:
    Step3Inputs   — bundled outputs of the load + setup stages.
    Step3Result   — bundled outputs of one ``run()``.
    Step3Pipeline — single-subject load → setup → EM → save lifecycle.

Single-subject only. For multi-subject runs use the unified pipeline
driver at :mod:`arealmshbm.pipeline` (``mode='modeA_batch'``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from arealmshbm.data_io import (
    fetch_data,
    load_group_prior,
    load_spatial_mask,
    load_avg_mesh,
    save_parcellation,
    derive_labels,
)
from arealmshbm.initialize_concentration import initialize_concentration
from arealmshbm.initialize_params import initialize_params
from arealmshbm.pipeline_setup import build_pipeline_setup
from arealmshbm.intra_em import intra_subject_var, intra_em_cost
from arealmshbm.em_stop_criterion import matlab_ratio_converged
from arealmshbm.vmf_clustering import VmfClusteringSession
from arealmshbm.postprocessing import remove_isolated_surface_components

from .config import Step3Config


# ─────────────────────────────────────────────────────────────────────
# Pre-EM artifacts (the "setup" stage's outputs).
# Bundled as a dataclass so a profiling caller can inspect them
# independently of the Pipeline's run() side-effects.
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Step3Inputs:
    """Loaded + derived inputs that feed the EM."""
    group_prior: Dict[str, np.ndarray]
    data: Dict[str, Any]                   # fetch_data output: series (N, T, ⌈D/8⌉) uint8 bit-packed, D_unpacked (int), gradient_mat (N, Dg) when gMSHBM
    boundary_mask: np.ndarray              # (N, L)
    neighborhood: np.ndarray               # (M1, N_active) int64
    row_idx: np.ndarray                    # (P,) int64 0-indexed
    col_idx: np.ndarray
    lh_inflated: Dict[str, np.ndarray]
    rh_inflated: Dict[str, np.ndarray]
    lh_sphere: Dict[str, np.ndarray]
    rh_sphere: Dict[str, np.ndarray]
    sphere_xyz_bilateral: np.ndarray       # (N, 3)
    ini_val: float
    Params: Dict[str, Any]
    setting_params: Dict[str, Any]


@dataclass
class Step3Result:
    """Outputs of one Pipeline.run()."""
    Params: Dict[str, Any]
    lh_labels: np.ndarray
    rh_labels: np.ndarray
    out_path: Optional[Path]               # None if save() was not called
    record: list                           # per-iter update_cost values
    iter_intra_em: int
    timings: Dict[str, Any] = field(default_factory=dict)


class Step3Pipeline:
    """Single-subject step-3 EM pipeline. Stateful — cache once, run once.

    Lifecycle:
      * ``__init__(cfg)``                       — store config; no IO.
      * ``load_inputs() -> Step3Inputs``         — read everything the EM needs.
      * ``build_session(inputs)``                — construct VmfClusteringSession.
      * ``run(inputs=None) -> Step3Result``      — execute intra_em loop (+ optional load + build).
      * ``save(result, out_path=None) -> Path``  — argmax + savemat.
      * ``close()``                              — release Session + Inputs refs and the
        cupy device-memory pool. **Required** between sequential subjects on
        the GPU backend; otherwise the cupy memory pool keeps holding the
        ~4 GB peak across runs (cupy's pool keeps free blocks for
        reuse-by-default, so ``nvidia-smi`` shows the peak even after the
        Pipeline goes out of scope).

    Pipeline is also a context manager: ``with Step3Pipeline(cfg) as pipe: ...``
    calls ``close()`` on exit (success or failure).

    For one-shot use: ``Step3Pipeline(cfg).run_and_save()``; remember to
    ``close()`` afterward in batch / interactive sessions.

    Not thread-safe (the underlying Sessions own scratch buffers).
    """

    def __init__(self, cfg: Step3Config,
                 *,
                 precomputed_gradient_mat: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        cfg : Step3Config
        precomputed_gradient_mat : (N, n_grad_components) fp32, optional
            In-memory gradient array. When set, fetch_data skips the
            per-hemi gradient .npy/.mat disk reads and uses this array
            directly. The unified pipeline driver passes step0's
            ``Step0Result.lh_emb_up``/``rh_emb_up`` (concatenated) here
            so step3 reads gradients from memory instead of re-reading
            the .npy step0 just wrote. Validated by fetch_data.
        """
        if not isinstance(cfg, Step3Config):
            raise TypeError(
                f"Step3Pipeline: cfg must be a Step3Config (got {type(cfg).__name__})"
            )
        self.cfg = cfg
        self._precomputed_gradient_mat: Optional[np.ndarray] = (
            precomputed_gradient_mat
        )

        # Lazy state.
        self._inputs: Optional[Step3Inputs] = None
        self._session: Optional[VmfClusteringSession] = None

        # Per-stage timings, populated as run() proceeds.
        self.timings: Dict[str, Any] = {}

    # ── Lifecycle: explicit cleanup ──
    def close(self) -> None:
        """Release Session + Inputs and free the cupy memory pool.

        Idempotent. The cupy free-blocks call is a no-op when ``cupy``
        isn't importable (CPU backend) or when the default pool is
        already empty.

        After ``close()`` the Pipeline is unusable — calling ``run()``
        again would re-allocate everything from scratch. To run a new
        subject, build a new Pipeline instance.
        """
        # Drop large Python references first so the underlying cupy /
        # numpy buffers become unreferenced before we ask the pool to
        # free.
        self._session = None
        self._inputs = None
        # Free cupy memory pools if cupy is loaded. Importing inside
        # the function (not at module top) keeps the CPU-only path
        # cupy-free.
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except ImportError:
            return
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def __enter__(self) -> "Step3Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Pre-loaded inputs injection (multi-subject runner path) ──
    def set_inputs(self, inputs: Step3Inputs) -> None:
        """Inject pre-loaded :class:`Step3Inputs` so :meth:`run` skips
        :meth:`load_inputs`.

        For a multi-subject runner that loads each subject's
        :class:`Step3Inputs` on a worker thread (overlapping IO with the
        previous subject's GPU EM): instead of every consumer poking
        ``self._inputs`` directly — which couples them to a private
        attribute that a future refactor might rename — they use this
        public method.

        Idempotent across distinct ``Step3Inputs`` is NOT supported on
        the same Pipeline instance: after :meth:`set_inputs`, the
        Pipeline is committed to that one subject's data; build a new
        Pipeline for each subject (the runner does this).

        ``pipe.timings`` gets a zero-valued ``load_total`` entry so
        downstream introspection of ``pipe.timings`` sees a complete
        per-stage map (the actual load happened in the runner's
        worker thread, not here).
        """
        if self._inputs is not None:
            raise RuntimeError(
                "set_inputs: this Pipeline already has inputs (either "
                "load_inputs() ran or set_inputs() was already called); "
                "build a new Step3Pipeline instead of reusing one."
            )
        if not isinstance(inputs, Step3Inputs):
            raise TypeError(
                f"set_inputs: expected Step3Inputs, got {type(inputs).__name__}"
            )
        self._inputs = inputs
        self.timings.setdefault("load_total", 0.0)

    # ── Stage 1: load ──
    def load_inputs(self) -> Step3Inputs:
        """Read every disk artifact the EM needs and build derived setup.

        Idempotent — caches on the Pipeline. Subsequent calls return the
        cached :class:`Step3Inputs`; no re-reads.
        """
        if self._inputs is not None:
            return self._inputs

        cfg = self.cfg
        t_load_total = time.perf_counter()

        t0 = time.perf_counter()
        group_prior = load_group_prior(cfg.group_prior_path)
        self.timings["load_group_prior"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        lh_b, rh_b = load_spatial_mask(cfg.spatial_mask_path)
        self.timings["load_spatial_mask"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        lh_inflated = load_avg_mesh("lh", cfg.mesh, "inflated")
        rh_inflated = load_avg_mesh("rh", cfg.mesh, "inflated")
        lh_sphere = load_avg_mesh("lh", cfg.mesh, "sphere")
        rh_sphere = load_avg_mesh("rh", cfg.mesh, "sphere")
        self.timings["load_avg_mesh"] = time.perf_counter() - t0

        # bilateral sphere coords (N, 3) for spatial_xyz_prior.
        sphere_xyz_bilateral = np.concatenate(
            [lh_sphere["vertices"].T, rh_sphere["vertices"].T], axis=0,
        )

        # ``with_gradient`` is gMSHBM-only — cMSHBM/dMSHBM never read
        # the diffusion-embedding ``.mat`` files (no gradient prior
        # term). Skipping the IO saves ~2 mat reads + the gradient
        # H2D upload on GPU.
        #
        # fetch_data always returns bit-packed BOLD; each consumer
        # (CPU / gpu_elambda / gpu_full Session) runs its own
        # unpack+normalize step on this buffer.
        t0 = time.perf_counter()
        data = fetch_data(
            project_dir=cfg.project_dir,
            num_session=cfg.num_session,
            subid=cfg.subid,
            mesh=cfg.mesh,
            lh_mesh=lh_inflated,
            rh_mesh=rh_inflated,
            n_grad_components=cfg.n_grad_components,
            with_gradient=cfg.variant.use_connect_prior,
            precomputed_gradient_mat=self._precomputed_gradient_mat,
        )
        self.timings["fetch_data"] = time.perf_counter() - t0

        # Setup stage — initialize_concentration, initialize_params,
        # build_pipeline_setup. Each is small but we still time them so
        # the profile breakdown is complete.
        # ``series`` is bit-packed (N, T, ⌈D/8⌉) uint8; the original
        # cell count along axis -1 lives in ``D_unpacked``.
        series = data["series"]
        N = int(series.shape[0])
        T = int(series.shape[1])
        D = int(data["D_unpacked"])
        if T != int(cfg.num_session):
            raise ValueError(
                f"fetch_data returned {T} sessions; cfg.num_session={cfg.num_session}"
            )
        L = int(cfg.num_clusters)
        dim = D - 1

        # MATLAB driver: ``ini_val=650`` is hardcoded for D-1=1482 (a
        # special-case for fs_LR_32k); the ``CBIG_ArealMSHBM_initialize_
        # concentration`` call runs otherwise. For fsaverage* the
        # special-case is unused.
        t0 = time.perf_counter()
        if dim == 1482:
            ini_val = 650.0
        else:
            ini_val = initialize_concentration(dim)
        self.timings["initialize_concentration"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        Params = initialize_params(
            group_prior=group_prior,
            ini_val=ini_val,
            num_session=T,
            num_clusters=L,
            num_verts=N,
        )
        self.timings["initialize_params"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        setup = build_pipeline_setup(
            theta=Params["theta"],
            s_lambda_init=Params["s_lambda"],
            lh_boundary=lh_b,
            rh_boundary=rh_b,
            lh_vertex_nbors=lh_inflated["vertexNbors"],
            rh_vertex_nbors=rh_inflated["vertexNbors"],
        )
        self.timings["build_pipeline_setup"] = time.perf_counter() - t0

        boundary_mask = setup["boundary_mask"]

        # MATLAB beta vector — repmat(beta * 1000, 1, L).
        beta_vec = np.full(L, cfg.beta_internal, dtype=np.float64)

        setting_params: Dict[str, Any] = {
            "mesh": cfg.mesh,
            "num_session": T,
            "num_clusters": L,
            "subid": int(cfg.subid),
            "w": float(cfg.w),
            "c": float(cfg.c),
            "beta": beta_vec,
            "epsilon": float(cfg.epsilon),
            "connect_th": float(cfg.connect_th),
            "dim": dim,
            "num_verts": N,
            "neighborhood": setup["neighborhood"],
            "row_idx": setup["row_idx"],
            "col_idx": setup["col_idx"],
        }

        self.timings["load_total"] = time.perf_counter() - t_load_total

        self._inputs = Step3Inputs(
            group_prior=group_prior,
            data=data,
            boundary_mask=boundary_mask,
            neighborhood=setup["neighborhood"],
            row_idx=setup["row_idx"],
            col_idx=setup["col_idx"],
            lh_inflated=lh_inflated,
            rh_inflated=rh_inflated,
            lh_sphere=lh_sphere,
            rh_sphere=rh_sphere,
            sphere_xyz_bilateral=sphere_xyz_bilateral,
            ini_val=ini_val,
            Params=Params,
            setting_params=setting_params,
        )
        return self._inputs

    # ── Stage 2: build EM Session ──
    def build_session(self, inputs: Optional[Step3Inputs] = None
                      ) -> VmfClusteringSession:
        """Construct the inner :class:`VmfClusteringSession`.

        Idempotent — caches on the Pipeline.
        """
        if self._session is not None:
            return self._session
        if inputs is None:
            inputs = self.load_inputs()

        cfg = self.cfg
        sp = inputs.setting_params

        t0 = time.perf_counter()
        # gMSHBM uses the gradient embedding; cMSHBM/dMSHBM omit it
        # (data["gradient_mat"] absent — see fetch_data ``with_gradient``).
        # cMSHBM/gMSHBM use sphere coords for check_connectedness;
        # dMSHBM doesn't run check_connectedness so meshes can be None.
        grad_data = inputs.data.get("gradient_mat")
        if cfg.variant.use_check_connectedness:
            lh_sphere_arg = inputs.lh_sphere
            rh_sphere_arg = inputs.rh_sphere
            sphere_xyz_arg = inputs.sphere_xyz_bilateral
        else:
            lh_sphere_arg = None
            rh_sphere_arg = None
            sphere_xyz_arg = None
        # BOLD is always bit-packed (N, T, ⌈D/8⌉) uint8 + D_unpacked;
        # each backend's Session runs its own unpack+normalize.
        sess = VmfClusteringSession(
            data_series_NTD=inputs.data["series"],
            grad_data=grad_data,
            sphere_xyz=sphere_xyz_arg,
            lh_sphere_mesh=lh_sphere_arg,
            rh_sphere_mesh=rh_sphere_arg,
            theta=inputs.Params["theta"],
            boundary_mask=inputs.boundary_mask,
            neighborhood=sp["neighborhood"],
            row_idx=sp["row_idx"],
            col_idx=sp["col_idx"],
            dim=int(sp["dim"]),
            num_clusters=int(sp["num_clusters"]),
            num_session=int(sp["num_session"]),
            w=float(sp["w"]),
            c=float(sp["c"]),
            beta=sp["beta"],
            connect_th=float(sp["connect_th"]),
            epsilon=float(sp["epsilon"]),
            max_iter_em=int(cfg.max_iter_em),
            max_iter_lambda=int(cfg.max_iter_lambda),
            max_iter_m=int(cfg.max_iter_m),
            max_iter_comp=int(cfg.max_iter_comp),
            cMSHBM_isolated_component_min_size=int(
                cfg.cMSHBM_isolated_component_min_size
            ),
            backend=cfg.backend,
            D_unpacked=int(inputs.data["D_unpacked"]),
            variant_spec=cfg.variant,
        )
        self.timings["session_init"] = time.perf_counter() - t0
        self._session = sess
        return sess

    # ── Stage 3: run intra_em outer loop ──
    def run(self, time_stages: bool = True) -> Step3Result:
        """Execute the full pipeline: load + build + intra_em loop.

        Parameters
        ----------
        time_stages : if True (default), record per-stage wall times for
            the inner :class:`VmfClusteringSession.run` calls. The
            aggregate is summed across intra_em iterations and stored on
            ``self.timings`` under ``per_stage_em``.

        Returns
        -------
        Step3Result : Params dict at convergence + derived labels.
        """
        cfg = self.cfg
        inputs = self.load_inputs()
        sess = self.build_session(inputs)

        Params = inputs.Params
        sp = inputs.setting_params
        ini_val = inputs.ini_val
        L = int(sp["num_clusters"])
        T = int(sp["num_session"])
        dim = int(sp["dim"])

        # Per-iter EM timings (aggregated across intra_em rounds).
        per_stage_em: Dict[str, float] = {
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
        per_iter_intra: list[float] = []     # outer-loop wall per round
        record: list = []
        cost = 0.0
        stop_intra = False
        iter_intra = 0

        t_em_total = time.perf_counter()

        while not stop_intra:
            iter_intra += 1
            t_round = time.perf_counter()

            # MATLAB driver: at the top of each intra_em iteration, reset
            # kappa + s_t_nu to their post-init values (the rest of
            # ``Params`` carries over from the previous iter).
            Params["kappa"] = ini_val * np.ones(L, dtype=np.float64)
            Params["s_t_nu"] = np.broadcast_to(
                Params["mu"][..., None],
                (Params["mu"].shape[0], L, T),
            ).copy()

            # MATLAB driver: ``vmf_clustering_subject_session`` (the EM body).
            Params = sess.run(Params, time_stages=time_stages)
            if time_stages and sess.last_timings:
                for k, v in sess.last_timings.items():
                    if k in per_stage_em and isinstance(v, (int, float)):
                        per_stage_em[k] += v

            # MATLAB driver: ``intra_subject_var``.
            Params["s_psi"] = intra_subject_var(
                s_t_nu=Params["s_t_nu"],
                sigma=Params["sigma"],
                epsil=Params["epsil"],
                mu=Params["mu"],
            )

            # MATLAB driver: ``intra_em_cost`` (the inline cost-update
            # block in the original; lifted to a helper here).
            update_cost = intra_em_cost(
                s_psi=Params["s_psi"],
                s_t_nu=Params["s_t_nu"],
                mu=Params["mu"],
                sigma=Params["sigma"],
                epsil=Params["epsil"],
                cost_em=Params.get("cost_em", 0.0),
                dim=dim,
            )
            record.append(update_cost)

            # Convergence test — same MATLAB ratio rule as the inner EM
            # loop (matlab_ratio_converged is the shared helper); single
            # site for the NaN/Inf branch semantics.
            converged = matlab_ratio_converged(update_cost, cost)
            if converged:
                stop_intra = True
                Params["cost_intra"] = update_cost
            if iter_intra >= int(cfg.max_iter_intra_em):
                stop_intra = True
                Params["cost_intra"] = update_cost
            cost = update_cost

            per_iter_intra.append(time.perf_counter() - t_round)

        Params["Record"] = np.asarray(record, dtype=np.float64)

        self.timings["em_total"] = time.perf_counter() - t_em_total
        self.timings["per_iter_intra_em"] = per_iter_intra
        self.timings["per_stage_em"] = per_stage_em

        # Argmax labels (no save yet — caller decides via save()).
        n_lh = int(inputs.lh_inflated["MARS_label"].shape[0])
        lh_labels, rh_labels = derive_labels(Params["s_lambda"], n_lh=n_lh)

        return Step3Result(
            Params=Params,
            lh_labels=lh_labels,
            rh_labels=rh_labels,
            out_path=None,
            record=record,
            iter_intra_em=iter_intra,
            timings=dict(self.timings),
        )

    # ── Stage 4: save artifact ──
    def save(self, result: Step3Result, out_path: Optional[Path] = None,
             ) -> Path:
        """Write ``Ind_parcellation_*.mat`` to disk.

        cMSHBM: applies :func:`remove_isolated_surface_components` (size 5)
        per hemisphere on the argmax labels BEFORE save (mirrors MATLAB
        cMSHBM lines 319-320). gMSHBM/dMSHBM skip the post-process.

        dMSHBM: filename omits the ``_beta<B>`` suffix.

        Parameters
        ----------
        result : :class:`Step3Result` from :meth:`run`.
        out_path : full target path. None -> auto-build from
                   ``cfg.out_dir`` + the standard filename. Both branches
                   route through :func:`save_parcellation` so the on-
                   disk schema stays bit-identical regardless of how the
                   caller picked the destination.
        """
        cfg = self.cfg
        beta_for_filename = (
            cfg.beta_scalar if cfg.variant.filename_includes_beta else None
        )

        # cMSHBM: argmax → RemoveIsolated(5) on lh / rh labels before
        # save. gMSHBM / dMSHBM emit raw argmax (the existing path).
        if cfg.variant.final_remove_isolated:
            inputs = self.load_inputs()  # cached; just retrieves
            lh_nbors = inputs.lh_inflated["vertexNbors"]
            rh_nbors = inputs.rh_inflated["vertexNbors"]
            thr = int(cfg.cMSHBM_isolated_component_min_size)
            lh_clean = remove_isolated_surface_components(
                result.lh_labels, lh_nbors, abs_threshold=thr,
            )
            rh_clean = remove_isolated_surface_components(
                result.rh_labels, rh_nbors, abs_threshold=thr,
            )
            result.lh_labels = lh_clean
            result.rh_labels = rh_clean
            out_path = save_parcellation(
                s_lambda=result.Params["s_lambda"],
                out_dir=cfg.out_dir,
                subid=cfg.out_subid or str(int(cfg.subid)),
                w=cfg.w,
                c=cfg.c,
                beta=beta_for_filename,
                out_path=out_path,
                lh_labels_override=lh_clean,
                rh_labels_override=rh_clean,
            )
        else:
            out_path = save_parcellation(
                s_lambda=result.Params["s_lambda"],
                out_dir=cfg.out_dir,
                subid=cfg.out_subid or str(int(cfg.subid)),
                w=cfg.w,
                c=cfg.c,
                beta=beta_for_filename,
                out_path=out_path,
            )
        result.out_path = Path(out_path)
        return Path(out_path)

    # ── Convenience ──
    def run_and_save(self, time_stages: bool = True,
                     out_path: Optional[Path] = None) -> Step3Result:
        """``run() + save()`` in one call. Returns the populated Result."""
        result = self.run(time_stages=time_stages)
        self.save(result, out_path=out_path)
        return result
