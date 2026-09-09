"""pipeline.py

End-to-end Python step-2 super-call (Mode-B group prior estimation).
Dispatches on ``mode ∈ {gMSHBM, dMSHBM}``. cMSHBM is not wired
(no xyz-vMF path in the master kernel).

Stages:

    1. load_inputs        — mesh + training-set txt lists + group.mat
                            + spatial_mask (→ boundary_mask) + preload all
                            subjects' (N, D, T) profiles and (N, 100, T)
                            gradients (gMSHBM/dMSHBM).
    2. initialize_params  — sigma/s_psi/epsil/mu/kappa/s_t_nu from group.mat;
                            init s_lambda per subject via mtc-argmax projection;
                            init theta from s_lambda.
    3. outer EM loop      — interleave intra-EM with inter_subject_var. Inside
                            each intra-EM iter, run ``vmf_clustering_batch`` +
                            ``intra_subject_var_loop`` + ``intra_em_cost_step2``.
    4. save_params        — Params_Final.mat.

Backends (``Step2Config.backend``):

    'cpu' — numba master + host outer-EM leaves. The numerical
            reference.
    'gpu' — the P-layout / bit-packed CuPy backend. Its stages are
            ``load_inputs_sparse`` → ``initialize_params_sparse`` →
            ``_run_em_sparse``, selected as a set in :meth:`run`
            because stage 1 returns a different dataclass
            (``Step2SparseInputs``). Contract:
            ``docs/step2_sparse_design.md``.

Public API:
    Step2Inputs   — bundled outputs of load_inputs (CPU backend).
    Step2Result   — bundled outputs of one run().
    Step2Pipeline — lifecycle.

For Mode-B production runs use the unified pipeline driver at
:mod:`arealmshbm.pipeline` (``mode='modeB_train_prior'``) — it
sequences step0 → step1 → step2 → step3 against a project manifest.
``Step2Pipeline`` here is the single-stage entry used by that driver
and by the step-2 regression scripts.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from arealmshbm.pipeline._progress import COHORT_SUB_ID, ProgressEmitter

from arealmshbm.data_io.cohort import read_cohort, resolve_path
from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
from arealmshbm.data_io.load_spatial_mask import load_spatial_mask
from arealmshbm.initialize_concentration import initialize_concentration
from arealmshbm.step2_io import (
    SubjectGradientLoader,
    SubjectProfileLoader,
    load_group_mtc,
)
from arealmshbm.step2_init import (
    build_step2_boundary_mask,
    compose_init_state,
)
# ``step2_em_outer`` and ``step2_em_iter_master`` are imported LAZILY,
# inside the CPU path's ``run_em`` — importing ``step2_em_outer``
# eagerly costs ~0.2 s of numba work at import time
# (``em_stop_criterion._cdln`` declares explicit ``float64(...)``
# signatures, so its @njit kernels compile/cache-load at import rather
# than at first call), and the ``gpu`` backend never calls a single one
# of them — L16/L17/L18 and both resets run on-device there.
# Note: ``reset_s_t_nu_from_mtc_STLD`` is not called directly here at all —
# the CPU Session wraps it as ``reset_s_t_nu_from_mtc()``.
from .config import Step2Config
from .vmf_clustering_batch import em_body_sparse, vmf_clustering_batch

if TYPE_CHECKING:  # pragma: no cover — typing only, no runtime import
    from arealmshbm.step2_io.sparse_inputs import Step2SparseInputs


# ─────────────────────────────────────────────────────────────────────
# Pre-EM artifacts
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Step2Inputs:
    """Pre-EM-loaded state.

    Holds per-subject **loaders** (not eager buffers) that decode one
    subject's BOLD / gradient from disk on demand when the master
    kernel visits subject ``s`` in Phase A.3 / Phase D (BOLD) or
    Phase C (gradient). Peak per-iter footprint is one subject's
    slab (~770 MB BOLD + ~200 MB grad at fsa6/T=6) independent of S.
    """
    # Mesh
    lh_mars: np.ndarray            # (N_lh,) bool — 1 for medial wall
    rh_mars: np.ndarray            # (N_rh,) bool
    n_lh: int
    n_rh: int

    # Per-subject loaders.
    bold_loader: SubjectProfileLoader
    grad_loader: Optional[SubjectGradientLoader]  # gMSHBM/dMSHBM only

    # Cached dims peeked from sub 1 at load_inputs time. Used by
    # initialize_params and Session ctor to size scratch buffers
    # before the first per-subject decode happens.
    N: int
    T: int
    D: int
    D_grad: int                     # 0 when no grad_loader (dMSHBM-no-grad)

    # Group prior
    mtc: np.ndarray                 # (D+1, L) fp64 — from group.mat
    dim: int                        # = D = mtc.shape[0] - 1

    # Spatial mask
    boundary_mask: np.ndarray       # (N, L) fp32


# ─────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Step2Result:
    """Bundled outputs of one ``Step2Pipeline.run()``.

    ``timings`` keys, CPU backend: ``load_inputs`` (+ ``.mesh`` /
    ``.cohort`` / ``.profiles`` / ``.gradients`` / ``.group_mtc`` /
    ``.boundary``), ``initialize_params`` (+ ``.compose``),
    ``session_ctor``, ``em_total``, ``run_em``, ``save``, ``total``.
    The ``gpu`` backend adds ``init_device`` and
    ``closure_total`` (and forwards the loader's own per-item keys
    under the same ``load_inputs.<item>`` prefix, plus Session timings
    under ``session.<name>``); it has no ``initialize_params.compose``
    (there is no host compose). The loader's own ``total`` key is NOT
    forwarded — there is no ``load_inputs.total``; that number is
    ``load_inputs`` itself, measured by the pipeline around the call.

    ``em_iters_total`` is the number of inner EM iterations actually run
    across the whole ``run()`` — Σ over every EM-body call of that call's
    ``em_iters``, i.e. the number of ``run_iter`` calls on either
    backend. It is the right
    denominator for a per-EM-iteration mean of ``timings['em_total']``
    (``len(intra_em_iters_per_inter)`` counts EM *bodies*, not iterations).
    """
    Params_final_path: Path
    inter_iters: int
    intra_em_iters_per_inter: List[int]
    final_cost: float
    timings: Dict[str, float] = field(default_factory=dict)
    em_iters_total: int = 0


# ─────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────
class Step2Pipeline:
    """End-to-end step-2 group-prior estimation."""

    def __init__(self, cfg: Step2Config,
                 *, progress: Optional[ProgressEmitter] = None) -> None:
        self.cfg = cfg
        self.timings: Dict[str, float] = {}
        # Frontend progress sink. Step2 has no per-subject boundary
        # (coupled cohort EM); it emits one ``running`` state for the
        # synthetic sub_id=COHORT_SUB_ID + per-iter sub-progress inside
        # the outer/inner EM loops, then ``failed`` on exception. The
        # terminal ``done`` emit lives in
        # ``Pipeline._run_step2_train_prior`` (driver wrapper) AFTER the
        # ``prior_dst.exists()`` artifact check — see :meth:`run` for
        # the rationale. ``progress=None`` → standalone unit tests and
        # legacy regression scripts; storing a disabled emitter rather
        # than ``None`` keeps every emit call site branchless (no
        # per-call ``is not None`` guard), matching the step0/1/3 stage
        # pipelines' pattern.
        self.progress: ProgressEmitter = (
            progress if progress is not None
            else ProgressEmitter.disabled()
        )

    # ── lifecycle ──
    def __enter__(self) -> "Step2Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # GPU backend: free the cupy memory pool on exit. Mirrors
        # Step3Pipeline.close() — without this, the cupy pool keeps the
        # ~2.3 GB peak across runs (the pool retains free blocks for
        # reuse by default; nvidia-smi shows the peak even after the
        # Pipeline goes out of scope). CPU backend: import is skipped.
        if self.cfg.backend == "gpu":
            try:
                import cupy as cp  # type: ignore[import-not-found]
            except ImportError:
                return
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)

    # ── stage 1 — load_inputs ──
    def load_inputs(self) -> Step2Inputs:
        cfg = self.cfg
        t0 = time.perf_counter()
        self._log(f"[step2] load_inputs  mode={cfg.mode}  S={cfg.num_sub}  T={cfg.num_session}  L={cfg.num_clusters}")

        # Read cohort.json — the explicit roster + artifact ledger
        # written by step1, used as the sole discovery mechanism.
        t_cohort = time.perf_counter()
        cohort = read_cohort(cfg.project_dir)
        if cohort.num_sub != cfg.num_sub:
            raise ValueError(
                f"Step2Pipeline: cfg.num_sub={cfg.num_sub} != "
                f"cohort.json num_sub={cohort.num_sub}"
            )
        if cohort.num_session != cfg.num_session:
            raise ValueError(
                f"Step2Pipeline: cfg.num_session={cfg.num_session} != "
                f"cohort.json num_session={cohort.num_session}"
            )
        if cohort.mesh.get("targ") != cfg.mesh:
            raise ValueError(
                f"Step2Pipeline: cfg.mesh={cfg.mesh!r} != "
                f"cohort.json mesh.targ={cohort.mesh.get('targ')!r}"
            )
        if cohort.mesh.get("seed") != cfg.seed_mesh:
            raise ValueError(
                f"Step2Pipeline: cfg.seed_mesh={cfg.seed_mesh!r} != "
                f"cohort.json mesh.seed={cohort.mesh.get('seed')!r}"
            )
        t_cohort = time.perf_counter() - t_cohort

        # Mesh — need MARS_label per hemi (for medial-wall zeroing in load_subject_profiles)
        t_mesh = time.perf_counter()
        lh_inf = load_avg_mesh("lh", cfg.mesh, "inflated")
        rh_inf = load_avg_mesh("rh", cfg.mesh, "inflated")
        # MARS_label is (1, N) per MATLAB; flatten + bool-typed (1 == medial wall).
        lh_mars = (np.asarray(lh_inf["MARS_label"]).ravel() == 1)
        rh_mars = (np.asarray(rh_inf["MARS_label"]).ravel() == 1)
        n_lh = int(lh_mars.size)
        n_rh = int(rh_mars.size)
        t_mesh = time.perf_counter() - t_mesh

        # Build per-subject streaming loaders from the cohort manifest.
        # NO eager (S, N, T, D) buffer — the master kernel decodes one
        # subject at a time inside Phase A.3 / Phase D (BOLD) and
        # Phase C (grad).
        #
        # Peeked dims are cached on Step2Inputs so initialize_params and
        # Session ctor can size scratch buffers without re-touching disk.
        t_profiles = time.perf_counter()
        N = int(n_lh + n_rh)
        # Loader-level cache mode: 'auto' / 'eager_bitpacked' both map
        # to the bitpacked host cache (the only loader cache mode);
        # 'stream' re-reads packed bytes from disk per call.
        if cfg.bold_cache_mode in ("auto", "eager_bitpacked"):
            loader_cache_mode = "eager_bitpacked"
        else:
            loader_cache_mode = "stream"
        bold_loader = SubjectProfileLoader(
            cohort=cohort,
            project_dir=cfg.project_dir,
            targ_mesh=cfg.mesh,
            seed_mesh=cfg.seed_mesh,
            lh_mars=lh_mars,
            rh_mars=rh_mars,
            cache_mode=loader_cache_mode,
        )
        N_peeked, T_peeked, D_peeked = bold_loader.dims()
        if cfg.verbose:
            self._log(
                f"  CPU bold_loader cache_mode={loader_cache_mode!r} "
                f"(requested cfg.bold_cache_mode={cfg.bold_cache_mode!r})"
            )
        if N_peeked != N:
            raise ValueError(
                f"bold_loader.dims() N={N_peeked} != n_lh+n_rh={N}"
            )
        if T_peeked != cfg.num_session:
            raise ValueError(
                f"bold_loader.dims() T={T_peeked} != cfg.num_session={cfg.num_session}"
            )
        t_profiles_total = time.perf_counter() - t_profiles
        self._log(
            f"  bold_loader   S={cfg.num_sub}  format={bold_loader.format}  "
            f"(N, T, D) = ({N_peeked}, {T_peeked}, {D_peeked})  "
            f"peek {t_profiles_total*1000:.0f} ms  "
            f"per-subject scratch ~{N_peeked*T_peeked*D_peeked*4/1e6:.0f} MB"
        )

        t_gradients_total = 0.0
        grad_loader: Optional[SubjectGradientLoader] = None
        D_grad = 0
        if cfg.mode == "gMSHBM":
            t_grad = time.perf_counter()
            grad_loader = SubjectGradientLoader(
                cohort=cohort,
                project_dir=cfg.project_dir,
                n_components=cfg.n_grad_components,
            )
            N_g, D_grad = grad_loader.dims()
            if N_g != N:
                raise ValueError(
                    f"grad_loader.dims() N={N_g} != n_lh+n_rh={N}"
                )
            t_gradients_total = time.perf_counter() - t_grad
            self._log(
                f"  grad_loader   S={cfg.num_sub}  format={grad_loader.format}  "
                f"(N, D_grad) = ({N_g}, {D_grad})  "
                f"peek {t_gradients_total*1000:.0f} ms  "
                f"per-subject scratch ~{T_peeked*N_g*D_grad*4/1e6:.0f} MB"
            )

        # Group prior
        t_group = time.perf_counter()
        group = load_group_mtc(cfg.project_dir / "group" / "group.mat")
        mtc = group["mtc"]                                          # (D+1, L) fp64
        dim = int(mtc.shape[0] - 1)
        assert mtc.shape[1] == cfg.num_clusters, (
            f"group.mtc has L={mtc.shape[1]} but cfg.num_clusters={cfg.num_clusters}"
        )
        t_group = time.perf_counter() - t_group

        # Boundary mask
        t_boundary = time.perf_counter()
        lh_b, rh_b = load_spatial_mask(
            cfg.project_dir / "spatial_mask" / f"spatial_mask_{cfg.mesh}.mat"
        )
        boundary_mask = build_step2_boundary_mask(
            lh_boundary=lh_b, rh_boundary=rh_b, mode=cfg.mode,
        )
        t_boundary = time.perf_counter() - t_boundary

        self.timings["load_inputs"] = time.perf_counter() - t0
        # Per-sub-stage breakdown (sum should match the total within noise).
        self.timings["load_inputs.mesh"] = t_mesh
        self.timings["load_inputs.cohort"] = t_cohort
        self.timings["load_inputs.profiles"] = t_profiles_total
        self.timings["load_inputs.gradients"] = t_gradients_total
        self.timings["load_inputs.group_mtc"] = t_group
        self.timings["load_inputs.boundary"] = t_boundary
        self._log(
            f"  load_inputs done in {self.timings['load_inputs']:.2f}s  "
            f"(mesh {t_mesh:.2f} + cohort {t_cohort*1000:.0f}ms + "
            f"profiles {t_profiles_total:.2f} + gradients {t_gradients_total:.2f} "
            f"+ group {t_group:.2f} + boundary {t_boundary:.2f})"
        )

        return Step2Inputs(
            lh_mars=lh_mars, rh_mars=rh_mars, n_lh=n_lh, n_rh=n_rh,
            bold_loader=bold_loader, grad_loader=grad_loader,
            N=N_peeked, T=T_peeked, D=D_peeked, D_grad=D_grad,
            mtc=mtc, dim=dim,
            boundary_mask=boundary_mask,
        )

    # ── stage 2 — init Params ──
    def initialize_params(self, inputs: Step2Inputs) -> Dict[str, Any]:
        cfg = self.cfg
        t0 = time.perf_counter()
        dim = inputs.dim
        S = cfg.num_sub
        L = cfg.num_clusters
        T = cfg.num_session
        N = inputs.boundary_mask.shape[0]

        ini_val = float(initialize_concentration(dim))
        self._log(f"  ini_val (vMF init concentration) = {ini_val:.2f}")

        # Internal layout:
        #   s_psi   : (S, L, D)    fp32
        #   s_t_nu  : (S, T, L, D) fp32
        #   mu      : (L, D)       fp32   (transposed from group.mat's (D, L))
        #   s_lambda: (S, N, L)    fp32
        mtc_LD = np.ascontiguousarray(
            inputs.mtc.astype(np.float32).T,
        )  # (L, D) C-contig
        D = mtc_LD.shape[1]

        Params: Dict[str, Any] = {
            "ini_val": ini_val,
            "sigma": (ini_val * np.ones(L, dtype=np.float32)).reshape(1, L),
            "epsil": (ini_val * np.ones(L, dtype=np.float32)).reshape(1, L),
            "kappa": (ini_val * np.ones(L, dtype=np.float32)).reshape(1, L),
            "mu":    mtc_LD,
            "s_psi": np.broadcast_to(mtc_LD, (S, L, D)).copy(),
            "s_t_nu": np.broadcast_to(mtc_LD, (S, T, L, D)).copy(),
            "iter_inter": 0,
            "Record": [],
        }

        # Fused compose of Params["s_lambda"] + Params["theta"] (MATLAB
        # lines 137-180). Streams per-subject BOLD through a single
        # scratch slot, writes Params["s_lambda"] directly in (S, N, L)
        # fp32 layout, and computes theta inline from the per-(n, l)
        # active-subject count.
        t_compose = time.perf_counter()
        Params["s_lambda"], Params["theta"] = compose_init_state(
            bold_loader=inputs.bold_loader,
            group_mtc=inputs.mtc,
            boundary_mask=inputs.boundary_mask,
            num_clusters=L,
        )
        self.timings["initialize_params.compose"] = time.perf_counter() - t_compose

        self.timings["initialize_params"] = time.perf_counter() - t0
        return Params

    # ── stage 3 — outer EM loop ──
    def run_em(self, Params: Dict[str, Any], inputs: Step2Inputs) -> Step2Result:
        # CPU path only. The 'gpu' backend runs ``_run_em_sparse``
        # instead — see :meth:`run`.
        #
        # Lazy leaf imports: keeping them here (rather than at module
        # scope) is what makes a GPU-backend process skip the ~0.2 s
        # eager-numba ``em_stop_criterion._cdln`` import chain that
        # ``step2_em_outer`` drags in.
        from arealmshbm.step2_em_outer import (
            intra_subject_var_loop,
            inter_subject_var,
            intra_em_cost_step2,
            reset_s_psi_from_mtc_SLD,
        )

        cfg = self.cfg
        dim = inputs.dim
        L = cfg.num_clusters
        S = cfg.num_sub
        T = cfg.num_session

        stop_inter = False
        cost_inter = 0.0
        intra_em_per_inter: List[int] = []
        t_run = time.perf_counter()
        em_wall_accum = 0.0
        em_iters_total = 0

        # ─── Session lifecycle ───────────────────────────────────────────
        # Build the Session ONCE per run_em. BOLD/grad/boundary mask are
        # static across all intra/inter iters; only s_psi/sigma change
        # between intra-EM iters (refreshed via ``sess.refresh_s_psi_sigma``
        # from within vmf_clustering_batch).
        from arealmshbm.step2_em_iter_master import Step2EmIterSession
        t_ctor = time.perf_counter()
        sess = Step2EmIterSession(
            bold_loader=inputs.bold_loader,
            grad_loader=inputs.grad_loader if cfg.mode == "gMSHBM" else None,
            num_sub=cfg.num_sub,
            N=inputs.N, T=inputs.T, D=inputs.D, D_grad=inputs.D_grad,
            boundary_mask=inputs.boundary_mask,
            s_psi=Params["s_psi"],
            sigma=Params["sigma"],
            mode=cfg.mode,
            dim=dim,
            num_clusters=L,
            ini_val=Params["ini_val"],
            beta_internal=cfg.beta_internal,
            n_lh=inputs.n_lh,
            eps_m_step=cfg.epsilon,
            max_iter_m=cfg.max_iter_m,
        )
        self.timings["session_ctor"] = time.perf_counter() - t_ctor

        # ─── Upload initial Params state into the Session ────────────────
        # Aliases Params['s_lambda'/'theta'/'s_t_nu'] to Session-owned
        # scratch (zero-copy fast path; the alternative is 90 ms / iter of
        # memcpy).
        #
        # CAUTION — Params['s_lambda' / 'theta' / 's_t_nu']
        # ARE views into Session-owned scratch after this call. Mid-run
        # inspection / checkpointing must ``.copy()`` first; the next
        # ``sess.run_iter`` overwrites them in place.
        sess.upload_initial_state(Params)

        # Pre-compute mtc_LD once for the reset kernels (host (L, D)
        # fp32 C-contig). The Session keeps a host reference.
        mtc_LD = np.ascontiguousarray(inputs.mtc.astype(np.float32).T)
        sess.cache_mtc(mtc_LD)

        while not stop_inter:
            Params["iter_inter"] = Params["iter_inter"] + 1
            self._log(f"\n=== [step2] Inter iter {Params['iter_inter']}/{cfg.max_iter_inter} ({cfg.mode}) ===")
            # Outer-iter heartbeat. Frontend renders "step2: outer
            # X/Y" between the running and done state events. No
            # intra index here — it's emitted below in the intra
            # loop body. ``self.progress`` is always a real emitter
            # (or a no-op disabled() one) — branchless on purpose.
            self.progress.emit_iter(
                "step2",
                iter_inter=Params["iter_inter"],
                max_inter=cfg.max_iter_inter,
            )

            # Reset sigma + s_psi each inter iter (MATLAB lines 212-213).
            # Both are host operations (the Session picks them up at the
            # start of the next vmf_clustering_batch via
            # refresh_s_psi_sigma). Keep this reset and the vmf_clustering_batch
            # call adjacent — inserting any consumer of sess._sigma_L
            # between them would read the stale pre-reset value.
            Params["sigma"][...] = Params["ini_val"]
            reset_s_psi_from_mtc_SLD(Params["s_psi"], mtc_LD)

            cost_intra_em = 0.0
            for iter_intra_em in range(1, cfg.max_iter_intra_em + 1):
                Params["iter_intra"] = iter_intra_em
                self._log(f"  Intra-EM iter {iter_intra_em} ...")
                self.progress.emit_iter(
                    "step2",
                    iter_inter=Params["iter_inter"],
                    max_inter=cfg.max_iter_inter,
                    iter_intra=iter_intra_em,
                    max_intra=cfg.max_iter_intra_em,
                )

                # Reset kappa + s_t_nu each intra iter (MATLAB lines 220-221).
                # kappa is host-only (read scalar-by-scalar by run_iter);
                # the s_t_nu reset is the Session's (it owns the buffer).
                Params["kappa"][...] = Params["ini_val"]
                sess.reset_s_t_nu_from_mtc()

                t_em = time.perf_counter()
                batch = vmf_clustering_batch(
                    Params,
                    sess=sess,
                    max_iter_em=cfg.max_iter_em,
                    em_convergence_eps=cfg.em_convergence_eps,
                    verbose=cfg.verbose,
                )
                em_wall_accum += (time.perf_counter() - t_em)
                em_iters_total += int(batch.em_iters)

                # intra_subject_var_loop — updates s_psi + sigma
                s_psi_new, sigma_new, _flag_psi = intra_subject_var_loop(
                    s_t_nu=Params["s_t_nu"],
                    s_psi_in=Params["s_psi"],
                    sigma_in=Params["sigma"],
                    epsil=Params["epsil"],
                    mu=Params["mu"],
                    dim=dim,
                    ini_val=Params["ini_val"],
                    epsilon=cfg.epsilon,
                    max_iter=cfg.max_iter_intra_var,
                )
                Params["s_psi"] = s_psi_new
                Params["sigma"] = sigma_new

                # intra_em_cost_step2
                update_cost = intra_em_cost_step2(Params, dim=dim)
                Params["cost_intra"] = update_cost

                # Convergence: |cost_new - cost_prev| / cost_prev < eps
                if cost_intra_em != 0.0:
                    rel = abs(abs(update_cost - cost_intra_em) / cost_intra_em)
                    if rel <= cfg.intra_em_convergence_eps:
                        self._log(f"  intra-EM converged at iter {iter_intra_em} (rel-diff {rel:.2e})")
                        break
                cost_intra_em = update_cost

            intra_em_per_inter.append(iter_intra_em)

            # inter_subject_var
            mu_new, epsil_new = inter_subject_var(
                s_psi=Params["s_psi"],
                prev_mu=Params["mu"],
                prev_epsil=Params["epsil"],
                dim=dim,
                ini_val=Params["ini_val"],
            )
            Params["mu"] = mu_new
            Params["epsil"] = epsil_new

            update_cost_inter = Params["cost_intra"]
            Params["Record"].append(float(update_cost_inter))
            self._log(f"  inter cost={float(update_cost_inter):+.4e}")

            # Outer convergence
            if cost_inter != 0.0:
                rel_outer = abs(abs(update_cost_inter - cost_inter) / cost_inter)
                if rel_outer <= cfg.inter_convergence_eps:
                    self._log(f"  inter converged (rel-diff {rel_outer:.2e})")
                    stop_inter = True
            if Params["iter_inter"] >= cfg.max_iter_inter:
                self._log(f"  reached max_iter_inter={cfg.max_iter_inter}; stopping")
                stop_inter = True
            cost_inter = update_cost_inter

        self.timings["em_total"] = em_wall_accum
        self.timings["run_em"] = time.perf_counter() - t_run

        # Final save (sequential nulls s_lambda/s_psi/s_t_nu to keep file small).
        Params["cost_inter"] = cost_inter
        Params_save = {k: v for k, v in Params.items() if k not in ("s_lambda", "s_psi", "s_t_nu")}
        final_path = Path(self.cfg.out_dir) / "Params_Final.mat"
        t_save = time.perf_counter()
        self._save_params(Params_save, final_path)
        self.timings["save"] = time.perf_counter() - t_save

        return Step2Result(
            Params_final_path=final_path,
            inter_iters=Params["iter_inter"],
            intra_em_iters_per_inter=intra_em_per_inter,
            final_cost=float(cost_inter),
            timings=self.timings,
            em_iters_total=em_iters_total,
        )

    # ── stage 4 — save_params helper ──
    #
    # Params dict during run_em holds INTERNAL layout. Conversion back
    # to the (D-leading) external convention happens only here, at
    # ``.mat`` save time.
    @staticmethod
    def _save_params(Params: Dict[str, Any], path: Path) -> None:
        from ._save import save_params_final
        save_params_final(Params, path)

    # ─────────────────────────────────────────────────────────────────
    # GPU ('gpu') backend — stages 1-3
    #
    # Same four stages as the CPU path, but every stage speaks the
    # P-layout contract in docs/step2_sparse_design.md §4/§5 instead of
    # the dense (N, L) one:
    #
    #   load_inputs_sparse       → Step2SparseInputs  (§5)
    #   initialize_params_sparse → host bookkeeping only
    #   _run_em_sparse           → Step2SparseSession  (§4)
    #
    # There is no ``Step2Inputs``, no host ``boundary_mask``, no
    # ``compose_init_state`` and no host outer-EM leaf on this path —
    # ``s_lambda`` / ``theta`` / ``s_psi`` / ``s_t_nu`` / ``mu`` never
    # exist as host arrays until ``export_params()`` at save time.
    # ─────────────────────────────────────────────────────────────────
    def load_inputs_sparse(self) -> "Step2SparseInputs":
        """Stage 1, GPU backend — see ``step2_io.load_step2_sparse_inputs``."""
        cfg = self.cfg
        t0 = time.perf_counter()
        self._log(
            f"[step2] load_inputs (sparse)  mode={cfg.mode}  S={cfg.num_sub}  "
            f"T={cfg.num_session}  L={cfg.num_clusters}"
        )
        # Lazy: the loader pulls blosc2 / scipy.sparse and must not be on
        # the CPU path's import graph.
        from arealmshbm.step2_io import load_step2_sparse_inputs

        inputs = load_step2_sparse_inputs(cfg)
        self.timings["load_inputs"] = time.perf_counter() - t0
        # Per-item breakdown, same ``load_inputs.<item>`` key style the
        # CPU path uses (mesh / cohort / profiles / gradients /
        # group_mtc / boundary), forwarded verbatim from the loader.
        for k, v in (inputs.timings or {}).items():
            if k == "total":            # already reported as ``load_inputs``
                continue
            self.timings[f"load_inputs.{k}"] = float(v)
        self._log(
            f"  P={inputs.layout.P} of N*L={inputs.N * inputs.layout.L} "
            f"({100.0 * inputs.layout.P / max(1, inputs.N * inputs.layout.L):.2f}%)  "
            f"(N, T, D) = ({inputs.N}, {inputs.T}, {inputs.D})  "
            f"D_grad={inputs.D_grad}  "
            f"packed_host={'yes' if inputs.packed_host is not None else 'no'}"
        )
        self._log(
            f"  load_inputs done in {self.timings['load_inputs']:.2f}s"
        )
        return inputs

    def initialize_params_sparse(
        self, inputs: "Step2SparseInputs",
    ) -> Dict[str, Any]:
        """Stage 2, GPU backend — host bookkeeping only.

        Everything the CPU ``initialize_params`` materialises on host
        (``mu`` / ``s_psi`` / ``s_t_nu`` from ``mtc``, ``s_lambda`` /
        ``theta`` from ``compose_init_state``) is built on device by
        ``Step2SparseSession.initialize_state()`` instead. What stays
        host-side is the scalar ``ini_val`` and the bookkeeping keys the
        outer loop and the save path read (``iter_inter``, ``Record``).
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        ini_val = float(initialize_concentration(inputs.dim))
        self._log(f"  ini_val (vMF init concentration) = {ini_val:.2f}")
        Params: Dict[str, Any] = {
            "ini_val": ini_val,
            "iter_inter": 0,
            "Record": [],
        }
        self.timings["initialize_params"] = time.perf_counter() - t0
        return Params

    def _run_em_sparse(
        self, Params: Dict[str, Any], inputs: "Step2SparseInputs",
    ) -> Step2Result:
        """Stage 3, GPU backend — the outer/intra/EM control flow.

        Structurally identical to :meth:`run_em`: same loop bounds, same
        log lines, same progress emits, same ``Record`` and convergence
        arithmetic. The differences are all on the Session side —

        * the host resets (``Params['sigma'][...] = ini_val`` +
          ``reset_s_psi_from_mtc_SLD``; ``Params['kappa'][...] = ini_val``
          + ``sess.reset_s_t_nu_from_mtc()``) become
          ``sess.reset_inter()`` / ``sess.reset_intra()``;
        * ``vmf_clustering_batch`` becomes ``em_body_sparse`` (identical
          convergence semantics, no ``Params`` round trip);
        * ``intra_subject_var_loop`` (L17) + ``intra_em_cost_step2`` (L16)
          become ``sess.intra_closure()``, which returns the same scalar;
        * ``inter_subject_var`` (L18) becomes ``sess.inter_closure()``;
        * the final host state comes from one ``sess.export_params()``.
        """
        cfg = self.cfg
        dim = inputs.dim
        L = cfg.num_clusters

        stop_inter = False
        cost_inter = 0.0
        intra_em_per_inter: List[int] = []
        t_run = time.perf_counter()
        em_wall_accum = 0.0
        closure_wall_accum = 0.0
        em_iters_total = 0

        # Lazy import — a CPU run must never touch cupy.
        from arealmshbm.step2_em_iter_master import (
            Step2SparseSession, warmup_step2_gpu,
        )

        # NVRTC compile + cuBLAS's process-wide module load, off the EM
        # path. A no-op once the driver's prewarm daemon has run; the
        # per-step API and the profiling harnesses do not prewarm, and
        # would otherwise charge ~150 ms to the first ``run_iter``.
        warmup_step2_gpu()

        t_ctor = time.perf_counter()
        sess = Step2SparseSession(
            inputs,
            mode=cfg.mode,
            num_clusters=L,
            dim=dim,
            ini_val=Params["ini_val"],
            beta_internal=cfg.beta_internal,
            eps_m_step=cfg.epsilon,
            max_iter_m=cfg.max_iter_m,
            eps_intra_var=cfg.epsilon,
            max_iter_intra_var=cfg.max_iter_intra_var,
            bold_cache_mode=cfg.bold_cache_mode,
            bold_cache_safety_margin_gb=cfg.gpu_cache_safety_margin_gb,
        )
        self.timings["session_ctor"] = time.perf_counter() - t_ctor
        if cfg.verbose:
            mode_resolved = getattr(sess, "bold_cache_mode", None)
            if mode_resolved is None:
                mode_resolved = getattr(sess, "_bold_cache_mode", "unknown")
            self._log(
                f"  GPU BOLD cache mode: requested={cfg.bold_cache_mode!r} "
                f"resolved={mode_resolved!r}"
            )

        # Device-side init: K1 (hard labels + compose) + log_theta +
        # the first active-support build. Replaces the CPU path's
        # host ``compose_init_state`` + ``upload_initial_state`` +
        # ``cache_mtc``.
        t_init_dev = time.perf_counter()
        sess.initialize_state()
        self.timings["init_device"] = time.perf_counter() - t_init_dev

        while not stop_inter:
            Params["iter_inter"] = Params["iter_inter"] + 1
            self._log(f"\n=== [step2] Inter iter {Params['iter_inter']}/{cfg.max_iter_inter} ({cfg.mode}) ===")
            self.progress.emit_iter(
                "step2",
                iter_inter=Params["iter_inter"],
                max_inter=cfg.max_iter_inter,
            )

            # Reset sigma + s_psi each inter iter (MATLAB lines 212-213),
            # on device. Same ordering hazard as the dense path: keep
            # this adjacent to the EM body — anything reading the
            # Session's sigma in between would see the pre-reset value.
            sess.reset_inter()

            cost_intra_em = 0.0
            for iter_intra_em in range(1, cfg.max_iter_intra_em + 1):
                Params["iter_intra"] = iter_intra_em
                self._log(f"  Intra-EM iter {iter_intra_em} ...")
                self.progress.emit_iter(
                    "step2",
                    iter_inter=Params["iter_inter"],
                    max_inter=cfg.max_iter_inter,
                    iter_intra=iter_intra_em,
                    max_intra=cfg.max_iter_intra_em,
                )

                # Reset kappa + s_t_nu each intra iter (MATLAB 220-221).
                sess.reset_intra()

                t_em = time.perf_counter()
                batch = em_body_sparse(
                    sess,
                    max_iter_em=cfg.max_iter_em,
                    em_convergence_eps=cfg.em_convergence_eps,
                    verbose=cfg.verbose,
                )
                em_wall_accum += (time.perf_counter() - t_em)
                em_iters_total += int(batch.em_iters)
                Params["cost_em"] = batch.cost_S

                # L17 (intra_subject_var_loop) + L16 (intra_em_cost_step2),
                # both on device, one host scalar out.
                t_cl = time.perf_counter()
                update_cost = float(sess.intra_closure())
                closure_wall_accum += (time.perf_counter() - t_cl)
                Params["cost_intra"] = update_cost

                # Convergence: |cost_new - cost_prev| / cost_prev < eps
                if cost_intra_em != 0.0:
                    rel = abs(abs(update_cost - cost_intra_em) / cost_intra_em)
                    if rel <= cfg.intra_em_convergence_eps:
                        self._log(f"  intra-EM converged at iter {iter_intra_em} (rel-diff {rel:.2e})")
                        break
                cost_intra_em = update_cost

            intra_em_per_inter.append(iter_intra_em)

            # L18 (inter_subject_var) — mu + epsil, on device.
            t_cl = time.perf_counter()
            sess.inter_closure()
            closure_wall_accum += (time.perf_counter() - t_cl)

            update_cost_inter = Params["cost_intra"]
            Params["Record"].append(float(update_cost_inter))
            self._log(f"  inter cost={float(update_cost_inter):+.4e}")

            # Outer convergence
            if cost_inter != 0.0:
                rel_outer = abs(abs(update_cost_inter - cost_inter) / cost_inter)
                if rel_outer <= cfg.inter_convergence_eps:
                    self._log(f"  inter converged (rel-diff {rel_outer:.2e})")
                    stop_inter = True
            if Params["iter_inter"] >= cfg.max_iter_inter:
                self._log(f"  reached max_iter_inter={cfg.max_iter_inter}; stopping")
                stop_inter = True
            cost_inter = update_cost_inter

        self.timings["em_total"] = em_wall_accum
        self.timings["closure_total"] = closure_wall_accum
        self.timings["run_em"] = time.perf_counter() - t_run

        # One D2H for the whole saved state. ``s_lambda`` / ``s_psi`` /
        # ``s_t_nu`` are never exported (the CPU path strips exactly
        # those three keys).
        exported = sess.export_params()
        # Handed to the writer as-is: ``_save.save_params_final`` owns the
        # densify policy.
        theta = exported["theta"]          # scipy csc_matrix (N, L) fp64
        Params["sigma"] = exported["sigma"]
        Params["epsil"] = exported["epsil"]
        Params["kappa"] = exported["kappa"]
        Params["mu"] = exported["mu"]
        Params["cost_em"] = exported["cost_em"]
        Params["theta"] = theta
        Params["cost_inter"] = cost_inter
        Params_save = {k: v for k, v in Params.items() if k not in ("s_lambda", "s_psi", "s_t_nu")}
        final_path = Path(self.cfg.out_dir) / "Params_Final.mat"
        t_save = time.perf_counter()
        self._save_params(Params_save, final_path)
        self.timings["save"] = time.perf_counter() - t_save

        sess_timings = getattr(sess, "timings", None)
        if sess_timings:
            # Forwarded VERBATIM under ``session.<name>``: four host
            # wall entries in SECONDS (``session_ctor``,
            # ``initialize_state``, ``intra_closure``, ``inter_closure``),
            # each cumulative over every call in the run.
            for k, v in sess_timings.items():
                self.timings[f"session.{k}"] = float(v)

        return Step2Result(
            Params_final_path=final_path,
            inter_iters=Params["iter_inter"],
            intra_em_iters_per_inter=intra_em_per_inter,
            final_cost=float(cost_inter),
            timings=self.timings,
            em_iters_total=em_iters_total,
        )

    # ── public ──
    def run(self) -> Step2Result:
        t0 = time.perf_counter()
        # Progress: emit ``running`` at entry and ``failed`` on
        # exception. The ``done`` emit deliberately lives in the
        # driver wrapper (see ``Pipeline._run_step2_train_prior``)
        # because the driver — not this class — is the authority on
        # "did step2 actually produce ``Params_Final.mat`` at the
        # expected slot." Emitting ``done`` here would mark the slot
        # complete BEFORE that artifact check, so a silent early-exit
        # in ``run_em`` (no exception, no file) would render as
        # success on the frontend even though the driver immediately
        # raises ``FileNotFoundError``. Standalone callers that don't
        # care about the artifact check can call
        # ``progress.emit_state("step2", COHORT_SUB_ID, "done")``
        # themselves if they want the terminal event.
        self.progress.emit_state("step2", COHORT_SUB_ID, "running")
        try:
            # Backend dispatch lives HERE, not below load_inputs: the
            # GPU backend's stage-1 output is a different dataclass
            # (Step2SparseInputs — no bold_loader, no boundary_mask), so
            # the three stages must be selected as a set. Splitting the
            # dispatch would hand a Step2SparseInputs to the CPU
            # ``run_em`` and fail with an ``AttributeError`` inside this
            # ``try``, i.e. as a confusing "failed" progress event.
            if self.cfg.backend == "gpu":
                inputs_sparse = self.load_inputs_sparse()
                Params = self.initialize_params_sparse(inputs_sparse)
                result = self._run_em_sparse(Params, inputs_sparse)
            else:
                inputs = self.load_inputs()
                Params = self.initialize_params(inputs)
                result = self.run_em(Params, inputs)
        except BaseException as e:
            self.progress.emit_state(
                "step2", COHORT_SUB_ID, "failed",
                error=f"{type(e).__name__}: {e}",
            )
            raise
        self.timings["total"] = time.perf_counter() - t0
        if self.cfg.verbose:
            print(f"\n[step2] Done. Total wall: {self.timings['total']:.1f}s", flush=True)
            for k, v in self.timings.items():
                print(f"  {k:24s} {v:8.2f}s", flush=True)
        return result
