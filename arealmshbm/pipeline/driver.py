"""driver.py — the unified project-driven pipeline orchestrator.

A single ``Pipeline(project_dir).run()`` call replaces the historical
mix of per-step entry points + MATLAB driver chains.

Flow per mode::

    modeA_*:  min-input check → step0 (×K) → step1 → step3 (×K)
    modeB_*:  min-input check → step0 (×K) → step1 → step2 → step3 (×K)

Min-input check is the only existence test performed: BOLD files (both
modes), and (Mode A only) a pre-existing group prior at the
project-local slot
``<project>/priors/<variant>/beta<beta_scalar>/Params_Final.mat``.
Staging that prior is the project creator's responsibility — the
pipeline never reaches outside the project (no HCP/CBIG fallback, no
cross-project lookup); Mode B produces the prior there itself via
step2. The driver does NOT check for intermediate artifacts to skip
steps — every ``run()`` is naive run-and-overwrite.

Step 1's four subgraphs (generate_profiles, avg_profiles, ini_params,
radius_mask) are called directly via :mod:`pipeline.step1_runners`,
and the cohort.json write is performed by :mod:`pipeline.cohort_writer`
at the end of the driver's run. Step 0 / 2 / 3 are dispatched through
their respective per-step ``Pipeline`` classes — those orchestrators
remain useful boundaries (per-subject lifecycle for step0/3,
multi-subject EM for step2).

The step0 → step3 gradient hand-off is in-memory (driver collects each
subject's ``(N, n_components)`` gradient matrix from step0 and forwards
it to ``Step3Pipeline(precomputed_gradient_mat=...)``); step1's avg
profile arrays are likewise threaded subgraph 2 → 3 without a disk
re-read.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._progress import COHORT_SUB_ID, ProgressEmitter, build_plan_entries
from .config import PipelineConfig, read_pipeline_config
from .inputs import BoldInputs, check_bold_files_exist, read_bold_inputs
from .layout import ProjectLayout


@dataclass
class PipelineRunResult:
    """One-shot record of what the driver did, written to ``logs/``.

    ``success`` is ``False`` and ``failed_at_step`` names the phase
    (``"step0"``, ``"step1"``, ...) when ``Pipeline.run()`` raises
    mid-flight; the log is still emitted from a try/finally so partial
    progress has an audit trail.
    """
    project_dir: str
    mode: str
    variant: str
    timestamp: str
    success: bool = True
    failed_at_step: Optional[str] = None
    error: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    step0_subjects: List[str] = field(default_factory=list)
    step1_subjects: List[str] = field(default_factory=list)
    step3_subjects: List[str] = field(default_factory=list)
    prior_path: Optional[str] = None
    cohort_json_path: Optional[str] = None


class Pipeline:
    """Project-driven pipeline driver.

    Construct with a project directory. The directory must contain
    ``bold_inputs.json`` (user-provided) and ``pipeline_config.json``
    (user-provided). The driver discovers the rest from there.
    """

    def __init__(self, project_dir: Path | str,
                 *, progress: Optional[ProgressEmitter] = None) -> None:
        self.layout = ProjectLayout(project_dir)
        self.inputs: BoldInputs = read_bold_inputs(self.layout.bold_inputs_path)
        self.config: PipelineConfig = read_pipeline_config(
            self.layout.pipeline_config_path
        )
        self._validate_inputs_vs_config()
        # Frontend progress sink. Default = on, writes
        # ``<project>/logs/progress.jsonl``. The caller can pass a
        # disabled emitter (``ProgressEmitter(..., enabled=False)``) to
        # opt out — e.g. perf scripts that must isolate zero file I/O.
        # Constructed last so any validation failure above short-
        # circuits before touching the logs dir.
        self._progress: ProgressEmitter = (
            progress if progress is not None
            else ProgressEmitter(self.layout.logs_dir)
        )

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────
    def _validate_inputs_vs_config(self) -> None:
        if self.config.mode == "modeA_single" and self.inputs.num_subjects != 1:
            raise ValueError(
                f"pipeline_config.json: mode='modeA_single' requires exactly "
                f"1 subject in bold_inputs.json (got {self.inputs.num_subjects})"
            )
        if (self.config.mode == "modeB_train_prior"
                and self.inputs.num_subjects < 2):
            raise ValueError(
                f"pipeline_config.json: mode='modeB_train_prior' requires "
                f">= 2 subjects (got {self.inputs.num_subjects}). Mode B's "
                f"coupled EM needs a cohort, not a single subject."
            )

    def _prior_includes_beta(self) -> bool:
        """Whether this variant's prior path carries a ``beta<X>`` segment.
        Single source of truth is the step3 ``VariantSpec`` — gMSHBM /
        cMSHBM include it, dMSHBM does not. Keeps the driver's prior path
        aligned with where step2 writes (Step2Config.out_dir) and where
        step3 reads (Step3Config.group_prior_path)."""
        from arealmshbm.step3_pipeline.variant import VariantSpec
        return VariantSpec.from_pipeline_type(
            self.config.variant).prior_path_includes_beta

    def _check_min_inputs(self) -> Path | None:
        """Raise on missing required inputs. Returns the project-local
        prior path for Mode A (None for Mode B)."""
        # All modes: every BOLD file must exist.
        check_bold_files_exist(self.inputs)

        if self.config.is_mode_a:
            # Mode A reads the prior ONLY from the project-local slot.
            # Staging it there is the project creator's job — the pipeline
            # does not reach into HCP/CBIG or other projects.
            prior_path = self.layout.prior_path(self.config.variant,
                                                self.config.beta_scalar,
                                                self._prior_includes_beta())
            if not prior_path.exists():
                raise FileNotFoundError(
                    f"Mode A min-input check: group prior not found at "
                    f"{prior_path}. Stage a Params_Final.mat there before "
                    f"running Mode A - e.g. copy one from a CBIG checkout's "
                    f"lib/group_priors/, from the in-tree example priors "
                    f"under arealmshbm/data/group_priors/, or from another "
                    f"project's modeB_train_prior output. (Mode B writes "
                    f"this slot itself via step2.)"
                )
            return prior_path
        return None

    # ─────────────────────────────────────────────────────────────────
    # Step staging — write data_list/fMRI_list/*.txt so step1 finds BOLD
    # ─────────────────────────────────────────────────────────────────
    def _stage_step1_bold_lists(self) -> None:
        """Step 1's ``compute_profile_arrays`` reads BOLD paths from
        ``project_dir/data_list/fMRI_list/{lh,rh}_sub<id>_sess<id>.txt``.
        Write them from the bold_inputs.json manifest before step1 runs.
        """
        list_dir = (self.layout.project_dir / "data_list" / "fMRI_list")
        list_dir.mkdir(parents=True, exist_ok=True)
        for sub in self.inputs.subjects:
            for sess in sub.sessions:
                (list_dir / f"lh_sub{sub.id}_sess{sess.id}.txt").write_text(
                    str(sess.lh) + "\n", encoding="utf-8"
                )
                (list_dir / f"rh_sub{sub.id}_sess{sess.id}.txt").write_text(
                    str(sess.rh) + "\n", encoding="utf-8"
                )

    # ─────────────────────────────────────────────────────────────────
    # Per-step runners
    # ─────────────────────────────────────────────────────────────────
    def _run_step0_all_subjects_maybe_tf32(self) -> Dict[str, "np.ndarray"]:
        """Driver wrapper around step0 that toggles cuBLAS TF32 math
        mode for the duration of the step (and only this step) when
        ``config.step0.enable_tf32`` is True and step0 runs on GPU.

        Only ``fc_similarity`` dispatches through cuBLAS sgemm inside
        step0; other GPU leaves (``diffusion_map`` eigsh, RawKernels)
        are unaffected by the math-mode toggle. The mode is restored
        on exit so adjacent step1/2/3 keep strict fp32.
        """
        if (self.config.step0.enable_tf32
                and self.config.backend_step0 == "gpu"):
            from ._cublas_math_mode import cublas_tf32_scope
            with cublas_tf32_scope():
                return self._run_step0_all_subjects()
        return self._run_step0_all_subjects()

    def _run_step0_all_subjects(self) -> Dict[str, "np.ndarray"]:
        """Step 0: gradients per subject. Each call writes
        ``project_dir/gradients/sub<id>/{lh,rh}_emb_<N>_distance_matrix.npy``
        as cache AND returns the concatenated ``(N, n_grad_components)``
        fp32 gradient matrix in memory.

        Returns a dict ``{sub_id: gradient_mat_NxDg}`` so step3 can
        consume gradients from memory without re-reading the .npy
        files step0 just wrote.

        **Pipeline-parallel by subgraph (GPU backend).** The single-
        subject pipeline runs four subgraphs sequentially (A: RSFC
        gradients, B: graph distance, C: diffusion embedding / eigsh,
        D: CPU upsample). Running whole subjects in parallel through
        a ThreadPoolExecutor stalls on the cuSOLVER / cuBLAS /
        cuSPARSE handle queues — only one eigsh runs at a time
        regardless of how many workers ask. So we pipeline by
        subgraph instead: one stage thread per subgraph, each on its
        own cupy stream, with cross-stream event sync on the B→C
        device-array handoff. cuSOLVER queue depth is always exactly
        1; subgraph A's RawKernels + cuBLAS sgemms run concurrently
        with stage C's eigsh on a different stream; stage D (CPU)
        overlaps the whole stack. See
        :mod:`arealmshbm.pipeline._step0_stage_pipeline`.

        CPU backend stays sequential (numba threads inside each call
        already saturate cores).
        """
        from arealmshbm.step0_pipeline import Step0Config, Step0Pipeline
        from ._step0_bold_prefetcher import Step0BoldPrefetcher
        from ._step0_stage_pipeline import Step0StagePipeline

        subjects = list(self.inputs.subjects)
        sess_list = tuple(
            f"ses-{int(s.id):02d}" for s in subjects[0].sessions
        )
        sess_paths_per_sub: List[List[Tuple[Path, Path]]] = [
            [(sess.lh, sess.rh) for sess in sub.sessions]
            for sub in subjects
        ]

        # One-off ``load_inputs`` to extract the mesh-derived
        # constants the prefetcher needs (medial_mask + hemi sizes).
        # Also populates the process-wide ``_INPUTS_CACHE`` so every
        # per-subject pipeline below hits tier 1.
        # ``**step0_kwargs`` splats every Step0Knobs algorithm field
        # into Step0Config via the field-name 1:1 mapping pinned by
        # test_knobs_field_match.test_step0_knobs_field_names_match_step0config
        # — adding an algorithm knob auto-flows here without a manual
        # edit. ``enable_tf32`` is the one exception: it's consumed by
        # the driver's tf32_scope wrapper (above), not by Step0Config
        # itself, so it's popped from the splat.
        step0_kwargs = asdict(self.config.step0)
        step0_kwargs.pop("enable_tf32")
        seed_cfg = Step0Config(
            project_dir=self.layout.project_dir,
            sub_id=subjects[0].id,
            sess_list=sess_list,
            out_subid=subjects[0].id,
            mesh=self.inputs.targ_mesh,
            num_components=self.config.n_grad_components,
            backend=self.config.backend_step0,
            bold_paths_override=tuple(sess_paths_per_sub[0]),
            **step0_kwargs,
        )
        with Step0Pipeline(seed_cfg) as seed_pipe:
            seed_inputs = seed_pipe.load_inputs()

        is_gpu = (self.config.backend_step0 == "gpu")
        # Lookahead = 4 (one ahead of each subgraph stage's in-flight
        # queue). Memory budget: 4 × 6 sessions × 72 MB ≈ 1.7 GB host
        # BOLD in flight. Fine.
        lookahead = 4

        # Issue #55: ``Step0BoldPrefetcher(backend='gpu')`` is
        # functionally complete (nvCOMP Deflate decode + on-device
        # concat_hemis_drop_medial, bit-equal to CPU — pinned by
        # ``arealmshbm/pipeline/tests/test_bold_prefetcher_gpu.py``).
        # The production driver still pins it to 'cpu' because under
        # cupy's legacy default stream the GPU decode would serialize
        # with subgraph A's compute on the same stream — i.e. add to
        # (not hide under) total wall. A real perf win needs per-
        # worker non-blocking streams; until that follow-up lands,
        # leave the cheap-and-overlapped CPU IO as the production
        # default. The class-level backend kwarg stays available for
        # benchmarking and future work.
        with Step0BoldPrefetcher(
            medial_mask=seed_inputs.medial_mask,
            n_lh=seed_inputs.n_lh, n_rh=seed_inputs.n_rh,
            backend="cpu",
        ) as prefetcher:
            for k in range(min(lookahead, len(subjects))):
                prefetcher.prime_subject(
                    subjects[k].id, sess_paths_per_sub[k])

            coord = Step0StagePipeline(
                project_dir=self.layout.project_dir,
                sess_list=sess_list,
                targ_mesh=self.inputs.targ_mesh,
                n_grad_components=self.config.n_grad_components,
                backend=self.config.backend_step0,
                prefetcher=prefetcher,
                sess_paths_per_sub=sess_paths_per_sub,
                lookahead=lookahead,
                step0_knobs=self.config.step0,
                progress=self._progress,
            )
            gradients = coord.run(subjects)

        # Belt-and-braces drain after all subjects done. Each stage-D
        # thread already free_all_blocks'd per subject; this also
        # catches the seed_inputs GPU mirrors before step2 starts.
        if is_gpu:
            try:
                import cupy as cp
                del seed_inputs
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

        return gradients

    def _run_step1_all_subjects(self) -> Dict[str, Any]:
        """Step 1: drives the four subgraphs directly via step1_runners.

        The driver owns the subgraph sequence, the inter-subgraph data
        flow (avg profile arrays threaded subgraph 2 → 3 in memory),
        and the cohort.json write at the end.

        Returns the bundle of artifacts the cohort writer needs to
        emit ``cohort.json``.
        """
        from .step1_runners import (
            resolve_group_labels,
            run_avg_profiles, run_generate_profiles,
            run_ini_params, run_radius_mask,
        )

        subjects = self.inputs.subject_ids()
        sessions = self.inputs.session_ids()
        targ = self.inputs.targ_mesh
        seed = self.inputs.seed_mesh
        backend = self.config.backend_step1
        k1 = self.config.step1

        # Subgraph 1 — per-subject profile arrays → bitpacked .b2nd.
        # The stage pipeline emits per-subject ``step1`` state events
        # internally; the driver injects ``self._progress`` via the
        # runner's keyword arg.
        profile_b2nd_paths = run_generate_profiles(
            project_dir=self.layout.project_dir,
            subjects=subjects,
            sessions=sessions,
            seed_mesh=seed,
            targ_mesh=targ,
            threshold=k1.threshold,
            split_flag=k1.split_flag,
            backend=backend,
            verbose=True,
            progress=self._progress,
        )

        # Subgraph 2 — cohort-average across subjects/sessions. Returns
        # both on-disk paths (cache) and in-memory ``lh_avg``/``rh_avg``
        # arrays, which subgraph 3 consumes directly.
        self._progress.emit_state("step1_avg", COHORT_SUB_ID, "running")
        try:
            avg_res = run_avg_profiles(
                project_dir=self.layout.project_dir,
                num_sub=len(subjects),
                num_sess=len(sessions),
                seed_mesh=seed,
                targ_mesh=targ,
                backend=backend,
            )
        except BaseException as e:
            self._progress.emit_state(
                "step1_avg", COHORT_SUB_ID, "failed",
                error=f"{type(e).__name__}: {e}")
            raise
        self._progress.emit_state("step1_avg", COHORT_SUB_ID, "done")

        # Resolve group labels once for both ini_params + radius_mask
        # (hot Schaefer .annot parse).
        lh_labels, rh_labels = resolve_group_labels(
            targ_mesh=targ,
            schaefer_resolution=str(self.config.num_clusters),
        )

        # Subgraph 3 — initialization prior (μ, ε). Consumes the avg
        # arrays from memory directly — no .npy re-read between
        # subgraphs 2 and 3.
        self._progress.emit_state("step1_ini", COHORT_SUB_ID, "running")
        try:
            group_mat_path, epsil = run_ini_params(
                project_dir=self.layout.project_dir,
                seed_mesh=seed,
                targ_mesh=targ,
                lh_labels=lh_labels,
                rh_labels=rh_labels,
                profile_dtype=np.dtype(k1.profile_dtype),
                reduction_dtype=np.dtype(k1.reduction_dtype),
                backend=backend,
                precomputed_lh_avg=avg_res.lh_avg,
                precomputed_rh_avg=avg_res.rh_avg,
            )
        except BaseException as e:
            self._progress.emit_state(
                "step1_ini", COHORT_SUB_ID, "failed",
                error=f"{type(e).__name__}: {e}")
            raise
        self._progress.emit_state("step1_ini", COHORT_SUB_ID, "done")

        # Subgraph 4 — spatial radius mask.
        self._progress.emit_state("step1_mask", COHORT_SUB_ID, "running")
        try:
            spatial_mask_path = run_radius_mask(
                project_dir=self.layout.project_dir,
                targ_mesh=targ,
                lh_labels=lh_labels,
                rh_labels=rh_labels,
                radius_mm=k1.radius_mask_radius_mm,
                backend=backend,
            )
        except BaseException as e:
            self._progress.emit_state(
                "step1_mask", COHORT_SUB_ID, "failed",
                error=f"{type(e).__name__}: {e}")
            raise
        self._progress.emit_state("step1_mask", COHORT_SUB_ID, "done")

        return {
            "profile_b2nd_paths": profile_b2nd_paths,
            "avg_profile_paths": (avg_res.lh_path, avg_res.rh_path),
            "group_mat_path": group_mat_path,
            "spatial_mask_path": spatial_mask_path,
            "epsil": epsil,
        }

    def _write_cohort(self, step1_outputs: Dict[str, Any]) -> Path:
        """Driver writes cohort.json directly (was step1 subgraph 5)."""
        from .cohort_writer import write_cohort_manifest
        return write_cohort_manifest(
            project_dir=self.layout.project_dir,
            subjects=self.inputs.subject_ids(),
            sessions=self.inputs.session_ids(),
            targ_mesh=self.inputs.targ_mesh,
            seed_mesh=self.inputs.seed_mesh,
            n_grad_components=self.config.n_grad_components,
            profile_b2nd_paths=step1_outputs["profile_b2nd_paths"],
            group_mat_path=step1_outputs["group_mat_path"],
            spatial_mask_path=step1_outputs["spatial_mask_path"],
            avg_profile_paths=step1_outputs["avg_profile_paths"],
        )

    def _run_step2_train_prior_maybe_tf32(self) -> Path:
        """Driver wrapper around step2 that toggles cuBLAS TF32 math
        mode for the duration of the step (and only this step) when
        ``config.step2.enable_tf32`` is True and step2 runs on GPU.

        Other steps see the prior math mode restored on exit (default
        strict fp32 unless ``NVIDIA_TF32_OVERRIDE=1`` is also set in
        the environment, in which case the prior mode IS the TF32 mode
        — restoring it is still a no-op).

        Mode A never reaches this method (step2 is not in its flow);
        Mode B's parser guarantees ``self.config.step2`` is non-None
        by the time the driver dispatches here.
        """
        assert self.config.step2 is not None, (
            "step2 block missing — driver invariant violated"
        )
        if (self.config.step2.enable_tf32
                and self.config.backend_step2 == "gpu"):
            from ._cublas_math_mode import cublas_tf32_scope
            with cublas_tf32_scope():
                return self._run_step2_train_prior()
        return self._run_step2_train_prior()

    def _run_step2_train_prior(self) -> Path:
        """Step 2: coupled EM across the cohort, produces Params_Final.mat
        at ``priors/<variant>/beta<X>/Params_Final.mat``."""
        from arealmshbm.step2_pipeline import Step2Config, Step2Pipeline

        # ``**k2_kwargs`` splats every Step2Knobs algorithm field into
        # Step2Config via the field-name 1:1 mapping pinned by
        # test_knobs_field_match.test_step2_knobs_field_names_match_step2config
        # — same pattern as step0. ``enable_tf32`` is the one exception:
        # it's consumed by the driver's tf32_scope wrapper (above), not
        # by Step2Config itself, so it's popped from the splat.
        #
        # k2 is guaranteed non-None here: this method is only invoked from
        # ``run()`` on the Mode B branch, and the parser requires the
        # step2 block under Mode B.
        k2 = self.config.step2
        assert k2 is not None, (
            "step2 block missing from PipelineConfig despite Mode B run "
            "— parser invariant violated"
        )
        k2_kwargs = asdict(k2)
        k2_kwargs.pop("enable_tf32")
        cfg = Step2Config(
            project_dir=self.layout.project_dir,
            num_sub=self.inputs.num_subjects,
            num_session=self.inputs.num_sessions,
            num_clusters=self.config.num_clusters,
            mode=self.config.variant,  # type: ignore[arg-type]
            beta_scalar=self.config.beta_scalar,
            mesh=self.inputs.targ_mesh,
            seed_mesh=self.inputs.seed_mesh,
            n_grad_components=self.config.n_grad_components,
            backend=self.config.backend_step2,
            **k2_kwargs,
        )
        with Step2Pipeline(cfg, progress=self._progress) as pipe:
            pipe.run()
        prior_dst = self.layout.prior_path(self.config.variant,
                                           self.config.beta_scalar,
                                           self._prior_includes_beta())
        if not prior_dst.exists():
            # Step2Pipeline.run() didn't raise but didn't write the
            # expected Params_Final.mat either (early exit, alt save
            # path, etc.). Surface immediately rather than letting
            # step3 fail with a confusing missing-prior error two
            # phases later. Emit ``failed`` on the step2 slot first —
            # Step2Pipeline.run() deliberately doesn't emit ``done``
            # so the frontend hasn't yet been told the slot finished;
            # this emit gives it the correct terminal state.
            self._progress.emit_state(
                "step2", COHORT_SUB_ID, "failed",
                error=f"FileNotFoundError: prior not written at {prior_dst}",
            )
            raise FileNotFoundError(
                f"Step2 finished without writing the expected prior at "
                f"{prior_dst}. Check Step2Pipeline.run() exit path."
            )
        # Artifact check passed — NOW the slot is genuinely done. The
        # ``done`` emit lives here (not inside Step2Pipeline.run) so
        # the frontend never sees ``done`` for a step2 run that
        # silently failed to write its prior.
        self._progress.emit_state("step2", COHORT_SUB_ID, "done")
        return prior_dst

    def _run_step3_all_subjects(
        self,
        gradient_by_sub: Dict[str, "np.ndarray"],
    ) -> None:
        """Step 3: per-subject parcellation EM. Each call writes
        ``ind_parcellation_<variant>/<T>_sess/beta<B>/<id>/Ind_parcellation_*.mat``.

        ``gradient_by_sub`` is the dict returned by step0 — each entry
        is the in-memory ``(N, n_grad_components)`` fp32 gradient_mat
        for one subject. fetch_data uses it directly instead of
        re-reading the cohort's gradient .npy files. After processing a
        subject we drop the reference so its ~33 MB (81924 × 100 × 4 B
        at fsa6 with n_grad_components=100) is reclaimable; this matters
        at large K.

        **Pipeline-parallel by phase (GPU backend, K ≥ 2).** The
        single-subject pipeline runs three phases sequentially (LOAD:
        disk reads + setup + H2D, EM: intra_em outer loop, SAVE: argmax
        + .mat write). On ``gpu_full`` the EM body (~2.4 s on the YS
        profile reference) gates the wall; LOAD (~0.9 s) and SAVE
        (~0.05 s) would otherwise serialize behind EM. We pipeline by
        phase across subjects: one stage thread per phase, with cupy
        streams on LOAD and EM (cross-stream event sync on the LOAD→EM
        device-buffer handoff), so subject K+1's LOAD overlaps subject
        K's EM and subject K-1's SAVE. See
        :mod:`arealmshbm.pipeline._step3_stage_pipeline`.

        CPU backend and single-subject runs stay on the serial
        ``Step3Pipeline.run_and_save()`` path — the stage pipeline has
        no benefit when there's nothing to overlap.
        """
        from arealmshbm.step3_pipeline import Step3Config, Step3Pipeline

        subjects = list(self.inputs.subjects)
        is_gpu = self.config.backend_step3 in ("gpu_elambda", "gpu_full")

        # ``**asdict(k3)`` splats every Step3Knobs field into Step3Config
        # via the field-name 1:1 mapping pinned by
        # test_knobs_field_match.test_step3_knobs_field_names_match_step3config
        # — same pattern as step0/step2. Adding a knob auto-flows here
        # without a manual edit.
        k3 = self.config.step3
        k3_kwargs = asdict(k3)
        configs: List[Step3Config] = [
            Step3Config(
                project_dir=self.layout.project_dir,
                num_session=self.inputs.num_sessions,
                num_clusters=self.config.num_clusters,
                subid=int(sub.id),
                mesh=self.inputs.targ_mesh,
                w=self.config.w,
                c=self.config.c,
                beta_scalar=self.config.beta_scalar,
                pipeline_type=self.config.variant,
                backend=self.config.backend_step3,
                n_grad_components=self.config.n_grad_components,
                out_subid=sub.id,
                **k3_kwargs,
            )
            for sub in subjects
        ]

        if is_gpu and len(subjects) >= 2:
            from ._step3_stage_pipeline import Step3StagePipeline
            # em_concurrency=1 by default — empirically 2-worker EM on
            # an RTX 5090 24 GB delivered +29 % step3 wall (worse than
            # single-EM) AND once triggered a WHEA hardware MCE under
            # the combined GPU + PCIe load. The cuBLAS / cupy
            # memory-pool / PCIe DMA paths all serialize on cross-
            # thread access, so two EM workers spend more time
            # contending than overlapping. Tuning ceiling left as an
            # explicit knob so future configurations (multi-GPU,
            # workstation-class PCIe, different cuBLAS revisions) can
            # opt in — see _step3_stage_pipeline.py docstring for the
            # measurement that motivated the default.
            coord = Step3StagePipeline(
                subject_ids=[sub.id for sub in subjects],
                configs=configs,
                gradients=gradient_by_sub,
                backend=self.config.backend_step3,
                em_concurrency=1,
                progress=self._progress,
            )
            coord.run()
            return

        # CPU backend or K=1: keep the original serial loop. The
        # stage pipeline's overhead (queues, threads, stream sync)
        # has no payoff to amortize without multiple subjects. The
        # serial path owns the running/done/failed emits directly —
        # the stage pipeline handles them internally on the GPU path.
        for sub_idx, sub in enumerate(subjects):
            grad_mat = gradient_by_sub.pop(sub.id, None)
            self._progress.emit_state("step3", sub.id, "running")
            try:
                with Step3Pipeline(
                    configs[sub_idx], precomputed_gradient_mat=grad_mat,
                ) as pipe:
                    pipe.run_and_save()
            except BaseException as e:
                self._progress.emit_state(
                    "step3", sub.id, "failed",
                    error=f"{type(e).__name__}: {e}")
                raise
            self._progress.emit_state("step3", sub.id, "done")

    # ─────────────────────────────────────────────────────────────────
    # Public entry
    # ─────────────────────────────────────────────────────────────────
    def run(self) -> PipelineRunResult:
        """Execute the full pipeline for the configured mode.

        Returns a :class:`PipelineRunResult` and also writes it to
        ``project_dir/logs/pipeline_run_<timestamp>.json``. On an
        exception mid-flight the log is still written via a finally
        clause (``success=False``, ``failed_at_step`` names the phase),
        then the exception is re-raised. K-1 successful subjects in a
        modeA_batch run keep their audit trail.
        """
        # UTC + millisecond suffix: avoids the local-time/DST ambiguity
        # the run-log filename would otherwise inherit, and the ms tail
        # disambiguates two runs starting in the same wall second.
        now = _dt.datetime.now(_dt.timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"
        t_total = time.perf_counter()
        timings: Dict[str, float] = {}
        cohort_path: Optional[Path] = None
        prior_dst: Optional[Path] = None
        current_phase: str = "init"
        failure: Optional[BaseException] = None

        try:
            # Emit the progress plan FIRST so the frontend has the full
            # ``(step, sub_id)`` roster before any state event arrives.
            # Mode A skips step2 — ``include_step2`` mirrors the dispatch
            # below. ``meta`` carries the run-level labels the frontend
            # would otherwise have to re-parse pipeline_config.json /
            # bold_inputs.json to learn. The plan emit also TRUNCATES
            # ``logs/progress.jsonl`` (see :meth:`ProgressEmitter.emit_plan`
            # docstring) so a fresh ``Pipeline.run()`` on a project
            # that's been run before starts with a clean file.
            #
            # Inside the try so the finally block (close + run-log
            # write) still runs if emit_plan raises ``ValueError`` on
            # bad meta payload — without this, a future change passing
            # an un-coerced ``np.int64 beta_scalar`` or ``Path`` would
            # skip the audit log and leak the open progress handle.
            current_phase = "emit_plan"
            subject_ids = [s.id for s in self.inputs.subjects]
            self._progress.emit_plan(
                build_plan_entries(
                    subject_ids,
                    include_step2=(not self.config.is_mode_a),
                ),
                meta={
                    "mode": self.config.mode,
                    "variant": self.config.variant,
                    "beta_scalar": self.config.beta_scalar,
                    "dataset_name": self.inputs.dataset_name,
                },
            )

            current_phase = "min_input_check"
            t = time.perf_counter()
            prior_src = self._check_min_inputs()
            timings["min_input_check"] = time.perf_counter() - t

            current_phase = "stage_bold_lists"
            t = time.perf_counter()
            self._stage_step1_bold_lists()
            timings["stage_bold_lists"] = time.perf_counter() - t

            # Step 0: gradients per subject. Returns the in-memory dict
            # ``{sub_id: gradient_mat}`` that step3 will consume directly,
            # skipping the disk re-read.
            current_phase = "step0"
            t = time.perf_counter()
            gradient_by_sub = self._run_step0_all_subjects_maybe_tf32()
            timings["step0"] = time.perf_counter() - t

            # Step 1: profiles + ini params + radius mask, driver-owned.
            current_phase = "step1"
            t = time.perf_counter()
            step1_outputs = self._run_step1_all_subjects()
            timings["step1"] = time.perf_counter() - t

            # Cohort.json: roster + artifact ledger that step2/step3 read
            # as their sole discovery mechanism.
            current_phase = "cohort_write"
            t = time.perf_counter()
            cohort_path = self._write_cohort(step1_outputs)
            timings["cohort_write"] = time.perf_counter() - t

            # Prior: Mode A reads it in place from the project slot
            # (validated to exist by the min-input check); Mode B trains
            # it via step2, writing the same slot.
            t = time.perf_counter()
            if self.config.is_mode_a:
                current_phase = "prior_check"
                assert prior_src is not None  # project-local prior_path
                prior_dst = prior_src
                timings["prior_check"] = time.perf_counter() - t
            else:
                current_phase = "step2"
                prior_dst = self._run_step2_train_prior_maybe_tf32()
                timings["step2"] = time.perf_counter() - t

            # Step 3: per-subject parcellation. Consumes step0's gradients
            # directly from memory — the cohort.json + .npy on disk stay
            # as cache for resume runs.
            current_phase = "step3"
            t = time.perf_counter()
            self._run_step3_all_subjects(gradient_by_sub)
            timings["step3"] = time.perf_counter() - t

            current_phase = "done"
        except BaseException as e:
            failure = e
            raise
        finally:
            timings["total"] = time.perf_counter() - t_total
            result = PipelineRunResult(
                project_dir=str(self.layout.project_dir),
                mode=self.config.mode,
                variant=self.config.variant,
                timestamp=timestamp,
                success=(failure is None),
                failed_at_step=(None if failure is None else current_phase),
                error=(None if failure is None else f"{type(failure).__name__}: {failure}"),
                timings=timings,
                step0_subjects=[s.id for s in self.inputs.subjects],
                step1_subjects=[s.id for s in self.inputs.subjects],
                step3_subjects=[s.id for s in self.inputs.subjects],
                prior_path=(str(prior_dst) if prior_dst is not None else None),
                cohort_json_path=(str(cohort_path) if cohort_path is not None else None),
            )
            try:
                self._write_run_log(result)
            except Exception as log_err:
                # Never let a log-write failure mask the real exception.
                import sys as _sys
                print(
                    f"WARNING: pipeline run log write failed: {log_err}",
                    file=_sys.stderr,
                )
            # Flush + close the progress sink. ``close`` itself is
            # exception-safe (swallows on failure) so it cannot mask
            # the original exception.
            self._progress.close()
        return result

    def _write_run_log(self, result: PipelineRunResult) -> None:
        self.layout.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.layout.run_log_path(result.timestamp)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            f.write("\n")
