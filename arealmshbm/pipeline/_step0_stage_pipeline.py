"""_step0_stage_pipeline.py — 4-stage pipeline-parallel coordinator for step0.

Background: the per-subject Step0Pipeline alternates GPU subgraphs (A: RSFC
gradients, B: graph distance, C: diffusion embedding / eigsh, D: CPU
upsample). Running many subjects concurrently as whole pipelines
(``ThreadPoolExecutor(max_workers=N)``) hits a wall — cuBLAS / cuSPARSE
/ cuSOLVER handles are per-device singletons, so when multiple workers
enter the same subgraph (e.g. all do eigsh in subgraph C) they
serialize behind the handle mutex and the GPU drops to ~⅓ utilization
while three workers wait their turn.

This module pipelines by subgraph instead of by subject. Four stage
threads, each handling its own subgraph for the full cohort:

    stage A  →  qAB  →  stage B  →  qBC  →  stage C  →  qCD  →  stage D
    [stream A]          [stream B]          [stream C]          [CPU only]
    (sub 0,1,2,…)        (sub 0,1,2,…)        (sub 0,1,2,…)         (sub 0,1,2,…)

At steady state, while stage A computes subgraph A for subject K,
stage B is doing graph distance for subject K-1, stage C is running
eigsh for subject K-2, and stage D is interpolating subject K-3.
Each stage owns its handle queue — cuSOLVER sees exactly one eigsh in
flight at any moment, cuBLAS sees stage A's sgemms, and they run on
independent streams so the GPU stays continuously busy across
subgraph mix.

The steady-state per-subject wall is ``max(t_A, t_B, t_C, t_D)``
instead of ``sum``. On the YS cohort that drops the per-subject cost
from ~5 s (sum) to ~2.7 s (max, gated by subgraph A).

Cross-stage device arrays
-------------------------
Subgraph A → B passes ``edge_density`` (host fp32, ~300 KB). Trivial.

Subgraph B → C passes ``lh_dist``, ``rh_dist`` — on GPU these are
``(N_down, N_down) fp32`` cupy arrays, ~670 MB each. Keeping them
device-resident saves a 2.7 GB / subject PCIe round trip. Cross-
stream ordering is enforced by recording a ``cp.cuda.Event`` after
stage B's writes and waiting for it on stage C's stream.

Subgraph C → D passes the embeddings (host arrays, small).

Cupy memory pool teardown
-------------------------
``Step0Pipeline.close()`` runs in stage D once the subject's emb is
host-side; it calls ``free_all_blocks()``, which reclaims only the
*unused* blocks in the pool (the per-pipe Step0Inputs GPU mirrors of
the about-to-be-discarded subject), so it cannot disturb live arrays
held by other stages' subjects.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from arealmshbm.step0_pipeline import Step0Config, Step0Pipeline
from arealmshbm.step0_pipeline.pipeline import (
    Step0Result, _bold_session_getter, _new_per_stage_dict,
)

from ._progress import ProgressEmitter
from ._step0_bold_prefetcher import Step0BoldPrefetcher
from .config import Step0Knobs


@dataclass
class _StageWork:
    """One subject's transit unit through the 4-stage pipeline.

    Each stage reads its inputs from the prior stage's outputs (stored
    on this object) and writes its own outputs back here before
    pushing to the next queue. The ``pipe`` reference is kept alive
    from stage A through stage D so all stages see the same
    Step0Inputs (fresh GPU mirrors per subject).
    """
    sub_idx: int
    sub_id: str
    cfg: Optional[Step0Config] = None
    pipe: Optional[Step0Pipeline] = None
    per_stage: Dict[str, float] = field(default_factory=_new_per_stage_dict)
    a_out: Optional[Dict[str, Any]] = None
    b_out: Optional[Dict[str, Any]] = None
    # Cupy event recorded at end of stage B; stage C waits on it
    # before consuming lh_dist / rh_dist on its own stream.
    b_event: Optional[Any] = None
    c_out: Optional[Dict[str, Any]] = None
    grad_mat: Optional[np.ndarray] = None
    failure: Optional[BaseException] = None


# Sentinel pushed through each queue to signal end-of-stream.
_SENTINEL = "__step0_pipeline_sentinel__"


class Step0StagePipeline:
    """Stage-pipelined step0 executor.

    Usage::

        coord = Step0StagePipeline(
            project_dir=..., sess_list=..., targ_mesh=...,
            n_grad_components=..., backend="gpu",
            prefetcher=prefetcher,
            sess_paths_per_sub=...,
            lookahead=...,
        )
        gradients = coord.run(subjects)
    """

    def __init__(
        self, *,
        project_dir: Path,
        sess_list: Tuple[str, ...],
        targ_mesh: str,
        n_grad_components: int,
        backend: str,
        prefetcher: Step0BoldPrefetcher,
        sess_paths_per_sub: List[List[Tuple[Path, Path]]],
        lookahead: int,
        step0_knobs: Step0Knobs,
        progress: Optional[ProgressEmitter] = None,
    ):
        self.project_dir = project_dir
        self.sess_list = sess_list
        self.targ_mesh = targ_mesh
        self.n_grad_components = n_grad_components
        self.backend = backend
        self.is_gpu = backend == "gpu"
        self.prefetcher = prefetcher
        self.sess_paths_per_sub = sess_paths_per_sub
        self.lookahead = lookahead
        # ``step0_knobs`` is the :class:`pipeline.config.Step0Knobs`
        # bundle the driver threads into each subject's
        # :class:`Step0Config`. Required — the driver always has a
        # parsed PipelineConfig.step0 to pass.
        self.step0_knobs = step0_knobs
        # Optional frontend progress sink. Per-subject ``step0``
        # running/done/failed events fire at queue seams in _stage_A
        # and _stage_D — see those methods. ``None`` → standalone
        # tests that don't care about progress (the driver always
        # passes a real emitter). Storing a disabled emitter rather
        # than ``None`` lets every emit call site stay branchless
        # (no per-call None check).
        self.progress = (
            progress if progress is not None
            else ProgressEmitter.disabled()
        )

    def run(self, subjects: List[Any]) -> Dict[str, np.ndarray]:
        # Backpressure: each queue holds at most 2 subjects so a
        # fast upstream stage can't pile up unbounded buffers.
        # Stage A → B: edge_density (small), 2 in flight is fine
        # Stage B → C: lh_dist/rh_dist on device (1.34 GB / subject);
        #              cap at 2 so peak device buffer is ~2.7 GB
        # Stage C → D: embeddings (small)
        qAB: queue.Queue = queue.Queue(maxsize=2)
        qBC: queue.Queue = queue.Queue(maxsize=2)
        qCD: queue.Queue = queue.Queue(maxsize=2)
        qDone: queue.Queue = queue.Queue()

        # Per-stage cupy streams (non_blocking=True so they don't
        # serialize against the legacy null stream). Stage D is CPU
        # only — no stream needed.
        if self.is_gpu:
            import cupy as cp
            stream_A = cp.cuda.Stream(non_blocking=True)
            stream_B = cp.cuda.Stream(non_blocking=True)
            stream_C = cp.cuda.Stream(non_blocking=True)
        else:
            stream_A = stream_B = stream_C = None

        # Sentinel-flag for stage D termination (we set after stage C
        # signals end-of-stream + we've drained any in-flight items).
        threads = [
            threading.Thread(
                target=self._stage_A, args=(subjects, qAB, stream_A),
                name="step0-stageA", daemon=True),
            threading.Thread(
                target=self._stage_B, args=(qAB, qBC, stream_B),
                name="step0-stageB", daemon=True),
            threading.Thread(
                target=self._stage_C, args=(qBC, qCD, stream_C),
                name="step0-stageC", daemon=True),
            threading.Thread(
                target=self._stage_D, args=(qCD, qDone),
                name="step0-stageD", daemon=True),
        ]
        for t in threads:
            t.start()

        # Collect from qDone in subject-id order (insertion order of
        # gradients dict). Each subject ends up here exactly once
        # (success or failure).
        gradients: Dict[str, np.ndarray] = {}
        failures: List[BaseException] = []
        for _ in range(len(subjects)):
            work: _StageWork = qDone.get()
            if work.failure is not None:
                failures.append(work.failure)
                continue
            gradients[work.sub_id] = work.grad_mat

        for t in threads:
            t.join()

        if failures:
            # Re-raise the first failure; later ones get lost but the
            # stage's stderr printed them.
            raise failures[0]

        # Belt-and-braces: drain the cupy pool one last time. Per-
        # subject stage-D close() already did this 40×, but the
        # seed inputs from the driver still pin a few MB.
        if self.is_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

        return gradients

    # ─────────────────────────────────────────────────────────────────
    # Stage A — RSFC gradients (subgraph A)
    # ─────────────────────────────────────────────────────────────────
    def _stage_A(self, subjects, qAB: queue.Queue, stream) -> None:
        ctx = stream if stream is not None else nullcontext()
        # ``asdict(self.step0_knobs)`` splats every Step0Knobs algorithm
        # field into Step0Config via the field-name 1:1 mapping pinned
        # by test_knobs_field_match. ``enable_tf32`` is the one
        # exception: it's consumed by the driver's tf32_scope wrapper
        # higher up the call stack, not by Step0Config itself, so it's
        # popped from the splat. Computed once at method entry rather
        # than per-subject — ``self.step0_knobs`` is bound at __init__
        # time and does not vary across subjects, so this saves one
        # dataclasses.fields walk per subject (~11 primitives, modest).
        extra_knob_kwargs = asdict(self.step0_knobs)
        extra_knob_kwargs.pop("enable_tf32")
        try:
            for sub_idx, sub in enumerate(subjects):
                cfg = Step0Config(
                    project_dir=self.project_dir,
                    sub_id=sub.id,
                    sess_list=self.sess_list,
                    out_subid=sub.id,
                    mesh=self.targ_mesh,
                    num_components=self.n_grad_components,
                    backend=self.backend,
                    bold_paths_override=tuple(
                        self.sess_paths_per_sub[sub_idx]),
                    bold_provider=self.prefetcher.get_provider(sub.id),
                    **extra_knob_kwargs,
                )
                work = _StageWork(
                    sub_idx=sub_idx, sub_id=sub.id, cfg=cfg)
                # Subject enters the pipeline here — stage A is the
                # first to touch it. Emitting at the head of this
                # iteration tags ``running`` slightly before the
                # subgraph-A compute starts; that's the user-facing
                # "now working on subject X" moment.
                self.progress.emit_state("step0", sub.id, "running")
                try:
                    pipe = Step0Pipeline(cfg)
                    ti = pipe.load_inputs()
                    work.pipe = pipe
                    with ctx:
                        with _bold_session_getter(cfg, ti) as get_bold:
                            work.a_out = pipe._subgraph_A(
                                ti, get_bold, work.per_stage)
                except BaseException as e:
                    work.failure = e

                # Slide the prefetcher window now that this subject's
                # BOLD has been fully consumed. Thread-safe via the
                # prefetcher's internal lock.
                next_to_prime = sub_idx + self.lookahead
                if next_to_prime < len(subjects):
                    self.prefetcher.prime_subject(
                        subjects[next_to_prime].id,
                        self.sess_paths_per_sub[next_to_prime],
                    )

                qAB.put(work)
        finally:
            qAB.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage B — graph distance (subgraph B)
    # ─────────────────────────────────────────────────────────────────
    def _stage_B(self, qAB: queue.Queue, qBC: queue.Queue, stream) -> None:
        ctx = stream if stream is not None else nullcontext()
        try:
            while True:
                work = qAB.get()
                if work is _SENTINEL:
                    return
                if work.failure is None:
                    try:
                        ti = work.pipe._inputs
                        with ctx:
                            work.b_out = work.pipe._subgraph_B(
                                ti, work.a_out["edge_density"],
                                work.per_stage,
                            )
                            if self.is_gpu:
                                import cupy as cp
                                # Event records the completion of every
                                # op queued on this stream so far,
                                # including the gradient-distance
                                # writes into lh_dist / rh_dist.
                                work.b_event = cp.cuda.Event(
                                    block=False, disable_timing=True)
                                work.b_event.record()
                    except BaseException as e:
                        work.failure = e
                qBC.put(work)
        finally:
            qBC.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage C — diffusion embedding / eigsh (subgraph C)
    # ─────────────────────────────────────────────────────────────────
    def _stage_C(self, qBC: queue.Queue, qCD: queue.Queue, stream) -> None:
        ctx = stream if stream is not None else nullcontext()
        try:
            while True:
                work = qBC.get()
                if work is _SENTINEL:
                    return
                if work.failure is None:
                    try:
                        with ctx:
                            if work.b_event is not None and self.is_gpu:
                                # Hold stage C's stream until stage
                                # B's writes are visible. Cheap on the
                                # GPU side (just a stream-level
                                # dependency edge).
                                import cupy as cp
                                cp.cuda.get_current_stream().wait_event(
                                    work.b_event)
                            work.c_out = work.pipe._subgraph_C(
                                work.b_out["lh_dist"],
                                work.b_out["rh_dist"],
                                work.per_stage,
                            )
                        # Free the device-resident lh_dist/rh_dist refs
                        # in b_out now that subgraph C has consumed
                        # them and emitted host copies in c_out.
                        # Otherwise they'd live until the work item is
                        # dequeued in stage D, doubling the device
                        # footprint while stage C is mid-eigsh.
                        work.b_out["lh_dist"] = None
                        work.b_out["rh_dist"] = None
                    except BaseException as e:
                        work.failure = e
                qCD.put(work)
        finally:
            qCD.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage D — upsample emb + save + close pipe (subgraph D)
    # ─────────────────────────────────────────────────────────────────
    def _stage_D(self, qCD: queue.Queue, qDone: queue.Queue) -> None:
        while True:
            work = qCD.get()
            if work is _SENTINEL:
                return
            if work.failure is None:
                try:
                    ti = work.pipe._inputs
                    d_out = work.pipe._subgraph_D(
                        ti,
                        work.c_out["lh_emb_down"],
                        work.c_out["rh_emb_down"],
                        work.b_out["down_v"],
                        work.b_out["down_f"],
                        work.b_out["down_vf"],
                        work.per_stage,
                    )
                    # Assemble the (N_full, n_components) gradient
                    # matrix the driver returns to step3.
                    work.grad_mat = np.ascontiguousarray(
                        np.concatenate(
                            [d_out["lh_emb_up"], d_out["rh_emb_up"]],
                            axis=0),
                        dtype=np.float32,
                    )

                    # Persist artifacts if requested. We reconstruct a
                    # ``Step0Result`` ad-hoc so Step0Pipeline.save()
                    # keeps its existing contract.
                    if work.cfg.save_artifacts:
                        result = Step0Result(
                            edge_density=work.a_out["edge_density"],
                            lh_dist=work.c_out["lh_dist_h"],
                            rh_dist=work.c_out["rh_dist_h"],
                            lh_emb_down=work.c_out["lh_emb_down"],
                            rh_emb_down=work.c_out["rh_emb_down"],
                            lh_emb_up=d_out["lh_emb_up"],
                            rh_emb_up=d_out["rh_emb_up"],
                            timings={
                                "load_inputs": work.pipe.timings.get(
                                    "load_inputs", 0.0),
                                "per_stage": work.per_stage,
                            },
                        )
                        work.pipe.save(result)

                    # Close the pipe: drops _inputs reference and
                    # calls free_all_blocks. ``free_all_blocks`` only
                    # reclaims unused blocks, so other concurrent
                    # stages' live cupy arrays are unaffected.
                    work.pipe.close()
                except BaseException as e:
                    work.failure = e
            # Terminal stage owns the transition out of ``running``.
            # work.failure may have been set in any prior stage; this
            # single emit point captures both happy + sad paths.
            if work.failure is None:
                self.progress.emit_state("step0", work.sub_id, "done")
            else:
                self.progress.emit_state(
                    "step0", work.sub_id, "failed",
                    error=f"{type(work.failure).__name__}: {work.failure}",
                )
            qDone.put(work)
