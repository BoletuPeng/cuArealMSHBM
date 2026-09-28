"""_step3_stage_pipeline.py — 3-stage pipeline-parallel coordinator for step3.

Background: per-subject ``Step3Pipeline`` runs three phases:

    load_inputs()  → build_session()  → run()  → save()
       (disk +         (H2D mirrors      (intra_em      (.mat
        .mat parse)     into GPU)         loop on GPU)   write)

``load_inputs + build_session`` (host reads + the BOLD / gradient H2D)
and ``save`` (argmax + .mat write) are off-GPU work that a sequential
subject loop would serialize behind the previous subject's EM.

This module pipelines step3 by phase, mirroring
``_step0_stage_pipeline.py`` (4 stages by subgraph) and
``_step1_stage_pipeline.py`` (4 stages by phase). Three stages, one
worker thread each, LOAD and EM on independent cupy streams::

    stage LOAD  →  qLE  →  stage EM  →  qES  →  stage SAVE
    [stream L]            [stream E]            (CPU only)

At steady state, while stage EM crunches subject K's intra_em loop,
the LOAD thread is reading subject K+1's BOLD + building its session
(H2D on stream L) and stage SAVE is writing subject K-1's parcellation
.mat, so the per-subject wall is ``max(t_LOAD, t_EM, t_SAVE)`` instead
of their sum. Stage LOAD reads the cohort constants (group prior,
spatial masks, candidate layout) once before the subject loop and
shares that ``Step3SparseCohort`` read-only with every subject, so each
subject's LOAD is only its BOLD + gradient. The chain is FIFO
end-to-end: ``.mat`` files are written in subject order. Measured walls
are in ``docs/step3_flow_and_subgraphs.md``.

Cross-stream device handoff
---------------------------
Stage LOAD's ``build_session`` H2D-copies the subject's packed BOLD,
gradient and candidate-set buffers into device buffers held by the
session. Those copies queue on ``stream L``; stage EM consumes them via
``pipe.run()``, which queues compute on ``stream E``. Without a sync
edge, EM might dispatch its first kernel before the H2D writes have
landed.

We record a ``cupy.Event`` on stream L at the end of LOAD and have
stage EM ``wait_event`` on stream E before invoking ``pipe.run``.
Cheap on the GPU (stream-level dependency edge — no host sync), and
the next subject's LOAD keeps H2D-ing while EM computes. The session's
per-stage timing syncs use ``cp.cuda.get_current_stream()``, i.e. the
stream set by the surrounding ``with stream_E:`` context.

Cupy memory-pool teardown
-------------------------
Stage SAVE calls ``pipe.close()`` once the .mat is written; that runs
``cp.get_default_memory_pool().free_all_blocks()``, reclaiming the
subject's EM buffers. ``free_all_blocks`` only releases *unused*
blocks, so the live device buffers of the subjects in stage LOAD / EM
are unaffected. The pool itself is thread-safe (cupy guards it with an
internal lock).

Memory budget
-------------
Queues capped at ``maxsize=2`` (matches step0). The structural upper
bound on subjects in flight is ``1 + 2 + 1 + 2 + 1 = 7`` (1 LOAD
active + 2 qLE + 1 EM active + 2 qES + 1 SAVE). Hitting that ceiling
requires SAVE to stall (e.g. the output disk backpressures) so the qES
backlog grows; in healthy steady state SAVE drains faster than EM, so
qES sits at 0 and qLE at ~1. If you push SAVE-side IO with slow disks,
watch the pool size and lower ``maxsize`` accordingly.

Per-stage timing fuzziness
--------------------------
``per_stage_em`` (m_step, e_step_lambda_loop, …) wall numbers come
from the in-Session timing block. Under the stage pipeline the syncs
land on a non-null stream while stream L's H2D runs concurrently, so
they include some contention. Stage totals (``em_total``,
``load_total``) are timed by the coordinator from the outside and
remain accurate. This matches step0's stage-pipeline trade-off; the
algorithmic correctness is unaffected (intra-stream ops still execute
in queue order).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from arealmshbm.step3_pipeline import (
    Step3Config, Step3Pipeline, Step3Result,
)

from ._progress import ProgressEmitter


# Sentinel pushed through each queue to signal end-of-stream.
_SENTINEL = "__step3_pipeline_sentinel__"


@dataclass
class _StageWork:
    """One subject's transit unit through the 3-stage pipeline.

    Each stage reads its inputs from prior-stage outputs stored on
    this object and writes its own outputs back here before pushing
    to the next queue. The ``pipe`` reference is kept alive from
    stage LOAD through stage SAVE so all stages see the same
    ``Step3Pipeline`` instance (its cached ``_inputs`` and
    ``_session`` attributes are populated in LOAD and consumed in
    EM / SAVE).
    """
    sub_id: str
    cfg: Step3Config
    grad_mat: Optional[np.ndarray]
    pipe: Optional[Step3Pipeline] = None
    # Cupy event recorded at the end of LOAD on stream L; EM waits on
    # it on stream E before running. ``None`` when LOAD failed first.
    load_event: Optional[Any] = None
    result: Optional[Step3Result] = None
    failure: Optional[BaseException] = None


class Step3StagePipeline:
    """Stage-pipelined step3 executor for ``backend='gpu'``.

    Usage::

        coord = Step3StagePipeline(
            subject_ids=[...],          # subject ids, same order as configs
            configs=[Step3Config, ...], # one per subject, backend='gpu'
            gradients={sub_id: NxDg, ...},
        )
        coord.run()

    Subjects are processed in input order; each subject's pipeline is
    constructed inside stage LOAD (so the BOLD H2D happens on stream L,
    not on whatever stream the caller was using).
    """

    def __init__(
        self, *,
        subject_ids: List[str],
        configs: List[Step3Config],
        gradients: Dict[str, np.ndarray],
        progress: Optional[ProgressEmitter] = None,
    ):
        if len(subject_ids) != len(configs):
            raise ValueError(
                f"Step3StagePipeline: subject_ids ({len(subject_ids)}) "
                f"and configs ({len(configs)}) length mismatch"
            )
        if not subject_ids:
            raise ValueError("Step3StagePipeline: empty subject list")
        self.subject_ids = subject_ids
        self.configs = configs
        # Per-subject gradient .npy ownership is transferred to this
        # coordinator; we pop entries as we dispatch to free memory
        # after each subject's LOAD.
        self.gradients = gradients
        # Optional frontend progress sink. Per-subject ``step3``
        # running fires from _stage_LOAD on subject entry; done/failed
        # fires from _stage_SAVE before qDone.put. ``None`` →
        # standalone tests; the driver always passes a real emitter.
        # Storing a disabled emitter rather than ``None`` lets every
        # emit call site stay branchless (no per-call None check).
        self.progress = (
            progress if progress is not None
            else ProgressEmitter.disabled()
        )

    def run(self) -> None:
        """Drive all subjects through LOAD → EM → SAVE.

        Returns once every subject has emitted on ``qDone`` (success
        or failure). Re-raises the first failure after the threads
        join. No return value — step3's only side effect is the
        per-subject ``Ind_parcellation_*.mat`` written by stage SAVE.
        """
        import cupy as cp

        # Backpressure: each inter-stage queue holds at most 2
        # subjects, so a fast upstream stage can't pile up unbounded
        # in-flight buffers.
        qLE: queue.Queue = queue.Queue(maxsize=2)
        qES: queue.Queue = queue.Queue(maxsize=2)
        qDone: queue.Queue = queue.Queue()

        # Per-stage cupy streams. ``non_blocking=True`` so they don't
        # serialize against the legacy null stream — required for the
        # H2D on stream L to actually overlap with EM compute on
        # stream E. Stage SAVE is CPU only — no stream needed.
        stream_L = cp.cuda.Stream(non_blocking=True)
        stream_E = cp.cuda.Stream(non_blocking=True)

        threads: List[threading.Thread] = [
            threading.Thread(
                target=self._stage_LOAD, args=(qLE, stream_L),
                name="step3-stageLOAD", daemon=True),
            threading.Thread(
                target=self._stage_EM, args=(qLE, qES, stream_E),
                name="step3-stageEM", daemon=True),
            threading.Thread(
                target=self._stage_SAVE, args=(qES, qDone),
                name="step3-stageSAVE", daemon=True),
        ]
        for t in threads:
            t.start()

        # Drain qDone in subject-id order matters only for failure
        # reporting; the .mat writes have already happened in stage
        # SAVE by the time we get here.
        failures: List[BaseException] = []
        for _ in range(len(self.subject_ids)):
            work: _StageWork = qDone.get()
            if work.failure is not None:
                failures.append(work.failure)

        for t in threads:
            t.join()

        if failures:
            # Re-raise the first failure here so the caller's stack
            # gets the exception object. Every stage thread already
            # printed its own traceback to stderr at failure capture
            # time (see traceback.print_exc calls in the stage
            # methods) — re-raising the first one preserves the
            # primary failure semantics; per-subject details are
            # already in the captured stderr.
            raise failures[0]

        # Belt-and-braces: drain the cupy pool one last time. Each
        # stage-SAVE close() already did this per subject; this catches
        # any residual blocks the driver may have allocated upstream
        # before handing off to us.
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # Stage LOAD — disk reads, .mat parse, fetch, build_session.
    # All H2D for the subject's Session runs on stream L. Pushes a
    # sentinel at the end so stage EM returns.
    # ─────────────────────────────────────────────────────────────────
    def _stage_LOAD(self, qLE: queue.Queue, stream) -> None:
        import cupy as cp

        # The group prior, the spatial masks and the candidate layout
        # are cohort constants (~110 ms of reads per subject on this
        # stage, which gates it). Load them once here and share them
        # read-only; every subject's ``Step3SparseCohort.check(cfg)``
        # still guards a heterogeneous config list. A failure here is
        # attached to every subject so each one still reaches qDone and
        # ``run()`` re-raises instead of hanging.
        cohort = None
        cohort_failure: Optional[BaseException] = None
        try:
            from arealmshbm.step3_pipeline.sparse_inputs import (
                load_step3_sparse_cohort,
            )
            cohort = load_step3_sparse_cohort(self.configs[0])
        except BaseException as e:
            cohort_failure = e
            traceback.print_exc()
        try:
            for sub_idx, sub_id in enumerate(self.subject_ids):
                cfg = self.configs[sub_idx]
                grad = self.gradients.pop(sub_id, None)
                work = _StageWork(
                    sub_id=sub_id, cfg=cfg, grad_mat=grad,
                )
                # Subject enters the pipeline at LOAD — emit before
                # any disk I/O so the frontend shows "now working on
                # X" the instant the worker picks the subject up.
                self.progress.emit_state("step3", sub_id, "running")
                if cohort_failure is not None:
                    work.failure = cohort_failure
                    qLE.put(work)
                    continue
                try:
                    pipe = Step3Pipeline(
                        cfg, precomputed_gradient_mat=grad,
                        sparse_cohort=cohort,
                    )
                    # Stash the pipe handle on the work item BEFORE
                    # the H2D / setup work below, so that if
                    # build_session() raises mid-construction the
                    # SAVE stage can still call pipe.close() on the
                    # partial Session — without this, any device
                    # buffers H2D'd before the failure would only be
                    # reclaimed when the local ``pipe`` reference
                    # falls out of scope on the next loop iter
                    # (no free_all_blocks(), pool stays sized).
                    work.pipe = pipe
                    with stream:
                        # load_inputs() is host-side (.mat / .b2nd
                        # reads). The H2D happens inside
                        # build_session() — the ``cp.asarray`` calls
                        # there queue on stream L because of the
                        # surrounding context.
                        inputs = pipe.load_inputs()
                        pipe.build_session(inputs)
                        # Records every op queued on stream L so far,
                        # including all the build_session H2D copies.
                        # Stage EM waits on the event from stream E
                        # before computing.
                        work.load_event = cp.cuda.Event(
                            block=False, disable_timing=True)
                        work.load_event.record()
                except BaseException as e:
                    work.failure = e
                    traceback.print_exc()
                qLE.put(work)
        finally:
            qLE.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage EM — intra_em outer loop on stream E. The gating stage.
    # ─────────────────────────────────────────────────────────────────
    def _stage_EM(self, qLE: queue.Queue, qES: queue.Queue, stream) -> None:
        import cupy as cp

        while True:
            work = qLE.get()
            if work is _SENTINEL:
                qES.put(_SENTINEL)
                return
            if work.failure is None:
                try:
                    with stream:
                        # Hold stream E until stream L's H2D writes
                        # are visible. Cheap (stream-level dependency
                        # edge).
                        cp.cuda.get_current_stream().wait_event(
                            work.load_event)
                        # pipe.run() finds _inputs and _session
                        # already cached from stage LOAD, so it
                        # skips straight into the intra_em loop.
                        work.result = work.pipe.run()
                except BaseException as e:
                    work.failure = e
                    traceback.print_exc()
            qES.put(work)

    # ─────────────────────────────────────────────────────────────────
    # Stage SAVE — argmax + .mat write + close (pool free).
    # Drains until stage EM forwards the sentinel.
    # ─────────────────────────────────────────────────────────────────
    def _stage_SAVE(self, qES: queue.Queue, qDone: queue.Queue) -> None:
        while True:
            work = qES.get()
            if work is _SENTINEL:
                return
            if work.failure is None:
                try:
                    work.pipe.save(work.result)
                    # close() drops _inputs / _session refs and calls
                    # free_all_blocks on the cupy pool. Only the
                    # current subject's *unused* blocks are reclaimed
                    # — concurrent LOAD / EM subjects' live device
                    # arrays stay pinned.
                    work.pipe.close()
                except BaseException as e:
                    work.failure = e
                    traceback.print_exc()
            else:
                # Failure path: close the pipe if LOAD got far enough
                # to construct one (work.pipe is set as soon as
                # Step3Pipeline(...) returns). Otherwise its partial
                # device buffers stay in the cupy pool until the
                # local ``pipe`` ref in stage LOAD is GC'd — and
                # free_all_blocks never runs, so the pool doesn't
                # shrink. Swallow secondary errors; the primary
                # failure is already captured in work.failure.
                if work.pipe is not None:
                    try:
                        work.pipe.close()
                    except Exception:
                        pass
            # Terminal stage owns the transition out of ``running``.
            # work.failure may have been set in LOAD, EM, or SAVE
            # itself; this single emit point captures all paths.
            if work.failure is None:
                self.progress.emit_state("step3", work.sub_id, "done")
            else:
                self.progress.emit_state(
                    "step3", work.sub_id, "failed",
                    error=f"{type(work.failure).__name__}: {work.failure}",
                )
            qDone.put(work)
