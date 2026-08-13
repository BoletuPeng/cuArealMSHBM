"""_step3_stage_pipeline.py — 3-stage pipeline-parallel coordinator for step3.

Background: per-subject ``Step3Pipeline`` runs three phases:

    load_inputs()  → build_session()  → run()  → save()
       (disk +         (H2D mirrors      (intra_em      (.mat
        .mat parse)     into GPU)         loop on GPU)   write)

On ``gpu_full`` the EM body (~2.4 s on the YS profile reference) gates
the per-subject wall; ``load_inputs + build_session`` (~0.9 s total —
``fetch_data`` BOLD H2D + numba JIT for setup leaves) and ``save``
(~50 ms — argmax + .mat write) are off-GPU work that the wall would
otherwise serialize behind EM. Running subjects sequentially through
``Pipeline._run_step3_all_subjects`` paid ~3.33 s / subject ≈ EM +
0.93 s of host work that could have overlapped with the previous
subject's GPU compute.

This module pipelines step3 by phase, mirroring
``_step0_stage_pipeline.py`` (4 stages by subgraph) and
``_step1_stage_pipeline.py`` (4 stages by phase). Three stage *tiers*,
each with ≥1 worker thread, on independent cupy streams::

    stage LOAD  →  qLE  →  stage EM (×em_concurrency)  →  qES  →  stage SAVE
    [stream L]            [stream E_0, stream E_1, …]            (CPU only)
     (sub 0,1,2…)          (workers pull from qLE)                (sub 0,1,2…)

At steady state, while one EM worker crunches subject K's intra_em
loop, the LOAD thread is reading subject K+M's BOLD + building its
``VmfClusteringSession`` (H2D on stream L), other EM workers are
running subjects K-1, K-2, … on their own streams, and stage SAVE is
writing subject K-M-1's parcellation .mat.

EM concurrency
--------------
``em_concurrency`` (default 1) controls how many EM workers run in
parallel. With concurrency 1 the coordinator runs the original 1-EM
design (per-subject wall = max(load, em, save)) and is the proven
fast path. The ``em_concurrency ≥ 2`` opt-in exists as a tuning knob
for future hardware configurations, but on the development reference
machine (RTX 5090 Laptop 24 GB, CUDA 13.1) it consistently regressed:

  serial (no stage pipeline)       129.35 s   (3.23 s/sub)
  em_concurrency=1 (this default)  108.30 s   (2.71 s/sub, -16 %)
  em_concurrency=2 (opt-in)        139.82 s   (3.50 s/sub, +29 %)

The +29 % regression at concurrency 2 has three compounding causes:

* **cuBLAS handle serialization** — cupy creates a per-thread
  cublasHandle, but the underlying cuBLAS context still funnels
  sgemms through a small set of internal queues; two threads both
  dispatching ``X @ s_t_nu`` simultaneously end up trading locks
  rather than computing.
* **cupy memory-pool mutex** — every ``cp.empty / cp.asarray`` enters
  the default pool's lock. EM allocates / releases scratch (N, L)
  buffers every comp_iter; with two workers the lock becomes a
  hotspot.
* **PCIe DMA saturation** — two workers' parallel H2D for the next
  comp_iter's scv / sxv copies share one PCIe lane group; the link
  saturates and per-copy latency rises.

Beyond the perf regression, the same configuration once tripped a
Windows ``BugCheck 0x00020001`` (WHEA_UNCORRECTABLE_ERROR) on the
reference machine — a hardware-level Machine Check Exception, not a
driver fault, but the sustained dual-stream + dual-thread load on
PCIe / DRAM / CPU was enough to expose it. Default 1 is therefore
the conservative ship setting; raise via the driver only on a
machine you've characterised under sustained mixed CPU+GPU load.

Cross-stream device handoff
---------------------------
Stage LOAD's ``build_session`` performs ~1 GB of H2D copies (BOLD,
boundary_mask, neighborhoods, theta, sphere, gradient) into device
buffers held by the ``VmfClusteringSession``. Those copies queue on
``stream L``. Each EM worker consumes them via ``sess.run()`` which
queues compute on its own ``stream E_k``. Without a sync edge, an EM
worker might dispatch the first cuBLAS sgemm before the H2D writes
have landed.

We record a ``cupy.Event`` on stream L at the end of LOAD and have
the consuming EM worker ``wait_event`` on its own stream E_k before
invoking ``sess.run``. Cheap on the GPU (stream-level dependency
edge — no host sync), and the next sub LOAD keeps H2D-ing while every
EM worker computes.

Cross-thread sync caveat
------------------------
``VmfClusteringSessionCUDA.run`` and ``ELambdaSessionCUDA`` previously
called ``cp.cuda.Stream.null.synchronize()`` for per-stage wall-time
accounting. The null stream (stream 0) has implicit global
synchronization with *all* other streams on the device, so under
multi-EM concurrency, worker A's null-sync would block until worker
B's stream E_B also drains — defeating the parallelism. Those 14
call sites were migrated to ``cp.cuda.get_current_stream().
synchronize()`` (which only syncs the stream set by the surrounding
``with stream_E_k:`` context). In single-EM mode the current stream
*is* the null stream, so behavior is bit-identical to the pre-
concurrency design.

Cupy memory-pool teardown
-------------------------
Stage SAVE calls ``pipe.close()`` once the .mat is written; that runs
``cp.get_default_memory_pool().free_all_blocks()``, reclaiming the
~4 GB EM peak. ``free_all_blocks`` only releases *unused* blocks, so
concurrent stage-LOAD / stage-EM subjects' live device buffers are
unaffected. The pool itself is thread-safe (cupy guards it with an
internal lock), so multiple EM workers can ``cp.empty`` / ``cp.asarray``
from the same pool concurrently without corruption.

Memory budget
-------------
Queues capped at ``maxsize=2`` (matches step0). The structural
upper bound on subjects in flight is ``1 + 2 + em_concurrency + 2 +
1`` = ``6 + em_concurrency`` (1 LOAD active + 2 qLE + N EM active +
2 qES + 1 SAVE), each holding 1.5–4 GB of device buffers. Hitting
that ceiling requires SAVE to stall (e.g. the output disk
backpressures) so the qES backlog grows.

In healthy steady state SAVE drains in ~50 ms while EM takes ~2.4 s,
so qES sits at 0 and qLE at ~1. The default ``em_concurrency=1`` was
measured at **~9.5 GB device peak** on the reference cohort (RTX 5090
24 GB) — well below the structural ceiling and well within budget.
Higher concurrencies fit in device memory but trip the cross-thread
bottlenecks described above — see the EM-concurrency note for the
measured cost. If you push SAVE-side IO with slow disks, watch the
pool size and lower ``maxsize`` accordingly.

Output ordering
---------------
With ``em_concurrency=1`` the LOAD→EM→SAVE chain is FIFO end-to-end,
so ``.mat`` files are written in subject order. With
``em_concurrency ≥ 2`` two EM workers race on qLE and can overtake
each other (subject K-1 may finish EM after subject K), so SAVE
order — and therefore output-file mtimes — is no longer guaranteed
to match input subject order. Filenames carry ``sub_id`` so the
content is still identifiable, but downstream tools that assume
write-order = subject-order must be re-validated.

Per-stage timing fuzziness
--------------------------
``per_stage_em`` (m_step, e_step_lambda_loop, …) wall numbers come
from the in-Session timing block. Under the stage pipeline the
syncs land on a non-null stream and capture only the queue→launch
time, not the actual GPU execution (which is overlapped with another
EM worker's kernels). Stage totals (``em_total``, ``load_total``) are
timed by the coordinator from the outside and remain accurate. This
matches step0's stage-pipeline trade-off; the algorithmic correctness
is unaffected (intra-stream ops still execute in queue order).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import queue
import threading
import traceback
from contextlib import nullcontext
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
    # it on stream E before running. ``None`` on the CPU backend
    # (where no stream sync is needed).
    load_event: Optional[Any] = None
    result: Optional[Step3Result] = None
    failure: Optional[BaseException] = None


class Step3StagePipeline:
    """Stage-pipelined step3 executor.

    Usage::

        coord = Step3StagePipeline(
            subjects=[...],            # iterable of BoldInputs.Subject
            configs=[Step3Config, ...], # one per subject, same order
            gradients={sub_id: NxDg, ...},
            backend="gpu_full",
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
        backend: str,
        em_concurrency: int = 1,
        progress: Optional[ProgressEmitter] = None,
    ):
        if len(subject_ids) != len(configs):
            raise ValueError(
                f"Step3StagePipeline: subject_ids ({len(subject_ids)}) "
                f"and configs ({len(configs)}) length mismatch"
            )
        if not subject_ids:
            raise ValueError("Step3StagePipeline: empty subject list")
        if em_concurrency < 1:
            raise ValueError(
                f"Step3StagePipeline: em_concurrency must be ≥ 1 (got {em_concurrency})"
            )
        self.subject_ids = subject_ids
        self.configs = configs
        # Per-subject gradient .npy ownership is transferred to this
        # coordinator; we pop entries as we dispatch to free memory
        # after each subject's LOAD.
        self.gradients = gradients
        self.backend = backend
        # Pipeline parallelism only helps when EM is on GPU. Caller
        # should gate this, but keep the flag here for clarity inside
        # the stage threads.
        self.is_gpu = backend in ("gpu_elambda", "gpu_full")
        # On CPU backend, parallel EM workers would fight for the
        # same numba threadpool — pin to 1.
        self.em_concurrency = em_concurrency if self.is_gpu else 1
        # Optional frontend progress sink. Per-subject ``step3``
        # running fires from _stage_LOAD on subject entry; done/failed
        # fires from _stage_SAVE before qDone.put. ``None`` →
        # standalone tests; the driver always passes a real emitter.
        # Storing a disabled emitter rather than ``None`` lets every
        # emit call site stay branchless (no per-call None check) —
        # and avoids the awkward "pass a dummy path you never write
        # to" idiom this class used before.
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
        # Backpressure: each inter-stage queue holds at most 2
        # subjects, so a fast upstream stage can't pile up unbounded
        # in-flight buffers. With EM gating, the steady state has
        # ~em_concurrency items being worked on + ~2 queued.
        qLE: queue.Queue = queue.Queue(maxsize=2)
        qES: queue.Queue = queue.Queue(maxsize=2)
        qDone: queue.Queue = queue.Queue()

        # Per-stage cupy streams. ``non_blocking=True`` so they don't
        # serialize against the legacy null stream — required for the
        # H2D on stream L to actually overlap with EM compute on the
        # E streams, and for the EM workers to overlap each other.
        # Stage SAVE is CPU only — no stream needed.
        if self.is_gpu:
            import cupy as cp
            stream_L = cp.cuda.Stream(non_blocking=True)
            em_streams = [
                cp.cuda.Stream(non_blocking=True)
                for _ in range(self.em_concurrency)
            ]
        else:
            stream_L = None
            em_streams = [None] * self.em_concurrency

        # Track EM workers separately so the SAVE thread knows when to
        # stop draining (it gets one sentinel per EM worker — see below).
        threads: List[threading.Thread] = [
            threading.Thread(
                target=self._stage_LOAD, args=(qLE, stream_L),
                name="step3-stageLOAD", daemon=True),
        ]
        for k, s in enumerate(em_streams):
            threads.append(threading.Thread(
                target=self._stage_EM, args=(qLE, qES, s, k),
                name=f"step3-stageEM-{k}", daemon=True))
        threads.append(threading.Thread(
            target=self._stage_SAVE,
            args=(qES, qDone, self.em_concurrency),
            name="step3-stageSAVE", daemon=True))

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
        if self.is_gpu:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    # Stage LOAD — disk reads, .mat parse, fetch_data, build_session.
    # All H2D for the subject's Session runs on stream L. Pushes one
    # sentinel per EM worker so every worker eventually returns.
    # ─────────────────────────────────────────────────────────────────
    def _stage_LOAD(self, qLE: queue.Queue, stream) -> None:
        ctx = stream if stream is not None else nullcontext()
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
                try:
                    pipe = Step3Pipeline(
                        cfg, precomputed_gradient_mat=grad,
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
                    with ctx:
                        # load_inputs() is mostly host-side
                        # (.mat / .b2nd reads, numba JIT). The H2D
                        # happens inside build_session() — the
                        # ``cp.asarray`` calls there will queue on
                        # stream L because of the surrounding ctx.
                        inputs = pipe.load_inputs()
                        pipe.build_session(inputs)
                        if self.is_gpu:
                            import cupy as cp
                            # Records every op queued on stream L so
                            # far, including all the build_session
                            # H2D copies. Whichever EM worker pulls
                            # this work item waits on the event from
                            # its own E_k stream before computing.
                            work.load_event = cp.cuda.Event(
                                block=False, disable_timing=True)
                            work.load_event.record()
                except BaseException as e:
                    work.failure = e
                    traceback.print_exc()
                qLE.put(work)
        finally:
            # One sentinel per EM worker so each one terminates.
            for _ in range(self.em_concurrency):
                qLE.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage EM — intra_em outer loop. The gating stage.
    # Multiple workers (each on its own stream_E_k) share qLE.
    # ─────────────────────────────────────────────────────────────────
    def _stage_EM(self, qLE: queue.Queue, qES: queue.Queue, stream,
                  worker_idx: int) -> None:
        ctx = stream if stream is not None else nullcontext()
        while True:
            work = qLE.get()
            if work is _SENTINEL:
                # Each EM worker drains exactly one sentinel and
                # forwards a SAVE-stage sentinel so SAVE sees one
                # sentinel per EM worker (it knows the count).
                qES.put(_SENTINEL)
                return
            if work.failure is None:
                try:
                    with ctx:
                        if work.load_event is not None and self.is_gpu:
                            # Hold stream E_k until stream L's H2D
                            # writes are visible. Cheap (stream-level
                            # dependency edge).
                            import cupy as cp
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
    # Drains until it has seen ``em_concurrency`` sentinels (one per
    # EM worker), since each EM worker forwards its own sentinel.
    # ─────────────────────────────────────────────────────────────────
    def _stage_SAVE(self, qES: queue.Queue, qDone: queue.Queue,
                    n_em_workers: int) -> None:
        sentinels_seen = 0
        while sentinels_seen < n_em_workers:
            work = qES.get()
            if work is _SENTINEL:
                sentinels_seen += 1
                continue
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
