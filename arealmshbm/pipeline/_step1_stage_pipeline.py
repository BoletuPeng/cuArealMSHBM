"""_step1_stage_pipeline.py — phase-pipelined coordinator for the step1
``generate_profiles`` CPU backend.

Background: ``run_generate_profiles(backend='cpu')`` is 240 serial
(sub, sess) iters on the YS cohort. Per-iter the numba leaf produces a
(K, V) fp32 pair, then the main thread spends another ~200 ms doing a
(K, V) → (V, K) ``ascontiguousarray(.T)`` copy on host before appending
to the per-subject buffer. With the :class:`Step1BoldPrefetcher` and
the background ``.b2nd`` writer already hiding gzip and disk I/O
respectively, the remaining wall is **main-thread serial CPU work**.

This module pipelines those phases, mirroring step0's
``_step0_stage_pipeline.py`` design — worker threads, ``queue.Queue``
with maxsize-backpressure, sentinel termination:

    [GZIP pool]  →  qIN  →  [Stage LEAF]  →  qPOST  →  [Stage POST]
       (8 workers)              (1 thread)              (1 thread)
                                                         │
                                                         ▼
                                              [Stage ASSEM]
                                              (1 thread; buffers
                                               6 sessions per sub,
                                               submits the bitpack +
                                               blosc2 + disk write to
                                               the [WRITE pool])

Steady-state throughput is gated by ``max(stage_walls)`` instead of
their sum.

The GPU backend does not come through here at all — it is one fused
whole-subject call per subject
(``generate_profiles.profiles_subject_gpu``), which owns its own
ingest, compute and writer.

NOT thread-safe across runs — one ``Step1GenerateProfilesStagePipeline``
instance per ``run_generate_profiles`` call; that caller owns the
prefetcher's lifecycle, and this coordinator's prime loop keeps its
sliding window filled.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._progress import ProgressEmitter
from ._step1_bold_prefetcher import Step1BoldPrefetcher


# Sentinel pushed through each queue to signal end-of-stream.
_SENTINEL = "__step1_pipeline_sentinel__"
# Sessions the prime loop keeps decoding ahead of the LEAF consumer.
_LOOKAHEAD = 8


@dataclass
class _SessionWork:
    """One (sub, sess) transit unit through the stage pipeline.

    Each stage reads the prior stage's outputs from this object and
    writes its own outputs back before pushing to the next queue.
    Fields are populated in stage order; cleared after consumption
    to let the GC reclaim the (K, V) ~193 MB host buffers.

    ``lh_leaf_out`` is ``(K, V_h) fp32`` as the leaf returns it; POST
    transposes to ``(V_h, K)`` in ``lh_sess_slab``; ASSEM stacks those;
    WRITE calls ``packbits`` inside ``write_subject_profile_tnd``.
    """
    sub_id: str
    sess_id: str
    is_last_sess_in_sub: bool
    out_path: Path                # destination .b2nd for this sub
    # Leaf output, as returned by ``compute_profile_arrays``: (K, V_h).
    lh_leaf_out: Optional[np.ndarray] = None
    rh_leaf_out: Optional[np.ndarray] = None
    # Post-POST per-session slab, V-major: the .T copy of the leaf
    # output.
    lh_sess_slab: Optional[np.ndarray] = None
    rh_sess_slab: Optional[np.ndarray] = None
    failure: Optional[BaseException] = None


@dataclass
class _SubjectResult:
    """One subject's pipeline output, ferried through qDone."""
    sub_id: str
    out_path: Path
    failure: Optional[BaseException] = None


class Step1GenerateProfilesStagePipeline:
    """Phase-pipelined executor for the step1 ``generate_profiles`` CPU
    backend.

    Usage::

        with Step1BoldPrefetcher(n_workers=8) as prefetcher:
            coord = Step1GenerateProfilesStagePipeline(
                project_dir=..., bold_paths=..., schedule=...,
                out_paths=..., threshold=...,
                prefetcher=prefetcher, verbose=True,
            )
            results = coord.run()  # -> [(sub_id, out_path), ...]
    """

    def __init__(
        self, *,
        project_dir: Path,
        schedule: List[Tuple[str, str]],
        bold_paths: Dict[Tuple[str, str], Tuple[List[str], List[str]]],
        out_paths: Dict[str, Path],
        threshold: float,
        prefetcher: Step1BoldPrefetcher,
        verbose: bool = True,
        progress: Optional[ProgressEmitter] = None,
    ):
        self.project_dir = project_dir
        self.schedule = schedule
        self.bold_paths = bold_paths
        self.out_paths = out_paths
        self.threshold = threshold
        self.prefetcher = prefetcher
        self.verbose = bool(verbose)
        # Optional frontend progress sink. Per-subject ``step1``
        # running fires from _prime_loop on the first session of each
        # sub; done/failed fires from _stage_ASSEM at WRITE-future
        # drain. ``None`` → standalone tests that don't care about
        # progress (the driver always passes a real emitter). Storing
        # a disabled emitter rather than ``None`` lets every emit call
        # site stay branchless (no per-call None check).
        self.progress = (
            progress if progress is not None
            else ProgressEmitter.disabled()
        )

    def run(self) -> List[Tuple[str, Path]]:
        # Backpressure caps.
        # qIN  → LEAF stage  : feeds (sub, sess) tags; the LEAF thread
        #                       calls prefetcher.get() to pull host
        #                       BOLD just-in-time. Small cap is fine.
        # qPOST→ POST stage  : carries (K, V) fp32 host ~386 MB per
        #                       session. Cap=2 → ≤2 × 386 MB ≈ 770 MB
        #                       buffered between LEAF and POST.
        # qASSEM→ ASSEM stage: (V, K) fp32 host, same size as POST in.
        #                       Cap=2.
        qIN: queue.Queue = queue.Queue(maxsize=4)
        qPOST: queue.Queue = queue.Queue(maxsize=2)
        qASSEM: queue.Queue = queue.Queue(maxsize=2)
        qDone: queue.Queue = queue.Queue()

        # One write worker — bitpack + blosc2 (LZ4 + bitshuffle)
        # saturates one core.
        write_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="b2nd-write-pl",
        )

        threads = [
            threading.Thread(
                target=self._prime_loop, args=(qIN,),
                name="step1-prime", daemon=True),
            threading.Thread(
                target=self._stage_LEAF, args=(qIN, qPOST),
                name="step1-stageLEAF", daemon=True),
            threading.Thread(
                target=self._stage_POST, args=(qPOST, qASSEM),
                name="step1-stagePOST", daemon=True),
            threading.Thread(
                target=self._stage_ASSEM,
                args=(qASSEM, qDone, write_pool),
                name="step1-stageASSEM", daemon=True),
        ]
        for t in threads:
            t.start()

        # Collect from qDone. One subject finishes here once its
        # ASSEM-thread submits the WRITE future and the future
        # resolves (the ASSEM thread blocks on .result() before
        # pushing to qDone).
        n_subjects = len(self.out_paths)
        out: List[Tuple[str, Path]] = []
        failures: List[BaseException] = []
        for _ in range(n_subjects):
            res: _SubjectResult = qDone.get()
            if res.failure is not None:
                failures.append(res.failure)
                continue
            out.append((res.sub_id, res.out_path))

        for t in threads:
            t.join()

        write_pool.shutdown(wait=True)

        if failures:
            raise failures[0]
        return out

    # ─────────────────────────────────────────────────────────────────
    # Prime loop — main coordination: feed qIN with (sub, sess) work
    # items in schedule order, and maintain the prefetcher's sliding
    # lookahead window of ``_LOOKAHEAD`` items ahead of the LEAF
    # consumer. Runs in its own thread so qIN's backpressure (maxsize)
    # naturally throttles us to the LEAF consumer's pace.
    # ─────────────────────────────────────────────────────────────────
    def _prime_loop(self, qIN: queue.Queue) -> None:
        # Prime the first lookahead window so the LEAF consumer never
        # blocks on a cold prime.
        for k in range(min(_LOOKAHEAD, len(self.schedule))):
            sub_id, sess_id = self.schedule[k]
            lh_p, rh_p = self.bold_paths[(sub_id, sess_id)]
            self.prefetcher.prime(sub_id, sess_id, lh_p, rh_p)

        n = len(self.schedule)
        try:
            for i, (sub_id, sess_id) in enumerate(self.schedule):
                # Prime the item ``_LOOKAHEAD`` steps ahead of i (if any),
                # so by the time LEAF consumes (i+_LOOKAHEAD)'s entry
                # via prefetcher.get(...), it's already decoded.
                ahead = i + _LOOKAHEAD
                if ahead < n:
                    a_sub, a_sess = self.schedule[ahead]
                    a_lh, a_rh = self.bold_paths[(a_sub, a_sess)]
                    self.prefetcher.prime(a_sub, a_sess, a_lh, a_rh)

                # The schedule is subject-major: a subject's sessions
                # are one contiguous run of entries, of any length.
                is_first = i == 0 or self.schedule[i - 1][0] != sub_id
                is_last = i + 1 == n or self.schedule[i + 1][0] != sub_id

                # First session of this subject enters the pipeline:
                # tag the user-visible ``step1`` slot as running. Done/
                # failed transitions happen at the WRITE-future drain
                # in _stage_ASSEM (the terminal stage for a subject).
                if is_first:
                    self.progress.emit_state("step1", sub_id, "running")

                work = _SessionWork(
                    sub_id=sub_id,
                    sess_id=sess_id,
                    is_last_sess_in_sub=is_last,
                    out_path=self.out_paths[sub_id],
                )
                qIN.put(work)
        finally:
            qIN.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage LEAF — the numba supercall: seed gather + zscore + GEMM +
    # threshold + binarize + MW-zero. One thread, owns it end-to-end.
    # ─────────────────────────────────────────────────────────────────
    def _stage_LEAF(self, qIN: queue.Queue, qPOST: queue.Queue) -> None:
        from arealmshbm.generate_profiles import compute_profile_arrays
        try:
            while True:
                work = qIN.get()
                if work is _SENTINEL:
                    return
                try:
                    bold_runs = self.prefetcher.get(work.sub_id, work.sess_id)
                    if self.verbose:
                        print(
                            f"  [1] compute_profile_arrays "
                            f"sub={work.sub_id} sess={work.sess_id}",
                            flush=True,
                        )
                    lh_bin, rh_bin = compute_profile_arrays(
                        out_dir=self.project_dir,
                        sub=work.sub_id,
                        sess=work.sess_id,
                        threshold=self.threshold,
                        bold_runs=bold_runs,
                    )
                    # Drop the prefetched host buffer eagerly — the
                    # leaf has taken what it needs.
                    del bold_runs
                    work.lh_leaf_out = lh_bin
                    work.rh_leaf_out = rh_bin
                except BaseException as e:
                    work.failure = e
                qPOST.put(work)
        finally:
            qPOST.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage POST — host layout swap: (K, V) fp32 -> (V, K) fp32 via
    # np.ascontiguousarray. Plain CPU memcpy; isolating it here lets
    # it overlap with Stage LEAF's next-session compute.
    # ─────────────────────────────────────────────────────────────────
    def _stage_POST(self, qPOST: queue.Queue, qASSEM: queue.Queue) -> None:
        try:
            while True:
                work = qPOST.get()
                if work is _SENTINEL:
                    return
                if work.failure is None:
                    try:
                        work.lh_sess_slab = np.ascontiguousarray(
                            work.lh_leaf_out.T)
                        work.rh_sess_slab = np.ascontiguousarray(
                            work.rh_leaf_out.T)
                        work.lh_leaf_out = None
                        work.rh_leaf_out = None
                    except BaseException as e:
                        work.failure = e
                qASSEM.put(work)
        finally:
            qASSEM.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage ASSEM — accumulate per-subject sessions; on the last sess
    # of a sub, np.empty + copy into (T, V_lh+V_rh, D) and hand off to
    # the WRITE pool. Block on the write future before reporting to
    # qDone so the caller sees the write's exception (if any).
    # ─────────────────────────────────────────────────────────────────
    def _stage_ASSEM(
        self, qASSEM: queue.Queue, qDone: queue.Queue,
        write_pool: ThreadPoolExecutor,
    ) -> None:
        # Per-subject session-buffers, keyed by sub_id. The same sub's
        # entries arrive in sess-order (the schedule is sub-major and
        # all earlier stages preserve order).
        buffers: Dict[
            str, Tuple[List[np.ndarray], List[np.ndarray]],
        ] = {}
        # In-flight WRITE futures: at most one (the pool has one worker).
        pending_writes: List[Tuple[str, Path, Future]] = []
        # Per-subject "first terminal state wins" set — sub_ids that
        # have already had a terminal (done/failed) state emitted +
        # qDone result pushed. Guards against the buffer-logic edge
        # case where a per-session failure followed
        # by later sessions of the same subject assembling + writing
        # successfully would emit ``failed`` → ``done`` (out-of-
        # order terminal states from the frontend's perspective). Also
        # short-circuits buffer accumulation for sessions of an
        # already-failed sub so we don't waste memory + CPU
        # transposing / assembling slabs that nobody will read.
        terminated: set[str] = set()

        # Single chokepoint for "subject N's terminal state". Every
        # qDone.put for a subject result routes through here so the
        # progress emit and the queue push stay coupled — no future
        # edit can add a 4th terminal site that forgets the emit.
        # **Idempotent on sub_id**: a second call for an already-
        # terminated sub is a silent no-op — suppressing BOTH the
        # progress emit AND the qDone.put. This matches the
        # caller contract (exactly one ``_SubjectResult`` per
        # subject; the upstream coordinator drains K results for K
        # subjects): a sub whose early session failed and whose later
        # sessions succeeded is put, and its state emitted, once.
        def _finish(sub_id: str, out_path: Path,
                    failure: Optional[BaseException] = None) -> None:
            if sub_id in terminated:
                # First-terminal-wins: drop redundant terminal emits
                # (e.g. a successful write completing after an earlier
                # session-level failure of the same sub) AND the
                # corresponding qDone.put. Upstream sees exactly one
                # result per planned subject regardless of mid-flight
                # failure ordering.
                return
            terminated.add(sub_id)
            if failure is None:
                self.progress.emit_state("step1", sub_id, "done")
            else:
                self.progress.emit_state(
                    "step1", sub_id, "failed",
                    error=f"{type(failure).__name__}: {failure}",
                )
            qDone.put(_SubjectResult(
                sub_id=sub_id, out_path=out_path, failure=failure,
            ))

        def _drain_oldest_write() -> None:
            sub_id, op, fut = pending_writes.pop(0)
            try:
                fut.result()
                _finish(sub_id, op)
                if self.verbose:
                    disk_mb = op.stat().st_size / 1e6
                    print(
                        f"      → wrote {op.name}  on-disk {disk_mb:.1f} MB",
                        flush=True,
                    )
            except BaseException as e:
                _finish(sub_id, op, failure=e)

        while True:
            work = qASSEM.get()
            if work is _SENTINEL:
                break
            if work.sub_id in terminated:
                # Earlier session of this sub already terminated (most
                # likely failed upstream). Drop subsequent sessions
                # without buffering, assembling, or re-emitting state
                # — _finish's idempotency would block a duplicate
                # emit anyway, but skipping here also saves the slab
                # transpose / np.empty assembly work.
                # Belt-and-braces buffer pop: the failure branch below
                # already pops on transition, but if the loop is re-
                # entered for a sub that's already terminated for any
                # reason (defensive cleanup), make sure no slab bucket
                # is left dangling.
                buffers.pop(work.sub_id, None)
                continue
            if work.failure is not None:
                # Pop any half-accumulated bucket BEFORE _finish so the
                # (V, K) fp32 slabs (~193 MB per session on fsa6, so
                # unbounded across many failures in one run) become
                # collectable as soon as the terminal state lands.
                # Without this, buffers[sub_id] would survive until
                # _stage_ASSEM exits at run end.
                buffers.pop(work.sub_id, None)
                _finish(work.sub_id, work.out_path, failure=work.failure)
                continue
            try:
                if work.sub_id not in buffers:
                    buffers[work.sub_id] = ([], [])
                bucket = buffers[work.sub_id]
                bucket[0].append(work.lh_sess_slab)
                bucket[1].append(work.rh_sess_slab)
                work.lh_sess_slab = None
                work.rh_sess_slab = None

                if not work.is_last_sess_in_sub:
                    continue

                # All sessions for this sub are ready. Build the
                # (T, V_lh + V_rh, K) fp32 array and submit the write.
                lh_per, rh_per = buffers.pop(work.sub_id)
                T_count = len(lh_per)
                V_lh = lh_per[0].shape[0]
                V_rh = rh_per[0].shape[0]
                D = lh_per[0].shape[1]
                slab_dtype = lh_per[0].dtype
                arr = np.empty(
                    (T_count, V_lh + V_rh, D), dtype=slab_dtype)
                for t in range(T_count):
                    arr[t, :V_lh] = lh_per[t]
                    arr[t, V_lh:] = rh_per[t]
                # Free the per-sess refs eagerly.
                del lh_per, rh_per

                if self.verbose:
                    print(
                        f"      submitting b2nd write sub={work.sub_id}  "
                        f"shape=({T_count}, {V_lh + V_rh}, {D})  "
                        f"in-mem {arr.nbytes / 1e6:.1f} MB  "
                        f"layout=fp32 K={D}",
                        flush=True,
                    )

                # One write in flight: drain the previous subject's.
                while pending_writes:
                    _drain_oldest_write()

                # Lazy import — keeps this module independent of
                # blosc2 at import time.
                from arealmshbm.data_io.profile_io import (
                    write_subject_profile_tnd,
                )
                fut = write_pool.submit(
                    write_subject_profile_tnd, work.out_path, arr,
                )
                pending_writes.append((work.sub_id, work.out_path, fut))
            except BaseException as e:
                # Pop any partial bucket if the exception fired BEFORE
                # the inline ``buffers.pop`` above (e.g., shape
                # mismatch on append, OOM at np.empty). The successful
                # path's pop already runs before write submission, so
                # this only matters for early-throws.
                buffers.pop(work.sub_id, None)
                _finish(work.sub_id, work.out_path, failure=e)

        # Drain remaining writes.
        while pending_writes:
            _drain_oldest_write()
