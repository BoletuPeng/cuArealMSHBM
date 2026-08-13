"""_step1_stage_pipeline.py — phase-pipelined coordinator for step1
``generate_profiles``.

Background: ``run_generate_profiles`` is 240 serial (sub, sess) iters
on the YS cohort. Per-iter the GPU does ~180 ms (H2D + compute + D2H
+ MW-zero inside the leaf), then the main thread spends another ~200
ms doing a (K, V) → (V, K) ``ascontiguousarray(.T)`` copy on host
before appending to the per-subject buffer. With the
:class:`Step1BoldPrefetcher` and the background ``.b2nd`` writer
already hiding gzip and disk I/O respectively, the remaining wall is
**main-thread serial CPU+GPU work**: 188 ms (GPU) + 200 ms (.T) +
overhead ≈ 513 ms / sess.

This module pipelines those two by-phase, mirroring step0's
``_step0_stage_pipeline.py`` design — worker threads, ``queue.Queue``
with maxsize-backpressure, sentinel termination. After this:

    [GZIP pool]  →  qIN  →  [Stage GPU]  →  qPOST  →  [Stage POST]
       (8 workers)              (1 thread, GPU)         (1 thread, CPU)
                                                         │
                                                         ▼
                                              [Stage ASSEM]
                                              (1 thread; buffers
                                               6 sessions per sub,
                                               submits the bitpack +
                                               LZ4 + disk write to
                                               the [WRITE pool])

Per-stage wall budget per (sub, sess):

* Stage GPU   : leaf wall (H2D + zscore + sgemm + threshold + binarize
                + D2H + MW-zero) ≈ 188 ms
* Stage POST  : (K, V) → (V, K) ascontiguousarray ≈ 200 ms
* Stage ASSEM : append + (on every 6th sess) np.empty + 6-slice copy
                ≈ 50 ms amortized, then async submit to WRITE
* WRITE pool  : 1.77 s / subject ≈ 295 ms / sess amortized

Steady-state throughput is gated by ``max(stage_walls)`` ≈ 295 ms / sess
(WRITE), down from the main-thread serial 513 ms / sess.

NOT thread-safe across runs — one ``Step1GenerateProfilesStagePipeline``
instance per ``run_generate_profiles`` call; the driver owns the
prefetcher's lifecycle and primes the sliding window externally.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ._progress import ProgressEmitter
from ._step1_bold_prefetcher import Step1BoldPrefetcher


# Sentinel pushed through each queue to signal end-of-stream.
_SENTINEL = "__step1_pipeline_sentinel__"


@dataclass
class _SessionWork:
    """One (sub, sess) transit unit through the stage pipeline.

    Each stage reads the prior stage's outputs from this object and
    writes its own outputs back before pushing to the next queue.
    Fields are populated in stage order; cleared after consumption
    to let the GC reclaim the (K, V) ~193 MB host buffers.

    Field semantics depend on ``K_unpacked`` — the two relevant data
    fields are deliberately named without layout suffixes because the
    shape/dtype change between backends, and any ``KxV``/``VxK`` tag
    would be a lie on one of the two paths:

      * ``K_unpacked is None`` (CPU leaf path) — ``lh_leaf_out`` is
        ``(K, V_h) fp32`` from the leaf; POST stage transposes to
        ``(V_h, K) fp32`` in ``lh_sess_slab``; ASSEM stacks fp32 slabs;
        WRITE calls ``packbits`` inside ``write_subject_profile_tnd``.

      * ``K_unpacked`` is an int (GPU fused-kernel path) —
        ``lh_leaf_out`` already holds ``(V_h, ⌈K/8⌉) uint8`` packed
        bytes (the GPU leaf emits this directly via
        ``binarize_mwzero_pack_cupy``, skipping the legacy fp32 D2H +
        host MW-zero + transpose + packbits chain). POST stage is a
        pass-through alias (no transpose needed — already V-major).
        ASSEM stacks packed-byte slabs. WRITE receives
        ``D_unpacked=K_unpacked`` so the writer skips its internal
        packbits and just hands the bytes to blosc2.
    """
    sub_id: str
    sess_id: str
    sess_idx_in_sub: int          # 0..n_sess_in_sub-1
    n_sess_in_sub: int
    is_last_sess_in_sub: bool
    out_path: Path                # destination .b2nd for this sub
    K_unpacked: Optional[int] = None  # set on GPU packed-path; None on CPU
    # Leaf output, as returned by ``compute_profile_arrays``. Layout +
    # dtype switch on ``K_unpacked`` — see class docstring. Layout-
    # neutral name on purpose; a ``KxV`` tag would lie on the GPU
    # packed path (which is actually ``(V_h, ⌈K/8⌉)``).
    lh_leaf_out: Optional[np.ndarray] = None
    rh_leaf_out: Optional[np.ndarray] = None
    # Post-POST per-session slab, V-major. On CPU this is the .T copy
    # of the leaf output; on GPU it aliases the leaf output (already
    # V-major). dtype is fp32 on CPU, uint8 on GPU.
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
    """Phase-pipelined executor for step1 ``generate_profiles``.

    Usage::

        with Step1BoldPrefetcher(n_workers=8) as prefetcher:
            coord = Step1GenerateProfilesStagePipeline(
                project_dir=..., bold_paths=..., schedule=...,
                sess_per_sub=..., out_paths=..., leaf_kwargs=...,
                backend=..., prefetcher=prefetcher, lookahead=8,
                verbose=True,
            )
            results = coord.run()  # -> [(sub_id, out_path), ...]
    """

    def __init__(
        self, *,
        project_dir: Path,
        schedule: List[Tuple[str, str]],
        bold_paths: Dict[Tuple[str, str], Tuple[List[str], List[str]]],
        sess_per_sub: Dict[str, int],
        out_paths: Dict[str, Path],
        leaf_kwargs: Dict[str, Any],
        backend: str,
        prefetcher: Step1BoldPrefetcher,
        lookahead: int = 8,
        write_workers: int = 1,
        verbose: bool = True,
        progress: Optional[ProgressEmitter] = None,
    ):
        self.project_dir = project_dir
        self.schedule = schedule
        self.bold_paths = bold_paths
        self.sess_per_sub = sess_per_sub
        self.out_paths = out_paths
        self.leaf_kwargs = leaf_kwargs
        self.backend = backend
        self.prefetcher = prefetcher
        self.lookahead = int(lookahead)
        self.write_workers = int(write_workers)
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
        # Pre-compute per-(schedule index) → sess-index-within-subject in
        # one O(N) pass so _prime_loop's per-iter lookup is O(1) instead
        # of an O(n_sess) backward scan. Cheap for YS (6 sess/sub) but
        # noticeable on 1000-sub × 12-sess cohorts.
        self._sess_idx_table: List[int] = []
        _sub_counts: Dict[str, int] = {}
        for sub_id, _sess_id in self.schedule:
            self._sess_idx_table.append(_sub_counts.get(sub_id, 0))
            _sub_counts[sub_id] = _sub_counts.get(sub_id, 0) + 1

    def run(self) -> List[Tuple[str, Path]]:
        # Backpressure caps.
        # qIN  → GPU stage   : feeds (sub, sess) tags; the GPU thread
        #                       calls prefetcher.get() to pull host
        #                       BOLD just-in-time. Small cap is fine.
        # qPOST→ POST stage  : carries (K, V) fp32 host ~386 MB per
        #                       session. Cap=2 → ≤2 × 386 MB ≈ 770 MB
        #                       buffered between GPU and POST.
        # qASSEM→ ASSEM stage: (V, K) fp32 host, same size as POST in.
        #                       Cap=2.
        qIN: queue.Queue = queue.Queue(maxsize=4)
        qPOST: queue.Queue = queue.Queue(maxsize=2)
        qASSEM: queue.Queue = queue.Queue(maxsize=2)
        qDone: queue.Queue = queue.Queue()

        # 1-worker write pool by default — LZ4 + bitpack saturates one
        # core. The driver can pass write_workers=2 if profiling shows
        # the write stage gating throughput.
        n_write_workers = max(1, self.write_workers)
        write_pool = ThreadPoolExecutor(
            max_workers=n_write_workers,
            thread_name_prefix="b2nd-write-pl",
        )

        threads = [
            threading.Thread(
                target=self._prime_loop, args=(qIN,),
                name="step1-prime", daemon=True),
            threading.Thread(
                target=self._stage_GPU, args=(qIN, qPOST),
                name="step1-stageGPU", daemon=True),
            threading.Thread(
                target=self._stage_POST, args=(qPOST, qASSEM),
                name="step1-stagePOST", daemon=True),
            threading.Thread(
                target=self._stage_ASSEM,
                args=(qASSEM, qDone, write_pool, n_write_workers),
                name="step1-stageASSEM", daemon=True),
        ]
        for t in threads:
            t.start()

        # Collect from qDone. One subject finishes here once its
        # ASSEM-thread submits the WRITE future and the future
        # resolves (the ASSEM thread blocks on .result() before
        # pushing to qDone).
        n_subjects = len(self.sess_per_sub)
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
    # lookahead window of ``self.lookahead`` items ahead of the GPU
    # consumer. Runs in its own thread so qIN's backpressure (maxsize)
    # naturally throttles us to the GPU consumer's pace.
    # ─────────────────────────────────────────────────────────────────
    def _prime_loop(self, qIN: queue.Queue) -> None:
        # Prime the first lookahead window so the GPU consumer never
        # blocks on a cold prime.
        for k in range(min(self.lookahead, len(self.schedule))):
            sub_id, sess_id = self.schedule[k]
            lh_p, rh_p = self.bold_paths[(sub_id, sess_id)]
            self.prefetcher.prime(sub_id, sess_id, lh_p, rh_p)

        try:
            for i, (sub_id, sess_id) in enumerate(self.schedule):
                # Prime the item `lookahead` steps ahead of i (if any),
                # so by the time the GPU consumes (i+lookahead)'s entry
                # via prefetcher.get(...), it's already decoded.
                ahead = i + self.lookahead
                if ahead < len(self.schedule):
                    a_sub, a_sess = self.schedule[ahead]
                    a_lh, a_rh = self.bold_paths[(a_sub, a_sess)]
                    self.prefetcher.prime(a_sub, a_sess, a_lh, a_rh)

                n_sess = self.sess_per_sub[sub_id]
                # O(1) lookup into the precomputed table (built once in
                # __init__ from the schedule). Robust to mixed-cohort
                # schedules — no modulo-uniformity assumption.
                sess_idx_in_sub = self._sess_idx_table[i]
                is_last = (sess_idx_in_sub + 1 == n_sess)

                # First session of this subject enters the pipeline:
                # tag the user-visible ``step1`` slot as running. Done/
                # failed transitions happen at the WRITE-future drain
                # in _stage_ASSEM (the terminal stage for a subject).
                if sess_idx_in_sub == 0:
                    self.progress.emit_state("step1", sub_id, "running")

                work = _SessionWork(
                    sub_id=sub_id,
                    sess_id=sess_id,
                    sess_idx_in_sub=sess_idx_in_sub,
                    n_sess_in_sub=n_sess,
                    is_last_sess_in_sub=is_last,
                    out_path=self.out_paths[sub_id],
                )
                qIN.put(work)
        finally:
            qIN.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage GPU — leaf call: H2D + zscore + GEMM + threshold + binarize
    # + D2H + MW-zero. One thread, owns the GPU work end-to-end.
    # ─────────────────────────────────────────────────────────────────
    def _stage_GPU(self, qIN: queue.Queue, qPOST: queue.Queue) -> None:
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
                            f"sub={work.sub_id} sess={work.sess_id} "
                            f"backend={self.backend}", flush=True,
                        )
                    lh_bin, rh_bin, K_unpacked = compute_profile_arrays(
                        seed_mesh=self.leaf_kwargs["seed_mesh"],
                        targ_mesh=self.leaf_kwargs["targ_mesh"],
                        out_dir=self.project_dir,
                        sub=work.sub_id,
                        sess=work.sess_id,
                        split_flag=self.leaf_kwargs.get("split_flag", "0"),
                        threshold=self.leaf_kwargs.get("threshold", 0.1),
                        dtype_reduce=self.leaf_kwargs.get(
                            "profile_dtype_reduce", np.float32),
                        backend=self.backend,
                        precomputed_bold_runs=bold_runs,
                    )
                    # Drop the prefetched host buffer eagerly — the
                    # leaf has H2D'd what it needs.
                    del bold_runs
                    work.lh_leaf_out = lh_bin
                    work.rh_leaf_out = rh_bin
                    # K_unpacked is None on CPU (legacy fp32 path),
                    # an int on GPU (packed path). Stage POST + ASSEM
                    # branch on this — see _SessionWork docstring.
                    work.K_unpacked = K_unpacked
                except BaseException as e:
                    work.failure = e
                qPOST.put(work)
        finally:
            qPOST.put(_SENTINEL)

    # ─────────────────────────────────────────────────────────────────
    # Stage POST — host layout swap.
    #   CPU leaf path : (K, V) fp32 -> (V, K) fp32 via np.ascontiguousarray.
    #                   Plain CPU memcpy; isolating it here lets it
    #                   overlap with Stage GPU's next-session compute.
    #   GPU packed path: leaf already emits (V, ⌈K/8⌉) uint8 packed bytes
    #                   in V-major layout, so this stage just aliases
    #                   the field — no work to do. Kept as a stage so
    #                   the threading topology stays uniform.
    # ─────────────────────────────────────────────────────────────────
    def _stage_POST(self, qPOST: queue.Queue, qASSEM: queue.Queue) -> None:
        try:
            while True:
                work = qPOST.get()
                if work is _SENTINEL:
                    return
                if work.failure is None:
                    try:
                        if work.K_unpacked is not None:
                            # Packed (V, ⌈K/8⌉) uint8 from the GPU fused
                            # kernel — already V-major; just pass through.
                            work.lh_sess_slab = work.lh_leaf_out
                            work.rh_sess_slab = work.rh_leaf_out
                        else:
                            # Legacy CPU fp32 path: transpose (K, V) -> (V, K).
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
        write_pool: ThreadPoolExecutor, max_inflight_writes: int,
    ) -> None:
        # Per-subject session-buffers, keyed by sub_id. The same sub's
        # entries arrive in sess-order (the schedule is sub-major and
        # all earlier stages preserve order). The third tuple slot
        # holds ``K_unpacked`` — None on the CPU fp32 path, an int on
        # the GPU packed path. Recorded at the first session and
        # forwarded to ``write_subject_profile_tnd(D_unpacked=...)`` on
        # the last session.
        buffers: Dict[
            str,
            Tuple[List[np.ndarray], List[np.ndarray], Optional[int]],
        ] = {}
        # Cap on in-flight WRITE futures (= pool worker count). >1 =
        # parallel LZ4; default 1.
        pending_writes: List[Tuple[str, Path, Future, np.ndarray]] = []
        # Per-subject "first terminal state wins" set — sub_ids that
        # have already had a terminal (done/failed) state emitted +
        # qDone result pushed. Guards against the pre-existing
        # buffer-logic edge case where a per-session failure followed
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
        # subjects). The pre-existing pre-this-PR behaviour would
        # double-put a sub whose early session failed then later
        # sessions succeeded, violating that contract — the
        # idempotency closes both holes simultaneously.
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
            sub_id, op, fut, _arr = pending_writes.pop(0)
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
                # bitpacked slabs (up to T_so_far × ~4 MB on fsa6 K=400
                # — small per sub but unbounded across many failures
                # in one run) become collectable as soon as the
                # terminal state lands. Without this, buffers[sub_id]
                # would survive until _stage_ASSEM exits at run end.
                buffers.pop(work.sub_id, None)
                _finish(work.sub_id, work.out_path, failure=work.failure)
                continue
            try:
                if work.sub_id not in buffers:
                    # First session for this sub — record K_unpacked
                    # (None on CPU, int on GPU packed path) alongside
                    # the per-sess slab lists.
                    buffers[work.sub_id] = ([], [], work.K_unpacked)
                bucket = buffers[work.sub_id]
                bucket[0].append(work.lh_sess_slab)
                bucket[1].append(work.rh_sess_slab)
                work.lh_sess_slab = None
                work.rh_sess_slab = None

                if not work.is_last_sess_in_sub:
                    continue

                # All sessions for this sub are ready. Build the
                # (T, V_lh + V_rh, D) array and submit the write.
                # D == K on the CPU fp32 path; D == ⌈K/8⌉ on the GPU
                # packed path. dtype follows the slabs' dtype directly
                # — no astype here.
                lh_per, rh_per, K_unpacked = buffers.pop(work.sub_id)
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
                    layout_tag = (
                        f"packed ⌈K/8⌉={D} K={K_unpacked}"
                        if K_unpacked is not None else f"fp32 K={D}"
                    )
                    print(
                        f"      submitting b2nd write sub={work.sub_id}  "
                        f"shape=({T_count}, {V_lh + V_rh}, {D})  "
                        f"in-mem {arr.nbytes / 1e6:.1f} MB  "
                        f"layout={layout_tag}",
                        flush=True,
                    )

                # Cap in-flight writes by draining oldest if needed.
                while len(pending_writes) >= max_inflight_writes:
                    _drain_oldest_write()

                # Lazy import — keeps this module independent of
                # blosc2 at import time.
                from arealmshbm.data_io.profile_io import (
                    write_subject_profile_tnd,
                )
                # On the packed path, tell the writer the bytes are
                # already packed so it skips its internal validation +
                # astype + packbits. The writer still emits the same
                # .b2nd schema (same vlmeta tags, same chunk shape).
                write_kwargs = {}
                if K_unpacked is not None:
                    write_kwargs["D_unpacked"] = K_unpacked
                fut = write_pool.submit(
                    write_subject_profile_tnd, work.out_path, arr,
                    **write_kwargs,
                )
                pending_writes.append(
                    (work.sub_id, work.out_path, fut, arr),
                )
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
