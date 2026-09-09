"""profiles_subject_gpu.py

The GPU path for subgraph 1, whole-subject and fused. This module owns
the **subject**: ingest of session ``g+1`` overlaps the compute of
session ``g``, every session writes into one device-resident packed
buffer, and each session's packed slab leaves the GPU straight into its
final place in the whole-subject pinned block while the .b2nd write
runs on a background thread.

Ingest has two forms and the compute has one. With
``nvidia-nvcomp-cu12`` installed the sessions arrive from
:func:`~arealmshbm.data_io.gifti_bold_gpu.iter_subject_bold_gpu`
(``raw_sessions=True``: GPU bytescan + base64 + batched Deflate);
without it they arrive from the CPU reader
(:func:`~arealmshbm.data_io.gifti_io.read_surface_gifti` on a small
thread pool, assembled into pinned staging and H2D'd). Both accept
the same files (canonical single-line base64) and hand
:func:`compute_subject_profiles_gpu` the same ``(T, n_lh + n_rh)``
fp32 device buffer -- byte-for-byte the same, pinned by
``data_io/tests/test_gifti_readers.py`` -- so the packed output does
not depend on which one ran; the result reports which did.

Public API:

    subject_seed_and_mw(seed_mesh, targ_mesh)
        ``(seed_idx, mw_mask, n_lh, n_rh)`` on the joint ``lh|rh`` axis.

    compute_subject_profiles_gpu(sessions, ...) -> (packed_host, K)
        The fused compute: one zscore launch over all ``n_full``
        columns, one sgemm into the joint ``(K, n_full)`` array, an
        exact radix-histogram threshold, binarize + MW-zero + pack.

    generate_subject_profiles_gpu(project_dir, sub_id, sess_ids, ...)
        Ingest + compute + streaming .b2nd write for one subject;
        returns a :class:`SubjectProfilesResult`.

    prewarm_generate_profiles_gpu(bold_paths=None)
        Idempotent first-call cost (NVRTC, cuBLAS, nvCOMP, blosc2).

    release_pinned_staging()
        Return every pooled pinned D2H block to the driver.

Sessions of differing length are supported. The nvCOMP reader wants
one ``T`` per call, so on that ingest the subject is driven as a chain
of calls split on the GIFTI ``NumberOfDataArrays`` header (one call
when ``T`` is uniform, which is the production case); the CPU-reader
ingest reads a pair at a time and needs no split.

Semantics: per-run zscore, sum over runs, ``* 1/n_runs``, one
threshold over the joint ``lh|rh`` array, ``>=`` binarize, medial wall
zeroed, LSB-first packing along K. The joint array the threshold sees
needs no ``cp.concatenate``, because the session buffer already stores
lh in columns ``[0, n_lh)`` and rh in ``[n_lh, n_full)``.

Bit-exactness
-------------
Gather-then-zscore and zscore-then-gather are the same computation:
the kernel's per-column result depends only on that column's values,
``T`` and the block size -- never on ``N``, which only enters the
addressing. So normalising all ``n_full`` columns once and gathering
the seed columns out of the result is bit-identical to the
"gather, then zscore the (T, K) copy" order the retired per-session
leaf used. The degenerate-column and ``+-0.0`` conventions of the two
kernels are documented in :mod:`._kernels_gpu`; neither is observable
after the ``>=`` compare.
``tests/test_subject_profiles_gpu.py`` pins both against a
transcription of that retired formulation.

cupy is imported at module top -- this file is only reachable through
the GPU entry points below, so the CPU path never pays for it.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import re
import threading
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cupy as cp
import numpy as np

from ..data_io.gifti_io import read_surface_gifti
from ._kernels_gpu import (
    binarize_mwzero_pack_cupy,
    threshold_top_fraction_exact_cupy,
    zscore_unit_norm_columns_zerovar_cupy,
)
from .profiles import _icosphere_nverts, _read_censor_vector, _read_lines



# ─────────────────────────────────────────────────────────────────────
# Pinned D2H staging pool (the packed buffer is the same size for every
# subject of a cohort, so blocks recycle)
# ─────────────────────────────────────────────────────────────────────
_PINNED_LOCK = threading.Lock()
_PINNED_FREE: List = []          # [(mem, nbytes)], LIFO
_PINNED_KEEP = 2                 # blocks retained across subjects


def _acquire_pinned(nbytes: int):
    """``(uint8 view, token)`` -- a pinned host block you must release.

    A pool rather than one cached block: the subject's packed bytes are
    D2H'd into it and then read by the background .b2nd writer, so the
    NEXT subject's D2H must not be allowed to land in the same memory
    while that writer is still going. The token is released by
    :meth:`SubjectProfilesResult.wait`, which joins the writer first,
    so the ordering is enforced rather than assumed.

    Never blocks: an empty pool allocates. At most :data:`_PINNED_KEEP`
    blocks are retained on release, so a caller that forgets to release
    leaks bounded memory instead of deadlocking.

    Best fit, not first fit: two block sizes circulate per subject (this
    packed block and, on the CPU-reader ingest, the ``(T, n_full)`` fp32
    staging block), and first fit would hand the larger one to the
    smaller request and then re-allocate the larger every other
    subject.
    """
    nbytes = int(nbytes)
    with _PINNED_LOCK:
        best = -1
        for i, (_mem, cap) in enumerate(_PINNED_FREE):
            if cap >= nbytes and (best < 0 or cap < _PINNED_FREE[best][1]):
                best = i
        if best >= 0:
            mem, cap = _PINNED_FREE.pop(best)
            return (np.frombuffer(mem, dtype=np.uint8, count=nbytes),
                    (mem, cap))
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return (np.frombuffer(mem, dtype=np.uint8, count=nbytes),
            (mem, nbytes))


def _release_pinned(token) -> None:
    if token is None:
        return
    with _PINNED_LOCK:
        if len(_PINNED_FREE) < _PINNED_KEEP:
            _PINNED_FREE.append(token)


def release_pinned_staging() -> None:
    """Give every pooled pinned D2H block back to the driver."""
    with _PINNED_LOCK:
        del _PINNED_FREE[:]


# ─────────────────────────────────────────────────────────────────────
# Seed / medial-wall index construction
# ─────────────────────────────────────────────────────────────────────
def subject_seed_and_mw(seed_mesh: str, targ_mesh: str):
    """``(seed_idx, mw_mask, n_lh, n_rh)`` for one (seed, targ) pair.

    ``seed_idx`` is the ``(K,) int64`` **global** column index into the
    joint ``lh|rh`` axis: the first ``_icosphere_nverts(seed_mesh)``
    vertices of each hemi whose ``MARS_label == 2``, with the rh block
    offset by ``n_lh``. ``mw_mask`` is ``(n_lh + n_rh,) uint8``, 1 at
    ``MARS_label == 1`` (medial wall).
    """
    from ..data_io.load_avg_mesh import load_avg_mesh

    if not targ_mesh.startswith("fsaverage"):
        raise ValueError(
            f"only fsaverage targets are supported; got {targ_mesh!r}")
    lh_mars = np.asarray(load_avg_mesh("lh", targ_mesh, "inflated")
                         ["MARS_label"]).ravel()
    rh_mars = np.asarray(load_avg_mesh("rh", targ_mesh, "inflated")
                         ["MARS_label"]).ravel()
    n_seed = _icosphere_nverts(seed_mesh)
    n_lh = int(lh_mars.shape[0])
    n_rh = int(rh_mars.shape[0])
    if n_lh < n_seed or n_rh < n_seed:
        raise ValueError(
            f"MARS_label arrays ({n_lh}, {n_rh}) shorter than the "
            f"{n_seed}-vertex seed mesh {seed_mesh!r}")
    lh_seed = np.flatnonzero(lh_mars[:n_seed] == 2).astype(np.int64)
    rh_seed = (np.flatnonzero(rh_mars[:n_seed] == 2).astype(np.int64)
               + np.int64(n_lh))
    seed_idx = np.concatenate([lh_seed, rh_seed])
    mw = np.concatenate([(lh_mars == 1), (rh_mars == 1)]).astype(np.uint8)
    return seed_idx, mw, n_lh, n_rh


# ─────────────────────────────────────────────────────────────────────
# The fused compute
# ─────────────────────────────────────────────────────────────────────
def _as_runs(item) -> List["cp.ndarray"]:
    """One session's device buffers, whether handed over singly or as a
    list of runs. A bare ``cupy.ndarray`` is a single-run session --
    exactly what ``iter_subject_bold_gpu`` yields."""
    if isinstance(item, cp.ndarray):
        return [item]
    runs = list(item)
    if not runs:
        raise ValueError("compute_subject_profiles_gpu: session has 0 runs")
    for r in runs:
        if not isinstance(r, cp.ndarray):
            raise TypeError(
                "compute_subject_profiles_gpu: runs must be cupy.ndarray "
                f"device buffers; got {type(r).__name__}")
    return runs


# How many sessions' raw device buffers may be alive at once. 2 is
# enough to keep the compute stream from ever waiting on the retire
# event: the event popped is from the session before last, which
# finished while the current one was being gathered.
_HOLD_DEPTH = 2


def compute_subject_profiles_gpu(sessions: Iterable,
                                  *,
                                  n_sess: int,
                                  n_full: int,
                                  seed_idx,
                                  mw_mask,
                                  threshold,
                                  censor: Optional[Dict] = None,
                                  stream=None,
                                  out: Optional[np.ndarray] = None,
                                  on_session=None):
    """Fused per-subject FC-profile compute, one D2H for the subject.

    Parameters
    ----------
    sessions : iterable of ``(sess_index, runs)``
        ``sess_index`` is the 0-based slot in the output's first axis;
        ``runs`` is either one ``(T, n_full) fp32`` C-contiguous device
        buffer (single-run session -- what
        :func:`~arealmshbm.data_io.gifti_bold_gpu.iter_subject_bold_gpu`
        with ``raw_sessions=True`` yields) or a sequence of them, one
        per run. Consumed lazily, so a generator lets ingest of session
        ``g+1`` overlap the compute of session ``g``.
        Columns ``[0, n_lh)`` are lh, ``[n_lh, n_full)`` rh.
    n_sess : int
        Number of sessions, i.e. the packed buffer's first-axis size.
    n_full : int
        ``n_lh + n_rh``.
    seed_idx : array-like of int
        Global seed column indices (``subject_seed_and_mw``). Host or
        device; uploaded once.
    mw_mask : array-like
        ``(n_full,)`` truthy at medial-wall vertices. Host or device.
    threshold : float or str
        Top fraction admitted by the binarize (production: ``0.1``).
    censor : dict, optional
        ``{sess_index: [keep_int32_or_None, ...]}`` -- one entry per
        run, matching ``profiles._SessionInputs.censor_runs``. A ``1``
        keeps the timepoint.
    stream : cupy.cuda.Stream, optional
        Stream every kernel here is issued on. Defaults to the current
        stream. Pass a non-blocking stream distinct from the ingest's
        decode stream to get real overlap.
    out : numpy.ndarray, optional
        Destination for the single D2H: ``(n_sess, n_full, ceil(K/8))``
        uint8, C-contiguous. Pass pinned memory to halve the copy;
        :func:`generate_subject_profiles_gpu` does. Omitted, a plain
        ``np.empty`` is allocated, which is always safe to keep.
    on_session : callable, optional
        ``on_session(sess_index, slab, event)``, called on THIS thread
        the moment session ``sess_index``'s pack is issued. ``slab`` is
        the ``(n_full, ceil(K/8))`` view of ``out`` that session's D2H
        was just enqueued into and ``event`` is a
        ``cupy.cuda.Event`` recorded after that copy -- so the callee
        must ``event.synchronize()`` (on some OTHER thread; doing it
        here would serialise the very pipeline this exists for) before
        reading ``slab``.

        Requires ``out``. With a callback the subject leaves the GPU as
        ``n_sess`` per-session copies into ``out`` instead of one
        whole-subject copy at the end; ``out`` is the same fully
        populated array either way by the time this returns (the last
        copy is joined before it does).

    Returns
    -------
    (packed_host, K)
        ``packed_host`` is ``(n_sess, n_full, ceil(K/8)) uint8`` --
        session ``t`` is exactly the slab
        ``SubjectProfileStreamWriter(D_unpacked=K).write_session(t, .)``
        takes (``out`` itself when one was given).
    """
    n_sess = int(n_sess)
    n_full = int(n_full)
    if n_sess <= 0 or n_full <= 0:
        raise ValueError(
            f"compute_subject_profiles_gpu: n_sess={n_sess}, "
            f"n_full={n_full} must both be positive")
    threshold_f = float(threshold)

    seed_idx_d = cp.ascontiguousarray(cp.asarray(seed_idx).ravel(),
                                      dtype=cp.int64)
    K = int(seed_idx_d.size)
    if K <= 0:
        raise ValueError("compute_subject_profiles_gpu: empty seed index")
    mw_d = cp.ascontiguousarray(
        (cp.asarray(mw_mask).ravel() != 0).astype(cp.uint8))
    if int(mw_d.size) != n_full:
        raise ValueError(
            f"compute_subject_profiles_gpu: mw_mask length {int(mw_d.size)}"
            f" != n_full {n_full}")
    D_bytes = (K + 7) // 8

    shape = (n_sess, n_full, D_bytes)
    if out is None:
        if on_session is not None:
            raise ValueError(
                "compute_subject_profiles_gpu: on_session needs out= "
                "(the callback hands out slab views of it)")
        packed_host = np.empty(shape, dtype=np.uint8)
    else:
        packed_host = out
        if (packed_host.shape != shape
                or packed_host.dtype != np.uint8
                or not packed_host.flags["C_CONTIGUOUS"]):
            raise ValueError(
                f"compute_subject_profiles_gpu: out must be a "
                f"C-contiguous {shape} uint8 array; got "
                f"{packed_host.shape} {packed_host.dtype}")

    ctx = stream if stream is not None else cp.cuda.Stream.null
    with ctx:
        packed_dev = cp.empty((n_sess, n_full, D_bytes), dtype=cp.uint8)
        # A session's buffers must outlive the kernels that read them.
        # They were allocated on the INGEST's decode stream, so dropping
        # the last reference returns them to that stream's free list and
        # the ingest is then free to hand the same bytes to its next
        # decode -- while our zscore/gemm on the compute stream is still
        # reading them. So each session's buffers are retired behind an
        # event recorded after its pack, and at most ``_HOLD_DEPTH``
        # sessions are retained: device memory is O(1) in ``n_sess``,
        # not O(n_sess).
        held: "deque" = deque()          # [(event, [buf, ...])]
        seen = 0
        try:
            for sess_index, item in sessions:
                si = int(sess_index)
                if not (0 <= si < n_sess):
                    raise ValueError(
                        f"compute_subject_profiles_gpu: sess_index {si} "
                        f"outside [0, {n_sess})")
                runs = _as_runs(item)
                n_runs = len(runs)
                corr_sum = None
                for r, buf in enumerate(runs):
                    if buf.ndim != 2 or int(buf.shape[1]) != n_full:
                        raise ValueError(
                            f"session {si} run {r}: expected "
                            f"(T, {n_full}); got {buf.shape}")
                    if buf.dtype != cp.float32:
                        raise ValueError(
                            f"session {si} run {r}: expected fp32; got "
                            f"{buf.dtype}")
                    if not buf.flags["C_CONTIGUOUS"]:
                        raise ValueError(
                            f"session {si} run {r}: buffer must be "
                            f"C-contiguous")
                    keep = None
                    if censor is not None:
                        per_run = censor.get(si) if hasattr(censor, "get") \
                            else censor[si]
                        if per_run is not None and per_run[r] is not None:
                            keep = np.asarray(per_run[r]) == 1
                            if keep.shape[0] != int(buf.shape[0]):
                                raise ValueError(
                                    f"session {si} run {r}: censor "
                                    f"length {keep.shape[0]} != "
                                    f"T={int(buf.shape[0])}")
                            buf = cp.ascontiguousarray(buf[cp.asarray(keep)],
                                                       dtype=cp.float32)
                    norm = cp.empty_like(buf)
                    zscore_unit_norm_columns_zerovar_cupy(buf, norm)
                    # (T, K) gather out of the already-normalised array --
                    # bit-identical to normalising the gathered copy.
                    s_norm = norm[:, seed_idx_d]
                    corr = s_norm.T @ norm          # (K, T) @ (T, n_full)
                    del s_norm, norm
                    if corr_sum is None:
                        corr_sum = corr
                    else:
                        corr_sum += corr
                        del corr
                if n_runs != 1:
                    corr_sum *= cp.float32(1.0 / n_runs)
                t = threshold_top_fraction_exact_cupy(corr_sum, threshold_f)
                binarize_mwzero_pack_cupy(corr_sum, mw_d, float(t),
                                          packed_dev[si])
                del corr_sum
                # D2H of just this session's packed slab, straight into
                # its final place in ``out``: the per-session copy IS the
                # whole-subject array, so nothing is copied twice. Async
                # on the compute stream, and the event recorded right
                # after it is what the .b2nd writer thread waits on --
                # never this thread, which has the next session to issue.
                slab = None
                if on_session is not None:
                    slab = packed_host[si]
                    packed_dev[si].get(out=slab, stream=ctx, blocking=False)
                    # block=True so the writer thread's synchronize()
                    # parks in the driver instead of spinning a core
                    # while it is behind; cupy's default
                    # (Stream.record()'s own Event) busy-waits, which
                    # would steal a core from the GIFTI decode.
                    evt = cp.cuda.Event(block=True, disable_timing=True)
                    ctx.record(evt)
                    on_session(si, slab, evt)
                else:
                    evt = ctx.record()
                # One event, two waiters: the writer above and the
                # raw-buffer retire below.
                held.append((evt, runs))
                del runs, slab
                while len(held) > _HOLD_DEPTH:
                    evt, old_runs = held.popleft()
                    evt.synchronize()        # ~0: it is _HOLD_DEPTH behind
                    del evt, old_runs
                seen += 1
            if seen != n_sess:
                raise ValueError(
                    f"compute_subject_profiles_gpu: consumed {seen} sessions "
                    f"but n_sess={n_sess}")

            if on_session is None:
                packed_dev.get(out=packed_host)   # blocking; retires the rest
            else:
                # Every slab was already copied; joining the stream is
                # what makes the LAST one readable (and retires the held
                # raw buffers), and by then it is the only one in flight.
                ctx.synchronize()
            held.clear()
        except BaseException:
            # Kernels reading the held session buffers, and with
            # ``on_session`` the per-session D2H into ``out`` the
            # caller is about to recycle, may still be in flight.
            # Drain the stream before the exception escapes -- on every
            # path, so a retired buffer is never handed back to the
            # ingest while a kernel still reads it.
            try:
                ctx.synchronize()
            except BaseException:           # pragma: no cover - device
                pass
            raise
    del packed_dev
    return packed_host, K


# ─────────────────────────────────────────────────────────────────────
# Subject orchestration: ingest + compute + background write
# ─────────────────────────────────────────────────────────────────────
class SubjectProfilesResult:
    """Handle returned by :func:`generate_subject_profiles_gpu`.

    ``.packed`` is the ``(n_sess, N, ceil(K/8)) uint8`` host array,
    ``.K`` the unpacked seed count, ``.out_path`` the .b2nd that
    ``.wait()`` joins on (``None`` when no write was asked for), and
    ``.ingest`` which decode fed the compute -- ``'nvcomp'`` or
    ``'cpu-reader'`` -- for the runner's per-subject line; nothing in
    ``.packed`` depends on it.

    ``.packed`` lives in a pooled **pinned** block that ``.wait()``
    hands back, so it is valid until you call ``.wait()`` and then
    start another subject; copy it if you need it beyond that. Calling
    ``.wait()`` is what makes the next subject's D2H safe -- it joins
    the writer that is reading this block before releasing it. It is
    idempotent on success, so calling it twice is fine; on a failed
    write every call re-raises the writer's exception.

    Because the .b2nd is written chunk-by-chunk as the sessions land,
    by the time you get this handle the writer is normally down to the
    LAST session, so ``.wait()`` owes one chunk rather than the whole
    subject.

    **Call ``.wait()`` (or use the result as a context manager).** It
    is the only thing that re-raises a .b2nd write failure into your
    call stack. A result that is dropped without it still cannot lose
    the error silently -- the writer's exception is reported through
    :mod:`warnings` the moment the write fails, and again if the
    handle is garbage-collected unconsumed -- but a warning is not a
    traceback, and the on-disk file is missing or truncated either way.
    """

    __slots__ = ("packed", "K", "out_path", "ingest", "_future", "_token",
                 "_exc")

    def __init__(self, packed, K: int, out_path, future: Optional[Future],
                 token=None, *, ingest: str):
        self.packed = packed
        self.K = int(K)
        self.out_path = out_path
        self.ingest = str(ingest)
        self._future = future
        self._token = token
        self._exc: Optional[BaseException] = None
        if future is not None:
            future.add_done_callback(_warn_if_write_failed)

    def wait(self):
        """Join the background .b2nd write, then release the pinned
        block ``.packed`` borrows. The writer's exception is latched and
        re-raised by EVERY call, not only the first: a late joiner must
        not read a failed write as a finished one."""
        try:
            if self._future is not None:
                fut, self._future = self._future, None
                try:
                    fut.result()
                except BaseException as exc:
                    self._exc = exc
                    raise
            elif self._exc is not None:
                raise self._exc
        finally:
            _release_pinned(self._token)
            self._token = None
        return self.out_path

    def __del__(self):                  # pragma: no cover - GC timing
        fut = getattr(self, "_future", None)
        if fut is None:
            return
        try:
            if fut.done() and fut.exception() is not None:
                warnings.warn(
                    "SubjectProfilesResult for "
                    f"{getattr(self, 'out_path', None)} was discarded "
                    "without .wait(); its .b2nd write had FAILED: "
                    f"{fut.exception()!r}",
                    RuntimeWarning, stacklevel=1)
            elif not fut.done():
                warnings.warn(
                    "SubjectProfilesResult for "
                    f"{getattr(self, 'out_path', None)} was discarded "
                    "while its .b2nd write was still running; call "
                    ".wait() (or use it as a context manager) so write "
                    "errors are raised and the pinned block is reused.",
                    RuntimeWarning, stacklevel=1)
        except BaseException:           # interpreter shutdown
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wait()
        return False


def _warn_if_write_failed(fut: Future) -> None:
    """Make a background .b2nd failure loud the instant it happens.

    ``.wait()`` re-raises properly, but a caller that never calls it
    would otherwise drop the exception on the floor when the Future is
    collected. This turns that into a ``RuntimeWarning`` at the moment
    of failure, which pytest turns into an error and a pipeline run
    prints.
    """
    try:
        exc = fut.exception()
    except BaseException:               # cancelled / shutting down
        return
    if exc is not None:
        warnings.warn(f"background .b2nd write failed: {exc!r}",
                      RuntimeWarning, stacklevel=1)


_INGEST_QUEUE_DEPTH = 2


def _threaded(make_iter, device_id: int, depth: int = _INGEST_QUEUE_DEPTH):
    """Run ``make_iter()`` on a producer thread, yield what it yields.

    The consumer of the ingest generator is the compute loop, and that
    loop blocks the host three times per session inside the radix
    select's histogram read-backs. Pumping the generator from the SAME
    thread therefore stalls the ingest's host-side prep (file reads,
    tag bookkeeping) for exactly as long as the compute is waiting on
    the GPU. CUDA's blocking syncs release the GIL, so a producer
    thread runs straight through them.

    ``depth`` bounds the queue so the ingest's own in-flight decode
    buffers do not pile up. It does NOT bound the sessions the compute
    loop is still holding -- that is the compute loop's job (see the
    ``held`` deque in :func:`compute_subject_profiles_gpu`).

    Abandonment
    -----------
    The consumer can leave early -- a shape/dtype/censor check in
    :func:`compute_subject_profiles_gpu` raises, or the caller simply
    stops iterating and the generator is closed by GC. The producer is
    then almost certainly parked in ``q.put`` on a full queue, so the
    ``finally`` must DRAIN the queue before joining; joining first
    deadlocks the process (and, because a generator's ``close()`` runs
    from ``gc``, it deadlocks at an arbitrary later point). ``stop``
    tells the producer to quit at its next item; the drain then lets it
    reach its own ``finally`` and post the sentinel.
    """
    import queue as _queue

    q: "_queue.Queue" = _queue.Queue(maxsize=max(1, int(depth)))
    sentinel = object()
    box: List = []
    stop = threading.Event()

    def _run():
        try:
            cp.cuda.Device(device_id).use()
            for item in make_iter():
                if stop.is_set():
                    break
                q.put(item)
        except BaseException as exc:      # re-raised on the consumer
            box.append(exc)
        finally:
            q.put(sentinel)

    th = threading.Thread(target=_run, name="profile-ingest", daemon=True)
    th.start()
    saw_sentinel = False
    try:
        while True:
            item = q.get()
            if item is sentinel:
                saw_sentinel = True
                break
            yield item
        if box:
            raise box[0]
    finally:
        stop.set()
        if not saw_sentinel:
            # Producer may be blocked in q.put(); drain until its
            # sentinel arrives so it can finish. Guaranteed to
            # terminate: ``stop`` makes it break out at the next item
            # and its ``finally`` always posts the sentinel.
            while q.get() is not sentinel:
                pass
        th.join()


_WRITE_POOL: Optional[ThreadPoolExecutor] = None
_WRITE_POOL_LOCK = threading.Lock()


def _write_pool() -> ThreadPoolExecutor:
    global _WRITE_POOL
    with _WRITE_POOL_LOCK:
        if _WRITE_POOL is None:
            _WRITE_POOL = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="profile-b2nd-write")
        return _WRITE_POOL


# Queue sentinels for _stream_write_job.
_WRITE_DONE = object()
_WRITE_ABORT = object()


def _stream_write_job(out_path, T: int, N: int, D_unpacked: int,
                      q, stop, device_id: int):
    """Own one subject's .b2nd from create to close, one chunk at a time.

    Runs on the single-worker :func:`_write_pool` thread. Items arrive
    as ``(sess_index, slab, event)``; ``event`` was recorded on the
    compute stream after that slab's D2H, so waiting on it here is what
    keeps ``evt.synchronize()`` off the compute thread. The queue is
    unbounded (``T`` items, each just a view into the producer's pinned
    block), so the producer can NEVER block on this thread -- that is
    what makes a mid-subject compute failure impossible to deadlock.

    Termination is guaranteed: every path out of
    :func:`generate_subject_profiles_gpu` past the ``submit`` posts a
    :data:`_WRITE_DONE` / :data:`_WRITE_ABORT` sentinel -- including a
    ``submit`` that raised after queueing the job -- and this loop
    returns on the first one it sees, so a second sentinel is harmless.
    ``stop`` short-circuits the compression of whatever is still queued
    behind a failure; the abort is on its way regardless.
    """
    from ..data_io.profile_io import SubjectProfileStreamWriter

    cp.cuda.Device(int(device_id)).use()
    w = None
    try:
        # Eager: the frame create + vlmeta stamp happens while this
        # thread still has the whole of session 0's compute ahead of it.
        w = SubjectProfileStreamWriter(out_path, T=T, N=N,
                                       D_unpacked=D_unpacked)
        while True:
            item = q.get()
            if item is _WRITE_DONE:
                return w.close()
            if item is _WRITE_ABORT:
                w.abort()
                return None
            if stop.is_set():
                continue                # drain; the abort is coming
            si, slab, evt = item
            evt.synchronize()
            w.write_session(si, slab)
            del item, slab, evt         # release the pinned view
    except BaseException:
        if w is not None:
            w.abort()               # no half-written .b2nd left behind
        raise


_RE_N_DARRAYS = re.compile(rb'NumberOfDataArrays="(\d+)"')
_HEADER_PROBE_BYTES = 8192

# Ingest pipeline depth forwarded to ``iter_subject_bold_gpu``: one
# batched nvCOMP decode per 2 sessions (1 = no overlap, 3+ = larger
# staging for no gain).
_INGEST_GROUP_SESSIONS = 2


def _t_hint(path) -> Optional[int]:
    """``NumberOfDataArrays`` off the first 8 KB of a .func.gii, or None.

    GIFTI puts the attribute on the root element, so it lands in the
    first ~200 bytes of every file this pipeline sees; the probe is two
    short reads per session. It is a **hint only**: the reader recounts ``<DataArray>`` occurrences and
    its own count is what any error message quotes. A wrong hint can
    only cost a pipeline bubble (an unnecessary block split) or produce
    the reader's own loud T-mismatch error -- never a silent misread.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(_HEADER_PROBE_BYTES)
    except OSError:
        return None                     # let the reader report it
    m = _RE_N_DARRAYS.search(head)
    return int(m.group(1)) if m else None


def _t_blocks(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[int, int]]:
    """Split ``pairs`` into maximal ``[lo, hi)`` runs of equal-T pairs.

    ``read_subject_bold_gpu`` requires every file of ONE call to share
    ``T`` (it decodes a whole group into one ``(T, n_full)`` buffer per
    session and checks T across groups). Real cohorts have sessions of
    different length, so the subject is driven as a chain of calls, one
    per equal-T block -- one block, and therefore one call, whenever
    the subject is uniform, which is the production case.

    A pair whose two hemispheres disagree, or whose header could not be
    probed, becomes its own singleton block; the reader then raises its
    own authoritative error for the genuinely-invalid ones.
    """
    hints: List[Optional[int]] = []
    for lh, rh in pairs:
        a, b = _t_hint(lh), _t_hint(rh)
        hints.append(a if (a is not None and a == b) else None)
    blocks: List[Tuple[int, int]] = []
    lo = 0
    for i in range(1, len(hints) + 1):
        if (i == len(hints) or hints[i] is None or hints[i - 1] is None
                or hints[i] != hints[i - 1]):
            blocks.append((lo, i))
            lo = i
    return blocks


# CPU-reader ingest: 8 decode workers (the isal_zlib saturation
# plateau, same as ``Step1BoldPrefetcher``) and up to 4 pairs submitted
# ahead of the one being consumed -- ramped up from one, so the first
# pair has the pool to itself and the compute starts after one file's
# decode rather than after four pairs contending for it.
_HOST_INGEST_WORKERS = 8
_HOST_INGEST_PAIRS_AHEAD = 4


def _host_ingest_sessions(pairs: Sequence[Tuple[str, str]],
                          n_lh: int, n_rh: int):
    """``(pair_index, (T, n_lh + n_rh) fp32 device buffer)``, in order.

    The stand-in for
    :func:`~arealmshbm.data_io.gifti_bold_gpu.iter_subject_bold_gpu`
    with ``raw_sessions=True`` when nvCOMP is absent, and contract-
    identical to it: no NaN cleaning, no medial drop, lh in columns
    ``[0, n_lh)``, and the same accepted input -- the reader runs with
    ``allow_wrapped_b64=False``, so a line-wrapped payload is refused
    here exactly as the device reader refuses it, rather than read on
    one machine and rejected on another. The bytes are identical too:
    ``read_subject_bold_gpu(raw_sessions=True)`` is pinned equal to the
    CPU reader's time-major output in
    ``data_io/tests/test_gifti_readers.py``, so the compute cannot tell
    the two ingests apart.

    Both hemispheres of up to :data:`_HOST_INGEST_PAIRS_AHEAD` pairs are
    submitted ahead of the pair being assembled; the reader's base64 +
    isal_zlib decode releases the GIL, so those workers overlap the
    caller's GPU compute. The pair is assembled into a pooled pinned
    block (:func:`_acquire_pinned`, so the H2D is a true async copy and
    the pages are not faulted in per session) and released once the
    copy is synchronised -- before the yield, since the consumer reads
    the buffer from its own compute stream, which has no ordering
    dependency on this thread's. Same contract as
    ``iter_subject_bold_gpu``: what is yielded is already there, usable
    from any stream.

    Runs on the :func:`_threaded` producer thread, which has already
    pinned the device; the H2D stream is that thread's cached one
    (:func:`_thread_stream`), not a fresh stream per subject.
    """
    n_full = int(n_lh) + int(n_rh)
    h2d = _thread_stream()
    with ThreadPoolExecutor(max_workers=_HOST_INGEST_WORKERS,
                            thread_name_prefix="profile-gifti") as pool:
        inflight: "deque" = deque()
        nxt = 0
        ahead = 1
        try:
            while nxt < len(pairs) or inflight:
                while nxt < len(pairs) and len(inflight) < ahead:
                    lh_p, rh_p = pairs[nxt]
                    inflight.append((
                        nxt, lh_p, rh_p,
                        pool.submit(read_surface_gifti, lh_p,
                                    time_major=True, allow_wrapped_b64=False),
                        pool.submit(read_surface_gifti, rh_p,
                                    time_major=True, allow_wrapped_b64=False)))
                    nxt += 1
                ahead = min(ahead + 1, _HOST_INGEST_PAIRS_AHEAD)
                i, lh_p, rh_p, lh_fut, rh_fut = inflight.popleft()
                lh = lh_fut.result()            # (T, n_lh) fp32 host
                rh = rh_fut.result()            # (T, n_rh) fp32 host
                if int(lh.shape[1]) != int(n_lh):
                    raise ValueError(
                        f"{lh_p}: Dim0 {int(lh.shape[1])} != n_lh "
                        f"{int(n_lh)}")
                if int(rh.shape[1]) != int(n_rh):
                    raise ValueError(
                        f"{rh_p}: Dim0 {int(rh.shape[1])} != n_rh "
                        f"{int(n_rh)}")
                if int(lh.shape[0]) != int(rh.shape[0]):
                    raise ValueError(
                        f"lh/rh time-axis mismatch: {lh_p} has "
                        f"T={int(lh.shape[0])}, {rh_p} has "
                        f"T={int(rh.shape[0])}")
                T = int(lh.shape[0])
                flat, token = _acquire_pinned(T * n_full * 4)
                try:
                    buf = flat.view(np.float32).reshape(T, n_full)
                    buf[:, :n_lh] = lh
                    buf[:, n_lh:] = rh
                    del lh, rh
                    with h2d:
                        dev = cp.empty((T, n_full), dtype=cp.float32)
                        dev.set(buf)
                    h2d.synchronize()
                    del buf, flat
                finally:
                    _release_pinned(token)
                yield i, dev
                del dev
        finally:
            # Abandoned early (a consumer-side raise): un-started reads
            # are dropped rather than paid for; the pool's shutdown
            # still joins whatever is already running.
            for _i, _lp, _rp, a, b in inflight:
                a.cancel()
                b.cancel()


def _session_censor(project_dir: Path, sub: str, sess: str, n_runs: int):
    """``profiles._load_session_inputs``' censor logic, verbatim."""
    p = project_dir / "data_list" / "censor_list" / f"sub{sub}_sess{sess}.txt"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content or content.upper() == "NONE":
        return None
    outlier_paths = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if len(outlier_paths) != n_runs:
        raise ValueError(
            f"censor list has {len(outlier_paths)} runs, BOLD has {n_runs}")
    return [_read_censor_vector(Path(q)) if q.upper() != "NONE" else None
            for q in outlier_paths]


def generate_subject_profiles_gpu(project_dir,
                                   sub_id: str,
                                   sess_ids: Sequence[str],
                                   bold_paths: Optional[Dict] = None,
                                   *,
                                   out_path=None,
                                   seed_mesh: str,
                                   targ_mesh: str,
                                   threshold=0.1,
                                   split_flag: str = "0",
                                   write: bool = True) -> SubjectProfilesResult:
    """Ingest + fused compute + .b2nd write for one subject.

    Parameters
    ----------
    project_dir : path
        Project root; used for ``data_list/fMRI_list`` (when
        ``bold_paths`` is omitted), ``data_list/censor_list`` and the
        default ``out_path``.
    sub_id : str
        Subject id as it appears in the list filenames.
    sess_ids : sequence of str
        Session ids, in output order (axis 0 of the .b2nd).
    bold_paths : dict, optional
        ``{sess_id: (lh_paths, rh_paths)}``. Read from
        ``data_list/fMRI_list`` when omitted.
    out_path : path, optional
        Target .b2nd. Defaults to
        ``profile_io.profile_path(project_dir, sub_id, targ, seed)``.
    threshold : float
        Top fraction admitted by the binarize.
    split_flag : str
        Only ``'0'`` is supported, as on the per-session path.
    write : bool
        Stream the .b2nd out on a background thread, one chunk per
        session, as the sessions finish. ``False`` skips it entirely
        (no file, no writer thread, and the subject leaves the GPU in
        one D2H at the end) though ``.out_path`` is still reported. A
        failure of that write is raised by ``.wait()``; a result dropped
        without ``.wait()`` reports it as a ``RuntimeWarning`` instead.

    Notes
    -----
    Ingest is chosen here, once, from
    :func:`~arealmshbm.data_io._nvcomp_batched.nvcomp_available`: the
    batched device decode when nvCOMP is installed, the CPU GIFTI
    reader (:func:`_host_ingest_sessions`) otherwise. Same accepted
    files, same bytes, same compute, same artifacts -- only where the
    DEFLATE runs, which the result reports as ``.ingest`` and the
    runner prints with the subject's timing.

    Runs may differ in length. ``read_subject_bold_gpu`` requires one
    ``T`` per call, so on the nvCOMP ingest the subject is driven as a
    chain of reader calls split on the ``NumberOfDataArrays`` header
    hint (:func:`_t_blocks`) -- a single call, and therefore a single
    uninterrupted ingest pipeline, whenever every run has the same
    ``T``, which is the production case. Only the block boundaries cost
    a pipeline bubble.

    Every kernel is issued on a module-owned non-blocking compute
    stream (:func:`_thread_stream`) so the ingest's decode stream can
    run ahead.

    Returns
    -------
    SubjectProfilesResult
        ``.packed``, ``.K``, ``.out_path``, ``.wait()``.
    """
    import queue as _queue

    from ..data_io import _nvcomp_batched
    from ..data_io.profile_io import profile_path

    if str(split_flag) != "0":
        raise NotImplementedError(
            f"split_flag={split_flag!r}: only '0' (no-split) is supported.")
    project_dir = Path(project_dir)
    sub = str(sub_id)
    sess_ids = [str(s) for s in sess_ids]
    if not sess_ids:
        raise ValueError("generate_subject_profiles_gpu: no sessions")

    if bold_paths is None:
        bold_paths = {}
        for sess in sess_ids:
            lh_lp = (project_dir / "data_list" / "fMRI_list"
                     / f"lh_sub{sub}_sess{sess}.txt")
            rh_lp = (project_dir / "data_list" / "fMRI_list"
                     / f"rh_sub{sub}_sess{sess}.txt")
            if not lh_lp.exists() or not rh_lp.exists():
                raise FileNotFoundError(
                    f"missing fMRI lists: {lh_lp} or {rh_lp}")
            bold_paths[sess] = (_read_lines(lh_lp), _read_lines(rh_lp))

    seed_idx, mw, n_lh, n_rh = subject_seed_and_mw(seed_mesh, targ_mesh)
    n_full = n_lh + n_rh

    # Flatten (session, run) into the reader's pair list; the reader
    # calls each pair a "session", which is exactly one run to us.
    pairs: List[Tuple[str, str]] = []
    run_counts: List[int] = []
    censor: Dict[int, object] = {}
    for si, sess in enumerate(sess_ids):
        lh_paths, rh_paths = bold_paths[sess]
        if len(lh_paths) != len(rh_paths):
            raise ValueError(
                f"sub{sub} sess{sess}: lh_runs ({len(lh_paths)}) != rh_runs "
                f"({len(rh_paths)})")
        if not lh_paths:
            raise ValueError(f"sub{sub} sess{sess}: empty fMRI list")
        pairs.extend(zip(lh_paths, rh_paths))
        run_counts.append(len(lh_paths))
        c = _session_censor(project_dir, sub, sess, len(lh_paths))
        if c is not None:
            censor[si] = c

    compute_stream = _thread_stream()

    dev_id = int(cp.cuda.runtime.getDevice())

    if _nvcomp_batched.nvcomp_available():
        ingest = "nvcomp"
        from ..data_io.gifti_bold_gpu import iter_subject_bold_gpu
        blocks = _t_blocks(pairs)

        def _flat():
            # One reader call per equal-T block (one call for a uniform
            # subject). Chaining keeps pair order, which is all _grouped
            # relies on -- the per-call session index is ignored.
            for lo, hi in blocks:
                for item in iter_subject_bold_gpu(
                        pairs[lo:hi], n_lh=n_lh, n_rh=n_rh,
                        raw_sessions=True,
                        group_sessions=_INGEST_GROUP_SESSIONS):
                    yield item
    else:
        ingest = "cpu-reader"

        def _flat():
            return _host_ingest_sessions(pairs, n_lh, n_rh)

    def _grouped():
        """Regroup the reader's flat per-run stream into sessions."""
        si = 0
        buf: List = []
        for _idx, arr in _threaded(_flat, dev_id):
            buf.append(arr)
            if len(buf) == run_counts[si]:
                yield si, (buf[0] if len(buf) == 1 else buf)
                si += 1
                buf = []
        if buf or si != len(run_counts):
            raise ValueError(
                f"generate_subject_profiles_gpu: ingest yielded a partial "
                f"subject ({si} of {len(run_counts)} sessions)")

    if out_path is None:
        out_path = profile_path(project_dir, sub, targ_mesh, seed_mesh)
    out_path = Path(out_path)

    # Pinned destination for the subject's D2H. The pool hands back a
    # block no live writer is reading (see _acquire_pinned).
    n_sess = len(sess_ids)
    K_hint = int(seed_idx.size)
    D_bytes = (K_hint + 7) // 8
    flat, token = _acquire_pinned(n_sess * n_full * D_bytes)
    packed_out = flat.reshape(n_sess, n_full, D_bytes)

    # The .b2nd is written a session at a time, on the writer thread,
    # while this thread computes the next one -- so the join at the end
    # waits for one chunk instead of the whole subject, which a single
    # subject has nothing to overlap.
    q = stop = on_session = fut = None
    if write:
        q = _queue.SimpleQueue()
        stop = threading.Event()
        try:
            fut = _write_pool().submit(
                _stream_write_job, out_path, n_sess, n_full, K_hint,
                q, stop, dev_id)
        except BaseException:
            # ``submit`` queues the work item BEFORE it starts the pool
            # thread, so a KeyboardInterrupt inside it can leave the job
            # running with no ``fut`` to steer it. Post the abort first:
            # the job returns on the first sentinel it sees, and an
            # extra sentinel nobody reads is harmless.
            q.put(_WRITE_ABORT)
            _release_pinned(token)
            raise

        def _enqueue(si, slab, evt, _put=q.put):
            _put((si, slab, evt))

        on_session = _enqueue

    try:
        packed, K = compute_subject_profiles_gpu(
            _grouped(), n_sess=n_sess, n_full=n_full,
            seed_idx=seed_idx, mw_mask=mw, threshold=threshold,
            censor=censor or None, stream=compute_stream,
            out=packed_out, on_session=on_session)
        if fut is not None:
            # Inside the try so the except below still owns every path
            # between the compute call and the sentinel.
            q.put(_WRITE_DONE)
    except BaseException:
        if fut is not None:
            # Stop the writer BEFORE the pinned block is recycled: it
            # holds slab views of it. The queue is unbounded, so the
            # sentinel is always accepted and the job always ends.
            stop.set()
            q.put(_WRITE_ABORT)
            try:
                fut.result()
            except BaseException:
                pass                    # the compute error is the story
        _release_pinned(token)
        raise
    return SubjectProfilesResult(packed, K, out_path, fut, token,
                                 ingest=ingest)


# ─────────────────────────────────────────────────────────────────────
# Prewarm
# ─────────────────────────────────────────────────────────────────────
_STREAM_TLS = threading.local()


def _thread_stream():
    """One cached non-blocking stream per thread: the compute loop's on
    the consumer thread, the CPU-reader ingest's H2D on the producer
    thread.

    Non-blocking so the ingest's decode stream is free to run group
    ``g+1`` while the compute chews on group ``g``; per-thread so two
    subjects driven from two threads do not interleave on one stream;
    cached because CuPy's memory pool keys its free lists by the
    allocating stream, so a stream created per call would strand each
    subject's buffers in an arena nothing reuses.
    """
    s = getattr(_STREAM_TLS, "stream", None)
    if s is None:
        s = cp.cuda.Stream(non_blocking=True)
        _STREAM_TLS.stream = s
    return s


_PREWARM_LOCK = threading.Lock()
_PREWARM_DONE = False


def prewarm_generate_profiles_gpu(bold_paths=None) -> None:
    """Pay every first-call cost of this module off the timed path.

    Compiles the three RawKernels (zscore-zerovar, radix histogram,
    binarize+pack), creates the cuBLAS handle, imports blosc2 for the
    writer, and -- when nvCOMP is what the ingest will use -- forwards
    to
    :func:`~arealmshbm.data_io.gifti_bold_gpu.prewarm_subject_bold_gpu`
    (nvCOMP library load, GIFTI kernels, pinned staging). The CPU
    reader has no device-side first-call cost, so on that ingest there
    is nothing to warm.

    Idempotent and thread-safe; safe to run on a daemon thread during
    pipeline init. ``bold_paths`` is an optional sequence of
    ``(lh, rh)`` pairs handed to the ingest prewarm so it can size its
    pinned staging buffer for the real files.
    """
    global _PREWARM_DONE
    with _PREWARM_LOCK:
        if _PREWARM_DONE:
            return
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            # RawKernel NVRTC compile + module load. Shapes are tiny;
            # only the cubin matters.
            x = cp.zeros((8, 64), dtype=cp.float32)
            y = cp.empty_like(x)
            zscore_unit_norm_columns_zerovar_cupy(x, y)
            threshold_top_fraction_exact_cupy(y, 0.1)
            mw = cp.zeros(64, dtype=cp.uint8)
            packed = cp.empty((64, 1), dtype=cp.uint8)
            binarize_mwzero_pack_cupy(
                cp.zeros((8, 64), dtype=cp.float32), mw, 0.5, packed)
        stream.synchronize()
        # cupy keeps one cuBLAS handle per (device, thread), so this
        # creates the handle for whichever thread runs the prewarm — but
        # the kernel-module load the gemm triggers is process-wide, and
        # that is the expensive part: it drops the profile loop's first
        # gemm from ~60 ms to <1 ms even when the prewarm ran on the
        # daemon thread.
        cp.cuda.device.get_cublas_handle()
        g = cp.zeros((32, 32), dtype=cp.float32)
        cp.matmul(g, g)
        cp.cuda.get_current_stream().synchronize()
        del x, y, mw, packed, g
        try:
            from ..data_io.profile_io import _blosc2
            _blosc2()
        except ImportError:
            pass
        from ..data_io import _nvcomp_batched
        if _nvcomp_batched.nvcomp_available():
            from ..data_io.gifti_bold_gpu import prewarm_subject_bold_gpu
            # Without paths we cannot size the pinned staging buffer,
            # so ask for a token 1 MB -- the NVRTC compile, the nvCOMP
            # ctypes load and the first cupy reduction are what matter,
            # and the real allocation happens at ingest.
            if bold_paths:
                prewarm_subject_bold_gpu(bold_paths)
            else:
                prewarm_subject_bold_gpu(nbytes=1 << 20)
        _PREWARM_DONE = True
