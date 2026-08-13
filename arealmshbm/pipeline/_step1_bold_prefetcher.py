"""_step1_bold_prefetcher.py — cross-(sub, sess) BOLD prefetcher for step1.

``run_generate_profiles`` walks 240 (sub, sess) iterations serially (40
subj × 6 sess on YS). Each iter pays the GIFTI cold-decode wall (~165 ms
warm) on the main thread before GPU work starts. The leaf's per-(sub,
sess) ``ThreadPoolExecutor`` degenerates to one worker on YS (n_runs=1
per session). This module hoists the decode pool to the driver level —
one persistent 8-worker pool decodes (sub_k, sess_{j+1}) and the
lookahead window beyond while the GPU loop crunches (sub_k, sess_j).

The lookahead is a SLIDING bound — the scheduler primes ahead, calls
``prime(...)`` after each consume. Eagerly submitting all 240 would
buffer ~19 GB of decoded fp32 BOLD in the futures dict before the GPU
catches up.

Backends (issue #55): ``cpu`` decodes via ``read_surface_gifti`` +
host transpose; ``gpu`` via ``read_surface_giftis_gpu_full_pipeline``
(per-file H2D → GPU bytescan + base64 RawKernels → batched nvCOMP
Deflate) + ``cp.ascontiguousarray(.T)`` on device — leaf's
``compute_profile_arrays_gpu`` sees the buffers zero-copy. Each GPU
worker owns a non_blocking CUDA stream (``_gpu_worker_init``) so
decode overlaps the consumer's compute instead of serializing on the
legacy default stream. Both backends are bit-equal; contract pinned
by ``arealmshbm/pipeline/tests/test_bold_prefetcher_gpu.py``.
**GPU is now the production default** for GPU-leaf step1 (3.56× E2E
vs CPU on YS); CPU stays the default on CPU-leaf step1.

Memory at fsa6 / T=240 / 1 run/sess: ~79 MB per session;
lookahead=8 ≈ 632 MB inflight (host on cpu / device on gpu).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from arealmshbm.data_io.gifti_io import (
    read_surface_gifti, read_surface_giftis_gpu_full_pipeline,
)

if TYPE_CHECKING:
    # ``from __future__ import annotations`` makes all type hints
    # strings at runtime, so cupy stays a soft dependency on the CPU
    # path. Same pattern as ``_step0_bold_prefetcher`` and
    # ``arealmshbm/data_io/gifti_io.py``.
    import cupy as cp


def _load_session_bold_runs_cpu(
    lh_paths: List[str], rh_paths: List[str],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """CPU worker: decode all runs in one (sub, sess), return one
    ``(lh_TxV, rh_TxV)`` host fp32 tuple per run pair."""
    if len(lh_paths) != len(rh_paths):
        raise ValueError(
            f"lh_runs ({len(lh_paths)}) != rh_runs ({len(rh_paths)}); "
            f"mismatched per-hemi fMRI lists"
        )
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for lh_p, rh_p in zip(lh_paths, rh_paths):
        lh_vol_VxT = read_surface_gifti(lh_p)
        rh_vol_VxT = read_surface_gifti(rh_p)
        lh_TxV = np.ascontiguousarray(lh_vol_VxT.T, dtype=np.float32)
        rh_TxV = np.ascontiguousarray(rh_vol_VxT.T, dtype=np.float32)
        out.append((lh_TxV, rh_TxV))
    return out


# Per-thread non-blocking CUDA stream for the GPU prefetcher. Issue #55
# fix: the legacy default stream serializes all GPU work in the process,
# so 8 prefetcher workers all queue their nvCOMP decode + transpose on
# one stream and run serially. A non_blocking stream per worker does NOT
# synchronize with the legacy default stream (or with each other), so:
#   * worker N's decode runs concurrently with worker M's decode
#   * worker decode runs concurrently with the consumer's compute on the
#     legacy default stream
# Bound to the worker thread via a ``threading.local`` populated by the
# ThreadPoolExecutor's ``initializer``. The initializer also pins the
# thread's current device so cupy lookups inside the worker honor the
# main thread's device selection.
_TLS = threading.local()


def _gpu_worker_init(device_id: int) -> None:
    """ThreadPoolExecutor initializer: bind a non_blocking stream +
    current device to this worker thread.

    Called once per worker at pool start. Subsequent calls into the
    worker reuse the same stream — stream creation has CUDA syscall
    overhead and would dwarf per-file decode if re-created.
    """
    import cupy as cp
    cp.cuda.Device(device_id).use()
    _TLS.stream = cp.cuda.Stream(non_blocking=True)
    _TLS.device_id = device_id


def _load_session_bold_runs_gpu(
    lh_paths: List[str], rh_paths: List[str],
    gpu_device_id: int,
) -> List[Tuple["cp.ndarray", "cp.ndarray"]]:
    """GPU worker: full device-stay GIFTI decode for all runs in one
    (sub, sess). Returns one ``(lh_TxV, rh_TxV)`` cupy fp32 tuple per
    run pair on ``gpu_device_id``.

    Both hemis go through a single ``read_surface_giftis_gpu_full_pipeline``
    call so one batched nvCOMP Deflate decode covers lh+rh chunks. All
    GPU work runs on the worker's per-thread non_blocking stream
    (``_gpu_worker_init``); ``stream.synchronize()`` at the end drains
    it so the consumer can use the buffers immediately on its own
    stream. ~83 ms / sess on YS vs ~225 ms / sess on CPU backend.
    """
    import cupy as cp

    if len(lh_paths) != len(rh_paths):
        raise ValueError(
            f"lh_runs ({len(lh_paths)}) != rh_runs ({len(rh_paths)}); "
            f"mismatched per-hemi fMRI lists"
        )
    stream = getattr(_TLS, "stream", None)
    if stream is None:
        with cp.cuda.Device(gpu_device_id):
            stream = cp.cuda.Stream(non_blocking=True)
    n_runs = len(lh_paths)

    out: List[Tuple["cp.ndarray", "cp.ndarray"]] = []
    with cp.cuda.Device(gpu_device_id), stream:
        # Concatenate lh+rh into one batch: outs[0..n_runs-1] are lh,
        # outs[n_runs..] are rh. One pipeline call -> one nvCOMP batch
        # decode covering both hemis. ``layout='TxN'`` skips the
        # reader's internal NxT-transpose AND the re-transpose this
        # worker would otherwise do; the leaf's GPU path consumes the
        # buffers as (T, V_h) C-contig fp32 directly.
        all_paths = list(lh_paths) + list(rh_paths)
        decoded = read_surface_giftis_gpu_full_pipeline(
            all_paths, to_host=False, n_workers=min(4, len(all_paths)),
            layout="TxN",
        )
        for r in range(n_runs):
            # Reader emits (T, N) C-contig fp32 directly under
            # layout='TxN'. The cp.ascontiguousarray + dtype assertion
            # of the legacy NxT path was load-bearing only because
            # there was a transpose between the reader's output and
            # the leaf's consumer expectation; with the reader giving
            # us TxV already, neither is needed.
            lh_TxV = decoded[r]
            rh_TxV = decoded[n_runs + r]
            out.append((lh_TxV, rh_TxV))
    stream.synchronize()
    return out


class Step1BoldPrefetcher:
    """Cross-(sub, sess) BOLD prefetcher with a long-lived worker pool.

    Lifecycle: ``__init__`` opens the pool → ``prime(sub, sess,
    lh_paths, rh_paths)`` submits one session's runs (idempotent on
    key) → ``get(sub, sess)`` blocks then pops the future (numpy tuples
    on cpu / cupy tuples on gpu) → ``close()`` drains + frees. Usable
    as a context manager. Lock is precautionary; the driver consumes
    from the main thread.
    """

    def __init__(self, *, n_workers: int = 8, backend: str = "cpu"):
        # n_workers=8 matches the isal_zlib saturation plateau; same
        # cap on gpu for symmetry. See Step0BoldPrefetcher.__init__ for
        # the thread-local cupy current-device capture rationale.
        if backend not in ("cpu", "gpu"):
            raise ValueError(
                f"Step1BoldPrefetcher: backend must be 'cpu' or 'gpu'; "
                f"got {backend!r}"
            )
        self._backend = backend
        self._gpu_device_id: Optional[int] = None
        pool_kwargs = dict(
            max_workers=int(n_workers),
            thread_name_prefix=f"bold-io-step1-{backend}",
        )
        if backend == "gpu":
            import cupy as cp
            self._gpu_device_id = int(cp.cuda.runtime.getDevice())
            # Initializer pins device + creates a per-worker
            # non_blocking stream — see ``_gpu_worker_init`` and
            # ``_load_session_bold_runs_gpu`` for the Issue #55 fix
            # rationale (without this, all worker decodes serialize
            # on the legacy default stream → multiple workers ≈ 1).
            pool_kwargs["initializer"] = _gpu_worker_init
            pool_kwargs["initargs"] = (self._gpu_device_id,)
        self._pool = ThreadPoolExecutor(**pool_kwargs)
        self._futures: Dict[Tuple[str, str], Future] = {}
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        """Backend selected at construction — read-only. Must match
        ``compute_profile_arrays(backend=...)`` or cupy buffers reach
        numba kernels and fail with an opaque deep-stack error."""
        return self._backend

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        # Drain pool, drop unpulled future refs (whose buffers — numpy
        # on cpu / cupy on gpu — can then be reclaimed promptly on
        # exception paths). Symmetric with Step0BoldPrefetcher.close().
        self._pool.shutdown(wait=True)
        self._futures.clear()

    def prime(
        self, sub: str, sess: str,
        lh_paths: List[str], rh_paths: List[str],
    ) -> None:
        """Submit one (sub, sess) session for background decode.
        Idempotent — re-priming an already-primed key is a no-op."""
        key = (str(sub), str(sess))
        with self._lock:
            if key in self._futures:
                return
            if self._backend == "gpu":
                # __init__ guarantees ``_gpu_device_id`` is set on the
                # GPU branch; narrow the Optional type for downstream
                # readers.
                assert self._gpu_device_id is not None
                self._futures[key] = self._pool.submit(
                    _load_session_bold_runs_gpu,
                    lh_paths, rh_paths, self._gpu_device_id,
                )
            else:
                self._futures[key] = self._pool.submit(
                    _load_session_bold_runs_cpu,
                    lh_paths, rh_paths,
                )

    def get(
        self, sub: str, sess: str,
    ) -> "List[Tuple[np.ndarray, np.ndarray]] | List[Tuple[cp.ndarray, cp.ndarray]]":
        """Block until (sub, sess) is decoded; pop the future.

        Returns the list-of-runs the leaf's
        ``compute_profile_arrays(precomputed_bold_runs=...)`` consumes.
        On gpu the leaf's ``cp.asarray(lh_TxV)`` is zero-copy. Raises
        ``KeyError`` (explicit message) if the key was never primed or
        already consumed — surfaces the scheduling bug at the consumer.
        """
        key = (str(sub), str(sess))
        with self._lock:
            fut = self._futures.pop(key, None)
        if fut is None:
            raise KeyError(
                f"Step1BoldPrefetcher.get: ({sub!r}, {sess!r}) was not "
                f"primed or has already been consumed"
            )
        return fut.result()
