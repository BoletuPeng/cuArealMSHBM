"""_step0_bold_prefetcher.py — cross-subject BOLD prefetcher for step0.

The standalone ``Step0Pipeline`` keeps a per-subject worker pool that
overlaps sessions 2..N within one subject but pays a ~1 s cold-start
on session 1 of every subject. This driver-level layer owns one
persistent ``ThreadPoolExecutor`` for the full step0 phase so subject
K+1's BOLD decode is in flight while subject K is on the GPU.

Backends (issue #55): ``cpu`` decodes via ``read_surface_gifti`` +
``concat_hemis_drop_medial`` (GIL-released isal_zlib, true 8-way
parallelism). ``gpu`` decodes via ``read_surface_gifti_gpu`` +
``concat_hemis_drop_medial_gpu`` with ``to_host=False`` so the H2D is
skipped. **Production pins cpu** — nvCOMP serializes on the legacy
default stream while the CPU pool saturates the codec; a real GPU win
needs per-worker non-blocking streams. Both backends are bit-equal;
the contract is pinned by
``arealmshbm/pipeline/tests/test_bold_prefetcher_gpu.py``.

Memory at fsa6 / T=240: ~432 MB per subject (6 sessions); lookahead=4
≈ 1.7 GB in flight (host on cpu backend, device on gpu).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from arealmshbm.bold_io import (
    read_surface_bold, concat_hemis_drop_medial,
)
from arealmshbm.data_io.gifti_io import read_surface_gifti_gpu

if TYPE_CHECKING:
    # ``from __future__ import annotations`` makes all type hints
    # strings at runtime, so importing cupy under TYPE_CHECKING gives
    # static analysers ``cp.ndarray`` resolution without forcing the
    # runtime import on CPU-only envs. Pattern matches the one in
    # ``arealmshbm/data_io/gifti_io.py``.
    import cupy as cp


def _load_session_bold_cpu(
    lh_path: Path, rh_path: Path,
    n_lh: int, n_rh: int, medial_mask: np.ndarray,
) -> np.ndarray:
    """CPU worker: read lh+rh GIFTI on host, concat, drop medial.
    Returns ``(N_cortex, T)`` host fp32. ``medial_mask`` / ``n_lh`` /
    ``n_rh`` are mesh-static — captured by the prefetcher at ``__init__``.
    """
    lh_bold = read_surface_bold(lh_path, expected_n=n_lh)
    rh_bold = read_surface_bold(rh_path, expected_n=n_rh)
    return concat_hemis_drop_medial(lh_bold, rh_bold, medial_mask)


def _load_session_bold_gpu(
    lh_path: Path, rh_path: Path,
    n_lh: int, n_rh: int, medial_mask_dev: "cp.ndarray",
    gpu_device_id: int,
) -> "cp.ndarray":
    """GPU worker: nvCOMP-decode lh+rh on device, concat, drop medial.
    Returns ``(N_cortex, T)`` cupy fp32 on ``gpu_device_id``.

    ``with cp.cuda.Device(gpu_device_id)`` is required: cupy's current
    device is thread-local and a fresh worker thread defaults to 0,
    so without this the decode would land on device 0 while the mask
    sits on main's device. Pre-concat mesh checks surface hemi+path on
    mismatch instead of an opaque ``concat_hemis_drop_medial_gpu`` error.

    Runs on the legacy default stream — see module docstring; a
    non-blocking-stream variant was measured and lost (3519 ms median
    vs 3037 ms CPU on YS, plus catastrophic per-rep growth from
    per-call ``cp.cuda.Stream`` allocation). SM contention with the
    consumer's subgraph A kernels is the actual bottleneck, not
    default-stream serialization.
    """
    import cupy as cp
    from arealmshbm.bold_io.bold_io_gpu import concat_hemis_drop_medial_gpu
    with cp.cuda.Device(gpu_device_id):
        lh_dev = read_surface_gifti_gpu(lh_path, to_host=False)
        rh_dev = read_surface_gifti_gpu(rh_path, to_host=False)
        if lh_dev.shape[0] != n_lh:
            raise ValueError(
                f"GPU BOLD load: lh GIFTI {lh_path} has "
                f"N={lh_dev.shape[0]} != expected n_lh={n_lh} "
                f"(mesh mismatch?)"
            )
        if rh_dev.shape[0] != n_rh:
            raise ValueError(
                f"GPU BOLD load: rh GIFTI {rh_path} has "
                f"N={rh_dev.shape[0]} != expected n_rh={n_rh} "
                f"(mesh mismatch?)"
            )
        return concat_hemis_drop_medial_gpu(
            lh_dev, rh_dev, medial_mask_dev)


class Step0BoldPrefetcher:
    """Cross-subject BOLD prefetcher with a long-lived worker pool.

    Lifecycle: ``__init__`` opens the pool (and on gpu, uploads
    ``medial_mask`` + captures the constructing thread's device id) →
    ``prime_subject(sub_id, sess_paths)`` submits one subject's
    sessions (idempotent on key) → ``get_provider(sub_id)`` returns a
    ``sess_idx -> (N_cortex, T)`` callable (numpy on cpu / cupy on gpu)
    → ``close()`` drains + drops state. Usable as a context manager.
    Single-GPU only (multi-GPU untested).
    """

    def __init__(
        self, *,
        medial_mask: np.ndarray, n_lh: int, n_rh: int,
        n_io_workers: int = 8,
        backend: str = "cpu",
    ):
        # n_io_workers=8 matches the isal_zlib saturation plateau
        # (~1.6 GB/s aggregate); decompress CPU is the limiter, not
        # disk bandwidth. GPU backend uses the same cap for symmetry.
        if backend not in ("cpu", "gpu"):
            raise ValueError(
                f"Step0BoldPrefetcher: backend must be 'cpu' or 'gpu'; "
                f"got {backend!r}"
            )
        self._backend = backend
        self._medial_mask_host = medial_mask
        self._n_lh = int(n_lh)
        self._n_rh = int(n_rh)
        # Lazy device-side state — CPU backend never imports cupy.
        # ``_gpu_device_id`` is captured on the main thread's current
        # device so worker threads (which default to device 0) re-enter
        # the right device per call; this also closes a multi-GPU
        # footgun. The mask is uploaded as bool (cheapest indexer).
        self._medial_mask_dev: Optional["cp.ndarray"] = None
        self._gpu_device_id: Optional[int] = None
        if backend == "gpu":
            import cupy as cp
            self._gpu_device_id = int(cp.cuda.runtime.getDevice())
            with cp.cuda.Device(self._gpu_device_id):
                self._medial_mask_dev = cp.asarray(
                    np.asarray(medial_mask).reshape(-1).astype(bool))
        self._pool = ThreadPoolExecutor(
            max_workers=int(n_io_workers),
            thread_name_prefix=f"bold-io-driver-{backend}",
        )
        self._futures: Dict[Tuple[str, int], Future] = {}
        # Guards _futures dict mutation across compute-side workers
        # (the dual-stream concurrent step0 driver primes from inside
        # worker threads after consuming a subject; without this lock
        # the ``if key in self._futures`` / ``self._futures[key] = ...``
        # check-then-set could race on duplicate keys).
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        """Backend selected at construction — read-only (mutating
        post-init would silently mix CPU/GPU output types)."""
        return self._backend

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        # Drain pool, drop unpulled future refs + device mask so the
        # driver's post-step0 ``free_all_blocks()`` actually reclaims.
        self._pool.shutdown(wait=True)
        self._futures.clear()
        self._medial_mask_dev = None

    def prime_subject(
        self, sub_id: str,
        sess_paths: List[Tuple[Path, Path]],
    ) -> None:
        """Submit all sessions for one subject. Idempotent on
        ``(sub_id, sess_idx)``. Thread-safe so the dual-stream concurrent
        step0 driver can prime from inside compute worker threads."""
        with self._lock:
            for sess_offset, paths in enumerate(sess_paths, start=1):
                key = (sub_id, sess_offset)
                if key in self._futures:
                    continue
                lh_path, rh_path = paths
                if self._backend == "gpu":
                    # __init__ guarantees both set on gpu branch.
                    assert self._gpu_device_id is not None
                    assert self._medial_mask_dev is not None
                    self._futures[key] = self._pool.submit(
                        _load_session_bold_gpu,
                        lh_path, rh_path,
                        self._n_lh, self._n_rh, self._medial_mask_dev,
                        self._gpu_device_id,
                    )
                else:
                    self._futures[key] = self._pool.submit(
                        _load_session_bold_cpu,
                        lh_path, rh_path,
                        self._n_lh, self._n_rh, self._medial_mask_host,
                    )

    def get_provider(self, sub_id: str) -> Callable[[int], "np.ndarray | cp.ndarray"]:
        """Return a ``sess_idx (1-indexed) -> (N_cortex, T)`` callable.

        The closure pops the future on consumption — prefetcher owns
        nothing past the pull. Return type: numpy on cpu / cupy on
        gpu; ``_subgraph_A`` calls ``cp.asarray()`` on both (zero-copy
        on device-resident input).
        """
        def _provider(sess_idx: int):
            key = (sub_id, sess_idx)
            with self._lock:
                fut = self._futures.pop(key, None)
            if fut is None:
                raise KeyError(
                    f"Step0BoldPrefetcher.get_provider: "
                    f"({sub_id!r}, sess_idx={sess_idx}) was not primed or "
                    f"has already been consumed"
                )
            return fut.result()
        return _provider
