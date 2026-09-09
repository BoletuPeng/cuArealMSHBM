"""_step0_bold_prefetcher.py — cross-subject BOLD prefetcher for step0.

The standalone ``Step0Pipeline`` keeps a per-subject worker pool that
overlaps sessions 2..N within one subject but pays a ~1 s cold-start
on session 1 of every subject. This driver-level layer owns one
persistent ``ThreadPoolExecutor`` for the full step0 phase so subject
K+1's BOLD decode is in flight while subject K is on the GPU.

Decode is CPU (``read_surface_gifti`` + ``concat_hemis_drop_medial``):
isal_zlib releases the GIL, so the pool's decode hides fully behind
the consumer's GPU compute and ``_subgraph_A``'s ``cp.asarray`` is the
H2D seam.

Memory at fsa6 / T=240: ~432 MB per subject (6 sessions); lookahead=4
≈ 1.7 GB of host BOLD in flight.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from arealmshbm.bold_io import (
    read_surface_bold, concat_hemis_drop_medial,
)


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


class Step0BoldPrefetcher:
    """Cross-subject BOLD prefetcher with a long-lived worker pool.

    Lifecycle: ``__init__`` opens the pool →
    ``prime_subject(sub_id, sess_paths)`` submits one subject's
    sessions (idempotent on key) → ``get_provider(sub_id)`` returns a
    ``sess_idx -> (N_cortex, T)`` host-numpy callable → ``close()``
    drains + drops state. Usable as a context manager.
    """

    def __init__(
        self, *,
        medial_mask: np.ndarray, n_lh: int, n_rh: int,
        n_io_workers: int = 8,
    ):
        # n_io_workers=8 matches the isal_zlib saturation plateau
        # (~1.6 GB/s aggregate); decompress CPU is the limiter, not
        # disk bandwidth.
        self._medial_mask_host = medial_mask
        self._n_lh = int(n_lh)
        self._n_rh = int(n_rh)
        self._pool = ThreadPoolExecutor(
            max_workers=int(n_io_workers),
            thread_name_prefix="bold-io-driver",
        )
        self._futures: Dict[Tuple[str, int], Future] = {}
        # Guards _futures dict mutation across compute-side workers
        # (the dual-stream concurrent step0 driver primes from inside
        # worker threads after consuming a subject; without this lock
        # the ``if key in self._futures`` / ``self._futures[key] = ...``
        # check-then-set could race on duplicate keys).
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        # Drain pool, drop unpulled future refs so the driver's
        # post-step0 ``free_all_blocks()`` actually reclaims.
        self._pool.shutdown(wait=True)
        self._futures.clear()

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
                self._futures[key] = self._pool.submit(
                    _load_session_bold_cpu,
                    lh_path, rh_path,
                    self._n_lh, self._n_rh, self._medial_mask_host,
                )

    def get_provider(self, sub_id: str) -> Callable[[int], np.ndarray]:
        """Return a ``sess_idx (1-indexed) -> (N_cortex, T)`` callable.

        The closure pops the future on consumption — prefetcher owns
        nothing past the pull. ``_subgraph_A`` calls ``cp.asarray()``
        on the returned host array.
        """
        def _provider(sess_idx: int) -> np.ndarray:
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
