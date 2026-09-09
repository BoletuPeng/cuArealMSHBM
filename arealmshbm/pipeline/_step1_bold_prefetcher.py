"""_step1_bold_prefetcher.py — cross-(sub, sess) BOLD prefetcher for the
step1 CPU backend.

``run_generate_profiles(backend='cpu')`` walks 240 (sub, sess)
iterations serially (40 subj × 6 sess on YS). Each iter pays the GIFTI
cold-decode wall (~165 ms warm) on the main thread before the leaf
starts. The leaf's per-(sub, sess) ``ThreadPoolExecutor`` degenerates
to one worker on YS (n_runs=1 per session). This module hoists the
decode pool to the driver level — one persistent 8-worker pool decodes
(sub_k, sess_{j+1}) and the lookahead window beyond while the compute
loop crunches (sub_k, sess_j).

The lookahead is a SLIDING bound — the scheduler primes ahead, calls
``prime(...)`` after each consume. Eagerly submitting all 240 would
buffer ~19 GB of decoded fp32 BOLD in the futures dict before the leaf
catches up.

Decode is ``read_surface_gifti(time_major=True)`` on 8 workers (the
isal_zlib saturation plateau), which releases the GIL and so overlaps
the leaf. The GPU leaf never uses this prefetcher: it owns the whole
subject and drives its own ingest (batched nvCOMP, or the same CPU
reader on its own pool when nvCOMP is absent).

Memory at fsa6 / T=240 / 1 run/sess: ~79 MB per session;
lookahead=8 ≈ 632 MB of host buffers inflight.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Tuple

import numpy as np

from arealmshbm.data_io.gifti_io import read_surface_gifti


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
        out.append((read_surface_gifti(lh_p, time_major=True),
                    read_surface_gifti(rh_p, time_major=True)))
    return out


class Step1BoldPrefetcher:
    """Cross-(sub, sess) BOLD prefetcher with a long-lived worker pool.

    Lifecycle: ``__init__`` opens the pool → ``prime(sub, sess,
    lh_paths, rh_paths)`` submits one session's runs (idempotent on
    key) → ``get(sub, sess)`` blocks then pops the future (host numpy
    tuples) → ``close()`` drains + frees. Usable as a context manager.
    Lock is precautionary; the driver consumes from the main thread.
    """

    def __init__(self, *, n_workers: int = 8):
        # n_workers=8 matches the isal_zlib saturation plateau.
        self._pool = ThreadPoolExecutor(
            max_workers=int(n_workers),
            thread_name_prefix="bold-io-step1",
        )
        self._futures: Dict[Tuple[str, str], Future] = {}
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        # Drain pool, drop unpulled future refs so their host buffers
        # are reclaimed promptly on exception paths. Symmetric with
        # Step0BoldPrefetcher.close().
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
            self._futures[key] = self._pool.submit(
                _load_session_bold_runs_cpu, lh_paths, rh_paths,
            )

    def get(
        self, sub: str, sess: str,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Block until (sub, sess) is decoded; pop the future.

        Returns the list-of-runs the leaf's
        ``compute_profile_arrays(precomputed_bold_runs=...)`` consumes.
        Raises ``KeyError`` (explicit message) if the key was never
        primed or already consumed — surfaces the scheduling bug at the
        consumer.
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
