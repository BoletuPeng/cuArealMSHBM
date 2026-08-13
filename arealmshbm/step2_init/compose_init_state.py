"""compose_init_state.py — fused step-2 init Params state composer.

End-to-end init for Step2Pipeline.initialize_params: streams per-
subject BOLD profiles (NTD per-sub) from disk via a
:class:`SubjectProfileLoader`, plus group mu + boundary mask, and
produces ``Params["s_lambda"] (S, N, L) fp32`` (one-hot sparse) and
``Params["theta"] (N, L) fp32`` in the EM body's expected internal
layout. One subject is decoded into a re-used scratch slot at a
time, so peak RAM stays at one subject's slab regardless of S.

See ``_kernels.profile_to_hard_labels_kernel`` and
``_kernels.compose_init_state_kernel`` for the kernel-side docs.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np

from arealmshbm.step2_io import SubjectProfileLoader

from ._kernels import (
    compose_init_state_kernel,
    profile_to_hard_labels_kernel,
)


# ─────────────────────────────────────────────────────────────────────
# Prefetch concurrency for the per-subject disk pump.
#
# Each subject's .b2nd packed slab is read on the worker pool BEFORE
# the main thread's widen+normalize + argmax stage consumes it. blosc2
# chunk decode releases the GIL during decompression, so concurrent
# workers fan out across NVMe sequential bandwidth (multi-MB sequential
# reads coalesce naturally on Windows / NTFS).
#
# Sizing: at fsa6 / T=6 each packed slab is ~47 MB; 4 concurrent reads
# pump ~190 MB of in-flight I/O — comfortably below the OS read-ahead
# window. Going past 4 is empirically wasted (blosc2 decode CPU starts
# to compete with the main-thread numba kernels at 24-core saturation).
# Override via ``MSHBM_STEP2_INIT_PREFETCH_WORKERS`` env var.
# ─────────────────────────────────────────────────────────────────────
def _default_prefetch_workers() -> int:
    env = os.environ.get("MSHBM_STEP2_INIT_PREFETCH_WORKERS")
    if env:
        try:
            n = int(env)
            if n >= 1:
                return n
        except ValueError:
            pass
    return 4


def compose_init_state(bold_loader: SubjectProfileLoader,
                       group_mtc: np.ndarray,
                       boundary_mask: np.ndarray,
                       num_clusters: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Fused init: per-sub argmax + theta + (S, N, L) Params.s_lambda.

    Parameters
    ----------
    bold_loader   : SubjectProfileLoader — streams one subject's
                    ``(N, T, D)`` fp32 profile from disk on each
                    :meth:`load_into` call. The number of subjects
                    ``S`` is taken from ``bold_loader.num_sub``.
    group_mtc     : (D+1, L) fp64 — group cluster centroids (from
                    group.mat).
    boundary_mask : (N, L) fp32 — bilateral block-diagonal radius mask
                    from build_step2_boundary_mask.
    num_clusters  : int — must match group_mtc.shape[1] and
                    boundary_mask.shape[1].

    Returns
    -------
    s_lambda_SNL_f32 : (S, N, L) fp32 — final Params["s_lambda"].
                       Sparse one-hot: each (s, n) slice has at most
                       one 1.0 at the argmax cluster; medial-wall and
                       boundary-mask-zero rows are entirely 0.
    theta            : (N, L) fp32 — final Params["theta"]. Active
                       cells = 1/S + eps_f64 cast to fp32 ≈ 1/S;
                       inactive cells = eps_f64 cast to fp32
                       (≈ 2.22e-16).
    """
    S = int(bold_loader.num_sub)
    if S < 1:
        raise ValueError(f"need at least 1 subject; got {S}")
    N_l, T, D_l = bold_loader.dims()

    bm = np.ascontiguousarray(boundary_mask, dtype=np.float32)
    L = int(num_clusters)
    src = np.asarray(group_mtc)
    if src.shape[1] != L:
        raise ValueError(
            f"group_mtc.shape[1]={src.shape[1]} != num_clusters={L}")
    if bm.shape[1] != L:
        raise ValueError(
            f"boundary_mask.shape[1]={bm.shape[1]} != num_clusters={L}")
    N = bm.shape[0]
    D = src.shape[0]
    if N_l != N:
        raise ValueError(
            f"bold_loader.dims() N={N_l} != boundary_mask N={N}")
    if D_l != D:
        raise ValueError(
            f"bold_loader.dims() D={D_l} != group_mtc.shape[0]={D}")
    # Cast group_mtc to fp32 once for the hard-label kernel. See
    # _kernels.profile_to_hard_labels_kernel for the precision
    # rationale (fp32 acc is within BLAS-vendor noise of fp64; the
    # bench showed 1 / 224k vertex flip and 2× speedup).
    g_mu_f32 = np.ascontiguousarray(src, dtype=np.float32)

    # Step 1 — per-sub hard_label + medial. Two pipelined fast-paths
    # depending on the loader's ``cache_mode``:
    #
    #   Path A (cache_mode='eager_bitpacked'): the CPU step-2 backend.
    #     The loader's host packed cache is reused across init + EM,
    #     so the prefetch *populates production-needed state* — not
    #     throwaway work. Workers call ``prefetch_packed(s)`` (disk +
    #     blosc2 decode, GIL-released), then the main thread runs the
    #     warm ``load_into`` path (widen+normalize numba kernel from
    #     cached packed bytes, ~56 ms) plus the argmax kernel (~134 ms).
    #     Cohort wall: 52s sequential → 8.75s on the S=40 reference cohort (5.8x).
    #
    #   Path B (cache_mode='stream', e.g. GPU backend): no host cache;
    #     each load_into goes back to disk. We can't prefetch into a
    #     shared cache, but we CAN parallelize the per-sub
    #     disk-read+widen+normalize across worker scratches and run
    #     argmax sequentially on the main thread. n_workers+1 (N, T, D)
    #     fp32 scratch slabs (~3.85 GB transient at fsa6/T=6/n=4) —
    #     released when compose_init_state returns.
    #
    # The widen+normalize kernel itself is numba-parallel (24-thread
    # prange across N); under Path A we keep it on the main thread to
    # avoid oversubscription, but under Path B the per-sub kernels run
    # concurrently on workers — each one is ~56 ms / 24-core kernel
    # so the contention is workable (workers spend most wall time in
    # blosc2 decode + numpy chunk-decode anyway).
    #
    # Pre-allocate per-sub argmax+medial outputs (S, N) layout for the
    # compose-kernel's tight inner loop.
    hard_label_SN = np.empty((S, N), dtype=np.int32)
    medial_SN = np.empty((S, N), dtype=np.bool_)

    cache_mode = getattr(bold_loader, "cache_mode", None)
    has_prefetch = (cache_mode == "eager_bitpacked"
                    and hasattr(bold_loader, "prefetch_packed"))
    n_workers_req = _default_prefetch_workers()
    n_workers = min(n_workers_req, S)

    if has_prefetch and n_workers > 1:
        # ── Path A: bitpacked-cache prefetch + warm widen+argmax ──
        bold_scratch_NTD = np.empty((N, T, D), dtype=np.float32)
        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="step2-init-prefetch",
        ) as ex:
            in_flight = {}
            for s_ahead in range(min(n_workers, S)):
                in_flight[s_ahead] = ex.submit(
                    bold_loader.prefetch_packed, s_ahead + 1,
                )
            next_to_submit = min(n_workers, S)
            for s in range(S):
                in_flight.pop(s).result()
                if next_to_submit < S:
                    in_flight[next_to_submit] = ex.submit(
                        bold_loader.prefetch_packed, next_to_submit + 1,
                    )
                    next_to_submit += 1
                bold_loader.load_into(s + 1, bold_scratch_NTD)
                profile_to_hard_labels_kernel(
                    bold_scratch_NTD, g_mu_f32,
                    hard_label_SN[s], medial_SN[s],
                )
    elif n_workers > 1:
        # ── Path B: per-worker scratch + parallel full-decode ──
        # n_workers+1 scratches: up to n_workers in-flight reads, one
        # active for main-thread argmax. Slot recycling via free-stack.
        n_slots = n_workers + 1
        scratches = [np.empty((N, T, D), dtype=np.float32)
                     for _ in range(n_slots)]
        free_slots = list(range(n_slots))

        def _decode_into(s_1, slot_idx):
            # Stream-mode load_into: per-session disk decode +
            # normalize. blosc2 + numba both GIL-release internally.
            bold_loader.load_into(s_1, scratches[slot_idx])
            return slot_idx

        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="step2-init-decode",
        ) as ex:
            in_flight = {}
            for s_ahead in range(min(n_workers, S)):
                slot = free_slots.pop()
                in_flight[s_ahead] = ex.submit(
                    _decode_into, s_ahead + 1, slot,
                )
            next_to_submit = min(n_workers, S)
            for s in range(S):
                slot = in_flight.pop(s).result()
                profile_to_hard_labels_kernel(
                    scratches[slot], g_mu_f32,
                    hard_label_SN[s], medial_SN[s],
                )
                # After argmax done, slot is safe to overwrite.
                free_slots.append(slot)
                if next_to_submit < S:
                    new_slot = free_slots.pop()
                    in_flight[next_to_submit] = ex.submit(
                        _decode_into, next_to_submit + 1, new_slot,
                    )
                    next_to_submit += 1
    else:
        # Degenerate path (n_workers=1 or S=1). One scratch, sequential
        # load + argmax. Preserved for tests and the env-var override.
        bold_scratch_NTD = np.empty((N, T, D), dtype=np.float32)
        for s in range(S):
            bold_loader.load_into(s + 1, bold_scratch_NTD)
            profile_to_hard_labels_kernel(
                bold_scratch_NTD, g_mu_f32, hard_label_SN[s], medial_SN[s],
            )

    # Step 2 — compose. np.zeros is calloc-backed → most pages stay
    # lazy (unmapped) since the kernel sparse-writes (only S*N ≈ 246k
    # nonzero cells out of S*N*L ≈ 98M). theta is fully written.
    #
    # s_lambda is fp32 storage; the compose kernel writes only 1.0 / 0.0
    # hard-label values, both exactly representable in fp32 — allocate
    # fp32 directly (no fp64 scratch + astype round-trip). At S=200 /
    # N=81924 / L=400 this halves transient init RAM: 26 GB → 13 GB.
    s_lambda_SNL_f32 = np.zeros((S, N, L), dtype=np.float32)
    theta = np.empty((N, L), dtype=np.float32)
    inv_S_f32 = np.float32(1.0) / np.float32(S)
    eps_f64 = np.float64(np.finfo(np.float64).eps)
    compose_init_state_kernel(
        hard_label_SN, medial_SN, bm,
        s_lambda_SNL_f32, theta,
        inv_S_f32, eps_f64,
    )

    return s_lambda_SNL_f32, theta
