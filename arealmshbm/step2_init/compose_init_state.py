"""compose_init_state.py — fused step-2 init Params state composer.

End-to-end init for Step2Pipeline.initialize_params: streams per-
subject BOLD profiles (NTD per-sub) from disk via a
:class:`SubjectProfileLoader`, plus group mu + boundary mask, and
produces ``Params["s_lambda"] (S, N, L) fp32`` (the stock soft masked
posterior) and ``Params["theta"] (N, L) fp32`` in the EM body's
expected internal layout. Subject reads run on a pool of up to four
worker threads; the (N, T, D) fp32 scratch is one slot when the loader
keeps its bit-packed host cache (cache_mode='eager_bitpacked', which
holds every subject's packed bytes) and one slot per worker with
cache_mode='stream'.

See ``_kernels.profile_to_soft_init_kernel`` and
``_kernels.compose_theta_kernel`` for the kernel-side docs.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np

from arealmshbm.step2_io import SubjectProfileLoader

from ._kernels import (
    compose_theta_kernel,
    profile_to_soft_init_kernel,
)


def init_kappa_T(ini_val: float, num_session: int) -> np.float32:
    """The one fp32 scalar both backends scale the init logits with.

    Stock forms ``kappa_ini * sum_t X_t * g_mu``; the port evaluates
    ``mean_t(X_t · g_mu) * kappa_T`` with ``kappa_T = f32(ini_val) *
    f32(T)``. CPU and GPU take the scalar from here so their init stays
    bit-identical.
    """
    return np.float32(np.float32(ini_val) * np.float32(num_session))


# ─────────────────────────────────────────────────────────────────────
# Prefetch concurrency for the per-subject disk pump.
#
# Each subject's .b2nd packed slab is read on the worker pool BEFORE
# the main thread's widen+normalize + soft-init stage consumes it.
#
# Sizing: at fsa6 / T=6 each packed slab is ~47 MB; 4 concurrent reads
# pump ~190 MB of in-flight I/O — comfortably below the OS read-ahead
# window. Going past 4 is empirically wasted (blosc2 decode CPU starts
# to compete with the main-thread numba kernels at 24-core saturation).
# ─────────────────────────────────────────────────────────────────────
_PREFETCH_WORKERS = 4


def compose_init_state(bold_loader: SubjectProfileLoader,
                       group_mtc: np.ndarray,
                       boundary_mask: np.ndarray,
                       num_clusters: int,
                       ini_val: float,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Fused stock init: per-sub soft posterior + theta, (S, N, L) s_lambda.

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
    ini_val       : float — the vMF init concentration (stock
                    ``ini_val``); scales the init logits.

    Returns
    -------
    s_lambda_SNL_f32 : (S, N, L) fp32 — final Params["s_lambda"]: per
                       (s, n) the in-mask soft posterior under the
                       group centroids, row-normalised (stock
                       ``ini_s_lambda .* boundary_mask`` then
                       ``./ sum``); medial-wall rows and rows whose
                       in-mask cells all underflow in fp32 are 0.
    theta            : (N, L) fp32 — final Params["theta"]: per cell
                       ``mean_s(s_lambda)`` divided again by the number
                       of subjects nonzero there (stock ``mean(., 3)``
                       then ``./ theta_num``) + eps; cells no subject
                       touches = eps_f64 cast to fp32 (≈ 2.22e-16).
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
    # Cast group_mtc to fp32 once for the init kernel (stock evaluates
    # the init logits in single precision; the fp32 serial accumulator
    # is also what the GPU kernel uses, so the two backends agree).
    g_mu_f32 = np.ascontiguousarray(src, dtype=np.float32)
    kappa_T = init_kappa_T(ini_val, T)

    # Step 1 — per-sub soft masked posterior, written straight into
    # the (S, N, L) output. Two pipelined fast-paths depending on the
    # loader's ``cache_mode``:
    #
    #   Path A (cache_mode='eager_bitpacked'): the CPU step-2 backend.
    #     The loader's host packed cache is reused across init + EM,
    #     so the prefetch *populates production-needed state* — not
    #     throwaway work. Workers call ``prefetch_packed(s)`` (disk +
    #     blosc2 decode), then the main thread runs the
    #     warm ``load_into`` path (widen+normalize numba kernel from
    #     cached packed bytes, ~56 ms) plus the init kernel.
    #     Cohort wall: 52s sequential → 8.75s on the S=40 reference cohort (5.8x).
    #
    #   Path B (cache_mode='stream'): the CPU step-2 backend with
    #     bold_cache_mode='stream'. No host cache;
    #     each load_into goes back to disk. We can't prefetch into a
    #     shared cache, but we CAN parallelize the per-sub
    #     disk-read+widen+normalize across worker scratches and run
    #     the init kernel sequentially on the main thread. n_workers
    #     (N, T, D) fp32 scratch slabs (~9.2 GB transient at
    #     fsa6/T=6/n=4) — released when compose_init_state returns.
    #
    # The widen+normalize kernel itself is numba-parallel (24-thread
    # prange across N); under Path A we keep it on the main thread to
    # avoid oversubscription, but under Path B the per-sub kernels run
    # concurrently on workers — each one is ~56 ms / 24-core kernel
    # so the contention is workable (workers spend most wall time in
    # blosc2 decode + numpy chunk-decode anyway).
    #
    # s_lambda is fully written per subject (every in-mask cell of a
    # live row carries a value), so it is allocated once, up front.
    s_lambda_SNL_f32 = np.empty((S, N, L), dtype=np.float32)

    cache_mode = getattr(bold_loader, "cache_mode", None)
    has_prefetch = (cache_mode == "eager_bitpacked"
                    and hasattr(bold_loader, "prefetch_packed"))
    n_workers = min(_PREFETCH_WORKERS, S)

    if has_prefetch:
        # ── Path A: bitpacked-cache prefetch + warm widen+soft init ──
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
                profile_to_soft_init_kernel(
                    bold_scratch_NTD, g_mu_f32, bm, kappa_T,
                    s_lambda_SNL_f32[s],
                )
    else:
        # ── Path B: per-worker scratch + parallel full-decode ──
        # One (N, T, D) fp32 scratch per worker. A slot is handed back
        # to the free-stack after the main-thread init kernel and only
        # then is the next decode submitted into it, so at most n_workers
        # scratches are ever live: n_workers-1 decoding while one feeds
        # the init kernel.
        n_slots = n_workers
        scratches = [np.empty((N, T, D), dtype=np.float32)
                     for _ in range(n_slots)]
        free_slots = list(range(n_slots))

        def _decode_into(s_1, slot_idx):
            # Stream-mode load_into: per-session disk decode +
            # normalize. numba GIL-releases internally.
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
                profile_to_soft_init_kernel(
                    scratches[slot], g_mu_f32, bm, kappa_T,
                    s_lambda_SNL_f32[s],
                )
                # After the init kernel, slot is safe to overwrite.
                free_slots.append(slot)
                if next_to_submit < S:
                    new_slot = free_slots.pop()
                    in_flight[next_to_submit] = ex.submit(
                        _decode_into, next_to_submit + 1, new_slot,
                    )
                    next_to_submit += 1

    # Step 2 — theta from the composed s_lambda (stock ``mean(., 3)``,
    # then ``./ theta_num``, then ``+ eps``).
    theta = np.empty((N, L), dtype=np.float32)
    eps_f64 = np.float64(np.finfo(np.float64).eps)
    compose_theta_kernel(s_lambda_SNL_f32, theta, eps_f64)

    return s_lambda_SNL_f32, theta
