"""avg_profiles_gpu.py

GPU supercall for averaging RSFC profiles across (subject × session).
Per-subject bitpacked ``.b2nd`` decode stays CPU-side (blosc2
LZ4+bitshuffle has no GPU counterpart); the accumulator and final scale
live on device, and the fused bit-unpack + accumulate happens in a
CUDA RawKernel.

Lifetime of GPU buffers:
    * Decode a subject on host (full ``(T, N, ⌈D/8⌉) uint8`` ndarray).
    * H2D the whole packed subject once per S iter.
    * Stream per-session: launch ``accum_packed_session_NhDb`` for lh
      and rh — one block per hemi vertex, threads stride over D_bytes;
      each thread owns its 8 bits → no atomics.
    * After loop: ``acc *= 1/n`` (single ufunc).
    * D2H once at the end into a *pinned* host buffer; the ``.npy``
      pair write runs on a background 2-thread pool over that same
      buffer (no second host copy).

Two entry points:

    ``avg_profiles_gpu(...)``            — disk-reading supercall, same
        signature/contract as the CPU twin. Discovers + decodes the
        per-subject ``.b2nd`` set, delegates, joins the writer, and
        DROPS the device accumulators before returning (``lh_avg_dev``
        / ``rh_avg_dev`` are ``None``) so the caller's
        ``free_all_blocks()`` can reclaim them -- this entry point has
        no device-side consumer.

    ``avg_profiles_from_packed_gpu(...)`` — memory-side twin. Takes the
        packed ``(T, N, ⌈D/8⌉)`` uint8 slabs the caller already holds
        (host numpy or cupy), skips the ``.b2nd`` round trip, and
        returns while the ``.npy`` writer is still running so the
        caller can overlap ``ini_params`` / ``radius_mask`` with it.
        The result carries the device fp32 means (``lh_avg_dev`` /
        ``rh_avg_dev``) that ``generate_ini_params_gpu`` consumes
        directly, so the fp32 means never round trip through host
        memory on the critical path.

cupy is imported at module top — this file is only loaded via the
lazy ``from .avg_profiles_gpu import …`` line in the dispatch branch,
so the CPU path never pays the cupy import cost.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cupy as cp
import numpy as np

from ..data_io._background_write import BackgroundWriteHandle
from .avg_profiles import (
    AvgProfilesResult,
    _avg_paths,
    _discover_and_decode_subjects,
    _hemi_vert_counts,
    _start_avg_npy_pair_async,
)


# Per-session bit-packed accumulator: ``acc[n, d] += popcount-style bit
# unpack of packed[n, d]`` for one (subject, session, hemi) slab.
# Mirrors the CPU :func:`_accum_packed_session_inplace_kernel` semantics
# bit-by-bit (for binary input the result is independent of summation
# order — every partial sum stays in fp32's exact-integer range).
_ACCUM_PACKED_BLOCK = 256

_accum_packed_session_kernel = None  # lazy compile (see _get_kernel)


def _get_accum_packed_kernel():
    """Lazy-compile the per-session packed-accumulate CUDA kernel.

    Compiled on first call so the import of this module stays cheap;
    cached as a module-global afterwards.
    """
    global _accum_packed_session_kernel
    if _accum_packed_session_kernel is None:
        _accum_packed_session_kernel = cp.RawKernel(r"""
extern "C" __global__
void accum_packed_session_NhDb(const unsigned char* __restrict__ packed,
                                float* __restrict__ acc,
                                int V_h, int D, int D_bytes) {
    // One block per hemi vertex; threads stride over D_bytes within
    // the row. Each thread processes its assigned bytes by unpacking 8
    // bits and conditionally adding 1.0f to 8 acc slots - no atomics
    // since each (n, d) slot is touched by exactly one thread.
    //
    // Bit convention: cell d <=> bit (d & 7) of byte (d >> 3),
    // LSB-first, matches numpy.packbits(bitorder='little'). Padding
    // bits past D in the last byte MUST be zero (writer guarantees);
    // the d < D check is defensive.
    //
    // fp32 accumulator: each summand is exactly 0.0f or 1.0f, partial
    // sums stay <= subjects * sessions << 2^24, so no rounding occurs
    // and the order-independence guarantee holds (bit-identical to
    // unpack-then-add-fp32).
    const int n = blockIdx.x;
    if (n >= V_h) return;
    const unsigned char* row_p = packed + (size_t)n * (size_t)D_bytes;
    float* row_a = acc + (size_t)n * (size_t)D;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    for (int b = tid; b < D_bytes; b += bs) {
        unsigned int bv = (unsigned int)row_p[b];
        if (bv != 0u) {
            int base = b * 8;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int d = base + k;
                if (d < D && ((bv >> k) & 1u)) {
                    row_a[d] += 1.0f;
                }
            }
        }
    }
}
""", "accum_packed_session_NhDb")
    return _accum_packed_session_kernel


def _accum_packed_session_device(
    packed_NhDb_dev: cp.ndarray,   # (V_h, D_bytes) uint8 device
    acc_VhD_dev: cp.ndarray,        # (V_h, D) fp32 device — modified in-place
    D: int,
) -> None:
    """Launch the per-session packed-accumulate kernel.

    Both inputs must be device-resident and C-contig. Shape checks
    surface mismatches at the launch boundary instead of letting them
    silently misindex inside the kernel.
    """
    if packed_NhDb_dev.dtype != cp.uint8:
        raise ValueError(
            f"packed_NhDb_dev must be uint8; got {packed_NhDb_dev.dtype}"
        )
    if acc_VhD_dev.dtype != cp.float32:
        raise ValueError(
            f"acc_VhD_dev must be fp32; got {acc_VhD_dev.dtype}"
        )
    V_h, D_bytes = packed_NhDb_dev.shape
    Va, Da = acc_VhD_dev.shape
    if Va != V_h:
        raise ValueError(
            f"V_h mismatch: packed has {V_h}, acc has {Va}"
        )
    if Da != int(D):
        raise ValueError(
            f"D mismatch: acc has {Da}, expected {D}"
        )
    if D_bytes != (int(D) + 7) // 8:
        raise ValueError(
            f"D_bytes ({D_bytes}) != ceil(D/8) ({(int(D)+7)//8}) for D={D}"
        )
    kernel = _get_accum_packed_kernel()
    kernel(
        (V_h,), (_ACCUM_PACKED_BLOCK,),
        (packed_NhDb_dev, acc_VhD_dev,
         cp.int32(V_h), cp.int32(D), cp.int32(D_bytes)),
    )


def _pinned_host_buffer(shape: Tuple[int, int],
                        dtype=np.float32) -> np.ndarray:
    """Page-locked host ndarray of ``shape`` / ``dtype``.

    Used as the single D2H landing buffer, then shared with the
    background ``.npy`` writer -- pinned staging roughly halves the
    192 MB/hemi copy vs a pageable destination, and the writer sees
    exactly those bytes (no second host copy).
    """
    if len(shape) != 2:
        raise ValueError(f"_pinned_host_buffer: shape must be 2-D; got {shape}")
    n = int(shape[0]) * int(shape[1])
    itemsize = int(np.dtype(dtype).itemsize)
    mem = cp.cuda.alloc_pinned_memory(n * itemsize)
    return np.frombuffer(mem, dtype=dtype, count=n).reshape(shape)


def _as_packed_device(arr, V_lh: int, V_rh: int, D: int, idx: int):
    """Validate + (if host) upload one subject's packed slab.

    Accepts a host numpy array (H2D'd here, one transfer) or an
    already-device cupy array (used in place). Every shape / dtype /
    contiguity precondition of :func:`_accum_packed_session_device` is
    checked once per subject, with the subject index in the message.
    """
    D_bytes = (int(D) + 7) // 8
    N = int(V_lh) + int(V_rh)
    on_device = isinstance(arr, cp.ndarray)
    probe = arr if on_device else np.asarray(arr)
    if probe.ndim != 3:
        raise ValueError(
            f"avg_profiles_from_packed_gpu: subject {idx} must be 3-D "
            f"(T, N, ceil(D/8)); got shape {probe.shape}"
        )
    if probe.dtype != np.uint8:
        raise ValueError(
            f"avg_profiles_from_packed_gpu: subject {idx} must be uint8; "
            f"got {probe.dtype}"
        )
    if probe.shape[1] != N:
        raise ValueError(
            f"avg_profiles_from_packed_gpu: subject {idx} has N="
            f"{probe.shape[1]} but the mesh yields V_lh+V_rh={N}"
        )
    if probe.shape[2] != D_bytes:
        raise ValueError(
            f"avg_profiles_from_packed_gpu: subject {idx} last axis "
            f"({probe.shape[2]}) != ceil(D/8) ({D_bytes}) for D={D}"
        )
    if not probe.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"avg_profiles_from_packed_gpu: subject {idx} must be C-contiguous"
        )
    return probe if on_device else cp.asarray(probe)


def avg_profiles_from_packed_gpu(
    packed_subjects: Sequence[np.ndarray],
    D: int,
    targ_mesh: str,
    seed_mesh: str,
    out_dir,
    *,
    save: bool = True,
    verbose: bool = False,
) -> AvgProfilesResult:
    """Average already-in-memory packed profiles on device.

    Memory-side twin of :func:`avg_profiles_gpu`: instead of
    discovering + decoding the ``.b2nd`` set it takes the packed slabs
    the caller already holds (step-1 ``generate_profiles`` output, or a
    ``read_subject_profile_packed_tnd`` load) and runs the identical
    device accumulation via :func:`_accum_packed_session_device`.

    Parameters
    ----------
    packed_subjects : sequence of ``(T, N, ceil(D/8))`` uint8 C-contig
        arrays -- host numpy (H2D'd here, one transfer per subject) or
        cupy (used in place). One entry per subject; all entries must
        share the same ``(T, N, ceil(D/8))``.
    D : int -- unpacked cell count (``D_unpacked`` from the .b2nd vlmeta).
    targ_mesh, seed_mesh : mesh names -- size the hemis and name the
        output ``.npy`` pair.
    out_dir : project root; the pair lands at
        ``profiles/avg_profile/{lh,rh}_<targ>_roi<seed>_avg_profile.npy``.
    save : when True (default) the ``.npy`` pair write is *started* on a
        background pool and the result carries a ``writer`` handle --
        the caller MUST call ``result.writer.wait()`` before exiting.
        When False nothing is written, ``writer`` is ``None``, and the
        result's paths name where the pair *would* live.

    Returns
    -------
    :class:`AvgProfilesResult` carrying the host fp32 means (``lh_avg``
    / ``rh_avg``, living in the pinned D2H buffer -- do not mutate
    before ``writer.wait()``), the device fp32 means (``lh_avg_dev`` /
    ``rh_avg_dev``, feed straight into
    ``generate_ini_params_gpu(precomputed_*_avg_dev=...)``), and the
    write handle.

    Numerics: identical to the CPU supercall. Every summand is exactly
    0.0f or 1.0f and partial sums stay far below 2**24, so the fp32
    accumulation is integer-exact and order-free.
    """
    if "fsaverage" not in targ_mesh:
        raise ValueError(
            f"avg_profiles_from_packed_gpu: only fsaverage* targ_mesh "
            f"supported, got {targ_mesh!r}."
        )
    subs = list(packed_subjects)
    if not subs:
        raise ValueError("avg_profiles_from_packed_gpu: packed_subjects is empty")
    D = int(D)
    if D <= 0:
        raise ValueError(f"avg_profiles_from_packed_gpu: D must be > 0; got {D}")
    out_dir = Path(out_dir)
    if save:
        (out_dir / "profiles" / "avg_profile").mkdir(parents=True, exist_ok=True)

    V_lh, V_rh = _hemi_vert_counts(targ_mesh)

    lh_acc_dev = cp.zeros((V_lh, D), dtype=cp.float32)
    rh_acc_dev = cp.zeros((V_rh, D), dtype=cp.float32)

    n_pairs = 0
    T0 = None
    for idx, arr in enumerate(subs):
        arr_dev = _as_packed_device(arr, V_lh, V_rh, D, idx)
        T = int(arr_dev.shape[0])
        if T0 is None:
            T0 = T
        elif T != T0:
            raise ValueError(
                f"avg_profiles_from_packed_gpu: subject {idx} has T={T} "
                f"but subject 0 has T={T0}"
            )
        for t in range(T):
            # Indexed inline: a loop-local name for the session view
            # would outlive the loop and keep ``arr_dev``'s block alive
            # across the next subject's upload.
            _accum_packed_session_device(arr_dev[t][:V_lh], lh_acc_dev, D)
            _accum_packed_session_device(arr_dev[t][V_lh:], rh_acc_dev, D)
            n_pairs += 1
        # Free the device slab as soon as it is folded in when we own it
        # (host input), so peak device usage stays at one subject.
        if arr_dev is not arr:
            del arr_dev
        if verbose:
            print(f"current subject:{idx + 1}")

    inv_n = cp.float32(1.0 / float(n_pairs))
    lh_acc_dev *= inv_n
    rh_acc_dev *= inv_n

    # Single D2H into pinned memory; the writer thread reads that same
    # buffer, so there is exactly one host copy of each hemi.
    lh_acc = _pinned_host_buffer((V_lh, D), np.float32)
    rh_acc = _pinned_host_buffer((V_rh, D), np.float32)
    lh_acc_dev.get(out=lh_acc)
    rh_acc_dev.get(out=rh_acc)

    lh_out, rh_out = _avg_paths(out_dir, targ_mesh, seed_mesh)
    writer: Optional[BackgroundWriteHandle] = None
    if save:
        writer = _start_avg_npy_pair_async(
            out_dir, targ_mesh, seed_mesh, lh_acc, rh_acc,
        )

    return AvgProfilesResult(
        lh_path=lh_out, rh_path=rh_out,
        lh_avg=lh_acc, rh_avg=rh_acc,
        lh_avg_dev=lh_acc_dev, rh_avg_dev=rh_acc_dev,
        writer=writer,
    )


def avg_profiles_gpu(
    seed_mesh: str,
    targ_mesh: str,
    out_dir,
    num_sub,
    num_sess,
    *,
    verbose: bool = True,
) -> AvgProfilesResult:
    """GPU port of :func:`avg_profiles.avg_profiles`. Same external
    contract (file discovery, output paths); the only difference is
    that the per-vertex accumulator lives on device.

    Disk-reading entry point: discovers + decodes the per-subject
    ``.b2nd`` set, then delegates to
    :func:`avg_profiles_from_packed_gpu`. The ``.npy`` pair write is
    joined before returning, so this call keeps its historical
    synchronous contract (``result.writer`` is already ``done``).

    Memory contract (differs from the memory-side twin on purpose):

    * ``lh_avg_dev`` / ``rh_avg_dev`` are ``None``. Nothing downstream
      of this entry point consumes device buffers -- the disk path
      exists for callers that go on to re-read the ``.npy`` pair or use
      the host arrays -- so retaining the two ``(V_h, D)`` fp32
      accumulators (~385 MB together at fsa6 / Schaefer-400) would just
      pin device memory that ``cp.get_default_memory_pool()
      .free_all_blocks()`` could otherwise reclaim. The delegate's
      result is rebuilt with those fields cleared and its own reference
      dropped, so the last reference dies here.
    * ``lh_avg`` / ``rh_avg`` ARE the pinned D2H landing buffers (one
      pinned block each, held only by the returned arrays -- the writer
      has been joined and the executor no longer references them).
      Returning pinned-backed host arrays is deliberate: it is the same
      memory the writer just streamed to disk, so there is exactly one
      host copy of each hemi. It is released to cupy's pinned pool when
      the caller drops the result.
    """
    if "fsaverage" not in targ_mesh:
        raise ValueError(
            f"avg_profiles_gpu: only fsaverage* targ_mesh supported, "
            f"got {targ_mesh!r}."
        )
    num_sub = int(num_sub)
    num_sess = int(num_sess)
    out_dir = Path(out_dir)
    (out_dir / "profiles" / "avg_profile").mkdir(parents=True, exist_ok=True)

    V_lh, V_rh = _hemi_vert_counts(targ_mesh)

    sub_paths, packed_arrs, _T0, _N0, _D0_bytes, D0 = _discover_and_decode_subjects(
        out_dir, targ_mesh, seed_mesh, num_sub, num_sess, V_lh, V_rh, verbose,
    )
    if verbose:
        for sub, _p in sub_paths:
            print(f"current subject:{sub}")

    res = avg_profiles_from_packed_gpu(
        packed_arrs, D0, targ_mesh, seed_mesh, out_dir,
        save=True, verbose=False,
    )
    res.writer.wait()
    # Drop the device accumulators: no consumer on this path, and the
    # frozen dataclass cannot be mutated in place. ``del res`` removes
    # the only other reference, so the cupy blocks become reclaimable
    # by ``free_all_blocks()`` as soon as we return.
    out = _dc_replace(res, lh_avg_dev=None, rh_avg_dev=None)
    del res
    return out
