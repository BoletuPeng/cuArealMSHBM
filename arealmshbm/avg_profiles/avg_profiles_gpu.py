"""avg_profiles_gpu.py

GPU supercall for averaging RSFC profiles across (subject × session).
The accumulator and the final scale live on device; the fused
bit-unpack + accumulate is the CUDA RawKernel
``accum_packed_session_NhDb`` over packed bytes — one block per hemi
vertex, threads stride over D_bytes, each thread owns its 8 bits, so
there are no atomics.

Two ways in:

    ``PackedProfileAccumulator`` — the step-1 GPU chain's way. Built
        once per cohort (``step1_runners.make_avg_accumulator``) and
        handed to the fused profile leaf as its ``on_packed`` hook, so
        every session is folded in from the device slab the pack kernel
        just wrote: the packed bytes never leave the GPU for the
        average, and no subject is held past its .b2nd write.
        ``avg_profiles_from_accumulator`` then scales in place, does one
        pinned D2H shared with the background ``.npy`` writer, and
        returns the device means (``lh_avg_dev`` / ``rh_avg_dev``) that
        ``generate_ini_params_gpu`` consumes directly, so the fp32 means
        never round trip through host memory on the critical path.

    ``avg_profiles_gpu(...)`` — disk-reading supercall, same
        signature/contract as the CPU twin. Decodes the per-subject
        ``.b2nd`` set on host (blosc2 LZ4+bitshuffle has no GPU
        counterpart), uploads one subject at a time into the same
        accumulator, joins the writer, and DROPS the device
        accumulators before returning (``lh_avg_dev`` / ``rh_avg_dev``
        are ``None``) so the caller's ``free_all_blocks()`` can reclaim
        them -- this entry point has no device-side consumer.

cupy is imported at module top — this file is only loaded via the
lazy ``from .avg_profiles_gpu import …`` line in the dispatch branch,
so the CPU path never pays the cupy import cost.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Optional, Tuple

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

    The launch boundary and the one validation layer: dtype, shape
    agreement and C-contiguity are checked here (the kernel indexes raw
    rows), so :meth:`PackedProfileAccumulator.add` only checks the joint
    lh|rh slab's layout before splitting it.
    """
    if packed_NhDb_dev.dtype != cp.uint8:
        raise ValueError(
            f"packed_NhDb_dev must be uint8; got {packed_NhDb_dev.dtype}"
        )
    if acc_VhD_dev.dtype != cp.float32:
        raise ValueError(
            f"acc_VhD_dev must be fp32; got {acc_VhD_dev.dtype}"
        )
    if not (packed_NhDb_dev.flags.c_contiguous
            and acc_VhD_dev.flags.c_contiguous):
        raise ValueError(
            "packed_NhDb_dev and acc_VhD_dev must be C-contiguous: the "
            "kernel indexes raw rows"
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


class PackedProfileAccumulator:
    """Device sums of set bits over every packed session fed to :meth:`add`.

    ``lh`` / ``rh`` are the ``(V_lh, D)`` / ``(V_rh, D)`` fp32 sums and
    ``n_pairs`` counts the (subject, session) slabs folded in. :meth:`add`
    has the fused profile leaf's ``on_packed`` signature, so an instance
    is handed straight to ``generate_subject_profiles_gpu(on_packed=acc.add)``
    and every session is accumulated from the device slab the pack
    kernel just wrote.

    Session structure. Every subject feeds its ``num_sess`` sessions in
    slot order ``0 .. num_sess-1`` -- the leaf's loop and the disk
    path's -- so ``sess_index`` must equal ``n_pairs % num_sess`` at
    every call: a subject that packs fewer or more sessions than the
    cohort's ``num_sess`` is caught at the call that breaks the order,
    not averaged in. ``n_subjects`` is the number of complete subjects.

    Streams. The zero-fill is complete when the constructor returns,
    :meth:`add` launches on the stream current at the call (the leaf's
    compute stream) and the finalize waits for the whole device before
    scaling, so the feeder may use any stream. One feeder at a time:
    ``n_pairs`` is a host counter and the kernel's ``+=`` is not atomic
    across concurrent launches.

    Numerics: identical to the CPU supercall. Every summand is exactly
    0.0f or 1.0f and partial sums stay far below 2**24, so the fp32
    accumulation is integer-exact and order-free -- folding a session in
    the moment it is packed gives the same bits as the CPU's
    subject-by-subject loop over the .b2nd files.
    """

    def __init__(self, V_lh: int, V_rh: int, D: int, num_sess: int) -> None:
        V_lh, V_rh, D, num_sess = int(V_lh), int(V_rh), int(D), int(num_sess)
        if V_lh <= 0 or V_rh <= 0 or D <= 0 or num_sess <= 0:
            raise ValueError(
                f"PackedProfileAccumulator: V_lh={V_lh}, V_rh={V_rh}, D={D}, "
                f"num_sess={num_sess} must all be positive")
        self.V_lh, self.V_rh, self.D = V_lh, V_rh, D
        self.num_sess = num_sess
        self.D_bytes = (D + 7) // 8
        self.lh = cp.zeros((V_lh, D), dtype=cp.float32)
        self.rh = cp.zeros((V_rh, D), dtype=cp.float32)
        # ``cp.zeros`` is an asynchronous memset on the current stream; a
        # feeder on another, non-blocking stream could otherwise fold a
        # session in before the zeros land and lose it to them.
        cp.cuda.get_current_stream().synchronize()
        self.n_pairs = 0
        self._finalized = False

    @property
    def n_subjects(self) -> int:
        """Complete subjects folded in so far (``n_pairs // num_sess``)."""
        return self.n_pairs // self.num_sess

    def add(self, sess_index: int, packed_NDb_dev) -> None:
        """Fold one session's ``(V_lh + V_rh, ceil(D/8))`` uint8 device
        slab in, on the current stream. ``sess_index`` is the leaf's
        session slot and must be the one due next for the current
        subject."""
        if self._finalized:
            raise RuntimeError(
                "PackedProfileAccumulator.add: the mean was already taken "
                "(avg_profiles_from_accumulator); nothing can be added now")
        due = self.n_pairs % self.num_sess
        if int(sess_index) != due:
            raise ValueError(
                f"PackedProfileAccumulator.add: session slot {sess_index} "
                f"arrived where {due} was due (subject {self.n_subjects + 1}, "
                f"num_sess={self.num_sess}): every subject must feed sessions "
                f"0..{self.num_sess - 1} in order")
        a = packed_NDb_dev
        if not isinstance(a, cp.ndarray):
            raise ValueError(
                f"PackedProfileAccumulator.add: slab must be a cupy ndarray; "
                f"got {type(a).__name__}")
        if a.ndim != 2 or int(a.shape[0]) != self.V_lh + self.V_rh:
            raise ValueError(
                f"PackedProfileAccumulator.add: slab shape {tuple(a.shape)} "
                f"!= (V_lh+V_rh, ceil(D/8)) = "
                f"{(self.V_lh + self.V_rh, self.D_bytes)}")
        # dtype, byte width and contiguity are checked at the launch
        # boundary, on the lh half first -- the rh half shares them, so
        # nothing is folded in when they are wrong.
        _accum_packed_session_device(a[:self.V_lh], self.lh, self.D)
        _accum_packed_session_device(a[self.V_lh:], self.rh, self.D)
        self.n_pairs += 1


def avg_profiles_from_accumulator(
    acc: PackedProfileAccumulator,
    targ_mesh: str,
    seed_mesh: str,
    out_dir,
    *,
    save: bool = True,
) -> AvgProfilesResult:
    """Turn the device sums into the cohort mean and start the ``.npy`` write.

    ``acc`` must have been built for ``targ_mesh`` (its hemi sizes are
    checked) and fed complete subjects only: at least one, the last one
    with all ``num_sess`` sessions. The mean is taken IN PLACE on
    ``acc.lh`` / ``acc.rh`` -- they become the result's ``lh_avg_dev``
    / ``rh_avg_dev`` -- and exactly once: a second call on the same
    accumulator raises instead of dividing the means by ``n_pairs``
    again, and ``acc.add`` refuses further sessions. The device is
    synchronised first, so the feeder's stream may still be running its
    last folds when this is called.

    ``save=True`` (default) starts the ``.npy`` pair write on a
    background pool and the result carries a ``writer`` handle the
    caller MUST ``wait()`` before exiting; ``save=False`` writes nothing
    (``writer`` is ``None``) though the result names the canonical
    paths.

    Returns the :class:`AvgProfilesResult` with the host fp32 means
    (``lh_avg`` / ``rh_avg``, living in the pinned D2H buffer -- do not
    mutate before ``writer.wait()``), the device means, and the handle.
    """
    if acc._finalized:
        raise RuntimeError(
            "avg_profiles_from_accumulator: the mean was already taken from "
            "this accumulator; build a new one for another cohort")
    if "fsaverage" not in targ_mesh:
        raise ValueError(
            f"avg_profiles_from_accumulator: only fsaverage* targ_mesh "
            f"supported, got {targ_mesh!r}.")
    if acc.n_pairs <= 0:
        raise ValueError(
            "avg_profiles_from_accumulator: no session was accumulated")
    if acc.n_pairs % acc.num_sess != 0:
        raise ValueError(
            f"avg_profiles_from_accumulator: the last subject is incomplete "
            f"({acc.n_pairs % acc.num_sess} of num_sess={acc.num_sess} "
            f"sessions were folded in)")
    V_lh, V_rh = _hemi_vert_counts(targ_mesh)
    if (V_lh, V_rh) != (acc.V_lh, acc.V_rh):
        raise ValueError(
            f"avg_profiles_from_accumulator: accumulator hemi sizes "
            f"({acc.V_lh}, {acc.V_rh}) != {targ_mesh!r} ({V_lh}, {V_rh})")
    out_dir = Path(out_dir)
    if save:
        (out_dir / "profiles" / "avg_profile").mkdir(parents=True, exist_ok=True)

    # The folds ran on the feeder's stream (the leaf drains its own; a
    # direct caller may not have): wait for the device before scaling.
    cp.cuda.Device().synchronize()
    inv_n = cp.float32(1.0 / float(acc.n_pairs))
    acc.lh *= inv_n
    acc.rh *= inv_n
    acc._finalized = True

    # Single D2H into pinned memory; the writer thread reads that same
    # buffer, so there is exactly one host copy of each hemi.
    lh_acc = _pinned_host_buffer((V_lh, acc.D), np.float32)
    rh_acc = _pinned_host_buffer((V_rh, acc.D), np.float32)
    acc.lh.get(out=lh_acc)
    acc.rh.get(out=rh_acc)

    lh_out, rh_out = _avg_paths(out_dir, targ_mesh, seed_mesh)
    writer: Optional[BackgroundWriteHandle] = None
    if save:
        writer = _start_avg_npy_pair_async(
            out_dir, targ_mesh, seed_mesh, lh_acc, rh_acc,
        )

    return AvgProfilesResult(
        lh_path=lh_out, rh_path=rh_out,
        lh_avg=lh_acc, rh_avg=rh_acc,
        lh_avg_dev=acc.lh, rh_avg_dev=acc.rh,
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
    ``.b2nd`` set, uploads one subject at a time into a
    :class:`PackedProfileAccumulator` and finishes through
    :func:`avg_profiles_from_accumulator`. The ``.npy`` pair write is
    joined before returning, so this call is synchronous
    (``result.writer`` is already ``done``).

    Memory contract (differs from :func:`avg_profiles_from_accumulator`
    on purpose):

    * ``lh_avg_dev`` / ``rh_avg_dev`` are ``None``. Nothing downstream
      of this entry point consumes device buffers -- the disk path
      exists for callers that go on to re-read the ``.npy`` pair or use
      the host arrays -- so retaining the two ``(V_h, D)`` fp32
      accumulators (~385 MB together at fsa6 / Schaefer-400) would just
      pin device memory that ``cp.get_default_memory_pool()
      .free_all_blocks()`` could otherwise reclaim. The accumulator's
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

    sub_paths, packed_arrs, T0, _N0, _D0_bytes, D0 = _discover_and_decode_subjects(
        out_dir, targ_mesh, seed_mesh, num_sub, num_sess, V_lh, V_rh, verbose,
    )

    acc = PackedProfileAccumulator(V_lh, V_rh, D0, T0)
    for (sub, _p), arr in zip(sub_paths, packed_arrs):
        arr_dev = cp.asarray(arr)            # one H2D per subject
        for t in range(T0):
            acc.add(t, arr_dev[t])
        del arr_dev                          # peak device usage: one subject
        if verbose:
            print(f"current subject:{sub}")

    res = avg_profiles_from_accumulator(
        acc, targ_mesh, seed_mesh, out_dir, save=True,
    )
    res.writer.wait()
    # Drop the device accumulators: no consumer on this path, and the
    # frozen dataclass cannot be mutated in place. ``del res`` / ``del
    # acc`` remove the only other references, so the cupy blocks become
    # reclaimable by ``free_all_blocks()`` as soon as we return.
    out = _dc_replace(res, lh_avg_dev=None, rh_avg_dev=None)
    del res, acc
    return out
