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
    * D2H once at the end; ``np.save`` write (parallel across hemis,
      same code path as the CPU supercall).

cupy is imported at module top — this file is only loaded via the
lazy ``from .avg_profiles_gpu import …`` line in the dispatch branch,
so the CPU path never pays the cupy import cost.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from .avg_profiles import (
    AvgProfilesResult,
    _discover_and_decode_subjects,
    _hemi_vert_counts,
    _save_avg_npy_pair,
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
    // bits and conditionally adding 1.0f to 8 acc slots — no atomics
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

    # Bit-packed GPU path:
    #   * Loader returns (T, N, ⌈D/8⌉) uint8 per subject.
    #   * H2D the whole packed array once per subject (~32x less bytes
    #     than the fp32 path: ~12 MB per session at fsa6 vs ~385 MB),
    #     then iterate sessions on device.
    #   * Per (session, hemi) launch ``accum_packed_session_NhDb`` —
    #     one block per hemi vertex; each block fuses the bit-unpack
    #     and the accumulator add. No atomics (each (n, d) slot
    #     belongs to one thread).
    sub_paths, packed_arrs, T0, _N0, _D0_bytes, D0 = _discover_and_decode_subjects(
        out_dir, targ_mesh, seed_mesh, num_sub, num_sess, V_lh, V_rh, verbose,
    )

    lh_acc_dev = cp.zeros((V_lh, D0), dtype=cp.float32)
    rh_acc_dev = cp.zeros((V_rh, D0), dtype=cp.float32)

    n_pairs = 0
    for (sub, _p), arr in zip(sub_paths, packed_arrs):
        # H2D the whole subject once. cp.asarray on a C-contig uint8
        # numpy array is a single transfer.
        arr_dev = cp.asarray(arr)
        for t in range(T0):
            slab_NDb = arr_dev[t]                   # (N, D_bytes) device view
            lh_slab = slab_NDb[:V_lh]               # (V_lh, D_bytes) view
            rh_slab = slab_NDb[V_lh:]               # (V_rh, D_bytes) view
            _accum_packed_session_device(lh_slab, lh_acc_dev, D0)
            _accum_packed_session_device(rh_slab, rh_acc_dev, D0)
            n_pairs += 1
        # Free this subject's device packed slab eagerly so peak
        # device usage stays bounded by one subject (not N_workers).
        del arr_dev
        if verbose:
            print(f"current subject:{sub}")

    inv_n = cp.float32(1.0 / float(n_pairs))
    lh_acc_dev *= inv_n
    rh_acc_dev *= inv_n

    # D2H once. np.save is GIL-bound but the write itself is fast
    # (~100 ms / hemi for 192 MB on NVMe); the pool overlaps lh + rh.
    lh_acc = cp.asnumpy(lh_acc_dev)
    rh_acc = cp.asnumpy(rh_acc_dev)

    lh_out, rh_out = _save_avg_npy_pair(
        out_dir, targ_mesh, seed_mesh, lh_acc, rh_acc,
    )
    return AvgProfilesResult(
        lh_path=lh_out, rh_path=rh_out,
        lh_avg=lh_acc, rh_avg=rh_acc,
    )
