"""_kernels.py

Numba kernels for the avg-profiles supercall.

    _accum_packed_session_inplace_kernel(acc_VhxD_flat, packed_VhxDb, D)
        Bit-packed accumulator add for one session × one hemisphere.
        Reads ``(V_h, ⌈D/8⌉) uint8`` and adds 1.0f to
        ``acc_flat[n * D + d]`` for every set bit. Bit-identical to
        "unpack to fp32 + ``acc += slab``" for binary {0, 1} input —
        each summand stays in fp32's exact-integer range regardless of
        how many subjects × sessions are folded in (partial sums
        ≤ S*T < 2^24 for any realistic S, T).

    _scale_inplace_kernel(buf_M, scale)
        In-place ``buf *= scale`` — the divide-by-count step at the end.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False, boundscheck=False)
def _accum_packed_session_inplace_kernel(
    acc_VhxD_flat,    # (V_h * D,) fp32 — in-place accumulator
    packed_VhxDb,     # (V_h, D_bytes) uint8 — one (session, hemi) slab
    D,                # int — true cell count
):
    """Bit-packed -> fp32 accumulator add, bit-identical to
    ``acc += unpack(packed).astype(fp32)`` for binary 0/1 input.

    Bit-convention: cell d <-> bit (d & 7) of byte (d >> 3) (LSB-first,
    matches ``numpy.packbits(bitorder='little')``). Padding bits past
    D in the last byte MUST be zero (the writer guarantees this); the
    inner loop's ``d < D`` guard is defensive.

    fp32 round-off: each summand contributed by one (n, d) cell is
    exactly 0.0f or 1.0f; the accumulator's value at any moment is a
    small non-negative integer (≤ subjects × sessions × number-of-
    accumulations-so-far). fp32 represents integers exactly up to 2^24,
    well beyond any realistic S × T, so no rounding occurs and the
    final value is order-independent — bit-identical to the legacy
    fp32-input path.
    """
    V_h = packed_VhxDb.shape[0]
    D_bytes = packed_VhxDb.shape[1]
    one = np.float32(1.0)

    for n in prange(V_h):
        row_off = n * D
        # Full-byte fast path: bytes whose 8 cells all fit inside D.
        # (b + 1) * 8 <= D <=> b < D // 8. ``b_full`` is the count.
        b_full = D // 8
        for b in range(b_full):
            bv = np.int32(packed_VhxDb[n, b])
            if bv != 0:
                base = b * 8
                if bv & 1:    acc_VhxD_flat[row_off + base + 0] += one
                if bv & 2:    acc_VhxD_flat[row_off + base + 1] += one
                if bv & 4:    acc_VhxD_flat[row_off + base + 2] += one
                if bv & 8:    acc_VhxD_flat[row_off + base + 3] += one
                if bv & 16:   acc_VhxD_flat[row_off + base + 4] += one
                if bv & 32:   acc_VhxD_flat[row_off + base + 5] += one
                if bv & 64:   acc_VhxD_flat[row_off + base + 6] += one
                if bv & 128:  acc_VhxD_flat[row_off + base + 7] += one
        # Trailing-byte slow path: only the bits in [0, D - b_full*8)
        # are valid. For typical D not a multiple of 8 this runs once
        # per row; for D == 0 mod 8 it's skipped.
        for b in range(b_full, D_bytes):
            bv = np.int32(packed_VhxDb[n, b])
            base = b * 8
            for k in range(8):
                d = base + k
                if d < D and ((bv >> k) & 1):
                    acc_VhxD_flat[row_off + d] += one


@njit(cache=True, parallel=True)
def _scale_inplace_kernel(buf_M, scale):
    M = buf_M.shape[0]
    s32 = np.float32(scale)
    for i in prange(M):
        buf_M[i] *= s32
