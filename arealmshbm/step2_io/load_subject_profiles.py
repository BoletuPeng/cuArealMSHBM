"""load_subject_profiles.py — per-session profile normalize helpers.

Provides the private leaf used by
:class:`arealmshbm.step2_io.subject_loaders.SubjectProfileLoader`:

* :func:`_widen_normalize_bitpacked_to_f32_NTD_kernel` — fused
  bit-unpack + per-(n, t) demean + L2-row-norm operating directly on
  bitpacked uint8 input. The on-disk packed bytes flow into this
  kernel without an intermediate fp32 widening.

fp32 throughout (reductions / row-norms in fp32, matching step-1 /
step-3 across the fork).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False, boundscheck=False,
      error_model="numpy", parallel=True)
def _widen_normalize_bitpacked_to_f32_NTD_kernel(
    packed_NTDb,   # (N, T, D_bytes) uint8
    out_NTD,       # (N, T, D)        fp32 — OUT
    D,             # int — true cell count along axis -1 (D_bytes = ⌈D/8⌉)
):
    """Bit-packed -> fp32 widen + per-(n,t) demean + L2-row-norm.

    Matches MATLAB ``CBIG_MSHBM_read_fmri`` semantics: per row, demean
    across D; rows with any post-demean zero cell are left at
    ``(s - mean)`` (MATLAB's ``all(series, 2) ~= 0`` gate), all others
    are L2-normalized. Bit convention (LSB-first, matching
    ``numpy.packbits(bitorder='little')``):

        cell index d  <->  bit (d & 7) of byte (d >> 3)

    Padding bits (d >= D within the last byte) MUST be zero — the
    bit-packed .b2nd writer (``write_subject_profile_tnd``) guarantees
    this via numpy.packbits zero-pad of the trailing input cells
    before packing.

    fp32 round-off contract — bit-identical to a hypothetical
    "unpack to fp32 + 3-pass per-row demean + L2-norm" reference
    (verified bit-equal by
    ``arealmshbm/step2_io/tests/test_subject_loaders.py``):

      * Pass 1 sum: integer popcount cast to fp32. For binary input
        the integer count k <= D < 2^24, so fp32(k) is exact and the
        resulting mean = fp32(k) * inv_D is bit-identical to the
        reference's serial fp32 accumulation of k 1's and (D-k) 0's.
      * Pass 2 sumsq: iterates d in [0, D) in ascending order
        (cell d = 8*b + k, b ascending, k ascending), so the fp32 acc
        is bit-identical to the reference.
      * Pass 3 normalize: identical 1/sqrt(sumsq) fp32 scale.

    prange parallel over flattened (n, t) — each row is independent.
    """
    N = packed_NTDb.shape[0]
    T = packed_NTDb.shape[1]
    D_bytes = packed_NTDb.shape[2]
    inv_D = np.float32(1.0) / np.float32(D)
    zero_f32 = np.float32(0.0)

    NT = N * T
    for nt in prange(NT):
        n = nt // T
        t = nt - n * T

        # Pass 1: row sum via integer popcount of bytes.
        s_sum_i = 0
        for b in range(D_bytes):
            byte_val = np.int32(packed_NTDb[n, t, b])
            # Per-bit unroll. Numba lowers this to bit-test ops; for
            # D=1175 / D_bytes=147 the inner body runs ~1176 times,
            # negligible at modern CPU speeds.
            if byte_val & 1:    s_sum_i += 1
            if byte_val & 2:    s_sum_i += 1
            if byte_val & 4:    s_sum_i += 1
            if byte_val & 8:    s_sum_i += 1
            if byte_val & 16:   s_sum_i += 1
            if byte_val & 32:   s_sum_i += 1
            if byte_val & 64:   s_sum_i += 1
            if byte_val & 128:  s_sum_i += 1
        mean = np.float32(s_sum_i) * inv_D

        # Pass 2: unpack + demean + sumsq.
        # Iterate d via (b, k) so the in-row scan order matches the
        # reference kernel's d=0..D-1 — preserves fp32 acc ordering.
        acc_n2 = np.float32(0.0)
        all_nonzero = True
        for b in range(D_bytes):
            byte_val = np.int32(packed_NTDb[n, t, b])
            base = b * 8
            for k in range(8):
                d = base + k
                if d < D:
                    bit = np.float32((byte_val >> k) & 1)
                    v = bit - mean
                    out_NTD[n, t, d] = v
                    if v == zero_f32:
                        all_nonzero = False
                    acc_n2 += v * v

        # Pass 3: scale if no post-mean zero AND sumsq > 0.
        # The "all_nonzero" gate handles medial-wall rows (k=0,
        # mean=0, every cell post-demean is 0) and fully-constant
        # rows (k=D, mean=1, every cell post-demean is 0): both trip
        # has_zero and skip the divide, leaving them at the demeaned
        # value (0).
        if all_nonzero and acc_n2 > zero_f32:
            inv = np.float32(1.0) / np.float32(math.sqrt(acc_n2))
            for d in range(D):
                out_NTD[n, t, d] *= inv
