"""bitpacked_norm.py — host-side bit-unpack + per-(n, t) demean + L2-row-norm
of bit-packed BOLD profiles.

The bit-packed ``.b2nd`` is the only on-disk BOLD format produced by
step-1's generate_profiles. Step-3 consumers come in two flavors:

  * **gpu_full** — packed bytes flow straight to device; the fused
    ``_normalize_bold_NTD_from_packed`` CUDA kernel
    (``arealmshbm/vmf_clustering/vmf_clustering_gpu.py``) does
    unpack + demean + L2-norm in one device pass.
  * **cpu / gpu_elambda** — packed bytes are unpacked + normalized on
    host via :func:`unpack_normalize_packed_NTD_host` here, which the
    CPU :class:`VmfClusteringSession` calls from its ``__init__``.

Both the host (this module) and the device (the fused
``_normalize_bold_NTD_from_packed`` CUDA kernel) produce numerically
equivalent ``(N, T, D)`` fp32 BOLD buffers; the only choice is *where*
the unpack+normalize runs.

Upstream invariant — **the kernel does not take a ``mw_full`` kwarg**.
MW rows MUST be zero in the packed input. The authoritative
zero-pass is :func:`arealmshbm.data_io.fetch_data._read_b2nd_series_packed`,
which mirrors MATLAB step3's ``series(medial_mask, :) = 0`` right
before demean+normalize. The kernel's ``has_zero`` gate produces zero
output for zero input rows naturally — see the docstring on
``_normalize_session_bitpacked_numba`` for the proof — but a caller
that constructs a packed buffer outside ``fetch_data`` and skips the
upstream MW-zero pass would get demeaned, L2-normalized vectors at
MW rows downstream, which is silently wrong (a contract violation,
not kernel misbehavior). The legal call sites are:

* :func:`arealmshbm.data_io.fetch_data.fetch_data` → CPU/GPU
  :class:`VmfClusteringSession{CUDA}` ctors (production).
* Tests in :mod:`arealmshbm.data_io.tests.test_bitpacked_norm` that
  explicitly construct a packed buffer; these pin the zero-row
  contract via :func:`test_all_zero_packed_row_stays_zero`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False, boundscheck=False)
def _normalize_session_bitpacked_numba(
    packed_ND_b: np.ndarray,   # (N, D_bytes) uint8 — single-session packed bytes
    D: int,                    # int — true cell count along axis -1
    out_ND: np.ndarray,        # (N, D) fp32 — pre-allocated
) -> None:
    """Bit-packed -> fp32 fused widen + per-row demean + L2-row-norm,
    bit-identical to "unpack to fp32 + per-row demean + L2-norm" on
    binary 0/1 input.

    Bit-identical proof (vs reading the .b2nd via the fp32 view and
    feeding any standard fp32 per-row normalize kernel):

    * Pass-1 fp64 accumulator sums fp32 cells in {0.0, 1.0}. Each
      summand is an exact integer in fp64; partial sums never exceed
      ``D < 2^31`` so no rounding occurs. ``s_sum (fp64) == popcount (int)``.
    * mean = ``fp32(fp64(popcount) / fp64(D))`` — same fp32 cast.
    * Pass-2 ``v = bit - mean`` is two fp32 operands; numba produces
      one fp32 result.
    * sumsq accumulation visits d ascending — same fp64 add order.
    * Pass-3 norm + scale uses the same fp64 sqrt -> fp32 cast and
      fp32 reciprocal multiply.

    Two row classes hit the ``has_zero`` gate and produce all-zero output:

      * **All-zero packed rows** (MW under the upstream contract):
        popcount=0 → mean=0 → all (0 - 0) = 0 → has_zero=True → divide
        skipped. No explicit MW parameter needed.
      * **All-ones packed rows** (degenerate constant input): popcount=D
        → mean=1 → all (1 - 1) = 0 → has_zero=True → divide skipped.
        Matches MATLAB's ``all_nonzero`` gate behavior — a fully
        constant row produces zero, not a normalized constant vector.

    Any post-mean zero on a non-degenerate row also triggers the gate;
    it's the algorithm's general "no L2 step on rows with any
    post-mean zero" rule, not a special-case for MW.
    """
    N = packed_ND_b.shape[0]
    D_bytes = packed_ND_b.shape[1]

    for n in prange(N):
        # Pass 1: integer popcount over the row's byte stream. numba
        # lowers the 8 bit-tests to a register-resident ALU sequence;
        # at D_bytes ~ 147 this is the cheapest possible per-byte loop.
        s_sum_i = 0
        for b in range(D_bytes):
            bv = np.int32(packed_ND_b[n, b])
            if bv & 1:    s_sum_i += 1
            if bv & 2:    s_sum_i += 1
            if bv & 4:    s_sum_i += 1
            if bv & 8:    s_sum_i += 1
            if bv & 16:   s_sum_i += 1
            if bv & 32:   s_sum_i += 1
            if bv & 64:   s_sum_i += 1
            if bv & 128:  s_sum_i += 1

        mean = np.float32(np.float64(s_sum_i) / np.float64(D))

        # Pass 2: bit-unpack on the fly, write (v - mean), accumulate
        # fp64 sumsq, check has_zero. ``(b, k)`` order visits d in
        # ascending 0..D-1.
        sumsq = 0.0          # fp64 accumulator
        has_zero = False
        for b in range(D_bytes):
            bv = np.int32(packed_ND_b[n, b])
            base = b * 8
            for k in range(8):
                d = base + k
                if d < D:
                    bit = np.float32((bv >> k) & 1)
                    v = bit - mean
                    out_ND[n, d] = v
                    if v == np.float32(0.0):
                        has_zero = True
                    sumsq += v * v   # fp32 v*v added to fp64 sumsq

        # Pass 3: L2-normalize if no post-mean zero.
        if has_zero or sumsq == 0.0:
            continue
        norm = np.float32(np.sqrt(sumsq))
        inv = np.float32(1.0) / norm
        for d in range(D):
            out_ND[n, d] = out_ND[n, d] * inv


def unpack_normalize_packed_NTD_host(
    packed_NTD: np.ndarray,   # (N, T, ⌈D/8⌉) uint8, MW rows assumed zero
    D: int,                   # original cell count (D_unpacked)
) -> np.ndarray:
    """Host bit-unpack + per-(n, t) demean + L2-row-norm.

    Iterates ``T`` sessions and calls :func:`_normalize_session_bitpacked_numba`
    per session. Returns a fresh ``(N, T, D)`` fp32 C-contig buffer —
    the layout the CPU :class:`VmfClusteringSession` consumes downstream.

    The per-session ``(N, D)`` scratch buf + strided copy into the
    ``out[:, t, :]`` slot keeps the kernel writing into a C-contig
    destination at the cost of a small recycle alloc; the alloc cost
    is negligible vs the popcount + unpack work.
    """
    if packed_NTD.dtype != np.uint8:
        raise ValueError(
            f"packed_NTD must be uint8; got {packed_NTD.dtype}"
        )
    if packed_NTD.ndim != 3:
        raise ValueError(
            f"packed_NTD must be 3D (N, T, D_bytes); got {packed_NTD.shape}"
        )
    N, T, D_bytes = packed_NTD.shape
    expected_bytes = (int(D) + 7) // 8
    if D_bytes != expected_bytes:
        raise ValueError(
            f"packed_NTD D_bytes ({D_bytes}) != ceil(D/8) ({expected_bytes}) "
            f"for D={D}"
        )
    out = np.empty((N, T, int(D)), dtype=np.float32)
    # Single (N, D) fp32 scratch reused across all T sessions. Reuse is
    # safe because the kernel's pass-2 ``out_ND[n, d] = v`` UNCONDITIONALLY
    # writes every cell in ``buf`` before pass-3 reads any of them (and
    # pass-1 never reads ``buf`` at all). If a future optimization were
    # to make pass-2 skip writes for popcount=0 rows, stale data from
    # session ``t-1`` would leak into session ``t`` here — re-add a
    # per-t alloc if that invariant changes. Hoisting saves T-1
    # ``(N, D) fp32`` allocs per Session ctor (~385 MB each at
    # fsa6/D=1175); at T=6 that's a 5× reduction in transient peak.
    buf = np.empty((N, int(D)), dtype=np.float32)
    for t in range(T):
        packed_ND_b = np.ascontiguousarray(packed_NTD[:, t, :])
        _normalize_session_bitpacked_numba(packed_ND_b, int(D), buf)
        out[:, t, :] = buf
    return out
