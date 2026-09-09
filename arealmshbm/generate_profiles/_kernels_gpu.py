"""_kernels_gpu.py

GPU kernels for the FC-profile leaf — CuPy counterparts to the numba
kernels in :mod:`._kernels`.

    binarize_mwzero_pack_cupy(corr_KxV_dev, mw_v_dev, threshold,
                              packed_VxDb_dev)
        Fused binarize + MW-zero + transpose-pack-along-K. Reads
        ``(K, V) fp32``, writes ``(V, ⌈K/8⌉) uint8`` directly. One
        thread per output byte; warps of 32 threads cover 32 adjacent
        v's at the same byte-index, giving coalesced fp32 reads from
        the K-major input. MW vertices are zeroed at the byte level
        (entire byte = 0). Bit convention matches
        ``numpy.packbits(bitorder='little')``: cell k ↔ bit (k & 7) of
        byte (k >> 3); padding bits past K in the last byte are zero.

    zscore_unit_norm_columns_zerovar_cupy(x_TxN_dev, out_TxN_dev)
        Fused per-column zero-mean + L2-unit-norm. One RawKernel pass:
        thread-block per column, shared-mem fp64 reduction for
        sum / sumsq, then a write-back pass that uses
        ``post_sumsq = sumsq - T * mean²`` to recover the centered L2
        norm without a separate centered-array pass. Matches the CPU
        ``_zscore_unit_norm_columns_kernel``'s fp64 reductions and fp32
        storage; a column with no usable variance is written as exact
        ``0.0f`` rather than leaking ``NaN`` / ``+-inf``, so no
        ``cp.nan_to_num`` sweep is needed downstream.

    exact_kth_smallest_cupy(flat_dev, kth)
    threshold_top_fraction_exact_cupy(corr_dev, fraction)
        Exact ``np.partition(flat, kth)[kth]`` by three streaming
        radix-histogram passes over an order-preserving uint32 key --
        the drop-in for ``cp.partition``, with no permutation
        buffer.

cupy is imported at module top — this file is only loaded from
:mod:`.profiles_subject_gpu`, so the CPU path never pays the cupy
import cost.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import cupy as cp
import numpy as _np


# ─────────────────────────────────────────────────────────────────────
# Fused binarize + MW-zero + transpose-pack-along-K
# ─────────────────────────────────────────────────────────────────────
_BINPACK_BLOCK_V = 128


_binarize_mwzero_pack_kernel = cp.RawKernel(r"""
extern "C" __global__
void binarize_mwzero_pack_KV_to_VDb(
    const float*   __restrict__ corr_KxV,    // (K, V) fp32 C-contig in
    const unsigned char* __restrict__ mw_v,  // (V,) uint8: 1 = MW, 0 = cortex
    float          threshold,
    int            K,
    int            V,
    int            D_bytes,
    unsigned char* __restrict__ packed_VxDb  // (V, D_bytes) uint8 C-contig out
) {
    // One thread = one output byte at (v, kb).
    //
    // Grid:   gridDim.x = ceil(V / blockDim.x),  gridDim.y = D_bytes
    // Block:  blockDim.x = _BINPACK_BLOCK_V (typically 128)
    //
    // Memory access:
    //   * Reads corr_KxV[k * V + v]: within a warp (32 threads, varying
    //     v at fixed k) consecutive v's hit consecutive fp32 cells ->
    //     COALESCED (one 128 B transaction per warp per k row).
    //   * Writes packed_VxDb[v * D_bytes + kb]: stride is D_bytes bytes
    //     between adjacent v's; uint8 writes coalesce through L1/L2.
    //
    // Bit convention: cell k -> bit (k & 7) of byte (k >> 3), LSB-first.
    // Matches numpy.packbits(bitorder='little'). Padding bits past K in
    // the last byte are zero (the byte starts at 0 and only sets bits
    // for k < K).

    const int v  = blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = blockIdx.y;
    if (v >= V) return;

    unsigned int byte = 0u;

    // MW vertex -> emit a zero byte. This enforces the writer-side
    // contract _apply_mw_zero used to enforce on host, but in the
    // packed-byte representation directly.
    if (mw_v[v] == 0u) {
        const int k_base = kb * 8;
        const int k_lim  = (k_base + 8 < K) ? (k_base + 8) : K;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int k = k_base + b;
            if (k < k_lim) {
                const float val = corr_KxV[(size_t)k * (size_t)V + (size_t)v];
                if (val >= threshold) {
                    byte |= (1u << b);
                }
            }
        }
    }
    packed_VxDb[(size_t)v * (size_t)D_bytes + (size_t)kb] =
        (unsigned char)byte;
}
""", "binarize_mwzero_pack_KV_to_VDb")


def binarize_mwzero_pack_cupy(corr_KxV: cp.ndarray,
                               mw_v: cp.ndarray,
                               threshold: float,
                               packed_VxDb: cp.ndarray) -> None:
    """Fused binarize + MW-zero + transpose-pack-along-K, GPU.

    Parameters
    ----------
    corr_KxV : cp.ndarray
        ``(K, V) fp32`` C-contig device array — the per-session
        correlation accumulator after ``*= 1/n_runs`` (i.e. the input
        the current ``(corr_sum_dev >= t).astype(fp32)`` consumes).
    mw_v : cp.ndarray
        ``(V,) uint8`` device array — 1 at medial-wall vertices,
        0 at cortex. Matches ``MARS_label == 1`` on the target mesh.
    threshold : float
        The fp32 threshold value (from
        :func:`threshold_top_fraction_exact_cupy` on the joint
        ``lh|rh`` correlation sum). Compared with ``>=``, matching the
        CPU path.
    packed_VxDb : cp.ndarray
        ``(V, ⌈K/8⌉) uint8`` C-contig device array — the output
        packed bytes. Bit convention matches
        ``numpy.packbits(bitorder='little')``.

    Outputs are written **in place** into ``packed_VxDb``. Padding bits
    past K in the last byte are guaranteed zero. MW rows in the output
    are entirely zero bytes.
    """
    if corr_KxV.dtype != cp.float32:
        raise ValueError(
            f"binarize_mwzero_pack: corr_KxV must be fp32; got {corr_KxV.dtype}"
        )
    if mw_v.dtype != cp.uint8:
        raise ValueError(
            f"binarize_mwzero_pack: mw_v must be uint8; got {mw_v.dtype}"
        )
    if packed_VxDb.dtype != cp.uint8:
        raise ValueError(
            f"binarize_mwzero_pack: packed_VxDb must be uint8; got {packed_VxDb.dtype}"
        )
    if corr_KxV.ndim != 2 or packed_VxDb.ndim != 2 or mw_v.ndim != 1:
        raise ValueError(
            f"binarize_mwzero_pack: shape ranks must be (2D, 1D, 2D); got "
            f"corr={corr_KxV.shape}, mw_v={mw_v.shape}, packed={packed_VxDb.shape}"
        )
    K, V = corr_KxV.shape
    V_out, D_bytes = packed_VxDb.shape
    if V_out != V:
        raise ValueError(
            f"binarize_mwzero_pack: V mismatch — corr has V={V}, packed has V={V_out}"
        )
    if mw_v.shape[0] != V:
        raise ValueError(
            f"binarize_mwzero_pack: mw_v.size ({mw_v.shape[0]}) != V ({V})"
        )
    expected_D_bytes = (int(K) + 7) // 8
    if D_bytes != expected_D_bytes:
        raise ValueError(
            f"binarize_mwzero_pack: D_bytes ({D_bytes}) != ⌈K/8⌉ "
            f"({expected_D_bytes}) for K={K}"
        )
    if not (corr_KxV.flags["C_CONTIGUOUS"] and
            packed_VxDb.flags["C_CONTIGUOUS"] and
            mw_v.flags["C_CONTIGUOUS"]):
        raise ValueError(
            "binarize_mwzero_pack: all arrays must be C-contiguous"
        )

    block_v = _BINPACK_BLOCK_V
    grid_x = (int(V) + block_v - 1) // block_v
    grid_y = int(D_bytes)
    _binarize_mwzero_pack_kernel(
        (grid_x, grid_y), (block_v,),
        (corr_KxV, mw_v, cp.float32(threshold),
         cp.int32(K), cp.int32(V), cp.int32(D_bytes),
         packed_VxDb),
    )


# ---------------------------------------------------------------------
# Per-column zscore + L2-unit-norm
# ---------------------------------------------------------------------
# The reduction tree is the one the retired ``zscore_unit_norm_columns``
# kernel used -- same block size, same fp64 accumulators, same pairing
# -- transcribed into
# ``tests/test_subject_profiles_gpu.py``'s oracle so the equality stays
# pinned. The only difference is what gets written when the column has
# no usable variance -- i.e. when
# ``!(post_sumsq > 0.0) || !isfinite((float)(1/sqrt(post_sumsq)))``,
# which covers the exactly-constant column, the cancellation case, any
# column containing a NaN or +-inf (those poison ``post_sumsq`` into
# NaN), and the fp32 overflow of a near-constant column of very small
# magnitude (post_sumsq positive but < ~8.6e-78):
#
#   retired      inv_norm = +inf  ->  column becomes NaN / +-inf
#                                 ->  every corr entry that touches it
#                                     is non-finite
#                                 ->  ``cp.nan_to_num`` sweeps the whole
#                                     row/column to 0
#   this kernel  store 0.0f       ->  column is exactly zero
#                                 ->  every corr entry that touches it
#                                     is exactly +-0.0f
#
# Both land on bit 0 after the threshold compare (the production
# threshold is a positive correlation and ``-0.0f >= t`` is false for
# any t > 0), so the packed output is unchanged while the
# ``cp.nan_to_num`` pass (a full read-modify-write of corr per session)
# is unnecessary. The only observable difference is the sign of the
# zero, which ``>=`` cannot see (-0.0f == +0.0f in IEEE compare); every
# NON-degenerate column comes out bit-identical.

_ZSCORE_BLOCK = 256

_zscore_zerovar_kernel = cp.RawKernel(r"""
extern "C" __global__
void zscore_unit_norm_columns_zerovar(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int T, int N) {
    const int j = blockIdx.x;
    if (j >= N) return;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (int i = tid; i < T; i += bs) {
        float v = x[(size_t)i * (size_t)N + (size_t)j];
        local_sum += (double)v;
        local_sumsq += (double)v * (double)v;
    }
    __shared__ double s_sum[256];
    __shared__ double s_sumsq[256];
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }
    __shared__ float s_mean;
    __shared__ float s_inv_norm;
    __shared__ int s_degenerate;
    if (tid == 0) {
        double mean_d = s_sum[0] / (double)T;
        double post_sumsq = s_sumsq[0] - (double)T * mean_d * mean_d;
        s_mean = (float)mean_d;
        // NOT ``post_sumsq <= 0.0``: NaN fails both comparisons, and a
        // NaN post_sumsq (a NaN or +-inf somewhere in the column) must
        // take the degenerate branch too.
        //
        // The second half of the test is the fp32 OVERFLOW case:
        // post_sumsq can be a perfectly ordinary positive double and
        // still send 1/sqrt(post_sumsq) past FLT_MAX (it does for
        // post_sumsq < ~8.6e-78, i.e. a near-constant column of
        // magnitude ~1e-38). The legacy kernel stored the resulting
        // +inf and let ``cp.nan_to_num`` mop up downstream; this one
        // has no such backstop, so the overflow must be caught here or
        // it would leak NaN into the corr and the selector.
        float inv_f = (float)(1.0 / sqrt(post_sumsq));
        int ok = (post_sumsq > 0.0) && isfinite(inv_f);
        s_degenerate = ok ? 0 : 1;
        s_inv_norm = ok ? inv_f : 0.0f;
    }
    __syncthreads();
    if (s_degenerate) {
        // Store the literal zero rather than ``(v - mean) * 0.0f``:
        // the multiply would leave NaN in place for a column that
        // contains one, and the whole point is that this column can no
        // longer poison anything downstream.
        for (int i = tid; i < T; i += bs) {
            out[(size_t)i * (size_t)N + (size_t)j] = 0.0f;
        }
    } else {
        for (int i = tid; i < T; i += bs) {
            const size_t o = (size_t)i * (size_t)N + (size_t)j;
            out[o] = (x[o] - s_mean) * s_inv_norm;
        }
    }
}
""", "zscore_unit_norm_columns_zerovar")


def zscore_unit_norm_columns_zerovar_cupy(x_TxN: cp.ndarray,
                                           out_TxN: cp.ndarray) -> None:
    """Per-column zscore + L2-unit-norm; degenerate columns -> exact 0.

    Non-degenerate columns come out bit-identical to the retired
    ``inv_norm = +inf`` formulation (same reduction tree); degenerate
    ones -- no variance, a non-finite value anywhere in the column, or
    an ``inv_norm`` that overflows fp32 -- are written as exact
    ``0.0f`` instead of ``NaN`` / ``+-inf``, which makes a downstream
    ``cp.nan_to_num`` pass unnecessary. See the comment block above.
    """
    if x_TxN.dtype != cp.float32 or out_TxN.dtype != cp.float32:
        raise ValueError(
            f"zscore_zerovar: inputs must be fp32; got {x_TxN.dtype}, "
            f"{out_TxN.dtype}"
        )
    if x_TxN.shape != out_TxN.shape or x_TxN.ndim != 2:
        raise ValueError(
            f"zscore_zerovar: shape mismatch / not 2-D: x={x_TxN.shape}, "
            f"out={out_TxN.shape}"
        )
    if not (x_TxN.flags["C_CONTIGUOUS"] and out_TxN.flags["C_CONTIGUOUS"]):
        raise ValueError("zscore_zerovar: both arrays must be C-contiguous")
    T, N = x_TxN.shape
    if T <= 0 or N <= 0:
        raise ValueError(f"zscore_zerovar: empty input {x_TxN.shape}")
    _zscore_zerovar_kernel((N,), (_ZSCORE_BLOCK,),
                            (x_TxN, out_TxN, cp.int32(T), cp.int32(N)))


# ---------------------------------------------------------------------
# Exact k-th smallest fp32 by three radix-histogram passes
# ---------------------------------------------------------------------
# Replaces ``cp.partition(flat, kth)[kth]`` (GPU introselect, which
# also materialises a second full-size buffer to permute). This returns
# the SAME value -- exactly ``np.partition(flat, kth)[kth]`` -- in
# three streaming passes over the array with no temporary bigger than
# 16 KB.
#
# Key transform (order-preserving fp32 -> uint32):
#
#     key = (u & 0x80000000) ? ~u : (u | 0x80000000)
#
# ascending float order becomes ascending unsigned order. Documented
# corner cases:
#
#   ties / quantized data   fine: bins carry counts, not identities.
#   all-equal array         one non-empty bin per pass; exact.
#   +-inf                   ordered correctly (they are ordinary fp32
#                           bit patterns under this map).
#   +0.0 / -0.0             -0.0 maps BELOW +0.0, whereas np.partition
#                           treats them as equal and may return either.
#                           The returned value can therefore be -0.0
#                           where numpy returned +0.0 (or vice versa).
#                           It is only ever used as the right-hand side
#                           of a ``>=`` compare, where -0.0 and +0.0 are
#                           indistinguishable, so the binarization is
#                           unaffected.
#   NaN                     positive NaN sorts above +inf (numpy also
#                           sorts NaN last); NEGATIVE NaN sorts below
#                           -inf, where numpy would still put it last.
#                           The production path cannot produce a NaN
#                           (``zscore_unit_norm_columns_zerovar_cupy``
#                           kills degenerate columns at the source), so
#                           this is documented, not defended against.

_RADIX_BINS_BITS = (12, 12, 8)      # 32 bits, most significant first
_RADIX_BLOCK = 256
_RADIX_MAX_BLOCKS = 4096

_radix_hist_kernel = cp.RawKernel(r"""
extern "C" __global__
void radix_hist_f32(const float* __restrict__ x,
                     long long n,
                     unsigned int prefix,
                     unsigned int prefix_mask,
                     int shift,
                     int nbins,
                     unsigned int* __restrict__ gbins) {
    // Shared sub-histogram, one atomicAdd per matching element, then
    // one atomicAdd per non-empty bin per block into the global
    // histogram. nbins <= 4096 keeps the shared allocation at 16 KB.
    // Plain shared atomics on purpose: the pass already runs at the
    // box's read bandwidth, so warp aggregation only adds traffic.
    extern __shared__ unsigned int sbin[];
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) sbin[b] = 0u;
    __syncthreads();

    const unsigned int binmask = (unsigned int)(nbins - 1);
    long long i = (long long)blockIdx.x * (long long)blockDim.x
                  + (long long)threadIdx.x;
    const long long stride = (long long)gridDim.x * (long long)blockDim.x;
    for (; i < n; i += stride) {
        unsigned int u = __float_as_uint(x[i]);
        unsigned int key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        if ((key & prefix_mask) == prefix) {
            atomicAdd(&sbin[(key >> shift) & binmask], 1u);
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) {
        unsigned int v = sbin[b];
        if (v) atomicAdd(&gbins[b], v);
    }
}
""", "radix_hist_f32")


def exact_kth_smallest_cupy(flat_dev: cp.ndarray, kth: int) -> float:
    """``float(np.partition(cp.asnumpy(flat_dev), kth)[kth])``, exactly.

    Parameters
    ----------
    flat_dev : cp.ndarray
        fp32 C-contiguous device array (any shape; flattened here).
    kth : int
        0-based rank in ASCENDING order, ``0 <= kth < flat_dev.size``.

    Returns
    -------
    float
        The value at that rank. See the module comment for the
        documented +-0.0 and NaN conventions.

    Three passes, each a full streaming read of ``flat_dev`` with a
    4096- / 4096- / 256-bin shared-memory histogram; the running rank
    is narrowed on the host between passes (16 KB D2H each).
    """
    if flat_dev.dtype != cp.float32:
        raise ValueError(
            f"exact_kth_smallest: array must be fp32; got {flat_dev.dtype}")
    if not flat_dev.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "exact_kth_smallest: array must be C-contiguous "
            "(call cp.ascontiguousarray on the caller side)")
    flat = flat_dev.reshape(-1)
    n = int(flat.size)
    k = int(kth)
    if n == 0:
        raise ValueError("exact_kth_smallest: empty array")
    if not (0 <= k < n):
        raise ValueError(
            f"exact_kth_smallest: kth={k} out of range for size {n}")

    prefix = 0
    prefix_mask = 0
    rank = k
    shift = 32
    n_blocks = min(_RADIX_MAX_BLOCKS,
                   max(1, (n + _RADIX_BLOCK - 1) // _RADIX_BLOCK))
    # One scratch histogram, reused by all three passes (the later
    # passes need fewer bins and just use a prefix of it).
    gbins_all = cp.empty(1 << max(_RADIX_BINS_BITS), dtype=cp.uint32)
    for bits in _RADIX_BINS_BITS:
        shift -= bits
        nbins = 1 << bits
        gbins = gbins_all[:nbins]
        gbins.fill(0)
        _radix_hist_kernel(
            (n_blocks,), (_RADIX_BLOCK,),
            (flat, cp.int64(n), cp.uint32(prefix), cp.uint32(prefix_mask),
             cp.int32(shift), cp.int32(nbins), gbins),
            shared_mem=nbins * 4)
        cum = _np.cumsum(cp.asnumpy(gbins).astype(_np.int64))
        b = int(_np.searchsorted(cum, rank, side="right"))
        if b >= nbins:
            # Unreachable unless the histogram lost elements, which the
            # prefix filter cannot do -- name it rather than index OOB.
            raise RuntimeError(
                f"exact_kth_smallest: rank {rank} beyond the "
                f"{int(cum[-1])} elements counted at shift {shift}")
        rank -= int(cum[b - 1]) if b > 0 else 0
        prefix |= b << shift
        prefix_mask |= (nbins - 1) << shift
    del gbins_all

    key = _np.uint32(prefix)
    u = ((key & _np.uint32(0x7FFFFFFF)) if (int(key) & 0x80000000)
         else _np.uint32(~key))
    return float(_np.asarray([u], dtype=_np.uint32).view(_np.float32)[0])


def threshold_top_fraction_exact_cupy(corr_dev: cp.ndarray,
                                       fraction: float) -> float:
    """Value at rank ``round(N*fraction)`` (1-indexed, descending).

    Same rank arithmetic as :func:`profiles._threshold_top_fraction`
    (half-away-from-zero rounding, clipped to ``[1, numel]``) and the
    same k-th order statistic -- only the selection algorithm differs.
    """
    flat = corr_dev.reshape(-1)
    numel = int(flat.size)
    raw = numel * float(fraction)
    if raw >= 0.0:
        idx_1based = int(_np.floor(raw + 0.5))
    else:
        idx_1based = -int(_np.floor(-raw + 0.5))
    idx_1based = max(1, min(idx_1based, numel))
    return exact_kth_smallest_cupy(flat, numel - idx_1based)
