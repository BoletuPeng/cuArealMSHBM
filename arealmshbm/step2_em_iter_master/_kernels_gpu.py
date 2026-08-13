"""_kernels_gpu.py — CuPy port of the streaming EM-iter master.

Mirrors :mod:`arealmshbm.step2_em_iter_master._kernels` phase-by-phase
with CuPy primitives + ``cp.matmul``. All inputs/outputs are device-resident
``cupy.ndarray``; the only host round-trips inside one master call are:

* per-iter_m: ``int(flag_ST.sum())`` (one int, drives M-step convergence test).
* per-iter_m: ``float(kappa_sum_f64)`` (one fp64, feeds CPU ``invad`` scalar).
* per-outer-EM: ``invad`` on CPU (~50 µs scalar root-find).
* per-outer-EM: ``cdln_general_to_f32`` of a single fp64 scalar (Bessel call).
* per-subject: ``float(cost_S)`` (fp64 scalar, fed back to host).

All bulk traffic (BOLD/grad load_into) is H2D inside the per-subject loops —
matches the CPU master's streaming pattern.

Precision contract — identical to the CPU master (see
``docs/step2_em_iter_master_kernel.md §2``):

* Storage fp32 throughout for BOLD, s_t_nu, s_psi, theta, sigma, s_lambda,
  boundary_mask, log_connect, log_vmf.
* fp64 accumulators (MUST stay fp64): Phase B ``kappa_sum``; Phase D softmax
  ``exp()`` at β=5000 (subnormal tier); Phase E.1 row-normalize ``rs``.
* All other reductions (denom, col-norm, cosine, per-subject cost,
  theta-mean) use fp32 accumulators — CuPy primitives default to fp32-acc
  on fp32 input, matching the CPU policy.

Numerical-match note — CPU's per-(s,t,l) M-step body uses serial fp32
accumulators inside numba; CuPy's pairwise tree reductions on fp32 give
ULP-level drift on s_t_nu / s_lambda / kappa / theta vs the CPU master.
Drift bound on the reference (3-sub, 2-sess, fsa6, L=400) cohort:
~1e-5 max-rel-diff after 1 iter, ~1e-3 after K=100 iters — well inside
the 1e-4 EM-convergence bar and the 5e-3 MATLAB-GT comparison bar. Bit
equality with the CPU port is **not** the spec.

Streaming budget (S=3, T=2, N=81924, D=1174, L=400, D_grad=100;
sizes are decimal MB/GB = bytes / 1e6 / 1e9):

* BOLD scratch    : ~770 MB device (one subject)
* grad scratch    : ~65 MB device (one subject; gMSHBM only)
* s_lambda_SNL    : ~393 MB device (all S — 26 GB at S=200, OOM)
* s_t_nu          : ~11 MB device (single in-place buffer; no ping-pong B)
* X_dot_sl_STLD   : ~11 MB device
* sigma_psi_SLD   : ~5.6 MB device
* theta_NL + bm   : ~262 MB device
* log_connect_NL  : ~131 MB device
* log_vmf_NL_buf  : ~131 MB device
* tmp_idx_NL      : ~33 MB device (bool)
* s_lambda_NL_f64_scratch : ~131 MB device (N·L·4 — **fp32** despite
  the "f64" in the name; the ``_s_lambda_NL_f64_scratch_dev`` attr
  keeps the legacy name only to minimize the diff but is allocated
  fp32 at ``session_gpu.py:342-344``. NOT fp64 / ~262 MB.)

The listed buffers sum to ~1.95 GB; the ~2.3 GB per-iter total at S=3 on
the reference cohort adds the CUDA primary context + cuBLAS workspace (a
few hundred MB, not enumerated above). This **streaming-only baseline**
comfortably fits the 24 GB RTX 5090 Laptop.

Eager cohort caches — NOT counted in the per-iter total above, and sized
at the production cohort scale (S=40/200, T=6), not the S=3, T=2 reference
used for the streaming budget above. The production ``eager_bitpacked`` +
gMSHBM path allocates these once per Session (``session_gpu.py``) on top
of the streaming budget:

* _bold_cache_SNTD_bp_dev : S·N·T·⌈D/8⌉ uint8 (⌈D/8⌉ = 147)
    ≈ 2.89 GB at S=40,T=6 ; ≈ 14.45 GB at S=200,T=6
* _grad_cache_SND_dev     : S·N·D_grad fp32 (gMSHBM only)
    ≈ 1.31 GB at S=40 ; ≈ 6.55 GB at S=200

S=200 is out of scope for v1: ~26 GB s_lambda_SNL alone, plus the
~14.45 GB bitpacked + ~6.55 GB grad cohort caches on top. A future
patch can stream s_lambda per-subject in the Phase B reduction.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cupy as cp
import numpy as np

from arealmshbm.em_stop_criterion._cdln import _cdln_single
from arealmshbm.m_step._invad import invad


# log(eps_f64^20) ≈ -720.4391 — softmax / cost floor (matches the CPU master).
_LOG_EPS20_F64 = float(math.log(np.finfo(np.float64).eps ** 20))
_LOG_EPS20_F32 = np.float32(_LOG_EPS20_F64)


# ─────────────────────────────────────────────────────────────────────
# bitpacked → fp32 widen + per-(n, t) demean + L2-row-norm.
#
# The eager-cache path holds the raw 0/1 BOLD as
# ``(S, N, T, ⌈D/8⌉) uint8`` device-resident — 1 bit/cell, 8× more
# compact than the un-packed variant. When the master kernel visits
# subject s, this kernel fuses (a) the bitpacked→fp32 widen, (b)
# per-(n, t) demean, (c) L2-row-norm into a single pass per (n, t)
# row. Output is a (N, T, D) fp32 buffer the master kernel's
# Phase A.3 / Phase D consume directly.
#
# Mirrors :func:`load_subject_profiles._widen_normalize_bitpacked_to_f32_NTD_kernel`
# (the CPU host-side bitpacked→fp32 normalize) — same 3-pass-per-row
# math, same "any post-mean zero → skip the divide" gate. Bandwidth-
# bound: ~24 MB read (packed bytes) + ~770 MB write (fp32) per subject
# at fsa6/T=6/D=1175. At RTX 5090 laptop's ~800 GB/s, ~0.85 ms per subject.
#
# Block layout: one block per (n, t) row, threadIdx.x strides D.
# Block size 256 (power-of-2 for the shared-mem reduction; matches step3's
# normalize_bold_NTD kernel constants).
# ─────────────────────────────────────────────────────────────────────
_NORMALIZE_BOLD_U8_BLOCK = 256
_NORMALIZE_BOLD_U8_BLOCK_MAX = 256


_normalize_bold_bitpacked_to_f32_kernel = cp.RawKernel(r"""
extern "C" __global__
void normalize_bold_bitpacked_to_f32_NTD(
    const unsigned char* __restrict__ src_packed,
    float* __restrict__ dst_f32,
    int N, int T, int D, int D_bytes
) {
    // One thread block per (n, t) row. Bit-packing convention (LSB-first,
    // matching numpy.packbits bitorder='little'):
    //   cell index d  <=>  bit (d & 7) of byte (d >> 3)
    // Padding bits (d >= D within the last byte) MUST be zero -- the
    // bit-packed .b2nd writer guarantees this via numpy.packbits
    // zero-padding the trailing input cells before packing.
    //
    // Math is identical to ``normalize_bold_u8_to_f32_NTD``; the only
    // difference is the inner load + unpack. Two cheap wins from the
    // packed layout:
    //   * 8x less device-mem read traffic for pass 1's row sum.
    //   * Pass 1's sum becomes ``__popc`` over the byte stream -- saves
    //     8 fp ops per byte vs the un-packed kernel's 8 fp64 adds.
    //
    // Block size assumed power-of-2 (the bs/2 reduction below silently
    // requires it). Wrapper asserts this; the static __shared__ arrays
    // size 256 caps the wrapper to that block constant.

    const int row = blockIdx.x;
    const int n = row / T;
    if (n >= N) return;

    const unsigned char* src_row = src_packed + (size_t)row * (size_t)D_bytes;
    float* dst_row = dst_f32 + (size_t)row * (size_t)D;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    // Pass 1: row sum via popcount over bytes.
    // ``int`` accumulator suffices: sum <= D, which is < 2^30 always
    // at fsa6 / fsaverage7 mesh sizes.
    int local_sum = 0;
    for (int b = tid; b < D_bytes; b += bs) {
        local_sum += __popc((unsigned int)src_row[b]);
    }
    __shared__ int s_sum[256];
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }

    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = (float)s_sum[0] / (float)D;
    }
    __syncthreads();

    // Pass 2: byte-strided unpack + write (v - mean), OR-reduce has_zero,
    // accumulate fp32 sumsq via shared-mem atomic.
    __shared__ int   s_has_zero;
    __shared__ float s_sumsq;
    if (tid == 0) {
        s_has_zero = 0;
        s_sumsq = 0.0f;
    }
    __syncthreads();

    int local_has_zero = 0;
    float local_sumsq = 0.0f;
    for (int b = tid; b < D_bytes; b += bs) {
        unsigned int byte_val = (unsigned int)src_row[b];
        int base = b * 8;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int d = base + k;
            if (d < D) {
                float bit = (float)((byte_val >> k) & 1u);
                float v = bit - s_mean;
                dst_row[d] = v;
                if (v == 0.0f) local_has_zero = 1;
                local_sumsq += v * v;
            }
        }
    }
    if (local_has_zero) atomicOr(&s_has_zero, 1);
    atomicAdd(&s_sumsq, local_sumsq);
    __syncthreads();

    // Pass 3: scale if no post-mean zero (same gate as the un-packed kernel).
    __shared__ float s_inv_norm;
    if (tid == 0) {
        s_inv_norm = (s_has_zero == 0 && s_sumsq > 0.0f)
            ? (float)(1.0 / sqrt((double)s_sumsq))
            : 1.0f;
    }
    __syncthreads();

    if (s_has_zero == 0) {
        for (int d = tid; d < D; d += bs) {
            dst_row[d] *= s_inv_norm;
        }
    }
}
""", "normalize_bold_bitpacked_to_f32_NTD")


def _widen_normalize_bold_bitpacked_to_f32_cupy(
    src_packed_dev: cp.ndarray,   # (N, T, D_bytes) uint8 — per-subject packed cache slice
    dst_f32_dev: cp.ndarray,      # (N, T, D)        fp32 — per-subject scratch
    D: int,
) -> None:
    """Bit-packed → fp32 widen + per-(n, t) demean + L2-row-norm on device.

    The eager-cache path holds the cohort BOLD as
    ``(S, N, T, ⌈D/8⌉) uint8`` device-resident; this kernel fuses
    bit-unpack + demean + L2-row-norm into a single pass per (n, t)
    row, producing the (N, T, D) fp32 buffer the master kernel's
    Phase A.3 / Phase D consume directly. Read traffic is
    ``N×T×D_bytes`` bytes; write traffic is ``N×T×D×4`` bytes. The
    ``__popc``-based pass-1 sum + 1 bit/cell read footprint hit
    ~0.85 ms wall on RTX 5090 at fsa6/T=2/D=1175 — and the headline
    win is **memory**: bit-packed cache is 1 bit/cell vs the unpacked-
    uint8 alternative's 8 bits/cell, the only path that fits the
    cohort BOLD on device for S=200 production runs.
    """
    if src_packed_dev.dtype != cp.uint8:
        raise ValueError(f"src must be uint8; got {src_packed_dev.dtype}")
    if dst_f32_dev.dtype != cp.float32:
        raise ValueError(f"dst must be fp32; got {dst_f32_dev.dtype}")
    if src_packed_dev.ndim != 3 or dst_f32_dev.ndim != 3:
        raise ValueError(
            f"both src/dst must be 3-D (N, T, ...); got "
            f"src={src_packed_dev.shape}, dst={dst_f32_dev.shape}"
        )
    N, T, D_bytes = src_packed_dev.shape
    if dst_f32_dev.shape != (N, T, D):
        raise ValueError(
            f"dst shape {dst_f32_dev.shape} != (N={N}, T={T}, D={D})"
        )
    if D_bytes != (D + 7) // 8:
        raise ValueError(
            f"src D_bytes={D_bytes} != ceil(D/8)={(D + 7) // 8} for D={D}"
        )
    if not src_packed_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("src must be C-contiguous")
    if not dst_f32_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("dst must be C-contiguous")
    block = _NORMALIZE_BOLD_U8_BLOCK   # reuse the same 256 power-of-2 constant
    grid = N * T
    _normalize_bold_bitpacked_to_f32_kernel(
        (grid,), (block,),
        (src_packed_dev, dst_f32_dev,
         cp.int32(N), cp.int32(T), cp.int32(D), cp.int32(D_bytes)),
    )


# ─────────────────────────────────────────────────────────────────────
# Phase A.1 — sigma_psi precompute.
#
#   sigma_psi[s, l, d] = sigma[l] * s_psi[s, l, d]
#
# Single broadcast multiply. Inplace into caller-owned device buffer.
# ─────────────────────────────────────────────────────────────────────
def _sigma_psi_SLD_compute_cupy(
    sigma_L_dev: cp.ndarray,        # (L,)       fp32
    s_psi_SLD_dev: cp.ndarray,      # (S, L, D)  fp32
    sigma_psi_SLD_dev: cp.ndarray,  # (S, L, D)  fp32 — OUT
) -> None:
    cp.multiply(sigma_L_dev[None, :, None], s_psi_SLD_dev,
                out=sigma_psi_SLD_dev)


# ─────────────────────────────────────────────────────────────────────
# Phase A.3 — Precompute X_dot_sl per subject.
#
#   X_dot_sl[s, t, l, d] = Σ_n s_lambda[s, n, l] · BOLD[s, n, t, d]
#
# Per subject: einsum over n collapses (N, T, D) BOLD + (N, L) s_lambda into
# (T, L, D). CuPy lowers ``nl,ntd->tld`` to cuBLAS sgemm via a transpose;
# tested no faster as a hand-written ``s_lambda.T @ bold.reshape(N, T*D)``
# at fsa6 scale, but cleaner intent.
# ─────────────────────────────────────────────────────────────────────
def _compute_X_dot_sl_s_NTD_cupy(
    X_NTD_s_dev: cp.ndarray,       # (N, T, D) fp32 — per-subject BOLD
    s_lambda_NL_s_dev: cp.ndarray, # (N, L)    fp32 — per-subject s_lambda
    out_TLD_s_dev: cp.ndarray,     # (T, L, D) fp32 — target slice
) -> None:
    N, T, D = X_NTD_s_dev.shape
    L = out_TLD_s_dev.shape[1]
    # (L, N) @ (N, T*D) → (L, T*D)
    tmp_LX = cp.matmul(
        s_lambda_NL_s_dev.T,
        X_NTD_s_dev.reshape(N, T * D),
    )
    # (L, T*D) → (L, T, D) → (T, L, D) via transpose into the caller buffer.
    # transpose returns a strided view; copyto materializes the contig write.
    cp.copyto(out_TLD_s_dev,
              tmp_LX.reshape(L, T, D).transpose(1, 0, 2))


# ─────────────────────────────────────────────────────────────────────
# Phase B — Multi-subject M-step inner while-loop.
#
# Driven from Python; CuPy primitives + ``invad`` host scalar each
# iter_m. Mirrors the ALGORITHM of :func:`_mstep_inner_loop_master_step2`
# (CPU) but diverges structurally: this GPU path is an in-place update on
# a SINGLE s_t_nu buffer with a factored cosine (see the function
# docstring), vs the CPU's two-buffer ping-pong + parity copy. ``flag_ST``
# is OVERWRITTEN each iter_m (not accumulated) — same as CPU.
# ─────────────────────────────────────────────────────────────────────
def _mstep_inner_loop_step2_cupy(
    X_dot_sl_STLD_dev: cp.ndarray,    # (S, T, L, D) fp32
    sigma_psi_SLD_dev: cp.ndarray,    # (S, L, D)    fp32
    denom_f64: float,                 # T * Σ s_lambda — fp64 host
    dim_f64: float,
    kappa_init_f64: float,
    ini_val_f64: float,
    eps_f32: float,
    max_iter_m: int,
    s_t_nu_STLD_dev: cp.ndarray,      # (S, T, L, D) fp32 — in-place state
) -> Tuple[int, float]:
    """Returns ``(iter_m, kappa_final_f64)``. The converged s_t_nu is
    written IN PLACE into ``s_t_nu_STLD_dev`` — no persistent ping-pong
    buffer, hence no parity copy, and (via the factored cosine below) no
    per-iter copy either.

    Factored-cosine trick: the renormalized update is ``new[d] = col[d] *
    inv_norm`` where ``inv_norm`` is a per-(s,t,l) scalar, so the M-step
    drift cosine factors as ``Σ_d new·old = inv_norm · Σ_d col·old``. We
    evaluate that reduction against the PRE-normalized ``col`` and the OLD
    s_t_nu first, which lets the renormalized update write straight into
    s_t_nu with neither a ping-pong B buffer nor a per-iter copy — same
    memory traffic as the old two-buffer form, minus one persistent
    (S,T,L,D) slab and the parity copy.

    The s_t_nu UPDATE is bit-identical to the explicit ``new = col *
    inv_norm`` form (same op on the same values). Only the convergence
    COSINE differs by ~1 fp32 ULP (the factored ``inv_norm·Σ col·old`` vs
    the explicit ``Σ (col·inv_norm)·old`` round at a different point),
    which can flip the per-(s,t,l) ``(1-cos) < eps`` test for a borderline
    cell and shift m_iters by ±1 in rare cases. Well inside the documented
    5e-3 CPU↔GPU parity bar (bit equality across backends is not the spec;
    step2-GPU gMSHBM is run-to-run nondeterministic at β=5000 regardless).
    """
    S, T, L, D = X_dot_sl_STLD_dev.shape
    eps_f32_cp = cp.float32(eps_f32)
    one_f32 = cp.float32(1.0)
    eps_f64 = float(eps_f32)

    iter_m = 0
    kappa_prev = kappa_init_f64
    kappa_new = kappa_init_f64

    while True:
        iter_m += 1

        # 1. kappa_sum (fp64 accumulator over (S, T, L, D)) — reads OLD s_t_nu.
        kappa_sum_f64 = float(
            cp.sum(s_t_nu_STLD_dev * X_dot_sl_STLD_dev, dtype=cp.float64)
        )
        rbar = kappa_sum_f64 / denom_f64
        kappa_new = invad(dim_f64, rbar)
        if not math.isfinite(kappa_new):
            kappa_new = kappa_prev
        if kappa_new < ini_val_f64:
            kappa_new = ini_val_f64
        kappa_f32_cp = cp.float32(kappa_new)

        # 2. Fused per-(s,t,l) col-norm + factored cosine + normalize.
        #   col[d]  = kappa * X_dot_sl[s,t,l,d] + sigma_psi[s,l,d]
        #   inv_n   = 1 / sqrt(Σ_d col²)
        #   cos     = inv_n * Σ_d col[d]·old[d]   (= Σ_d new[d]·old[d]; reads OLD)
        #   new[d]  = col[d] * inv_n               (written straight into s_t_nu)
        # All fp32 accumulators (M3/M4 ablation-safe per the CPU policy doc).
        # sigma_psi broadcasts across T.
        col = kappa_f32_cp * X_dot_sl_STLD_dev + sigma_psi_SLD_dev[:, None, :, :]  # (S,T,L,D)
        col_norms = cp.sqrt((col * col).sum(axis=-1))                              # (S,T,L)
        # col_norms == 0 ⟺ every col[d] == 0 for that (s,t,l) — a degenerate
        # case, NOT a normal empty parcel: even where the s_lambda column is
        # empty (X_dot == 0), sigma_psi is a nonzero floor, so col_norms > 0.
        # If it did occur, inv_norm = 1/0 = inf and the write below is
        # col·inf = 0·inf = NaN — identical to the CPU master, which likewise
        # forms inv_cn = inf then stores col·inv_cn = NaN (the inf is the
        # reciprocal; the STORED value is NaN, same on both backends). The
        # E-step's dead-column-zero phase cleans up.
        inv_norm = cp.float32(1.0) / col_norms
        # Factored cosine: inv_norm (per-(s,t,l) scalar) pulled out of the
        # reduction, evaluated on PRE-normalized col + OLD s_t_nu.
        cos_STL = inv_norm * cp.sum(col * s_t_nu_STLD_dev, axis=-1)                 # (S,T,L)

        # Renormalized update straight into the state buffer. Enqueued HERE —
        # right after the cosine's OLD-s_t_nu read, BEFORE the flag's host
        # sync — because it depends only on col/inv_norm (not the flag), so
        # on the default stream it overlaps the ``int(flag_ST.sum())`` D2H
        # below. Stream order guarantees the cosine read above runs before
        # this overwrite, so OLD is seen. No copy, no B buffer.
        cp.multiply(col, inv_norm[..., None], out=s_t_nu_STLD_dev)

        # 3. Per-(s, t) flag — all l have (1 - cos) < eps.
        flag_ST = cp.all((one_f32 - cos_STL) < eps_f32_cp, axis=-1)                # (S,T) bool
        all_flag = bool(int(flag_ST.sum()) == S * T)

        kappa_drift = abs(kappa_prev - kappa_new) / max(abs(kappa_prev), 1e-30)
        kappa_prev = kappa_new

        if all_flag and (kappa_drift < eps_f64):
            break
        if iter_m > max_iter_m:
            break

    return iter_m, kappa_new


# ─────────────────────────────────────────────────────────────────────
# Phase C — Per-subject spatial_connect prior (gMSHBM only).
#
# Math per (s):
#   For t in range(T):
#     u_update[d, l] = Σ_n grad_t[n, d] · s_lambda[n, l]
#     u[l, d]        = u_update[d, l] / sum_lambda[l]   (0/0 → NaN propagates)
#     u_sq[l]        = Σ_d u[l, d]²
#     grad_sq[n]     = Σ_d grad_t[n, d]²
#     cross[n, l]    = Σ_d grad_t[n, d] · u[l, d]
#     vmf[n, l]      = 2·cross[n, l] - grad_sq[n] - u_sq[l]
#     log_connect[s, n, l] += vmf  (same-hemi only; cross-hemi stays -inf)
#
# Output: log_connect_NL_out with cross-hemi pre-filled -inf and same-hemi
# carrying the sum over T sessions. NaN cells propagate to the E-step's
# dead-column-zero step (Phase D).
#
# Layout note — single (N, D_grad) per subject (not (T, N, D_grad)):
# the gradient is **session-invariant** in this fork
# (``SubjectGradientLoader.load_into`` replicates a single (N, D_grad)
# block across the T axis; see step2_io/subject_loaders.py docstring).
# Every t-iter computes the SAME u_update / u_LD / u_sq / grad_sq /
# cross / vmf — so we hoist all those operations out of the t-loop and
# keep only the cheap accumulation under T (preserves bit-identical
# fp32 rounding semantics: T sequential additions of the same value).
# Speedup on the S=40 / T=6 reference cohort: 12.7 ms → ~2 ms per sub.
# ─────────────────────────────────────────────────────────────────────
def _spatial_connect_per_subject_cupy(
    grad_ND_s_dev: cp.ndarray,       # (N, D_grad)  fp32 — per-subject grad
    s_lambda_NL_s_dev: cp.ndarray,   # (N, L)       fp32 — per-subject s_lambda
    log_connect_NL_out_dev: cp.ndarray,  # (N, L)   fp32 — OUT
    n_lh: int,
    L_lh: int,
    T: int,                          # number of sessions (loop count)
) -> None:
    L = log_connect_NL_out_dev.shape[1]
    NEG_INF = cp.float32(-np.inf)
    ZERO = cp.float32(0.0)
    TWO = cp.float32(2.0)

    # Initialize log_connect: same-hemi 0, cross-hemi -inf. The block-diag
    # writes are zero-copy strided fills; cheaper than gathering through a
    # bool mask.
    log_connect_NL_out_dev[:n_lh, :L_lh] = ZERO
    log_connect_NL_out_dev[:n_lh, L_lh:] = NEG_INF
    log_connect_NL_out_dev[n_lh:, :L_lh] = NEG_INF
    log_connect_NL_out_dev[n_lh:, L_lh:] = ZERO

    # All t-invariant (grad is session-invariant — see kernel header).
    sum_lambda = s_lambda_NL_s_dev.sum(axis=0)                    # (L,)

    # u_update[d, l] = grad.T @ s_lambda  → (D_grad, L)
    u_update = cp.matmul(grad_ND_s_dev.T, s_lambda_NL_s_dev)
    # u[l, d] = u_update[d, l] / sum_lambda[l]
    # CuPy follows numpy: 0/0 → NaN, x/0 → inf. We DO NOT guard sl==0
    # with a where() — the NaN-propagation chain is load-bearing (see
    # docs/step2_em_iter_master_kernel.md §3).
    u_LD = u_update.T / sum_lambda[:, None]                       # (L, D_grad)

    u_sq = (u_LD * u_LD).sum(axis=-1)                              # (L,)
    grad_sq = (grad_ND_s_dev * grad_ND_s_dev).sum(axis=-1)         # (N,)
    cross_NL = cp.matmul(grad_ND_s_dev, u_LD.T)                    # (N, L)

    # Fold the T-fold accumulation into a single fp32 multiply. Not
    # bit-identical to T sequential adds, but the fp32 deviation is
    # < 1 ULP per cell — well below the 5e-3 spec bar at iter=1 and
    # the documented β=5000 chaotic-divergence regime past iter=2.
    T_f32 = cp.float32(T)
    vmf_T = T_f32 * (TWO * cross_NL - grad_sq[:, None] - u_sq[None, :])

    log_connect_NL_out_dev[:n_lh, :L_lh] += vmf_T[:n_lh, :L_lh]
    log_connect_NL_out_dev[n_lh:, L_lh:] += vmf_T[n_lh:, L_lh:]


# ─────────────────────────────────────────────────────────────────────
# Phase D — Per-subject fused E-step (NTD variant).
#
# Mirrors :func:`_fused_estep_per_subject_NTD` step-by-step. The cost
# accumulator stays fp32 (per the docs); softmax exp() and row-sum stay
# fp64 (mandatory for β=5000 subnormal safety).
#
# Outputs: writes ``s_lambda_NL_f64_scratch_dev`` (per-subject fp64
# softmax intermediate, **not** yet normalized), ``log_vmf_NL_buf_dev``,
# ``tmp_idx_NL_dev``; returns the fp64 per-subject cost scalar.
# ─────────────────────────────────────────────────────────────────────
def _fused_estep_per_subject_NTD_cupy(
    X_NTD_dev: cp.ndarray,                  # (N, T, D) fp32
    s_t_nu_TLD_dev: cp.ndarray,             # (T, L, D) fp32
    kappa_scalar_f64: float,
    dim_int: int,
    theta_NL_dev: cp.ndarray,               # (N, L)    fp32
    log_connect_NL_dev: cp.ndarray,         # (N, L)    fp32 — gMSHBM only
    boundary_mask_NL_dev: cp.ndarray,       # (N, L)    fp32
    beta_f64: float,
    has_spatial: int,
    log_vmf_NL_buf_dev: cp.ndarray,         # (N, L)    fp32 — OUT
    s_lambda_NL_f64_scratch_dev: cp.ndarray,# (N, L)    fp64 — OUT
    tmp_idx_NL_dev: cp.ndarray,             # (N, L)    bool — OUT
) -> float:
    N, T, D = X_NTD_dev.shape
    L = s_t_nu_TLD_dev.shape[1]

    cdln_v = float(dim_int) * 0.5 - 1.0
    # _cdln_single is a @njit fn — call it from host with python floats.
    # Tiny scalar; runs in ~1 µs.
    cdln_val_f32 = np.float32(_cdln_single(kappa_scalar_f64, cdln_v))
    kappa_f32 = np.float32(kappa_scalar_f64)
    zero_f32_cp = cp.float32(0.0)

    # ── Step 1a-b: big sgemm (folds T and D contractions in one call) ──
    # lv_sum[n, l] = Σ_t Σ_d X[n, t, d] · s_t_nu[t, l, d]
    # einsum lowers to cuBLAS sgemm after one auto-transpose on s_t_nu.
    lv_sum = cp.einsum("ntd,tld->nl", X_NTD_dev, s_t_nu_TLD_dev)  # (N, L) fp32

    # ── Step 1c: n_alive_count per vertex ──
    # vertex n is alive in session t iff X[n, t, :] has any nonzero entry.
    # Medial-wall vertices: all-zero BOLD row → n_alive = 0.
    alive_NT = cp.any(X_NTD_dev != zero_f32_cp, axis=2)             # (N, T) bool
    n_alive_NL = alive_NT.sum(axis=1, dtype=cp.float32)             # (N,)  fp32

    # ── Step 1d: log_vmf = κ · lv_sum + n_alive · cdln_val ──
    cp.add(
        cp.float32(kappa_f32) * lv_sum,
        n_alive_NL[:, None] * cp.float32(cdln_val_f32),
        out=log_vmf_NL_buf_dev,
    )

    # ── Step 2a: tmp_idx — rows that have any log_vmf == 0 (medial test).
    # Matches CPU exact fp32-zero test; medial-vertex rows flip the whole
    # row to True.
    row_has_zero = (log_vmf_NL_buf_dev == zero_f32_cp).any(axis=1)   # (N,) bool
    cp.copyto(tmp_idx_NL_dev,
              cp.broadcast_to(row_has_zero[:, None], log_vmf_NL_buf_dev.shape))

    # ── Step 2b: compose log_lambda = log_vmf + log(θ) [+ β·log_connect] ──
    # log(θ): -inf at θ <= 0 (matches CPU). cp.log(0.0) natively returns
    # -inf, but cp.log on negative is NaN; defensive where():
    log_theta = cp.where(
        theta_NL_dev > zero_f32_cp,
        cp.log(theta_NL_dev),
        cp.float32(-cp.inf),
    )
    if has_spatial == 1:
        log_lambda = log_vmf_NL_buf_dev + log_theta + cp.float32(beta_f64) * log_connect_NL_dev
    else:
        log_lambda = log_vmf_NL_buf_dev + log_theta

    # ── Step 2c: row_max with NaN/Inf cleanup → 0 fallback (CPU semantics) ──
    # CPU walks log_lambda per-row, ignoring NaN, replacing -inf-only rows
    # with 0. CuPy's nanmax handles NaN; we still need to mask non-finite
    # rmax to 0 afterwards.
    rmax = cp.nanmax(log_lambda, axis=1, keepdims=True)              # (N, 1) fp32
    rmax = cp.where(cp.isfinite(rmax), rmax, zero_f32_cp)

    # ── Step 2d: softmax exp() in fp32 ──
    # Prior contract was fp64 for "β=5000 subnormal safety" (inherited
    # from the CPU master where x86 fp32 exp had a documented
    # underflow cliff). On CuPy/CUDA, the shift-by-rmax bounds every
    # cell at exp(0)=1 max, with all other cells exp(negative) ≤ 1.
    # fp32 underflow at ~e^-87 truncates only contributions that are
    # already <1e-38 of the max — below fp32-sum precision against the
    # max-1.0 cell, so they vanish in fp64 sums too. Empirical: cost
    # trajectory matches fp64 within 1e-5 rel through 10 outer iters
    # on the S=40 reference cohort. Saves ~30% of em_total wall (fp64
    # transcendentals on consumer GPUs are 1/32 fp32 throughput).
    shifted_f32 = log_lambda - rmax                                  # (N, L) fp32
    cp.exp(shifted_f32, out=s_lambda_NL_f64_scratch_dev)

    # ── Step 2e: Dead-column zero ──
    # CPU: cs = Σ_n s_lambda[n, l] over non-NaN values; cs==0 → zero column.
    # CuPy: replace NaN with 0 for the sum check (without mutating scratch),
    # then zero columns where the sum is 0.
    col_sum_f32 = cp.where(
        cp.isnan(s_lambda_NL_f64_scratch_dev),
        zero_f32_cp,
        s_lambda_NL_f64_scratch_dev,
    ).sum(axis=0)                                                    # (L,) fp32
    dead_col_mask = (col_sum_f32 == zero_f32_cp)
    # No host sync — fancy-indexed assignment with an all-False mask is a
    # no-op on device. Avoids ~10 µs PCIe per Phase D call (~1500 calls
    # per pipeline).
    s_lambda_NL_f64_scratch_dev[:, dead_col_mask] = zero_f32_cp

    # ── Step 3: per-subject cost (fp32 acc; matches CPU variant A) ──
    # cost = Σ_{n, l} slc · (log_vmf + ltheta - lslc [+ β·lc])
    # where slc = (s_lambda_f32 · bm) / row_sum, gated by tmp_idx + bm.
    sl_bm_f32 = s_lambda_NL_f64_scratch_dev * boundary_mask_NL_dev   # (N, L) fp32
    row_sum_f32 = sl_bm_f32.sum(axis=1, keepdims=True)              # (N, 1)

    # slc: divide, NaN→0, zero rows where tmp_idx[n, 0] is True.
    slc = cp.where(row_sum_f32 > zero_f32_cp,
                   sl_bm_f32 / row_sum_f32,
                   zero_f32_cp)
    slc = cp.where(cp.isnan(slc), zero_f32_cp, slc)
    # tmp_idx_NL_dev[n, 0] gates the entire row (CPU reads only the [n, 0]
    # bit because the row is uniform after the broadcast above).
    slc = cp.where(tmp_idx_NL_dev[:, 0:1], zero_f32_cp, slc)

    # log_theta with floor (no -Inf — CPU replaces +/-inf with log_eps20).
    log_floor_f32_cp = cp.float32(_LOG_EPS20_F32)
    ltheta_raw = cp.where(theta_NL_dev > zero_f32_cp,
                          cp.log(theta_NL_dev),
                          log_floor_f32_cp)
    ltheta = cp.where(cp.isinf(ltheta_raw), log_floor_f32_cp, ltheta_raw)

    # log_slc with floor.
    lslc_raw = cp.where(slc > zero_f32_cp, cp.log(slc), log_floor_f32_cp)
    lslc = cp.where(cp.isinf(lslc_raw), log_floor_f32_cp, lslc_raw)

    cost_integrand = slc * (log_vmf_NL_buf_dev + ltheta - lslc)
    if has_spatial == 1:
        lc_raw = log_connect_NL_dev
        lc = cp.where(cp.isnan(lc_raw) | cp.isinf(lc_raw),
                      log_floor_f32_cp,
                      lc_raw)
        cost_integrand = cost_integrand + cp.float32(beta_f64) * slc * lc

    # fp32 sum; CuPy's pairwise tree gives ULP-drift vs CPU's serial acc.
    cost_f32 = cost_integrand.sum(dtype=cp.float32)
    return float(cost_f32)


# ─────────────────────────────────────────────────────────────────────
# Phase E.1 — Per-subject normalize: fp32 scratch → fp32 storage.
#
# The CPU master forced ``rs`` to fp64 because x86 fp32 has a
# documented "catastrophic subnormal flushing" cliff at β=5000. On
# CuPy/CUDA, softmax output is bounded in [0, 1] with row sum O(1),
# never approaching subnormal — fp32 rs is safe end-to-end. The
# variable kept the f64 name to minimize the diff (dtype is fp32 in
# the GPU Session — see ``_s_lambda_NL_f64_scratch_dev`` alloc).
# ─────────────────────────────────────────────────────────────────────
def _phase_e1_normalize_per_subject_cupy(
    s_lambda_NL_f64_scratch_dev: cp.ndarray,  # (N, L) fp32 — modified in-place
    tmp_idx_NL_dev: cp.ndarray,                # (N, L) bool
    boundary_mask_NL_dev: cp.ndarray,          # (N, L) fp32
    s_lambda_NL_f32_out_dev: cp.ndarray,       # (N, L) fp32 — OUT
) -> None:
    # In-place multiply (both fp32 now).
    cp.multiply(s_lambda_NL_f64_scratch_dev, boundary_mask_NL_dev,
                out=s_lambda_NL_f64_scratch_dev)
    rs = s_lambda_NL_f64_scratch_dev.sum(axis=1, keepdims=True)     # (N, 1) fp32
    zero_f32 = cp.float32(0.0)
    one_f32 = cp.float32(1.0)
    safe_rs = cp.where(rs > zero_f32, rs, one_f32)
    normalized = cp.where(
        rs > zero_f32,
        s_lambda_NL_f64_scratch_dev / safe_rs,
        zero_f32,
    )
    normalized = cp.where(
        tmp_idx_NL_dev[:, 0:1],
        zero_f32,
        normalized,
    )
    cp.copyto(s_lambda_NL_f32_out_dev, normalized)


# ─────────────────────────────────────────────────────────────────────
# Phase E.2 — theta = mean over S.
# ─────────────────────────────────────────────────────────────────────
def _phase_e2_theta_only_cupy(
    s_lambda_SNL_f32_dev: cp.ndarray,   # (S, N, L) fp32
    theta_NL_out_dev: cp.ndarray,       # (N, L)    fp32 — OUT
) -> None:
    cp.mean(s_lambda_SNL_f32_dev, axis=0, out=theta_NL_out_dev)


# ─────────────────────────────────────────────────────────────────────
# Streaming master kernel — CuPy port of em_iter_master_kernel_streaming.
#
# Mirrors the CPU master's Phase ordering exactly:
#   A.1 — sigma_psi precompute
#   A.3 — per-subject X_dot_sl (decode BOLD per subject)
#   B   — M-step inner while-loop
#   A.4 — copy B → A if odd m_iters
#   C+D+E.1 — per-subject (spatial_connect → fused E-step → normalize)
#   E.2 — theta = mean over S
# ─────────────────────────────────────────────────────────────────────
def em_iter_master_kernel_streaming_cupy(
    # Streaming source (device scratch) + BOLD loader callback:
    bold_scratch_NTD_dev: cp.ndarray,
    load_bold_into_dev,                # callable(s_1, out_dev)
    # Gradient eager cache (gMSHBM only — dummy 0-d buffer in dMSHBM mode).
    # Shape (S, N, D_grad) fp32; populated once at Session construction
    # from ``SubjectGradientLoader.load_into`` (which produces a (T, N,
    # D_grad) replicated block — we keep just one (N, D_grad) per sub
    # since the gradient is session-invariant). Per-iter, per-sub access
    # is ``grad_cache_SND_dev[s]`` — zero disk / zero H2D, just a device-
    # resident view into the cache.
    grad_cache_SND_dev: cp.ndarray,
    # Static buffers (mutated only at construction):
    boundary_mask_NL_dev: cp.ndarray,
    has_spatial: int,
    # Mutable state (in-place updates):
    s_lambda_SNL_f32_dev: cp.ndarray,  # (S, N, L) fp32
    s_t_nu_STLD_dev: cp.ndarray,     # (S, T, L, D) fp32 — in-place state
    theta_NL_dev: cp.ndarray,          # (N, L) fp32
    cost_S_out_dev: cp.ndarray,        # (S,) fp64 — OUT
    # Per-iter scratch (Session-owned):
    s_lambda_NL_f64_scratch_dev: cp.ndarray,  # (N, L) fp64
    X_dot_sl_STLD_dev: cp.ndarray,
    sigma_psi_SLD_dev: cp.ndarray,
    log_connect_NL_buf_dev: cp.ndarray,
    log_vmf_NL_buf_dev: cp.ndarray,
    tmp_idx_NL_dev: cp.ndarray,
    # Per-EM-iter precomputed (Session-owned):
    sigma_L_dev: cp.ndarray,
    s_psi_SLD_dev: cp.ndarray,
    # Scalars:
    dim_int: int,
    kappa_init_f64: float,
    ini_val_f64: float,
    beta_f64: float,
    eps_f32: float,
    max_iter_m: int,
    n_lh: int,
    L_lh: int,
) -> Tuple[int, float]:
    """One outer EM iter on device. Returns ``(m_iters, kappa_final_f64)``.

    Per-subject cost is written to ``cost_S_out_dev``; caller D2Hs.

    All buffers must be device-resident, fp32 (or as documented for the
    fp64 scratch); shapes match the CPU master's signature.
    """
    S = int(s_lambda_SNL_f32_dev.shape[0])
    T = int(s_t_nu_STLD_dev.shape[1])

    # Phase A.1 — sigma_psi precompute.
    _sigma_psi_SLD_compute_cupy(sigma_L_dev, s_psi_SLD_dev, sigma_psi_SLD_dev)

    # Phase A.3 — per-subject streamed X_dot_sl.
    # Per-subject H2D (decode BOLD into bold_scratch_NTD_dev). The loader
    # callback writes device memory directly (via cp.ndarray.set() or a
    # pinned-host staging copy — see the Session for the contract).
    for s in range(S):
        load_bold_into_dev(s + 1, bold_scratch_NTD_dev)
        _compute_X_dot_sl_s_NTD_cupy(
            bold_scratch_NTD_dev,
            s_lambda_SNL_f32_dev[s],
            X_dot_sl_STLD_dev[s],
        )

    # Phase B — M-step inner while-loop. denom is a global fp64 reduction.
    denom_f64 = float(T) * float(
        s_lambda_SNL_f32_dev.sum(dtype=cp.float64)
    )
    m_iters, kappa_new = _mstep_inner_loop_step2_cupy(
        X_dot_sl_STLD_dev,
        sigma_psi_SLD_dev,
        denom_f64,
        float(dim_int),
        kappa_init_f64,
        ini_val_f64,
        eps_f32,
        max_iter_m,
        s_t_nu_STLD_dev,
    )
    # The M-step renormalizes s_t_nu in place into s_t_nu_STLD_dev, so
    # downstream phases read it directly — no ping-pong, no parity copy.

    # Phase C+D+E.1 — per-subject streamed.
    for s in range(S):
        load_bold_into_dev(s + 1, bold_scratch_NTD_dev)
        if has_spatial == 1:
            # grad is already on device in ``grad_cache_SND_dev`` (built
            # once at Session __init__); no disk read, no H2D.
            _spatial_connect_per_subject_cupy(
                grad_cache_SND_dev[s],
                s_lambda_SNL_f32_dev[s],
                log_connect_NL_buf_dev,
                n_lh, L_lh, T,
            )
        cost_s = _fused_estep_per_subject_NTD_cupy(
            bold_scratch_NTD_dev,
            s_t_nu_STLD_dev[s],
            kappa_new,
            int(dim_int),
            theta_NL_dev,
            log_connect_NL_buf_dev,
            boundary_mask_NL_dev,
            beta_f64,
            has_spatial,
            log_vmf_NL_buf_dev,
            s_lambda_NL_f64_scratch_dev,
            tmp_idx_NL_dev,
        )
        cost_S_out_dev[s] = cp.float64(cost_s)

        _phase_e1_normalize_per_subject_cupy(
            s_lambda_NL_f64_scratch_dev,
            tmp_idx_NL_dev,
            boundary_mask_NL_dev,
            s_lambda_SNL_f32_dev[s],
        )

    # Phase E.2 — theta = mean over S.
    _phase_e2_theta_only_cupy(s_lambda_SNL_f32_dev, theta_NL_dev)

    return m_iters, kappa_new


# ─────────────────────────────────────────────────────────────────────
# Warmup — CuPy compiles its ufuncs / reductions per dtype combo on
# first invocation. Pre-warm at process start so the first real call
# doesn't pay JIT cost.
# ─────────────────────────────────────────────────────────────────────
def warmup_step2_gpu() -> None:
    """Pre-warm every CuPy primitive used by the master kernel.

    Tiny synthetic shapes so the JIT compiles for the dtypes we actually
    use (fp32 / fp64 / bool / int8). ~100 ms one-shot.
    """
    S, T, N, D, L, D_grad = 2, 2, 8, 4, 4, 3
    n_lh = N // 2
    L_lh = L // 2
    rng = np.random.default_rng(0)

    BOLD = cp.asarray(rng.standard_normal((S, N, T, D)).astype(np.float32))
    # Eager grad cache (S, N, D_grad) — session-invariant gradient.
    grad_cache = cp.asarray(
        rng.standard_normal((S, N, D_grad)).astype(np.float32)
    )
    bm = cp.ones((N, L), dtype=cp.float32)
    s_lambda_f32 = cp.full((S, N, L), 0.25, dtype=cp.float32)
    s_t_nu = cp.asarray(
        (rng.standard_normal((S, T, L, D)) * 0.1).astype(np.float32)
    )
    theta = cp.full((N, L), 0.25, dtype=cp.float32)
    cost = cp.zeros(S, dtype=cp.float64)
    X_dot_sl = cp.empty((S, T, L, D), dtype=cp.float32)
    sigma_psi = cp.empty((S, L, D), dtype=cp.float32)
    log_connect = cp.zeros((N, L), dtype=cp.float32)
    log_vmf = cp.zeros((N, L), dtype=cp.float32)
    tmp_idx = cp.zeros((N, L), dtype=cp.bool_)
    sigma = cp.full(L, 0.1, dtype=cp.float32)
    s_psi = cp.asarray((rng.standard_normal((S, L, D)) * 0.05).astype(np.float32))
    s_lambda_NL_f64_scratch = cp.empty((N, L), dtype=cp.float32)  # fp32 now (see header)
    bold_scratch = cp.empty((N, T, D), dtype=cp.float32)

    def _bold_into(s_1, out_dev):
        out_dev[...] = BOLD[s_1 - 1]

    # gMSHBM path — uses grad_cache.
    em_iter_master_kernel_streaming_cupy(
        bold_scratch, _bold_into, grad_cache,
        bm, 1,
        s_lambda_f32, s_t_nu, theta, cost,
        s_lambda_NL_f64_scratch,
        X_dot_sl, sigma_psi,
        log_connect, log_vmf, tmp_idx,
        sigma, s_psi,
        D, 50.0, 30.0, 5000.0, np.float32(1e-4), 3, n_lh, L_lh,
    )
    # dMSHBM path — has_spatial=0; grad_cache is not read but the param
    # is still passed (zero-size dummy in production session_gpu).
    em_iter_master_kernel_streaming_cupy(
        bold_scratch, _bold_into, grad_cache,
        bm, 0,
        s_lambda_f32, s_t_nu, theta, cost,
        s_lambda_NL_f64_scratch,
        X_dot_sl, sigma_psi,
        log_connect, log_vmf, tmp_idx,
        sigma, s_psi,
        D, 50.0, 30.0, 5000.0, np.float32(1e-4), 3, n_lh, L_lh,
    )
