"""_kernels_gpu.py — CUDA kernels of the step-2 ``gpu`` backend (P-layout, bit-packed).

One process-global :class:`cupy.RawModule` holding every device kernel of the
P-layout step-2 EM (``P = nnz(boundary_mask)``, CSR order). The numerical
reference is the **CPU numba backend**
(``step2_em_iter_master/_kernels.py`` + ``step2_em_outer/_kernels.py``);
the contract is ``docs/step2_sparse_design.md`` §1-§3.

Module rules (design §3), all load-bearing:

* compiled ``-std=c++17 -fmad=false``, no ``--use_fast_math`` — the fp32 op
  sequences that mirror numba's ``fastmath=False`` code must not contract
  into FMAs, and the empty-parcel contract needs IEEE ``0*inf``;
* CUDA source is **ASCII only** (this box is zh-CN) and NVRTC compiles
  without host headers, so ``INFINITY`` / ``NAN`` are undefined — the
  bit-pattern helpers ``f_inf`` / ``d_inf`` / ``d_nan`` are used instead;
* **no floating-point atomics**; every reduction is a fixed-shape tree over
  a fixed grid, so a run is bit-reproducible;
* kernels never allocate — the Session owns every buffer;
* ``long long`` indexing wherever a product can exceed 2^31.

Flush-to-zero (design §1.10). CuPy appends ``-ftz=true`` after every user
option (``cupy/cuda/compiler.py``), so every fp32 arithmetic op, compare and
conversion in this module flushes fp32 subnormals. The CPU keeps subnormal
``s_lambda`` / ``theta`` cells and ``theta == 0`` is absorbing, so the sites
listed in §1.10 go through ``f32_rn`` / ``f32_to_f64``, which emulate a
subnormal-aware fp32 round / widen in integer + fp64 arithmetic.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised only on CPU-only hosts
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────
# Launch geometry / compile-time limits
# ─────────────────────────────────────────────────────────────────────
XDOT_BLOCK = 256        # threads own BYTES of the packed row
XDOT_MAX_NB = 1         # bytes per thread -> Db <= 256 (D <= 2048)
XDOT_CHUNK = 64         # members per fp32->fp64 fold (fixes the K3 op order)
FUSED_BLOCK = 256       # M-step block reduce over D
REDUCE_BLOCK = 256
REDUCE_GRID = 1024      # FIXED grid -> fixed reduction order
ROW_BLOCK = 256         # thread-per-row kernels
INIT_WARPS = 8          # warps per init_hard_labels block
INIT_RR = 4             # rows carried per warp (register tile)
INIT_ROWS = INIT_WARPS * INIT_RR   # rows per block
INIT_BLOCK = 32 * INIT_WARPS       # threads per block
INIT_MAXJ = 16          # per-lane accumulators -> L <= 32 * INIT_MAXJ
MAX_CLUSTERS = 32 * INIT_MAXJ   # the L ceiling ``check_dims`` enforces
ACC_BLOCK = 768         # K6 members per block (see the acc_P comment)
CONNECT_BLOCK = 128
FLAG_BLOCK = 256        # intra_flags_rel: one block, power of two
# ``connect_u`` sizes its dynamic shared memory as ``D_grad`` floats on top of
# its static sh_w[256] + sh_n[256] + sh_sl (2064 B as the compiler reports it),
# against the 48 KiB default per-block limit. ``check_dims`` enforces the
# resulting D_grad ceiling; ``test_connect_u_shared_limit_is_exact`` pins both
# numbers against the compiled kernel and the device attribute.
CONNECT_STATIC_SHARED = 2064
MAX_D_GRAD = (49152 - CONNECT_STATIC_SHARED) // 4


# ─────────────────────────────────────────────────────────────────────
# CUDA source
# ─────────────────────────────────────────────────────────────────────
_SRC_HELPERS = r"""
// =====================================================================
// Bit-pattern literals (NVRTC has no <math.h> macros).
// =====================================================================
__device__ __forceinline__ float  f_inf()  { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float  f_ninf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ double d_inf()  { return __longlong_as_double(0x7ff0000000000000ULL); }
__device__ __forceinline__ double d_nan()  { return __longlong_as_double(0x7ff8000000000000ULL); }

// =====================================================================
// Subnormal-aware fp64 -> fp32 round-to-nearest-even.
// CuPy compiles with -ftz=true, so a plain (float)x returns +0 for every
// |x| < 2^-126; the CPU reference keeps those cells alive (design 1.10).
// =====================================================================
__device__ __forceinline__ float f32_rn(double x) {
    const double ax = fabs(x);
    if (!(ax < 0x1p-126)) {           // normal, inf or NaN -> hardware convert
        return (float)x;
    }
    // |x| < 2^-126: build the subnormal (or smallest-normal) bit pattern by
    // hand.  ax * 2^149 is exact and <= 2^23, rint() is round-half-even.
    const int m = (int)rint(ax * 0x1p149);
    const int sign = (__double2hiint(x) < 0) ? (int)0x80000000 : 0;
    return __int_as_float(sign | m);
}

// Subnormal-aware fp32 -> fp64 widen.
__device__ __forceinline__ double f32_to_f64(float f) {
    const int b = __float_as_int(f);
    if ((b & 0x7f800000) == 0) {      // zero or subnormal
        const double v = (double)(b & 0x007fffff) * 0x1p-149;
        return (b < 0) ? -v : v;
    }
    return (double)f;
}

// Emulated native-fp32 binary ops with subnormal support.
__device__ __forceinline__ float f32_add_rn(float a, float b) {
    return f32_rn(f32_to_f64(a) + f32_to_f64(b));
}
__device__ __forceinline__ float f32_mul_rn(float a, float b) {
    return f32_rn(f32_to_f64(a) * f32_to_f64(b));
}

// True for any nonzero fp32 including subnormals (FTZ-proof).
__device__ __forceinline__ bool f32_nonzero(float f) {
    return (__float_as_int(f) & 0x7fffffff) != 0;
}
// True for strictly positive fp32 including subnormals (theta >= 0 always).
__device__ __forceinline__ bool f32_pos(float f) {
    return __float_as_int(f) > 0;
}

// =====================================================================
// Fixed-tree fp64 block reduction (blockDim.x a power of two <= 1024).
// =====================================================================
__device__ __forceinline__ double block_reduce_f64(double v, double* sh) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    const double r = sh[0];
    __syncthreads();
    return r;
}

// =====================================================================
// Scalar special functions -- verbatim ports of
//   em_stop_criterion/_cdln.py::_log_bessel_i_debye / _cdln_single
//   m_step/_invad.py::invad_numba
// =====================================================================
#define LOG_2PI 1.8378770664093454835606594728112

__device__ double dev_log_bessel_i_debye(double nu, double x) {
    if (x <= 0.0) return (nu == 0.0) ? 0.0 : -d_inf();
    if (nu < 1e-10) nu = 1e-10;

    const double z = x / nu;
    const double z2 = z * z;
    const double sqrt_1pz2 = sqrt(1.0 + z2);
    const double eta = sqrt_1pz2 + log(z / (1.0 + sqrt_1pz2));

    const double p = 1.0 / sqrt_1pz2;
    const double p2 = p*p, p3 = p*p*p, p4 = p*p*p*p;
    const double p5 = p4*p, p6 = p4*p2, p7 = p4*p3;
    const double p9 = p4*p5, p10 = p5*p5, p11 = p5*p6;
    const double p12 = p6*p6, p13 = p6*p7, p15 = p7*p7*p;

    const double u1 = (3.0*p - 5.0*p3) / 24.0;
    const double u2 = (81.0*p2 - 462.0*p4 + 385.0*p6) / 1152.0;
    const double u3 = (30375.0*p3 - 369603.0*p5 + 765765.0*p7 - 425425.0*p9) / 414720.0;
    const double u4 = (4465125.0*p4 - 94121676.0*p6 + 349922430.0*p4*p4
                       - 446185740.0*p10 + 185910725.0*p12) / 39813120.0;
    const double u5 = (1519035525.0*p5 - 49286948607.0*p7 + 284499769554.0*p9
                       - 614135872350.0*p11 + 566098157625.0*p13
                       - 188699385875.0*p15) / 6688604160.0;

    const double inv_nu = 1.0 / nu;
    const double correction = (1.0 + u1*inv_nu + u2*inv_nu*inv_nu
                               + u3*inv_nu*inv_nu*inv_nu
                               + u4*inv_nu*inv_nu*inv_nu*inv_nu
                               + u5*inv_nu*inv_nu*inv_nu*inv_nu*inv_nu);

    double log_iv = nu*eta - 0.5*(LOG_2PI + log(nu)) - 0.25*log(1.0 + z2);
    if (correction > 0.0) log_iv += log(correction);
    return log_iv;
}

// Debye for v >= 25 (the only branch reachable at step-2's v = dim/2 - 1);
// a smaller order must fail loudly rather than silently take a stub.
__device__ __forceinline__ double dev_log_bessel_i(double v, double x) {
    if (v >= 25.0) return dev_log_bessel_i_debye(v, x);
    return d_nan();
}

__device__ __forceinline__ double dev_cdln_single(double k, double v) {
    return v * log(k) - dev_log_bessel_i(v, k);
}

__device__ double dev_invad(double D, double rbar) {
    if (rbar <= 0.0) return 0.0;
    if (rbar >= 1.0) return d_inf();

    const double v = D / 2.0 - 1.0;
    const double kappa0 = (D - 1.0) * rbar / (1.0 - rbar * rbar)
                          + (D / (D - 1.0)) * rbar;
    const double asymp = kappa0 - (D / (D - 1.0)) * rbar * 0.5;

    const double log_iv = dev_log_bessel_i(v, kappa0);
    if ((log_iv > 709.0) || (log_iv < -708.0) || !(log_iv == log_iv)) return asymp;

    double x0 = kappa0;
    double x1 = kappa0 * 1.001;
    double lA0 = dev_log_bessel_i(v + 1.0, x0) - dev_log_bessel_i(v, x0);
    double f0 = exp(lA0) - rbar;
    double lA1 = dev_log_bessel_i(v + 1.0, x1) - dev_log_bessel_i(v, x1);
    double f1 = exp(lA1) - rbar;

    for (int it = 0; it < 50; ++it) {
        const double df = f1 - f0;
        if (fabs(df) < 1e-300) return x1;
        const double x_new = x1 - f1 * (x1 - x0) / df;
        if (x_new <= 0.0 || !(x_new == x_new)) return asymp;
        if (fabs(x_new - x1) < 1e-12) return x_new;
        x0 = x1;
        x1 = x_new;
        f0 = f1;
        lA1 = dev_log_bessel_i(v + 1.0, x1) - dev_log_bessel_i(v, x1);
        f1 = exp(lA1) - rbar;
    }
    return asymp;
}

// Debug hooks for the unit tests (never used by the Session).
extern "C" __global__
void probe_invad(const double* rbar, double* out, double D, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = dev_invad(D, rbar[i]);
}
extern "C" __global__
void probe_cdln(const double* k, double* out, double v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = dev_cdln_single(k[i], v);
}
extern "C" __global__
void probe_f32_rn(const double* x, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_rn(x[i]);
}
extern "C" __global__
void probe_f32_to_f64(const float* x, double* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_to_f64(x[i]);
}
"""


_SRC_ROWSTATS = r"""
// =====================================================================
// K0 -- per-(s,t,n) BOLD row statistics, EXACT vs the CPU widen kernel
// (step2_io/load_subject_profiles.py). One thread per row: the sumsq
// accumulation must stay a serial ascending-d fp32 sum.
// =====================================================================
extern "C" __global__
void row_stats_exact(const unsigned char* __restrict__ packed,  // (n_rows, Db)
                     float* __restrict__ row_mean,              // (n_rows,)
                     float* __restrict__ row_inv,               // (n_rows,)
                     long long n_rows, int D, int Db, float inv_D)
{
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const unsigned char* src = packed + row * (long long)Db;

    int k = 0;
    for (int b = 0; b < Db; ++b) k += __popc((unsigned int)src[b]);

    const float mean = (float)k * inv_D;
    const bool dead = (mean == 0.0f) || (mean == 1.0f);

    float acc = 0.0f;
    for (int b = 0; b < Db; ++b) {
        const unsigned int bv = (unsigned int)src[b];
        const int base = b * 8;
        for (int j = 0; j < 8; ++j) {
            const int d = base + j;
            if (d < D) {
                const float v = (float)((bv >> j) & 1u) - mean;
                acc += v * v;
            }
        }
    }
    row_mean[row] = mean;
    row_inv[row] = (dead || !(acc > 0.0f))
                   ? 0.0f
                   : (1.0f / (float)sqrt((double)acc));
}

// n_alive[s, n] = #{t : row_inv[s, t, n] != 0}
extern "C" __global__
void count_alive(const float* __restrict__ row_inv,   // (S, T, N)
                 int* __restrict__ n_alive,           // (S, N)
                 int S, int T, int N)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)S * N) return;
    const long long s = i / N;
    const long long n = i - s * N;
    int c = 0;
    for (int t = 0; t < T; ++t) {
        if (row_inv[(s * T + t) * N + n] != 0.0f) c += 1;
    }
    n_alive[i] = c;
}

// Widen one subject's packed rows to a dense (n_tile, T*D) fp32 slab --
// exactly (f32(bit) - mean) * inv, the K7 iteration-1 operand.
//
// PERF (F3): one WARP per (i, t) row with the lanes striding ``d``, not one
// thread per row.  With a thread per row the 32 lanes of a warp wrote 32
// different rows, i.e. every store instruction scattered over 32 x 4.7 KB;
// with a warp per row each store is 32 consecutive floats.  The 8 lanes that
// share a packed byte hit L1.  Bit-exact (every output is independent).
// Measured 1505 us -> 356 us per 8192-row tile on the S=1 bench.
extern "C" __global__
void widen_exact(const unsigned char* __restrict__ packed,  // (T, N, Db)
                 const float* __restrict__ row_mean,        // (T, N)
                 const float* __restrict__ row_inv,         // (T, N)
                 float* __restrict__ out,                   // (n_tile, T, D)
                 int T, int N, int D, int Db, int row_off, int n_tile)
{
    const long long w =
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (w >= (long long)n_tile * T) return;
    const int lane = threadIdx.x & 31;
    const int i = (int)(w / T);
    const int t = (int)(w - (long long)i * T);
    const int n = row_off + i;
    const float mean = row_mean[(long long)t * N + n];
    const float inv = row_inv[(long long)t * N + n];
    const unsigned char* src = packed + ((long long)t * N + n) * Db;
    float* dst = out + ((long long)i * T + t) * D;
    for (int d = lane; d < D; d += 32) {
        const unsigned int bv = (unsigned int)__ldg(src + (d >> 3));
        dst[d] = ((float)((bv >> (d & 7)) & 1u) - mean) * inv;
    }
}
"""


_SRC_INIT = r"""
// =====================================================================
// K1a -- per-(s,n) hard label + medial flag (design 1.2).
//
// INIT_WARPS warps per block, each warp owning INIT_RR consecutive rows, so
// a block covers INIT_ROWS = INIT_WARPS*INIT_RR rows.  Each (n, l)
// accumulator stays a private serial ascending-d fp32 sum, which is what
// makes every (INIT_WARPS, INIT_RR, DT) choice bit-identical.
//
// PERF (F3), two changes over the original one-warp-per-row form, both
// bit-exact (measured on the S=1 bench: 22.7 ms -> 7.9 ms):
//  1. the ``at`` stage is BYTE-based -- one packed byte yields 8 d values,
//     so the packed / row_mean / row_inv loads drop 8x (they used to be
//     re-issued once per d, i.e. N*D*T = 577 M byte loads for 72 MB of data);
//  2. a warp carries INIT_RR rows in registers, so each ``g_DL`` load feeds
//     INIT_RR fmaf instead of one.  With one row per warp the loop is
//     1 load : 1 FMA and runs at ~1.3 TFLOP/s; INIT_RR = 4 makes it 1 : 4.
//     INIT_RR = 8 spills ``acc[RR][INIT_MAXJ]`` and costs 30 ms.
// ``g_DL`` is read straight from global: it is 1.4 MB, L2-resident, and every
// block sweeps it in the same order, so a shared-memory stage measured no
// better (19.6 vs 19.9 ms at INIT_RR = 1) while capping DT.
// =====================================================================
#define INIT_WARPS INIT_WARPS_VALUE
#define INIT_RR    INIT_RR_VALUE
#define INIT_ROWS  (INIT_WARPS * INIT_RR)
#define INIT_MAXJ INIT_MAXJ_VALUE

extern "C" __global__
void init_hard_labels(const unsigned char* __restrict__ packed,  // (S,T,N,Db)
                      const float* __restrict__ row_mean,        // (S,T,N)
                      const float* __restrict__ row_inv,         // (S,T,N)
                      const float* __restrict__ g_DL,            // (D,L)
                      int* __restrict__ hard_label,              // (S,N)
                      unsigned char* __restrict__ medial,        // (S,N)
                      int S, int T, int N, int D, int Db, int L,
                      int DT, float inv_T)
{
    extern __shared__ float at[];             // INIT_ROWS * DT

    const int s = blockIdx.y;
    const int base_n = blockIdx.x * INIT_ROWS;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    float acc[INIT_RR][INIT_MAXJ];
    #pragma unroll
    for (int r = 0; r < INIT_RR; ++r) {
        #pragma unroll
        for (int j = 0; j < INIT_MAXJ; ++j) acc[r][j] = 0.0f;
    }
    float nsq[INIT_RR];
    #pragma unroll
    for (int r = 0; r < INIT_RR; ++r) nsq[r] = 0.0f;

    // DT is a multiple of 8 (host ``init_dt``), so d0 is byte-aligned.
    for (int d0 = 0; d0 < D; d0 += DT) {
        const int dt = (D - d0 < DT) ? (D - d0) : DT;
        const int nby = (dt + 7) >> 3;
        __syncthreads();
        for (int i = tid; i < INIT_ROWS * nby; i += bs) {
            const int r = i / nby;
            const int q = i - r * nby;
            const int n = base_n + r;
            float a8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) a8[k] = 0.0f;
            if (n < N) {
                const int b = (d0 >> 3) + q;
                for (int t = 0; t < T; ++t) {
                    const long long ridx = ((long long)s * T + t) * N + n;
                    const float mean = row_mean[ridx];
                    const float inv = row_inv[ridx];
                    const unsigned int bv =
                        (unsigned int)packed[ridx * Db + b];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k)
                        a8[k] += ((float)((bv >> k) & 1u) - mean) * inv;
                }
            }
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int dd = q * 8 + k;
                if (dd < dt) at[r * DT + dd] = a8[k] * inv_T;
            }
        }
        __syncthreads();
        for (int dd = 0; dd < dt; ++dd) {
            float a[INIT_RR];
            #pragma unroll
            for (int r = 0; r < INIT_RR; ++r)
                a[r] = at[(wid * INIT_RR + r) * DT + dd];
            if (lane == 0) {
                #pragma unroll
                for (int r = 0; r < INIT_RR; ++r) nsq[r] += a[r] * a[r];
            }
            const float* grow = g_DL + (long long)(d0 + dd) * L;
            #pragma unroll
            for (int j = 0; j < INIT_MAXJ; ++j) {
                const int l = lane + 32 * j;
                if (l < L) {
                    const float gv = __ldg(grow + l);
                    #pragma unroll
                    for (int r = 0; r < INIT_RR; ++r) acc[r][j] += a[r] * gv;
                }
            }
        }
    }

    // First index of the strict maximum (matches the CPU's '>' update).
    #pragma unroll
    for (int r = 0; r < INIT_RR; ++r) {
        const int n_my = base_n + wid * INIT_RR + r;
        if (n_my >= N) continue;
        float best = 0.0f;
        int bl = -1;
        #pragma unroll
        for (int j = 0; j < INIT_MAXJ; ++j) {
            const int l = lane + 32 * j;
            if (l < L) {
                if (bl < 0) { best = acc[r][j]; bl = l; }
                else if (acc[r][j] > best) { best = acc[r][j]; bl = l; }
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, best, off);
            const int oi = __shfl_down_sync(0xffffffffu, bl, off);
            if (oi >= 0 && (bl < 0 || ov > best || (ov == best && oi < bl))) {
                best = ov; bl = oi;
            }
        }
        if (lane == 0) {
            hard_label[(long long)s * N + n_my] = bl;
            medial[(long long)s * N + n_my] = (nsq[r] == 0.0f) ? 1 : 0;
        }
    }
}

// =====================================================================
// K1b(i) -- sparse one-hot write of s_lambda on P.
extern "C" __global__
void init_slambda(const int* __restrict__ hard_label,       // (S,N)
                  const unsigned char* __restrict__ medial, // (S,N)
                  const int* __restrict__ row_ptr,          // (N+1,)
                  const int* __restrict__ col,              // (P,)
                  float* __restrict__ s_lambda,             // (S,P) zero-filled
                  int S, int N, long long P)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)S * N) return;
    const long long s = i / N;
    const int n = (int)(i - s * N);
    if (medial[i]) return;
    const int l_act = hard_label[i];
    int lo = row_ptr[n], hi = row_ptr[n + 1] - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const int c = col[mid];
        if (c == l_act) { s_lambda[s * P + mid] = 1.0f; return; }
        if (c < l_act) lo = mid + 1; else hi = mid - 1;
    }
}

// K1b(ii) -- theta / log_theta / active flag from the composed s_lambda.
extern "C" __global__
void init_theta(const float* __restrict__ s_lambda,   // (S,P)
                float* __restrict__ theta,            // (P,)
                float* __restrict__ log_theta,        // (P,)
                int* __restrict__ active,             // (P,)
                int S, long long P,
                float theta_on, float theta_off)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    bool any = false;
    for (int s = 0; s < S; ++s) {
        if (f32_nonzero(s_lambda[(long long)s * P + p])) { any = true; break; }
    }
    const float th = any ? theta_on : theta_off;
    theta[p] = th;
    const bool on = f32_pos(th);
    active[p] = on ? 1 : 0;
    log_theta[p] = on ? (float)log(f32_to_f64(th)) : f_ninf();
}
"""


_SRC_MSTEP = r"""
// =====================================================================
// K2 -- sigma_psi[s,l,d] = sigma[l] * s_psi[s,l,d]
// =====================================================================
extern "C" __global__
void sigma_psi_SLD(const float* __restrict__ sigma_L,
                   const float* __restrict__ s_psi,
                   float* __restrict__ out,
                   int L, int D, long long n_total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    out[i] = sigma_L[(i / D) % L] * s_psi[i];
}

// =====================================================================
// K3 -- X_dot_sl[s,t,l,d] = sum_{n in members(l)} s_lambda[s,n,l]*X[s,n,t,d]
// Port of step-3's ``x_dot_sl_bits`` with the subject axis added and the
// member lists restricted to the ACTIVE CSC (design 2.3).
// One block per (t, l); threads own BYTES of the packed row.
//
// PERF (F3): the step-3 original stages each member's packed row in shared
// memory.  Here it does not pay: every thread of the block walks the SAME
// member row at the same time, so ``__ldg(rp + tid)`` is already a fully
// coalesced 147-byte read and the staging loop only adds a serial
// ``for i < cnt`` global->smem copy plus a __syncthreads per chunk.  Dropping
// it (and moving from 128 threads x 2 bytes to 256 threads x 1 byte, which
// halves the per-thread accumulator arrays) is BIT-EXACT -- the per-``d``
// fp32 accumulator still folds the same members in the same order, and the
// fp32->fp64 fold still happens every ``chunk`` members.  Measured on the
// real S=1 active support (73 693 members): 696 us -> 490 us.
// ``XDOT_BLOCK`` must stay a power of two: ``block_reduce_f64``'s fixed tree
// (used for ``B``) is only correct for one.
// =====================================================================
#define XDOT_MAX_NB XDOT_MAX_NB_VALUE
#define XDOT_BS     XDOT_BS_VALUE

extern "C" __global__
void x_dot_sl_bits(const unsigned char* __restrict__ packed,  // (T, N, Db)
                   const float* __restrict__ row_mean,        // (T, N)
                   const float* __restrict__ row_inv,         // (T, N)
                   const int* __restrict__ col_ptr,           // (L+1,)
                   const int* __restrict__ csc_row,           // (P,)
                   const int* __restrict__ csc_pidx,          // (P,)
                   const float* __restrict__ s_lambda_P,      // (P,)
                   float* __restrict__ out,                   // (T, L, D)
                   int N, int L, int D, int D_bytes, int chunk)
{
    extern __shared__ float sh_w[];          // (chunk,) w, then (chunk,) row id
    int* sh_n = (int*)(sh_w + chunk);
    __shared__ double sh_red[XDOT_BS];
    __shared__ double sh_B;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int blk = blockIdx.x;
    const int t = blk / L;
    const int l = blk - t * L;

    const int start = col_ptr[l];
    const int end = col_ptr[l + 1];

    const unsigned char* base_t =
        packed + (size_t)t * (size_t)N * (size_t)D_bytes;
    const float* rm_t = row_mean + (size_t)t * (size_t)N;
    const float* ri_t = row_inv + (size_t)t * (size_t)N;
    float* out_tl = out + ((size_t)t * (size_t)L + (size_t)l) * (size_t)D;

    double bl = 0.0;
    for (int i = start + tid; i < end; i += bs) {
        const int n = csc_row[i];
        const float w = s_lambda_P[csc_pidx[i]] * ri_t[n];
        bl += (double)(w * rm_t[n]);
    }
    bl = block_reduce_f64(bl, sh_red);
    if (tid == 0) sh_B = bl;
    __syncthreads();
    const double B = sh_B;

    float a32[XDOT_MAX_NB * 8];
    double acc[XDOT_MAX_NB * 8];
    #pragma unroll
    for (int k = 0; k < XDOT_MAX_NB * 8; ++k) acc[k] = 0.0;

    for (int cs = start; cs < end; cs += chunk) {
        const int cnt = (end - cs) < chunk ? (end - cs) : chunk;
        __syncthreads();
        for (int i = tid; i < cnt; i += bs) {
            const int n = csc_row[cs + i];
            sh_w[i] = s_lambda_P[csc_pidx[cs + i]] * ri_t[n];
            sh_n[i] = n;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < XDOT_MAX_NB * 8; ++k) a32[k] = 0.0f;

        // 4-deep member prefetch: the four ``__ldg`` are issued before any of
        // the dependent ``fmaf``, which is what hides the 1-byte load latency
        // (measured 403 -> 284 us per launch on the real S=1 active support).
        // The per-``d`` accumulator still folds members in ascending order,
        // so this is bit-exact.
        #pragma unroll
        for (int j = 0; j < XDOT_MAX_NB; ++j) {
            const int b = tid + j * XDOT_BS;
            if (b < D_bytes) {
                int i = 0;
                for (; i + 4 <= cnt; i += 4) {
                    const unsigned int v0 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i] * (size_t)D_bytes + b);
                    const unsigned int v1 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i + 1] * (size_t)D_bytes + b);
                    const unsigned int v2 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i + 2] * (size_t)D_bytes + b);
                    const unsigned int v3 = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i + 3] * (size_t)D_bytes + b);
                    const float w0 = sh_w[i], w1 = sh_w[i + 1];
                    const float w2 = sh_w[i + 2], w3 = sh_w[i + 3];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float a = a32[j * 8 + k];
                        a = fmaf(w0, (float)((v0 >> k) & 1u), a);
                        a = fmaf(w1, (float)((v1 >> k) & 1u), a);
                        a = fmaf(w2, (float)((v2 >> k) & 1u), a);
                        a = fmaf(w3, (float)((v3 >> k) & 1u), a);
                        a32[j * 8 + k] = a;
                    }
                }
                for (; i < cnt; ++i) {
                    const float w = sh_w[i];
                    const unsigned int bv = (unsigned int)__ldg(
                        base_t + (size_t)sh_n[i] * (size_t)D_bytes + b);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const float bit = (float)((bv >> k) & 1u);
                        a32[j * 8 + k] = fmaf(w, bit, a32[j * 8 + k]);
                    }
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < XDOT_MAX_NB * 8; ++k) acc[k] += (double)a32[k];
    }

    #pragma unroll
    for (int j = 0; j < XDOT_MAX_NB; ++j) {
        const int b = tid + j * XDOT_BS;
        if (b < D_bytes) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = b * 8 + k;
                if (d < D) out_tl[d] = (float)(acc[j * 8 + k] - B);
            }
        }
    }
}

// =====================================================================
// K4a -- fixed-grid fp64 stage-1 reductions.
// =====================================================================
extern "C" __global__
void reduce_sum_f32_stage1(const float* __restrict__ x,
                           double* __restrict__ part, long long n)
{
    __shared__ double sh[256];
    double acc = 0.0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        acc += (double)x[i];
    }
    acc = block_reduce_f64(acc, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = acc;
}

extern "C" __global__
void kappa_sum_stage1(const float* __restrict__ a, const float* __restrict__ b,
                      double* __restrict__ part, long long n)
{
    __shared__ double sh[256];
    double acc = 0.0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const float p = a[i] * b[i];
        acc += (double)p;
    }
    acc = block_reduce_f64(acc, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = acc;
}

extern "C" __global__
void reduce_stage2(const double* __restrict__ part, double* __restrict__ out,
                   int slot, int n_part)
{
    __shared__ double sh[256];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_part; i += blockDim.x) acc += part[i];
    acc = block_reduce_f64(acc, sh);
    if (threadIdx.x == 0) out[slot] = acc;
}

// =====================================================================
// K4b -- M-step scalar epilogue: stage-2 of kappa_sum, rbar, invad,
// clamps, drift, kappa_f32.  ONE block; the scalar tail is thread 0.
//   mstate[0] = kappa_prev, [1] = kappa_cur, [2] = kappa_sum, [3] = denom
//   result[0] = drift, [1] = kappa
// =====================================================================
extern "C" __global__
void mstep_kappa(const double* __restrict__ part, int n_part,
                 double* __restrict__ mstate,
                 float* __restrict__ kappa_f32_dev,
                 double* __restrict__ result,
                 double dim, double ini_val, double denom_scale)
{
    __shared__ double sh[256];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_part; i += blockDim.x) acc += part[i];
    acc = block_reduce_f64(acc, sh);
    if (threadIdx.x != 0) return;

    const double kappa_sum = acc;
    const double denom = mstate[3] * denom_scale;
    const double kappa_prev = mstate[0];
    double kappa = dev_invad(dim, kappa_sum / denom);
    if (!(isfinite(kappa))) kappa = kappa_prev;
    if (kappa < ini_val) kappa = ini_val;

    const double drift = fabs(kappa_prev - kappa) / kappa_prev;
    mstate[0] = kappa;
    mstate[1] = kappa;
    mstate[2] = kappa_sum;
    kappa_f32_dev[0] = (float)kappa;
    result[0] = drift;
    result[1] = kappa;
}

// =====================================================================
// K4c -- fused per-(s,t,l) M-step body, IN PLACE on s_t_nu.
// The old value is read into a register BEFORE the store (design 3 K4c).
// ``st_old`` and ``s_t_nu`` are the SAME pointer in production (hence no
// __restrict__ on either); the tests pass distinct buffers to get the
// ping-pong reference the CPU master uses.
// =====================================================================
extern "C" __global__
void mstep_fused_body(const float* __restrict__ kappa_f32_dev,
                      const float* __restrict__ X_dot_sl,   // (S,T,L,D)
                      const float* __restrict__ sigma_psi,  // (S,L,D)
                      const float* st_old,                  // (S,T,L,D)
                      float* s_t_nu,                        // (S,T,L,D)
                      float* __restrict__ cos_STL,          // (S,T,L)
                      int T, int L, int D)
{
    __shared__ double sh[256];
    __shared__ float sh_inv;
    extern __shared__ float sh_col[];

    const int blk = blockIdx.x;
    const int s = blk / (T * L);
    const int rem = blk - s * (T * L);
    const int t = rem / L;
    const int l = rem - t * L;
    const long long off = (long long)blk * D;
    const long long off_sp = ((long long)s * L + l) * D;
    const float kappa_f32 = kappa_f32_dev[0];

    double acc_n = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = kappa_f32 * X_dot_sl[off + d] + sigma_psi[off_sp + d];
        sh_col[d] = v;
        acc_n += (double)(v * v);
    }
    acc_n = block_reduce_f64(acc_n, sh);
    if (threadIdx.x == 0) {
        // cn == 0 -> inv_cn = inf -> 0*inf = NaN (empty parcel), as the CPU.
        sh_inv = (float)(1.0 / sqrt(acc_n));
    }
    __syncthreads();
    const float inv_cn = sh_inv;

    double acc_c = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float ov = st_old[off + d];        // READ BEFORE THE WRITE
        const float nv = sh_col[d] * inv_cn;
        s_t_nu[off + d] = nv;
        acc_c += (double)(nv * ov);
    }
    acc_c = block_reduce_f64(acc_c, sh);
    if (threadIdx.x == 0) cos_STL[blk] = (float)acc_c;
}

// =====================================================================
// K4d -- per-(s,t) convergence flag; OVERWRITTEN every iter_m (not latched).
// =====================================================================
extern "C" __global__
void mstep_flags(const float* __restrict__ cos_STL,
                 double* __restrict__ result, float eps, int L, int slot0)
{
    __shared__ int sh[256];
    const int st = blockIdx.x;
    const float* row = cos_STL + (long long)st * L;
    int c = 0;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        if ((1.0f - row[l]) < eps) c += 1;   // NaN fails '<' -> not converged
    }
    sh[threadIdx.x] = c;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) result[slot0 + st] = (sh[0] >= L) ? 1.0 : 0.0;
}

// =====================================================================
// K4e -- Cdln of the final kappa, once per run_iter.
// =====================================================================
extern "C" __global__
void cdln_after_loop(const double* __restrict__ mstate,
                     float* __restrict__ cdln_dev,
                     float* __restrict__ kappa_f32_dev,
                     double cdln_v)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const double kappa = mstate[1];
    cdln_dev[0] = (float)dev_cdln_single(kappa, cdln_v);
    kappa_f32_dev[0] = (float)kappa;
}

// s_t_nu row sums S[t,l] = sum_d nu[s,t,l,d] (fp64), block per (s,t,l).
extern "C" __global__
void stnu_row_sum(const float* __restrict__ s_t_nu, double* __restrict__ S_out,
                  int D)
{
    __shared__ double sh[256];
    const long long off = (long long)blockIdx.x * D;
    double acc = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) acc += (double)s_t_nu[off + d];
    acc = block_reduce_f64(acc, sh);
    if (threadIdx.x == 0) S_out[blockIdx.x] = acc;
}
"""


_SRC_CONNECT = r"""
// =====================================================================
// K5a -- spatial_connect parcel state (design 1.5).  Block per parcel.
//   sum_lambda : serial ascending-n fp32 sum over the ACTIVE members
//   u_update   : fp32 product, fp64 accumulate, fp32 store
//   inv_l      : 1/0 -> +inf ; 0 * inf -> NaN, which MUST propagate
//   u_sq       : serial ascending-d fp32 sum
//
// PERF (F3): the member list is staged in shared memory in CONNECT_CH chunks
// and BOTH consumers read it from there.  Before, thread 0 walked the column
// with ~250 dependent global loads (~500 cycles each) while the other 127
// threads sat at the barrier, and every thread then re-read the same
// ``s_lambda_P`` gather inside its own d-loop.  The staging also lets the
// fp64 ``u_update`` accumulation prefetch four ``grad`` loads per step.
// All three sums keep their exact order (sum_lambda serial ascending-n fp32
// on thread 0, u_update fp64 in member order, u_sq serial ascending-d), so
// this is bit-exact; measured 148 us -> 60 us per launch.
// =====================================================================
#define CONNECT_CH 256
extern "C" __global__
void connect_u(const int* __restrict__ act_col_ptr,
               const int* __restrict__ act_csc_row,
               const int* __restrict__ act_csc_pidx,
               const float* __restrict__ s_lambda_P,
               const float* __restrict__ grad,      // (N, Dg)
               float* __restrict__ sum_lambda,      // (L,)
               float* __restrict__ u_LD,            // (L, Dg)
               float* __restrict__ u_sq,            // (L,)
               int Dg)
{
    extern __shared__ float sh_u[];
    __shared__ float sh_sl;
    __shared__ float sh_w[CONNECT_CH];
    __shared__ int sh_n[CONNECT_CH];
    const int l = blockIdx.x;
    const int start = act_col_ptr[l];
    const int end = act_col_ptr[l + 1];
    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    float sl_acc = 0.0f;
    const int ndl = (Dg + bs - 1) / bs;
    for (int q = 0; q < ndl; ++q) {
        const int d = tid + q * bs;
        const bool have = (d < Dg);
        double acc = 0.0;
        for (int cs = start; cs < end; cs += CONNECT_CH) {
            const int cnt = (end - cs) < CONNECT_CH ? (end - cs) : CONNECT_CH;
            __syncthreads();
            for (int i = tid; i < cnt; i += bs) {
                sh_n[i] = act_csc_row[cs + i];
                sh_w[i] = s_lambda_P[act_csc_pidx[cs + i]];
            }
            __syncthreads();
            if (q == 0 && tid == 0) {
                for (int i = 0; i < cnt; ++i) sl_acc += sh_w[i];
            }
            if (have) {
                int i = 0;
                for (; i + 4 <= cnt; i += 4) {
                    const float g0 = grad[(long long)sh_n[i] * Dg + d];
                    const float g1 = grad[(long long)sh_n[i + 1] * Dg + d];
                    const float g2 = grad[(long long)sh_n[i + 2] * Dg + d];
                    const float g3 = grad[(long long)sh_n[i + 3] * Dg + d];
                    acc += (double)(g0 * sh_w[i]);
                    acc += (double)(g1 * sh_w[i + 1]);
                    acc += (double)(g2 * sh_w[i + 2]);
                    acc += (double)(g3 * sh_w[i + 3]);
                }
                for (; i < cnt; ++i) {
                    acc += (double)(grad[(long long)sh_n[i] * Dg + d]
                                    * sh_w[i]);
                }
            }
        }
        if (q == 0) {
            __syncthreads();
            if (tid == 0) {
                sum_lambda[l] = sl_acc;
                sh_sl = sl_acc;
            }
            __syncthreads();
        }
        const float inv_l = 1.0f / sh_sl;
        if (have) {
            const float uv = (float)acc * inv_l;
            u_LD[(long long)l * Dg + d] = uv;
            sh_u[d] = uv;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int d = 0; d < Dg; ++d) acc += sh_u[d] * sh_u[d];
        u_sq[l] = acc;
    }
}

// grad_sq[n] = serial ascending-d fp32 sum of grad[n,d]^2 (once per subject).
extern "C" __global__
void grad_sq_rows(const float* __restrict__ grad, float* __restrict__ grad_sq,
                  int N, int Dg)
{
    const long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = grad + n * (long long)Dg;
    float acc = 0.0f;
    for (int d = 0; d < Dg; ++d) acc += row[d] * row[d];
    grad_sq[n] = acc;
}

// =====================================================================
// K5b -- per-active-cell log_connect. NaN is kept (no guard).
// =====================================================================
extern "C" __global__
void connect_scv_P(const int* __restrict__ act_p,
                   const int* __restrict__ p_row,
                   const int* __restrict__ col,
                   const float* __restrict__ grad,     // (N, Dg)
                   const float* __restrict__ grad_sq,  // (N,)
                   const float* __restrict__ u_LD,     // (L, Dg)
                   const float* __restrict__ u_sq,     // (L,)
                   float* __restrict__ log_connect,    // (P,)
                   int n_act, int Dg, int T)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_act) return;
    const int p = act_p[i];
    const int n = p_row[p];
    const int l = col[p];
    const float* g = grad + (long long)n * Dg;
    const float* uu = u_LD + (long long)l * Dg;
    float cross = 0.0f;
    for (int d = 0; d < Dg; ++d) cross += g[d] * uu[d];
    const float vmf = (2.0f * cross - grad_sq[n]) - u_sq[l];
    float lc = 0.0f;
    for (int t = 0; t < T; ++t) lc += vmf;   // T sequential adds, NOT T*vmf
    log_connect[p] = lc;
}
"""


_SRC_ESTEP = r"""
// =====================================================================
// K6 -- lv_sum on the active support (design 1.6 / 3 K6).
// Block per (s, l); the nu row is staged as Db*8 floats with a zero tail
// so the branch-free bit fold can never read past it.
//
// PERF (F3): ``ACC_BLOCK`` is 768, not 256.  The kernel is one block per
// parcel and one thread per member, so the member loop's trip count is
// ceil(col_size / blockDim); on the real S=1 active support the columns run
// 75..533 members (mean 246), which at 256 threads makes 116 of the 300
// blocks take two or three passes while the rest take one -- a pure tail.
// Measured per launch on that support: 688 us at 256, 436 at 512, 305 at
// 640, 291 at 768, 421 at 1024 (32 warps/block leaves one block per SM).
// Every block size is BIT-EXACT: each member's fp32 fold and the fp64
// ascending-t combine are per-thread and independent of the partition, and
// this kernel has no block-wide reduction.  A member-tiled (L, NT) grid and
// a t-unrolled ILP variant were both measured slower (491 us / 650 us).
// =====================================================================
#define ACC_PF 8
extern "C" __global__
void acc_P(const unsigned char* __restrict__ packed,  // (T, N, Db)
           const float* __restrict__ row_mean,        // (T, N)
           const float* __restrict__ row_inv,         // (T, N)
           const float* __restrict__ stnu,            // (T, L, D)
           const double* __restrict__ S_TL,           // (T, L)
           const int* __restrict__ act_col_ptr,
           const int* __restrict__ act_csc_row,
           const int* __restrict__ act_csc_pidx,
           float* __restrict__ lv_sum,                // (P,)
           int T, int N, int D, int Db, int L)
{
    extern __shared__ float nu_sm[];
    const int l = blockIdx.x;
    const int start = act_col_ptr[l];
    const int end = act_col_ptr[l + 1];
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int Dp = Db * 8;

    for (int cs = start; cs < end; cs += bs) {
        const int i = cs + tid;
        const bool have = (i < end);
        const int n = have ? act_csc_row[i] : 0;
        const int p = have ? act_csc_pidx[i] : 0;
        double part = 0.0;
        for (int t = 0; t < T; ++t) {
            __syncthreads();
            const float* src = stnu + ((long long)t * L + l) * D;
            for (int d = tid; d < Dp; d += bs) nu_sm[d] = (d < D) ? src[d] : 0.0f;
            __syncthreads();
            if (have) {
                const float inv = row_inv[(long long)t * N + n];
                if (inv != 0.0f) {
                    const unsigned char* prow =
                        packed + ((long long)t * N + n) * Db;
                    float pt = 0.0f;
                    // 8 bytes fetched before the dependent fmaf chain
                    // consumes them (each thread walks its OWN member row, so
                    // the loads are scattered and their latency has to be
                    // hidden by hand).  Order is untouched -> bit-exact;
                    // measured 298 -> 263 us per launch.
                    int b = 0;
                    for (; b + ACC_PF <= Db; b += ACC_PF) {
                        unsigned int v[ACC_PF];
                        #pragma unroll
                        for (int j = 0; j < ACC_PF; ++j)
                            v[j] = (unsigned int)__ldg(prow + b + j);
                        #pragma unroll
                        for (int j = 0; j < ACC_PF; ++j) {
                            const float* sb = nu_sm + (b + j) * 8;
                            #pragma unroll
                            for (int k = 0; k < 8; ++k)
                                pt = fmaf((float)((v[j] >> k) & 1u), sb[k], pt);
                        }
                    }
                    for (; b < Db; ++b) {
                        const unsigned int bv = (unsigned int)__ldg(prow + b);
                        const float* sb = nu_sm + b * 8;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            pt = fmaf((float)((bv >> k) & 1u), sb[k], pt);
                        }
                    }
                    const float mean = row_mean[(long long)t * N + n];
                    part += (double)inv
                            * ((double)pt
                               - (double)mean * S_TL[(long long)t * L + l]);
                }
            }
        }
        if (have) lv_sum[p] = (float)part;
    }
}

// =====================================================================
// K7 -- iteration-1 out-of-P row max (only while theta_out != 0).
// One warp per row; lanes stride l.  gMSHBM: cross-hemisphere cells are
// -inf on the CPU (dense log_connect pre-fill), so they are skipped.
// dMSHBM: every out-of-P l participates and there is no lc term at all.
// =====================================================================
extern "C" __global__
void rmax_out_k(const int* __restrict__ row_ptr,
                const int* __restrict__ col,
                const float* __restrict__ lv_dense,    // (n_tile, L)
                const float* __restrict__ cross_dense, // (n_tile, L)
                const float* __restrict__ grad_sq,     // (N,)
                const float* __restrict__ u_sq,        // (L,)
                const int* __restrict__ n_alive,       // (N,)
                const float* __restrict__ kappa_f32_dev,
                const float* __restrict__ cdln_dev,
                float log_theta_out, float beta_f32, int has_spatial,
                int n_lh, int L_lh, int L, int T,
                int row_off, int n_tile,
                float* __restrict__ rmax_dense)        // (N,)
{
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (gw >= n_tile) return;
    const int lane = threadIdx.x & 31;
    const int n = row_off + gw;

    const int p0 = row_ptr[n], p1 = row_ptr[n + 1];
    const float kf = kappa_f32_dev[0];
    const float cdln_add = (float)n_alive[n] * cdln_dev[0];
    const bool n_lh_side = (n < n_lh);
    const float* lvr = lv_dense + (long long)gw * L;
    const float* crr = cross_dense + (long long)gw * L;
    const float gsq = grad_sq[n];

    float best = f_ninf();
    for (int l = lane; l < L; l += 32) {
        // in P(n)?  (row columns are ascending, <= 15 of them)
        bool inP = false;
        for (int p = p0; p < p1; ++p) {
            const int c = col[p];
            if (c == l) { inP = true; break; }
            if (c > l) break;
        }
        if (inP) continue;
        if (has_spatial && (n_lh_side != (l < L_lh))) continue;  // -inf on CPU

        float lam = (kf * lvr[l] + cdln_add) + log_theta_out;
        if (has_spatial) {
            const float vmf = (2.0f * crr[l] - gsq) - u_sq[l];
            float lc = 0.0f;
            for (int t = 0; t < T; ++t) lc += vmf;
            lam = lam + beta_f32 * lc;
        }
        if (lam == lam && lam > best) best = lam;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, off);
        if (ov == ov && ov > best) best = ov;
    }
    if (lane == 0) rmax_dense[n] = best;
}

// =====================================================================
// K8 -- per-row lam assembly, rmax and the fp64 softmax numerator.
// scr[] carries lam (fp64) between the two passes.
// =====================================================================
extern "C" __global__
void estep_row1(const int* __restrict__ row_ptr,
                const int* __restrict__ col,
                const float* __restrict__ lv_sum,
                const float* __restrict__ theta,
                const float* __restrict__ log_theta,
                const float* __restrict__ log_connect,
                const int* __restrict__ n_alive,
                const float* __restrict__ kappa_f32_dev,
                const float* __restrict__ cdln_dev,
                const float* __restrict__ rmax_dense,
                float beta_f32, int has_spatial, int use_rmax_dense, int N,
                float* __restrict__ log_vmf,
                double* __restrict__ scr,
                float* __restrict__ rmax_out)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int p0 = row_ptr[n], p1 = row_ptr[n + 1];
    const float kf = kappa_f32_dev[0];
    const float cdln_add = (float)n_alive[n] * cdln_dev[0];

    // INVARIANT (load-bearing, design 1.6 / 2.3).  At ``theta[p] == 0`` the
    // device writes ``lam = -inf`` WITHOUT the ``beta * log_connect`` term, so
    // ``scr = 0``, where the CPU (_kernels.py:711-717 adds ``beta*lc``
    // unconditionally) would produce ``-inf + beta*NaN = NaN``.  This pairs
    // with K9 (``dead_col``) sweeping NaN only over the ACTIVE CSC: the
    // device's explicit 0 at inactive cells is what keeps a NaN out of
    // ``estep_row2``'s ``rs`` accumulation.  The two backends agree because
    // the CPU's dead-column pass zeroes any all-NaN column.  Do not "align"
    // one of the two halves without the other.
    //
    // ``lv_sum`` and ``log_connect`` are only written for cells of the ACTIVE
    // support (K6 / K5b), so at an inactive cell they hold a value frozen from
    // an earlier EM iteration -- possibly NaN, if the parcel went momentarily
    // degenerate before it died.  ``log_vmf`` is published from here and is the
    // one term K10 does NOT sanitize (it computes ``0.0f * log_vmf``, which is
    // NaN for a NaN input), so the read is gated on the same ``theta > 0``.
    float rmax = f_ninf();
    for (int p = p0; p < p1; ++p) {
        const bool on = f32_pos(theta[p]);
        const float lvmf = on ? (kf * lv_sum[p] + cdln_add) : cdln_add;
        log_vmf[p] = lvmf;
        float lam;
        if (on) {
            lam = lvmf + log_theta[p];
            if (has_spatial) lam = lam + beta_f32 * log_connect[p];
        } else {
            lam = f_ninf();
        }
        scr[p] = (double)lam;
        if (lam == lam && lam > rmax) rmax = lam;
    }
    if (use_rmax_dense) {
        const float ro = rmax_dense[n];
        if (ro == ro && ro > rmax) rmax = ro;
    }
    if (!(rmax == rmax) || isinf(rmax)) rmax = 0.0f;
    rmax_out[n] = rmax;

    const double rmax_f64 = (double)rmax;
    for (int p = p0; p < p1; ++p) scr[p] = exp(scr[p] - rmax_f64);
}

// =====================================================================
// K9 -- dead-column zero on the active support (NaN-skipping fp64 sum).
// =====================================================================
extern "C" __global__
void dead_col(const int* __restrict__ act_col_ptr,
              const int* __restrict__ act_csc_pidx,
              double* __restrict__ scr)
{
    __shared__ double sh[256];
    const int l = blockIdx.x;
    const int start = act_col_ptr[l];
    const int end = act_col_ptr[l + 1];
    double acc = 0.0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const double v = scr[act_csc_pidx[i]];
        if (v == v) acc += v;
    }
    acc = block_reduce_f64(acc, sh);
    if (acc == 0.0) {
        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            scr[act_csc_pidx[i]] = 0.0;
        }
    }
}

// =====================================================================
// K10 -- per-row cost (fp32 chain) + E.1 normalize (fp64 rs, f32_rn store).
// =====================================================================
extern "C" __global__
void estep_row2(const int* __restrict__ row_ptr,
                const double* __restrict__ scr,
                const float* __restrict__ theta,
                const float* __restrict__ log_vmf,
                const float* __restrict__ log_connect,
                const int* __restrict__ n_alive,
                float beta_f32, int has_spatial, float log_eps20, int N,
                float* __restrict__ s_lambda_out,
                double* __restrict__ cost_part)
{
    __shared__ double sh[256];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    double row_cost = 0.0;

    if (n < N) {
        const int p0 = row_ptr[n], p1 = row_ptr[n + 1];
        const bool tmp_idx = (n_alive[n] == 0);

        // ``row_sum`` / ``slc`` go through the emulated fp32 ops: at the
        // first EM iteration an out-of-P cell can hold the row max by ~100
        // log-units, which puts EVERY in-P ``scr`` in the fp32 subnormal
        // band.  A native fp32 add would flush the whole row to zero (FTZ)
        // and drop its entire cost contribution, while the CPU keeps it.
        float row_sum = 0.0f;
        for (int p = p0; p < p1; ++p) row_sum = f32_add_rn(row_sum, f32_rn(scr[p]));

        float c = 0.0f;
        for (int p = p0; p < p1; ++p) {
            const float slc_raw = f32_rn(scr[p]);
            float slc = f32_pos(row_sum)
                        ? f32_rn(f32_to_f64(slc_raw) / f32_to_f64(row_sum))
                        : 0.0f;
            if (slc != slc) slc = 0.0f;
            if (tmp_idx) slc = 0.0f;

            const float th = theta[p];
            float ltheta;
            if (f32_pos(th)) {
                const double lt = log(f32_to_f64(th));
                ltheta = isinf(lt) ? log_eps20 : (float)lt;
            } else {
                ltheta = log_eps20;
            }
            float lslc;
            if (f32_pos(slc)) {
                const double ls = log(f32_to_f64(slc));
                lslc = isinf(ls) ? log_eps20 : (float)ls;
            } else {
                lslc = log_eps20;
            }
            c += slc * log_vmf[p];
            c += slc * ltheta;
            c -= slc * lslc;
            if (has_spatial) {
                float lc = log_connect[p];
                if (!(lc == lc)) lc = log_eps20;
                if (isinf(lc)) lc = log_eps20;
                c += (beta_f32 * slc) * lc;
            }
        }
        row_cost = (double)c;

        // E.1 -- bm == 1 on P, so the multiply is a no-op.
        double rs = 0.0;
        for (int p = p0; p < p1; ++p) rs += scr[p];
        if (rs > 0.0 && !tmp_idx) {
            for (int p = p0; p < p1; ++p) s_lambda_out[p] = f32_rn(scr[p] / rs);
        } else {
            for (int p = p0; p < p1; ++p) s_lambda_out[p] = 0.0f;
        }
    }

    row_cost = block_reduce_f64(row_cost, sh);
    if (threadIdx.x == 0) cost_part[blockIdx.x] = row_cost;
}

// =====================================================================
// K12 -- E.2 theta (emulated fp32 serial sum + fp32 scale), log_theta and
// the active flag.
// =====================================================================
extern "C" __global__
void theta_mean_logtheta(const float* __restrict__ s_lambda,  // (S,P)
                         float* __restrict__ theta,
                         float* __restrict__ log_theta,
                         int* __restrict__ active,
                         int S, long long P, float inv_S_f32)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float acc = 0.0f;
    bool any_nz = false;
    for (int s = 0; s < S; ++s) {
        const float v = s_lambda[(long long)s * P + p];
        any_nz = any_nz || f32_nonzero(v);
        acc = f32_add_rn(acc, v);
    }
    const float th = f32_mul_rn(acc, inv_S_f32);
    theta[p] = th;
    const bool on = f32_pos(th);
    // The active flag is the UNION of supp(theta) and supp(s_lambda), not
    // supp(theta) alone.  For S >= 2 a cell whose only nonzero weight is the
    // minimum subnormal 2^-149 gives acc = 2^-149 and acc * f32(1/2) = 2^-150,
    // an exact half-way tie that rounds to +0 -- so theta dies while
    // s_lambda[s,p] is still nonzero.  Compacting on theta alone would then
    // drop that member from K3's x_dot_sl and from K5a's sum_lambda /
    // u_update, which the CPU does visit (measured on testdata/step2_bench/proj2
    // at S=2: 158 such cells after EM iter 1, all exactly 1.401298e-45).
    // theta and log_theta keep their values; only the compaction widens.
    active[p] = (on || any_nz) ? 1 : 0;
    log_theta[p] = on ? (float)log(f32_to_f64(th)) : f_ninf();
}

// Stream-compaction scatter: out[incl[p]-1] = p wherever flag[p] != 0.
extern "C" __global__
void compact_scatter(const int* __restrict__ flag,
                     const int* __restrict__ incl,
                     int* __restrict__ out, long long P)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    if (flag[p]) out[incl[p] - 1] = (int)p;
}

// Gather an int32 P-vector through an index array (CSR order -> CSC order).
extern "C" __global__
void gather_i32(const int* __restrict__ src, const int* __restrict__ idx,
                int* __restrict__ out, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = src[idx[i]];
}

// Compact the CSC member lists down to the active cells.
extern "C" __global__
void compact_csc(const int* __restrict__ flag_csc,
                 const int* __restrict__ incl,     // inclusive scan of flag_csc
                 const int* __restrict__ csc_row,
                 const int* __restrict__ csc_pidx,
                 int* __restrict__ out_row, int* __restrict__ out_pidx,
                 long long P)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    if (flag_csc[i]) {
        const int o = incl[i] - 1;
        out_row[o] = csc_row[i];
        out_pidx[o] = csc_pidx[i];
    }
}

// Park an int32 device scalar in an fp64 slot (folds a D2H into another one).
extern "C" __global__
void write_scalar_d(const int* __restrict__ src, double* __restrict__ dst,
                    int slot)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[slot] = (double)src[0];
}
"""


_SRC_OUTER = r"""
// =====================================================================
// K13a -- one intra_subject_var iteration (L17), block per (s, l).
// Serial fp64 norm^2 and serial fp32 cosine on thread 0 keep the CPU's
// ascending-d op order; the zero-norm branch writes 0 (NOT the M-step's
// inf/NaN convention).
// =====================================================================
extern "C" __global__
void intra_psi_iter(const float* __restrict__ nu_STLD,
                    const float* __restrict__ psi_prev,
                    float* __restrict__ psi_new,
                    const float* __restrict__ sigma_cur,
                    const float* __restrict__ eps_mu_LD,
                    float* __restrict__ cos_LS,
                    int S, int T, int L, int D)
{
    extern __shared__ float sh_col[];
    __shared__ double sh_nrm;
    const int blk = blockIdx.x;
    const int s = blk / L;
    const int l = blk - s * L;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const float sig = sigma_cur[l];
    const long long off = ((long long)s * L + l) * D;

    for (int d = tid; d < D; d += bs) {
        float nu_sum = 0.0f;
        for (int t = 0; t < T; ++t) {
            nu_sum += nu_STLD[(((long long)s * T + t) * L + l) * D + d];
        }
        sh_col[d] = sig * nu_sum + eps_mu_LD[(long long)l * D + d];
    }
    __syncthreads();
    if (tid == 0) {
        double acc = 0.0;
        for (int d = 0; d < D; ++d) {
            const double v = (double)sh_col[d];
            acc += v * v;
        }
        sh_nrm = acc;
    }
    __syncthreads();
    const double nrm = sqrt(sh_nrm);

    if (nrm > 0.0) {
        const float inv = (float)(1.0 / nrm);
        for (int d = tid; d < D; d += bs) {
            const float vn = sh_col[d] * inv;
            psi_new[off + d] = vn;
            sh_col[d] = vn;
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += sh_col[d] * psi_prev[off + d];
            cos_LS[(long long)l * S + s] = acc;
        }
    } else {
        for (int d = tid; d < D; d += bs) psi_new[off + d] = 0.0f;
        if (tid == 0) cos_LS[(long long)l * S + s] = 0.0f;
    }
}

// sigma input: acc[l] = sum_{s,t,d} psi[s,l,d] * nu[s,t,l,d].
//
// Block per parcel (design 3, K13b: "block per l, fp64 dot over (s,t,d),
// fixed tree").  Thread ``tid`` folds ITS strided d-elements serially in
// ascending (s, t, d) order and the 256 partials go through the same
// fixed-shape ``block_reduce_f64`` tree the M-step uses, so the result is
// bit-reproducible run to run.  Reassociated vs the CPU's fully serial loop;
// design 7.1 bars sigma/epsil/mu at 1e-9 rel, which this is far inside.
extern "C" __global__
void intra_sigma_dot(const float* __restrict__ psi,   // (S,L,D)
                     const float* __restrict__ nu,    // (S,T,L,D)
                     double* __restrict__ out,        // (L,)
                     int S, int T, int L, int D)
{
    __shared__ double sh[256];
    const int l = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    double acc = 0.0;
    for (int s = 0; s < S; ++s) {
        const long long op = ((long long)s * L + l) * D;
        for (int t = 0; t < T; ++t) {
            const long long on = (((long long)s * T + t) * L + l) * D;
            for (int d = tid; d < D; d += bs) {
                acc += (double)psi[op + d] * (double)nu[on + d];
            }
        }
    }
    acc = block_reduce_f64(acc, sh);
    if (tid == 0) out[l] = acc;
}

// invad epilogue -- L blocks x 1 thread (2.4x faster than one L-thread block).
extern "C" __global__
void invad_L(const double* __restrict__ acc_L,
             const float* __restrict__ prev_L,
             float* __restrict__ out_L,
             double scale, double dim, double ini_val)
{
    if (threadIdx.x != 0) return;
    const int l = blockIdx.x;
    double v_in = acc_L[l] * scale;
    if (v_in > 1.0) v_in = 1.0;
    double v = dev_invad(dim, v_in);
    if (v < ini_val) v = ini_val;
    if (!isfinite(v)) v = (double)prev_L[l];
    out_L[l] = (float)v;
}

// Cdln epilogue -- L blocks x 1 thread.
extern "C" __global__
void cdln_L(const float* __restrict__ k_L, double* __restrict__ out_L, double v)
{
    if (threadIdx.x != 0) return;
    const int l = blockIdx.x;
    out_L[l] = dev_cdln_single((double)k_L[l], v);
}

// L17 flag latch + sigma relative drift.  ONE block, FLAG_BLOCK threads
// (a power of two; L <= 512 and S is small).
//
// PERF (F3): this used to be a single thread walking S*L + 3*L dependent
// global loads -- ~1 200 scalar ops, but 125 us per launch because every load
// was a fresh ~500-cycle round trip with nothing to overlap it.  The counts
// are integer (order-free) so they take a block reduction; ``rel_acc`` keeps
// the CPU's serial ascending-l fp64 order on thread 0 but now reads
// ``sigma_cur`` / ``sigma_new`` out of shared memory, staged coalesced by the
// whole block.  Bit-exact.
#define INTRA_FLAG_MAXL 512
extern "C" __global__
void intra_flags_rel(const float* __restrict__ cos_LS,
                     int* __restrict__ flag_psi,
                     float* __restrict__ sigma_cur,
                     const float* __restrict__ sigma_new,
                     double* __restrict__ result,
                     float eps, int S, int L)
{
    __shared__ float sh_cur[INTRA_FLAG_MAXL];
    __shared__ float sh_new[INTRA_FLAG_MAXL];
    __shared__ int sh_i[1024];
    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    for (int l = tid; l < L; l += bs) {
        sh_cur[l] = sigma_cur[l];
        sh_new[l] = sigma_new[l];
    }

    for (int s = 0; s < S; ++s) {
        int c = 0;
        for (int l = tid; l < L; l += bs) {
            if ((1.0f - cos_LS[(long long)l * S + s]) < eps) c += 1;
        }
        sh_i[tid] = c;
        __syncthreads();
        for (int q = bs >> 1; q > 0; q >>= 1) {
            if (tid < q) sh_i[tid] += sh_i[tid + q];
            __syncthreads();
        }
        if (tid == 0 && sh_i[0] >= L) flag_psi[s] = 1;
        __syncthreads();
    }

    if (tid == 0) {
        double rel_acc = 0.0;
        for (int l = 0; l < L; ++l) {
            const float diff = sh_cur[l] - sh_new[l];   // fp32 subtract first
            rel_acc += fabs((double)diff / (double)sh_cur[l]);
        }
        int fs = 0;
        for (int s = 0; s < S; ++s) fs += flag_psi[s];
        result[0] = (double)fs;
        result[1] = rel_acc / (double)L;
    }
    __syncthreads();
    for (int l = tid; l < L; l += bs) sigma_cur[l] = sh_new[l];
}

// =====================================================================
// K13c -- inter_subject_var (L18).
// =====================================================================
// mu_new[l,d] = sum_s psi[s,l,d]; the norm^2 accumulates the UNROUNDED fp64
// column in the CPU's serial ascending-d order (thread 0 over a smem stage).
extern "C" __global__
void inter_mu(const float* __restrict__ psi,      // (S,L,D)
              const float* __restrict__ prev_mu,  // (L,D)
              float* __restrict__ mu_new,         // (L,D)
              int S, int L, int D)
{
    extern __shared__ double sh_v[];              // D doubles
    __shared__ double sh_n2;
    const int l = blockIdx.x;
    const long long off = (long long)l * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        double v = 0.0;
        for (int s = 0; s < S; ++s) v += (double)psi[((long long)s * L + l) * D + d];
        mu_new[off + d] = (float)v;
        sh_v[d] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double acc = 0.0;
        for (int d = 0; d < D; ++d) acc += sh_v[d] * sh_v[d];
        sh_n2 = acc;
    }
    __syncthreads();
    const double nrm = sqrt(sh_n2);
    if (nrm > 0.0) {
        const float inv = (float)(1.0 / nrm);
        for (int d = threadIdx.x; d < D; d += blockDim.x) mu_new[off + d] *= inv;
    } else {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            mu_new[off + d] = prev_mu[off + d];
        }
    }
}

// Serial fp64 in the CPU's exact (s, d) order -- one thread per parcel.
//
// DELIBERATELY NOT the block-per-parcel tree the other two dots use.  This dot
// feeds ``epsil = invAd(dim, min(acc/S, 1))``, which is knife-edge: at S=1
// ``mu_new == s_psi/||s_psi||`` so ``acc/S`` sits at ``1 - O(1e-16)`` and
// invAd amplifies a last-ulp change by ~5e9.  MEASURED on this box with the
// tree here: the unit bar ``epsil <= 1e-9 rel`` (design 7.1) becomes 5.40e-07,
// and the S=1 end-to-end A/B against ``out_cpu_ref`` degrades from
// (Record len 10, theta argmax flips 0, support-diff rows 1, kappa max-rel
// 0.0) to (len 7 -- the outer-inter loop hits its stop test three iterations
// early, flips 4, rows 13, kappa 2.45e-06).  Reverting only this kernel
// restores every one of those numbers exactly.  The tree would have bought
// 159 us -> 20 us per launch, x10 launches: 1.6 ms at S=1 / 3.0 ms at S=2,
// i.e. ~11 % / ~4 % of the post-fix closure.  Not worth it.
extern "C" __global__
void inter_eps_dot(const float* __restrict__ psi,    // (S,L,D)
                   const float* __restrict__ mu,     // (L,D)
                   double* __restrict__ out, int S, int L, int D)
{
    if (threadIdx.x != 0) return;
    const int l = blockIdx.x;
    const long long om = (long long)l * D;
    double acc = 0.0;
    for (int s = 0; s < S; ++s) {
        const long long op = ((long long)s * L + l) * D;
        for (int d = 0; d < D; ++d) {
            acc += (double)psi[op + d] * (double)mu[om + d];
        }
    }
    out[l] = acc;
}

// =====================================================================
// K13d -- intra_em_cost_step2 (L16).
// =====================================================================
// Block per parcel + fixed fp64 tree (design 3, K13d "block per l + reduce").
// Both dots use the same per-thread-serial / tree-combine shape as
// ``intra_sigma_dot``; design 7.1 bars the cost at 1e-10 rel.
extern "C" __global__
void cost_terms_perl(const float* __restrict__ psi,   // (S,L,D)
                     const float* __restrict__ nu,    // (S,T,L,D)
                     const float* __restrict__ mu,    // (L,D)
                     double* __restrict__ term1,      // (L,)
                     double* __restrict__ term2,      // (L,)
                     int S, int T, int L, int D)
{
    __shared__ double sh[256];
    const int l = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const long long om = (long long)l * D;
    double acc1 = 0.0;
    for (int s = 0; s < S; ++s) {
        const long long op = ((long long)s * L + l) * D;
        for (int t = 0; t < T; ++t) {
            const long long on = (((long long)s * T + t) * L + l) * D;
            for (int d = tid; d < D; d += bs) {
                acc1 += (double)psi[op + d] * (double)nu[on + d];
            }
        }
    }
    acc1 = block_reduce_f64(acc1, sh);
    if (tid == 0) term1[l] = acc1;

    double acc2 = 0.0;
    for (int s = 0; s < S; ++s) {
        const long long op = ((long long)s * L + l) * D;
        for (int d = tid; d < D; d += bs) {
            acc2 += (double)psi[op + d] * (double)mu[om + d];
        }
    }
    acc2 = block_reduce_f64(acc2, sh);
    if (tid == 0) term2[l] = acc2;
}

extern "C" __global__
void cost_epilogue(const double* __restrict__ term1,
                   const double* __restrict__ term2,
                   const double* __restrict__ cdln_sigma,
                   const double* __restrict__ cdln_epsil,
                   const float* __restrict__ sigma,
                   const float* __restrict__ epsil,
                   const double* __restrict__ cost_em,
                   double* __restrict__ out, int S, int T, int L)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double t1d = 0.0, t1c = 0.0, t2d = 0.0, t2c = 0.0;
    for (int l = 0; l < L; ++l) {
        t1d += (double)sigma[l] * term1[l];
        t2d += (double)epsil[l] * term2[l];
        t1c += cdln_sigma[l];
        t2c += cdln_epsil[l];
    }
    t1c *= (double)(S * T);
    t2c *= (double)S;
    double ce = 0.0;
    for (int s = 0; s < S; ++s) ce += cost_em[s];
    out[0] = t1d + t1c + t2d + t2c + ce;
}

// =====================================================================
// K13e -- resets and small fills.
// =====================================================================
extern "C" __global__
void broadcast_LD(const float* __restrict__ src_LD, float* __restrict__ dst,
                  int L, int D, long long n_total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    const long long ld = i % ((long long)L * D);
    dst[i] = src_LD[ld];
}

extern "C" __global__
void fill_f32(float* __restrict__ x, float v, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

extern "C" __global__
void eps_mu_kernel(const float* __restrict__ epsil, const float* __restrict__ mu,
                   float* __restrict__ out, int D, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = epsil[i / D] * mu[i];
}
"""


# ``(key, RawModule)`` published as ONE tuple so a concurrent reader can never
# observe a module paired with a stale key.  ``module()`` is called from the
# driver's prewarm daemon thread AND from ``Step2SparseSession.__init__`` on
# the main thread, so the build itself is serialised by a lock: without it two
# threads can each start a 1.9 s NVRTC compile of the identical source.
_MODULE_CELL: Optional[Tuple[Tuple[int, int, int, int], Any]] = None
_MODULE_LOCK = threading.Lock()


def module():
    """Compile (once per process) and return the RawModule. Thread-safe.

    Note that ``cupy.RawModule(backend='nvrtc')`` is LAZY: the constructor
    parses nothing.  The NVRTC compile fires on the first ``get_function`` --
    which is why :func:`warmup_step2_gpu` asks for a symbol.
    """
    global _MODULE_CELL
    key = (XDOT_MAX_NB, XDOT_BLOCK, INIT_WARPS, INIT_RR, INIT_MAXJ)
    cell = _MODULE_CELL                      # single read of the published pair
    if cell is not None and cell[0] == key:
        return cell[1]
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for the step-2 gpu backend")
    with _MODULE_LOCK:
        cell = _MODULE_CELL
        if cell is not None and cell[0] == key:
            return cell[1]
        src = (_SRC_HELPERS + _SRC_ROWSTATS + _SRC_INIT + _SRC_MSTEP
               + _SRC_CONNECT + _SRC_ESTEP + _SRC_OUTER)
        src = (src.replace("XDOT_MAX_NB_VALUE", str(XDOT_MAX_NB))
                  .replace("XDOT_BS_VALUE", str(XDOT_BLOCK))
                  .replace("INIT_WARPS_VALUE", str(INIT_WARPS))
                  .replace("INIT_RR_VALUE", str(INIT_RR))
                  .replace("INIT_MAXJ_VALUE", str(INIT_MAXJ)))
        if not src.isascii():
            raise RuntimeError("CUDA source must be ASCII")
        mod = cp.RawModule(code=src, options=("-std=c++17", "-fmad=false"),
                           backend="nvrtc")
        _MODULE_CELL = (key, mod)
        return mod


def warmup_step2_gpu() -> None:
    """Force the NVRTC compile and the cuBLAS setup. Thread-safe.

    ``cp.RawModule`` compiles lazily, so building the object is not enough:
    the ``get_function`` below is what drives the whole module through NVRTC
    (~1.9 s on a cold cupy kernel cache, ~150 ms on a warm one).  Doing it
    here instead of inside :class:`Step2SparseSession` keeps it off the timed
    path -- measured fresh-process ``session_ctor`` 0.95 s with no prewarm,
    0.12 s with a ``module()``-only prewarm.

    The tiny gemm pays cuBLAS's process-wide kernel-module load, which the
    iteration-1 dense pass (K7) would otherwise charge to the first
    ``run_iter``. cupy's cuBLAS handle is per **thread**, so the gemm issued
    here cannot re-point the handle another thread is using; it is safe on
    the driver's prewarm daemon.
    """
    module().get_function("row_stats_exact")
    a = cp.zeros((32, 32), dtype=cp.float32)
    cp.matmul(a, a)
    cp.cuda.get_current_stream().synchronize()


# ─────────────────────────────────────────────────────────────────────
# Host-side constants shared with the Session
# ─────────────────────────────────────────────────────────────────────
LOG_EPS20_F32 = np.float32(math.log(float(np.finfo(np.float64).eps) ** 20))
EPS_F64 = float(np.finfo(np.float64).eps)


def grid(n: int, block: int) -> int:
    return (int(n) + block - 1) // block


def init_dt() -> int:
    """d-tile width of ``init_hard_labels``: the largest 8-multiple of
    ``d`` values whose ``at`` stage (``INIT_ROWS * DT`` floats) fits in
    shared memory.  DT must be a multiple of 8 because the stage is
    byte-based (one packed byte -> 8 ``d`` values); every DT gives
    bit-identical labels (the per-(n,l) accumulator stays a private
    serial ascending-d fp32 sum).
    """
    dt = (20000 // (4 * INIT_ROWS)) & ~7
    return max(8, min(128, int(dt)))


def check_dims(D: int, Db: int, L: int, dim: int, D_grad: int) -> None:
    """Static limits of the backend (design §3 / §9)."""
    if Db > 256:
        raise ValueError(
            f"backend='gpu' supports ceil(D/8) <= 256 (D <= 2048); got "
            f"D={D} (Db={Db}). The profile length is the seed-mesh vertex "
            f"count + 1, so this needs a seed mesh of at most fsaverage3 "
            f"(D=1175). Use backend='cpu' for larger seeds."
        )
    if L > MAX_CLUSTERS:
        raise ValueError(
            f"backend='gpu' supports num_clusters <= {MAX_CLUSTERS}; got L={L}"
        )
    if D_grad > MAX_D_GRAD:
        raise ValueError(
            f"backend='gpu' supports n_grad_components <= {MAX_D_GRAD}; "
            f"got D_grad={D_grad}. ``connect_u`` stages one float per gradient "
            f"component in shared memory. Use backend='cpu'."
        )
    v = float(dim) * 0.5 - 1.0
    if not (v >= 25.0):
        raise ValueError(
            f"backend='gpu' requires dim/2 - 1 >= 25 (the Debye branch of "
            f"log I_v); got dim={dim} (v={v})."
        )


__all__ = [
    "module", "warmup_step2_gpu", "grid", "init_dt", "check_dims",
    "LOG_EPS20_F32", "EPS_F64",
    "XDOT_BLOCK", "XDOT_MAX_NB", "XDOT_CHUNK", "FUSED_BLOCK",
    "REDUCE_BLOCK", "REDUCE_GRID", "ROW_BLOCK", "INIT_ROWS", "INIT_MAXJ",
    "MAX_CLUSTERS", "MAX_D_GRAD", "CONNECT_STATIC_SHARED",
    "ACC_BLOCK", "CONNECT_BLOCK", "FLAG_BLOCK", "INIT_WARPS", "INIT_RR", "INIT_BLOCK",
]
