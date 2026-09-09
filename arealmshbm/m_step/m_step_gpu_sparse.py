"""m_step_gpu_sparse.py

GPU M-step for the step-3 ``gpu_sparse`` backend (design contract:
``docs/step3_sparse_design.md`` section 1 + section 2.3).

Differences vs the dense backends
---------------------------------
* ``X`` is **never materialised**. The BOLD lives on device as the
  on-disk bit-packed ``(T, N, D_bytes)`` uint8 buffer plus the two
  per-row fp32 statistics ``row_mean`` / ``row_inv``; the M-step's only
  contact with it is::

      X_dot_sl[t, l, d] = sum_{n in members(l)} sl[n, l] * X[n, t, d]
                        = sum_{n : bit(t,n,d)} w_nt - sum_n w_nt*row_mean[t,n]
        with w_nt = sl[n,l] * row_inv[t, n]

  which :func:`x_dot_sl_bits` evaluates as a bit-sum (the second term is
  per-``(t, l)``, independent of ``d``).
* ``s_lambda`` is the ``(P,)`` candidate vector (CSR order); members of
  parcel ``l`` are read through the layout's CSC arrays
  (``col_ptr`` / ``csc_row`` / ``csc_pidx``).
* Per-``(t, l, d)`` buffers are in **``(T, L, D)`` layout** (D
  contiguous), not the CPU's ``(T, D, L)``.

Determinism
-----------
No floating-point atomics anywhere. Every reduction is either a fixed
per-thread serial accumulation or a fixed-shape shared-memory tree over
a fixed grid, so the backend is run-to-run bit-reproducible.

Numerical mirroring of the CPU reference
(``arealmshbm/m_step/_kernels.py``)
----------------------------------------
* ``kappa_sum``: product formed in fp32, accumulated in fp64 (numba's
  ``acc += a*b`` with an fp64 ``acc`` and fp32 operands).
* column norm: ``v*v`` in fp32, summed in fp64; ``inv_cn = fp32(1.0/cn)``
  (fp64 divide, then a single cast).
* ``cos``: ``new*old`` in fp32, summed in fp64, stored fp32.
* empty parcels: ``cn == 0 -> inv_cn = inf -> 0*inf = NaN`` (IEEE, not
  trapping). Note CuPy appends ``-ftz=true`` to every nvrtc compile, so
  fp32 denormals flush to zero in this module; none of the quantities
  here get within ~1e-30 of that boundary.
* This module compiles with FMA contraction ON (``options=("-std=c++14",)``
  — no ``-fmad=false``, unlike ``vmf_clustering/_kernels_gpu_sparse``).
  That is the flag the design contract was validated under; leave it.
* ``kappa`` comes from the host scalar root-finder :func:`invad`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

try:  # cupy is optional at import time (CPU-only boxes import the package)
    import cupy as cp
except Exception:  # pragma: no cover - exercised only without cupy
    cp = None  # type: ignore

from ._invad import invad


# ---------------------------------------------------------------------
# Launch geometry (also the compile-time limits baked into the kernels)
# ---------------------------------------------------------------------
ROW_STATS_BLOCK = 128          # 4 warps -> 4 rows per block
XDOT_BLOCK = 128               # threads own BYTES of the packed row
XDOT_MAX_NB = 2                # bytes per thread -> D_bytes <= MAX_D_BYTES
                               # (vmf_clustering/sparse_layout.py)
XDOT_CHUNK = 64                # members staged per shared-memory chunk
FUSED_BLOCK = 256              # block-reduce over D
REDUCE_BLOCK = 256
REDUCE_GRID = 1024             # FIXED grid -> fixed reduction order


_CUDA_SRC = r"""
// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
// (a) per-(t, n) row statistics from the packed bit stream.
//     One warp per row; the popcount is exact integer -> deterministic.
// ---------------------------------------------------------------------
extern "C" __global__
void row_stats(const unsigned char* __restrict__ packed,   // (n_rows, D_bytes)
               float* __restrict__ row_mean,               // (n_rows,)
               float* __restrict__ row_inv,                // (n_rows,)
               int n_rows, int D, int D_bytes)
{
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int row  = blockIdx.x * (blockDim.x >> 5) + wid;
    if (row >= n_rows) return;

    const unsigned char* src = packed + (size_t)row * (size_t)D_bytes;
    int pop = 0;
    for (int b = lane; b < D_bytes; b += 32) {
        pop += __popc((unsigned int)src[b]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        pop += __shfl_down_sync(0xffffffffu, pop, off);
    }
    if (lane == 0) {
        double sum_d  = (double)pop;              // exact in fp64
        double mean_d = sum_d / (double)D;
        double post   = sum_d - (double)D * mean_d * mean_d;
        row_mean[row] = (float)mean_d;
        // has_zero gate: for binary rows it fires exactly when the row is
        // identically zero after demeaning, i.e. pop in {0, D}.
        row_inv[row]  = (pop == 0 || pop == D)
                        ? 0.0f
                        : (float)(1.0 / sqrt(post));
    }
}

// ---------------------------------------------------------------------
// (b) X_dot_sl[t, l, d] = sum_{n in members(l)} sl[n,l] * X[n,t,d]
//     One block per (t, l); threads own BYTES of the packed row so a
//     single shared-memory byte load feeds 8 accumulators.
// ---------------------------------------------------------------------
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
    extern __shared__ unsigned char smem_raw[];
    float* sh_w = (float*)smem_raw;                          // chunk floats
    unsigned char* sh_bytes = smem_raw + (size_t)chunk * 4;  // chunk*D_bytes
    __shared__ double sh_red[XDOT_BS];
    __shared__ double sh_B;

    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const int blk = blockIdx.x;
    const int t   = blk / L;
    const int l   = blk - t * L;

    const int start = col_ptr[l];
    const int end   = col_ptr[l + 1];

    const unsigned char* base_t =
        packed + (size_t)t * (size_t)N * (size_t)D_bytes;
    const float* rm_t = row_mean + (size_t)t * (size_t)N;
    const float* ri_t = row_inv  + (size_t)t * (size_t)N;
    float* out_tl = out + ((size_t)t * (size_t)L + (size_t)l) * (size_t)D;

    // --- B = sum_n w_nt * row_mean[t, n]  (per-(t,l), independent of d) ---
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

    // --- A_d = sum_{n : bit(t,n,d)} w_nt ---
    float  a32[XDOT_MAX_NB * 8];
    double acc[XDOT_MAX_NB * 8];
    #pragma unroll
    for (int k = 0; k < XDOT_MAX_NB * 8; ++k) acc[k] = 0.0;

    for (int cs = start; cs < end; cs += chunk) {
        const int cnt = (end - cs) < chunk ? (end - cs) : chunk;
        __syncthreads();
        // Member-outer / byte-inner staging: consecutive threads read
        // consecutive bytes of one member row (coalesced) and there is no
        // integer division by the runtime ``D_bytes`` in the loop.
        for (int i = 0; i < cnt; ++i) {
            const int n = csc_row[cs + i];
            const unsigned char* g = base_t + (size_t)n * (size_t)D_bytes;
            unsigned char* s = sh_bytes + (size_t)i * (size_t)D_bytes;
            for (int b = tid; b < D_bytes; b += bs) s[b] = g[b];
        }
        for (int i = tid; i < cnt; i += bs) {
            const int n = csc_row[cs + i];
            sh_w[i] = s_lambda_P[csc_pidx[cs + i]] * ri_t[n];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < XDOT_MAX_NB * 8; ++k) a32[k] = 0.0f;

        // <= chunk (64) members folded in fp32, then one fp64 fold.
        for (int i = 0; i < cnt; ++i) {
            const float w = sh_w[i];
            const unsigned char* rp = sh_bytes + (size_t)i * (size_t)D_bytes;
            #pragma unroll
            for (int j = 0; j < XDOT_MAX_NB; ++j) {
                const int b = tid + j * XDOT_BS;
                if (b < D_bytes) {
                    const unsigned int bv = (unsigned int)rp[b];
                    // Branch-free form. ``bit`` is exactly 0.0f or 1.0f, so
                    // fma(w, bit, a) == a + w (bit=1) / == a (bit=0) with a
                    // single rounding either way -- bit-identical to the
                    // predicated ``if (bit) a += w`` but divergence-free.
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

// ---------------------------------------------------------------------
// sigma_psi[l, d] = sigma[l] * s_psi[l, d]
// ---------------------------------------------------------------------
extern "C" __global__
void sigma_psi_LD(const float* __restrict__ sigma_L,
                  const float* __restrict__ s_psi_LD,
                  float* __restrict__ out_LD,
                  int L, int D)
{
    const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (size_t)L * (size_t)D) return;
    out_LD[n] = sigma_L[n / (size_t)D] * s_psi_LD[n];
}

// ---------------------------------------------------------------------
// fp64 sum of an fp32 vector; fixed grid -> fixed order (stage 1)
// ---------------------------------------------------------------------
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

// kappa_sum stage 1: product in fp32, accumulate fp64 (mirrors numba).
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

// stage 2: single block, fixed tree over the (REDUCE_GRID,) partials.
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

// ---------------------------------------------------------------------
// (c) fused per-(t, l) M-step body, (T, L, D) layout.
//     One block per (t, l); block reduction over D.
// ---------------------------------------------------------------------
extern "C" __global__
void fused_iter_body(float kappa_f32,
                     const float* __restrict__ X_dot_sl,   // (T, L, D)
                     const float* __restrict__ sigma_psi,  // (L, D)
                     const float* __restrict__ st_old,     // (T, L, D)
                     float* __restrict__ st_new,           // (T, L, D)
                     float* __restrict__ cos_TL,           // (T, L)
                     int L, int D)
{
    __shared__ double sh[256];
    __shared__ float sh_inv;
    extern __shared__ float sh_col[];   // D floats -- keeps ``col`` on chip

    const int blk = blockIdx.x;
    const int t = blk / L;
    const int l = blk - t * L;
    const size_t off = (size_t)blk * (size_t)D;
    const size_t off_sp = (size_t)l * (size_t)D;

    // pass 1: col = kappa*X_dot_sl + sigma_psi ; acc_n = sum col^2 (fp64)
    double acc_n = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = kappa_f32 * X_dot_sl[off + d] + sigma_psi[off_sp + d];
        sh_col[d] = v;
        acc_n += (double)(v * v);
    }
    acc_n = block_reduce_f64(acc_n, sh);
    if (threadIdx.x == 0) {
        // cn == 0 -> inv_cn = inf -> 0*inf = NaN (empty parcel), as CPU.
        sh_inv = (float)(1.0 / sqrt(acc_n));
    }
    __syncthreads();
    const float inv_cn = sh_inv;

    // pass 2: normalize in place + cosine vs old
    double acc_c = 0.0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float nv = sh_col[d] * inv_cn;
        st_new[off + d] = nv;
        acc_c += (double)(nv * st_old[off + d]);
    }
    acc_c = block_reduce_f64(acc_c, sh);
    if (threadIdx.x == 0) cos_TL[(size_t)t * (size_t)L + l] = (float)acc_c;
}

// ---------------------------------------------------------------------
// per-t monotone convergence flag: all_l ((1 - cos) < eps).
// grid = T blocks. Writes the accumulated flag into result[slot0 + t].
// ---------------------------------------------------------------------
extern "C" __global__
void conv_flags(const float* __restrict__ cos_TL,   // (T, L)
                signed char* __restrict__ flag_acc, // (T,)
                double* __restrict__ result,        // (>= slot0 + T,)
                float eps, int L, int slot0)
{
    __shared__ int sh[256];
    const int t = blockIdx.x;
    const float* row = cos_TL + (size_t)t * (size_t)L;
    int c = 0;
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        // NaN fails '<' -> counts as NOT converged (matches CPU).
        if ((1.0f - row[l]) < eps) c += 1;
    }
    sh[threadIdx.x] = c;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (sh[0] >= L) flag_acc[t] = (signed char)1;
        result[slot0 + t] = (double)flag_acc[t];
    }
}
"""


_MODULE = None


def _module():
    """Compile (once per process) and return the RawModule."""
    global _MODULE
    if _MODULE is None:
        if cp is None:  # pragma: no cover
            raise RuntimeError("cupy is required for the gpu_sparse M-step")
        src = (_CUDA_SRC
               .replace("XDOT_MAX_NB_VALUE", str(XDOT_MAX_NB))
               .replace("XDOT_BS_VALUE", str(XDOT_BLOCK)))
        # NO --use_fast_math: the empty-parcel contract needs IEEE 0*inf.
        _MODULE = cp.RawModule(code=src, options=("-std=c++14",))
    return _MODULE


def _k(name: str):
    return _module().get_function(name)


# ---------------------------------------------------------------------
# (a) row statistics
# ---------------------------------------------------------------------
def compute_row_stats(bold_packed_dev, D: int):
    """Per-``(t, n)`` ``row_mean`` / ``row_inv`` from the packed bit stream.

    Parameters
    ----------
    bold_packed_dev : (T, N, D_bytes) uint8 device array, C-contiguous.
        Padding bits (``d >= D``) must be zero.
    D : int -- unpacked cell count.

    Returns
    -------
    (row_mean (T, N) fp32, row_inv (T, N) fp32) device arrays.
    """
    if bold_packed_dev.dtype != cp.uint8:
        raise ValueError(
            f"bold_packed must be uint8; got {bold_packed_dev.dtype}")
    if bold_packed_dev.ndim != 3:
        raise ValueError(f"bold_packed must be 3D (T, N, D_bytes); got "
                         f"{bold_packed_dev.shape}")
    if not bold_packed_dev.flags["C_CONTIGUOUS"]:
        raise ValueError("bold_packed must be C-contiguous")
    T, N, D_bytes = bold_packed_dev.shape
    if D_bytes != (int(D) + 7) // 8:
        raise ValueError(f"D_bytes {D_bytes} != ceil(D/8) for D={D}")

    row_mean = cp.empty((T, N), dtype=cp.float32)
    row_inv = cp.empty((T, N), dtype=cp.float32)
    n_rows = T * N
    rows_per_block = ROW_STATS_BLOCK // 32
    grid = (n_rows + rows_per_block - 1) // rows_per_block
    _k("row_stats")(
        (grid,), (ROW_STATS_BLOCK,),
        (bold_packed_dev, row_mean, row_inv,
         np.int32(n_rows), np.int32(D), np.int32(D_bytes)),
    )
    return row_mean, row_inv


# ---------------------------------------------------------------------
# (b) X_dot_sl
# ---------------------------------------------------------------------
def x_dot_sl_bits(bold_packed_dev, row_mean_dev, row_inv_dev,
                  col_ptr_dev, csc_row_dev, csc_pidx_dev,
                  s_lambda_P_dev, D: int, out_TLD_dev) -> None:
    """``out[t, l, d] = sum_{n in members(l)} sl[n,l] * X[n, t, d]``.

    ``out_TLD_dev`` is ``(T, L, D)`` fp32 and is fully overwritten.
    """
    T, N, D_bytes = bold_packed_dev.shape
    L = int(col_ptr_dev.size) - 1
    if out_TLD_dev.shape != (T, L, int(D)):
        raise ValueError(f"out must be (T, L, D) = ({T}, {L}, {D}); got "
                         f"{out_TLD_dev.shape}")
    if D_bytes > XDOT_MAX_NB * XDOT_BLOCK:
        raise ValueError(
            f"D_bytes={D_bytes} exceeds the kernel limit "
            f"{XDOT_MAX_NB * XDOT_BLOCK} (raise XDOT_MAX_NB)"
        )
    shared = XDOT_CHUNK * 4 + XDOT_CHUNK * D_bytes
    _k("x_dot_sl_bits")(
        (T * L,), (XDOT_BLOCK,),
        (bold_packed_dev, row_mean_dev, row_inv_dev,
         col_ptr_dev, csc_row_dev, csc_pidx_dev, s_lambda_P_dev,
         out_TLD_dev,
         np.int32(N), np.int32(L), np.int32(D), np.int32(D_bytes),
         np.int32(XDOT_CHUNK)),
        shared_mem=shared,
    )


# ---------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------
class MStepGPU:
    """Device-resident M-step inner-while loop (design contract 2.3).

    Reproduces :meth:`arealmshbm.m_step.m_step.MStepSession.run` -- same
    convergence test, same monotone per-``t`` flag latch, same empty-parcel
    NaN, same ``invad`` host root-find -- on the sparse candidate layout.

    Buffers are allocated once in ``__init__``; ``run`` allocates nothing.
    Exactly one host synchronisation per ``iter_m`` (a single
    ``8*(T+2)``-byte D2H that carries both the next iteration's
    ``kappa_sum`` and the accumulated per-``t`` convergence flags), plus
    one at per-call setup.
    """

    __slots__ = (
        "T", "D", "L", "P", "N", "dim", "epsilon", "max_iter",
        "bold_packed", "row_mean", "row_inv", "layout",
        "col_ptr", "csc_row", "csc_pidx",
        "_st_A", "_st_B", "_X_dot_sl", "_sigma_psi", "_cos_TL",
        "_part", "_result", "_flag_acc", "_host",
    )

    def __init__(self, T: int, D: int, L: int, dim: float,
                 epsilon: float, max_iter: int,
                 bold_packed_dev, row_mean_dev, row_inv_dev,
                 layout_dev: Dict[str, Any]):
        if cp is None:  # pragma: no cover
            raise RuntimeError("cupy is required for MStepGPU")
        self.T = int(T)
        self.D = int(D)
        self.L = int(L)
        self.dim = float(dim)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)

        if int(bold_packed_dev.shape[0]) != self.T:
            raise ValueError("bold_packed T mismatch")
        self.N = int(bold_packed_dev.shape[1])
        self.bold_packed = bold_packed_dev
        self.row_mean = row_mean_dev
        self.row_inv = row_inv_dev
        self.layout = layout_dev
        self.col_ptr = layout_dev["col_ptr"]
        self.csc_row = layout_dev["csc_row"]
        self.csc_pidx = layout_dev["csc_pidx"]
        self.P = int(layout_dev["P"])
        if int(self.col_ptr.size) - 1 != self.L:
            raise ValueError("layout L mismatch")

        T_, L_, D_ = self.T, self.L, self.D
        self._st_A = cp.empty((T_, L_, D_), dtype=cp.float32)
        self._st_B = cp.empty((T_, L_, D_), dtype=cp.float32)
        self._X_dot_sl = cp.empty((T_, L_, D_), dtype=cp.float32)
        self._sigma_psi = cp.empty((L_, D_), dtype=cp.float32)
        self._cos_TL = cp.empty((T_, L_), dtype=cp.float32)
        self._part = cp.empty(REDUCE_GRID, dtype=cp.float64)
        # result[0] = kappa_sum, result[1] = sum(s_lambda), result[2+t] = flag
        self._result = cp.empty(2 + T_, dtype=cp.float64)
        self._flag_acc = cp.empty(T_, dtype=cp.int8)
        import cupyx
        self._host = cupyx.empty_pinned(2 + T_, dtype=np.float64)

    # ---------- internals ----------
    def _launch_kappa_sum(self, buf) -> None:
        _k("kappa_sum_stage1")(
            (REDUCE_GRID,), (REDUCE_BLOCK,),
            (buf, self._X_dot_sl, self._part, np.int64(buf.size)),
        )
        _k("reduce_stage2")(
            (1,), (REDUCE_BLOCK,),
            (self._part, self._result, np.int32(0), np.int32(REDUCE_GRID)),
        )

    def _launch_flags(self) -> None:
        _k("conv_flags")(
            (self.T,), (REDUCE_BLOCK,),
            (self._cos_TL, self._flag_acc, self._result,
             np.float32(self.epsilon), np.int32(self.L), np.int32(2)),
        )

    # ---------- per-call API ----------
    def run(self, s_t_nu_TLD_in_dev, s_lambda_P_dev, s_psi_LD_dev,
            sigma_L_dev, kappa_init: float) -> Tuple[Any, float, int]:
        """One full M-step inner-while loop.

        Parameters
        ----------
        s_t_nu_TLD_in_dev : (T, L, D) fp32 device -- read only.
        s_lambda_P_dev    : (P,) fp32 device -- CSR-order candidate weights.
        s_psi_LD_dev      : (L, D) fp32 device.
        sigma_L_dev       : (L,) fp32 device.
        kappa_init        : float -- uniform kappa seed.

        Returns
        -------
        (s_t_nu_out_dev, kappa, iter_m) -- the output is one of the two
        internal ping-pong buffers; treat it as read-only until the next
        ``run``.
        """
        T, L, D = self.T, self.L, self.D
        if s_t_nu_TLD_in_dev.shape != (T, L, D):
            raise ValueError(f"s_t_nu must be (T, L, D) = ({T}, {L}, {D}); "
                             f"got {s_t_nu_TLD_in_dev.shape}")
        if s_psi_LD_dev.shape != (L, D):
            raise ValueError(f"s_psi must be (L, D) = ({L}, {D}); got "
                             f"{s_psi_LD_dev.shape}")
        if s_lambda_P_dev.shape != (self.P,):
            raise ValueError(f"s_lambda_P must be ({self.P},); got "
                             f"{s_lambda_P_dev.shape}")

        cp.copyto(self._st_A, s_t_nu_TLD_in_dev)

        # ---- per-call setup ----
        nLD = L * D
        _k("sigma_psi_LD")(
            ((nLD + 255) // 256,), (256,),
            (sigma_L_dev, s_psi_LD_dev, self._sigma_psi,
             np.int32(L), np.int32(D)),
        )
        x_dot_sl_bits(self.bold_packed, self.row_mean, self.row_inv,
                      self.col_ptr, self.csc_row, self.csc_pidx,
                      s_lambda_P_dev, D, self._X_dot_sl)
        self._flag_acc.fill(0)

        # denom = T * sum_P sl   (slot 1)  and the first kappa_sum (slot 0):
        # both land in ONE D2H below.
        _k("reduce_sum_f32_stage1")(
            (REDUCE_GRID,), (REDUCE_BLOCK,),
            (s_lambda_P_dev, self._part, np.int64(self.P)),
        )
        _k("reduce_stage2")(
            (1,), (REDUCE_BLOCK,),
            (self._part, self._result, np.int32(1), np.int32(REDUCE_GRID)),
        )
        self._launch_kappa_sum(self._st_A)
        self._result.get(out=self._host)
        kappa_sum = float(self._host[0])
        denom = float(self._host[1]) * float(T)

        st_buf = (self._st_A, self._st_B)
        kappa_prev = float(kappa_init)
        kappa_new = kappa_prev
        iter_m = 0
        final_idx = 0

        while True:
            iter_m += 1
            old_idx = (iter_m - 1) % 2
            new_idx = iter_m % 2
            final_idx = new_idx

            rbar = kappa_sum / denom
            kappa_new = invad(self.dim, rbar)
            kappa_f32 = np.float32(kappa_new)

            _k("fused_iter_body")(
                (T * L,), (FUSED_BLOCK,),
                (kappa_f32, self._X_dot_sl, self._sigma_psi,
                 st_buf[old_idx], st_buf[new_idx], self._cos_TL,
                 np.int32(L), np.int32(D)),
                shared_mem=4 * D,
            )
            self._launch_flags()
            # Fold the NEXT iteration's kappa_sum into the same launch
            # sequence so ONE D2H returns both it and the flags.
            self._launch_kappa_sum(st_buf[new_idx])
            self._result.get(out=self._host)          # the single sync

            kappa_sum = float(self._host[0])
            all_flag = all(self._host[2 + t] != 0.0 for t in range(T))
            kappa_drift = abs(kappa_prev - kappa_new) / kappa_prev
            kappa_prev = kappa_new

            if all_flag and kappa_drift < self.epsilon:
                break
            if iter_m > self.max_iter:
                break

        return st_buf[final_idx], kappa_new, iter_m


__all__ = ["compute_row_stats", "x_dot_sl_bits", "MStepGPU"]
