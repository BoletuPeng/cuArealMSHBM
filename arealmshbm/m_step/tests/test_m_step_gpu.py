"""test_m_step_gpu.py

Validation of the ``gpu`` M-step
(:mod:`arealmshbm.m_step.m_step_gpu`) against the CPU reference
:class:`arealmshbm.m_step.m_step.MStepSession`.

Real-data checks run on sub-001 (fsaverage6, T=6, N=81924, D=1175,
L=300, P=251502) via the shared fixture; they skip when the profile
store or cupy is missing. The synthetic case is self-contained and only
needs cupy.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arealmshbm.m_step.m_step import MStepSession
from arealmshbm.vmf_clustering.tests._sub001_fixture import (
    skip_unless_cupy, skip_unless_sub001,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _csc_from_dense_support(mask_NL: np.ndarray):
    """Hand-built ``(col_ptr, csc_row, csc_pidx)`` for a dense support.

    Mirrors :func:`arealmshbm.vmf_clustering.tests._layout_oracle.build_candidate_layout_dense`:
    the ``(P,)`` CSR order is the row-major nonzero order of the support,
    and the CSC arrays are that order sorted by ``(col, n)``.
    """
    rows, cols = np.nonzero(mask_NL)              # row-major -> CSR order
    L = mask_NL.shape[1]
    order = np.lexsort((rows, cols)).astype(np.int32)
    col_ptr = np.zeros(L + 1, dtype=np.int64)
    np.cumsum(np.bincount(cols, minlength=L), out=col_ptr[1:])
    return (col_ptr.astype(np.int32),
            np.ascontiguousarray(rows[order], dtype=np.int32),
            np.ascontiguousarray(order, dtype=np.int32),
            rows.astype(np.int64), cols.astype(np.int64))


def _gather_P(x_NL, rows, cols):
    return np.ascontiguousarray(x_NL[rows, cols], dtype=np.float32)


# ---------------------------------------------------------------------
# module-scope sub-001 reference (built once: the host BOLD is 2.3 GB)
# ---------------------------------------------------------------------
class _Ref:
    pass


_REF = None


def _sub001_ref():
    """Fixture + host fp32 BOLD + CPU MStepSession + device inputs."""
    global _REF
    if _REF is not None:
        return _REF
    import cupy as cp
    from arealmshbm.data_io.bitpacked_norm import (
        unpack_normalize_packed_NTD_host,
    )
    from arealmshbm.vmf_clustering.sparse_layout import layout_to_device
    from arealmshbm.m_step import m_step_gpu as G

    fx = skip_unless_sub001()
    r = _Ref()
    r.fx = fx
    r.T, r.N, r.Db = fx.packed_TND.shape
    r.D = int(fx.D)
    r.L = int(fx.layout.L)
    r.dim = float(fx.setting_params["dim"])
    r.eps = 1e-4
    r.max_iter = 50

    # Host reference BOLD (N, T, D) fp32.
    r.ds_NTD = unpack_normalize_packed_NTD_host(
        np.ascontiguousarray(np.transpose(fx.packed_TND, (1, 0, 2))), r.D)
    r.session = MStepSession(r.ds_NTD, r.dim, r.L, r.T,
                             epsilon=r.eps, max_iter=r.max_iter)

    # Inputs.
    r.s_lambda = np.ascontiguousarray(fx.Params["theta"], dtype=np.float32)
    r.s_lambda_P = fx.layout.gather(r.s_lambda)
    r.s_psi = np.ascontiguousarray(fx.Params["mu"], dtype=np.float32)   # (D, L)
    r.sigma = np.ascontiguousarray(
        np.asarray(fx.Params["sigma"]).ravel(), dtype=np.float32)
    r.s_t_nu_DLT = np.ascontiguousarray(
        np.repeat(r.s_psi[:, :, None], r.T, axis=2))
    r.kappa_init = np.full(r.L, float(fx.ini_val), dtype=np.float64)

    # Device state.
    r.packed_dev = cp.asarray(np.ascontiguousarray(fx.packed_TND))
    r.row_mean_dev, r.row_inv_dev = G.compute_row_stats(r.packed_dev, r.D)
    r.layout_dev = layout_to_device(fx.layout)
    r.s_lambda_P_dev = cp.asarray(r.s_lambda_P)
    r.s_psi_LD_dev = cp.asarray(np.ascontiguousarray(r.s_psi.T))     # (L, D)
    r.sigma_dev = cp.asarray(r.sigma)
    r.s_t_nu_TLD_dev = cp.asarray(
        np.ascontiguousarray(np.transpose(r.s_t_nu_DLT, (2, 1, 0))))
    r.mstep = G.MStepGPU(r.T, r.D, r.L, r.dim, r.eps, r.max_iter,
                         r.packed_dev, r.row_mean_dev, r.row_inv_dev,
                         r.layout_dev)
    _REF = r
    return r


# ---------------------------------------------------------------------
# 1. row statistics
# ---------------------------------------------------------------------
def test_row_stats_vs_host():
    skip_unless_cupy()
    import cupy as cp
    r = _sub001_ref()

    packed = r.fx.packed_TND                       # (T, N, Db)
    bits_pop = np.zeros((r.T, r.N), dtype=np.int64)
    for b in range(r.Db):
        col = packed[:, :, b].astype(np.uint32)
        for k in range(8):
            bits_pop += ((col >> k) & 1).astype(np.int64)

    pop_d = bits_pop.astype(np.float64)
    mean_ref = (pop_d / float(r.D)).astype(np.float32)
    post = pop_d - float(r.D) * (pop_d / float(r.D)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ref = (1.0 / np.sqrt(post)).astype(np.float32)
    zero_rows = (bits_pop == 0) | (bits_pop == r.D)
    inv_ref = np.where(zero_rows, np.float32(0.0), inv_ref).astype(np.float32)

    mean_gpu = cp.asnumpy(r.row_mean_dev)
    inv_gpu = cp.asnumpy(r.row_inv_dev)
    assert np.array_equal(mean_gpu, mean_ref), "row_mean must be bit-exact"

    # ULP distance on row_inv.
    a = mean_gpu  # keep flake quiet
    ulp = np.abs(inv_gpu.view(np.int32).astype(np.int64)
                 - inv_ref.view(np.int32).astype(np.int64))
    print(f"\n[row_stats] rows={r.T * r.N} zero/const rows={int(zero_rows.sum())} "
          f"row_inv max ULP diff={int(ulp.max())}")
    assert int(ulp.max()) <= 1

    # Cross-check against the actual host fp32 rows: row_mean is the value
    # subtracted, row_inv the scale applied (checked on a random sample of
    # non-degenerate rows).
    rng = np.random.default_rng(0)
    ok = np.flatnonzero(~zero_rows[0])
    sample = rng.choice(ok, size=64, replace=False)
    for n in sample:
        bits = np.zeros(r.D, dtype=np.float32)
        for d in range(r.D):
            bits[d] = (packed[0, n, d >> 3] >> (d & 7)) & 1
        ref_row = r.ds_NTD[n, 0, :]
        recon = (bits - mean_gpu[0, n]) * inv_gpu[0, n]
        assert np.allclose(recon, ref_row, rtol=0, atol=1e-7), n
    del a


# ---------------------------------------------------------------------
# 2. x_dot_sl_bits vs the CPU sgemm
# ---------------------------------------------------------------------
def test_x_dot_sl_bits_vs_sgemm():
    skip_unless_cupy()
    import cupy as cp
    from arealmshbm.m_step import m_step_gpu as G
    r = _sub001_ref()

    out = cp.empty((r.T, r.L, r.D), dtype=cp.float32)
    G.x_dot_sl_bits(r.packed_dev, r.row_mean_dev, r.row_inv_dev,
                    r.layout_dev["col_ptr"], r.layout_dev["csc_row"],
                    r.layout_dev["csc_pidx"], r.s_lambda_P_dev, r.D, out)
    got = cp.asnumpy(out)

    max_abs = 0.0
    max_rel = 0.0
    max_scale_rel = 0.0
    for t in range(r.T):
        ref = (r.ds_NTD[:, t, :].T @ r.s_lambda).astype(np.float32)  # (D, L)
        g = got[t].T                                                # (D, L)
        d = np.abs(g.astype(np.float64) - ref.astype(np.float64))
        max_abs = max(max_abs, float(d.max()))
        scale = float(np.abs(ref).max())
        max_scale_rel = max(max_scale_rel, float(d.max()) / scale)
        m = np.abs(ref) > 1e-3
        if m.any():
            max_rel = max(max_rel,
                          float((d[m] / np.abs(ref[m].astype(np.float64))).max()))
    print(f"\n[x_dot_sl_bits] max_abs={max_abs:.3e} "
          f"max_rel(|ref|>1e-3)={max_rel:.3e} "
          f"max_abs/max|ref|={max_scale_rel:.3e}")

    # The fp32 sgemm is NOT ground truth: it accumulates 81924 fp32 terms
    # per output cell, while the bit kernel folds <=64-member fp32 partials
    # into an fp64 running sum. Anchor both against an fp64 matmul at t=0.
    Xt = r.ds_NTD[:, 0, :]
    ref64 = Xt.T.astype(np.float64) @ r.s_lambda.astype(np.float64)
    ref32 = (Xt.T @ r.s_lambda).astype(np.float32)
    e_sgemm = np.abs(ref32.astype(np.float64) - ref64)
    e_gpu = np.abs(got[0].T.astype(np.float64) - ref64)
    print(f"[x_dot_sl_bits] vs fp64 truth (t=0, scale={np.abs(ref64).max():.3f}): "
          f"sgemm max={e_sgemm.max():.3e} rms={np.sqrt((e_sgemm**2).mean()):.3e} | "
          f"gpu max={e_gpu.max():.3e} rms={np.sqrt((e_gpu**2).mean()):.3e}")

    # Contract: agreement at the SCALE of the output (the elementwise
    # |ref|>1e-3 relative metric is dominated by near-zero cells where the
    # fp32 sgemm reference itself carries the larger absolute error).
    assert max_scale_rel <= 1e-6
    assert e_gpu.max() <= e_sgemm.max()
    assert np.sqrt((e_gpu ** 2).mean()) <= np.sqrt((e_sgemm ** 2).mean())


# ---------------------------------------------------------------------
# 2b. x_dot_sl_bits vs the shared-memory-staged oracle
# ---------------------------------------------------------------------
# The oracle stages each member's packed row in shared memory and folds
# one member per step. The shipped text (``m_step/_xdot_kernel.py``)
# reads the member bytes straight from global memory with a 4-deep
# ``__ldg`` prefetch instead, keeping every per-``d`` fp32 fold and every
# fp32->fp64 fold in the same order, so the two must agree to the bit.
# The staged text lives here as the oracle so that a later edit of the
# shared kernel cannot drift silently.
#
# The text has two consumers and the oracle is instantiated at each one's
# geometry -- step 3 at 128 threads x 2 bytes (``-std=c++14``), step 2 at
# 256 x 1 (``-std=c++17 -fmad=false``) -- and compared against that
# consumer's OWN compiled kernel, launched the way the consumer launches
# it. The fp64 B-term tree spans ``blockDim.x`` lanes, so only the same
# geometry is bit-comparable. The geometry and the 64-member fold chunk
# are written out here, not read from the modules: the labels' bit-exact
# contract against the merge-base rests on that op order, so a changed
# module constant must fail here (``test_x_dot_geometry_is_pinned``), not
# re-tune the oracle.
_STAGED_ORACLE_SRC = r"""
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
"""


# (threads, bytes per thread, fold chunk, nvrtc options) the oracle is
# instantiated at for each consumer.
_ORACLE_GEOMETRY = {
    "step3": (128, 2, 64, ("-std=c++14",)),
    "step2": (256, 1, 64, ("-std=c++17", "-fmad=false")),
}
_CONSUMERS = tuple(_ORACLE_GEOMETRY)


def _staged_oracle(consumer):
    import cupy as cp
    bs, nb, _, opts = _ORACLE_GEOMETRY[consumer]
    src = (_STAGED_ORACLE_SRC
           .replace("XDOT_MAX_NB_VALUE", str(nb))
           .replace("XDOT_BS_VALUE", str(bs)))
    return cp.RawModule(code=src, options=opts).get_function("x_dot_sl_bits")


def _shipped(consumer):
    """The consumer's own compiled ``x_dot_sl_bits``: step 3 through its
    wrapper, step 2 with ``Step2SparseSession.run_iter``'s arguments and
    geometry (a copy of that launch; the constants are pinned by
    ``test_x_dot_geometry_is_pinned``)."""
    if consumer == "step3":
        from arealmshbm.m_step import m_step_gpu as G

        def run(packed, rm, ri, lay, sl, D, out):
            G.x_dot_sl_bits(packed, rm, ri, lay["col_ptr"], lay["csc_row"],
                            lay["csc_pidx"], sl, D, out)
        return run
    from arealmshbm.step2_em_iter_master import _kernels_gpu as K
    kern = K.module().get_function("x_dot_sl_bits")

    def run(packed, rm, ri, lay, sl, D, out):
        T, N, Db = packed.shape
        L = int(lay["col_ptr"].size) - 1
        kern((T * L,), (K.XDOT_BLOCK,),
             (packed, rm, ri, lay["col_ptr"], lay["csc_row"], lay["csc_pidx"],
              sl, out, np.int32(N), np.int32(L), np.int32(D), np.int32(Db),
              np.int32(K.XDOT_CHUNK)),
             shared_mem=K.XDOT_CHUNK * 8)
    return run


def _assert_bit_exact_vs_staged(consumer, packed_dev, D, lay, sl_dev):
    """Consumer's shipped kernel (twice) == staged oracle at the
    consumer's geometry, uint32-exact."""
    import cupy as cp
    from arealmshbm.m_step import m_step_gpu as G
    T, N, Db = packed_dev.shape
    L = int(lay["col_ptr"].size) - 1
    bs, _, chunk, _ = _ORACLE_GEOMETRY[consumer]
    rm, ri = G.compute_row_stats(packed_dev, D)
    got = cp.empty((T, L, D), dtype=cp.float32)
    got2 = cp.empty_like(got)
    ref = cp.empty_like(got)
    run = _shipped(consumer)
    run(packed_dev, rm, ri, lay, sl_dev, D, got)
    run(packed_dev, rm, ri, lay, sl_dev, D, got2)
    _staged_oracle(consumer)(
        (T * L,), (bs,),
        (packed_dev, rm, ri, lay["col_ptr"], lay["csc_row"], lay["csc_pidx"],
         sl_dev, ref, np.int32(N), np.int32(L), np.int32(D), np.int32(Db),
         np.int32(chunk)),
        shared_mem=chunk * 4 + chunk * Db,   # the staged layout
    )
    g, g2, r = (cp.asnumpy(x).view(np.uint32) for x in (got, got2, ref))
    assert np.array_equal(g, g2), f"{consumer}: x_dot_sl_bits is not deterministic"
    assert np.array_equal(g, r), (
        f"{consumer}: x_dot_sl_bits differs from the staged oracle in "
        f"{int((g != r).sum())} of {g.size} cells")


def test_x_dot_geometry_is_pinned():
    """Each consumer launches at the geometry its oracle is written at;
    ``XDOT_CHUNK`` fixes the fp32->fp64 fold order the labels contract
    rests on. Retuning one means retuning the oracle on purpose."""
    skip_unless_cupy()
    from arealmshbm.m_step import m_step_gpu as G
    from arealmshbm.step2_em_iter_master import _kernels_gpu as K
    assert (G.XDOT_BLOCK, G.XDOT_MAX_NB, G.XDOT_CHUNK) == _ORACLE_GEOMETRY["step3"][:3]
    assert (K.XDOT_BLOCK, K.XDOT_MAX_NB, K.XDOT_CHUNK) == _ORACLE_GEOMETRY["step2"][:3]


def test_x_dot_src_is_ascii():
    """On a kernel-cache miss CuPy writes each ``RawModule`` source to a
    ``.cu`` file in the locale codec, so the shared text must stay ASCII
    for both consumers (step 3's M-step and step 2's ``module()``)."""
    from arealmshbm.m_step._xdot_kernel import XDOT_SL_BITS_SRC
    bad = [i for i, ch in enumerate(XDOT_SL_BITS_SRC) if ord(ch) > 127]
    assert not bad, f"non-ASCII at offsets {bad[:5]}"


@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_x_dot_sl_bits_bit_exact_vs_staged_synthetic(consumer):
    """Db = 200 > 128, so step 3's second byte per thread runs; D is not
    a multiple of 8, so the ``d < D`` store guard runs; columns hold ~390
    members (several 64-member folds plus a tail that is not a multiple of
    the 4-deep prefetch); s_lambda spans 1e-30..1."""
    skip_unless_cupy()
    import cupy as cp
    N, T, D, L = 3000, 2, 1597, 7
    rng = np.random.default_rng(5)
    bits = rng.random((T, N, D)) < 0.1
    bits[:, :3, :] = False                  # pop == 0 rows -> row_inv = 0
    bits[:, 3:5, :] = True                  # pop == D rows -> row_inv = 0
    packed = np.packbits(bits, axis=2, bitorder="little")
    mask = rng.random((N, L)) < 0.13
    col_ptr, csc_row, csc_pidx, rows, _ = _csc_from_dense_support(mask)
    lay = {"col_ptr": cp.asarray(col_ptr), "csc_row": cp.asarray(csc_row),
           "csc_pidx": cp.asarray(csc_pidx)}
    sl = cp.asarray((10.0 ** rng.uniform(-30, 0, size=rows.size)).astype(np.float32))
    _assert_bit_exact_vs_staged(consumer, cp.asarray(packed), D, lay, sl)


@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_x_dot_sl_bits_bit_exact_vs_staged_sub001(consumer):
    skip_unless_cupy()
    import cupy as cp
    r = _sub001_ref()
    _assert_bit_exact_vs_staged(consumer, r.packed_dev, r.D, r.layout_dev,
                                r.s_lambda_P_dev)
    rng = np.random.default_rng(7)
    wide = cp.asarray((10.0 ** rng.uniform(-30, 0, size=r.s_lambda_P.size)).astype(np.float32))
    _assert_bit_exact_vs_staged(consumer, r.packed_dev, r.D, r.layout_dev, wide)


@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_x_dot_sl_bits_bit_exact_vs_staged_cancellation(consumer):
    """Every cell is ``A_d - B = 0`` in exact arithmetic, so what the
    kernel stores is the summation residue -- the one input on which the
    order of the fp64 B term is visible (on the synthetic and sub-001
    fixtures a reversed B loop is invisible after the fp32 store). Rows
    come in complementary pairs with exactly D/2 ones each, so
    ``row_mean = 0.5`` and ``row_inv = 1/16`` are exact and every B term
    is exact; both members of a pair share the weight; the head pair has
    ``w = 1`` and the rest ``2^U(-44, -38)``, so every add into the partials
    that carry the head pair rounds at ulp(0.5) (the tail-only lanes sum
    exactly) and the rounding walk depends on which tail members share a
    lane or tree level with the head pair."""
    skip_unless_cupy()
    import cupy as cp
    npairs, T, D, L = 20000, 2, 1024, 16
    rng = np.random.default_rng(12)
    half = np.zeros((T, npairs, D), dtype=bool)
    half[..., : D // 2] = True
    half = rng.permuted(half, axis=2)           # exactly D/2 ones per row
    bits = np.empty((T, 2 * npairs, D), dtype=bool)
    bits[:, 0::2] = half
    bits[:, 1::2] = ~half                       # the complement row
    packed = np.packbits(bits, axis=2, bitorder="little")
    mask = np.repeat(rng.random((npairs, L)) < 0.5, 2, axis=0)   # a pair is in or out
    col_ptr, csc_row, csc_pidx, rows, _ = _csc_from_dense_support(mask)
    sl = np.empty(rows.size, dtype=np.float32)
    for l in range(L):
        pos = csc_pidx[col_ptr[l]:col_ptr[l + 1]]       # (n ascending) = pairs adjacent
        w = np.exp2(rng.uniform(-44, -38, size=pos.size // 2))
        w[0] = 1.0
        sl[pos] = np.repeat((w * 16.0).astype(np.float32), 2)   # w = sl * row_inv
    lay = {"col_ptr": cp.asarray(col_ptr), "csc_row": cp.asarray(csc_row),
           "csc_pidx": cp.asarray(csc_pidx)}
    _assert_bit_exact_vs_staged(consumer, cp.asarray(packed), D, lay, cp.asarray(sl))


# ---------------------------------------------------------------------
# 3. full run vs MStepSession.run
# ---------------------------------------------------------------------
def test_full_run_vs_cpu():
    skip_unless_cupy()
    import cupy as cp
    r = _sub001_ref()

    st_cpu, kappa_cpu, iter_cpu = r.session.run(
        r.s_t_nu_DLT, r.s_lambda, r.s_psi, r.sigma, r.kappa_init)
    st_gpu_dev, kappa_gpu, iter_gpu = r.mstep.run(
        r.s_t_nu_TLD_dev, r.s_lambda_P_dev, r.s_psi_LD_dev,
        r.sigma_dev, float(r.fx.ini_val))
    st_gpu = np.transpose(cp.asnumpy(st_gpu_dev), (2, 1, 0))   # -> (D, L, T)

    k_cpu = float(kappa_cpu[0])
    rel_k = abs(k_cpu - kappa_gpu) / abs(k_cpu)
    diff = np.abs(st_gpu.astype(np.float64) - st_cpu.astype(np.float64))
    print(f"\n[full run] iter_m cpu={iter_cpu} gpu={iter_gpu}\n"
          f"           kappa cpu={k_cpu!r} gpu={kappa_gpu!r} rel={rel_k:.3e}\n"
          f"           s_t_nu max|diff|={float(diff.max()):.3e} "
          f"nan cpu={int(np.isnan(st_cpu).sum())} "
          f"gpu={int(np.isnan(st_gpu).sum())}")
    assert iter_cpu == iter_gpu
    assert rel_k <= 1e-6
    assert np.array_equal(np.isnan(st_cpu), np.isnan(st_gpu))
    assert float(np.nanmax(diff)) <= 1e-5

    # Convergence flags derived from cos must agree with the CPU latch.
    cos_gpu = cp.asnumpy(r.mstep._cos_TL)
    flags_gpu = cp.asnumpy(r.mstep._flag_acc)
    print(f"           final-iter cos: min={float(np.nanmin(cos_gpu)):.9f} "
          f"max(1-cos)={float(np.nanmax(1.0 - cos_gpu)):.3e} "
          f"flags={flags_gpu.tolist()} cpu_flags="
          f"{r.session._flag_T_acc.tolist()}")
    assert np.array_equal(flags_gpu.astype(np.int8),
                          r.session._flag_T_acc.astype(np.int8))


# ---------------------------------------------------------------------
# 4. determinism
# ---------------------------------------------------------------------
def test_determinism():
    skip_unless_cupy()
    import cupy as cp
    r = _sub001_ref()

    a_dev, ka, ia = r.mstep.run(r.s_t_nu_TLD_dev, r.s_lambda_P_dev,
                                r.s_psi_LD_dev, r.sigma_dev,
                                float(r.fx.ini_val))
    a = a_dev.copy()
    b_dev, kb, ib = r.mstep.run(r.s_t_nu_TLD_dev, r.s_lambda_P_dev,
                                r.s_psi_LD_dev, r.sigma_dev,
                                float(r.fx.ini_val))
    assert (ia, ka) == (ib, kb)
    # bitwise (stronger than array_equal; also pins NaN payloads)
    assert bool(cp.all(a.view(cp.uint32) == b_dev.view(cp.uint32)))
    assert bool(cp.array_equal(a, b_dev))


# ---------------------------------------------------------------------
# 5. synthetic tiny case (empty parcel, pop == 0 and pop == D rows)
# ---------------------------------------------------------------------
def _synthetic(with_empty_parcel: bool):
    import cupy as cp
    from arealmshbm.data_io.bitpacked_norm import (
        unpack_normalize_packed_NTD_host,
    )
    from arealmshbm.m_step import m_step_gpu as G

    rng = np.random.default_rng(7)
    N, T, D, L = 40, 2, 20, 4
    bits = (rng.random((T, N, D)) < 0.35).astype(np.uint8)
    bits[:, 0, :] = 0        # pop == 0
    bits[:, 1, :] = 1        # pop == D
    packed_TND = np.ascontiguousarray(
        np.packbits(bits, axis=-1, bitorder="little"))

    # Candidate support: a few members per parcel; parcel L-1 optionally empty.
    mask = np.zeros((N, L), dtype=bool)
    for n in range(N):
        ls = rng.choice(L - 1 if with_empty_parcel else L,
                        size=2, replace=False)
        mask[n, ls] = True
    if not with_empty_parcel:
        mask[0, L - 1] = True

    col_ptr, csc_row, csc_pidx, rows, cols = _csc_from_dense_support(mask)

    s_lambda = np.zeros((N, L), dtype=np.float32)
    s_lambda[mask] = rng.random(int(mask.sum())).astype(np.float32) + 0.1
    s_lambda_P = _gather_P(s_lambda, rows, cols)

    s_psi = rng.standard_normal((D, L)).astype(np.float32)
    s_psi /= np.linalg.norm(s_psi, axis=0, keepdims=True)
    if with_empty_parcel:
        s_psi[:, L - 1] = 0.0                       # -> cn == 0 -> NaN
    sigma = (rng.random(L).astype(np.float32) + 0.5)
    s_t_nu_DLT = np.ascontiguousarray(np.repeat(s_psi[:, :, None], T, axis=2))

    ds_NTD = unpack_normalize_packed_NTD_host(
        np.ascontiguousarray(np.transpose(packed_TND, (1, 0, 2))), D)
    dim, eps, max_iter = float(D - 1), 1e-4, 12
    ses = MStepSession(ds_NTD, dim, L, T, epsilon=eps, max_iter=max_iter)
    kappa_init = np.full(L, 12.0, dtype=np.float64)
    st_cpu, kappa_cpu, iter_cpu = ses.run(
        s_t_nu_DLT, s_lambda, s_psi, sigma, kappa_init)

    packed_dev = cp.asarray(packed_TND)
    rm, ri = G.compute_row_stats(packed_dev, D)
    layout_dev = {
        "col_ptr": cp.asarray(col_ptr), "csc_row": cp.asarray(csc_row),
        "csc_pidx": cp.asarray(csc_pidx), "P": int(csc_row.size),
    }
    ms = G.MStepGPU(T, D, L, dim, eps, max_iter, packed_dev, rm, ri,
                    layout_dev)
    st_dev, kappa_gpu, iter_gpu = ms.run(
        cp.asarray(np.ascontiguousarray(np.transpose(s_t_nu_DLT, (2, 1, 0)))),
        cp.asarray(s_lambda_P),
        cp.asarray(np.ascontiguousarray(s_psi.T)),
        cp.asarray(sigma), 12.0)
    st_gpu = np.transpose(cp.asnumpy(st_dev), (2, 1, 0))
    return (st_cpu, float(kappa_cpu[0]), iter_cpu,
            st_gpu, float(kappa_gpu), iter_gpu)


def test_synthetic_with_empty_parcel():
    skip_unless_cupy()
    st_cpu, k_cpu, i_cpu, st_gpu, k_gpu, i_gpu = _synthetic(True)
    print(f"\n[synthetic empty] iter cpu={i_cpu} gpu={i_gpu} "
          f"kappa cpu={k_cpu!r} gpu={k_gpu!r} "
          f"nan cpu={int(np.isnan(st_cpu).sum())} "
          f"gpu={int(np.isnan(st_gpu).sum())}")
    assert i_cpu == i_gpu
    assert np.array_equal(np.isnan(st_cpu), np.isnan(st_gpu))
    assert np.isnan(st_cpu).any(), "empty parcel must produce NaN"
    assert (np.isnan(k_cpu) and np.isnan(k_gpu)) or \
        abs(k_cpu - k_gpu) / abs(k_cpu) <= 1e-6


def test_synthetic_no_empty_parcel():
    skip_unless_cupy()
    st_cpu, k_cpu, i_cpu, st_gpu, k_gpu, i_gpu = _synthetic(False)
    d = np.abs(st_gpu.astype(np.float64) - st_cpu.astype(np.float64))
    print(f"\n[synthetic dense] iter cpu={i_cpu} gpu={i_gpu} "
          f"kappa rel={abs(k_cpu - k_gpu) / abs(k_cpu):.3e} "
          f"max|diff|={float(d.max()):.3e}")
    assert i_cpu == i_gpu
    assert not np.isnan(st_cpu).any() and not np.isnan(st_gpu).any()
    assert abs(k_cpu - k_gpu) / abs(k_cpu) <= 1e-6
    assert float(d.max()) <= 1e-5


# ---------------------------------------------------------------------
# 6. timing
# ---------------------------------------------------------------------
def test_timing_report():
    skip_unless_cupy()
    import cupy as cp
    from arealmshbm.m_step import m_step_gpu as G
    r = _sub001_ref()

    out = cp.empty((r.T, r.L, r.D), dtype=cp.float32)

    def _xdot():
        G.x_dot_sl_bits(r.packed_dev, r.row_mean_dev, r.row_inv_dev,
                        r.layout_dev["col_ptr"], r.layout_dev["csc_row"],
                        r.layout_dev["csc_pidx"], r.s_lambda_P_dev, r.D, out)

    for _ in range(3):
        _xdot()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        _xdot()
    cp.cuda.runtime.deviceSynchronize()
    t_xdot = (time.perf_counter() - t0) / 5 * 1e3

    args = (r.s_t_nu_TLD_dev, r.s_lambda_P_dev, r.s_psi_LD_dev,
            r.sigma_dev, float(r.fx.ini_val))
    _, _, iter_m = r.mstep.run(*args)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        r.mstep.run(*args)
    cp.cuda.runtime.deviceSynchronize()
    t_run = (time.perf_counter() - t0) / 5 * 1e3

    # CPU reference wall for context.
    t0 = time.perf_counter()
    r.session.run(r.s_t_nu_DLT, r.s_lambda, r.s_psi, r.sigma, r.kappa_init)
    t_cpu = (time.perf_counter() - t0) * 1e3

    print(f"\n[timing sub-001] x_dot_sl_bits={t_xdot:.3f} ms | "
          f"full run={t_run:.3f} ms (iter_m={iter_m}, "
          f"{(t_run - t_xdot) / max(iter_m, 1):.3f} ms/iter_m) | "
          f"CPU MStepSession.run={t_cpu:.1f} ms")


if __name__ == "__main__":   # pragma: no cover
    pytest.main([__file__, "-q", "-s"])
