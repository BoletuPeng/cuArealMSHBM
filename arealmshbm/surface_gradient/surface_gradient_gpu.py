"""surface_gradient_gpu.py

CuPy RawKernel port of :mod:`.surface_gradient`. Per-vertex tangent-
plane least-squares gradient magnitude on a mesh-attached scalar field
with K columns.

Algorithm class is identical to the CPU path
(``_hemi_gradient`` in ``surface_gradient.py``): build a 3x3 weighted
normal-equations matrix ``A`` from the 1-ring neighborhood (in tangent
coordinates), then for each column ``k`` solve ``A @ [a, b, _] =
[b0, b1, b2]`` via Cramer's rule, with the documented Workbench
fallback for singular ``A``. Output is the magnitude of the projected
2-D gradient ``sqrt((xhat*a + yhat*b)^2)`` lifted back to 3-D.

Layout
------
Grid:  (n_verts,)   — one block per vertex
Block: (K_thr,)     — K threads per block (256 by default); each thread
                      handles columns ``[tid, tid + K_thr, ...]`` of K.

Precision
---------
fp32 throughout the kernel. The CPU path uses fp64 in the
normal-equations accumulator + Cramer's solve, but the 3x3 ``A`` is
tiny and well-conditioned on the fsa6 midthickness mesh (κ(A) ~ O(10)
based on 1-ring regularity), so fp32 is safe. The downstream consumers
(per-block accumulator + smoothing SpMM) are themselves fp32, so any
sub-1e-4 drift is below the consumer noise floor. ``test_gpu_correctness.py``
gates this as a pytest assertion.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import cupy as cp


# Mirror the CPU constant.
_MAX_DEG = 32


# CUDA C kernel — one block per vertex, K threads per block.
_GRADIENT_KERNEL_SRC = r"""
#define MAX_DEG 32

extern "C" __global__
void hemi_gradient_kernel(
    const int n_verts,
    const int K,
    const float* __restrict__ verts,    // (n_verts, 3)
    const int*   __restrict__ deg,      // (n_verts,)
    const int*   __restrict__ nbors,    // (n_verts, MAX_DEG)  -1 padded
    const float* __restrict__ vn,       // (n_verts, 3)        unit normal
    const float* __restrict__ va,       // (n_verts,)          vertex area
    const float* __restrict__ data,     // (n_verts, K)
    const unsigned char* __restrict__ roi, // (n_verts,) 1 = in ROI
    float* __restrict__ out             // (n_verts, K)        zero-initialized by caller
)
{
    int i = blockIdx.x;
    int tid = threadIdx.x;
    int K_thr = blockDim.x;

    // Shared per-vertex state (computed once by warp 0; broadcast).
    __shared__ int   sm_m;
    __shared__ int   sm_nb[MAX_DEG];
    __shared__ float sm_xmag[MAX_DEG];
    __shared__ float sm_ymag[MAX_DEG];
    __shared__ float sm_ww[MAX_DEG];
    __shared__ float sm_unrollM[MAX_DEG];
    __shared__ float sm_mag2dM[MAX_DEG];
    __shared__ float sm_xhat[3];
    __shared__ float sm_yhat[3];
    __shared__ float sm_A[6];   // a00, a01, a02, a11, a12, a22
    __shared__ float sm_det;

    // NOTE: this early-return is safe ONLY because the launch site below
    // (``_hemi_gradient_gpu``) uses ``grid = (n_verts, 1, 1)`` exactly —
    // every block has ``blockIdx.x < n_verts``, so the condition is
    // either uniformly false (no thread returns) or, if a future change
    // launches with grid.x > n_verts, uniformly true for the over-
    // launched blocks (every thread returns; the per-block sync below
    // is never reached). Do NOT introduce a case where this condition
    // diverges within a block — the ``__syncthreads()`` after the
    // ``if (tid == 0)`` setup would then deadlock the early-return
    // threads. If you need a larger grid, gate work on ``i < n_verts``
    // without returning, or move the early-return after the sync.
    if (i >= n_verts) return;

    if (tid == 0) {
        // ---- Early-exit screening ----
        sm_m = 0;
        int d_full = deg[i];
        unsigned char in_roi = roi[i];
        if (!in_roi || d_full < 2) {
            sm_m = 0;
        } else {
            float nx = vn[i*3 + 0];
            float ny = vn[i*3 + 1];
            float nz = vn[i*3 + 2];
            if (nx == 0.f && ny == 0.f && nz == 0.f) {
                sm_m = 0;
            } else {
                // ---- ROI-filter neighbors ----
                int m = 0;
                for (int kk = 0; kk < d_full; kk++) {
                    int j = nbors[i*MAX_DEG + kk];
                    // Note: nbors uses -1 as padding sentinel; deg
                    // bounds the loop, so all visited entries are valid
                    // vertex indices (in [0, n_verts)).
                    if (j >= 0 && roi[j]) {
                        sm_nb[m] = j;
                        m++;
                    }
                }
                if (m < 2) { sm_m = 0; }
                else {
                    sm_m = m;
                    // ---- Tangent basis (Workbench convention) ----
                    float sx, sy, sz;
                    if (fabsf(nx) > fabsf(ny)) {
                        sx = 0.f; sy = 1.f; sz = 0.f;
                    } else {
                        sx = 1.f; sy = 0.f; sz = 0.f;
                    }
                    // xhat = n x seed
                    float xhx = ny*sz - nz*sy;
                    float xhy = nz*sx - nx*sz;
                    float xhz = nx*sy - ny*sx;
                    float xn = sqrtf(xhx*xhx + xhy*xhy + xhz*xhz);
                    if (xn > 0.f) { float inv = 1.f/xn; xhx*=inv; xhy*=inv; xhz*=inv; }
                    // yhat = n x xhat
                    float yhx = ny*xhz - nz*xhy;
                    float yhy = nz*xhx - nx*xhz;
                    float yhz = nx*xhy - ny*xhx;
                    float yn = sqrtf(yhx*yhx + yhy*yhy + yhz*yhz);
                    if (yn > 0.f) { float inv = 1.f/yn; yhx*=inv; yhy*=inv; yhz*=inv; }
                    sm_xhat[0] = xhx; sm_xhat[1] = xhy; sm_xhat[2] = xhz;
                    sm_yhat[0] = yhx; sm_yhat[1] = yhy; sm_yhat[2] = yhz;

                    float cvx = verts[i*3 + 0];
                    float cvy = verts[i*3 + 1];
                    float cvz = verts[i*3 + 2];

                    // ---- Per-neighbor scratch + A accumulator ----
                    float a00=0.f, a01=0.f, a02=0.f, a11=0.f, a12=0.f, a22=0.f;
                    for (int q = 0; q < m; q++) {
                        int j = sm_nb[q];
                        float ox = verts[j*3 + 0] - cvx;
                        float oy = verts[j*3 + 1] - cvy;
                        float oz = verts[j*3 + 2] - cvz;
                        float origMag = sqrtf(ox*ox + oy*oy + oz*oz);
                        float opp = ox*nx + oy*ny + oz*nz;
                        float unrollMag = origMag;
                        if (origMag > 0.f && fabsf(opp) > 0.035f * origMag) {
                            float r = opp / origMag;
                            if (r > 1.f) r = 1.f;
                            else if (r < -1.f) r = -1.f;
                            unrollMag = origMag * asinf(r) * origMag / opp;
                        }
                        float xm0 = ox*xhx + oy*xhy + oz*xhz;
                        float ym0 = ox*yhx + oy*yhy + oz*yhz;
                        float mag2d = sqrtf(xm0*xm0 + ym0*ym0);
                        float xm, ym;
                        if (mag2d > 0.f) {
                            float s = unrollMag / mag2d;
                            xm = xm0 * s;
                            ym = ym0 * s;
                        } else { xm = 0.f; ym = 0.f; }
                        float wj = va[j];
                        sm_xmag[q] = xm;
                        sm_ymag[q] = ym;
                        sm_ww[q]   = wj;
                        sm_unrollM[q] = unrollMag;
                        sm_mag2dM[q]  = mag2d;
                        a00 += xm*xm*wj;
                        a01 += xm*ym*wj;
                        a02 += xm*wj;
                        a11 += ym*ym*wj;
                        a12 += ym*wj;
                        a22 += wj;
                    }
                    a22 += va[i];
                    sm_A[0]=a00; sm_A[1]=a01; sm_A[2]=a02;
                    sm_A[3]=a11; sm_A[4]=a12; sm_A[5]=a22;
                    // |A| via cofactor expansion along row 0.
                    float det = a00 * (a11*a22 - a12*a12)
                              - a01 * (a01*a22 - a12*a02)
                              + a02 * (a01*a12 - a11*a02);
                    sm_det = det;
                }
            }
        }
    }
    __syncthreads();

    int m = sm_m;
    if (m < 2) return;   // out was zero-initialised by caller — leave as 0

    float a00 = sm_A[0], a01 = sm_A[1], a02 = sm_A[2];
    float a11 = sm_A[3], a12 = sm_A[4], a22 = sm_A[5];
    float det = sm_det;
    float xhx = sm_xhat[0], xhy = sm_xhat[1], xhz = sm_xhat[2];
    float yhx = sm_yhat[0], yhy = sm_yhat[1], yhz = sm_yhat[2];

    for (int kk = tid; kk < K; kk += K_thr) {
        float f_v = data[i*K + kk];
        float b0 = 0.f, b1 = 0.f, b2 = 0.f;
        for (int q = 0; q < m; q++) {
            int j = sm_nb[q];
            float tempf = data[j*K + kk] - f_v;
            float wt = sm_ww[q] * tempf;
            b0 += sm_xmag[q] * wt;
            b1 += sm_ymag[q] * wt;
            b2 += sm_ww[q]  * tempf;
        }
        float sol_a = 0.f, sol_b = 0.f;
        bool ok = false;
        if (det != 0.f) {
            float inv_det = 1.f / det;
            float num_a = b0*(a11*a22 - a12*a12)
                        - a01*(b1*a22 - a12*b2)
                        + a02*(b1*a12 - a11*b2);
            float num_b = a00*(b1*a22 - a12*b2)
                        - b0*(a01*a22 - a12*a02)
                        + a02*(a01*b2 - b1*a02);
            sol_a = num_a * inv_det;
            sol_b = num_b * inv_det;
            ok = isfinite(sol_a) && isfinite(sol_b);
        }
        if (!ok) {
            // Workbench fallback: weighted finite-difference average.
            float tot = 0.f, acc_a = 0.f, acc_b = 0.f;
            for (int q = 0; q < m; q++) {
                int j = sm_nb[q];
                float denom = sm_unrollM[q] * sm_mag2dM[q];
                float wj = sm_ww[q];
                float tempf = data[j*K + kk] - f_v;
                if (denom > 0.f) {
                    float t2 = tempf / denom;
                    acc_a += sm_xmag[q] * t2 * wj;
                    acc_b += sm_ymag[q] * t2 * wj;
                }
                tot += wj;
            }
            if (tot > 0.f) {
                sol_a = acc_a / tot;
                sol_b = acc_b / tot;
            } else {
                sol_a = 0.f; sol_b = 0.f;
            }
        }
        float gx = xhx*sol_a + yhx*sol_b;
        float gy = xhy*sol_a + yhy*sol_b;
        float gz = xhz*sol_a + yhz*sol_b;
        out[i*K + kk] = sqrtf(gx*gx + gy*gy + gz*gz);
    }
}
"""


_GRADIENT_KERNEL = cp.RawKernel(_GRADIENT_KERNEL_SRC, "hemi_gradient_kernel")


GradientMeshGPU = Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]


def prepare_gradient_mesh_gpu(cpu_prep) -> GradientMeshGPU:
    """Push a CPU prepare_gradient_mesh output to device.

    Parameters
    ----------
    cpu_prep : (deg, nbors, vn, va) — the 4-tuple from
        :func:`.surface_gradient.prepare_gradient_mesh`.

    Returns
    -------
    (deg_d, nbors_d, vn_d, va_d, verts_d=None placeholder)

    The verts array is uploaded separately by the pipeline (it's
    already part of Step0Inputs as ``lh_mid_verts``/``rh_mid_verts``
    and the pipeline pushes it to device once at load_inputs).
    """
    deg, nbors, vn, va = cpu_prep
    # Ensure nbors is fixed-width MAX_DEG (the CPU code uses this same
    # constant; -1 is the absent-slot sentinel).
    if nbors.shape[1] != _MAX_DEG:
        raise ValueError(
            f"nbors must be (n_verts, {_MAX_DEG}); got {nbors.shape}")
    deg_d = cp.asarray(deg, dtype=cp.int32)
    nbors_d = cp.asarray(nbors, dtype=cp.int32)
    vn_d = cp.asarray(vn, dtype=cp.float32)
    va_d = cp.asarray(va, dtype=cp.float32)
    return deg_d, nbors_d, vn_d, va_d


def _hemi_gradient_gpu(verts_d, deg_d, nbors_d, vn_d, va_d,
                       data_full_d, roi_d, K_thr: int = 256) -> cp.ndarray:
    """Per-hemisphere kernel launch. Returns (n_verts, K) fp32 device array."""
    n_verts = int(verts_d.shape[0])
    K = int(data_full_d.shape[1])
    out_d = cp.zeros((n_verts, K), dtype=cp.float32)
    # Threads-per-block: 128/256 is a good balance for K=100-150 on
    # most CC 8.x+ cards. We clamp to K (don't bother launching idle
    # threads when K is tiny) but keep a floor of 32 to keep one full
    # warp resident on the SM for the per-vertex setup.
    K_thr = max(32, min(K_thr, K))
    grid = (n_verts, 1, 1)
    block = (K_thr, 1, 1)
    _GRADIENT_KERNEL(
        grid, block,
        (
            np.int32(n_verts),
            np.int32(K),
            verts_d, deg_d, nbors_d, vn_d, va_d,
            data_full_d, roi_d, out_d,
        ),
    )
    return out_d


def cifti_gradient_gpu(data: Union[np.ndarray, cp.ndarray],
                       *,
                       medial_mask: np.ndarray,
                       lh_verts_d: cp.ndarray,
                       lh_mesh_gpu: GradientMeshGPU,
                       rh_verts_d: cp.ndarray,
                       rh_mesh_gpu: GradientMeshGPU,
                       ) -> cp.ndarray:
    """GPU mirror of :func:`.surface_gradient.cifti_gradient`.

    Same signature except mesh inputs are device-resident bundles and
    verts is split per-hemi (the CPU function packs both hemis into a
    single call; the GPU call uses two kernel launches, one per hemi,
    so we accept the splits directly to avoid an extra H2D).

    ``data`` may be either a host ``np.ndarray`` or a device
    ``cp.ndarray``. The production caller hands off an on-device
    ``FC_simi_block`` straight from ``compute_FC_simi_block_gpu``;
    ``cp.asarray`` below is then a no-op alias (no copy, no sync). A
    host array still works — it triggers a one-shot H2D, matching the
    legacy contract used by ``test_gpu_correctness``.

    Returns a device-resident ``cp.ndarray`` (post-PR #56-pattern
    refactor). The production caller's per-iter_a accumulator now lives
    on device too, so the legacy ``cp.asnumpy`` at exit was a 60 MB D2H
    × 18 calls / subject ping-pong — eliminated in favour of one final
    D2H at end of subgraph A (~300 KB of edge_count). Tests that need a
    numpy view call ``cp.asnumpy`` at their boundary.

    Parameters
    ----------
    data : (N_cortex, K) fp32, host or device
        Cortex-only data — same layout as CPU.
    medial_mask : (N_full,) bool host
        ``True`` = medial wall.
    lh_verts_d, rh_verts_d : (N_lh, 3) / (N_rh, 3) fp32 device
        Cached midthickness vertex coordinates.
    lh_mesh_gpu, rh_mesh_gpu : GradientMeshGPU
        4-tuples from :func:`prepare_gradient_mesh_gpu`, one per hemi.

    Returns
    -------
    grads : (N_cortex, K) device fp32  — same shape + dtype as CPU.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (N_cortex, K), got {data.shape}")
    n_lh = int(lh_verts_d.shape[0])
    n_rh = int(rh_verts_d.shape[0])
    n_full = n_lh + n_rh
    medial = np.asarray(medial_mask).reshape(-1)
    if medial.shape[0] != n_full:
        raise ValueError(
            f"medial_mask length {medial.shape[0]} != lh+rh verts {n_full}")
    cortex_bool = ~medial.astype(bool)
    n_cortex = int(cortex_bool.sum())
    if data.shape[0] != n_cortex:
        raise ValueError(
            f"data has {data.shape[0]} rows; expected N_cortex={n_cortex}")
    K = int(data.shape[1])

    # H2D (if data is host) + scatter to full layout on device. When
    # the caller passes a device cp.ndarray fp32 (production path),
    # cp.asarray is an alias — no copy, no stream sync.
    data_d = cp.asarray(data, dtype=cp.float32)
    cortex_bool_d = cp.asarray(cortex_bool)
    full_d = cp.zeros((n_full, K), dtype=cp.float32)
    full_d[cortex_bool_d] = data_d

    lh_full_d = cp.ascontiguousarray(full_d[:n_lh])
    rh_full_d = cp.ascontiguousarray(full_d[n_lh:])
    lh_roi_d = cp.ascontiguousarray(cortex_bool_d[:n_lh].astype(cp.uint8))
    rh_roi_d = cp.ascontiguousarray(cortex_bool_d[n_lh:].astype(cp.uint8))

    lh_grads_d = _hemi_gradient_gpu(
        lh_verts_d, *lh_mesh_gpu, lh_full_d, lh_roi_d)
    rh_grads_d = _hemi_gradient_gpu(
        rh_verts_d, *rh_mesh_gpu, rh_full_d, rh_roi_d)

    # Concatenate and gather cortex rows back to (N_cortex, K). Output
    # stays on device — the production caller's per-iter_a accumulator
    # consumes it directly; tests cp.asnumpy at their boundary.
    full_out_d = cp.concatenate([lh_grads_d, rh_grads_d], axis=0)
    cortex_out_d = full_out_d[cortex_bool_d]
    return cortex_out_d
