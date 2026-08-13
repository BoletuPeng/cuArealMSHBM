"""vmf_clustering_gpu.py

Full-device GPU port of :class:`VmfClusteringSession` (the ``backend=
'gpu_full'`` path). All state device-resident across the entire EM loop:
BOLD, every (N, L) ping-pong buffer, M-step buffers, and per-L scratch
are pre-allocated at __init__. A handful of short-lived (N, L)
intermediates inside the EM-stop step are allocated per call and freed
back to CuPy's pool.

GPU stages: M-step (cuBLAS sgemms + CuPy primitives; ``invad`` on CPU,
~50 µs), E-step lambda loop, spatial_connect_prior, spatial_xyz_prior,
EM stop criterion.

CPU stages:
    * check_connectedness (BFS on labels — GPU-hostile). H↔D boundary
      per comp_iter is a (N,) int64 ``labels`` (~640 KB) plus a (L,)
      fp64 ``xyz_gamma`` update (2.4 KB).
    * ``Cdln(kappa, dim)`` and ``Cdln(xyz_gamma, 3)`` (L=300 fp64 vector
      + scipy Bessel, ~1 ms each).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import numpy as np

from arealmshbm.V_lambda import Session as VLambdaSession
from arealmshbm.spatial_priors._cdln import cdln_d3_to_f32
from arealmshbm.spatial_priors.spatial_xyz import compute_unit_sphere_xyz
from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32
from arealmshbm.m_step._invad import invad
from arealmshbm.check_connectedness.component_distance import (
    component_distance,
    compute_components_general,
)
from arealmshbm.postprocessing import remove_isolated_surface_components
from arealmshbm.step3_pipeline.variant import VariantSpec

from . import _kernels_gpu
from ._session_common import (
    _stage_1d, _stage_2d, _stage_3d,
    validate_packed_bold_shape,
    validate_variant_requirements,
)
from arealmshbm.em_stop_criterion._kernels import LOG_EPS_POW20


# The kernel's __shared__ s_sum / s_sumsq arrays are statically sized
# to this constant; bumping the wrapper constant past 256 without also
# bumping the kernel arrays would silently OOB the shared-mem write.
_NORMALIZE_BOLD_BLOCK = 256
_NORMALIZE_BOLD_BLOCK_MAX = 256   # static shared-mem ceiling — see kernel.


# Bit-packed -> fp32 demean + L2-norm. The device counterpart of the
# host numba kernel
# :func:`arealmshbm.data_io.bitpacked_norm._normalize_session_bitpacked_numba`:
# the two backends ingest the same on-disk packed bytes and produce
# numerically equivalent ``(N, T, D)`` fp32 BOLD (host = pass-2 fp64
# ``sumsq += (bit - mean)^2`` accumulator; device = fp64 closed-form
# ``sum - D * mean^2`` per-row). Both reduce to the same algebraic
# value and agree after the fp32 cast at the precision the downstream
# cuBLAS/MKL sgemms operate on. The legacy fp32-input device kernel
# (``_normalize_bold_kernel`` + ``_normalize_bold_NTD_inplace``) was
# removed when bitpacked became the only production format
# (commit 97742cd, 2026-06).
#
# Convention: input cells are 0/1; cell d <=> bit (d & 7) of byte (d >> 3),
# matching ``numpy.packbits(bitorder='little')``. Padding bits in the last
# byte (d >= D) MUST be zero; the step1 / step2 writers guarantee this.
#
# For binary input the per-row normalize algebra simplifies: sumsq == sum
# (v in {0, 1} implies v*v == v), so a single integer popcount pass
# replaces separate fp64 sum/sumsq reductions, and the post-mean form
# ``sum((v - mean)^2) = sum - D * mean^2`` is closed-form (one fp64 fma).
# Both backends use this identity; the host kernel keeps the serial
# fp64 accumulator form for its own cross-check against any standard
# fp32 per-row normalize on equivalent binary input.
_normalize_bold_bitpacked_kernel = cp.RawKernel(r"""
extern "C" __global__
void normalize_bold_bitpacked_NTD(const unsigned char* __restrict__ src_packed,
                                   float* __restrict__ dst_f32,
                                   int N, int T, int D, int D_bytes) {
    // One thread block per (n, t) row. Three passes:
    //   pass 1: integer popcount over byte stream → block-reduce → fp64 mean
    //   pass 2: byte-strided unpack + write (v - mean) + OR-reduce has_zero
    //   pass 3: if no post-mean zeros, multiply by 1/sqrt(post_sumsq)
    //
    // ``int`` accumulator for the popcount sum: D < 2^30 always at
    // fsa6 / fsaverage7 mesh sizes, so no overflow.

    const int row = blockIdx.x;
    const int n = row / T;
    if (n >= N) return;

    const unsigned char* src_row = src_packed + (size_t)row * (size_t)D_bytes;
    float* dst_row = dst_f32 + (size_t)row * (size_t)D;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    // Pass 1: integer popcount over byte stream.
    int local_sum_i = 0;
    for (int b = tid; b < D_bytes; b += bs) {
        local_sum_i += __popc((unsigned int)src_row[b]);
    }
    __shared__ int s_sum_i[256];
    s_sum_i[tid] = local_sum_i;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_i[tid] += s_sum_i[tid + s];
        }
        __syncthreads();
    }

    // Mean + inv-norm derivation. For binary input, sumsq == sum
    // (v in {0, 1} implies v*v == v), so post_sumsq = sum - D * mean^2.
    __shared__ float s_mean;
    __shared__ float s_inv_norm;
    __shared__ int   s_has_zero;
    if (tid == 0) {
        double sum_d = (double)s_sum_i[0];   // exact: popcount < 2^31 fits in double.
        double mean_d = sum_d / (double)D;
        double post_sumsq = sum_d - (double)D * mean_d * mean_d;
        s_mean = (float)mean_d;
        s_inv_norm = (post_sumsq > 0.0)
            ? (float)(1.0 / sqrt(post_sumsq))
            : 1.0f;
        s_has_zero = 0;
    }
    __syncthreads();

    // Pass 2: byte-strided unpack + write (v - mean) + OR-reduce has_zero.
    int local_has_zero = 0;
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
            }
        }
    }
    if (local_has_zero) atomicOr(&s_has_zero, 1);
    __syncthreads();

    // Pass 3: scale if no post-mean zero (same MATLAB ``all_nonzero`` gate).
    if (!s_has_zero) {
        for (int d = tid; d < D; d += bs) {
            dst_row[d] *= s_inv_norm;
        }
    }
}
""", "normalize_bold_bitpacked_NTD")


def _normalize_bold_NTD_from_packed(
    dst_f32: cp.ndarray,     # (N, T, D) fp32 — out
    src_packed: cp.ndarray,  # (N, T, D_bytes) uint8 — in
    D: int,
) -> None:
    """Bit-unpack + per-(n, t) demean + L2-row-norm in one device kernel.

    ``dst_f32`` and ``src_packed`` must both be device-resident,
    C-contiguous, and shape-consistent (same N, T; last axis of src
    must be ``⌈D/8⌉``).
    """
    if dst_f32.dtype != cp.float32:
        raise ValueError(f"dst_f32 must be fp32; got {dst_f32.dtype}")
    if src_packed.dtype != cp.uint8:
        raise ValueError(f"src_packed must be uint8; got {src_packed.dtype}")
    if dst_f32.ndim != 3 or src_packed.ndim != 3:
        raise ValueError(
            f"dst_f32 and src_packed must be 3D; got {dst_f32.shape}, "
            f"{src_packed.shape}"
        )
    N, T, D_dst = dst_f32.shape
    Ns, Ts, D_bytes = src_packed.shape
    if (Ns, Ts) != (N, T):
        raise ValueError(
            f"shape mismatch: dst (N, T) = ({N}, {T}); src (N, T) = ({Ns}, {Ts})"
        )
    if D_dst != int(D):
        raise ValueError(
            f"dst_f32 D-axis ({D_dst}) must equal D_unpacked ({D})"
        )
    if D_bytes != (int(D) + 7) // 8:
        raise ValueError(
            f"src_packed D_bytes ({D_bytes}) must equal ceil(D/8) "
            f"({(int(D) + 7) // 8}) for D={D}"
        )
    if not dst_f32.flags["C_CONTIGUOUS"]:
        raise ValueError("dst_f32 must be C-contiguous")
    if not src_packed.flags["C_CONTIGUOUS"]:
        raise ValueError("src_packed must be C-contiguous")
    block = _NORMALIZE_BOLD_BLOCK
    if block <= 0 or (block & (block - 1)) != 0:
        raise ValueError(
            f"normalize_bold block size must be power-of-2; got {block}"
        )
    if block > _NORMALIZE_BOLD_BLOCK_MAX:
        raise ValueError(
            f"normalize_bold block size {block} exceeds shared-mem ceiling "
            f"{_NORMALIZE_BOLD_BLOCK_MAX}"
        )
    grid = N * T
    _normalize_bold_bitpacked_kernel(
        (grid,), (block,),
        (src_packed, dst_f32,
         cp.int32(N), cp.int32(T), cp.int32(D), cp.int32(D_bytes)),
    )


class VmfClusteringSessionCUDA:
    """Full-GPU vmf_clustering super-call (single-subject).

    All buffers resident on device throughout the EM loop. Only
    check_connectedness runs CPU-side (BFS on labels); its boundary is
    a 640 KB labels array per comp_iter.

    Not thread-safe — ``run()`` mutates device-resident scratch and
    ping-pong buffers; concurrent calls on the same Session would race.
    Build one Session per worker.
    """

    def __init__(self,
                 data_series_NTD: np.ndarray,              # (N, T, ⌈D/8⌉) uint8 packed
                 grad_data: Optional[np.ndarray],
                 sphere_xyz: Optional[np.ndarray],
                 lh_sphere_mesh: Optional[Dict[str, np.ndarray]],
                 rh_sphere_mesh: Optional[Dict[str, np.ndarray]],
                 theta: np.ndarray,
                 boundary_mask: np.ndarray,
                 neighborhood: np.ndarray,
                 row_idx: np.ndarray, col_idx: np.ndarray,
                 dim: int, num_clusters: int, num_session: int,
                 w: float, c: float,
                 beta: np.ndarray,
                 connect_th: float = 15.0,
                 epsilon: float = 1e-4,
                 max_iter_em: int = 100,
                 max_iter_lambda: int = 50,
                 max_iter_m: int = 50,
                 max_iter_comp: int = 15,
                 cMSHBM_isolated_component_min_size: int = 5,
                 *,
                 D_unpacked: int,
                 variant_spec: Optional[VariantSpec] = None):
        if variant_spec is None:
            variant_spec = VariantSpec.from_pipeline_type("gMSHBM")
        self.variant = variant_spec
        # Variant + BOLD validation shared with CPU Session via
        # ``_session_common`` — both backends must agree on what
        # they accept.
        validate_variant_requirements(
            variant_spec, grad_data, sphere_xyz,
            lh_sphere_mesh, rh_sphere_mesh,
        )
        # gpu_full accepts ONLY bit-packed uint8 BOLD ``(N, T, ⌈D/8⌉)``
        # (one bit per cell, LSB-first within each byte). The on-device
        # kernel fuses unpack + demean + L2-norm.
        N, T, _D_bytes = validate_packed_bold_shape(
            data_series_NTD, num_session, D_unpacked,
        )
        # validate_packed_bold_shape already did np.asarray; just ensure
        # C-contig for the H2D upload below.
        ds = np.ascontiguousarray(data_series_NTD)
        D = int(D_unpacked)
        L = int(num_clusters)

        self.N = int(N); self.D = int(D); self.T = int(T); self.L = L
        self.n_lh = N // 2; self.L_lh = L // 2
        self.dim = int(dim)
        self.w = float(w); self.c = float(c)
        self.connect_th = float(connect_th)
        self.epsilon = float(epsilon)
        self.max_iter_em = int(max_iter_em)
        self.max_iter_lambda = int(max_iter_lambda)
        self.max_iter_m = int(max_iter_m)
        self.max_iter_comp = int(max_iter_comp)
        self.cMSHBM_isolated_component_min_size = int(
            cMSHBM_isolated_component_min_size
        )

        # ── H2D static caches ──────────────────────────────────────────
        # BOLD: H2D the packed bytes once into a ``(N, T, ⌈D/8⌉)`` scratch
        # buffer, run the fused unpack + normalize kernel writing into
        # ``bold_NTD_dev`` (the (N, T, D) fp32 device buffer + zero-copy
        # (N, T·D) view used by the fused sgemm in ELambda), then drop
        # the packed scratch. ~9x smaller H2D traffic vs a hypothetical
        # fp32 path (288 MB packed vs 2.3 GB fp32 at fsa6/D=1175/T=6)
        # and saves the host-side bit-unpack + fp32 cast.
        packed_dev = cp.asarray(ds)            # (N, T, D_bytes) uint8 — H2D
        self.bold_NTD_dev = cp.empty((N, T, D), dtype=cp.float32)
        _normalize_bold_NTD_from_packed(self.bold_NTD_dev, packed_dev, D)
        del packed_dev
        self.bold_NxTD_view_dev = self.bold_NTD_dev.reshape(N, T * D)

        # log(θ) — TWO versions, both computed on GPU directly from
        # ``theta_dev`` (no host roundtrip):
        #   * log_theta_dev      : log(θ) with -Inf at θ==0  (used by ELambda
        #                          mega kernel; -Inf propagates additively)
        #   * log_theta_cost_dev : log(θ) with floor LOG_EPS_POW20
        #                          ≈ -720.87 at θ==0 (used by EMStop's
        #                          cost reduction; floor avoids 0 * -Inf
        #                          = NaN at theta==0 cells that also
        #                          have s_lambda == 0).
        th_f32 = np.ascontiguousarray(theta, dtype=np.float32)
        if th_f32.shape != (N, L):
            raise ValueError(f"theta shape {th_f32.shape} != ({N}, {L})")
        self.theta_dev = cp.asarray(th_f32)

        # log_theta_dev: log(θ) with -Inf at θ<=0. Equivalent to the
        # CPU ``compute_log_theta_with_neginf_cap_f32`` kernel:
        #   v = θ[n, l]; out = -inf if v <= 0 else log(v).
        # ``cp.log(0.0)`` natively returns -Inf in fp32, so for the
        # non-negative θ we have in production this is just ``cp.log``;
        # we still mask defensively in case a future caller passes
        # negative values (which would yield NaN from raw cp.log).
        # ``cp.where`` with two fp32 branches already returns fp32 — no
        # ``.astype`` needed (which would copy an extra (N, L) buffer).
        self.log_theta_dev = cp.where(
            self.theta_dev > cp.float32(0.0),
            cp.log(self.theta_dev),
            cp.float32(-np.inf),
        )

        # log_theta_cost_dev: same shape, floor at log(eps_f64^20) ≈ -720.87.
        # Equivalent to the CPU ``log_with_neginf_floor`` kernel.
        log_floor_f32 = cp.float32(LOG_EPS_POW20)
        self.log_theta_cost_dev = cp.where(
            self.theta_dev > cp.float32(0.0),
            cp.log(self.theta_dev),
            log_floor_f32,
        )

        # boundary_mask
        bm = np.ascontiguousarray(boundary_mask, dtype=np.float32)
        # Invariant: cross-hemi (LH-vert × RH-parcel and vice versa) cells are
        # zero. ``_em_stop`` overwrites cross-hemi spatial_connect_vmf cells
        # with ``-737`` (NaN/Inf cleanup); the next em_iter's mega-fused
        # E-step kernel relies on bm == 0 at those cells to early-skip the
        # exp() pass, masking the -737-vs--Inf difference. One-time check
        # at construction so a malformed boundary_mask fails loudly here
        # rather than producing silently-wrong cost terms downstream.
        if not (np.all(bm[:self.n_lh, self.L_lh:] == 0.0)
                and np.all(bm[self.n_lh:, :self.L_lh] == 0.0)):
            raise ValueError(
                "boundary_mask must be zero at cross-hemi cells "
                "(LH-vert × RH-parcel and RH-vert × LH-parcel)"
            )
        self.boundary_mask_dev = cp.asarray(bm)

        # beta
        beta_f32 = np.ascontiguousarray(np.asarray(beta).ravel(),
                                          dtype=np.float32)
        self.beta_dev = cp.asarray(beta_f32)

        # Active row mask + indices.
        theta_row_sum = th_f32.sum(axis=1)
        row_active_mask_host = (theta_row_sum != 0)
        row_idx_active_host = np.ascontiguousarray(
            np.where(row_active_mask_host)[0], dtype=np.int64
        )
        self.M_active = int(row_idx_active_host.shape[0])
        self.row_idx_active_dev = cp.asarray(row_idx_active_host)
        inv_active = np.full(N, -1, dtype=np.int64)
        inv_active[row_idx_active_host] = np.arange(self.M_active, dtype=np.int64)
        self.inv_active_idx_dev = cp.asarray(inv_active)

        # V_lambda state on GPU. Build the CPU Session purely as an
        # input validator (range checks, neighborhood transpose); ship
        # the validated arrays to GPU. Construction is cheap — the
        # (M_active, K) ``V_lam_f32`` zero-fill is the only non-trivial
        # allocation, and we don't use it on the GPU path.
        cpu_vlam = VLambdaSession(
            neighborhood=neighborhood,
            row_idx=row_idx, col_idx=col_idx, K=L,
            row_idx_active=row_idx_active_host,
        )
        if cpu_vlam.N != self.M_active:
            raise ValueError(
                f"V_lambda Session N={cpu_vlam.N} != M_active={self.M_active}"
            )
        self.vlam_neighborhood_NM_dev = cp.asarray(cpu_vlam.neighborhood_NM)
        self.vlam_row_idx_dev = cp.asarray(cpu_vlam.row_idx)
        self.vlam_col_idx_dev = cp.asarray(cpu_vlam.col_idx)
        self.vlam_V_lam_dev = cp.zeros((self.M_active, L), dtype=cp.float32)

        # spatial_xyz: pre-normalize sphere on CPU, H2D, build LH/RH views.
        # Built only when the variant uses the xyz prior OR runs
        # check_connectedness (which needs sphere coords for distance).
        if variant_spec.use_xyz_prior or variant_spec.use_check_connectedness:
            sphere_unit = compute_unit_sphere_xyz(sphere_xyz)   # (N, 3) fp32
            self.sphere_xyz_unit_dev = cp.asarray(sphere_unit)
            self.sphere_xyz_unit_lh_dev = self.sphere_xyz_unit_dev[:self.n_lh]
            self.sphere_xyz_unit_rh_dev = self.sphere_xyz_unit_dev[self.n_lh:]
        else:
            self.sphere_xyz_unit_dev = None
            self.sphere_xyz_unit_lh_dev = None
            self.sphere_xyz_unit_rh_dev = None

        # spatial_connect: gMSHBM only — cache grad_data + grad_sq_norms.
        if variant_spec.use_connect_prior:
            gd = np.ascontiguousarray(grad_data, dtype=np.float32)
            if gd.shape[0] != N:
                raise ValueError(f"grad_data has {gd.shape[0]} rows, expected {N}")
            self.grad_data_dev = cp.asarray(gd)
            self.grad_lh_dev = self.grad_data_dev[:self.n_lh]
            self.grad_rh_dev = self.grad_data_dev[self.n_lh:]
            self.grad_sq_norms_dev = (self.grad_data_dev ** 2).sum(axis=1).astype(cp.float32)
            self.D_grad = int(gd.shape[1])
        else:
            self.grad_data_dev = None
            self.grad_lh_dev = None
            self.grad_rh_dev = None
            self.grad_sq_norms_dev = None
            self.D_grad = 0

        # ── Mesh data for check_connectedness (CPU-side only) ──────────
        self.lh_sphere_mesh = lh_sphere_mesh
        self.rh_sphere_mesh = rh_sphere_mesh

        # ── Per-call scratch on GPU ────────────────────────────────────
        # M-step ping-pong buffers for s_t_nu (TWO independent allocations).
        self._s_t_nu_TDL_A_dev = cp.empty((T, D, L), dtype=cp.float32)
        self._s_t_nu_TDL_B_dev = cp.empty((T, D, L), dtype=cp.float32)
        # M-step per-call scratch.
        self._X_dot_sl_TDL_dev = cp.empty((T, D, L), dtype=cp.float32)
        self._sigma_psi_DL_dev = cp.empty((D, L), dtype=cp.float32)
        self._cos_TL_dev = cp.empty((T, L), dtype=cp.float32)
        # M-step per-call inputs (re-populated each em_iter).
        self._s_psi_DL_dev = cp.empty((D, L), dtype=cp.float32)
        self._sigma_L_dev = cp.empty(L, dtype=cp.float32)
        # ELambda per-em_iter cache.
        self._s_t_nu_TDxL_dev = cp.empty((T * D, L), dtype=cp.float32)
        self._acc_NL_dev = cp.empty((N, L), dtype=cp.float32)
        self._kappa_f32_dev = cp.empty(L, dtype=cp.float32)
        self._cdln_per_k_f32_dev = cp.empty(L, dtype=cp.float32)
        self._cdln_T_f32_dev = cp.empty(L, dtype=cp.float32)
        self._col_zero_mask_dev = cp.empty(L, dtype=cp.bool_)
        # ELambda per-comp_iter ping-pong for s_lambda + V_temp output.
        self._s_lambda_A_dev = cp.empty((N, L), dtype=cp.float32)
        self._s_lambda_B_dev = cp.empty((N, L), dtype=cp.float32)
        self._V_temp_dev = cp.empty((N, L), dtype=cp.float32)
        # spatial_xyz outputs (allocated only when variant uses xyz prior).
        if variant_spec.use_xyz_prior:
            self._s_muc_dev = cp.empty((3, L), dtype=cp.float32)
            self._spatial_xyz_vmf_dev = cp.zeros((N, L), dtype=cp.float32)
        else:
            self._s_muc_dev = None
            self._spatial_xyz_vmf_dev = None
        # spatial_connect outputs (gMSHBM only). Cross-hemi pre-filled with -Inf.
        if variant_spec.use_connect_prior:
            scv_init = np.zeros((N, L), dtype=np.float32)
            scv_init[:self.n_lh, self.L_lh:] = np.float32(-np.inf)
            scv_init[self.n_lh:, :self.L_lh] = np.float32(-np.inf)
            self._spatial_connect_vmf_dev = cp.asarray(scv_init)
            self._u_dev = cp.empty((L, self.D_grad), dtype=cp.float32)
        else:
            self._spatial_connect_vmf_dev = None
            self._u_dev = None
        # Variant-routing zero buffer: reused as the kernel's scv / sxv slot
        # whenever a variant doesn't compute that prior. One device-resident
        # (N, L) fp32 buffer of zeros, shared across all absent-prior slots
        # (the kernel only reads it).
        self._zero_NL_dev = cp.zeros((N, L), dtype=cp.float32)
        # EMStop scratch.
        self._log_lambda_prop_dev = cp.empty((N, L), dtype=cp.float32)
        # xyz_gamma + scratch (mostly tiny).
        self._xyz_gamma_f32_dev = cp.empty(L, dtype=cp.float32)
        self._cdln_xyz_per_k_dev = cp.empty(L, dtype=cp.float32)

        # Last timings + intermediate state.
        self.last_timings: Dict[str, float] = {}
        self._row_active_mask_host = row_active_mask_host  # for argmax masking

    # ─────────────────────────────────────────────────────────────────
    # GPU sub-stages — operate on session GPU buffers in place.
    # ─────────────────────────────────────────────────────────────────
    def _m_step(self, s_t_nu_in_TDL_dev, s_lambda_NL_dev, s_psi_in_DL_dev,
                sigma_in_dev, kappa_old_L_dev) -> Tuple[Any, Any, int]:
        """Run M-step inner-while loop on GPU.

        Inputs are GPU arrays. Returns ``(s_t_nu_out_TDL_dev, kappa_new_L_dev,
        iter_m)`` — the s_t_nu output is a view of one of the ping-pong
        buffers; caller can re-pass on next em_iter.
        """
        L = self.L; T = self.T; D = self.D
        # Stage inputs.
        cp.copyto(self._s_t_nu_TDL_A_dev, s_t_nu_in_TDL_dev)
        cp.copyto(self._s_psi_DL_dev, s_psi_in_DL_dev)
        cp.copyto(self._sigma_L_dev, sigma_in_dev)
        # sigma_psi[d, l] = sigma[l] * s_psi[d, l]
        cp.multiply(self._sigma_L_dev, self._s_psi_DL_dev,
                    out=self._sigma_psi_DL_dev)
        # X_dot_sl[t, d, l] = X[:, t, :].T @ s_lambda
        _kernels_gpu.mstep_X_dot_sl_TDL_cupy(
            self.bold_NTD_dev, s_lambda_NL_dev, self._X_dot_sl_TDL_dev,
        )
        # denom = num_session * sum(s_lambda) — fp64 scalar.
        denom_f64 = float(s_lambda_NL_dev.astype(cp.float64).sum() * T)

        # Ping-pong setup.
        st_buf = (self._s_t_nu_TDL_A_dev, self._s_t_nu_TDL_B_dev)
        kappa_old_host = float(cp.asnumpy(kappa_old_L_dev[0]))   # scalar
        kappa_new_scalar = kappa_old_host
        # flag_T_acc lives on device; OR-accumulated each iter; reduces to a
        # scalar host bool via the (T,)-sum check below. Avoids the per-iter
        # (T, L) D2H sync the prior implementation paid.
        flag_T_acc_dev = cp.zeros(T, dtype=cp.bool_)
        eps = self.epsilon
        eps_f32 = cp.float32(eps)
        iter_m = 0
        final_idx = 0

        while True:
            iter_m += 1
            old_idx = (iter_m - 1) % 2
            new_idx = iter_m % 2
            final_idx = new_idx

            # kappa_sum_f64 = sum_{t, d, l} s_t_nu[t, d, l] * X_dot_sl[t, d, l]
            kappa_sum_f64 = float((st_buf[old_idx] * self._X_dot_sl_TDL_dev)
                                   .astype(cp.float64).sum())
            rbar = kappa_sum_f64 / denom_f64
            kappa_new_scalar = invad(self.dim, rbar)
            kappa_f32 = np.float32(kappa_new_scalar)

            # Per-t fused: s_t_nu_new + cosine.
            _kernels_gpu.mstep_fused_iter_m_body_cupy(
                kappa_f32, self._X_dot_sl_TDL_dev, self._sigma_psi_DL_dev,
                st_buf[old_idx], st_buf[new_idx], self._cos_TL_dev,
            )
            # Per-t convergence test on device: all (1 - cos) < eps along L.
            # Monotone-OR into flag_T_acc_dev (matches MATLAB's flag_nu latch
            # in the M-step). The .sum() == T reduction below forces the only
            # D2H sync per iter_m for this branch.
            this_iter_flags = cp.all(
                (cp.float32(1.0) - self._cos_TL_dev) < eps_f32, axis=1,
            )
            flag_T_acc_dev |= this_iter_flags
            all_flag = bool(int(flag_T_acc_dev.sum()) == T)

            kappa_drift = abs(kappa_old_host - kappa_new_scalar) / max(abs(kappa_old_host), 1e-30)
            kappa_old_host = kappa_new_scalar

            if all_flag and kappa_drift < eps:
                break
            if iter_m > self.max_iter_m:
                break

        # Build kappa_L on device (replicated scalar).
        kappa_L_new_dev = cp.full(L, kappa_new_scalar, dtype=cp.float64)
        return st_buf[final_idx], kappa_L_new_dev, iter_m

    def _elambda_prepare_em_iter(self,
                                  s_t_nu_TDL_dev,    # (T, D, L)
                                  kappa_L_dev,       # (L,) fp64
                                  ) -> None:
        """Compute em-iter-invariant cache: acc = X @ s_t_nu, Cdln(kappa)."""
        # Permute s_t_nu (T, D, L) → (T*D, L) reshape after axis swap.
        # For our ping-pong buffers (T, D, L) is already the natural layout;
        # we need (T*D, L) which is just the contiguous reshape.
        # Note: s_t_nu_TDL_dev is (T, D, L) C-contig; .reshape(T*D, L) is zero-copy.
        st_TDxL_view = s_t_nu_TDL_dev.reshape(self.T * self.D, self.L)
        cp.matmul(self.bold_NxTD_view_dev, st_TDxL_view, out=self._acc_NL_dev)
        cp.all(self._acc_NL_dev == 0, axis=0, out=self._col_zero_mask_dev)
        # kappa fp32 mirror.
        cp.copyto(self._kappa_f32_dev, kappa_L_dev.astype(cp.float32))
        # Cdln on CPU (scipy Bessel; tiny L=300 vector).
        kappa_host = cp.asnumpy(kappa_L_dev)
        cdln_host = np.empty(self.L, dtype=np.float32)
        cdln_general_to_f32(kappa_host, self.dim, cdln_host)
        self._cdln_per_k_f32_dev.set(cdln_host)
        cp.multiply(self._cdln_per_k_f32_dev, np.float32(self.T),
                     out=self._cdln_T_f32_dev)

    def _spatial_connect(self, s_lambda_dev) -> None:
        """Run spatial_connect_prior on GPU. Writes to ``self._u_dev`` and
        ``self._spatial_connect_vmf_dev``.
        """
        _kernels_gpu.spatial_connect_compute_cupy(
            self.grad_lh_dev, self.grad_rh_dev,
            self.grad_sq_norms_dev,
            s_lambda_dev,
            self.n_lh, self.L_lh, self.D_grad,
            self._u_dev, self._spatial_connect_vmf_dev,
        )

    def _spatial_xyz(self, s_lambda_dev, xyz_gamma_host) -> None:
        """Run spatial_xyz_prior on GPU. xyz_gamma is host (small L vector)."""
        # Cdln(xyz_gamma, 3) on CPU.
        cdln_host = np.empty(self.L, dtype=np.float32)
        cdln_d3_to_f32(np.ascontiguousarray(xyz_gamma_host, dtype=np.float64),
                        cdln_host)
        self._cdln_xyz_per_k_dev.set(cdln_host)
        self._xyz_gamma_f32_dev.set(np.ascontiguousarray(xyz_gamma_host,
                                                           dtype=np.float32))
        _kernels_gpu.spatial_xyz_compute_cupy(
            self.sphere_xyz_unit_lh_dev, self.sphere_xyz_unit_rh_dev,
            self.sphere_xyz_unit_dev,
            s_lambda_dev,
            self._xyz_gamma_f32_dev,
            self._cdln_xyz_per_k_dev,
            self.n_lh, self.L_lh,
            self._s_muc_dev, self._spatial_xyz_vmf_dev,
        )

    def _lambda_loop_body(self, s_lambda_curr_dev, s_lambda_new_dev,
                           scv_dev, sxv_dev,
                           ) -> Tuple[Any, Any, int]:
        """Run inner λ-loop on GPU. Returns (s_lambda_final, V_temp, n_iter).

        ``scv_dev`` / ``sxv_dev`` are the device buffers plugged into the
        kernel's ``β·scv + sxv`` slots — variant-routed by the caller
        (see :class:`VariantSpec` and ``run`` for the assignment table).
        """
        eps = self.epsilon
        checklam = 0.0
        lambda_iter = 0
        curr = s_lambda_curr_dev
        new = s_lambda_new_dev
        while True:
            lambda_iter += 1
            # V_lambda fused (3-phase Potts close-form).
            tmp_lam = curr[self.row_idx_active_dev]
            _kernels_gpu.vlambda_potts_closeform_fused_cupy(
                self.vlam_neighborhood_NM_dev, tmp_lam,
                self.vlam_row_idx_dev, self.vlam_col_idx_dev,
                self.vlam_V_lam_dev,
            )
            # Mega-mega-fused.
            checklam_update = _kernels_gpu.fused_v_lambda_assemble_softmax_drift_cupy(
                self._acc_NL_dev, self._kappa_f32_dev,
                self._cdln_T_f32_dev, self._col_zero_mask_dev,
                self.log_theta_dev, self.w,
                self.vlam_V_lam_dev, self.inv_active_idx_dev, self.c,
                self.beta_dev,
                scv_dev,
                sxv_dev,
                self.boundary_mask_dev,
                curr,
                self._V_temp_dev,
                new,
            )
            curr, new = new, curr
            converged = abs(checklam_update - checklam) <= eps
            checklam = checklam_update
            if converged:
                break
            if lambda_iter > self.max_iter_lambda:
                break
        return curr, self._V_temp_dev, lambda_iter

    def _check_connectedness(self, s_lambda_dev, xyz_gamma_host
                              ) -> Tuple[np.ndarray, float, float]:
        """GPU argmax → CPU BFS → return (xyz_gamma_new_host, max_conn, max_comp).

        Per variant:
          * gMSHBM: components_threshold=3, no RemoveIsolated pre-pass.
          * cMSHBM: components_threshold=1, RemoveIsolated(5) on argmax
            labels BEFORE the components / distance test.
          * dMSHBM: this method is never called.
        """
        N = self.N
        components_threshold = int(self.variant.components_threshold)
        # GPU argmax + active mask.
        row_active_dev = (s_lambda_dev.sum(axis=1) != 0)
        # Need (N,) labels: 0 at inactive, argmax+1 at active.
        labels_dev = cp.zeros(N, dtype=cp.int64)
        active_idx = cp.where(row_active_dev)[0]
        if active_idx.size > 0:
            argmax_active = s_lambda_dev[active_idx, :].argmax(axis=1) + 1
            labels_dev[active_idx] = argmax_active
        labels_host = cp.asnumpy(labels_dev)
        # CPU BFS.
        n_per_hemi = N // 2
        lh_labels = labels_host[:n_per_hemi]
        rh_labels = labels_host[n_per_hemi:]
        if self.variant.pre_predicate_remove_isolated:
            thr = self.cMSHBM_isolated_component_min_size
            lh_labels = remove_isolated_surface_components(
                lh_labels, self.lh_sphere_mesh["vertexNbors"], abs_threshold=thr,
            )
            rh_labels = remove_isolated_surface_components(
                rh_labels, self.rh_sphere_mesh["vertexNbors"], abs_threshold=thr,
            )
        parcel_components, lh_ci, rh_ci = compute_components_general(
            lh_labels, rh_labels,
            self.lh_sphere_mesh["vertexNbors"],
            self.rh_sphere_mesh["vertexNbors"],
            self.L, return_ci=True,
        )
        eucli_dist = component_distance(
            lh_labels, rh_labels,
            self.lh_sphere_mesh, self.rh_sphere_mesh, self.L,
            parcel_components=parcel_components,
            lh_ci_full=lh_ci, rh_ci_full=rh_ci,
        )
        distrib_mask = (eucli_dist > self.connect_th) | (
            parcel_components > components_threshold
        )
        if distrib_mask.any():
            max_connectedness = float(eucli_dist[distrib_mask].max())
            max_components = float(parcel_components[
                distrib_mask & np.isfinite(parcel_components)].max())
            xyz_gamma_new = xyz_gamma_host.copy()
            xyz_gamma_new[distrib_mask] += 1000.0
        else:
            max_connectedness = 0.0
            max_components = float(components_threshold)
            xyz_gamma_new = xyz_gamma_host.copy()
        return xyz_gamma_new, max_connectedness, max_components

    def _em_stop(self, s_t_nu_TDL_dev, kappa_L_dev, s_lambda_dev,
                  V_temp_dev, cost_host, iter_em,
                  scv_dev) -> Tuple[float, int, Optional[np.ndarray]]:
        """GPU EMStop. Returns (update_cost_scalar, stop_em, cost_em_host).

        ``scv_dev`` is the device buffer plugged into the cost
        integrand's ``+ β·scv·s_lambda`` slot — gMSHBM passes the
        gradient ``spatial_connect_vmf`` (with NaN/Inf cleanup);
        cMSHBM/dMSHBM pass ``self._zero_NL_dev`` (no spatial term in
        cost).

        Mirrors :func:`em_stop_criterion._kernels.compute_update_cost_kernel`
        — coefficients matter (no w on log_theta term, -w on log_sl,
        -c on V_temp). Uses ``log_theta_cost_dev`` (with-floor) NOT
        ``log_theta_dev`` (with -Inf), to avoid ``0 * -Inf = NaN``.
        """
        # acc + assemble — reuse cached acc (s_t_nu unchanged this em_iter).
        # log_lambda_prop[n, l] = T * cdln[l] + kappa[l] * acc[n, l]
        cp.add(cp.float32(self.T) * self._cdln_per_k_f32_dev,
                self._kappa_f32_dev * self._acc_NL_dev,
                out=self._log_lambda_prop_dev)

        # gMSHBM only: spatial_connect_vmf cleanup (NaN / Inf → log(eps^20)
        # ≈ -737). OVERWRITES cross-hemi -Inf cells with -737 (finite);
        # next em_iter's spatial_connect block-diag rewrite leaves cross-
        # hemi alone, so they stay -737. The ELambda mega kernel reads
        # spatial_connect_vmf in iter_em > 1 — at cross-hemi cells
        # boundary_mask is 0, so the contribution is zeroed by Pass B's
        # bm==0 early-skip; the -737-vs--Inf difference is
        # mathematically irrelevant downstream.
        LOG_FLOOR = cp.float32(-737.0)
        if self.variant.em_stop_uses_beta_scv:
            finite = cp.isfinite(scv_dev)
            scv_dev[:] = cp.where(finite, scv_dev, LOG_FLOOR)

        # Cost reduction. Coefficients per CPU compute_update_cost_kernel.
        log_sl = cp.where(s_lambda_dev > cp.float32(0.0),
                           cp.log(cp.maximum(s_lambda_dev, cp.float32(1e-38))),
                           LOG_FLOOR)
        w_f32 = cp.float32(self.w)
        c_f32 = cp.float32(self.c)
        integrand = s_lambda_dev * (
            self._log_lambda_prop_dev
            + self.log_theta_cost_dev
            - w_f32 * log_sl
            - c_f32 * V_temp_dev
            + self.beta_dev * scv_dev
        )
        update_cost = float(integrand.astype(cp.float64).sum())

        # Convergence test (single-subject). Mirror the CPU-side
        # ``em_stop_criterion.convergence_test`` exactly: numpy-divide so
        # ``0/0 -> NaN`` (treated as converged via the ``not (NaN > 1e-4)``
        # idiom), ``nonzero/0 -> Inf`` (treated as NOT converged).
        # A naive ``if cost_scalar == 0: ratio = inf`` shortcut would force
        # non-convergence in the 0/0 case, diverging from MATLAB on
        # degenerate-but-valid inputs (e.g., heavily masked subjects where
        # both update_cost and prior cost remain zero).
        cost_scalar = float(cost_host.ravel()[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = float(np.abs(
                (np.float64(update_cost) - np.float64(cost_scalar))
                / np.float64(cost_scalar)
            ))
        # NaN-as-converged: ``NaN > 1e-4`` is False -> ``not False`` -> True.
        converged = not (ratio > 1e-4)
        stop_em = 0
        cost_em: Optional[np.ndarray] = None
        if converged:
            stop_em = 1
            cost_em = np.array([[cost_scalar]], dtype=np.float32)
        # Hard ceiling at 100 — mirrors MATLAB step3 line 757 and the CPU
        # ``em_stop_criterion.convergence_test``. The outer ``run()`` loop
        # tests ``self.max_iter_em`` as a soft cap; setting it above 100
        # has no effect because this fires first.
        if iter_em > 100:
            stop_em = 1
            cost_em = np.array([[update_cost]], dtype=np.float32)
        return update_cost, stop_em, cost_em

    # ─────────────────────────────────────────────────────────────────
    # Top-level run
    # ─────────────────────────────────────────────────────────────────
    def run(self, params_in: Dict[str, Any], time_stages: bool = False
            ) -> Dict[str, Any]:
        """Run one full vmf_clustering super-call entirely on GPU."""
        # Stage initial Params (host) into GPU buffers. Strict shapes —
        # single-subject pipeline; (..., 1) MATLAB leftovers are rejected.
        s_lambda_host = _stage_2d(params_in["s_lambda"], self.N, self.L, "s_lambda")
        s_t_nu_host   = _stage_3d(params_in["s_t_nu"],
                                   self.D, self.L, self.T, "s_t_nu")
        s_psi_host    = _stage_2d(params_in["s_psi"], self.D, self.L, "s_psi")
        sigma_host    = _stage_1d(params_in["sigma"], self.L, "sigma", np.float32)
        kappa_host    = _stage_1d(params_in["kappa"], self.L, "kappa", np.float64)
        epsil         = _stage_1d(params_in["epsil"], self.L, "epsil", np.float32)
        mu            = np.asarray(params_in["mu"])
        xyz_gamma_host = _stage_1d(params_in["xyz_gamma"], self.L,
                                    "xyz_gamma", np.float64)
        spatial_xyz_vmf_host = np.ascontiguousarray(
            params_in["spatial_xyz_vmf"], dtype=np.float32,
        )

        # H2D initial state.
        # s_lambda goes into A; the lambda loop ping-pongs between A and B.
        self._s_lambda_A_dev.set(s_lambda_host)
        # s_t_nu in TDL layout — CPU input is (D, L, T), permute to (T, D, L).
        st_TDL_host = np.ascontiguousarray(np.transpose(s_t_nu_host, (2, 0, 1)))
        self._s_t_nu_TDL_A_dev.set(st_TDL_host)
        self._s_psi_DL_dev.set(s_psi_host)
        self._sigma_L_dev.set(sigma_host)
        if self.variant.use_xyz_prior:
            self._spatial_xyz_vmf_dev.set(spatial_xyz_vmf_host)
        # kappa as fp64 device vector (initially replicated scalar).
        kappa_L_dev = cp.asarray(kappa_host) if kappa_host.shape[0] == self.L \
                      else cp.full(self.L, float(kappa_host[0]), dtype=cp.float64)

        s_lambda_curr_dev = self._s_lambda_A_dev
        s_lambda_new_dev = self._s_lambda_B_dev
        s_t_nu_curr_TDL_dev = self._s_t_nu_TDL_A_dev

        # Per-stage timing accumulators.
        timings = {
            "m_step": 0.0, "spatial_connect_prior": 0.0,
            "e_step_lambda_loop": 0.0, "check_connectedness": 0.0,
            "spatial_xyz_prior": 0.0, "em_stop_criterion": 0.0,
            "iter_count_em_total": 0, "iter_count_m_total": 0,
            "iter_count_lambda_total": 0, "iter_count_comp_total": 0,
            "iter_count_check_conn": 0, "iter_count_spatial_xyz": 0,
        }

        cost_host = np.zeros((1, 1), dtype=np.float32)
        max_conn = 0.0; max_comp = 0.0
        cost_em_out = None
        u_centroid_host = None; s_muc_host = None
        iter_em = 0

        while True:
            iter_em += 1
            timings["iter_count_em_total"] = iter_em

            # ── M-step ──
            t0 = time.perf_counter()
            s_t_nu_curr_TDL_dev, kappa_L_dev, iter_m = self._m_step(
                s_t_nu_curr_TDL_dev, s_lambda_curr_dev,
                self._s_psi_DL_dev, self._sigma_L_dev, kappa_L_dev,
            )
            cp.cuda.get_current_stream().synchronize()
            timings["m_step"] += time.perf_counter() - t0
            timings["iter_count_m_total"] += iter_m

            # ── ELambda prepare_em_iter (sgemm + Cdln) ──
            t0 = time.perf_counter()
            self._elambda_prepare_em_iter(s_t_nu_curr_TDL_dev, kappa_L_dev)
            cp.cuda.get_current_stream().synchronize()
            timings["e_step_lambda_loop"] += time.perf_counter() - t0

            # ── Pre-comp_iter prior refresh (variant-specific) ──
            # gMSHBM: spatial_connect_prior. cMSHBM: spatial_xyz_prior.
            # dMSHBM: no-op. See VmfClusteringSession.run for the
            # rationale; the GPU mirror keeps the same dispatch shape.
            if self.variant.use_connect_prior:
                t0 = time.perf_counter()
                self._spatial_connect(s_lambda_curr_dev)
                cp.cuda.get_current_stream().synchronize()
                timings["spatial_connect_prior"] += time.perf_counter() - t0
            elif self.variant.use_xyz_prior:
                t0 = time.perf_counter()
                self._spatial_xyz(s_lambda_curr_dev, xyz_gamma_host)
                cp.cuda.get_current_stream().synchronize()
                timings["spatial_xyz_prior"] += time.perf_counter() - t0
                timings["iter_count_spatial_xyz"] += 1

            # ── E-step routing (per VariantSpec; mirrors CPU run()) ──
            #   gMSHBM: scv = spatial_connect_vmf, sxv = spatial_xyz_vmf.
            #   cMSHBM: scv = spatial_xyz_vmf,    sxv = zero.
            #   dMSHBM: scv = sxv = zero.
            if self.variant.use_connect_prior:
                scv_estep_dev = self._spatial_connect_vmf_dev
                sxv_estep_dev = self._spatial_xyz_vmf_dev
            elif self.variant.use_xyz_prior:
                scv_estep_dev = self._spatial_xyz_vmf_dev
                sxv_estep_dev = self._zero_NL_dev
            else:
                scv_estep_dev = self._zero_NL_dev
                sxv_estep_dev = self._zero_NL_dev

            # ── E-step + (optional) check_connectedness + xyz_prior ──
            if not self.variant.wrap_comp_iter:
                # dMSHBM: single λ-pass per EM iter.
                t0 = time.perf_counter()
                s_lambda_curr_dev, V_temp_dev, lambda_iter = self._lambda_loop_body(
                    s_lambda_curr_dev, s_lambda_new_dev,
                    scv_estep_dev, sxv_estep_dev,
                )
                if s_lambda_curr_dev is self._s_lambda_A_dev:
                    s_lambda_new_dev = self._s_lambda_B_dev
                else:
                    s_lambda_new_dev = self._s_lambda_A_dev
                cp.cuda.get_current_stream().synchronize()
                timings["e_step_lambda_loop"] += time.perf_counter() - t0
                timings["iter_count_lambda_total"] += lambda_iter
                timings["iter_count_comp_total"] += 1
            else:
                stop_comp = False
                comp_iter = 0
                while not stop_comp:
                    comp_iter += 1
                    timings["iter_count_comp_total"] += 1

                    # E-step lambda body.
                    t0 = time.perf_counter()
                    s_lambda_curr_dev, V_temp_dev, lambda_iter = self._lambda_loop_body(
                        s_lambda_curr_dev, s_lambda_new_dev,
                        scv_estep_dev, sxv_estep_dev,
                    )
                    if s_lambda_curr_dev is self._s_lambda_A_dev:
                        s_lambda_new_dev = self._s_lambda_B_dev
                    else:
                        s_lambda_new_dev = self._s_lambda_A_dev
                    cp.cuda.get_current_stream().synchronize()
                    timings["e_step_lambda_loop"] += time.perf_counter() - t0
                    timings["iter_count_lambda_total"] += lambda_iter

                    run_conn = (
                        self.variant.use_check_connectedness
                        and iter_em >= self.variant.first_em_iter_with_conn_xyz
                    )
                    if run_conn:
                        t0 = time.perf_counter()
                        xyz_gamma_host, max_conn, max_comp = self._check_connectedness(
                            s_lambda_curr_dev, xyz_gamma_host,
                        )
                        timings["check_connectedness"] += time.perf_counter() - t0
                        timings["iter_count_check_conn"] += 1

                        if self.variant.use_xyz_prior:
                            t0 = time.perf_counter()
                            self._spatial_xyz(s_lambda_curr_dev, xyz_gamma_host)
                            cp.cuda.get_current_stream().synchronize()
                            timings["spatial_xyz_prior"] += time.perf_counter() - t0
                            timings["iter_count_spatial_xyz"] += 1
                            # cMSHBM: refresh the scv slot to follow the
                            # newly-updated xyz buffer (which IS the scv slot).
                            if not self.variant.use_connect_prior:
                                scv_estep_dev = self._spatial_xyz_vmf_dev
                    else:
                        # gMSHBM iter_em==1: exit comp_iter wrap after
                        # one λ-pass (matches MATLAB ``stop_comp2 = 1``).
                        stop_comp = True

                    if (max_conn <= self.connect_th) and (
                        max_comp <= float(self.variant.components_threshold)
                    ):
                        stop_comp = True
                    if comp_iter >= self.max_iter_comp:
                        stop_comp = True

            # ── EM stop criterion ──
            scv_em_stop_dev = (
                self._spatial_connect_vmf_dev
                if self.variant.em_stop_uses_beta_scv
                else self._zero_NL_dev
            )
            t0 = time.perf_counter()
            update_cost, stop_em, cost_em_view = self._em_stop(
                s_t_nu_curr_TDL_dev, kappa_L_dev, s_lambda_curr_dev,
                V_temp_dev, cost_host, iter_em,
                scv_em_stop_dev,
            )
            cp.cuda.get_current_stream().synchronize()
            timings["em_stop_criterion"] += time.perf_counter() - t0

            if stop_em:
                cost_em_out = cost_em_view if cost_em_view is not None else cost_host.copy()
                break
            if iter_em > self.max_iter_em:
                cost_em_out = np.array([[update_cost]], dtype=np.float32)
                break

            cost_host = np.array([[update_cost]], dtype=np.float32)

        # D2H final outputs.
        s_lambda_out = cp.asnumpy(s_lambda_curr_dev)
        s_t_nu_out_TDL = cp.asnumpy(s_t_nu_curr_TDL_dev)
        # External shape is (D, L, T) — permute back from (T, D, L).
        s_t_nu_out = np.ascontiguousarray(s_t_nu_out_TDL.transpose(1, 2, 0))
        kappa_out = cp.asnumpy(kappa_L_dev)   # (L,) flat

        if time_stages:
            self.last_timings = dict(timings)

        params_out = dict(params_in)
        params_out["s_lambda"] = s_lambda_out
        params_out["s_t_nu"] = s_t_nu_out
        params_out["kappa"] = kappa_out
        params_out["s_psi"] = cp.asnumpy(self._s_psi_DL_dev)
        # 1-D vectors flat (L,) — single-subject pipeline; match CPU path.
        params_out["sigma"] = cp.asnumpy(self._sigma_L_dev)
        params_out["epsil"] = epsil
        params_out["mu"] = mu
        params_out["theta"] = cp.asnumpy(self.theta_dev)
        params_out["xyz_gamma"] = xyz_gamma_host
        # Variant-conditional output fields. Match CPU run() emission rules:
        # always emit spatial_*_vmf for shape compatibility (zero where the
        # variant didn't compute the prior); emit u / s_muc only when the
        # corresponding session was built.
        if self.variant.use_xyz_prior:
            params_out["spatial_xyz_vmf"] = cp.asnumpy(self._spatial_xyz_vmf_dev)
            params_out["s_muc"] = cp.asnumpy(self._s_muc_dev)
        else:
            params_out["spatial_xyz_vmf"] = np.zeros((self.N, self.L), dtype=np.float32)
        if self.variant.use_connect_prior:
            params_out["spatial_connect_vmf"] = cp.asnumpy(self._spatial_connect_vmf_dev)
            params_out["u"] = cp.asnumpy(self._u_dev)
        else:
            params_out["spatial_connect_vmf"] = np.zeros((self.N, self.L), dtype=np.float32)
        params_out["max_connectedness"] = max_conn
        params_out["max_components"] = max_comp
        if cost_em_out is not None:
            params_out["cost_em"] = cost_em_out
        params_out["iter_em"] = iter_em
        return params_out
