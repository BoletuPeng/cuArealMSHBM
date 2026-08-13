"""spatial_connect.py

Gradient-embedding squared-distance prior (the β term,
``spatial_connect_prior`` added to ``log_vmf`` in the EM E-step,
Gordon-2016 edge-detection style). For each parcel ``p``, compute its
centroid ``u[p]`` in the 100-dim diffusion-embedding space; the prior
at each (vertex, parcel) is the negative squared L2 distance from the
vertex's embedding to the parcel centroid. Cross-hemisphere entries
(LH-vertex × RH-parcel and vice versa) are pinned to ``-Inf``.

Public API:
    ConnectSession — caches grad_data + ``||grad||²`` once; recomputes
                     the (N, L) prior each call.

Per call:
    1. ``u = (grad_data.T @ s_lambda).T / sum_lambda_per_k`` — (L, D) fp32
    2. ``cross = grad_data @ u.T``                          — (N, L) sgemm
    3. ``vmf = -||grad - u||²`` via the gemm-trick
       ``-||g||² + 2 g·u - ||u||²``                         — (N, L) fp32
       with cross-hemisphere mask + ``NaN → -Inf`` cleanup.

The explicit-broadcast form
``-((grad[:,None,:] - u[None,:,:])**2).sum(-1)`` would allocate ~10 GB
at production shapes (N=82k, L=300, D=100). The gemm-trick replaces it
with a single (N, D) @ (D, L) sgemm + fused (N, L) elementwise pass.

All intermediates are fp32; output is fp32.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import _kernels


# ─────────────────────────────────────────────────────────────────────────
# Step functions — one per algorithmic step in the MATLAB source.
# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# ConnectSession — block-diagonal hot path.
#
# Three structural exploits driven by step-3 invariants:
#
#  (1) Cross-hemi block-diagonal. ``s_lambda`` is cross-hemi-zero
#      (boundary_mask), so both sgemms split into LH-LH + RH-RH halves.
#      ``spatial_connect_vmf``'s cross-hemi block is statically -Inf
#      (driver lines 768-770); pre-filled once at __init__.
#  (2) ``grad_sq_norms`` is session-invariant; cached at __init__.
#  (3) All per-call temporaries pre-allocated.
#
# Per call: 4 sgemms (LH+RH × {u_update, cross}) + sums + divides + 1
# row-sq-norm + 2 numba assemble leaves. Zero numpy temporaries.
# ─────────────────────────────────────────────────────────────────────────
class ConnectSession:
    """Pre-allocated state for the spatial_connect_prior fast path.

    Construction: takes ``(grad_data, num_verts, num_clusters)``; pre-
    computes ``grad_sq_norms[n] = ||grad_data[n]||^2``; pre-fills the
    cross-hemi block of the output buffer with -Inf.

    Per call: :meth:`compute` runs the pipeline and returns ``(u,
    spatial_connect_vmf)`` as VIEWS into cached buffers. The next
    ``compute`` overwrites them; copy if you need to keep across calls.

    Hemisphere split (LH = first half, RH = second half) assumes:
      * vertex order is ``[lh_verts; rh_verts]`` with ``n_lh = N // 2``.
      * parcel order is ``[lh_parcels; rh_parcels]`` with ``L_lh = L // 2``.
    """

    __slots__ = (
        "N", "L", "D", "n_lh", "L_lh",
        # Cached gradient + per-vertex sq norms (session-invariant).
        "grad_data_f32",                # (N, D) fp32 C-contig — owns memory
        "_grad_lh",                     # (n_lh, D) view
        "_grad_rh",                     # (n_lh, D) view
        "_grad_sq_norms",               # (N,) fp32 — cached
        "_grad_sq_lh",                  # (n_lh,) view
        "_grad_sq_rh",                  # (n_lh,) view
        # Per-call scratch.
        "_u_update_lh",                 # (D, L_lh) fp32 — scratch
        "_u_update_rh",                 # (D, L_lh) fp32 — scratch
        "_sum_lam_lh",                  # (L_lh,) fp32 — scratch
        "_sum_lam_rh",                  # (L_lh,) fp32 — scratch
        "_u_f32",                       # (L, D) fp32 — output
        "_u_lh",                        # (L_lh, D) view of _u_f32
        "_u_rh",                        # (L_lh, D) view of _u_f32
        "_u_sq_norms",                  # (L,) fp32 — scratch
        # Output buffer — cross-hemi pre-filled with -Inf, same-hemi rewritten per call.
        "_spatial_connect_vmf_f32",     # (N, L) fp32 C-contig — output
        "_vmf_lh",                      # (n_lh, L_lh) view (LH-LH block)
        "_vmf_rh",                      # (n_lh, L_lh) view (RH-RH block)
        # Cast destination for fp64 callers.
        "_s_lambda_f32_buf",            # (N, L) fp32 C-contig — scratch
    )

    def __init__(self,
                 grad_data: np.ndarray,
                 num_verts: int,
                 num_clusters: int):
        g = np.ascontiguousarray(grad_data, dtype=np.float32)
        if g.ndim != 2:
            raise ValueError(f"grad_data must be 2D, got {g.shape}")
        N, D = g.shape
        if N != num_verts:
            raise ValueError(
                f"grad_data has {N} rows, expected num_verts={num_verts}"
            )
        L = int(num_clusters)
        if N % 2 != 0:
            raise ValueError(
                f"N={N} must be even (bilateral mesh, half LH half RH)"
            )
        if L % 2 != 0:
            raise ValueError(
                f"L={L} must be even (half LH parcels, half RH parcels)"
            )

        self.N = int(N)
        self.L = L
        self.D = int(D)
        self.n_lh = N // 2
        self.L_lh = L // 2

        # Gradient + LH/RH views.
        self.grad_data_f32 = g
        self._grad_lh = g[:self.n_lh]
        self._grad_rh = g[self.n_lh:]
        # Per-vertex ||grad[n]||^2 — session-invariant, computed once.
        self._grad_sq_norms = np.ascontiguousarray(
            (g ** 2).sum(axis=1).astype(np.float32),
        )
        self._grad_sq_lh = self._grad_sq_norms[:self.n_lh]
        self._grad_sq_rh = self._grad_sq_norms[self.n_lh:]

        # Per-half scratch.
        self._u_update_lh = np.empty((D, self.L_lh), dtype=np.float32)
        self._u_update_rh = np.empty((D, self.L_lh), dtype=np.float32)
        self._sum_lam_lh  = np.empty(self.L_lh, dtype=np.float32)
        self._sum_lam_rh  = np.empty(self.L_lh, dtype=np.float32)

        # u (full L rows) + LH/RH views into it.
        self._u_f32 = np.empty((L, D), dtype=np.float32)
        self._u_lh  = self._u_f32[:self.L_lh]
        self._u_rh  = self._u_f32[self.L_lh:]
        self._u_sq_norms = np.empty(L, dtype=np.float32)

        # ── Pre-filled output: cross-hemi -Inf, same-hemi 0 (will be overwritten) ──
        # Cross-hemi cells are static -Inf (driver lines 768-770) — per-call we
        # only touch the LH-LH and RH-RH diagonal blocks.
        vmf = np.zeros((N, L), dtype=np.float32)
        NEG_INF = np.float32(-np.inf)
        vmf[:self.n_lh, self.L_lh:] = NEG_INF        # LH-vert × RH-parcel
        vmf[self.n_lh:, :self.L_lh] = NEG_INF        # RH-vert × LH-parcel
        self._spatial_connect_vmf_f32 = vmf
        self._vmf_lh = vmf[:self.n_lh, :self.L_lh]
        self._vmf_rh = vmf[self.n_lh:, self.L_lh:]

        self._s_lambda_f32_buf = np.empty((N, L), dtype=np.float32)

    # ────────── Per-call API ──────────
    def compute(self,
                s_lambda: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full spatial_connect_prior pipeline (block-diagonal hot path).

        Parameters
        ----------
        s_lambda : (N, L) array, fp32 or fp64.
            Cross-hemi entries assumed zero (boundary_mask upstream).

        Returns
        -------
        u                    : (L, D) float32 view of cached buffer.
        spatial_connect_vmf  : (N, L) float32 view of cached buffer.
        """
        s_lam = self._stage_s_lambda(s_lambda)
        n_lh = self.n_lh
        L_lh = self.L_lh

        # ── Step 1 (block-diagonal sgemm + sum + divide) ──
        # u_update[:, :L_lh] = grad_lh.T @ s_lam[:n_lh, :L_lh]
        # u_update[:, L_lh:] = grad_rh.T @ s_lam[n_lh:, L_lh:]
        # Cross-hemi entries of s_lam are 0 (boundary_mask), so block-diagonal
        # is mathematically exact. ~2x FLOPS reduction on the dominant 5-GFlop
        # gemm.
        s_lam_lh = s_lam[:n_lh, :L_lh]
        s_lam_rh = s_lam[n_lh:, L_lh:]
        np.matmul(self._grad_lh.T, s_lam_lh, out=self._u_update_lh)
        np.matmul(self._grad_rh.T, s_lam_rh, out=self._u_update_rh)

        # sum_axis0 via numpy: a per-call ~1 ms ufunc on the (non-contig)
        # block, vs ~7 ms in our naive numba kernel. numpy's reduction
        # auto-dispatches to a SIMD-vectorized inner loop along the contig
        # axis (k) within each row. Critical correction: the previous
        # numba ``sum_axis0_f32`` on the non-C-contig slice forced a
        # scalar-loop — measured 7-8 ms per hemi (15 ms total / call).
        np.sum(s_lam_lh, axis=0, dtype=np.float32, out=self._sum_lam_lh)
        np.sum(s_lam_rh, axis=0, dtype=np.float32, out=self._sum_lam_rh)
        _kernels.divide_uupdate_f32(self._u_update_lh, self._sum_lam_lh, self._u_lh)
        _kernels.divide_uupdate_f32(self._u_update_rh, self._sum_lam_rh, self._u_rh)

        # ── Step 2 (block-diagonal cross term written to vmf diagonal blocks) ──
        # ``cross[:n_lh, :L_lh] = grad_lh @ u_lh.T``  (sliced output).
        # Cross-hemi blocks of vmf stay -Inf (set once at __init__).
        # Per-bench the sliced sgemm dispatches via sgemm with ldc=L (no copy);
        # ~2x FLOPS reduction + skips writing the cross-hemi block (~50 MB).
        np.matmul(self._grad_lh, self._u_lh.T, out=self._vmf_lh)
        np.matmul(self._grad_rh, self._u_rh.T, out=self._vmf_rh)

        # ── Step 3 (per-row u_sq_norms) ──
        _kernels.row_sq_norms_f32(self._u_f32, self._u_sq_norms)

        # ── Step 4 (FULL contig assemble — auto-vectorized) ──
        # In-place over the (N, L) C-contig buffer. Cross-hemi cells were
        # pre-filled with -Inf at __init__; the full kernel re-asserts the
        # mask per-call. Two explicit sub-loops (``_diag2`` variant) measure
        # within noise of the full kernel here — branch prediction on the
        # cross-hemi check costs ~0 once warm, and the full kernel keeps
        # the code path simple. Diag-2 is kept in ``_kernels`` for the
        # case where pre-fill alone needs to hold (no per-call -Inf write).
        _kernels.assemble_connect_vmf(
            self._spatial_connect_vmf_f32,
            self._grad_sq_norms,
            self._u_sq_norms,
            n_lh, L_lh,
        )
        return self._u_f32, self._spatial_connect_vmf_f32

    # ────────── Internal ──────────
    def _stage_s_lambda(self, s_lambda: np.ndarray) -> np.ndarray:
        """Get an ``(N, L)`` float32 contiguous view of ``s_lambda``."""
        if s_lambda.shape != (self.N, self.L):
            raise ValueError(
                f"s_lambda shape mismatch: expected ({self.N}, {self.L}), "
                f"got {s_lambda.shape}"
            )
        if s_lambda.dtype == np.float32 and (
            s_lambda.flags["C_CONTIGUOUS"] or s_lambda.flags["F_CONTIGUOUS"]
        ):
            return s_lambda
        if s_lambda.dtype == np.float64 and s_lambda.flags["C_CONTIGUOUS"]:
            _kernels.cast_f64_to_f32_2d(s_lambda, self._s_lambda_f32_buf)
            return self._s_lambda_f32_buf
        np.copyto(self._s_lambda_f32_buf,
                  np.ascontiguousarray(s_lambda, dtype=np.float32))
        return self._s_lambda_f32_buf
