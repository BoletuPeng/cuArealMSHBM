"""v_lambda.py

V_lambda — the Potts close-form MRF smoothness term in the gMSHBM
E-step.

Public API:
    Session — built once with ``(neighborhood, row_idx, col_idx, K,
              row_idx_active)``; per-call entry
              ``session.compute_full(full_lam) -> V_lam``. All buffers
              cached; no per-call allocation, no caller-side gather (the
              kernel reads the full ``(N_full, K)`` lam directly via
              ``row_idx_active`` indirection).

Potts edge weights (``V_same=0, V_diff=1``, the MATLAB driver's fixed
choice) are baked into the kernel; the Session takes no edge-weight
matrices. Non-Potts weights are not supported.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np

from . import _kernels


class Session:
    """Pre-allocated state for the V_lambda EM hot path (Potts only).

    Construction: validates inputs, transposes ``neighborhood`` for
    cache-friendly access, builds candidate-set bookkeeping, allocates
    all per-call scratch + output buffers in fp32. ``V_lam`` zeroed
    exactly once; non-candidate entries stay 0 forever (the gMSHBM EM's
    candidate set is static across all calls).

    Per call: :meth:`compute_full` runs the fused 3-phase Potts kernel
    on the FULL ``(N_full, K)`` lam directly. Returns a view into
    ``self.V_lam_f32``; the next call overwrites it.
    """

    __slots__ = (
        "N", "K", "M1", "P",
        "neighborhood_NM",   # int64 (N_active, M1) — fast-path layout
        "row_idx", "col_idx",
        "row_idx_active",    # int64 (N_active,) — full-vertex id per active row
        # fp32 fast-path buffers (all owned, contiguous)
        "row_sum_f32",       # (N_active,)
        "nbr_sum_per_m",     # (N_active,)
        "V_lam_f32",         # (N_active, K) — pre-zeroed once; only candidate
                             # entries written by the kernel
    )

    def __init__(self,
                 neighborhood: np.ndarray,
                 row_idx: np.ndarray,
                 col_idx: np.ndarray,
                 K: int,
                 row_idx_active: np.ndarray):
        nbh_MN = np.ascontiguousarray(neighborhood, dtype=np.int64)
        if nbh_MN.ndim != 2:
            raise ValueError(f"neighborhood must be 2D, got {nbh_MN.shape}")
        M1, N = nbh_MN.shape
        if not 1 <= M1 <= 16:
            raise ValueError(
                f"neighborhood first dim looks wrong (M1={M1}); "
                f"expected MATLAB-shape (M1, N) with M1 ~= 6"
            )

        self.N = int(N)
        self.K = int(K)
        self.M1 = int(M1)
        self.neighborhood_NM = np.ascontiguousarray(nbh_MN.T)

        rows = np.ascontiguousarray(row_idx, dtype=np.int64).ravel()
        cols = np.ascontiguousarray(col_idx, dtype=np.int64).ravel()
        if rows.shape != cols.shape:
            raise ValueError("row_idx / col_idx must be same length")
        if rows.size > 0:
            # Symmetric range checks. With ``boundscheck=False`` on the
            # numba kernels, a negative index would silently wrap and
            # segfault rather than raise — so check both ends here.
            if int(rows.min()) < 0:
                raise ValueError("row_idx contains negative entries")
            if int(cols.min()) < 0:
                raise ValueError("col_idx contains negative entries")
            if int(rows.max()) >= N:
                raise ValueError("row_idx out of range vs N")
            if int(cols.max()) >= K:
                raise ValueError("col_idx out of range vs K")
        self.row_idx = rows
        self.col_idx = cols
        self.P = int(rows.size)

        # Active-row → full-vertex id table — required by the kernel.
        ria = np.ascontiguousarray(row_idx_active, dtype=np.int64).ravel()
        if ria.size != N:
            raise ValueError(
                f"row_idx_active length {ria.size} != N (M_active) {N}"
            )
        # Symmetric range check (matches the row_idx / col_idx checks
        # above). With ``boundscheck=False`` on the numba kernels, a
        # negative entry would silently wrap to the last full_lam row;
        # the upper-bound check happens in compute_full() against the
        # caller's full_lam.
        if ria.size > 0 and int(ria.min()) < 0:
            raise ValueError("row_idx_active contains negative entries")
        self.row_idx_active = ria

        # fp32 buffers. ``V_lam_f32`` zeroed exactly once: the candidate
        # set is static across all EM calls (pipeline_setup builds it
        # once before the EM enters its main loop and never modifies it).
        # The kernel only writes at candidate (m, k) cells; non-candidate
        # entries stay 0 forever.
        self.row_sum_f32   = np.empty(N, dtype=np.float32)
        self.nbr_sum_per_m = np.empty(N, dtype=np.float32)
        self.V_lam_f32     = np.zeros((N, K), dtype=np.float32)

    # ────────── Per-call API ──────────
    def compute_full(self, full_lam: np.ndarray) -> np.ndarray:
        """Run the fused Potts close-form kernel on the FULL lam.

        Parameters
        ----------
        full_lam : (N_full, K) float32 C-contiguous.
            Full soft posterior (active + inactive rows). Must be fp32
            and contiguous — strict to avoid hidden allocation; the
            super-call only ever passes its ping-pong buffer here.

            INVARIANT: ``full_lam`` must be a row-stochastic-or-zero
            soft posterior — ``Σ_zj full_lam[m, zj] ∈ {0, 1}`` for every
            m. The close-form Potts kernel folds the row-sum into the
            row / neighbor-sum reductions; if a caller routes V_lambda
            BEFORE the softmax (or with un-normalized rows for any
            reason), the kernel returns silently wrong answers — no
            NaN, no crash. The only call site (``e_step_lambda``'s
            λ-loop body) guarantees the invariant by construction.

        Returns
        -------
        V_lam : (N_active, K) float32 view into the session's cached
            buffer. The next ``compute_full()`` call overwrites it.
            Copy if you need to keep it.
        """
        if not (full_lam.dtype == np.float32 and full_lam.flags["C_CONTIGUOUS"]):
            raise ValueError(
                f"compute_full() requires fp32 C-contig full_lam; got "
                f"dtype={full_lam.dtype}, "
                f"C_CONTIG={full_lam.flags['C_CONTIGUOUS']}"
            )
        if full_lam.ndim != 2 or full_lam.shape[1] != self.K:
            raise ValueError(
                f"full_lam shape {full_lam.shape} incompatible with K={self.K}"
            )
        if int(self.row_idx_active.max(initial=-1)) >= full_lam.shape[0]:
            raise ValueError(
                f"row_idx_active points past full_lam: "
                f"max={int(self.row_idx_active.max())}, "
                f"N_full={full_lam.shape[0]}"
            )

        _kernels.v_lambda_potts_closeform_fused_full_lam_f32(
            self.neighborhood_NM, full_lam, self.row_idx_active,
            self.row_idx, self.col_idx, self.V_lam_f32,
            self.row_sum_f32, self.nbr_sum_per_m,
        )
        return self.V_lam_f32
