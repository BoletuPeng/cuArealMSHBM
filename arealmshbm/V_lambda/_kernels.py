"""_kernels.py

Numba kernel for V_lambda_Product_with_theta (Potts close-form).

A single kernel. Operates on the full ``(N_full, K)`` lam directly via
``row_idx_active`` indirection, so the caller (e_step_lambda's λ-loop)
never has to gather a ``(M_active, K)`` intermediate.

Layout convention:
    ``neighborhood_NM`` is stored as ``(N_active, M1)`` — the transpose
    of the MATLAB-side ``(M1, N)`` layout. The inner 6-neighbor loop
    reads M1=6 neighbors of vertex m at fixed m, so (N, M1) C-contiguous
    puts those 6 entries on a single 48-byte cache line. The transpose
    is done once per Session.

Algorithmic shortcut:
    Exploits ``V_same==0, V_diff==1`` AND row-normalization of ``lam``
    (``Σ_zj lam[m, zj] ∈ {0, 1}`` for all m at every λ-iteration, since
    the caller renormalizes via softmax). Each per-(m, k) inner sum
    collapses to ~10 ops via

        V_lam[m, k] = (Σ_{n,j>0} row_sum[j-1]) - (Σ_{n,j>0} lam[j-1, k])
                    = nbr_sum_per_m[m]        - Σ_{n,j>0} lam[j-1, k]

    Result is ULP-level agreement with the MATLAB MEX (not bit-equal —
    different reduction order, fp32 vs fp64).

Buffer reference:
    N_active  : active vertex count (rows of masked s_lambda; sub-001 ~74,947)
    N_full    : total vertex count (sub-001 fsaverage6: 81,924)
    M1        : max neighbors per vertex (6 on fsaverage6)
    K         : number of clusters (Mode A: 300)
    P         : candidate (m, k) pairs (= nnz(masked theta); sub-001 ~251k)

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=False, boundscheck=False)
def v_lambda_potts_closeform_fused_full_lam_f32(
    neighborhood_NM, full_lam, row_idx_active,
    row_idx, col_idx, V_lam,
    row_sum_scratch, nbr_sum_scratch,
):
    """3-phase fused close-form V_lambda Potts kernel.

    Operates on the FULL ``(N_full, K)`` lam directly — for each
    active-row index ``i``, the full vertex id is ``row_idx_active[i]``
    and the corresponding row of ``full_lam`` is read at that index.
    No caller-side gather required.

    Phases (single JIT region):
      Phase 1 (per-active-row row-sum):
          row_sum_scratch[i] = Σ_k full_lam[row_idx_active[i], k]
      Phase 2 (per-active-vertex neighbor-sum):
          nbr_sum_scratch[m] = Σ_n: j>0 row_sum_scratch[j-1]
      Phase 3 (per-candidate close-form):
          V_lam[m, k] = nbr_sum_scratch[m]
                        - Σ_n: j>0 full_lam[row_idx_active[j-1], k]

    Reduction order: outer per-vertex (or per-candidate) loops in
    ascending order; inner reductions in ascending k / ascending n.
    Indirection through ``row_idx_active`` only changes which entries
    in ``full_lam`` are read, not the summation order over those
    entries.

    Caller responsibilities:
      * ``V_lam`` candidate-set entries are overwritten; non-candidate
        entries are NOT touched (Session pre-zeros once at construction
        per the static-candidate-set invariant — the gMSHBM EM never
        changes the candidate set after pipeline_setup).
      * ``row_sum_scratch`` / ``nbr_sum_scratch`` are pure scratch.

    Inputs:
        neighborhood_NM   : (N_active, M1) int64 read-only — j entries
                             are M_active-space (1-indexed; 0 = absent)
        full_lam          : (N_full, K)    fp32   read-only — full soft
                             posterior (active and inactive rows)
        row_idx_active    : (N_active,)    int64  read-only — maps
                             active-row id ``i`` to full vertex id
        row_idx           : (P,)           int64  read-only
        col_idx           : (P,)           int64  read-only
    Outputs (caller-provided):
        V_lam             : (N_active, K)  fp32   rewritten at candidates
        row_sum_scratch   : (N_active,)    fp32   scratch
        nbr_sum_scratch   : (N_active,)    fp32   scratch
    """
    N_active, M1 = neighborhood_NM.shape
    K = full_lam.shape[1]
    P = row_idx.shape[0]

    # Phase 1: row_sums per active row, reading full_lam at the indirected
    # vertex id.
    for i in range(N_active):
        n_full = row_idx_active[i]
        s = np.float32(0.0)
        for k in range(K):
            s += full_lam[n_full, k]
        row_sum_scratch[i] = s

    # Phase 2: nbr_sum_per_m operates entirely on the active-space
    # row_sum_scratch.
    for m in range(N_active):
        s = np.float32(0.0)
        for n in range(M1):
            j = neighborhood_NM[m, n]
            if j != 0:
                s += row_sum_scratch[j - 1]
        nbr_sum_scratch[m] = s

    # Phase 3: per-candidate close-form. Neighbor read indirects through
    # row_idx_active to land in full_lam.
    for p in range(P):
        m = row_idx[p]
        k = col_idx[p]
        acc_lam = np.float32(0.0)
        for n in range(M1):
            j = neighborhood_NM[m, n]
            if j != 0:
                n_full = row_idx_active[j - 1]
                acc_lam += full_lam[n_full, k]
        V_lam[m, k] = nbr_sum_scratch[m] - acc_lam


# ─────────────────────────────────────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────────────────────────────────────
def warmup() -> None:
    """Compile the kernel once with realistic dtypes/shapes. ~30 ms one-shot."""
    M1, N_active, K, P = 6, 4, 3, 2
    N_full = N_active + 2

    nbh_NM = np.zeros((N_active, M1), dtype=np.int64)
    nbh_NM[0, 0] = 2
    full_lam = np.full((N_full, K), 1.0 / K, dtype=np.float32)
    row_idx_active = np.array([1, 0, N_full - 1, 2], dtype=np.int64)
    rows = np.array([0, 1], dtype=np.int64)
    cols = np.array([0, 1], dtype=np.int64)
    V_lam = np.zeros((N_active, K), dtype=np.float32)
    row_sum = np.empty(N_active, dtype=np.float32)
    nbr_sum = np.empty(N_active, dtype=np.float32)
    v_lambda_potts_closeform_fused_full_lam_f32(
        nbh_NM, full_lam, row_idx_active,
        rows, cols, V_lam, row_sum, nbr_sum,
    )
