"""vmf_clustering_batch.py

The ⟨A⟩ subgraph of step 2 — the inter-region vMF clustering EM body.
Equivalent to MATLAB's ``vmf_clustering_batch`` subfunction in
``CBIG_ArealMSHBM_cdgMSHBM_estimate_group_priors_sequential.m`` (lines
425-676).

This is a thin Python wrapper around
:class:`arealmshbm.step2_em_iter_master.Step2EmIterSession`. The
Session owns the fused numba master kernel that fuses every leaf in
one outer-EM iter body (M-step inner while-loop + spatial_connect +
per-subject E-step + normalize + theta). One outer-EM iter wall at
fsaverage6 / S=3 / T=2 / L=400 drops from ~688 s (the legacy naive
numpy path) to ~5 s with the fused master.

The wrapper refreshes the mutable per-intra-EM-iter state on the
caller-owned Session, loops up to ``max_iter_em`` outer iters,
checks the per-subject cost rel-diff convergence between iters, and
returns a :class:`VmfBatchResult` with the updated Params delta plus
diagnostics. The caller (``Step2Pipeline.run_em``) owns the Session
lifecycle.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from arealmshbm.step2_em_iter_master import Step2EmIterSession


# ─────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────
@dataclass
class VmfBatchResult:
    """The leaf-level outputs (the Params delta) plus diagnostics."""
    Params: Dict[str, np.ndarray]   # updated in-place but returned for clarity
    em_iters: int                   # outer-EM iters consumed
    m_iters_per_em: List[int]       # M-step inner-iter count per outer EM iter
    converged: bool                 # True iff all subjects converged within cap


# ─────────────────────────────────────────────────────────────────────
# Subgraph supercall
# ─────────────────────────────────────────────────────────────────────
def vmf_clustering_batch(
    Params: Dict[str, np.ndarray],
    *,
    sess: Step2EmIterSession,
    max_iter_em: int = 100,
    em_convergence_eps: float = 1e-4,
    verbose: bool = True,
) -> VmfBatchResult:
    """Run the EM body for one intra-EM iteration via the fused master.

    Mutates and returns Params with updated kappa, s_t_nu, s_lambda,
    theta, cost_em.
    """
    # Internal layout: s_lambda is (S, N, L).
    S = Params["s_lambda"].shape[0]

    # Refresh the mutable per-intra-EM-iter state on the caller-owned
    # Session (BOLD/grad/boundary_mask + per-iter scratch buffers stay
    # untouched across intra-EM iters).
    sess.refresh_s_psi_sigma(Params["s_psi"], Params["sigma"])

    prev_cost = np.zeros(S, dtype=np.float64)
    m_iters_per_em: List[int] = []
    converged = False
    iter_em = 0
    update_cost = prev_cost

    for iter_em in range(1, max_iter_em + 1):
        t_iter = time.perf_counter()
        m_iters, cost_total = sess.run_iter(Params)
        m_iters_per_em.append(m_iters)
        update_cost = sess.get_cost_S()

        # Per-subject rel-diff convergence: ``|cost - prev| / |prev| ≤ eps``
        # for ALL subjects. MATLAB uses absolute |.| around both sides
        # because cost can be negative; we mirror.
        if iter_em > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(np.abs(update_cost - prev_cost) / prev_cost)
            per_sub_ok = np.where(rel <= em_convergence_eps, 1, 0)
            stop_em = (per_sub_ok.sum() == S)
        else:
            stop_em = False

        if verbose:
            dur = time.perf_counter() - t_iter
            print(
                f"    EM iter {iter_em:3d}  m_iters={m_iters:2d}  "
                f"cost={cost_total:+.4e}  ({dur:.1f}s)",
                flush=True,
            )

        prev_cost = update_cost.copy()
        if stop_em:
            converged = True
            break

    Params["cost_em"] = update_cost
    # Sync the device → host state that downstream outer-EM leaves
    # (intra_subject_var_loop / intra_em_cost_step2) will read. On the
    # CPU Session this is a no-op (Params['s_t_nu'] is already aliased
    # to the Session-owned buffer). On the GPU Session this D2Hs the
    # ~11 MB s_t_nu slab — the only bulk traffic per outer-EM iter.
    # theta is NOT in this set (it's only read at save time, once per
    # pipeline; Step2Pipeline.run_em syncs it explicitly before save).
    sess.sync_to_host(Params, fields=("s_t_nu",))
    return VmfBatchResult(
        Params=Params,
        em_iters=iter_em,
        m_iters_per_em=m_iters_per_em,
        converged=converged,
    )
