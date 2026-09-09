"""vmf_clustering_batch.py

The ⟨A⟩ subgraph of step 2 — the inter-region vMF clustering EM body.
Equivalent to MATLAB's ``vmf_clustering_batch`` subfunction in
``CBIG_ArealMSHBM_cdgMSHBM_estimate_group_priors_sequential.m`` (lines
425-676).

One control loop, :func:`_em_loop`, serves both Session families:

* :func:`vmf_clustering_batch` drives the CPU
  :class:`arealmshbm.step2_em_iter_master.Step2EmIterSession` (numba
  master kernel). The Session owns the fused kernel
  that runs every leaf of one outer-EM iter body (M-step inner
  while-loop + spatial_connect + per-subject E-step + normalize +
  theta); the wrapper refreshes the per-intra-EM-iter state on it,
  loops, and returns the updated Params delta.
* :func:`em_body_sparse` drives the ``gpu`` backend's
  ``Step2SparseSession`` (``docs/step2_sparse_design.md`` §4): no
  ``Params`` round trip — the Session keeps every bulk field on device,
  so the only host traffic per EM iter is the ``(S,)`` fp64 ``cost_S``.

Convergence (shared): per-subject ``|cost - prev| / |prev| <= eps`` for
ALL subjects, ``iter_em == 1`` never stops, cap ``max_iter_em``. MATLAB
uses absolute |.| around both sides because cost can be negative; we
mirror. The caller (``Step2Pipeline``) owns the Session lifecycle.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — typing only
    # Imported lazily/never at runtime: this module is on the import path
    # of every backend, and ``step2_em_iter_master`` drags in the numba
    # master kernels a sparse-GPU run never calls.
    from arealmshbm.step2_em_iter_master import Step2EmIterSession


# ─────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────
@dataclass
class VmfBatchResult:
    """The leaf-level outputs (the Params delta) plus diagnostics."""
    Params: Dict[str, np.ndarray]   # updated in-place but returned for clarity
    em_iters: int                   # outer-EM iters consumed
    m_iters_per_em: List[int]       # M-step inner-iter count per outer EM iter
    converged: bool                 # True iff all subjects converged within cap


@dataclass
class SparseBatchResult:
    """``VmfBatchResult`` without the ``Params`` slot.

    The sparse Session owns every bulk field on device, so there is no
    Params delta to hand back — only ``cost_S``, which the caller stores
    as ``Params['cost_em']`` for the save path (the on-device L16 reads
    the device copy, not this one).
    """
    cost_S: np.ndarray              # (S,) fp64 — per-subject cost, host copy
    em_iters: int                   # outer-EM iters consumed
    m_iters_per_em: List[int]       # M-step inner-iter count per outer EM iter
    converged: bool                 # True iff all subjects converged within cap


# ─────────────────────────────────────────────────────────────────────
# The shared loop
# ─────────────────────────────────────────────────────────────────────
def _em_loop(
    step: Callable[[], Tuple[int, np.ndarray]],
    *,
    n_subjects: int,
    max_iter_em: int,
    em_convergence_eps: float,
    verbose: bool,
) -> Tuple[np.ndarray, int, List[int], bool]:
    """Run ``step()`` (one outer-EM iteration; returns ``(m_iters,
    cost_S)`` with ``cost_S`` the ``(S,)`` fp64 per-subject cost) until
    every subject's cost rel-diff is within ``em_convergence_eps`` or
    ``max_iter_em`` is hit. Returns ``(cost_S, em_iters, m_iters_per_em,
    converged)``. ``n_subjects`` seeds the previous-cost vector, so a
    loop that never runs (``max_iter_em <= 0``, which the config
    validator rejects) still hands back an ``(S,)`` zero ``cost_S``.
    """
    prev_cost = np.zeros(int(n_subjects), dtype=np.float64)
    m_iters_per_em: List[int] = []
    converged = False
    iter_em = 0
    update_cost = prev_cost

    for iter_em in range(1, max_iter_em + 1):
        t_iter = time.perf_counter()
        m_iters, update_cost = step()
        m_iters_per_em.append(int(m_iters))

        if iter_em > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(np.abs(update_cost - prev_cost) / prev_cost)
            stop_em = int((rel <= em_convergence_eps).sum()) == int(update_cost.size)
        else:
            stop_em = False

        if verbose:
            dur = time.perf_counter() - t_iter
            print(
                f"    EM iter {iter_em:3d}  m_iters={int(m_iters):2d}  "
                f"cost={float(update_cost.sum()):+.4e}  ({dur:.1f}s)",
                flush=True,
            )

        prev_cost = update_cost.copy()
        if stop_em:
            converged = True
            break

    return update_cost, iter_em, m_iters_per_em, converged


# ─────────────────────────────────────────────────────────────────────
# Dense backends ('cpu' / 'gpu')
# ─────────────────────────────────────────────────────────────────────
def vmf_clustering_batch(
    Params: Dict[str, np.ndarray],
    *,
    sess: "Step2EmIterSession",
    max_iter_em: int = 100,
    em_convergence_eps: float = 1e-4,
    verbose: bool = True,
) -> VmfBatchResult:
    """Run the EM body for one intra-EM iteration via the fused master.

    Mutates and returns Params with updated kappa, s_t_nu, s_lambda,
    theta, cost_em.
    """
    # Refresh the mutable per-intra-EM-iter state on the caller-owned
    # Session (BOLD/grad/boundary_mask + per-iter scratch buffers stay
    # untouched across intra-EM iters).
    sess.refresh_s_psi_sigma(Params["s_psi"], Params["sigma"])

    def _step():
        m_iters, _cost_total = sess.run_iter(Params)
        return m_iters, sess.get_cost_S()

    cost_S, em_iters, m_iters_per_em, converged = _em_loop(
        _step, n_subjects=sess.S, max_iter_em=max_iter_em,
        em_convergence_eps=em_convergence_eps, verbose=verbose,
    )

    Params["cost_em"] = cost_S
    return VmfBatchResult(
        Params=Params,
        em_iters=em_iters,
        m_iters_per_em=m_iters_per_em,
        converged=converged,
    )


# ─────────────────────────────────────────────────────────────────────
# GPU ('gpu') backend
# ─────────────────────────────────────────────────────────────────────
def em_body_sparse(
    sess: Any,
    *,
    max_iter_em: int = 100,
    em_convergence_eps: float = 1e-4,
    verbose: bool = True,
) -> SparseBatchResult:
    """The EM body for one intra-EM iteration on the sparse backend.

    Same loop as :func:`vmf_clustering_batch`, driven through the §4
    Session API instead of the dense duck-typed one:

    * no ``refresh_s_psi_sigma``: ``sigma`` / ``s_psi`` never leave the
      device, so the reset the caller did (``sess.reset_inter()``) is
      already visible;
    * ``sess.run_iter()`` takes no ``Params`` and returns
      ``(m_iters, cost_S)`` directly, so there is no ``get_cost_S()``
      round trip.
    """
    def _step():
        m_iters, cost_S = sess.run_iter()
        return m_iters, np.asarray(cost_S, dtype=np.float64)

    cost_S, em_iters, m_iters_per_em, converged = _em_loop(
        _step, n_subjects=sess.S, max_iter_em=max_iter_em,
        em_convergence_eps=em_convergence_eps, verbose=verbose,
    )
    return SparseBatchResult(
        cost_S=cost_S,
        em_iters=em_iters,
        m_iters_per_em=m_iters_per_em,
        converged=converged,
    )
