"""vmf_clustering

The full EM body of step 3 — one outer call corresponds to one ``intra_em``
iteration. Composes the leaf modules into a single Session.

Public API:
    VmfClusteringSession — caches all sub-Sessions and per-call scratch.
                           ``backend='cpu' | 'gpu_elambda' | 'gpu_full'``.
    warmup               — pre-compile every numba kernel reachable from
                           the super-call. Idempotent; useful before
                           benchmarking to avoid JIT cold-start cost.

Composes:
    m_step              — M-step inner-while loop body.
    spatial_priors      — spatial_xyz_prior + spatial_connect_prior.
    V_lambda            — Potts close-form MRF smoothness term.
    check_connectedness — connectedness BFS + component distance.
    em_stop_criterion   — EM stop criterion.
    e_step_lambda       — local E-step λ-loop body (in this package).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .vmf_clustering import VmfClusteringSession


def warmup() -> None:
    """Pre-compile every numba kernel reachable from the super-call.

    Calls ``warmup()`` on each composed module (m_step, spatial_priors,
    em_stop_criterion, V_lambda kernels, check_connectedness kernels,
    plus this package's e_step_lambda kernels). Idempotent.
    """
    from arealmshbm.m_step import warmup as _m_warmup
    from arealmshbm.spatial_priors import warmup as _sp_warmup
    from arealmshbm.em_stop_criterion import warmup as _es_warmup
    from arealmshbm.V_lambda import _kernels as _vl_kernels
    from arealmshbm.check_connectedness import _kernels as _cc_kernels
    from . import _kernels as _local_kernels

    _m_warmup()
    _sp_warmup()
    _es_warmup()
    _vl_kernels.warmup()
    _cc_kernels.warmup()
    _local_kernels.warmup()


__all__ = [
    "VmfClusteringSession",
    "warmup",
]
