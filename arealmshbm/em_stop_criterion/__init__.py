"""em_stop_criterion

EM stop-criterion block: cost reduction + convergence test that closes the
outer EM loop in :class:`vmf_clustering.VmfClusteringSession`.

Public API:
    EMStopSession         — caches per-call scratch.
    matlab_ratio_converged — shared MATLAB-faithful ratio convergence helper
                             (also used by the outer ``intra_em`` loop).
    cdln_general_to_f32   — generic-d log-vMF normalization via Debye
                             expansion; used here and by ``intra_em_cost``.
    warmup                — pre-compile every numba kernel. Idempotent.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from ._cdln import cdln_general_to_f32
from .em_stop_criterion import (
    EMStopSession,
    matlab_ratio_converged,
)


def warmup() -> None:
    """Pre-compile every numba kernel in this module. Idempotent."""
    from . import _cdln, _kernels
    _cdln.warmup()
    _kernels.warmup()


__all__ = [
    "EMStopSession",
    "matlab_ratio_converged",
    "cdln_general_to_f32",
    "warmup",
]
