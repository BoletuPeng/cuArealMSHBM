"""m_step

M-step inner-while loop body of the EM (Mode A, single-subject).

Public API:
    MStepSession — caches per-call scratch; takes ``data_series_NTD``
                   (N, T, D) by reference.
    warmup       — pre-compile every numba kernel in this module.
                   Idempotent; useful before benchmarking.

External shapes (S axis dropped — Mode A always has S=1):
    data_series_NTD : (N, T, D) fp32 — caller-owned, shared by reference
                      across the super-call's ELambda + EMStop + MStep
                      sub-Sessions (single BOLD copy per pipeline).
    s_t_nu          : (D, L, T) fp32
    s_lambda        : (N, L)    fp32
    s_psi           : (D, L)    fp32

Internal layout: ``s_t_nu_TDL`` per-t slice (D, L) is C-contig; the BLAS
sgemm fast path uses the strided per-t view ``data_series_NTD[:, t, :]``
(N, D) (lda = T·D). MKL handles the lda — empirically <1% slower than a
dedicated (T, N, D) C-contig copy.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .m_step import MStepSession


def warmup() -> None:
    """Pre-compile every numba kernel in this module with realistic
    dtypes / shapes. ~50 ms one-shot."""
    from . import _kernels
    _kernels.warmup()


__all__ = [
    "MStepSession",
    "warmup",
]
