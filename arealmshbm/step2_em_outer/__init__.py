"""step2_em_outer

Outer EM closed-form update leaves for the Python port of CBIG MSHBM
step-2 (group prior estimation). Three leaves:

* ``intra_subject_var_loop`` (L17) — while-loop over (s_psi, sigma)
  updates with convergence flagging. Multi-subject form of step-3's
  ``intra_em.intra_subject_var``.
* ``inter_subject_var`` (L18) — closed-form group ``(mu, epsil)``
  update from the per-subject ``s_psi``.
* ``intra_em_cost_step2`` (L16) — outer-loop convergence cost
  (multi-subject form of step-3's ``intra_em.intra_em_cost``).

All three leaves dispatch to ``@njit`` kernels under
``_kernels.py`` (fp32 storage / fp64 reduction accumulator,
d-axis unit-stride innermost loops). ``invAd`` / ``Cdln`` are
called fp64 from inside the kernels via the shared CPU helpers
``arealmshbm.m_step._invad`` and
``arealmshbm.em_stop_criterion._cdln``.

Also re-exports two in-place ``@njit`` reset kernels used by the
outer pipeline (``reset_s_t_nu_from_mtc_STLD``,
``reset_s_psi_from_mtc_SLD``) and the JIT-warmup entry point.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .intra_subject_var_loop import intra_subject_var_loop
from .inter_subject_var import inter_subject_var
from .intra_em_cost import intra_em_cost_step2
from ._kernels import (
    reset_s_t_nu_from_mtc_STLD,
    reset_s_psi_from_mtc_SLD,
    warmup_step2_em_outer,
)

__all__ = [
    "intra_subject_var_loop",
    "inter_subject_var",
    "intra_em_cost_step2",
    "reset_s_t_nu_from_mtc_STLD",
    "reset_s_psi_from_mtc_SLD",
    "warmup_step2_em_outer",
]
