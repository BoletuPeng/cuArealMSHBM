"""intra_em

Outer ``intra_em`` loop math: ``s_psi`` update + outer-loop convergence cost.

Public API:
    intra_subject_var — update ``s_psi`` from (s_t_nu, sigma, epsil, mu).
    intra_em_cost     — outer-loop convergence cost.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .intra_subject_var import intra_subject_var
from .intra_em_cost import intra_em_cost

__all__ = ["intra_subject_var", "intra_em_cost"]
