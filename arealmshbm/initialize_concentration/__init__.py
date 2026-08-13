"""initialize_concentration

vMF concentration parameter init: solve ``|I_{D/2-1}(λ)| = 1e10`` for λ.

Public API:
    initialize_concentration(D) -> float — initial λ for feature dimension D.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .initialize_concentration import initialize_concentration

__all__ = ["initialize_concentration"]
