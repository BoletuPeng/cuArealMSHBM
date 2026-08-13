"""local_minima

Per-column local-minima detection on a K-nearest-neighbor mesh
graph. Port of CBIG ``CBIG_SPGrad_find_minima.m``.

A vertex is a local minimum on a metric column iff it is strictly
less than every one of its K-hop graph neighbors on that column.
The MATLAB implementation appends a sentinel row of ``+inf`` to the
data so NaN-padded neighbor slots dereference safely (the sentinel
fails the strict-less-than test for every real vertex).

Public API:
    find_minima(data, K_neighbors) -> (N, K) bool

External shapes:
    data        : (N_cortex, K) fp32 — smoothed gradient stack.
    K_neighbors : (N_cortex, M) fp64 — col 0 is self (1-indexed
                  cortex IDs); cols 1..M-1 are K-hop neighbors;
                  NaN entries are padding past per-vertex degree.
    output      : (N_cortex, K) bool — True at local minima.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .local_minima import find_minima


def __getattr__(name):
    # Lazy import for the GPU variant so ``cupy`` stays a soft
    # dependency on CPU-only environments. Access pattern:
    # ``from arealmshbm.local_minima import find_minima_gpu`` triggers
    # the cupy import only at that line; CPU-only callers that only
    # ever touch :func:`find_minima` never load cupy. Deliberately NOT
    # listed in ``__all__`` so ``from ... import *`` doesn't walk the
    # GPU names and eagerly load cupy on CPU-only environments —
    # wildcard imports of this package stay CPU-clean.
    if name in ("find_minima_gpu", "prepare_K_neighbors_for_gpu"):
        from .local_minima_gpu import (
            find_minima_gpu, prepare_K_neighbors_for_gpu,
        )
        return {
            "find_minima_gpu": find_minima_gpu,
            "prepare_K_neighbors_for_gpu": prepare_K_neighbors_for_gpu,
        }[name]
    raise AttributeError(name)


__all__ = [
    "find_minima",
]
