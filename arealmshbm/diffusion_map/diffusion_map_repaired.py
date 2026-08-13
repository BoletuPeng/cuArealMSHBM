"""diffusion_map_repaired.py

Production diffusion-map embedding. Applies the canonical distance →
affinity transform ``A = exp(-D / D.max())`` and then delegates to
:func:`.diffusion_map.compute_diffusion_map`, which runs the
mapalign-class alpha-normalisation + Lanczos pipeline on the
resulting affinity. Upstream CBIG fed a normalised *distance* matrix
straight into the mapalign function, which expects an *affinity* —
this wrapper inserts the missing transform.

The wrapper is kept in a separate file from the algorithm body so the
distinction between "mapalign-class core" and "the repair" is explicit
at every call site, and so a future audit of the upstream bug only
needs to touch this file.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np

from .diffusion_map import compute_diffusion_map


def _distance_to_affinity(dist: np.ndarray) -> np.ndarray:
    """Standard distance → affinity transform ``A = exp(-D / D.max())``.

    The single argument used by Margulies 2016 and the broader diffusion-
    map literature. Diagonal goes from 0 (self-distance) to 1 (self-
    affinity); the most distant pair goes from 1 to ``exp(-1) ≈ 0.368``.
    Output is non-negative, symmetric, and entrywise positive — the
    contract mapalign's ``compute_diffusion_map`` actually expects.
    """
    D = np.asarray(dist, dtype=np.float64)
    m = float(D.max())
    if m <= 0.0:
        # Pathological all-zero input — return the identity-like matrix
        # (every point self-similar, no relations). Downstream eigsh
        # would still work but the embedding would be degenerate;
        # surface a clear error instead.
        raise ValueError(
            "diffusion_map_repaired: input distance matrix has D.max() <= 0; "
            "cannot construct a meaningful affinity matrix."
        )
    return np.exp(-D / m)


def compute_diffusion_map_repaired(
    dist: np.ndarray,
    *,
    alpha: float = 0.5,
    n_components: int = 100,
) -> np.ndarray:
    """Repaired diffusion-map embedding of a precomputed distance matrix.

    Identical signature to :func:`compute_diffusion_map`; the only
    difference is the prepended distance → affinity transform.

    Parameters
    ----------
    dist : (N, N) ndarray
        Symmetric, non-negative DISTANCE matrix (CBIG step-0 normalises
        to ``[0, 1]`` so the diagonal is 0 and the most distant pair
        is 1). The function applies ``A = exp(-D / D.max())`` and
        forwards the result to the standard mapalign pipeline.
    alpha, n_components : see :func:`compute_diffusion_map`.

    Returns
    -------
    emb : (N, n_components) fp32
    """
    A = _distance_to_affinity(dist)
    return compute_diffusion_map(
        A,
        alpha=alpha,
        n_components=n_components,
    )
