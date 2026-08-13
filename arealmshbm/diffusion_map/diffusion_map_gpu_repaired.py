"""diffusion_map_gpu_repaired.py

GPU production diffusion-map embedding. Applies the canonical
distance → affinity transform ``A = exp(-D / D.max())`` and then runs
the alpha-normalisation + Lanczos pipeline.

The affinity transform itself is done **on the GPU** (host transform
on a 12962×12962 matrix is the dominant cost — measured 1.4 s out of
~1.7 s/hemi when host-side). Memory layout: the input fp32 distance
is pushed to device once, then ``exp(-D/dm)`` runs in-place. No fp64
working copy is materialised; see
:func:`compute_diffusion_map_gpu` for the precision rationale.

This mirrors :func:`compute_diffusion_map_repaired` (CPU) semantically
but the CPU path keeps the host transform because the matrix never
leaves host anyway.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from .diffusion_map_gpu import compute_diffusion_map_gpu

if TYPE_CHECKING:  # type-only import — cupy stays a soft dependency at runtime
    import cupy as cp


def compute_diffusion_map_gpu_repaired(
    dist: "Union[np.ndarray, cp.ndarray]",
    *,
    alpha: float = 0.5,
    n_components: int = 100,
    working_dtype=None,
) -> np.ndarray:
    """Repaired GPU diffusion-map embedding of a precomputed distance matrix.

    Accepts either a host ``np.ndarray`` or a device ``cp.ndarray``. The
    host case triggers a single H2D copy at ``cp.asarray(dist, dtype=…)``;
    the device case aliases (no copy) when the dtype already matches
    ``working_dtype``. Identical semantics to
    :func:`compute_diffusion_map_gpu`; the only difference is the
    prepended distance → affinity transform, done on the GPU side.
    """
    import cupy as cp

    if working_dtype is None:
        working_dtype = cp.float32

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"dist must be square 2D, got {dist.shape}")

    # H2D as working_dtype; if the caller already passed a cupy array
    # with the matching dtype this is an alias (no copy).
    L = cp.asarray(dist, dtype=working_dtype)
    dm = float(cp.max(L))
    if dm <= 0.0:
        raise ValueError(
            "diffusion_map_gpu_repaired: input distance matrix has "
            "D.max() <= 0; cannot construct a meaningful affinity matrix."
        )
    # In-place A = exp(-D / D.max()).
    # ``-1.0 / dm`` cast to working_dtype keeps the in-place multiply
    # type-stable; cp.exp(out=...) avoids a temporary allocation.
    L *= working_dtype(-1.0 / dm)
    cp.exp(L, out=L)

    return compute_diffusion_map_gpu(
        L,
        alpha=alpha,
        n_components=n_components,
        working_dtype=working_dtype,
    )
