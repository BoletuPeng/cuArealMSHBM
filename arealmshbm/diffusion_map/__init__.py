"""diffusion_map

Diffusion-map embedding of a precomputed (dis)similarity matrix.
Port of ``mapalign.embed.compute_diffusion_map(alpha=0.5,
n_components=100, skip_checks=True, overwrite=True)`` with the
canonical ``A = exp(-D/D.max())`` distance → affinity transform
prepended.

Public API:
    compute_diffusion_map_repaired      — CPU, production path
    compute_diffusion_map_gpu_repaired  — GPU mirror, production path

The two ``_repaired`` wrappers prepend the distance → affinity
transform and then delegate to ``compute_diffusion_map`` /
``compute_diffusion_map_gpu`` (defined in :mod:`.diffusion_map` /
:mod:`.diffusion_map_gpu`), which implement the same
partial-Lanczos algorithm on the symmetric similar matrix; the GPU
mirror runs the matvecs on cuBLAS. The step-0 pipeline routes the
repaired variants by default (selected via ``Step0Config.backend``).
The repair: CBIG fed a normalised *distance* matrix into the
diffusion map, which expects an *affinity*; the production path
prepends the canonical ``A = exp(-D / D.max())`` transform first.

Precision policy:
    CPU path runs fp64 internally (scipy eigsh is fp64-only for the
    Lanczos workspace anyway). GPU path runs **fp32** by default —
    the affinity matrix ``exp(-D/D.max())`` is bounded in (0, 1] so
    the alpha-normalised symmetric similar matrix has well-controlled
    dynamic range; on the fsa6 down-sphere (N=12962) every one of the
    top 100 diffusion components matches the fp64 reference to
    |cos| = 1.0000. ``working_dtype=cp.float64`` is still selectable
    for audit. The returned ``emb`` is cast to fp32 in both cases to
    match the MATLAB GT artifact.

Sign ambiguity:
    Eigenvectors are defined up to a sign: ``(lambda, psi)`` and
    ``(lambda, -psi)`` are equally valid. Per-component |cosine|
    is the right comparator vs any external reference.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .diffusion_map_repaired import compute_diffusion_map_repaired
from .diffusion_map_gpu_repaired import compute_diffusion_map_gpu_repaired


__all__ = [
    "compute_diffusion_map_repaired",
    "compute_diffusion_map_gpu_repaired",
]
