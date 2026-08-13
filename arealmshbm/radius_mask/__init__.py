"""radius_mask

Step-1 leaf — per-parcel spatial-prior mask on a fsaverage* mesh.

For each parcel l: bounded multi-source SSSP from every vertex labeled
l on the area-rescaled inflated surface (≤ ``radius`` mm) → boolean
column for ``lh_boundary`` / ``rh_boundary``. Pre/post-central sulcus
is computed once per hemi (SSSP over union of verts in aparc==25 ∨ 23,
unbounded, used to derive mean parcel-to-sulcus distance) and used to
zero parcels whose distance is too small (``truncate_kernel``).

Backends (dispatched by the ``backend=`` kwarg on the supercall):
    cpu — numba per-parcel Dijkstra-with-binary-heap, ``prange`` over
          parcels / sources.
    gpu — batched pull-based Bellman-Ford on a (V, K) device distance
          matrix (one BF iter = one RawKernel pass over V*K cells,
          loop until ``changed`` flag is 0). Classify / truncate stay
          CPU (small, heavy control flow). cupy imported lazily.

Public API:
    generate_radius_mask(lh_labels, rh_labels, mesh, radius, out_dir, …,
                          backend='cpu' | 'gpu')

Output:
    <out_dir>/spatial_mask/spatial_mask_<mesh>.mat with
        ``lh_boundary``, ``rh_boundary`` — scipy.sparse.csc_matrix

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .radius_mask import generate_radius_mask

__all__ = ["generate_radius_mask"]
