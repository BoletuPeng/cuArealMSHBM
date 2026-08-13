"""step0_neighbors

Port of the two SPGrad neighbor utilities used by step 0:

    CBIG_SPGrad_neighbors_exclude_medial.m  →  neighbors_exclude_medial
    CBIG_SPGrad_find_neighbors.m            →  find_K_neighbors

Both work on the (N+1, K+1) "self-padded NaN-sentinel" neighbor table
that the SPGrad pipeline uses internally — column 0 is the self-vertex
ID, columns 1..K are ring-1 neighbors, NaN marks an absent slot or a
medial-wall neighbor that has been masked out.

The folder is named ``step0_neighbors`` (not just ``neighbors``) to
disambiguate from ``check_connectedness/`` which already lives in
``arealmshbm/`` and serves step 3.

Public API:
    neighbors_exclude_medial(orig_neighbors, medial_mask) -> reorg_neighbors
    find_K_neighbors(neighbors, K)                       -> K_neighbors

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .step0_neighbors import find_K_neighbors, neighbors_exclude_medial


__all__ = [
    "neighbors_exclude_medial",
    "find_K_neighbors",
]
