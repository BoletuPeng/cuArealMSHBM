"""check_connectedness

Connectedness check + per-parcel component distance, called inside the EM
body to decide which parcels need the spatial-xyz pull.

Public API:
    compute_components_general — BFS over parcel labels on the bilateral
                                 mesh; returns per-parcel component count.
    component_distance         — pairwise centroid distance between
                                 components inside each parcel.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .component_distance import (
    compute_components_general,
    component_distance,
)

__all__ = [
    "compute_components_general",
    "component_distance",
]
