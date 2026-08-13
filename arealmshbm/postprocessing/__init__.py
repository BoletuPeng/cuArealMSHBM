"""postprocessing

Surface-label post-processing leaves used by step 3.

Public API:
    remove_isolated_surface_components(lh_labels, rh_labels, lh_mesh,
                                       rh_mesh, abs_threshold=5)
        — relabel each connected component of vertices sharing a parcel
          label whose size is below ``abs_threshold``, by mode-vote of
          its neighbours' labels.

Used by the cMSHBM variant in two places:
    1. inside ``check_connectedness`` (per comp_iter), before the
       components / distance predicate, on the current argmax labels.
    2. as a final cleanup on ``lh_labels`` / ``rh_labels`` after the
       pipeline-level argmax.

Both invocations use ``abs_threshold`` configurable via
``Step3Config.cMSHBM_isolated_component_min_size`` (default 5). The
gMSHBM and dMSHBM variants do not invoke this leaf.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .remove_isolated import remove_isolated_surface_components

__all__ = ["remove_isolated_surface_components"]
