"""surface_smoothing

Port of HCP Workbench ``wb_command -cifti-smoothing`` for the surface
portion of CIFTI dtseries data.

Public API:
    cifti_smoothing — geodesic Gaussian smoothing per hemisphere,
        applied from the per-hemi gather shipped in the step-0 inputs
        bundle (``Step0Inputs.{lh,rh}_smooth_gather``).

The gather encodes the polyhedral-geodesic Gaussian kernel of
``src/Files/MetricSmoothingObject.cxx`` (GEO_GAUSS_AREA branch, ROI
variant), with ``src/Files/GeodesicHelper.cxx`` as the geodesic-distance
oracle. Per hemisphere, on the midthickness mesh:

    * Vertex areas are 1/3 of the incident face areas (Workbench
      ``SurfaceFile::computeNodeAreas``).
    * Geodesic distances d_ij come from a modified Dijkstra bounded at
      ``3 * sigma`` over two neighbor layers — the 1-ring edges and the
      across-face "unfolded" pairs (``GeodesicHelperBase`` ctor,
      GeodesicHelper.cxx:62-164; ``GeodesicHelper::dijkstra(root,
      maxdist, ..., smooth=true)``, GeodesicHelper.cxx:217-293) — so
      paths cross triangle interiors rather than following the edge
      graph.
    * Each cortex source i scatters ``w_ij = exp(-d_ij^2/(2σ^2)) *
      area[j]``, summed over ALL reachable j (including outside the ROI;
      lines 455-463 of MetricSmoothingObject.cxx) and rescaled so
      ``sum_j w_ij = area[i]``; targets outside the ROI are dropped.
    * The scatters are transposed into gathers: row j of ``gather_W``
      holds every source i that scattered into j, and
      ``inv_weight_sum[j]`` is the reciprocal of that row's sum.

:func:`cifti_smoothing` computes:

    1. Scatter cortex-only data to a full-mesh layout (medial → 0).
    2. Per hemisphere: ``out[j, k] = sum_i w_{i→j} f[i, k] / weightSum[j]``.
    3. Re-mask: drop medial entries → return (N_cortex, K).

Numerical precision policy: the gather weights are fp64 and the
per-target weighted mean runs in fp64; the output is cast to fp32 to
match the upstream ``avg_grads`` dtype.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .surface_smoothing import cifti_smoothing

__all__ = ["cifti_smoothing"]
