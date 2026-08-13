"""surface_smoothing

Numba port of HCP Workbench ``wb_command -cifti-smoothing`` for the
surface portion of CIFTI dtseries data.

Public API:
    prepare_smoothing_mesh — per-hemi mesh-only precomputation (vertex
        areas + Layer 1/2 polyhedral-geodesic neighbor CSR tables).
        Step0Pipeline builds one of these per hemi at ``load_inputs``
        and threads them into every ``cifti_smoothing`` call.
    cifti_smoothing — geodesic Gaussian smoothing per hemisphere.

Algorithm — polyhedral geodesic. Mirrors
``src/Files/MetricSmoothingObject.cxx`` GEO_GAUSS_AREA branch (ROI
variant) and ``src/Files/GeodesicHelper.cxx`` for the geodesic-distance
oracle:

    1. Scatter cortex-only data to a full-mesh layout (medial → 0).
    2. For each hemisphere independently:
         * Build a per-vertex area array (1/3 sum of incident face
           areas — Workbench ``SurfaceFile::computeNodeAreas``).
         * Build per-vertex 1-ring neighbors with euclidean edge
           lengths (layer 1) AND across-face "unfolded" neighbor pairs
           with their straight-line unfolded distances (layer 2),
           mirroring ``GeodesicHelperBase`` ctor at
           GeodesicHelper.cxx:62-164. Layer 2 is what makes the
           propagation produce true polyhedral geodesics (paths that
           cross triangle interiors) rather than discrete edge-graph
           paths.
         * For each cortex source vertex i, run modified bounded
           Dijkstra (limit ``3 * sigma``) over BOTH neighbor layers,
           mirroring ``GeodesicHelper::dijkstra(root, maxdist, ...,
           smooth=true)`` at GeodesicHelper.cxx:217-293. Record
           reachable vertices j with distance d_ij.
         * Compute scatter weights ``w_ij = exp(-d_ij^2/(2σ^2)) *
           area[j]``, sum over ALL reachable j (including outside ROI;
           lines 455-463 of MetricSmoothingObject.cxx), then rescale
           so ``sum_j w_ij = area[i]``. Drop outside-ROI targets from
           the gather list.
         * Transpose scatters into gathers: ``gather[j].nodes`` = all
           sources i that scattered into j.
    3. Smooth: ``out[j, k] = sum_i w_{i→j} f[i, k] / weightSum[j]``.
    4. Re-mask: drop medial entries → return (N_cortex, K).

Numerical precision policy: distance accumulation runs in fp64
(matches Workbench's `output[whichneigh] = tempf` semantics with float
accumulation across short paths to ~1e-7 at fsaverage6 scale).
The final per-target weighted mean and output cast to fp32 to match
the upstream ``avg_grads`` dtype.

Per-thread scratch footprint: the prange-over-sources parallelization
costs ~T × 8 × N_full × 8 bytes of fp64 scratch (T = nb.get_num_threads(),
N_full = per-hemi vertex count). For the canonical step-0 setup
(24 threads × fs6 hemi at ~81 920 verts) this is ~125 MB per hemi or
~250 MB total during ``cifti_smoothing``. Scales linearly with mesh
resolution — on fsaverage7 (~163 842 verts) it would roughly double.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .surface_smoothing import cifti_smoothing, prepare_smoothing_gather
from ._geodesic_kernels import prepare_smoothing_mesh

__all__ = ["cifti_smoothing", "prepare_smoothing_mesh",
           "prepare_smoothing_gather"]
