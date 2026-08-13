"""surface_gradient

Pure-numpy port of HCP Workbench ``wb_command -cifti-gradient`` for the
surface portion of CIFTI dtseries data.

Public API:
    cifti_gradient — per-vertex gradient magnitude over a surface mesh.

Algorithm (mirrors ``src/Algorithms/AlgorithmMetricGradient.cxx`` from
the Workbench source — area-weighted tangent-plane LS with the
"unrolled" curvature correction, applied per hemisphere, per column):

    1. Scatter cortex-only data to a full-mesh layout where medial
       vertices hold zero, then split into lh / rh halves. Smoothing
       does not bleed across the medial wall or the inter-hemisphere
       boundary because each hemisphere's mesh is processed
       independently.
    2. For each vertex v:
         * Vertex normal n_v = normalized average of incident face
           normals (each face contributes once, no area weight, then
           the sum is renormalized — matches Workbench's
           ``computeNormals`` defaults).
         * Tangent basis (xhat, yhat) seeded from a permuted unit
           vector against n_v (Workbench convention — picks (0,1,0)
           or (1,0,0) depending on which component of n_v has larger
           magnitude).
         * For each 1-ring neighbor j of v, compute the 3D offset
           ``somevec = coord_j - coord_v``, then "unroll" its
           tangential length to ``unrollMag = origMag * asin(o/origMag)
           * origMag / o`` where ``o = somevec.dot(n_v)`` (curvature
           correction; falls back to origMag when |o|/origMag <= 0.035).
           Project onto (xhat, yhat) and rescale to ``unrollMag``.
         * Accumulate the area-weighted normal equations of the LS
           system ``[x_j, y_j, 1] · [a, b, c]^T = (f_j - f_v)`` with
           weight ``vertAreas[j]``. The center vertex itself is added
           only as ``myRegress[2][2] += vertAreas[v]`` (its tempf=0
           row contributes nothing else).
         * Solve the symmetric 3x3 system in fp64; gradient magnitude
           is ``|| xhat * a + yhat * b ||`` (a 3D vector length, equals
           sqrt(a^2 + b^2) since xhat,yhat are an orthonormal pair, but
           computing in 3D matches Workbench bit-for-bit).
    3. Re-mask: drop medial entries → return (N_cortex, K).

Numerical precision policy: per-vertex normal-equation accumulation and
the 3x3 solve run in fp64 (LS in fp32 shows visible noise on near-flat
columns). The output is cast to fp32 to match the upstream
``FC_simi_block`` dtype.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .surface_gradient import cifti_gradient, prepare_gradient_mesh

__all__ = ["cifti_gradient", "prepare_gradient_mesh"]
