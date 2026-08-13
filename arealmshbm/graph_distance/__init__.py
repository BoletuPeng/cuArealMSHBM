"""graph_distance

All-pairs geodesic distance on a triangle-mesh graph weighted by
node-attached gradient values. Port of CBIG
``CBIG_SPGrad_create_graph.m`` + MATLAB's ``distances(G)``.

Used by step-0 subgraph B to turn the down-sampled gradient edge
density into a (12962, 12962) per-hemi distance matrix that step-0
subgraph C feeds into the diffusion map.

Public API:
    gradient_geodesic_distance(verts, vertex_nbors, grad_data)
        -> (N, N) fp32 in [0, 1]

External shapes:
    verts        : (N, 3) fp32 — sphere vertex coordinates (used only
                    for shape consistency / future graph debugging).
    vertex_nbors : (N, max_neigh) int32 1-indexed (``0`` = absent).
    grad_data    : (N,) fp32 — gradient density at each vertex.
    output       : (N, N) fp32 — symmetric, zero diagonal, normalised
                    so ``output.max() == 1`` (matches MATLAB
                    ``lh_dist = single(lh_dist / max(max(abs(lh_dist))))``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .graph_distance import gradient_geodesic_distance


def __getattr__(name):
    # Lazy import for the GPU variants so ``cupy`` stays a soft
    # dependency on CPU-only environments.
    if name in ("gradient_geodesic_distance_gpu",
                "gradient_geodesic_distance_gpu_device"):
        from .graph_distance_gpu import (
            gradient_geodesic_distance_gpu,
            gradient_geodesic_distance_gpu_device,
        )
        return {
            "gradient_geodesic_distance_gpu": gradient_geodesic_distance_gpu,
            "gradient_geodesic_distance_gpu_device":
                gradient_geodesic_distance_gpu_device,
        }[name]
    raise AttributeError(name)


__all__ = [
    "gradient_geodesic_distance",
    "gradient_geodesic_distance_gpu",
    "gradient_geodesic_distance_gpu_device",
]
