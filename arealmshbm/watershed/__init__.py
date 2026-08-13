"""watershed

Threshold-sweep watershed flooding on a triangulated mesh. Port of
WashU ``watershed_algorithm_all_par_cifti.m`` with the K-mod-4
column-drop bug repaired (the upstream loop's 4-way chunking had a
dead ``else`` that silently skipped the trailing ``K mod 4`` columns,
putting a uniform ``+3/375 ≈ 0.008`` floor on ``edge_density``).

For each metric column, vertices flagged as local minima seed
catchments; thresholds are swept from ``minh`` to ``maxh*fracmaxh``
in ``stepnum`` linear steps. At every threshold, every below-
threshold unlabeled vertex is visited in a random order: if its
ring-1 neighbors carry a single nonzero label, the vertex joins
that catchment; if they carry two or more distinct nonzero labels,
the vertex becomes a permanent watershed boundary (label 0). The
visitation order is random by design — different runs produce
different catchment ID assignments and slightly different boundary
masks. The downstream consumer
(``edge_density = sum(label == 0, axis=1) / num_sample_S``) is
robust to this perturbation, hence the agreement bar is on the
boundary mask rather than the label values.

Public API:
    watershed_algorithm_repaired(edge_metrics, minima, neighbors,
                                 stepnum, fracmaxh) -> (N, K) int32

The CPU symbol is a numba ``@njit(parallel=True, cache=True)``
dispatcher; the K-column loop is parallelised via ``prange`` and
each iteration seeds its own RNG state with the column index, so
two Python runs against identical inputs produce byte-identical
labels.

This leaf is **not bit-equivalent** to MATLAB. The MATLAB driver
uses ``randn + sort`` for label assignment and ``randperm`` for
vertex visitation order; we use a per-column ``np.random.seed`` /
``np.random.shuffle`` (legacy API, numba-supported) seeded by the
column index. Deterministic across Python runs but does not match
MATLAB's twister state. The downstream consumer averages over
``block_a_size`` columns so the boundary noise cancels — see
``docs/step0_flow_and_subgraphs.md`` for the algorithm-class
equivalence note.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .watershed_repaired import watershed_algorithm_repaired


__all__ = [
    "watershed_algorithm_repaired",
    "watershed_edge_count_gpu",
    "prepare_neighbors_for_gpu",
]


def __getattr__(name):
    """Lazy import of GPU symbols so the cupy import only fires when
    backend=gpu is actually selected; CPU users never pay it.
    """
    if name == "watershed_edge_count_gpu":
        from .watershed_gpu import watershed_edge_count_gpu
        return watershed_edge_count_gpu
    if name == "prepare_neighbors_for_gpu":
        from .watershed_gpu import prepare_neighbors_for_gpu
        return prepare_neighbors_for_gpu
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
