"""local_minima_gpu.py

CuPy RawKernel port of :func:`.local_minima.find_minima`. Replaces the
CPU numba prange leaf so the step-0 subgraph-A device-stay chain
(``cifti_smoothing_gpu`` → minima → ``watershed_edge_count_gpu``) does
not have to D2H+H2D the (N_cortex, K) fp32 smoothed buffer around the
former CPU call.

Algorithm
---------
Per-column local minimum on a K-NN graph: vertex ``i`` is a local
minimum on column ``k`` iff ``data[i, k]`` is strictly less than
``data[nb, k]`` for every K-hop neighbor ``nb`` of ``i``. Padding
slots (no neighbor at that position) are skipped — equivalent to the
CPU implementation's ``+inf`` sentinel.

Layout
------
Grid : ceil(N * K / 256)   — one thread per (vertex, column) pair
Block: (256,)              — tuned for fsa6 / K=200; bound by
                             ``data`` global-memory bandwidth.

Precision / semantics contract vs CPU
-------------------------------------
Bit-equivalent to the CPU reference. The comparison is a single
``>=`` between two fp32 values per neighbor — no accumulation, no
rounding decisions to drift on. Tested in
``arealmshbm/local_minima/tests/test_gpu_correctness.py``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import Union

import numpy as np
import cupy as cp


_KERNEL_SRC = r"""
extern "C" __global__
void find_minima_kernel(
    const float* __restrict__ data,    // (N, K) fp32, row-major
    const int*   __restrict__ nbors,   // (N, M) int32, -1 padded, 0-indexed
    const int N,
    const int K,
    const int M,
    unsigned char* __restrict__ out    // (N, K) uint8 (interpreted as bool)
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int i = idx / K;
    int k = idx - i * K;

    float v_ik = data[i * K + k];
    unsigned char is_min = 1;

    // Sequential scan over the (compact, -1-padded) neighbor list. M is
    // small for K-hop on fsa6 (~30-60 entries), so unrolled register
    // reads dominate over any clever prefetch. Early-out on first
    // greater-or-equal neighbor.
    int base = i * M;
    for (int m = 0; m < M; ++m) {
        int nb = nbors[base + m];
        if (nb < 0) continue;            // padding sentinel
        float v_nb_k = data[nb * K + k];
        if (v_ik >= v_nb_k) {
            is_min = 0;
            break;
        }
    }
    out[i * K + k] = is_min;
}
"""

_FIND_MINIMA_KERNEL = cp.RawKernel(_KERNEL_SRC, "find_minima_kernel")


def prepare_K_neighbors_for_gpu(K_neighbors: np.ndarray) -> cp.ndarray:
    """One-shot conversion of the fp64 NaN-padded 1-indexed
    ``K_neighbors`` table into a ``(N, M-1) int32`` 0-indexed,
    ``-1``-padded device array suitable for repeated
    :func:`find_minima_gpu` calls.

    The fsa6 step-0 pipeline calls find_minima ``iter_a=3..4`` times
    per subject against the SAME ``ti.K_neighbors``; hoist this prep
    out of the per-iter_a hot loop (see
    ``arealmshbm.step0_pipeline.pipeline._subgraph_A``) so we pay the
    NaN→int + H2D once per subject.

    Parameters
    ----------
    K_neighbors : (N, M) fp64
        Per-vertex neighbor table. Column 0 is self (1-indexed cortex
        ID); columns 1..M-1 hold K-hop neighbor IDs (1-indexed; NaN for
        absent slots). Same layout as the CPU ``find_minima``'s second
        arg.

    Returns
    -------
    nbors_int_d : (N, M-1) int32 on device
        Self column dropped; ``-1`` marks absent slots.
    """
    # Drop column 0 (self), matching the CPU loop's ``range(1, n_cols)``.
    # NaN sentinel is replaced with -1 in fp64 BEFORE the int32 cast —
    # float→int on NaN is implementation-defined at the C level (numpy
    # typically yields INT_MIN but it's not spec'd). Doing np.where in
    # float space, then casting, keeps every astype input well-defined.
    nb_1idx = np.ascontiguousarray(K_neighbors[:, 1:])
    nan_mask = np.isnan(nb_1idx)
    nb_int = np.where(nan_mask, -1, nb_1idx - 1).astype(np.int32)
    # ``np.where(...).astype(...)`` is C-contiguous (numpy default), and
    # ``cp.asarray`` of a contiguous host array stays contiguous — no
    # explicit ascontiguousarray wrap needed.
    return cp.asarray(nb_int)


def find_minima_gpu(data: Union[np.ndarray, cp.ndarray],
                    K_neighbors_int_d: cp.ndarray) -> cp.ndarray:
    """GPU twin of :func:`.local_minima.find_minima`.

    Parameters
    ----------
    data : (N, K) fp32, host or device
        Smoothed gradient metric stack. Production caller passes a
        device cp.ndarray from ``cifti_smoothing_gpu``; ``cp.asarray``
        is then a no-op alias.
    K_neighbors_int_d : (N, M-1) int32 device
        Output of :func:`prepare_K_neighbors_for_gpu`. Self column
        dropped; ``-1`` marks absent slots.

    Returns
    -------
    minima : (N, K) bool device
        ``True`` where the vertex is strictly less than every K-hop
        neighbor on that column. Output stays on device for the
        downstream ``watershed_edge_count_gpu`` consumer.
    """
    data_d = cp.asarray(data, dtype=cp.float32)
    if data_d.ndim != 2:
        raise ValueError(f"data must be 2-D (N, K); got {data_d.shape}")
    if K_neighbors_int_d.ndim != 2:
        raise ValueError(
            f"K_neighbors_int_d must be 2-D (N, M-1); "
            f"got {K_neighbors_int_d.shape}")
    if data_d.shape[0] != K_neighbors_int_d.shape[0]:
        raise ValueError(
            f"data N ({data_d.shape[0]}) != nbors N "
            f"({K_neighbors_int_d.shape[0]})")
    if K_neighbors_int_d.dtype != cp.int32:
        raise TypeError(
            f"K_neighbors_int_d must be int32; got {K_neighbors_int_d.dtype}")

    N, K = int(data_d.shape[0]), int(data_d.shape[1])
    M = int(K_neighbors_int_d.shape[1])

    # Output: (N, K) uint8 viewed as bool. uint8 storage matches the
    # numba CPU output's bool memory layout (1 byte / cell).
    out_d = cp.empty((N, K), dtype=cp.uint8)

    threads = 256
    total = N * K
    blocks = (total + threads - 1) // threads
    _FIND_MINIMA_KERNEL(
        (blocks,), (threads,),
        (data_d, K_neighbors_int_d,
         np.int32(N), np.int32(K), np.int32(M),
         out_d),
    )
    return out_d.view(cp.bool_)
