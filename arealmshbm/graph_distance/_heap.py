"""_heap.py

Indexed binary min-heap for numba Dijkstra loops (numba has no heapq),
stored in flat caller-owned arrays so every thread keeps its own heap:

    heap_node : int32 (N,)  — vertex at each heap slot
    heap_key  : (N,)        — key at each heap slot (fp32 or fp64; numba
                              specialises per dtype, compares and moves only)
    heap_pos  : int32 (N,)  — heap slot of each vertex, -1 when absent

Decrease-key goes through the ``heap_pos`` back-reference. Used by
:func:`arealmshbm.graph_distance.graph_distance._all_pairs_dijkstra`,
:func:`arealmshbm.surface_smoothing._geodesic_kernels.bounded_dijkstra_smooth`
and :func:`arealmshbm.radius_mask._kernels.add_spatial_constraint_kernel` /
:func:`arealmshbm.radius_mask._kernels.central_sulcus_kernel`.

numba's ``cache=True`` keys a caller on its own file's mtime/size and
bytecode, not on this module's: after editing this file, clear the
callers' ``__pycache__/*.nbi`` / ``*.nbc``. The same holds for every
cross-module ``@njit`` helper (``em_stop_criterion._cdln``,
``m_step._invad``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numba as nb


@nb.njit(cache=True, fastmath=False, boundscheck=False, inline="always")
def _heap_push(heap_node, heap_key, heap_pos, heap_size, node, key):
    """Push ``(node, key)``; returns the new size."""
    i = heap_size
    heap_node[i] = node
    heap_key[i] = key
    heap_pos[node] = i
    while i > 0:
        parent = (i - 1) >> 1
        if heap_key[parent] > heap_key[i]:
            pn = heap_node[parent]
            pk = heap_key[parent]
            heap_node[parent] = heap_node[i]
            heap_key[parent] = heap_key[i]
            heap_node[i] = pn
            heap_key[i] = pk
            heap_pos[heap_node[parent]] = parent
            heap_pos[heap_node[i]] = i
            i = parent
        else:
            break
    return heap_size + 1


@nb.njit(cache=True, fastmath=False, boundscheck=False, inline="always")
def _heap_pop(heap_node, heap_key, heap_pos, heap_size):
    """Remove the minimum (the caller has already read ``heap_node[0]`` /
    ``heap_key[0]``); returns the new size."""
    last = heap_size - 1
    heap_pos[heap_node[0]] = -1
    if last == 0:
        return 0
    heap_node[0] = heap_node[last]
    heap_key[0] = heap_key[last]
    heap_pos[heap_node[0]] = 0
    new_size = last
    i = 0
    while True:
        l = 2 * i + 1
        r = 2 * i + 2
        smallest = i
        if l < new_size and heap_key[l] < heap_key[smallest]:
            smallest = l
        if r < new_size and heap_key[r] < heap_key[smallest]:
            smallest = r
        if smallest == i:
            break
        sn = heap_node[smallest]
        sk = heap_key[smallest]
        heap_node[smallest] = heap_node[i]
        heap_key[smallest] = heap_key[i]
        heap_node[i] = sn
        heap_key[i] = sk
        heap_pos[heap_node[smallest]] = smallest
        heap_pos[heap_node[i]] = i
        i = smallest
    return new_size


@nb.njit(cache=True, fastmath=False, boundscheck=False, inline="always")
def _heap_decrease(heap_node, heap_key, heap_pos, idx_in_heap, new_key):
    """Lower the key at heap slot ``idx_in_heap`` to ``new_key``."""
    i = idx_in_heap
    heap_key[i] = new_key
    while i > 0:
        parent = (i - 1) >> 1
        if heap_key[parent] > heap_key[i]:
            pn = heap_node[parent]
            pk = heap_key[parent]
            heap_node[parent] = heap_node[i]
            heap_key[parent] = heap_key[i]
            heap_node[i] = pn
            heap_key[i] = pk
            heap_pos[heap_node[parent]] = parent
            heap_pos[heap_node[i]] = i
            i = parent
        else:
            break
