"""ini_params_gpu.py

GPU supercall for the step-1 vMF init-parameter leaf. Profile reads
stay CPU (bitpacked ``.b2nd`` via blosc2 LZ4 + bitshuffle, then a
single uint8→fp64 widen pass); once H2D'd, everything up to and
including the ``inner = profile @ mtc`` cuBLAS dgemm runs on device:

    H2D concat profile (fp64) → zero MW + detect keep → row demean +
    L2-unit-norm (RawKernel) → host-side label merge / renumber / mask
    → one-hot ``mtc = profile.T @ one_hot`` GEMM → column-renorm →
    ``inner = profile @ mtc`` GEMM → ``epsil_input`` gather-sum → host
    scalar → invAd (CPU scipy Bessel) → savemat on host.

invAd stays CPU — D=1175 fp64 root solve on a single scalar, ~ms; not
worth a GPU port.

cupy import lives at module top — only loaded via the lazy import
inside :mod:`.ini_params`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np
import scipy.io as sio

from ._invad import invAd
from ._kernels_gpu import (
    demean_l2norm_inplace_cupy,
    epsil_input_cupy,
    groupsum_cupy,
    zero_mw_and_detect_nonzero_cupy,
)
from .ini_params import (
    _concat_precomputed_avg,
    _load_medial_mask,
    _read_profile_concat,
    _renumber_labels,
)


def generate_ini_params_gpu(
    seed_mesh: str,
    targ_mesh: str,
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
    out_dir: str,
    *,
    profile_dtype: np.dtype = np.float64,
    reduction_dtype: np.dtype = np.float64,
    save: bool = True,
    precomputed_lh_avg: Optional[np.ndarray] = None,
    precomputed_rh_avg: Optional[np.ndarray] = None,
) -> dict:
    """GPU port of :func:`ini_params.generate_ini_params`. Same outputs
    (mtc, epsil, lambda, lh_labels, rh_labels) on host; the heavy
    arithmetic is device-resident.

    ``precomputed_lh_avg`` / ``precomputed_rh_avg``: when both set,
    skip the .npy disk read and use these in-memory arrays. See the
    CPU :func:`generate_ini_params` doc for the contract.
    """
    lh_labels = np.asarray(lh_labels, dtype=np.int64).ravel()
    rh_labels = np.asarray(rh_labels, dtype=np.int64).ravel()
    n_lh = lh_labels.size
    n_rh = rh_labels.size

    medial_mask = _load_medial_mask(targ_mesh)
    if medial_mask.shape[0] != n_lh + n_rh:
        raise ValueError(
            f"medial_mask size ({medial_mask.shape[0]}) != lh+rh labels "
            f"({n_lh + n_rh})"
        )
    lh_labels = lh_labels.copy()
    rh_labels = rh_labels.copy()
    lh_labels[medial_mask[:n_lh]] = 0
    rh_labels[medial_mask[n_lh:]] = 0

    if precomputed_lh_avg is not None and precomputed_rh_avg is not None:
        profile_host = _concat_precomputed_avg(
            precomputed_lh_avg, precomputed_rh_avg, n_lh, n_rh, profile_dtype,
        )
    elif precomputed_lh_avg is None and precomputed_rh_avg is None:
        # CPU read + concat at requested profile_dtype.
        profile_host = _read_profile_concat(
            out_dir, targ_mesh, seed_mesh, profile_dtype,
        )
    else:
        raise ValueError(
            "generate_ini_params_gpu: must pass BOTH precomputed_lh_avg "
            "and precomputed_rh_avg (or neither)."
        )
    n_total, d_orig = profile_host.shape
    if n_total != n_lh + n_rh:
        raise ValueError(
            f"profile_mat first axis ({n_total}) != lh+rh ({n_lh + n_rh})"
        )
    # Promote storage to reduction_dtype so the demean+L2 RawKernel's
    # double* pointer is valid. Production has both at fp64 -> no copy.
    if profile_host.dtype != reduction_dtype:
        profile_host = profile_host.astype(reduction_dtype, copy=False)

    # H2D — profile + medial mask.
    profile_dev = cp.asarray(np.ascontiguousarray(profile_host))
    del profile_host
    medial_dev = cp.asarray(medial_mask)

    keep_dev = zero_mw_and_detect_nonzero_cupy(profile_dev, medial_dev)
    n_keep = int(keep_dev.sum().item())

    demean_l2norm_inplace_cupy(profile_dev, keep_dev)

    # Label merge / renumber / mask is small + host-side (cheap; the
    # control-flow walk in _renumber_labels is hard to express well on GPU).
    max_lh = int(lh_labels.max()) if lh_labels.size else 0
    rh_labels_offset = rh_labels.copy()
    rh_labels_offset[rh_labels_offset != 0] += max_lh
    labels_host = _renumber_labels(
        np.concatenate([lh_labels, rh_labels_offset]).astype(np.int64)
    )
    keep_host = cp.asnumpy(keep_dev)
    labels_host[~keep_host] = 0
    L = int(labels_host.max()) if labels_host.size else 0
    if L == 0:
        raise RuntimeError("ini_params_gpu: no non-zero labels — empty parcellation")
    labels_dev = cp.asarray(labels_host)

    # mtc (D, L) — fp64 cuBLAS dgemm via dense one-hot.
    mtc_dev = cp.empty((d_orig, L), dtype=reduction_dtype)
    groupsum_cupy(profile_dev, labels_dev, L, mtc_dev)
    col_norm = cp.sqrt((mtc_dev * mtc_dev).sum(axis=0, keepdims=True))
    col_norm = cp.where(col_norm == 0, mtc_dev.dtype.type(1.0), col_norm)
    mtc_dev /= col_norm

    # inner = profile @ mtc, then gather-sum to get epsil input scalar.
    inner_dev = profile_dev @ mtc_dev
    epsil_sum = epsil_input_cupy(inner_dev, labels_dev)
    epsil_input = float(epsil_sum) / float(n_keep)
    epsil = invAd(d_orig - 1, epsil_input)

    # Materialize dense one-hot lambda on host (small uint8, no point on GPU).
    keep_idx = np.flatnonzero(keep_host)
    lam = np.zeros((n_keep, L), dtype=np.uint8)
    keep_labels = labels_host[keep_idx]
    has_label = keep_labels > 0
    lam[np.flatnonzero(has_label), keep_labels[has_label] - 1] = 1

    # D2H mtc once at the end.
    mtc_host = cp.asnumpy(mtc_dev).astype(np.float64, copy=False)

    out = {
        "mtc": mtc_host,
        "epsil": np.array([[epsil]], dtype=np.float64),
        "lambda": lam,
        "lh_labels": lh_labels.reshape(-1, 1),
        "rh_labels": rh_labels_offset.reshape(-1, 1),
    }
    if save:
        out_path = os.path.join(out_dir, "group", "group.mat")
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        sio.savemat(out_path, out, do_compression=True, format="5")
    return out
