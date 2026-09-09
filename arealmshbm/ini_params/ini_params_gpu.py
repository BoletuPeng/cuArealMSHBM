"""ini_params_gpu.py

GPU supercall for the step-1 vMF init-parameter leaf.

Two ways in:

  * **device-resident** — the caller hands the fp32 ``(V_h, D)`` avg
    profiles that ``avg_profiles_from_packed_gpu`` left on device
    (``precomputed_lh_avg_dev`` / ``precomputed_rh_avg_dev``). Nothing
    round-trips through host memory: the lh/rh concat and the fp32→fp64
    widen happen in one device allocation.
  * **host** — the historical path: ``.npy`` read (or in-memory host
    arrays via ``precomputed_{lh,rh}_avg``), host concat, one H2D.

From there everything up to the ε scalar is device-resident::

    concat+widen → zero MW + detect keep → row demean + L2-unit-norm
    (RawKernel) → host-side label merge / renumber / mask → parcel CSR
    → ordered fp64 groupsum (RawKernel) → column L2-renorm (RawKernel)
    → fused row-dot ε reduction (RawKernel) → host scalar → invAd
    (CPU scipy Bessel) → savemat (optionally on a background thread).

Neither reduction goes through a GEMM. ``mtc`` is a CSR-ordered fp64
reduction, bit-exact against the numba CPU kernel **given the same
fp64 profile matrix** (identical ascending-row summation order); the
saved ``group.mat`` still differs between backends by ~1e-15 because
``demean_l2norm_inplace`` is not CPU-bit-exact. ``epsil`` is a per-row
dot against that row's own parcel column, never the full ``(N, L)``
product.

``invAd`` stays CPU — a D=1175 fp64 root solve on a single scalar.

cupy import lives at module top — only loaded via the lazy import
inside :mod:`.ini_params`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Optional

import cupy as cp
import numpy as np

from ._group_mat_writer import IniParamsResult, write_group_mat
from ._invad import invAd
from ._kernels_gpu import (
    build_parcel_csr_cupy,
    colnorm_scale_cupy,
    demean_l2norm_inplace_cupy,
    epsil_input_rowdot_cupy,
    groupsum_csr_cupy,
    zero_mw_and_detect_nonzero_cupy,
)
from .ini_params import (
    _concat_precomputed_avg,
    _load_medial_mask,
    _read_profile_concat,
    _renumber_labels,
)


def _concat_precomputed_avg_device(
    lh_avg_dev, rh_avg_dev, n_lh: int, n_rh: int,
    profile_dtype, reduction_dtype,
):
    """Concat the device ``(V_h, D)`` avg profiles into ``(N, D)`` at
    ``reduction_dtype``, on device.

    Cast semantics match the host path exactly. The host path does
    ``source → profile_dtype → reduction_dtype``; when those two dtypes
    are equal the composition collapses to a single cast, which is what
    production (fp64 / fp64) takes. When they differ we materialise the
    ``profile_dtype`` intermediate so a deliberately lossy
    ``profile_dtype`` still rounds at the same place.
    """
    for name, a in (("lh", lh_avg_dev), ("rh", rh_avg_dev)):
        if not isinstance(a, cp.ndarray):
            raise ValueError(
                f"precomputed_{name}_avg_dev must be a cupy ndarray; got "
                f"{type(a).__name__}"
            )
        if a.ndim != 2:
            raise ValueError(
                f"precomputed_{name}_avg_dev must be 2-D (V_h, D); got "
                f"shape {a.shape}"
            )
        if a.dtype not in (cp.float32, cp.float64):
            raise ValueError(
                f"precomputed_{name}_avg_dev must be fp32 or fp64; got "
                f"{a.dtype}"
            )
    if lh_avg_dev.shape[0] != n_lh:
        raise ValueError(
            f"precomputed_lh_avg_dev has V={lh_avg_dev.shape[0]} but labels "
            f"imply V_lh={n_lh}"
        )
    if rh_avg_dev.shape[0] != n_rh:
        raise ValueError(
            f"precomputed_rh_avg_dev has V={rh_avg_dev.shape[0]} but labels "
            f"imply V_rh={n_rh}"
        )
    if lh_avg_dev.shape[1] != rh_avg_dev.shape[1]:
        raise ValueError(
            f"precomputed device lh/rh D mismatch: {lh_avg_dev.shape[1]} vs "
            f"{rh_avg_dev.shape[1]}"
        )
    D = int(lh_avg_dev.shape[1])
    p_dt = np.dtype(profile_dtype)
    r_dt = np.dtype(reduction_dtype)

    def _stack(dtype):
        out = cp.empty((n_lh + n_rh, D), dtype=dtype)
        out[:n_lh] = lh_avg_dev
        out[n_lh:] = rh_avg_dev
        return out

    if p_dt == r_dt:
        return _stack(r_dt)
    mid = _stack(p_dt)
    out = mid.astype(r_dt)
    del mid
    return out


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
    precomputed_lh_avg_dev=None,
    precomputed_rh_avg_dev=None,
    save_async: bool = False,
) -> IniParamsResult:
    """GPU port of :func:`ini_params.generate_ini_params`. Same outputs
    (mtc, epsil, lambda, lh_labels, rh_labels) on host; the heavy
    arithmetic is device-resident.

    Profile source — exactly one of:
      * ``precomputed_lh_avg_dev`` + ``precomputed_rh_avg_dev``: device
        fp32/fp64 ``(V_h, D)`` arrays (``AvgProfilesResult.lh_avg_dev``
        / ``rh_avg_dev``). No host round trip at all.
      * ``precomputed_lh_avg`` + ``precomputed_rh_avg``: host arrays.
      * neither: read the ``.npy`` pair under
        ``out_dir/profiles/avg_profile/``.

    ``save_async``: submit the ``group.mat`` write to a background
    thread and return immediately; the handle is on ``result.writer``
    and the caller MUST ``.wait()`` on it.

    ``reduction_dtype`` must be fp64 on this backend — the demean/L2,
    groupsum, colnorm and ε RawKernels are all fp64. Use
    ``backend='cpu'`` for an fp32 reduction.
    """
    if np.dtype(reduction_dtype) != np.float64:
        raise ValueError(
            f"generate_ini_params_gpu: reduction_dtype must be float64 on "
            f"the GPU backend (the demean/groupsum/colnorm/epsil RawKernels "
            f"are fp64); got {np.dtype(reduction_dtype)}. Use backend='cpu' "
            f"for a lower-precision reduction."
        )
    have_dev = (precomputed_lh_avg_dev is not None
                and precomputed_rh_avg_dev is not None)
    if (precomputed_lh_avg_dev is None) != (precomputed_rh_avg_dev is None):
        raise ValueError(
            "generate_ini_params_gpu: must pass BOTH precomputed_lh_avg_dev "
            "and precomputed_rh_avg_dev (or neither)."
        )
    have_host = (precomputed_lh_avg is not None
                 and precomputed_rh_avg is not None)
    if (precomputed_lh_avg is None) != (precomputed_rh_avg is None):
        raise ValueError(
            "generate_ini_params_gpu: must pass BOTH precomputed_lh_avg "
            "and precomputed_rh_avg (or neither)."
        )
    if have_dev and have_host:
        raise ValueError(
            "generate_ini_params_gpu: pass either the device pair "
            "(precomputed_*_avg_dev) or the host pair (precomputed_*_avg), "
            "not both."
        )

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

    if have_dev:
        profile_dev = _concat_precomputed_avg_device(
            precomputed_lh_avg_dev, precomputed_rh_avg_dev,
            n_lh, n_rh, profile_dtype, reduction_dtype,
        )
        n_total, d_orig = profile_dev.shape
    else:
        if have_host:
            profile_host = _concat_precomputed_avg(
                precomputed_lh_avg, precomputed_rh_avg,
                n_lh, n_rh, profile_dtype,
            )
        else:
            profile_host = _read_profile_concat(
                out_dir, targ_mesh, seed_mesh, profile_dtype,
            )
        n_total, d_orig = profile_host.shape
        # Promote storage to reduction_dtype so the demean+L2 RawKernel's
        # double* pointer is valid. Production has both at fp64 -> no copy.
        if profile_host.dtype != reduction_dtype:
            profile_host = profile_host.astype(reduction_dtype, copy=False)
        profile_dev = cp.asarray(np.ascontiguousarray(profile_host))
        del profile_host
    if n_total != n_lh + n_rh:
        raise ValueError(
            f"profile_mat first axis ({n_total}) != lh+rh ({n_lh + n_rh})"
        )

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

    # Parcel CSR: rows grouped by label, ascending within each parcel —
    # the numba _groupsum_kernel's exact accumulation order.
    offsets_dev, rows_dev = build_parcel_csr_cupy(labels_dev, L)

    mtc_dev = cp.empty((d_orig, L), dtype=reduction_dtype)
    groupsum_csr_cupy(profile_dev, offsets_dev, rows_dev, L, mtc_dev)
    colnorm_scale_cupy(mtc_dev)

    epsil_sum = epsil_input_rowdot_cupy(profile_dev, mtc_dev, labels_dev)
    epsil_input = epsil_sum / float(n_keep)
    epsil = invAd(d_orig - 1, epsil_input)

    # Materialize dense one-hot lambda on host (small uint8, no point on GPU).
    keep_idx = np.flatnonzero(keep_host)
    lam = np.zeros((n_keep, L), dtype=np.uint8)
    keep_labels = labels_host[keep_idx]
    has_label = keep_labels > 0
    lam[np.flatnonzero(has_label), keep_labels[has_label] - 1] = 1

    # D2H mtc once at the end.
    mtc_host = cp.asnumpy(mtc_dev).astype(np.float64, copy=False)

    out = IniParamsResult({
        "mtc": mtc_host,
        "epsil": np.array([[epsil]], dtype=np.float64),
        "lambda": lam,
        "lh_labels": lh_labels.reshape(-1, 1),
        "rh_labels": rh_labels_offset.reshape(-1, 1),
    })
    if save:
        out.writer = write_group_mat(
            out_dir, out, compress=True, background=save_async,
        )
    return out
