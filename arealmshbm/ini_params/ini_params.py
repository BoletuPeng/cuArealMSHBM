"""ini_params.py

Supercall for the step-1 vMF init-parameter leaf.

CPU path: numba kernels in :mod:`._kernels` called directly.
Dispatch on ``backend='gpu'`` defers to :mod:`.ini_params_gpu` (cuBLAS
fp64 dgemm + RawKernel row-demean + CuPy scatter one-hot; invAd
stays CPU). cupy is imported lazily — the CPU path never touches it.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio

from ._invad import invAd
from ._kernels import (
    _zero_mw_and_detect_nonzero_kernel,
    _demean_l2norm_inplace_kernel,
    _groupsum_kernel,
    _epsil_input_kernel,
)


def _concat_precomputed_avg(lh_avg: np.ndarray, rh_avg: np.ndarray,
                             n_lh: int, n_rh: int,
                             profile_dtype: np.dtype) -> np.ndarray:
    """vstack the in-memory ``(V_h, D)`` arrays + dtype-cast.

    Validates shapes against the label-derived vertex counts so a
    mis-aligned hand-off fails early (before any reduction kernel).
    """
    if lh_avg.ndim != 2 or rh_avg.ndim != 2:
        raise ValueError(
            f"_concat_precomputed_avg: arrays must be 2-D (V_h, D); got "
            f"lh_avg.ndim={lh_avg.ndim} rh_avg.ndim={rh_avg.ndim}"
        )
    if lh_avg.shape[0] != n_lh:
        raise ValueError(
            f"precomputed_lh_avg has V={lh_avg.shape[0]} but labels imply V_lh={n_lh}"
        )
    if rh_avg.shape[0] != n_rh:
        raise ValueError(
            f"precomputed_rh_avg has V={rh_avg.shape[0]} but labels imply V_rh={n_rh}"
        )
    if lh_avg.shape[1] != rh_avg.shape[1]:
        raise ValueError(
            f"precomputed lh/rh D mismatch: {lh_avg.shape[1]} vs {rh_avg.shape[1]}"
        )
    profile = np.vstack([lh_avg, rh_avg])
    if profile_dtype != profile.dtype:
        profile = profile.astype(profile_dtype, copy=False)
    return profile


def _read_profile_concat(out_dir: str, targ_mesh: str, seed_mesh: str,
                          profile_dtype: np.dtype) -> np.ndarray:
    """Read lh+rh avg-profile .npy files in parallel and concat → (V, D).

    The .npy is already ``(V_h, D)`` fp32 C-contig as written by
    :mod:`avg_profiles` — concatenated directly along the V axis.
    """
    base = os.path.join(out_dir, "profiles", "avg_profile")
    lh_path = os.path.join(base, f"lh_{targ_mesh}_roi{seed_mesh}_avg_profile.npy")
    rh_path = os.path.join(base, f"rh_{targ_mesh}_roi{seed_mesh}_avg_profile.npy")
    if not (os.path.exists(lh_path) and os.path.exists(rh_path)):
        raise FileNotFoundError(
            f"avg profile not found: {lh_path} / {rh_path}"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_lh = pool.submit(np.load, lh_path, allow_pickle=False)
        f_rh = pool.submit(np.load, rh_path, allow_pickle=False)
        lh_vol = f_lh.result()
        rh_vol = f_rh.result()
    profile = np.vstack([lh_vol, rh_vol])
    if profile_dtype != np.float32:
        profile = profile.astype(profile_dtype, copy=False)
    return profile


def _load_medial_mask(targ_mesh: str) -> np.ndarray:
    """Concatenated lh+rh medial-wall mask (1=MW)."""
    if "fsaverage" not in targ_mesh:
        raise ValueError(
            f"ini_params: only fsaverage* targ_mesh supported, got {targ_mesh!r}."
        )
    from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
    lh = load_avg_mesh("lh", targ_mesh, "inflated")
    rh = load_avg_mesh("rh", targ_mesh, "inflated")
    return np.concatenate(
        [lh["MARS_label"] == 1, rh["MARS_label"] == 1]
    ).astype(bool)


def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    """Compress label ids to [1, n_unique] by decrementing every label
    above each gap in ``unique(labels)`` ∩ [1..max]. 0 stays as the MW
    sentinel and is excluded from the gap search.
    """
    out = labels.copy()
    if out.size == 0:
        return out
    max_lbl = int(out.max())
    present = set(int(x) for x in np.unique(out))
    for p in sorted(p for p in range(1, max_lbl + 1) if p not in present):
        out[out > p] -= 1
    return out


def generate_ini_params(
    seed_mesh: str,
    targ_mesh: str,
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
    out_dir: str,
    *,
    profile_dtype: np.dtype = np.float64,
    reduction_dtype: np.dtype = np.float64,
    save: bool = True,
    backend: str = "cpu",
    precomputed_lh_avg: Optional[np.ndarray] = None,
    precomputed_rh_avg: Optional[np.ndarray] = None,
) -> dict:
    """vMF initial parameters from a group-level parcellation.

    Parameters
    ----------
    seed_mesh, targ_mesh : str
        e.g. ``'fsaverage3'`` and ``'fsaverage6'``.
    lh_labels, rh_labels : (V_h,) int — group parcellation labels in
        [0, L_h], 0 = medial wall.
    out_dir : str — project root containing ``profiles/avg_profile/...``.
        Output written to ``<out_dir>/group/group.mat`` when ``save=True``.
    profile_dtype : on-load precision for the avg profile.
    reduction_dtype : precision for demean / normalize / matmul / sum.
    save : whether to write ``group.mat``.
    precomputed_lh_avg, precomputed_rh_avg : in-memory (V_h, D) fp32 (or
        ``profile_dtype``) arrays — when both are set, ini_params skips
        the .npy disk read under ``out_dir/profiles/avg_profile/`` and
        uses the arrays directly. The unified pipeline driver passes
        the just-computed ``AvgProfilesResult.lh_avg`` / ``rh_avg``
        here so step1 subgraph 3 reads from memory, not disk.

    Returns dict with keys ``mtc``, ``epsil``, ``lambda``, ``lh_labels``,
    ``rh_labels``.
    """
    if backend == "gpu":
        from .ini_params_gpu import generate_ini_params_gpu
        return generate_ini_params_gpu(
            seed_mesh=seed_mesh, targ_mesh=targ_mesh,
            lh_labels=lh_labels, rh_labels=rh_labels, out_dir=out_dir,
            profile_dtype=profile_dtype, reduction_dtype=reduction_dtype,
            save=save,
            precomputed_lh_avg=precomputed_lh_avg,
            precomputed_rh_avg=precomputed_rh_avg,
        )
    if backend != "cpu":
        raise ValueError(f"generate_ini_params: unknown backend {backend!r}")
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
        profile_mat = _concat_precomputed_avg(
            precomputed_lh_avg, precomputed_rh_avg, n_lh, n_rh, profile_dtype,
        )
    elif precomputed_lh_avg is None and precomputed_rh_avg is None:
        profile_mat = _read_profile_concat(out_dir, targ_mesh, seed_mesh, profile_dtype)
    else:
        raise ValueError(
            "generate_ini_params: must pass BOTH precomputed_lh_avg and "
            "precomputed_rh_avg (or neither). Mixing one in-memory and "
            "one from disk is not supported."
        )
    n_total, d_orig = profile_mat.shape
    if n_total != n_lh + n_rh:
        raise ValueError(
            f"profile_mat first axis ({n_total}) != lh+rh ({n_lh + n_rh})"
        )
    profile_mat = profile_mat.astype(reduction_dtype, copy=False)

    keep_mask = np.empty(profile_mat.shape[0], dtype=np.bool_)
    _zero_mw_and_detect_nonzero_kernel(profile_mat, medial_mask, keep_mask)
    n_keep = int(keep_mask.sum())

    _demean_l2norm_inplace_kernel(profile_mat, keep_mask)

    # Merge labels with rh offset by max(lh_labels); zero-out rows that
    # the keep-detector rejected so the kernels can use ``label > 0`` as
    # the unified "row contributes" guard.
    max_lh = int(lh_labels.max()) if lh_labels.size else 0
    rh_labels_offset = rh_labels.copy()
    rh_labels_offset[rh_labels_offset != 0] += max_lh
    labels = _renumber_labels(
        np.concatenate([lh_labels, rh_labels_offset]).astype(np.int64)
    )
    labels[~keep_mask] = 0
    L = int(labels.max()) if labels.size else 0
    if L == 0:
        raise RuntimeError("ini_params: no non-zero labels — empty parcellation")

    mtc = np.empty((d_orig, L), dtype=reduction_dtype)
    _groupsum_kernel(profile_mat, labels, L, mtc)
    col_norm = np.sqrt((mtc * mtc).sum(axis=0, keepdims=True))
    col_norm = np.where(col_norm == 0, 1.0, col_norm)
    mtc /= col_norm

    inner = profile_mat @ mtc
    epsil_input = float(_epsil_input_kernel(inner, labels) / n_keep)
    epsil = invAd(d_orig - 1, epsil_input)

    # Materialize the dense one-hot lambda for the saved group.mat
    # (the kernels above never needed it).
    keep_idx = np.flatnonzero(keep_mask)
    lam = np.zeros((n_keep, L), dtype=np.uint8)
    keep_labels = labels[keep_idx]
    has_label = keep_labels > 0
    lam[np.flatnonzero(has_label), keep_labels[has_label] - 1] = 1

    out = {
        "mtc": mtc.astype(np.float64, copy=False),
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
