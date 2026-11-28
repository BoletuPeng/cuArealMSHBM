"""profiles.py

CPU supercall for per-session FC-profile generation. Takes one
session's decoded BOLD runs and returns binarized top-fraction profile
arrays. The caller
(step-1 pipeline) routes the resulting arrays into the per-subject
blosc2 ``.b2nd`` writer (:mod:`arealmshbm.data_io.profile_io`).

numba kernels in :mod:`._kernels`, called directly; no cupy anywhere on
this path. The GPU path is a different shape entirely — one fused
whole-subject call, :mod:`.profiles_subject_gpu` — not a backend of
this function.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np

from ..data_io.load_avg_mesh import load_avg_mesh
from ._kernels import (
    _zscore_unit_norm_columns_kernel,
    _nan_to_zero_kernel,
    _threshold_kernel,
)


def _apply_mw_zero(lh_bin_KxV: np.ndarray, rh_bin_KxV: np.ndarray,
                   lh_mars: np.ndarray, rh_mars: np.ndarray) -> None:
    """Zero whole columns at medial-wall vertices, in place.

    Writer-side enforcement of the bitpacked ``.b2nd`` contract: rows
    flagged ``MARS_label == 1`` (medial wall) MUST be zero in every
    subject profile that step2 ``SubjectProfileLoader`` will consume.
    Factored out of the supercall so the unit test in
    ``tests/test_mw_clamp.py`` can exercise it without disk I/O; the
    GPU leaf enforces the same contract inside
    ``binarize_mwzero_pack_cupy``, in packed form.

    Asserts strict equality between the (lh, rh) MARS_label sizes and
    the bin column counts (``==``, not ``>=`` — an oversized
    ``MARS_label`` would silently truncate to a wrong-vertex prefix).
    """
    V_lh = int(lh_bin_KxV.shape[1])
    V_rh = int(rh_bin_KxV.shape[1])
    lh_mars_flat = np.asarray(lh_mars).ravel()
    rh_mars_flat = np.asarray(rh_mars).ravel()
    assert lh_mars_flat.size == V_lh, (
        f"lh MARS_label.size={lh_mars_flat.size} != V_lh={V_lh}; "
        f"mesh/BOLD mismatch?"
    )
    assert rh_mars_flat.size == V_rh, (
        f"rh MARS_label.size={rh_mars_flat.size} != V_rh={V_rh}; "
        f"mesh/BOLD mismatch?"
    )
    lh_mw_full = (lh_mars_flat == 1)
    rh_mw_full = (rh_mars_flat == 1)
    if lh_mw_full.any():
        lh_bin_KxV[:, lh_mw_full] = 0.0
    if rh_mw_full.any():
        rh_bin_KxV[:, rh_mw_full] = 0.0


def _read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _read_censor_vector(p: Path) -> np.ndarray:
    with p.open("r", encoding="utf-8") as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    return np.asarray([int(r) for r in rows], dtype=np.int32)


class _SessionInputs(NamedTuple):
    """Inputs for one (sub, sess) compute call: the decoded runs, mesh
    metadata and censor vectors :func:`compute_profile_arrays` sets up
    before its per-run kernels.
    """
    lh_runs: List           # n_runs × (T, V_lh) host numpy
    rh_runs: List
    n_runs: int
    censor_runs: Optional[List]   # n_runs × (T,) int32 or None per run; None when all-keep.
    lh_mars: np.ndarray
    rh_mars: np.ndarray
    threshold_f: float


# The RSFC seeds are the fsaverage3 vertices. An icosphere's coarser
# levels are a prefix of its vertex list, so they are the first 642
# vertices (10·4³ + 2) of each fsaverage6 hemisphere.
N_SEED_VERTS = 642


def _load_session_inputs(out_dir,
                          sub: str,
                          sess: str,
                          threshold,
                          bold_runs,
                          ) -> _SessionInputs:
    """Mesh + censor setup for :func:`compute_profile_arrays` around
    the session's decoded BOLD runs.
    """
    out_dir = Path(out_dir)
    threshold_f = float(threshold)

    lh_mars = load_avg_mesh("lh", "inflated")["MARS_label"]
    rh_mars = load_avg_mesh("rh", "inflated")["MARS_label"]

    n_runs = len(bold_runs)
    lh_runs: List = [b[0] for b in bold_runs]
    rh_runs: List = [b[1] for b in bold_runs]

    # Censor list — the prefetcher does not cover it.
    censor_path = out_dir / "data_list" / "censor_list" / f"sub{sub}_sess{sess}.txt"
    censor_runs: Optional[List] = None
    if censor_path.exists():
        with censor_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
        if content and content.upper() != "NONE":
            outlier_paths = [ln.strip() for ln in content.splitlines() if ln.strip()]
            if len(outlier_paths) != n_runs:
                raise ValueError(
                    f"censor list has {len(outlier_paths)} runs, BOLD has {n_runs}"
                )
            censor_runs = [
                _read_censor_vector(Path(p)) if p.upper() != "NONE" else None
                for p in outlier_paths
            ]

    return _SessionInputs(
        lh_runs=lh_runs, rh_runs=rh_runs, n_runs=n_runs,
        censor_runs=censor_runs,
        lh_mars=lh_mars, rh_mars=rh_mars,
        threshold_f=threshold_f,
    )


def top_fraction_rank(numel: int, fraction: float) -> int:
    """1-indexed rank ``round(numel·fraction)`` (half up) in the
    descending order, clipped to ``[1, numel]``: the position of the
    cutoff that admits the top ``fraction`` of ``numel`` values. The CPU
    and GPU selects share it."""
    return max(1, min(int(np.floor(numel * float(fraction) + 0.5)), numel))


def _threshold_top_fraction(combined_corr: np.ndarray, fraction: float) -> np.float32:
    """Threshold value at rank :func:`top_fraction_rank` in the
    descending order — i.e. the cutoff that admits the top ``fraction``
    of all (K, V) entries. O(N) introselect via ``np.partition``; no
    full sort needed.
    """
    flat = np.ascontiguousarray(combined_corr, dtype=np.float32).ravel()
    numel = flat.size
    # k-th smallest = (numel-k)-th largest in ascending
    kth = numel - top_fraction_rank(numel, fraction)
    return np.float32(np.partition(flat, kth)[kth])


def compute_profile_arrays(out_dir,
                            sub: str,
                            sess: str,
                            threshold: float,
                            bold_runs,
                            ):
    """Compute one (sub, sess) FC-profile array pair WITHOUT writing.

    Parameters
    ----------
    threshold : float
        Fraction of the (K, V) correlations kept, in ``[0, 1]``.
    bold_runs : list of (lh_TxV, rh_TxV)
        The session's decoded host BOLD runs, as
        :class:`Step1BoldPrefetcher` hands them over. ``out_dir`` /
        ``sub`` / ``sess`` locate the session's censor list, which the
        prefetcher does not handle.

    Returns
    -------
    (lh_arr, rh_arr)
        ``(K, V_h) fp32`` C-contig binarized FC-profile arrays with MW
        columns already zeroed. The writer still owes the transpose +
        packbits — cheap on host, so there is no win in pre-packing here
        the way the GPU leaf does.

    CPU only; the GPU path is
    :func:`~arealmshbm.generate_profiles.profiles_subject_gpu.generate_subject_profiles_gpu`,
    a whole-subject call with its own entry point.
    """
    if not bold_runs:
        raise ValueError(f"sub{sub} sess{sess}: no BOLD runs")
    # Backend-mismatch guard: a caller can hand this CPU leaf device
    # buffers (e.g. straight out of ``iter_subject_bold_gpu``). They
    # would reach the numba kernel below and fail deep in numpy's
    # fancy-index / @ overloads with an opaque error. Surface a named
    # TypeError at the boundary.
    # ``type(...).__module__`` avoids a cupy import on CPU-only systems.
    if type(bold_runs[0][0]).__module__.startswith("cupy"):
        raise TypeError(
            "compute_profile_arrays: bold_runs contains "
            "device cupy arrays — this leaf is CPU-only. Convert to "
            "host numpy first, or use generate_subject_profiles_gpu "
            "for the GPU path."
        )
    inputs = _load_session_inputs(
        out_dir=out_dir,
        sub=sub, sess=sess, threshold=threshold,
        bold_runs=bold_runs,
    )
    lh_runs = inputs.lh_runs
    rh_runs = inputs.rh_runs
    n_runs = inputs.n_runs
    censor_runs = inputs.censor_runs
    lh_mars = inputs.lh_mars
    rh_mars = inputs.rh_mars
    threshold_f = inputs.threshold_f

    V_lh = int(lh_runs[0].shape[1])
    V_rh = int(rh_runs[0].shape[1])
    lh_seed_mask = np.zeros(V_lh, dtype=bool)
    rh_seed_mask = np.zeros(V_rh, dtype=bool)
    lh_seed_mask[:N_SEED_VERTS] = (lh_mars[:N_SEED_VERTS] == 2)
    rh_seed_mask[:N_SEED_VERTS] = (rh_mars[:N_SEED_VERTS] == 2)
    K = int(lh_seed_mask.sum() + rh_seed_mask.sum())

    lh_corr_sum = np.zeros((K, V_lh), dtype=np.float32)
    rh_corr_sum = np.zeros((K, V_rh), dtype=np.float32)

    for r in range(n_runs):
        lh_TxV = lh_runs[r]
        rh_TxV = rh_runs[r]
        if lh_TxV.shape[0] != rh_TxV.shape[0]:
            raise ValueError(
                f"run {r}: lh/rh time-axis mismatch: "
                f"{lh_TxV.shape[0]} vs {rh_TxV.shape[0]}"
            )
        if censor_runs is not None and censor_runs[r] is not None:
            keep = (censor_runs[r] == 1)
            if keep.shape[0] != lh_TxV.shape[0]:
                raise ValueError(
                    f"run {r}: censor length {keep.shape[0]} != T={lh_TxV.shape[0]}"
                )
            lh_TxV = np.ascontiguousarray(lh_TxV[keep], dtype=np.float32)
            rh_TxV = np.ascontiguousarray(rh_TxV[keep], dtype=np.float32)

        lh_seed_TxK = np.ascontiguousarray(lh_TxV[:, lh_seed_mask], dtype=np.float32)
        rh_seed_TxK = np.ascontiguousarray(rh_TxV[:, rh_seed_mask], dtype=np.float32)
        s_series = np.concatenate([lh_seed_TxK, rh_seed_TxK], axis=1)

        s_norm = np.empty_like(s_series)
        lh_norm = np.empty_like(lh_TxV)
        rh_norm = np.empty_like(rh_TxV)
        _zscore_unit_norm_columns_kernel(s_series, s_norm)
        _zscore_unit_norm_columns_kernel(lh_TxV, lh_norm)
        _zscore_unit_norm_columns_kernel(rh_TxV, rh_norm)
        lh_corr = np.ascontiguousarray(s_norm.T @ lh_norm, dtype=np.float32)
        rh_corr = np.ascontiguousarray(s_norm.T @ rh_norm, dtype=np.float32)

        _nan_to_zero_kernel(lh_corr)
        _nan_to_zero_kernel(rh_corr)

        lh_corr_sum += lh_corr
        rh_corr_sum += rh_corr

    inv_n = np.float32(1.0 / n_runs)
    lh_corr_sum *= inv_n
    rh_corr_sum *= inv_n

    combined = np.concatenate([lh_corr_sum, rh_corr_sum], axis=1)
    t = _threshold_top_fraction(combined, threshold_f)

    lh_bin_KxV = np.empty((K, V_lh), dtype=np.float32)
    rh_bin_KxV = np.empty((K, V_rh), dtype=np.float32)
    _threshold_kernel(lh_corr_sum, t, lh_bin_KxV)
    _threshold_kernel(rh_corr_sum, t, rh_bin_KxV)

    # Writer-side MW zero (pre-pack zero pass). YS-style BOLD
    # preprocessing leaves signal at MW
    # vertices, which the threshold→binarize can mark as 1; the
    # bitpacked .b2nd contract + step2 SubjectProfileLoader require
    # zero rows at MW vertices. The actual clamp + shape assertion live
    # in ``_apply_mw_zero`` so a no-disk unit test can exercise the same
    # code (the GPU leaf zeroes inside ``binarize_mwzero_pack_cupy``).
    _apply_mw_zero(lh_bin_KxV, rh_bin_KxV, lh_mars, rh_mars)

    return lh_bin_KxV, rh_bin_KxV
