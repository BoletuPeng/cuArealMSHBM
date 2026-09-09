"""profiles.py

CPU supercall for per-session FC-profile generation. Reads BOLD path
lists and returns binarized top-fraction profile arrays. The caller
(step-1 pipeline) routes the resulting arrays into the per-subject
blosc2 ``.b2nd`` writer (:mod:`arealmshbm.data_io.profile_io`).

numba kernels in :mod:`._kernels`, called directly; no cupy anywhere on
this path. The GPU path is a different shape entirely — one fused
whole-subject call, :mod:`.profiles_subject_gpu` — not a backend of
this function.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np

from ..data_io.load_avg_mesh import load_avg_mesh
from ..data_io.gifti_io import read_surface_gifti
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


def _read_run_pair(lh_path, rh_path):
    """Read one BOLD run's lh + rh as ``(T, V_h)`` fp32 buffers.

    Used as a thread-pool task; the GIFTI reader's per-darray base64 +
    isal_zlib decompress releases the GIL, so a multi-worker pool over
    the cohort's BOLD files actually parallelizes the CPU decode. The
    reader hands back its own time-major layout, so there is no host
    transpose on either side.
    """
    lh_TxV = read_surface_gifti(lh_path, time_major=True)
    rh_TxV = read_surface_gifti(rh_path, time_major=True)
    return lh_TxV, rh_TxV


def _read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _read_censor_vector(p: Path) -> np.ndarray:
    with p.open("r", encoding="utf-8") as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    return np.asarray([int(r) for r in rows], dtype=np.int32)


class _SessionInputs(NamedTuple):
    """Decoded inputs for one (sub, sess) compute call: the disk reads,
    mesh metadata and censor logic :func:`compute_profile_arrays` runs
    before its per-run kernels.
    """
    lh_runs: List           # n_runs × (T, V_lh) — host numpy unless
    rh_runs: List           # the caller pre-staged on device.
    n_runs: int
    censor_runs: Optional[List]   # n_runs × (T,) int32 or None per run; None when all-keep.
    lh_mars: np.ndarray
    rh_mars: np.ndarray
    n_seed_lh_verts: int
    n_seed_rh_verts: int
    threshold_f: float


def _icosphere_nverts(mesh: str) -> int:
    """Vertex count of an fsaverage icosphere: ``10·4^order + 2``.

    fsaverage3→642, 4→2562, 5→10242, 6→40962. The seed only needs its
    vertex count (the seed mask is the first N cortex verts of the targ
    mesh), so we compute it in closed form instead of loading the seed
    mesh — the count is topology-invariant, no surface read required.
    """
    if not mesh.startswith("fsaverage"):
        raise ValueError(f"not an fsaverage mesh: {mesh!r}")
    suffix = mesh[len("fsaverage"):]
    if suffix == "":
        order = 7
    elif suffix.isdigit():
        order = int(suffix)
    else:
        # fs_LR-style names (e.g. 'fsaverage_LR32k') are not icospheres;
        # reject clearly instead of a cryptic int() ValueError.
        raise ValueError(f"not an icosphere fsaverage mesh: {mesh!r}")
    return 10 * (4 ** order) + 2


def _load_session_inputs(seed_mesh: str,
                          targ_mesh: str,
                          out_dir,
                          sub: str,
                          sess: str,
                          split_flag: str,
                          threshold,
                          precomputed_bold_runs,
                          ) -> _SessionInputs:
    """Mesh + BOLD + censor setup for :func:`compute_profile_arrays`:
    the decoded per-session inputs it then builds its seed mask from
    and runs its per-run kernels over.
    """
    out_dir = Path(out_dir)
    if not targ_mesh.startswith("fsaverage"):
        raise ValueError(
            f"only fsaverage targets are supported; got {targ_mesh!r}"
        )
    if str(split_flag) != "0":
        raise NotImplementedError(
            f"split_flag={split_flag!r}: only '0' (no-split) is supported."
        )
    threshold_f = float(threshold)

    lh_targ = load_avg_mesh("lh", targ_mesh, "inflated")
    rh_targ = load_avg_mesh("rh", targ_mesh, "inflated")
    n_seed_lh_verts = _icosphere_nverts(seed_mesh)
    n_seed_rh_verts = _icosphere_nverts(seed_mesh)
    lh_mars = lh_targ["MARS_label"]
    rh_mars = rh_targ["MARS_label"]

    if precomputed_bold_runs is not None:
        # Driver-level Step1BoldPrefetcher already gzip-decoded ahead of
        # us. Skip the disk-read pool. Censor-list still comes from disk
        # (orthogonal, rarely populated).
        lh_list_path = out_dir / "data_list" / "fMRI_list" / f"lh_sub{sub}_sess{sess}.txt"
        rh_list_path = out_dir / "data_list" / "fMRI_list" / f"rh_sub{sub}_sess{sess}.txt"
        lh_paths = _read_lines(lh_list_path) if lh_list_path.exists() else []
        rh_paths = _read_lines(rh_list_path) if rh_list_path.exists() else []
        n_runs = len(precomputed_bold_runs)
        # Mirror the non-precomputed path's lh/rh symmetry invariants.
        # The prefetcher worker also asserts these, but keep the leaf-
        # boundary check so any future caller of the precomputed_bold_runs
        # API gets the same correctness contract.
        if lh_paths and rh_paths and len(lh_paths) != len(rh_paths):
            raise ValueError(
                f"lh_runs ({len(lh_paths)}) != rh_runs ({len(rh_paths)}); "
                f"mismatched per-hemi fMRI lists"
            )
        if lh_paths and len(lh_paths) != n_runs:
            raise ValueError(
                f"precomputed_bold_runs has {n_runs} runs but lh fMRI_list "
                f"has {len(lh_paths)} paths"
            )
        if rh_paths and len(rh_paths) != n_runs:
            raise ValueError(
                f"precomputed_bold_runs has {n_runs} runs but rh fMRI_list "
                f"has {len(rh_paths)} paths"
            )
        lh_runs: List = [b[0] for b in precomputed_bold_runs]
        rh_runs: List = [b[1] for b in precomputed_bold_runs]
    else:
        lh_list_path = out_dir / "data_list" / "fMRI_list" / f"lh_sub{sub}_sess{sess}.txt"
        rh_list_path = out_dir / "data_list" / "fMRI_list" / f"rh_sub{sub}_sess{sess}.txt"
        if not lh_list_path.exists() or not rh_list_path.exists():
            raise FileNotFoundError(
                f"missing fMRI lists: {lh_list_path} or {rh_list_path}"
            )
        lh_paths = _read_lines(lh_list_path)
        rh_paths = _read_lines(rh_list_path)
        if len(lh_paths) != len(rh_paths):
            raise ValueError(
                f"lh_runs ({len(lh_paths)}) != rh_runs ({len(rh_paths)})"
            )

        # Parallel reads — the GIFTI reader's base64 + isal_zlib
        # decode releases the GIL (see _read_run_pair).
        n_runs = len(lh_paths)
        lh_runs = [None] * n_runs
        rh_runs = [None] * n_runs
        with ThreadPoolExecutor(max_workers=min(n_runs, 8)) as pool:
            futures = [
                pool.submit(_read_run_pair, lh_p, rh_p)
                for lh_p, rh_p in zip(lh_paths, rh_paths)
            ]
            for r, fut in enumerate(futures):
                lh_TxV, rh_TxV = fut.result()
                lh_runs[r] = lh_TxV
                rh_runs[r] = rh_TxV

    # Censor list — orthogonal to BOLD decode; the prefetcher does not
    # cover it, so read it here regardless of the precomputed path.
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
        n_seed_lh_verts=n_seed_lh_verts, n_seed_rh_verts=n_seed_rh_verts,
        threshold_f=threshold_f,
    )


def _threshold_top_fraction(combined_corr: np.ndarray, fraction: float) -> np.float32:
    """Threshold value at rank ``round(N·fraction)`` (1-indexed) in the
    descending order — i.e. the cutoff that admits the top ``fraction``
    of all (K, V) entries. O(N) introselect via ``np.partition``; no
    full sort needed.
    """
    flat = np.ascontiguousarray(combined_corr, dtype=np.float32).ravel()
    numel = flat.size
    raw = numel * float(fraction)
    if raw >= 0.0:
        idx_1based = int(np.floor(raw + 0.5))
    else:
        idx_1based = -int(np.floor(-raw + 0.5))
    idx_1based = max(1, min(idx_1based, numel))
    kth = numel - idx_1based  # k-th smallest = (numel-k)-th largest in ascending
    return np.float32(np.partition(flat, kth)[kth])


def compute_profile_arrays(seed_mesh: str,
                            targ_mesh: str,
                            out_dir,
                            sub: str,
                            sess: str,
                            split_flag: str = "0",
                            threshold="0.1",
                            dtype_reduce: np.dtype = np.float32,
                            precomputed_bold_runs=None,
                            ):
    """Compute one (sub, sess) FC-profile array pair WITHOUT writing.

    Parameters
    ----------
    precomputed_bold_runs : list of (lh_TxV, rh_TxV), optional
        When set, skip the internal gzip-decode ThreadPoolExecutor and
        use the pre-decoded BOLD buffers. Used by the driver-level
        :class:`Step1BoldPrefetcher` to hoist gzip decompress out of the
        serial per-session loop. The leaf still consumes
        ``out_dir / sub / sess`` for the censor list (which the
        prefetcher does not handle).

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
    if np.dtype(dtype_reduce) != np.float32:
        raise ValueError(
            f"compute_profile_arrays: dtype_reduce must be np.float32 (got "
            f"{np.dtype(dtype_reduce)!r}); the leaf is hard-coded fp32."
        )
    # Backend-mismatch guard: ``precomputed_bold_runs`` is public, so a
    # caller can hand this CPU leaf device buffers (e.g. straight out of
    # ``read_subject_bold_gpu``). They would reach the numba kernel below
    # and fail deep in numpy's fancy-index / @ overloads with an opaque
    # error. Surface a named TypeError at the boundary.
    # ``type(...).__module__`` avoids a cupy import on CPU-only systems.
    if precomputed_bold_runs is not None and len(precomputed_bold_runs) > 0:
        first = precomputed_bold_runs[0][0]
        if type(first).__module__.startswith("cupy"):
            raise TypeError(
                "compute_profile_arrays: precomputed_bold_runs contains "
                "device cupy arrays — this leaf is CPU-only. Convert to "
                "host numpy first, or use generate_subject_profiles_gpu "
                "for the GPU path."
            )
    inputs = _load_session_inputs(
        seed_mesh=seed_mesh, targ_mesh=targ_mesh, out_dir=out_dir,
        sub=sub, sess=sess, split_flag=split_flag, threshold=threshold,
        precomputed_bold_runs=precomputed_bold_runs,
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
    if lh_mars.shape[0] < inputs.n_seed_lh_verts or rh_mars.shape[0] < inputs.n_seed_rh_verts:
        raise ValueError("MARS_label arrays shorter than seed-vert count")
    lh_seed_mask = np.zeros(V_lh, dtype=bool)
    rh_seed_mask = np.zeros(V_rh, dtype=bool)
    lh_seed_mask[:inputs.n_seed_lh_verts] = (lh_mars[:inputs.n_seed_lh_verts] == 2)
    rh_seed_mask[:inputs.n_seed_rh_verts] = (rh_mars[:inputs.n_seed_rh_verts] == 2)
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
    # in ``_apply_mw_zero`` so the GPU path and a no-disk unit test can
    # exercise the same code.
    _apply_mw_zero(lh_bin_KxV, rh_bin_KxV, lh_mars, rh_mars)

    return lh_bin_KxV, rh_bin_KxV
