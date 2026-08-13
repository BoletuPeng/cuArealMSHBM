"""step1_runners.py — step-1 subgraph orchestration as module functions.

Step 1 produces four artifacts on disk via four subgraphs:

  1. ``generate_profiles`` — per-subject RSFC profile, written as bitpacked
     .b2nd to ``profiles_raw/sub<id>/...``.
  2. ``avg_profiles``      — across-cohort average, written as .npy to
     ``profiles/avg_profile/{lh,rh}_..._avg_profile.npy``.
  3. ``ini_params``        — initialization prior, written to
     ``group/group.mat``.
  4. ``radius_mask``       — spatial mask, written to
     ``spatial_mask/spatial_mask_<mesh>.mat``.

Each runner returns the artifact (in-memory where reasonable, plus the
written path) so the caller can thread data through memory between
subgraphs and write cohort.json from the result. The disk writes are
the *cache*, not the bus.

Group-labels resolution is shared (``resolve_group_labels``) — both
``run_ini_params`` and ``run_radius_mask`` consume the same pair.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from ._progress import ProgressEmitter


# ─────────────────────────────────────────────────────────────────────
# generate_profiles
# ─────────────────────────────────────────────────────────────────────
def run_generate_profiles(
    *,
    project_dir: Path,
    subjects: Sequence[str],
    sessions: Sequence[str],
    seed_mesh: str,
    targ_mesh: str,
    threshold: float = 0.1,
    split_flag: str = "0",
    profile_dtype_reduce=np.float32,
    backend: str = "cpu",
    verbose: bool = True,
    progress: Optional["ProgressEmitter"] = None,
) -> List[Tuple[str, Path]]:
    """Per-subject (T, N, D) profile writer.

    For each subject, walks its sessions in order, calls
    :func:`compute_profile_arrays` per (sub, sess), stacks bilateral
    hemis along the N axis and writes one bitpacked .b2nd file per
    subject.

    Returns ``[(sub_id, b2nd_path), ...]`` — one entry per subject in
    the order ``subjects`` was given.

    NB: Returns paths, not the in-memory packed bytes. The next
    subgraph (avg_profiles) does a fast .b2nd decompress from OS cache
    (~150 ms / subject), so the small wall savings of an in-memory
    pass aren't worth the kernel-interface churn. See
    docs/profile_disk_format.md for the format.
    """
    from arealmshbm.data_io.profile_io import profile_path
    from ._step1_bold_prefetcher import Step1BoldPrefetcher
    from ._step1_stage_pipeline import Step1GenerateProfilesStagePipeline

    # ── BOLD path resolution (cheap; reads small text files) ─────────
    # Resolve all (sub, sess) → (lh_paths, rh_paths) up-front so the
    # prefetcher's prime() does no disk I/O of its own.
    schedule: List[Tuple[str, str]] = [
        (str(sub), str(sess)) for sub in subjects for sess in sessions
    ]
    bold_paths: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}
    for sub_id, sess_id in schedule:
        lh_lp = project_dir / "data_list" / "fMRI_list" / f"lh_sub{sub_id}_sess{sess_id}.txt"
        rh_lp = project_dir / "data_list" / "fMRI_list" / f"rh_sub{sub_id}_sess{sess_id}.txt"
        if not lh_lp.exists() or not rh_lp.exists():
            raise FileNotFoundError(f"missing fMRI list: {lh_lp} or {rh_lp}")
        with lh_lp.open() as f:
            lh_runs = [ln.strip() for ln in f if ln.strip()]
        with rh_lp.open() as f:
            rh_runs = [ln.strip() for ln in f if ln.strip()]
        bold_paths[(sub_id, sess_id)] = (lh_runs, rh_runs)

    # Per-subject session counts + canonical .b2nd output paths.
    sess_per_sub: Dict[str, int] = {}
    out_paths: Dict[str, Path] = {}
    for sub in subjects:
        sub_id = str(sub)
        if sub_id not in sess_per_sub:
            sess_per_sub[sub_id] = 0
            out_paths[sub_id] = profile_path(
                project_dir, sub_id, targ_mesh, seed_mesh)
        sess_per_sub[sub_id] = sum(
            1 for s in schedule if s[0] == sub_id)

    # ── Stage pipeline — step0-style by-phase coordinator ────────────
    # GZIP (8-worker pool) → GPU (1 thread) → POST (1 thread, .T copy)
    #                 → ASSEM (1 thread; submits writes to WRITE pool).
    # Sliding lookahead = 8 sessions; queue maxsize=2 caps each
    # inter-stage buffer.
    leaf_kwargs = {
        "seed_mesh": seed_mesh,
        "targ_mesh": targ_mesh,
        "split_flag": split_flag,
        "threshold": threshold,
        "profile_dtype_reduce": profile_dtype_reduce,
    }

    # Prefetcher backend matches the leaf backend:
    #   * GPU leaf → 'gpu' prefetcher (full device-stay GIFTI pipeline:
    #     bytescan + base64 as CuPy RawKernels on device, then one
    #     batched nvCOMP Deflate decode per session, per-worker
    #     non_blocking CUDA stream). ~83 ms / sess on YS reference.
    #   * CPU leaf → 'cpu' prefetcher (isal_zlib + 8-worker thread pool
    #     that hides gzip decode behind the leaf's CPU compute).
    # Crossing the backends would produce host-vs-device buffer type
    # confusion in the leaf — the CPU leaf rejects cupy precomputed
    # buffers explicitly (compute_profile_arrays guards this).
    if backend == "gpu":
        prefetcher_backend = "gpu"
        # GPU prefetcher: 2 concurrent (sub, sess) decodes. Each is
        # one ``read_surface_giftis_gpu_full_pipeline`` call covering
        # both hemis (= 2 GIFTI files, one batched nvCOMP decode).
        # 2 outer workers is enough to hide the per-call host stalls
        # behind the leaf's GPU compute on YS (leaf wall ~92 ms vs
        # prefetcher wall ~83 ms / sess); raising to 4+ risks decode
        # work piling up on a single nvCOMP internal stream and
        # pushing peak device-side memory through the working-set
        # ceiling (~632 MB lookahead + transient bytescan/base64
        # buffers ~360 MB at YS scale, vs ~1 GB headroom on the
        # 5090 Laptop's 16 GB).
        prefetcher_workers = 2
    else:
        prefetcher_backend = "cpu"
        # CPU prefetcher: 8 workers matches the isal_zlib saturation
        # plateau for GIFTI gzip decode on the YS reference (probed
        # in PR #54 — raising beyond 8 gives no further wall win
        # because the codec itself is the bottleneck).
        prefetcher_workers = 8
    with Step1BoldPrefetcher(
        n_workers=prefetcher_workers, backend=prefetcher_backend,
    ) as prefetcher:
        coord = Step1GenerateProfilesStagePipeline(
            project_dir=project_dir,
            schedule=schedule,
            bold_paths=bold_paths,
            sess_per_sub=sess_per_sub,
            out_paths=out_paths,
            leaf_kwargs=leaf_kwargs,
            backend=backend,
            prefetcher=prefetcher,
            lookahead=8,
            write_workers=1,
            verbose=verbose,
            progress=progress,
        )
        completed = coord.run()

    # Preserve the input ``subjects`` order in the returned list — the
    # coordinator returns subjects in completion order, which on a
    # multi-worker WRITE pool may interleave.
    by_sub = {sub_id: op for (sub_id, op) in completed}
    out: List[Tuple[str, Path]] = []
    for sub in subjects:
        sub_id = str(sub)
        if sub_id in by_sub:
            out.append((sub_id, by_sub[sub_id]))
    return out


# ─────────────────────────────────────────────────────────────────────
# avg_profiles
# ─────────────────────────────────────────────────────────────────────
def run_avg_profiles(
    *,
    project_dir: Path,
    num_sub: int,
    num_sess: int,
    seed_mesh: str,
    targ_mesh: str,
    backend: str = "cpu",
    verbose: bool = True,
):
    """Average RSFC profiles across (subject × session).

    Returns the underlying :class:`AvgProfilesResult` —
    ``(lh_path, rh_path, lh_avg, rh_avg)``. The averaged arrays are
    on disk under ``profiles/avg_profile/...`` (cache) AND in the
    return object's ``lh_avg``/``rh_avg`` fields (in-memory). The
    next subgraph (``run_ini_params``) consumes the in-memory arrays
    directly via ``precomputed_lh_avg``/``precomputed_rh_avg``,
    skipping the .npy re-read.
    """
    from arealmshbm.avg_profiles import avg_profiles
    return avg_profiles(
        seed_mesh=seed_mesh,
        targ_mesh=targ_mesh,
        out_dir=str(project_dir),
        num_sub=num_sub,
        num_sess=num_sess,
        verbose=verbose,
        backend=backend,
    )


# ─────────────────────────────────────────────────────────────────────
# ini_params
# ─────────────────────────────────────────────────────────────────────
def run_ini_params(
    *,
    project_dir: Path,
    seed_mesh: str,
    targ_mesh: str,
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
    profile_dtype=np.float64,
    reduction_dtype=np.float64,
    backend: str = "cpu",
    precomputed_lh_avg: Optional[np.ndarray] = None,
    precomputed_rh_avg: Optional[np.ndarray] = None,
) -> Tuple[Path, float]:
    """Compute the initialization prior (μ, ε) from the averaged profile.

    When ``precomputed_{lh,rh}_avg`` are passed, ini_params skips the
    .npy disk read and consumes the in-memory ``AvgProfilesResult``
    arrays directly. Returns ``(group_mat_path, epsil)``; the .mat
    write happens inside :func:`generate_ini_params`.
    """
    from arealmshbm.ini_params import generate_ini_params
    out = generate_ini_params(
        seed_mesh=seed_mesh,
        targ_mesh=targ_mesh,
        lh_labels=lh_labels,
        rh_labels=rh_labels,
        out_dir=str(project_dir),
        profile_dtype=profile_dtype,
        reduction_dtype=reduction_dtype,
        save=True,
        backend=backend,
        precomputed_lh_avg=precomputed_lh_avg,
        precomputed_rh_avg=precomputed_rh_avg,
    )
    epsil = float(np.asarray(out["epsil"]).ravel()[0])
    return project_dir / "group" / "group.mat", epsil


# ─────────────────────────────────────────────────────────────────────
# radius_mask
# ─────────────────────────────────────────────────────────────────────
def run_radius_mask(
    *,
    project_dir: Path,
    targ_mesh: str,
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
    radius_mm: float = 30.0,
    cbig_code_dir: Optional[Path] = None,
    dtype=np.float32,
    backend: str = "cpu",
    verbose: bool = True,
) -> Path:
    """Compute the spatial-radius mask. Returns the mat path.

    The ``radius_mm`` parameter is exposed in pipeline_config.json as
    ``step1.radius_mask_radius_mm`` (the longer name disambiguates
    from other ``radius`` knobs at the project-config level). The
    driver maps ``k1.radius_mask_radius_mm`` → ``radius_mm`` explicitly;
    the name divergence is pinned by
    test_knobs_field_match.test_step1_radius_mask_radius_mm_maps_to_run_radius_mask.
    """
    from arealmshbm.radius_mask import generate_radius_mask
    cbig_str = str(cbig_code_dir) if cbig_code_dir is not None else None
    result = generate_radius_mask(
        lh_labels=lh_labels,
        rh_labels=rh_labels,
        mesh=targ_mesh,
        radius=radius_mm,
        out_dir=str(project_dir),
        dtype=dtype,
        verbose=verbose,
        backend=backend,
        cbig_code_dir=cbig_str,
    )
    return Path(result["mat_path"])


# ─────────────────────────────────────────────────────────────────────
# Group labels resolver — shared between ini_params and radius_mask
# ─────────────────────────────────────────────────────────────────────
def resolve_group_labels(
    *,
    targ_mesh: str,
    schaefer_resolution: Optional[str] = None,
    lh_labels_path: Optional[Path] = None,
    rh_labels_path: Optional[Path] = None,
    cbig_code_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(lh_labels, rh_labels)`` with -1 → 0 normalisation.

    Exactly one of ``schaefer_resolution`` or the explicit
    ``{lh,rh}_labels_path`` pair must be set. Memoised module-level
    via :func:`functools.lru_cache` on :func:`_load_labels_cached`,
    keyed on the input paths.
    """
    if schaefer_resolution is not None:
        import nibabel.freesurfer.io as fsio
        from arealmshbm.data_io.load_avg_mesh import _atlas_dir

        base = cbig_code_dir if cbig_code_dir is not None else _atlas_dir()
        stem = (
            f"Schaefer2018_{schaefer_resolution}Parcels_"
            f"Kong2022_17Networks_order.annot"
        )
        annot_dir = Path(base) / targ_mesh / "label"
        lh_p = annot_dir / f"lh.{stem}"
        rh_p = annot_dir / f"rh.{stem}"
        if not (lh_p.exists() and rh_p.exists()):
            raise FileNotFoundError(
                f"Schaefer annot not found: {lh_p} / {rh_p}"
            )
        lh_raw, _, _ = fsio.read_annot(str(lh_p))
        rh_raw, _, _ = fsio.read_annot(str(rh_p))
    elif lh_labels_path is not None and rh_labels_path is not None:
        lh_raw = _load_labels_file(lh_labels_path)
        rh_raw = _load_labels_file(rh_labels_path)
    else:
        raise ValueError(
            "resolve_group_labels: must pass schaefer_resolution OR "
            "both lh_labels_path / rh_labels_path."
        )

    lh_labels = np.where(lh_raw < 0, 0, lh_raw).astype(np.int64)
    rh_labels = np.where(rh_raw < 0, 0, rh_raw).astype(np.int64)
    return lh_labels, rh_labels


def _load_labels_file(p: Path) -> np.ndarray:
    """Load a group-labels file: .annot (FreeSurfer) or .npy."""
    p = Path(p)
    ext = p.suffix.lower()
    if ext == ".annot":
        import nibabel.freesurfer.io as fsio
        arr, _, _ = fsio.read_annot(str(p))
        return arr
    if ext == ".npy":
        return np.load(p)
    raise ValueError(f"unsupported group-labels file: {p}")
