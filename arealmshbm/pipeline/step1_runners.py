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

GPU path (``backend='gpu'``)
----------------------------
On the GPU backend the four subgraphs are chained **in memory and on
device**, and every disk write runs on a background thread that the
caller joins at the end (:func:`join_step1_writers`):

    generate_profiles  → packed (T, N, ⌈K/8⌉) uint8 per subject
                         (``packed_sink``; .b2nd write in flight)
    avg_profiles       ← ``packed_subjects``: device fp32 means
                         (``lh_avg_dev`` / ``rh_avg_dev``; .npy write in flight)
    ini_params         ← device means (no host round trip; group.mat
                         write in flight)
    radius_mask        ← unchanged (mesh geodesics, no cohort input)

The CPU backend keeps the disk-mediated flow (stage pipeline →
.b2nd → avg reads it back), unchanged.

Group-labels resolution is shared (``resolve_group_labels``) — both
``run_ini_params`` and ``run_radius_mask`` consume the same pair.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple,
)

import numpy as np

if TYPE_CHECKING:
    from ._progress import ProgressEmitter


# ─────────────────────────────────────────────────────────────────────
# Shared bookkeeping
# ─────────────────────────────────────────────────────────────────────
def _read_bold_lists(
    project_dir: Path,
    subjects: Sequence[str],
    sessions: Sequence[str],
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], Tuple[List[str], List[str]]]]:
    """Resolve every ``(sub, sess)`` → ``(lh_paths, rh_paths)`` from
    ``data_list/fMRI_list``. Cheap (small text files); done up-front so
    no downstream stage touches the lists again."""
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
    return schedule, bold_paths


def bold_pairs_for_prewarm(project_dir: Path, subjects: Sequence[str],
                           sessions: Sequence[str]) -> List[Tuple[str, str]]:
    """``[(lh, rh), ...]`` of the FIRST subject's runs — what
    :func:`prewarm_step1_gpu` needs to size the pinned staging buffer.
    Never raises on a missing list (the real run reports that)."""
    try:
        _, bold_paths = _read_bold_lists(project_dir, subjects[:1], sessions)
    except FileNotFoundError:
        return []
    pairs: List[Tuple[str, str]] = []
    for (_sub, _sess), (lh, rh) in bold_paths.items():
        pairs.extend(zip(lh, rh))
    return pairs


_PREWARM_THREAD: Optional[threading.Thread] = None
_PREWARM_LOCK = threading.Lock()
_log = logging.getLogger(__name__)


def prewarm_step1_gpu(bold_pairs: Optional[Sequence[Tuple[str, str]]] = None,
                      *, background: bool = True) -> Optional[threading.Thread]:
    """Pay step 1's one-time GPU costs (RawKernel NVRTC compiles, the
    cuBLAS handle, and — when the leaf's ingest will be nvCOMP — the
    nvCOMP library + GIFTI kernels + pinned staging) off the timed
    path.

    Idempotent: the leaf's prewarm is lock-guarded, so a second call —
    including the one ``run_generate_profiles`` makes itself — simply
    waits for the first to finish. Returns the daemon thread when
    backgrounded (``None`` when run inline). Never raises: a prewarm
    failure just means the run pays the cost itself, so the exception
    is logged at DEBUG rather than propagated.

    Join the thread with :func:`join_step1_prewarm` before the process
    exits — otherwise a still-running NVRTC compile can outlive the
    interpreter's cupy teardown.
    """
    global _PREWARM_THREAD

    # A fresh thread defaults to device 0, so the prewarm must be
    # pinned to the device the CALLER is on or it warms the wrong one.
    dev_id = None
    try:
        import cupy as cp
        dev_id = int(cp.cuda.runtime.getDevice())
    except Exception:      # noqa: BLE001 — no cupy / no device
        _log.debug("step1 prewarm: no CUDA device to pin", exc_info=True)

    def _work() -> None:
        try:
            if dev_id is not None:
                import cupy as cp
                cp.cuda.Device(dev_id).use()
            from arealmshbm.generate_profiles.profiles_subject_gpu import (
                prewarm_generate_profiles_gpu,
            )
            prewarm_generate_profiles_gpu(
                list(bold_pairs) if bold_pairs else None)
        except Exception:      # noqa: BLE001 — warm-up is best effort
            _log.debug("step1 GPU prewarm failed; the run will pay the "
                       "first-call cost itself", exc_info=True)

    if not background:
        _work()
        return None
    with _PREWARM_LOCK:
        th = _PREWARM_THREAD
        if th is not None and th.is_alive():
            return th
        th = threading.Thread(target=_work, name="step1-prewarm", daemon=True)
        _PREWARM_THREAD = th
        th.start()
        return th


def join_step1_prewarm(timeout: Optional[float] = 30.0) -> None:
    """Join the background thread :func:`prewarm_step1_gpu` started.

    No-op when none is running. A thread still alive after ``timeout``
    is left alone (it is a daemon); that is logged at DEBUG.
    """
    with _PREWARM_LOCK:
        th = _PREWARM_THREAD
    if th is None or not th.is_alive():
        return
    th.join(timeout)
    if th.is_alive():
        _log.debug("step1 GPU prewarm still running after %.1fs; leaving it "
                   "to the daemon-thread teardown", timeout)


# ─────────────────────────────────────────────────────────────────────
# generate_profiles
# ─────────────────────────────────────────────────────────────────────
class PendingSubjectWrite(NamedTuple):
    """One subject's in-flight ``.b2nd`` write (GPU path). ``handle``
    is a :class:`SubjectProfilesResult`; ``join_step1_writers`` waits
    on it and emits the subject's terminal progress state."""
    sub_id: str
    handle: Any


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
    packed_sink: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    write_handles: Optional[List[PendingSubjectWrite]] = None,
) -> List[Tuple[str, Path]]:
    """Per-subject (T, N, D) profile writer.

    Returns ``[(sub_id, b2nd_path), ...]`` — one entry per subject in
    the order ``subjects`` was given.

    ``backend='gpu'`` runs the fused whole-subject leaf
    (:func:`~arealmshbm.generate_profiles.profiles_subject_gpu.generate_subject_profiles_gpu`:
    ingest → one zscore / sgemm / exact-select / pack per session →
    one D2H → background .b2nd write). The leaf picks its own ingest
    — batched nvCOMP when the library is installed, the CPU GIFTI
    reader otherwise, byte-identical either way — so there is nothing
    for this runner to route. Two optional sinks let the caller keep
    the result in memory instead of re-reading the .b2nd:

    * ``packed_sink`` — filled with ``sub_id → (packed (T, N, ⌈K/8⌉)
      uint8 host array, K)``. For a multi-subject cohort each array is
      a pageable copy (the leaf's pinned block is recycled subject to
      subject); for a single subject it is the leaf's own buffer.
    * ``write_handles`` — when given, the single-subject case does
      NOT join its .b2nd write here: the pending write is appended and
      the caller must :func:`join_step1_writers` it (after consuming
      ``packed_sink``, which borrows the pinned block the writer is
      still reading). Multi-subject cohorts always join per subject so
      pinned memory stays bounded.

    ``backend='cpu'`` keeps the stage pipeline (GZIP pool → numba leaf
    → POST → ASSEM → WRITE); the sinks are ignored (the CPU
    ``avg_profiles`` reads the .b2nd back from disk).
    """
    from arealmshbm.data_io.profile_io import profile_path

    schedule, bold_paths = _read_bold_lists(project_dir, subjects, sessions)

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

    if backend not in ("cpu", "gpu"):
        raise ValueError(f"run_generate_profiles: unknown backend {backend!r}")

    if backend == "gpu":
        # The fused leaf reduces in fp32 throughout (its zscore, sgemm
        # and select kernels are fp32). Anything else must fail loudly
        # rather than be silently ignored.
        if np.dtype(profile_dtype_reduce) != np.float32:
            raise ValueError(
                f"run_generate_profiles: backend='gpu' computes in float32; "
                f"profile_dtype_reduce={np.dtype(profile_dtype_reduce)} is "
                f"not supported. Use backend='cpu' for another reduction "
                f"dtype."
            )
        return _run_generate_profiles_gpu_fused(
            project_dir=project_dir, subjects=subjects, sessions=sessions,
            bold_paths=bold_paths, out_paths=out_paths,
            seed_mesh=seed_mesh, targ_mesh=targ_mesh,
            threshold=threshold, split_flag=split_flag,
            verbose=verbose, progress=progress,
            packed_sink=packed_sink, write_handles=write_handles,
        )

    from ._step1_bold_prefetcher import Step1BoldPrefetcher
    from ._step1_stage_pipeline import Step1GenerateProfilesStagePipeline

    # ── Stage pipeline — step0-style by-phase coordinator ────────────
    # GZIP (8-worker pool) → leaf (1 thread) → POST (1 thread, .T copy)
    # → ASSEM (1 thread; submits writes to WRITE pool). Sliding
    # lookahead = 8 sessions; queue maxsize=2 caps each inter-stage
    # buffer.
    leaf_kwargs = {
        "seed_mesh": seed_mesh,
        "targ_mesh": targ_mesh,
        "split_flag": split_flag,
        "threshold": threshold,
        "profile_dtype_reduce": profile_dtype_reduce,
    }
    # The prefetcher decodes on the CPU (isal_zlib + 8 workers, the
    # saturation plateau for GIFTI gzip decode on the YS reference).
    with Step1BoldPrefetcher(n_workers=8) as prefetcher:
        coord = Step1GenerateProfilesStagePipeline(
            project_dir=project_dir,
            schedule=schedule,
            bold_paths=bold_paths,
            sess_per_sub=sess_per_sub,
            out_paths=out_paths,
            leaf_kwargs=leaf_kwargs,
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


def _run_generate_profiles_gpu_fused(
    *,
    project_dir: Path,
    subjects: Sequence[str],
    sessions: Sequence[str],
    bold_paths: Dict[Tuple[str, str], Tuple[List[str], List[str]]],
    out_paths: Dict[str, Path],
    seed_mesh: str,
    targ_mesh: str,
    threshold: float,
    split_flag: str,
    verbose: bool,
    progress: Optional["ProgressEmitter"],
    packed_sink: Optional[Dict[str, Tuple[np.ndarray, int]]],
    write_handles: Optional[List[PendingSubjectWrite]],
) -> List[Tuple[str, Path]]:
    """GPU path of :func:`run_generate_profiles`: one fused
    whole-subject leaf call per subject, sequential over subjects."""
    from arealmshbm.generate_profiles.profiles_subject_gpu import (
        generate_subject_profiles_gpu, prewarm_generate_profiles_gpu,
    )

    sess_ids = [str(s) for s in sessions]
    sub_ids = [str(s) for s in subjects]
    first_pairs: List[Tuple[str, str]] = []
    if sub_ids:
        for s in sess_ids:
            lh, rh = bold_paths[(sub_ids[0], s)]
            first_pairs.extend(zip(lh, rh))
    # Idempotent; if ``prewarm_step1_gpu`` already started it on a
    # background thread this just waits for it.
    prewarm_generate_profiles_gpu(first_pairs or None)

    defer_join = write_handles is not None and len(sub_ids) == 1
    out: List[Tuple[str, Path]] = []
    for sub_id in sub_ids:
        if progress is not None:
            progress.emit_state("step1", sub_id, "running")
        t0 = time.perf_counter()
        try:
            res = generate_subject_profiles_gpu(
                project_dir, sub_id, sess_ids,
                bold_paths={s: bold_paths[(sub_id, s)] for s in sess_ids},
                out_path=out_paths[sub_id],
                seed_mesh=seed_mesh, targ_mesh=targ_mesh,
                threshold=threshold, split_flag=split_flag,
            )
            if packed_sink is not None:
                packed = res.packed if defer_join else np.array(res.packed, copy=True)
                packed_sink[sub_id] = (packed, int(res.K))
            if defer_join:
                write_handles.append(PendingSubjectWrite(sub_id, res))
            else:
                res.wait()
        except BaseException as e:
            if progress is not None:
                progress.emit_state(
                    "step1", sub_id, "failed",
                    error=f"{type(e).__name__}: {e}")
            raise
        if not defer_join and progress is not None:
            progress.emit_state("step1", sub_id, "done")
        if verbose:
            print(f"  [1] generate_subject_profiles_gpu sub={sub_id}  "
                  f"sessions={len(sess_ids)}  K={int(res.K)} "
                  f"D={res.packed.shape[-1]}  ingest={res.ingest}  "
                  f"{time.perf_counter() - t0:.2f} s", flush=True)
        out.append((sub_id, Path(res.out_path)))
    return out


def join_step1_writers(
    write_handles: Optional[Sequence[PendingSubjectWrite]] = None,
    avg_result=None,
    ini_result=None,
    *,
    progress: Optional["ProgressEmitter"] = None,
) -> None:
    """Join every in-flight step-1 disk write.

    ``write_handles`` are the pending .b2nd writes from
    :func:`run_generate_profiles` (their subject's ``step1`` progress
    slot is closed here); ``avg_result`` / ``ini_result`` are the
    :class:`AvgProfilesResult` / :class:`IniParamsRun` whose ``writer``
    (when not ``None``) is joined. Every handle is waited on even when
    an earlier one raised; the first exception is re-raised at the end,
    so no writer thread is left running behind a failure.
    """
    first: Optional[BaseException] = None
    for pending in (write_handles or ()):
        try:
            pending.handle.wait()
        except BaseException as e:      # noqa: BLE001 — reported below
            first = first or e
            if progress is not None:
                progress.emit_state(
                    "step1", pending.sub_id, "failed",
                    error=f"{type(e).__name__}: {e}")
            continue
        if progress is not None:
            progress.emit_state("step1", pending.sub_id, "done")
    for result in (avg_result, ini_result):
        if result is None or getattr(result, "writer", None) is None:
            continue
        try:
            result.writer.wait()
        except BaseException as e:      # noqa: BLE001 — reported below
            first = first or e
    if first is not None:
        raise first


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
    packed_subjects: Optional[Sequence[np.ndarray]] = None,
    D: Optional[int] = None,
):
    """Average RSFC profiles across (subject × session).

    Returns the underlying :class:`AvgProfilesResult` —
    ``(lh_path, rh_path, lh_avg, rh_avg[, lh_avg_dev, rh_avg_dev, writer])``.

    Disk flow (default): reads every subject's .b2nd back from
    ``profiles_raw/`` and writes the .npy pair synchronously.

    In-memory flow (``packed_subjects`` + ``D`` given; GPU only): the
    ``(T, N, ⌈D/8⌉)`` uint8 arrays from ``run_generate_profiles``'s
    ``packed_sink`` are accumulated on device; the result carries the
    device fp32 means (``lh_avg_dev`` / ``rh_avg_dev``) for
    ``run_ini_params`` and a ``writer`` handle for the background .npy
    pair that the caller must join (:func:`join_step1_writers`).
    """
    if packed_subjects is not None:
        if backend != "gpu":
            raise ValueError(
                "run_avg_profiles: the in-memory packed_subjects hand-off "
                "is GPU-only; the CPU backend reads the .b2nd files back."
            )
        if D is None:
            raise ValueError("run_avg_profiles: packed_subjects needs D")
        # Same invariant the disk branch enforces on the decoded .b2nd:
        # the accumulated slab must carry exactly ``num_sess`` sessions.
        # (The leaf pins T equal across subjects, so subject 0 speaks
        # for all of them; an empty sequence is the leaf's own error.)
        if len(packed_subjects) > 0:
            T0 = int(np.shape(packed_subjects[0])[0])
            if T0 != num_sess:
                raise ValueError(
                    f"run_avg_profiles: packed_subjects has T={T0} but "
                    f"num_sess={num_sess}."
                )
        from arealmshbm.avg_profiles.avg_profiles_gpu import (
            avg_profiles_from_packed_gpu,
        )
        return avg_profiles_from_packed_gpu(
            list(packed_subjects), int(D), targ_mesh, seed_mesh,
            str(project_dir), save=True, verbose=verbose,
        )
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
class IniParamsRun(NamedTuple):
    """``run_ini_params`` result: the group.mat path, the scalar ε, and
    the background write handle (``None`` when the write was
    synchronous). Unpacks as the historical ``(path, epsil)`` pair via
    ``path, epsil = res[:2]``."""
    group_mat_path: Path
    epsil: float
    writer: Any


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
    precomputed_lh_avg_dev=None,
    precomputed_rh_avg_dev=None,
    save_async: bool = False,
) -> IniParamsRun:
    """Compute the initialization prior (μ, ε) from the averaged profile.

    Profile source, in priority order: device arrays
    (``precomputed_*_avg_dev``, GPU only — no host round trip), host
    arrays (``precomputed_*_avg``), else the .npy pair on disk. When
    both device and host arrays are given the device pair wins on the
    GPU backend and the host pair on the CPU backend, so a caller can
    hand over an :class:`AvgProfilesResult` without branching.

    ``save_async`` submits the group.mat write to a background thread
    on either backend; join it via :func:`join_step1_writers`.
    """
    from arealmshbm.ini_params import generate_ini_params

    use_dev = (backend == "gpu"
               and precomputed_lh_avg_dev is not None
               and precomputed_rh_avg_dev is not None)
    kwargs: Dict[str, Any] = {}
    if use_dev:
        kwargs.update(precomputed_lh_avg_dev=precomputed_lh_avg_dev,
                      precomputed_rh_avg_dev=precomputed_rh_avg_dev)
    else:
        kwargs.update(precomputed_lh_avg=precomputed_lh_avg,
                      precomputed_rh_avg=precomputed_rh_avg)
    # group.mat keeps its compressed on-disk form on both backends; the
    # zlib cost just moves to the writer thread.
    kwargs["save_async"] = bool(save_async)
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
        **kwargs,
    )
    epsil = float(np.asarray(out["epsil"]).ravel()[0])
    return IniParamsRun(
        project_dir / "group" / "group.mat", epsil,
        getattr(out, "writer", None),
    )


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
    ``{lh,rh}_labels_path`` pair must be set. The ``.annot`` parse is
    the numpy-only :func:`~arealmshbm.data_io.annot_io.read_annot_labels`
    (equal to nibabel's labels vector; avoids the ~0.8 s cold
    ``import nibabel``).
    """
    if schaefer_resolution is not None:
        from arealmshbm.data_io.annot_io import read_annot_labels
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
        lh_raw = read_annot_labels(lh_p)
        rh_raw = read_annot_labels(rh_p)
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
        from arealmshbm.data_io.annot_io import read_annot_labels
        return read_annot_labels(p)
    if ext == ".npy":
        return np.load(p)
    raise ValueError(f"unsupported group-labels file: {p}")
