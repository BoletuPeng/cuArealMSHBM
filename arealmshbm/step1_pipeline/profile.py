"""profile.py

Per-stage timing report for step 1 on a single subject. Calls each
subgraph runner from :mod:`arealmshbm.pipeline.step1_runners`
directly — chained exactly the way the unified driver chains them
(in-memory / on-device hand-off + background writers on the GPU
backend) — and aggregates wall time per stage across ``--runs``
repeats.

Per-stage numbers are the **critical path** of each subgraph (the
time the next subgraph has to wait); ``join_writes`` is the time
spent at the end waiting for the background .b2nd / .npy / group.mat
writes that were still in flight. ``TOTAL`` is the wall of one full
step-1 pass including that join. Run 1 of a fresh process is a cold
start (RawKernel JIT, cuBLAS handle, nvCOMP load, pinned staging);
runs 2+ are warm.

Usage::

    python -m arealmshbm.step1_pipeline.profile
    python -m arealmshbm.step1_pipeline.profile --subject 1
    python -m arealmshbm.step1_pipeline.profile --skip radius_mask
    python -m arealmshbm.step1_pipeline.profile --runs 3        # avg
    python -m arealmshbm.step1_pipeline.profile --prewarm       # GPU one-time costs off the timed path

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arealmshbm.pipeline.step1_runners import (
    bold_pairs_for_prewarm,
    join_step1_writers,
    prewarm_step1_gpu,
    resolve_group_labels,
    run_avg_profiles,
    run_generate_profiles,
    run_ini_params,
    run_radius_mask,
)


_DEFAULT_PROJECT = Path("profile_runs/modeA/sub-001")

_STAGES = ("generate_profiles", "avg_profiles", "ini_params",
           "radius_mask", "join_writes")


# ─────────────────────────────────────────────────────────────────────
# Local helpers (inlined from the now-retired step1 validate.py when the
# pipeline decoupled from MATLAB — profile.py was the only consumer).
# ─────────────────────────────────────────────────────────────────────
def _release_cupy_pool(*, pinned: bool = False) -> None:
    """Free the cupy device memory pool (and, on request, the pinned
    pool). No-op when cupy isn't installed or the CPU backend is in
    use.

    The pinned pool is left alone between runs on purpose: page-locking
    the 2×192 MB avg-profile D2H buffers costs ~70 ms, and the driver
    keeps that pool warm across a cohort as well.
    """
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError:
        return
    cp.get_default_memory_pool().free_all_blocks()
    if pinned:
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _stage_inputs(src: Path, dst: Path, sessions: List[str]) -> None:
    """Stage the ``data_list/fMRI_list`` BOLD lists from ``src`` into
    ``dst`` for subgraph 1 (generate_profiles). No BOLD bytes are
    copied — the lists carry absolute paths to the source GIFTI, which
    ``run_generate_profiles`` reads in place.
    """
    if not src.exists():
        raise FileNotFoundError(f"source project missing: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    list_src = src / "data_list" / "fMRI_list"
    list_dst = dst / "data_list" / "fMRI_list"
    if not list_dst.exists():
        list_dst.mkdir(parents=True, exist_ok=True)
        for f in list_src.iterdir():
            if f.is_file():
                shutil.copy(f, list_dst / f.name)


def _info(msg: str = "") -> None:
    print(msg, flush=True)


def _section(title: str) -> None:
    bar = "=" * 80
    _info(f"\n{bar}\n{title}\n{bar}")


def _fmt_pct(n: float, d: float) -> str:
    return f"{(100.0 * n / d if d else 0.0):5.1f}%"


def _run_once(out: Path, args) -> Dict[str, float]:
    """Execute the enabled subgraphs once against ``out``, chained the
    way the driver chains them. Returns per-stage wall (seconds) plus
    ``__wall__`` total."""
    timings: Dict[str, float] = {}
    is_gpu = args.backend == "gpu"
    group_labels: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _group_labels() -> Tuple[np.ndarray, np.ndarray]:
        nonlocal group_labels
        if group_labels is None:
            group_labels = resolve_group_labels(
                targ_mesh=args.targ_mesh,
                schaefer_resolution=args.schaefer_resolution,
            )
        return group_labels

    run_avg = "avg_profiles" not in args.skip
    run_ini = "ini_params" not in args.skip
    run_gen = "generate_profiles" not in args.skip
    run_mask = "radius_mask" not in args.skip

    t_total = time.perf_counter()

    packed_sink: Optional[Dict] = {} if (is_gpu and run_avg) else None
    write_handles: Optional[list] = [] if is_gpu else None
    if run_gen:
        t = time.perf_counter()
        run_generate_profiles(
            project_dir=out,
            subjects=args.subjects,
            sessions=args.sessions,
            seed_mesh=args.seed_mesh,
            targ_mesh=args.targ_mesh,
            backend=args.backend,
            verbose=False,
            packed_sink=packed_sink,
            write_handles=write_handles,
        )
        timings["generate_profiles"] = time.perf_counter() - t

    # Subgraphs 2-4 run under one try/finally — the same shape the
    # driver uses — so a failure in any of them (e.g. a missing .annot
    # in _group_labels) still joins the background writers instead of
    # letting their files land after the run reported failure.
    avg_res = None
    ini_res = None
    joined = False
    try:
        if run_avg:
            t = time.perf_counter()
            num_sub = max(int(s) for s in args.subjects)
            if packed_sink:
                packed_subjects = [packed_sink[str(s)][0] for s in args.subjects]
                D = packed_sink[str(args.subjects[0])][1]
            else:
                packed_subjects, D = None, None
            avg_res = run_avg_profiles(
                project_dir=out,
                num_sub=num_sub,
                num_sess=len(args.sessions),
                seed_mesh=args.seed_mesh,
                targ_mesh=args.targ_mesh,
                backend=args.backend,
                verbose=False,
                packed_subjects=packed_subjects,
                D=D,
            )
            timings["avg_profiles"] = time.perf_counter() - t

        if run_ini:
            t = time.perf_counter()
            lh_labels, rh_labels = _group_labels()
            ini_res = run_ini_params(
                project_dir=out,
                seed_mesh=args.seed_mesh,
                targ_mesh=args.targ_mesh,
                lh_labels=lh_labels,
                rh_labels=rh_labels,
                backend=args.backend,
                precomputed_lh_avg=(avg_res.lh_avg if avg_res is not None else None),
                precomputed_rh_avg=(avg_res.rh_avg if avg_res is not None else None),
                precomputed_lh_avg_dev=(avg_res.lh_avg_dev if avg_res is not None else None),
                precomputed_rh_avg_dev=(avg_res.rh_avg_dev if avg_res is not None else None),
                save_async=True,
            )
            timings["ini_params"] = time.perf_counter() - t

        if run_mask:
            t = time.perf_counter()
            lh_labels, rh_labels = _group_labels()
            run_radius_mask(
                project_dir=out,
                targ_mesh=args.targ_mesh,
                lh_labels=lh_labels,
                rh_labels=rh_labels,
                backend=args.backend,
                verbose=False,
            )
            timings["radius_mask"] = time.perf_counter() - t

        t = time.perf_counter()
        joined = True
        join_step1_writers(write_handles, avg_res, ini_res)
        timings["join_writes"] = time.perf_counter() - t
    finally:
        if not joined:
            try:
                join_step1_writers(write_handles, avg_res, ini_res)
            except BaseException:      # noqa: BLE001 — never mask
                pass                   # the failure being propagated

    timings["__wall__"] = time.perf_counter() - t_total
    del avg_res, ini_res, packed_sink, write_handles
    if is_gpu:
        _release_cupy_pool()
    return timings


def profile_run(args) -> Dict[str, object]:
    src = Path(args.source_project)
    if args.temp_out:
        out = Path(args.temp_out)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = Path(tempfile.mkdtemp(prefix="step1_profile_", dir=str(src.parent)))

    _info(f"  source: {src}")
    _info(f"  out:    {out}")
    _stage_inputs(src, out, args.sessions)

    if args.prewarm and args.backend == "gpu":
        t = time.perf_counter()
        prewarm_step1_gpu(
            bold_pairs_for_prewarm(out, args.subjects, args.sessions),
            background=False)
        _info(f"  prewarm (off the timed path): {time.perf_counter() - t:.2f} s")

    runs: List[Dict[str, float]] = []
    for r in range(int(args.runs)):
        # Wipe outputs between runs so each one re-runs from scratch.
        if r > 0:
            for sub in ("profiles/avg_profile", "group", "spatial_mask"):
                p = out / sub
                if p.exists():
                    shutil.rmtree(p)
            # Also wipe per-session profiles so generate_profiles redoes work.
            if "generate_profiles" not in args.skip:
                for sub_id in args.subjects:
                    sub_dir = out / "profiles_raw" / f"sub{sub_id}"
                    if sub_dir.exists():
                        shutil.rmtree(sub_dir)

        timings = _run_once(out, args)
        runs.append(timings)
        _info(f"  [run {r+1}/{args.runs}] total wall: {timings['__wall__']:.3f} s"
              + ("   (cold)" if r == 0 else ""))
    _release_cupy_pool(pinned=True)

    # Aggregate.
    keys = sorted({k for r in runs for k in r if not k.startswith("__")})
    agg: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vs = np.array([r.get(k, 0.0) for r in runs])
        agg[k] = {
            "mean": float(vs.mean()),
            "min": float(vs.min()),
            "max": float(vs.max()),
        }
    walls = np.array([r["__wall__"] for r in runs])
    agg["TOTAL"] = {
        "mean": float(walls.mean()),
        "min": float(walls.min()),
        "max": float(walls.max()),
    }
    warm = [r for r in runs[1:]] or runs
    agg["TOTAL_WARM"] = {
        "mean": float(np.mean([r["__wall__"] for r in warm])),
        "min": float(np.min([r["__wall__"] for r in warm])),
        "max": float(np.max([r["__wall__"] for r in warm])),
    }
    return {"agg": agg, "runs": runs, "out_dir": str(out)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-project", default=str(_DEFAULT_PROJECT))
    p.add_argument("--temp-out", default=None)
    p.add_argument("--subjects", nargs="*", default=["1"])
    p.add_argument("--sessions", nargs="*",
                   default=["1", "2", "3", "4", "5", "6"])
    p.add_argument("--targ-mesh", default="fsaverage6")
    p.add_argument("--seed-mesh", default="fsaverage3")
    p.add_argument("--schaefer-resolution", default="300")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["generate_profiles", "avg_profiles",
                            "ini_params", "radius_mask"])
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"],
                   help="Compute backend (per-subgraph dispatch; falls "
                        "back to CPU for subgraphs without a GPU port).")
    p.add_argument("--prewarm", action="store_true",
                   help="GPU: pay the one-time RawKernel / cuBLAS / nvCOMP "
                        "costs before the timed runs (the driver does this "
                        "on a background thread).")
    args = p.parse_args(argv)

    _section(f"Step1 profile — runs={args.runs}, backend={args.backend}, "
             f"subjects={args.subjects}, sessions={args.sessions}")
    res = profile_run(args)
    agg = res["agg"]   # type: ignore

    _section("Per-stage wall-clock (mean over runs; run 1 is cold)")
    total_mean = float(agg["TOTAL"]["mean"])
    for stage in _STAGES:
        if stage in agg:
            mean = float(agg[stage]["mean"])
            mn = float(agg[stage]["min"])
            mx = float(agg[stage]["max"])
            _info(f"  {stage:24s}  {mean:7.3f} s   "
                  f"[{mn:.3f}, {mx:.3f}]   {_fmt_pct(mean, total_mean)}")
    _info(f"  {'TOTAL':24s}  {total_mean:7.3f} s   "
          f"[{float(agg['TOTAL']['min']):.3f}, "
          f"{float(agg['TOTAL']['max']):.3f}]")
    _info(f"  {'TOTAL (warm runs)':24s}  {float(agg['TOTAL_WARM']['mean']):7.3f} s   "
          f"[{float(agg['TOTAL_WARM']['min']):.3f}, "
          f"{float(agg['TOTAL_WARM']['max']):.3f}]")
    _info(f"\n  out_dir: {res['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
