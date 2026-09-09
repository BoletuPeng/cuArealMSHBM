"""profile.py

Per-stage timing report for one step-2 super-call. Runs
:class:`arealmshbm.step2_pipeline.Step2Pipeline` ``--runs`` times inside
ONE process against a real project and reports each stage's mean and
span, so run 1 (cold: NVRTC compiles, cuBLAS handle, numba cache loads,
OS page cache) and runs 2+ (warm) are visible separately.

Stages reported (``Step2Pipeline.timings`` keys):

    load        load_inputs            — cohort + BOLD + gradient + group.mat
                                         + spatial mask → layout
    init        initialize_params      — ini_val + host bookkeeping
    ctor        session_ctor           — device buffers + module compile
    init_device init_device            — K1 init on device (gpu backend)
    em_total    em_total               — the accumulated EM-body wall; the
                                         per-EM-iteration mean uses
                                         ``Step2Result.em_iters_total`` (the
                                         real ``run_iter`` count), and the
                                         per-EM-body mean the number of
                                         EM-body calls
    closure     closure_total          — L16/L17/L18 (gpu backend)
    save        save                   — Params_Final.mat
    total       total                  — whole run()

The GPU Session's own host-wall timings (``session_ctor`` /
``initialize_state`` / ``intra_closure`` / ``inter_closure``) reach the
pipeline under ``session.<name>``, in SECONDS, each CUMULATIVE over
every call across the whole run — printed in their own table so a total
summed over 20 calls is not read as a per-call cost.

Usage::

    python -m arealmshbm.step2_pipeline.profile \\
        --project testdata/step2_bench/proj --backend gpu --runs 5
    python -m arealmshbm.step2_pipeline.profile \\
        --project testdata/step2_bench/proj --backend gpu --runs 5 --prewarm
    python -m arealmshbm.step2_pipeline.profile \\
        --project testdata/step2_bench/proj2 --num-sub 2 --backend cpu --runs 2
    python -m arealmshbm.step2_pipeline.profile \\
        --project testdata/step2_bench/proj --backend gpu --runs 3 --fresh-process

``--fresh-process`` additionally spawns a subprocess that runs the
pipeline exactly once and reports its wall — the number a driver run
actually pays on a cold interpreter.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arealmshbm.step2_pipeline.config import Step2Config          # noqa: E402
from arealmshbm.step2_pipeline.pipeline import Step2Pipeline      # noqa: E402


_DEFAULT_PROJECT = Path("testdata/step2_bench/proj")

# (label, timings key) in report order.
_STAGES = (
    ("load", "load_inputs"),
    ("init", "initialize_params"),
    ("ctor", "session_ctor"),
    ("init_device", "init_device"),
    ("em_total", "em_total"),
    ("closure", "closure_total"),
    ("save", "save"),
    ("total", "total"),
)


def _info(msg: str = "") -> None:
    print(msg, flush=True)


def _section(title: str) -> None:
    bar = "=" * 78
    _info(f"\n{bar}\n{title}\n{bar}")


def _release_cupy_pool(*, pinned: bool = False) -> None:
    """Free the cupy device pool (and optionally the pinned pool). No-op
    without cupy / on the CPU backend."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError:
        return
    cp.get_default_memory_pool().free_all_blocks()
    if pinned:
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _build_cfg(args, out_dir: Path) -> Step2Config:
    return Step2Config(
        project_dir=Path(args.project),
        num_sub=int(args.num_sub),
        num_session=int(args.num_session),
        num_clusters=int(args.num_clusters),
        mode=args.mode,
        beta_scalar=float(args.beta),
        backend=args.backend,
        out_dir=out_dir,
        verbose=False,
    )


def _run_once(args, out_dir: Path) -> Dict[str, object]:
    """One full ``Step2Pipeline.run()``. Returns its timings dict plus
    the EM iteration bookkeeping from the Step2Result."""
    cfg = _build_cfg(args, out_dir)
    t0 = time.perf_counter()
    with Step2Pipeline(cfg) as pipe:
        res = pipe.run()
    wall = time.perf_counter() - t0
    timings = dict(pipe.timings)
    timings["__wall__"] = wall
    return {
        "timings": timings,
        "inter_iters": int(res.inter_iters),
        "intra_em_iters_per_inter": list(res.intra_em_iters_per_inter),
        "em_iters_total": int(getattr(res, "em_iters_total", 0)),
    }


def _agg(values: List[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "min": float(v.min()),
            "max": float(v.max())}


def _fresh_process_wall(args, out_dir: Path) -> Optional[float]:
    """Spawn a subprocess that runs the pipeline exactly once; return its
    reported wall (seconds), or None if it failed."""
    code = (
        "import json,sys,time\n"
        f"sys.path.insert(0, {str(_REPO)!r})\n"
        "from arealmshbm.step2_pipeline import Step2Config, Step2Pipeline\n"
        f"cfg = Step2Config(project_dir={str(args.project)!r}, "
        f"num_sub={int(args.num_sub)}, num_session={int(args.num_session)}, "
        f"num_clusters={int(args.num_clusters)}, mode={args.mode!r}, "
        f"beta_scalar={float(args.beta)!r}, backend={args.backend!r}, "
        f"out_dir={str(out_dir)!r}, verbose=False)\n"
        "t0 = time.perf_counter()\n"
        "with Step2Pipeline(cfg) as p:\n"
        "    p.run()\n"
        "print('__FRESH__' + json.dumps({'wall': time.perf_counter() - t0, "
        "'timings': p.timings}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("__FRESH__"):
            return float(json.loads(line[len("__FRESH__"):])["wall"])
    _info(f"  fresh-process run failed (rc={proc.returncode}):\n"
          f"{proc.stderr[-2000:]}")
    return None


def profile_run(args) -> Dict[str, object]:
    out_dir = Path(args.out) if args.out else (
        Path(args.project) / "profile_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    _info(f"  project: {args.project}")
    _info(f"  out:     {out_dir}")

    if args.prewarm and args.backend == "gpu":
        t = time.perf_counter()
        from arealmshbm.pipeline.step2_runners import prewarm_step2_gpu
        prewarm_step2_gpu(background=False)
        _info(f"  prewarm (off the timed path): "
              f"{time.perf_counter() - t:.2f} s")

    runs: List[Dict[str, object]] = []
    for r in range(int(args.runs)):
        rec = _run_once(args, out_dir)
        runs.append(rec)
        tm = rec["timings"]                       # type: ignore[index]
        _info(f"  [run {r+1}/{args.runs}] total wall: "
              f"{tm['__wall__']:.3f} s"           # type: ignore[index]
              + ("   (cold)" if r == 0 else ""))
        if args.backend != "cpu":
            _release_cupy_pool()
    if args.backend != "cpu":
        _release_cupy_pool(pinned=True)

    fresh_wall = None
    if args.fresh_process:
        fresh_wall = _fresh_process_wall(args, out_dir)

    return {"runs": runs, "out_dir": str(out_dir), "fresh_wall": fresh_wall}


def _report(args, res: Dict[str, object]) -> None:
    runs: List[Dict[str, object]] = res["runs"]        # type: ignore[assignment]
    all_t = [r["timings"] for r in runs]               # type: ignore[index]
    warm = all_t[1:] or all_t

    _section("Per-stage wall (mean [min, max]; run 1 is cold, "
             "warm = runs 2+)")
    for label, key in _STAGES:
        vals = [float(t[key]) for t in all_t if key in t]   # type: ignore[index]
        if not vals:
            continue
        a = _agg(vals)
        w = _agg([float(t[key]) for t in warm if key in t])  # type: ignore[index]
        _info(f"  {label:12s}  all {a['mean']:8.4f} s "
              f"[{a['min']:.4f}, {a['max']:.4f}]   "
              f"warm {w['mean']:8.4f} s "
              f"[{w['min']:.4f}, {w['max']:.4f}]")

    # EM iteration bookkeeping. Two denominators, both reported:
    #   EM-body calls = Σ intra_em_iters_per_inter (one em_body_* call each)
    #   EM iters      = Step2Result.em_iters_total (one run_iter each) — the
    #                   real per-iteration denominator.
    _section("EM iterations")
    for i, r in enumerate(runs):
        intra = r["intra_em_iters_per_inter"]           # type: ignore[index]
        tm = r["timings"]                               # type: ignore[index]
        n_batches = int(sum(intra))                     # type: ignore[arg-type]
        n_iters = int(r.get("em_iters_total", 0))       # type: ignore[union-attr]
        em = float(tm["em_total"])                      # type: ignore[index]
        per_iter = (f"{em / n_iters * 1e3:.2f} ms" if n_iters
                    else "n/a (em_iters_total not reported)")
        _info(f"  [run {i+1}] inter={r['inter_iters']}  "     # type: ignore[index]
              f"intra/inter={list(intra)}  "
              f"EM-body calls={n_batches}  "
              f"EM iters={n_iters}  "
              f"em_total={em:.4f} s  "
              f"mean/EM-body={em / max(1, n_batches) * 1e3:.2f} ms  "
              f"mean/EM-iter={per_iter}")

    # Loader sub-items (means over warm runs).
    sub_keys = sorted({k for t in warm for k in t                 # type: ignore[union-attr]
                       if k.startswith("load_inputs.")})
    if sub_keys:
        _section("load_inputs breakdown (warm mean)")
        for k in sub_keys:
            vals = [float(t[k]) for t in warm if k in t]          # type: ignore[index]
            _info(f"  {k[len('load_inputs.'):]:24s} {np.mean(vals)*1e3:8.2f} ms")

    # Session timings; they arrive from the pipeline in seconds, printed
    # in ms.
    wall_keys = sorted({k for t in warm for k in t                # type: ignore[union-attr]
                        if k.startswith("session.")})

    if wall_keys:
        _section("Session cumulative wall (total over every call in the run, "
                 "warm mean over runs)")
        for k in wall_keys:
            vals = [float(t[k]) for t in warm if k in t]          # type: ignore[index]
            _info(f"  {k[len('session.'):]:24s} {np.mean(vals)*1e3:8.3f} ms "
                  f"(cumulative)")

    if res.get("fresh_wall") is not None:
        _info(f"\n  fresh-process total (1 run, cold interpreter): "
              f"{float(res['fresh_wall']):.3f} s")   # type: ignore[arg-type]
    _info(f"\n  out_dir: {res['out_dir']}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default=str(_DEFAULT_PROJECT))
    p.add_argument("--backend", default="gpu",
                   choices=["cpu", "gpu"])
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--num-sub", type=int, default=1)
    p.add_argument("--num-session", type=int, default=6)
    p.add_argument("--num-clusters", type=int, default=300)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--mode", default="gMSHBM", choices=["gMSHBM", "dMSHBM"])
    p.add_argument("--out", default=None,
                   help="output dir for Params_Final.mat "
                        "(default <project>/profile_out)")
    p.add_argument("--prewarm", action="store_true",
                   help="GPU: compile the RawModule and warm cuBLAS before "
                        "the timed runs (as the driver does).")
    p.add_argument("--fresh-process", action="store_true",
                   help="also time one run in a freshly spawned process.")
    args = p.parse_args(argv)

    _section(f"Step2 profile - backend={args.backend}, runs={args.runs}, "
             f"S={args.num_sub}, T={args.num_session}, L={args.num_clusters}, "
             f"mode={args.mode}")
    res = profile_run(args)
    _report(args, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
