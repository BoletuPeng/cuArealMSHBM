"""profile.py

Per-stage timing report for the step-3 super-call. Runs the Python
pipeline on a single subject across the requested backends and prints
a per-stage wall-time breakdown of the EM body.

Usage::

    python -m arealmshbm.step3_pipeline.profile
    python -m arealmshbm.step3_pipeline.profile --subject sub-001
    python -m arealmshbm.step3_pipeline.profile --backends cpu       # skip GPU
    python -m arealmshbm.step3_pipeline.profile --warmup             # JIT warmup before timing
    python -m arealmshbm.step3_pipeline.profile --runs 3             # avg over 3 EM runs

Output per backend:
    * Load + setup wall (one-shot per pipeline).
    * EM wall (one full ``intra_em`` outer loop).
    * Per-stage breakdown of the EM body (m_step, e_step_lambda_loop,
      check_connectedness, spatial_xyz_prior, spatial_connect_prior,
      em_stop_criterion).
    * Iteration counts.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arealmshbm.step3_pipeline import Step3Config, Step3Pipeline


_DEFAULT_PROJECT_ROOT = Path("profile_runs/modeA")


def _info(msg: str = "") -> None:
    print(msg, flush=True)


def _section(title: str) -> None:
    bar = "═" * 80
    _info(f"\n{bar}\n{title}\n{bar}")


def _fmt_pct(n: float, d: float) -> str:
    return f"{(100.0 * n / d if d else 0.0):5.1f}%"


def profile_backend(sub: str, backend: str, *,
                    runs: int, warmup: bool, args
                    ) -> Optional[Dict[str, object]]:
    """Run one (subject, backend) configuration and report timings.

    Returns a dict suitable for cross-backend comparison, or None if
    the backend is unavailable (cupy missing for GPU runs).
    """
    if backend.startswith("gpu"):
        try:
            import cupy  # noqa: F401
        except ImportError:
            _info(f"[skip {backend}] cupy not installed")
            return None

    project_dir = Path(args.project_root) / sub
    if not project_dir.exists():
        _info(f"[skip {backend}] project_dir missing: {project_dir}")
        return None

    cfg = Step3Config(
        project_dir=project_dir,
        num_session=int(args.num_session),
        num_clusters=int(args.num_clusters),
        subid=1,
        mesh=args.mesh,
        w=float(args.w),
        c=float(args.c),
        beta_scalar=float(args.beta),
        backend=backend,
    )

    if warmup and backend == "cpu":
        from arealmshbm.vmf_clustering import warmup as wm
        t0 = time.perf_counter()
        wm()
        _info(f"  numba warmup: {time.perf_counter() - t0:.2f} s")

    _section(f"{sub}  backend={backend}  (runs={runs})")

    pipe = Step3Pipeline(cfg)
    pipe.load_inputs()
    pipe.build_session()
    load_total = pipe.timings.get("load_total", 0.0)
    sess_init = pipe.timings.get("session_init", 0.0)
    _info(f"  Setup wall:")
    _info(f"    load_inputs       {load_total:7.3f} s")
    _info(f"    session_init      {sess_init:7.3f} s")
    for k in ("load_group_prior", "load_spatial_mask", "load_avg_mesh",
              "fetch_data", "initialize_concentration",
              "initialize_params", "build_pipeline_setup"):
        v = pipe.timings.get(k, 0.0)
        _info(f"      {k:24s} {v:7.3f} s")

    em_walls = []
    last_stage_breakdown: Dict[str, float] = {}
    last_iters: Dict[str, int] = {}
    for r in range(runs):
        # Recreate the Pipeline between runs so the cupy memory pool
        # actually releases blocks. The prior in-place reset
        # (``pipe._inputs = None; ...; pipe.load_inputs()``) only dropped
        # Python references — the cupy default pool keeps free blocks
        # for reuse, so peak GPU usage accumulated across ``--runs N``.
        # ``close()`` is what calls ``free_all_blocks()``.
        if r > 0:
            pipe.close()
            pipe = Step3Pipeline(cfg)
            pipe.load_inputs()
            pipe.build_session()
        t0 = time.perf_counter()
        result = pipe.run(time_stages=True)
        em_walls.append(time.perf_counter() - t0)
        last_stage_breakdown = pipe.timings.get("per_stage_em", {})
        last_iters = {
            k: int(last_stage_breakdown.get(k, 0))
            for k in ("iter_count_em_total", "iter_count_m_total",
                      "iter_count_lambda_total", "iter_count_comp_total",
                      "iter_count_check_conn", "iter_count_spatial_xyz")
        }

    em_mean = float(np.mean(em_walls))
    em_min = float(np.min(em_walls))
    _info(f"  EM wall (runs={runs}):  mean {em_mean:.3f} s   min {em_min:.3f} s")

    stages = ["m_step", "spatial_connect_prior", "e_step_lambda_loop",
              "check_connectedness", "spatial_xyz_prior", "em_stop_criterion"]
    em_total = sum(last_stage_breakdown.get(s, 0.0) for s in stages)
    _info(f"  Per-stage breakdown (last run, sum over intra_em rounds):")
    for s in stages:
        v = float(last_stage_breakdown.get(s, 0.0))
        _info(f"    {s:24s} {v:7.3f} s   {_fmt_pct(v, em_total)}")
    _info(f"  Iter counts: em_total={last_iters.get('iter_count_em_total')}  "
          f"m_total={last_iters.get('iter_count_m_total')}  "
          f"lambda_total={last_iters.get('iter_count_lambda_total')}  "
          f"comp_total={last_iters.get('iter_count_comp_total')}")

    out = {
        "subject": sub,
        "backend": backend,
        "load_total": load_total,
        "session_init": sess_init,
        "em_wall_mean": em_mean,
        "em_wall_min": em_min,
        "stage_breakdown": dict(last_stage_breakdown),
        "iters": last_iters,
        "iter_intra_em": int(result.iter_intra_em),
    }
    # Release Session + Inputs and the cupy memory pool. Without this,
    # successive (subject, backend) profile_backend calls in the same
    # process keep ~4 GB of GPU pool blocks across runs.
    pipe.close()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", default="sub-001")
    p.add_argument("--backends", nargs="*",
                   default=["cpu", "gpu_full"],
                   choices=["cpu", "gpu_elambda", "gpu_full"])
    p.add_argument("--project-root", default=str(_DEFAULT_PROJECT_ROOT))
    p.add_argument("--mesh", default="fsaverage6")
    p.add_argument("--w", type=float, default=50.0)
    p.add_argument("--c", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--num-clusters", type=int, default=300)
    p.add_argument("--num-session", type=int, default=6)
    p.add_argument("--runs", type=int, default=1,
                   help="number of EM runs to average (default 1)")
    p.add_argument("--warmup", action="store_true",
                   help="warm numba kernels before CPU timing")
    args = p.parse_args(argv)

    results: List[Dict[str, object]] = []
    for backend in args.backends:
        r = profile_backend(args.subject, backend, runs=args.runs,
                            warmup=args.warmup, args=args)
        if r is not None:
            results.append(r)

    # Cross-backend comparison.
    if len(results) > 1:
        _section("Cross-backend comparison")
        _info(f"  {'stage':<26} " + "  ".join(f"{r['backend']:>14}" for r in results))
        stages = ["m_step", "spatial_connect_prior", "e_step_lambda_loop",
                  "check_connectedness", "spatial_xyz_prior", "em_stop_criterion"]
        for s in stages:
            row = [f"{float(r['stage_breakdown'].get(s, 0.0)):14.3f}" for r in results]  # type: ignore[arg-type]
            _info(f"  {s:<26} " + "  ".join(row) + " s")
        em = [f"{float(r['em_wall_mean']):14.3f}" for r in results]
        _info(f"  {'EM wall (mean)':<26} " + "  ".join(em) + " s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
