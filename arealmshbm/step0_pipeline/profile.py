"""profile.py

Per-stage wall-time breakdown of the Python step-0 super-call. Mirrors
:mod:`arealmshbm.step3_pipeline.profile` in shape — one report
per subject listing each leaf's accumulated time across the full run.

Usage::

    python -m arealmshbm.step0_pipeline.profile
    python -m arealmshbm.step0_pipeline.profile --subjects sub-001

Stages reported:

    subgraph A (RSFC gradients):
      fc_similarity, surface_gradient, surface_smoothing,
      local_minima, watershed
    subgraph B (gradient distance per hemi)
    subgraph C (diffusion embedding per hemi)
    subgraph D (upsample per hemi)

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arealmshbm.step0_pipeline import Step0Config, Step0Pipeline


_PROJECT_ROOT = Path("profile_runs/modeA")
_DEFAULT_SESS = ("ses-01", "ses-02", "ses-03", "ses-04", "ses-05", "ses-06")


def _info(msg: str) -> None:
    print(msg, flush=True)


def run_subject(sub: str, args) -> Optional[dict]:
    project_dir = Path(args.project_root) / sub
    if not project_dir.exists():
        _info(f"[skip] {sub}: project_dir missing ({project_dir})")
        return None

    cfg = Step0Config(
        project_dir=project_dir,
        sub_id=sub,
        sess_list=tuple(args.sess_list),
        save_artifacts=False,
        backend=args.backend,
    )
    timings = {}
    t_total = time.perf_counter()
    with Step0Pipeline(cfg) as pipe:
        t0 = time.perf_counter()
        inputs = pipe.load_inputs()
        timings["load_inputs"] = time.perf_counter() - t0
        result = pipe.run(inputs=inputs, time_stages=True)
    timings["wall_total"] = time.perf_counter() - t_total
    timings["per_stage"] = result.timings.get("per_stage", {})
    timings["meta"] = {
        "n_cortex": int(inputs.n_cortex),
        "iter_a":   int(result.timings.get("iter_a", 0)),
        "block_size_a": int(result.timings.get("block_size_a", 0)),
        "n_down":   int(result.timings.get("n_down_sphere", 0)),
        "num_sess": cfg.num_sess,
    }
    return timings


def _format_report(sub: str, t: dict) -> str:
    lines = []
    lines.append(f"\n=== {sub} ({t['meta']['num_sess']} sess, "
                 f"N_cortex={t['meta']['n_cortex']}, "
                 f"N_down={t['meta']['n_down']}) ===")
    lines.append(f"  load_inputs            {t['load_inputs']:8.2f} s")
    s = t["per_stage"]
    a_total = s.get("subgraph_A_total", 0.0)
    b_total = s.get("subgraph_B_total", 0.0)
    c_total = s.get("subgraph_C_total", 0.0)
    d_total = s.get("subgraph_D_total", 0.0)
    wall = t["wall_total"]
    lines.append(f"  subgraph A (RSFC gradients)        {a_total:8.2f} s "
                 f"({100*a_total/wall:5.1f}%)")
    for k in ("bold_io_wait", "fc_similarity", "surface_gradient",
              "surface_smoothing", "local_minima", "watershed"):
        v = s.get(k, 0.0)
        lines.append(f"    {k:30s} {v:8.2f} s")
    lines.append(f"  subgraph B (gradient distance)     {b_total:8.2f} s "
                 f"({100*b_total/wall:5.1f}%)")
    lines.append(f"  subgraph C (diffusion embedding)   {c_total:8.2f} s "
                 f"({100*c_total/wall:5.1f}%)")
    lines.append(f"  subgraph D (upsample emb)          {d_total:8.2f} s "
                 f"({100*d_total/wall:5.1f}%)")
    lines.append(f"  WALL TOTAL                         {wall:8.2f} s")
    cands = [(k, s.get(k, 0.0)) for k in
             ("bold_io_wait", "fc_similarity", "surface_gradient",
              "surface_smoothing", "local_minima", "watershed")]
    cands += [("subgraph_B_total", b_total),
              ("subgraph_C_total", c_total),
              ("subgraph_D_total", d_total)]
    cands.sort(key=lambda kv: -kv[1])
    lines.append("  Top 3 hotspots:")
    for k, v in cands[:3]:
        lines.append(f"    {k:24s} {v:8.2f} s ({100*v/wall:5.1f}%)")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="*", default=["sub-001"])
    p.add_argument("--project-root", default=str(_PROJECT_ROOT))
    p.add_argument("--sess-list", nargs="*", default=list(_DEFAULT_SESS))
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    args = p.parse_args(argv)

    overall = []
    for sub in args.subjects:
        t = run_subject(sub, args)
        if t is None:
            continue
        overall.append((sub, t))
        _info(_format_report(sub, t))
    if not overall:
        _info("\nNo subjects profiled.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
