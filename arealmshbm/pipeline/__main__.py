# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""CLI entry: ``python -m arealmshbm.pipeline <project_dir>``.

The project_dir is expected to already contain ``bold_inputs.json`` +
``pipeline_config.json``. Everything else is produced by the run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .driver import Pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m arealmshbm.pipeline",
        description=(
            "Run the unified MSHBM pipeline driver against a project "
            "directory. The project must contain bold_inputs.json + "
            "pipeline_config.json; everything else is produced by the run."
        ),
    )
    p.add_argument(
        "project_dir",
        type=Path,
        help="Path to projects/<name>/ (relative or absolute).",
    )
    args = p.parse_args(argv)

    project_dir = args.project_dir
    if not project_dir.exists():
        print(f"error: project_dir not found: {project_dir}", file=sys.stderr)
        return 2

    result = Pipeline(project_dir).run()

    print(f"\n[pipeline] done — mode={result.mode} variant={result.variant} "
          f"total={result.timings.get('total', 0):.1f}s")
    print(f"[pipeline] cohort: {result.cohort_json_path}")
    print(f"[pipeline] prior:  {result.prior_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
