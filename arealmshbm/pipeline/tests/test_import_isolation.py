"""Every ``arealmshbm.*`` sub-package must import standalone.

Regression guard for the circular import that ``arealmshbm/__init__.py``'s
PEP-562 lazy re-export un-masked: ``arealmshbm.vmf_clustering`` imported
``arealmshbm.step3_pipeline.variant`` at module scope, whose package
``__init__`` imported ``step3_pipeline.pipeline``, which imported
``arealmshbm.vmf_clustering`` back — an ImportError in a fresh
interpreter. The eager ``arealmshbm/__init__.py`` used to pre-order the
two, so ``python -m pytest arealmshbm/`` stayed green while
``python -c "import arealmshbm.vmf_clustering"`` and
``pytest arealmshbm/vmf_clustering/tests`` both failed.

The check has to run outside this interpreter: by the time pytest gets
here, conftest / collection has already imported half the tree, which is
exactly the pre-ordering that hides the bug. So we spawn ONE fresh
interpreter that imports every sub-package one at a time, purging
``arealmshbm*`` from ``sys.modules`` between packages so each import
re-executes from scratch, and report every package that fails.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import arealmshbm


_REPO_ROOT = Path(arealmshbm.__file__).resolve().parent.parent


def _subpackages():
    """Names of every sub-package directly under ``arealmshbm/``."""
    return sorted(
        m.name for m in pkgutil.iter_modules(arealmshbm.__path__) if m.ispkg
    )


# The child script: import each package in isolation, print failures.
_CHILD = r"""
import importlib, json, sys

names = json.loads(sys.argv[1])
failures = []
for name in names:
    mod = "arealmshbm." + name
    try:
        importlib.import_module(mod)
    except Exception as exc:            # noqa: BLE001 - we report everything
        failures.append([mod, type(exc).__name__, str(exc)[:300]])
    # Purge so the next package re-executes from scratch rather than
    # picking up a pre-ordered sys.modules (which is what hides cycles).
    for m in [m for m in sys.modules if m.split(".")[0] == "arealmshbm"]:
        del sys.modules[m]
print(json.dumps(failures))
"""


def test_every_subpackage_imports_standalone():
    """``import arealmshbm.<pkg>`` succeeds in a fresh interpreter, for all pkgs."""
    import json

    names = _subpackages()
    assert "vmf_clustering" in names and "step2_pipeline" in names, names

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(names)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"child interpreter died (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The child prints exactly one JSON line last.
    line = proc.stdout.strip().splitlines()[-1]
    failures = json.loads(line)
    assert failures == [], "sub-packages that do not import standalone:\n" + "\n".join(
        f"  {mod}: {kind}: {msg}" for mod, kind, msg in failures
    )


def test_vmf_clustering_imports_standalone():
    """The specific cycle the guard above was written for, on its own."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import arealmshbm.vmf_clustering as m; "
            "assert m.VmfClusteringSession is not None; print('ok')",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
