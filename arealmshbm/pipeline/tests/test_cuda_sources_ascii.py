# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""test_cuda_sources_ascii.py - compiled CUDA sources are pure ASCII.

CuPy writes each source to a temporary ``.cu`` file with the locale
encoding on a kernel-cache miss, so a non-ASCII character in a compiled
string raises ``UnicodeEncodeError`` on a cp1252 box. The guard checks
every string literal that carries a CUDA marker (``__global__``,
``extern "C"``, ``__device__``); every compiled piece in the package,
tests included, carries one. It lives in ``pipeline/tests`` with the
other repo-wide guards.
"""
from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2]
_MARKERS = ("__global__", 'extern "C"', "__device__")


def test_cuda_sources_are_ascii():
    bad = []
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and any(m in node.value for m in _MARKERS)):
                continue
            for i, line in enumerate(node.value.split("\n")):
                if not line.isascii():
                    chars = sorted({c for c in line if not c.isascii()})
                    bad.append(f"{path.relative_to(_PKG.parent)}:"
                               f"{node.lineno + i}: {chars}")
    assert not bad, "non-ASCII in compiled CUDA source:\n" + "\n".join(bad)
