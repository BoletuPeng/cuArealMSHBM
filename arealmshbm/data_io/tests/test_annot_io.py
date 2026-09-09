"""test_annot_io.py — ``read_annot_labels`` ≡ nibabel ``read_annot`` labels.

Two layers:

* synthetic version-2 ``.annot`` files written here, covering
  annotation value 0 → -1, out-of-order vertex ids, and the
  ``r + g*256 + b*65536`` key with a non-zero alpha byte that must NOT
  enter the key; plus the refusals (absent key, old-format colour
  table, a vertex no row mentions);
* every ``.annot`` under ``$MSHBM_ATLAS_DIR`` (the Schaefer + aparc
  files step 1 actually reads), compared against nibabel when both the
  env var and nibabel are available.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from arealmshbm.data_io.annot_io import read_annot_labels


def _write_annot(path: Path, ann_values: np.ndarray, rows, *,
                 old_format: bool = False, vertex_order=None) -> None:
    """``rows`` = list of (name, r, g, b, a). ``vertex_order`` permutes
    the on-disk vertex sequence (ids are stored explicitly).
    ``old_format`` writes the positive-``n_entries`` table the reader
    refuses — it exists only to pin that refusal."""
    n = int(ann_values.shape[0])
    vid = np.arange(n) if vertex_order is None else np.asarray(vertex_order)
    out = bytearray()
    out += struct.pack(">i", n)
    pairs = np.empty(2 * n, dtype=">i4")
    pairs[0::2] = vid
    pairs[1::2] = ann_values[vid]
    out += pairs.tobytes()
    out += struct.pack(">i", 1)                      # has ctab
    if old_format:
        out += struct.pack(">i", len(rows))
        for name, r, g, b, a in rows:
            nb = name.encode() + b"\0"
            out += struct.pack(">i", len(nb)) + nb
            out += struct.pack(">4i", r, g, b, a)
    else:
        out += struct.pack(">i", -2)                 # version 2
        out += struct.pack(">i", len(rows))
        orig = b"synthetic.ctab\0"
        out += struct.pack(">i", len(orig)) + orig
        out += struct.pack(">i", len(rows))
        for idx, (name, r, g, b, a) in enumerate(rows):
            nb = name.encode() + b"\0"
            out += struct.pack(">i", idx)
            out += struct.pack(">i", len(nb)) + nb
            out += struct.pack(">4i", r, g, b, a)
    path.write_bytes(bytes(out))


def _key(r, g, b):
    return r + g * 256 + b * 65536


_ROWS = [("Unknown", 0, 0, 0, 0), ("A", 20, 30, 40, 0),
         ("B", 200, 100, 50, 255), ("C", 1, 2, 3, 128)]


def test_synthetic_roundtrip(tmp_path):
    keys = [_key(r, g, b) for _, r, g, b, _ in _ROWS]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(_ROWS), size=257)
    ann = np.asarray(keys, dtype=np.int64)[idx]
    # Vertex ids stored out of order to exercise the scatter.
    perm = rng.permutation(ann.shape[0])
    p = tmp_path / "synthetic.annot"
    _write_annot(p, ann, _ROWS, vertex_order=perm)
    got = read_annot_labels(p)
    expected = idx.astype(np.int32)
    expected[ann == 0] = -1          # nibabel: value 0 is always -1
    assert got.dtype == np.int32
    assert np.array_equal(got, expected)


def test_absent_key_is_refused(tmp_path):
    rows = [("A", 1, 0, 0, 0)]
    ann = np.array([_key(1, 0, 0), _key(2, 0, 0)], dtype=np.int64)
    p = tmp_path / "bad.annot"
    _write_annot(p, ann, rows)
    with pytest.raises(ValueError, match="not in the colour table"):
        read_annot_labels(p)


def test_old_format_colour_table_is_refused(tmp_path):
    ann = np.asarray([_key(20, 30, 40)] * 4, dtype=np.int64)
    p = tmp_path / "old.annot"
    _write_annot(p, ann, _ROWS, old_format=True)
    with pytest.raises(NotImplementedError, match="old-style colour table"):
        read_annot_labels(p)


def test_a_vertex_no_row_mentions_is_refused(tmp_path):
    ann = np.asarray([_key(20, 30, 40)] * 4, dtype=np.int64)
    p = tmp_path / "dup.annot"
    # Vertex 3 is never listed; vertex 0 appears twice.
    _write_annot(p, ann, _ROWS, vertex_order=np.array([0, 1, 2, 0]))
    with pytest.raises(ValueError, match="never listed"):
        read_annot_labels(p)


def _atlas_annots():
    root = os.getenv("MSHBM_ATLAS_DIR")
    if not root:
        return []
    return sorted(Path(root).glob("*/label/*.annot"))


def test_matches_nibabel_on_atlas():
    annots = _atlas_annots()
    if not annots:
        pytest.skip("MSHBM_ATLAS_DIR unset or holds no */label/*.annot")
    fsio = pytest.importorskip("nibabel.freesurfer.io")
    for annot in annots:
        ref, _, _ = fsio.read_annot(str(annot))
        got = read_annot_labels(annot)
        assert np.array_equal(got, ref.astype(np.int32)), annot
