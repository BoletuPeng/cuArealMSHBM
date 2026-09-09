"""annot_io.py — minimal FreeSurfer ``.annot`` label reader (numpy only).

Step 1 reads two kinds of ``.annot`` files live from ``MSHBM_ATLAS_DIR``:
the Schaefer group parcellation (``resolve_group_labels``) and the DK
``aparc`` (``radius_mask._read_aparc``). Both consumers only need the
per-vertex **label index** vector — the colour table and the structure
names are discarded.

``nibabel.freesurfer.io.read_annot`` returns exactly that vector, but
importing ``nibabel`` costs ~0.8 s cold — more than a whole
single-subject step-1 GPU run — so the parse is re-implemented here on
``struct`` + numpy. Output is asserted equal to nibabel's ``labels``
array in ``tests/test_annot_io.py``.

Format (FreeSurfer ``read_annotation.m`` / nibabel ``read_annot``):

    int32 BE  n_vertices
    int32 BE  [vertex_id, rgba_key] * n_vertices      (interleaved)
    int32 BE  has_ctab (1 = present)
    if present:
        int32 BE  n_entries
        > 0  : "old" format — not supported (see below)
        < 0  : "new" format — version = -n_entries (must be 2),
                 int32 n_entries, int32 len, bytes orig_tab[len],
                 then n_entries × (int32 index, int32 len, bytes name,
                 int32 r,g,b,t)

Only the version-2 table is read: every ``.annot`` this project ships
(Schaefer + aparc) is version 2, and an untested old-format branch is
worse than a ``NotImplementedError``.

The per-vertex ``labels`` array is the index into the colour table of
each vertex's annotation value (``r + g*256 + b*65536`` of the table
row; the alpha byte is NOT part of the key), with ``-1`` wherever the
annotation value is 0 — nibabel's ``read_annot(orig_ids=False)``
semantics, which is what both consumers rely on (``-1`` → 0
normalisation happens at the call sites). Unlike nibabel, a non-zero
value absent from the table is a ``ValueError`` rather than a silent
mis-index.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_annot_labels(path: "str | Path") -> np.ndarray:
    """Per-vertex colour-table index of a FreeSurfer ``.annot`` file.

    Returns an ``(n_vertices,) int32`` array equal to the first element
    of ``nibabel.freesurfer.io.read_annot(path)`` (``-1`` for vertices
    whose annotation value is 0). Vertices are returned in vertex-id
    order regardless of the on-disk order. Raises ``ValueError`` on a
    non-zero annotation value that is absent from the colour table.
    """
    p = Path(path)
    with open(p, "rb") as f:
        buf = f.read()
    off = 0
    (n_vert,) = struct.unpack_from(">i", buf, off)
    off += 4
    if n_vert < 0:
        raise ValueError(f"annot {p}: negative vertex count {n_vert}")
    pairs = np.frombuffer(buf, dtype=">i4", count=2 * n_vert, offset=off)
    off += 8 * n_vert
    vid = pairs[0::2].astype(np.int64)
    keys = pairs[1::2].astype(np.int64)
    # Vertex ids are 0..n-1 in file order for every atlas we ship, but
    # the format does not promise it, so the scatter is checked: an id
    # outside range, or a vertex no row ever mentions, would otherwise
    # leave uninitialised memory in the label vector.
    ann = np.zeros(n_vert, dtype=np.int64)
    if n_vert:
        if int(vid.min()) < 0 or int(vid.max()) >= n_vert:
            raise ValueError(
                f"annot {p}: vertex id outside [0, {n_vert}) "
                f"(min {int(vid.min())}, max {int(vid.max())})")
        ann[vid] = keys
        seen = np.zeros(n_vert, dtype=bool)
        seen[vid] = True
        if not seen.all():
            raise ValueError(
                f"annot {p}: {int((~seen).sum())} vertex/vertices never "
                f"listed — the file does not cover every vertex")

    if off + 4 > len(buf):
        # No colour table: nibabel would return the raw annotation
        # values as labels with ``ctab=None``; neither consumer here
        # can use that (both index a table), so refuse loudly.
        raise ValueError(f"annot {p}: no colour table present")
    (has_ctab,) = struct.unpack_from(">i", buf, off)
    off += 4
    if has_ctab != 1:
        raise ValueError(f"annot {p}: has_ctab={has_ctab}, expected 1")
    (n_entries,) = struct.unpack_from(">i", buf, off)
    off += 4

    if n_entries > 0:
        raise NotImplementedError(
            f"annot {p}: old-style colour table (n_entries={n_entries}); "
            f"only the version-2 table is supported — the shipped atlases "
            f"are version 2")
    version = -n_entries
    if version != 2:
        raise ValueError(
            f"annot {p}: unsupported colour-table version {version}")
    (n_table,) = struct.unpack_from(">i", buf, off)
    off += 4
    (orig_len,) = struct.unpack_from(">i", buf, off)
    off += 4 + orig_len
    (n_entries_to_read,) = struct.unpack_from(">i", buf, off)
    off += 4
    table_keys = [None] * n_table
    for _ in range(n_entries_to_read):
        (idx,) = struct.unpack_from(">i", buf, off)
        off += 4
        (name_len,) = struct.unpack_from(">i", buf, off)
        off += 4 + name_len
        r, g, b, _t = struct.unpack_from(">4i", buf, off)
        off += 16
        if idx < 0 or idx >= n_table:
            raise ValueError(
                f"annot {p}: colour-table entry index {idx} outside "
                f"[0, {n_table})")
        table_keys[idx] = r + g * 256 + b * 65536
    if any(k is None for k in table_keys):
        # nibabel fills unread entries with zeros; the key of such an
        # entry would be 0 (r=g=b=t=0). Mirror that so a vertex
        # annotated 0 maps to the same index nibabel reports.
        table_keys = [0 if k is None else k for k in table_keys]

    tab = np.asarray(table_keys, dtype=np.int64)
    # nibabel semantics (read_annot, orig_ids=False):
    #   ord = argsort(ctab_keys); labels[ann == 0] = -1;
    #   labels[ann != 0] = ord[searchsorted(ctab_keys[ord], ann[ann != 0])]
    # i.e. annotation value 0 is ALWAYS -1 (even when the table has a
    # key-0 row such as a2009s' "Unknown"), and every other value is
    # looked up by sorted position. nibabel does not check that the
    # value is actually present (an absent key silently maps to the
    # next row); both consumers here index a table with the result, so
    # an absent non-zero key is refused instead.
    order = np.argsort(tab, kind="stable")
    tab_sorted = tab[order]
    labels = np.full(n_vert, -1, dtype=np.int32)
    nz = ann != 0
    if nz.any():
        vals = ann[nz]
        pos = np.searchsorted(tab_sorted, vals)
        pos_c = np.minimum(pos, tab_sorted.shape[0] - 1)
        missing = tab_sorted[pos_c] != vals
        if missing.any():
            bad = int(vals[np.flatnonzero(missing)[0]])
            raise ValueError(
                f"annot {p}: annotation value {bad} is not in the colour "
                f"table ({tab.shape[0]} entries)")
        labels[nz] = order[pos_c].astype(np.int32)
    return labels


__all__ = ["read_annot_labels"]
