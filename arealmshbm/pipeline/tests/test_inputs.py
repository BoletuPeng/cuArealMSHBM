"""test_inputs.py — pin the bold_inputs.json parse-time contract.

The reader-level GIFTI-only guard at
:func:`arealmshbm.bold_io.read_surface_bold` is covered by
``arealmshbm/bold_io/tests/test_bold_io.py``. This file pins the
*front-line* guard at
:func:`arealmshbm.pipeline.inputs.read_bold_inputs` — the one that
fires BEFORE any reader runs, when ``Pipeline(project_dir).run()``
first parses the manifest. Surfacing the GIFTI-only contract at parse
time is what catches stale projects (e.g. pre-PR-#54 ``.nii.gz``
paths) before any expensive setup happens.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arealmshbm.pipeline.inputs import read_bold_inputs


def _make_manifest(lh: str, rh: str) -> dict:
    """Build the minimal valid bold_inputs.json shape (one subject,
    one session) with the given (lh, rh) path strings."""
    return {
        "schema_version": "1",
        "dataset_name": "synth",
        "targ_mesh": "fsaverage6",
        "seed_mesh": "fsaverage3",
        "subjects": [
            {
                "id": "1",
                "sessions": [
                    {"id": "1", "lh": lh, "rh": rh},
                ],
            },
        ],
    }


def test_rejects_nii_gz_path_at_parse_time(tmp_path: Path):
    """A bold_inputs.json carrying ``.nii.gz`` paths is refused at
    parse time with an error that names the offending field and the
    GIFTI-only contract. This guards against stale pre-strip projects
    silently running and then failing deep inside the BOLD reader."""
    p = tmp_path / "bold_inputs.json"
    p.write_text(json.dumps(_make_manifest(
        lh="/data/sub-001_ses-01_hemi-L_bold.nii.gz",
        rh="/data/sub-001_ses-01_hemi-R_bold.nii.gz",
    )), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.func\.gii"):
        read_bold_inputs(p)


def test_rejects_nii_gz_on_rh_only(tmp_path: Path):
    """The check fires for either hemi, not just lh. ``rh`` is a
    separate iteration of the validation loop — pin it so a future
    refactor that only checks lh doesn't silently regress."""
    p = tmp_path / "bold_inputs.json"
    p.write_text(json.dumps(_make_manifest(
        lh="/data/sub-001_ses-01_hemi-L_bold.func.gii",
        rh="/data/sub-001_ses-01_hemi-R_bold.nii.gz",
    )), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.rh="):
        read_bold_inputs(p)


def test_accepts_func_gii_paths(tmp_path: Path):
    """Sanity backstop: the .gii enforcement doesn't accidentally
    reject the canonical ``.func.gii`` shape (compound suffix). The
    suffix check uses ``Path.suffix`` which returns just ``.gii`` for
    ``...bold.func.gii`` — verify that's the right behavior."""
    p = tmp_path / "bold_inputs.json"
    p.write_text(json.dumps(_make_manifest(
        lh="/data/sub-001_ses-01_hemi-L_bold.func.gii",
        rh="/data/sub-001_ses-01_hemi-R_bold.func.gii",
    )), encoding="utf-8")
    bi = read_bold_inputs(p)
    assert len(bi.subjects) == 1
    assert bi.subjects[0].sessions[0].lh.suffix == ".gii"


def test_rejects_unknown_suffix(tmp_path: Path):
    """A non-``.gii`` non-``.nii.gz`` suffix (e.g. ``.dat``) is also
    rejected — the guard is whitelist-style, not nii.gz-blacklist."""
    p = tmp_path / "bold_inputs.json"
    p.write_text(json.dumps(_make_manifest(
        lh="/data/sub-001_ses-01_hemi-L_bold.dat",
        rh="/data/sub-001_ses-01_hemi-R_bold.dat",
    )), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.func\.gii"):
        read_bold_inputs(p)
