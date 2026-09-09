"""inputs.py — reader + validator for ``bold_inputs.json``.

``bold_inputs.json`` is the user-provided contract: a list of subjects,
each with a list of sessions, each with explicit ``lh``/``rh`` BOLD
``.func.gii`` paths. The pipeline driver consumes this list and emits
a :class:`BoldInputs` object — read once, validated, then handed off.

Schema v1::

    {
      "schema_version": "1",
      "dataset_name": "<str>",
      "targ_mesh":    "fsaverage6",      # surface BOLD lives on
      "seed_mesh":    "fsaverage3",      # step1 RSFC seed mesh
      "subjects": [
        {
          "id": "1",                     # 1-based positional integer-string
          "sessions": [
            {"id": "1", "lh": "...func.gii", "rh": "...func.gii"},
            ...
          ]
        },
        ...
      ]
    }

There is intentionally no ``label`` / display-name field. The ``id``
the user writes is the only identifier the pipeline uses — anywhere on
disk under the project, in cohort.json, in step3 output filenames. If
the user wants to remember "id=7 means sub-009", that mapping lives
outside this manifest.

Validation rules:

  * ``schema_version`` must equal ``"1"``.
  * ``subjects[*].id`` must be a positive integer-string, **sequential**
    starting at 1 — matches step1's 1-based positional indexing into
    cohort.json.
  * Per-subject session counts must match (uniform N across subjects).
  * Within a subject, ``sessions[*].id`` must be a positive integer-string
    sequential from 1.
  * Every ``lh`` / ``rh`` path must exist on disk at read time.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class BoldInputSession:
    id: str
    lh: Path
    rh: Path


@dataclass(frozen=True)
class BoldInputSubject:
    id: str           # "1", "2", ... — positional integer-string
    sessions: Tuple[BoldInputSession, ...]


@dataclass(frozen=True)
class BoldInputs:
    schema_version: str
    dataset_name: str
    targ_mesh: str
    seed_mesh: str
    subjects: Tuple[BoldInputSubject, ...]

    @property
    def num_subjects(self) -> int:
        return len(self.subjects)

    @property
    def num_sessions(self) -> int:
        # All subjects have the same N (enforced at parse time).
        return len(self.subjects[0].sessions)

    def subject_ids(self) -> List[str]:
        return [s.id for s in self.subjects]

    def session_ids(self) -> List[str]:
        return [s.id for s in self.subjects[0].sessions]


def read_bold_inputs(path: Path | str) -> BoldInputs:
    """Parse + validate ``bold_inputs.json``. Raises on any defect."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bold_inputs.json not found at {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    sv = raw.get("schema_version")
    if sv != "1":
        raise ValueError(
            f"bold_inputs.json: schema_version must be '1' (got {sv!r})"
        )
    for k in ("dataset_name", "targ_mesh", "seed_mesh", "subjects"):
        if k not in raw:
            raise ValueError(f"bold_inputs.json: missing required key {k!r}")

    raw_subjects = raw["subjects"]
    if not isinstance(raw_subjects, list) or len(raw_subjects) == 0:
        raise ValueError("bold_inputs.json: 'subjects' must be a non-empty list")

    n_sess_expected: int | None = None
    parsed_subjects: List[BoldInputSubject] = []
    for sub_i, rs in enumerate(raw_subjects):
        expected_id = str(sub_i + 1)
        sub_id = str(rs.get("id", "")).strip()
        if sub_id != expected_id:
            raise ValueError(
                f"bold_inputs.json: subjects[{sub_i}].id must be "
                f"{expected_id!r} (sequential 1-based), got {sub_id!r}"
            )
        sessions_raw = rs.get("sessions", [])
        if not isinstance(sessions_raw, list) or len(sessions_raw) == 0:
            raise ValueError(
                f"bold_inputs.json: subjects[{sub_i}].sessions must be a "
                f"non-empty list"
            )
        if n_sess_expected is None:
            n_sess_expected = len(sessions_raw)
        elif len(sessions_raw) != n_sess_expected:
            raise ValueError(
                f"bold_inputs.json: subjects[{sub_i}] has "
                f"{len(sessions_raw)} sessions; expected "
                f"{n_sess_expected} (matching subjects[0])"
            )

        parsed_sess: List[BoldInputSession] = []
        for sess_i, rsess in enumerate(sessions_raw):
            expected_sess_id = str(sess_i + 1)
            sess_id = str(rsess.get("id", "")).strip()
            if sess_id != expected_sess_id:
                raise ValueError(
                    f"bold_inputs.json: subjects[{sub_i}].sessions[{sess_i}].id "
                    f"must be {expected_sess_id!r}, got {sess_id!r}"
                )
            lh = Path(rsess.get("lh", ""))
            rh = Path(rsess.get("rh", ""))
            if not str(lh) or not str(rh):
                raise ValueError(
                    f"bold_inputs.json: subjects[{sub_i}].sessions[{sess_i}] "
                    f"missing lh/rh path"
                )
            # GIFTI-only since the NIFTI surface-BOLD reader was retired.
            # The bit-equality argument that justified the strip is
            # captured in the PR #54 description (10/10 numerical
            # artifacts match between the direct-GIFTI route and the
            # converted-NIFTI mirror on real YS sub-001 data under
            # backend_step0=cpu); the e2e harness was retired with the
            # NIFTI reader.
            for hemi, p in (("lh", lh), ("rh", rh)):
                # Whitelist: any ``.gii`` is accepted. GIFTI BOLD is
                # canonically ``.func.gii`` (BIDS naming convention) but
                # the actual file extension is just ``.gii`` —
                # ``.surf.gii`` (geometry) / ``.shape.gii`` (single-frame)
                # also pass the suffix check; they fail the content
                # contract instead at :func:`_scan_gifti_spans`
                # because their DataArray shape isn't a per-vertex time
                # series.
                if p.suffix.lower() != ".gii":
                    raise ValueError(
                        f"bold_inputs.json: subjects[{sub_i}]."
                        f"sessions[{sess_i}].{hemi}={str(p)!r} — only "
                        f"``.gii`` surface BOLD is supported (typically "
                        f"``.func.gii``). The NIFTI mirror path was "
                        f"removed; re-derive your BOLD from "
                        f"DeepPrep/fmriprep GIFTI output."
                    )
            parsed_sess.append(BoldInputSession(id=sess_id, lh=lh, rh=rh))
        parsed_subjects.append(BoldInputSubject(
            id=sub_id, sessions=tuple(parsed_sess),
        ))

    return BoldInputs(
        schema_version=sv,
        dataset_name=str(raw["dataset_name"]),
        targ_mesh=str(raw["targ_mesh"]),
        seed_mesh=str(raw["seed_mesh"]),
        subjects=tuple(parsed_subjects),
    )


def check_bold_files_exist(inputs: BoldInputs) -> None:
    """Confirm every BOLD path on disk. Hard error on any missing file."""
    missing: List[str] = []
    for sub in inputs.subjects:
        for sess in sub.sessions:
            if not sess.lh.exists():
                missing.append(f"sub{sub.id} sess{sess.id} lh: {sess.lh}")
            if not sess.rh.exists():
                missing.append(f"sub{sub.id} sess{sess.id} rh: {sess.rh}")
    if missing:
        raise FileNotFoundError(
            f"bold_inputs.json references {len(missing)} missing BOLD file(s):\n  "
            + "\n  ".join(missing[:10])
            + (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
        )
