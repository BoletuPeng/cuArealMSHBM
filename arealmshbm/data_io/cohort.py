"""cohort.py — single source of truth for cohort roster + artifact paths.

A ``project_dir`` hosting a step1 output carries a single
``cohort.json`` at its root. The file lists the cohort roster
(subjects, sessions, mesh config) and points at every artifact step1
wrote. Step2 and step3 use it as the sole discovery mechanism.

Concurrency: the writer is single-writer-safe (atomic os.replace),
but the manifest itself has no file lock. Two concurrent driver runs
against the same ``project_dir`` produce a lost-update — the second
``os.replace`` wins outright. Treat ``project_dir`` as owned by one
running pipeline at a time.

Schema v1
---------

::

    {
      "schema_version": 1,
      "created_by": "arealmshbm.step1_pipeline",
      "created_at": "<ISO 8601 UTC>",
      "run_id": "<sha256 of (sorted subjects, sorted sessions, mesh, n_grad)>",
      "mesh": {"targ": "fsaverage6", "seed": "fsaverage3"},
      "n_grad_components": 100,
      "subjects": [
        {
          "id": "1",
          "sessions": ["1", "2", ...],
          "profile_b2nd":  "<path>" | null,
          "gradient_lh":   "<path>" | null,
          "gradient_rh":   "<path>" | null
        },
        ...
      ],
      "group_mat":          "<path>" | null,
      "spatial_mask_mat":   "<path>" | null,
      "avg_profile_lh_npy": "<path>" | null,
      "avg_profile_rh_npy": "<path>" | null
    }

Path conventions
----------------

Every ``<path>`` is interpreted relative to ``project_dir`` when it's a
relative path, and as-is when it's an absolute path (POSIX or Windows).
Absolute paths are used by the step-2 GT builder, which points
gradient fields at external mode-A per-subject directories instead of
staging files into the cohort tree.

Merge semantics
---------------

``write_cohort_partial`` reads an existing ``cohort.json`` (if any),
merges per-field with the new partial manifest, and writes the result.
Fields set to ``None`` in the partial do NOT overwrite existing
non-None values — this lets users run step1 incrementally (e.g.
generate_profiles in one invocation, avg_profiles in another) without
losing the artifact-path entries from prior runs.

Subjects list is replaced on every write (the cohort roster is a
property of the current step1 invocation), but per-subject artifact
fields go through the same partial-merge rule keyed on subject id.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence


COHORT_JSON_NAME = "cohort.json"
SCHEMA_VERSION = 1
DEFAULT_CREATED_BY = "arealmshbm.step1_pipeline"


# ─────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────
@dataclass
class CohortSubject:
    """One subject's roster entry + artifact paths.

    ``id`` is a string (the step1 runners coerce subject IDs to str).
    ``sessions`` is the per-subject session list (usually identical
    across subjects, but the schema allows ragged cohorts for forward
    compat). Artifact path fields are ``None`` when the corresponding
    artifact wasn't produced (e.g. ``gradient_lh=None`` on a dMSHBM
    cohort that doesn't need gradients).
    """
    id: str
    sessions: List[str]
    profile_b2nd: Optional[str] = None
    gradient_lh: Optional[str] = None
    gradient_rh: Optional[str] = None


@dataclass
class CohortManifest:
    """In-memory mirror of ``cohort.json``.

    ``mesh`` is a 2-key dict ``{"targ": ..., "seed": ...}``. ``run_id``
    is a sha256-derived string identifying the cohort definition
    (subjects + sessions + mesh + n_grad) — does NOT include artifact
    paths, so the same logical cohort regenerated on a different disk
    yields the same run_id.
    """
    subjects: List[CohortSubject]
    mesh: Dict[str, str]
    n_grad_components: int
    run_id: str
    schema_version: int = SCHEMA_VERSION
    created_by: str = DEFAULT_CREATED_BY
    created_at: str = field(default_factory=lambda: _utc_now_iso())
    group_mat: Optional[str] = None
    spatial_mask_mat: Optional[str] = None
    avg_profile_lh_npy: Optional[str] = None
    avg_profile_rh_npy: Optional[str] = None

    # ── convenience ──
    @property
    def num_sub(self) -> int:
        return len(self.subjects)

    @property
    def num_session(self) -> int:
        """Session count — required to be uniform across subjects.

        Raises if the cohort has ragged session lists (forward-compat
        possibility that today's callers don't support)."""
        ts = {len(s.sessions) for s in self.subjects}
        if len(ts) > 1:
            raise ValueError(
                f"CohortManifest.num_session: ragged sessions across "
                f"subjects ({sorted(ts)}); current consumers require "
                f"uniform T."
            )
        return ts.pop() if ts else 0

    def subject_by_id(self, sub_id: str | int) -> CohortSubject:
        """Look up one subject by id. Raises ``KeyError`` if absent."""
        key = str(sub_id)
        for s in self.subjects:
            if s.id == key:
                return s
        raise KeyError(
            f"CohortManifest: subject id={key!r} not in cohort "
            f"({[s.id for s in self.subjects]})"
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _utc_now_iso() -> str:
    """RFC 3339 UTC timestamp, second precision."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def cohort_json_path(project_dir: str | Path) -> Path:
    """Canonical ``cohort.json`` location."""
    return Path(project_dir) / COHORT_JSON_NAME


def compute_run_id(
    subjects: Sequence[str | int],
    sessions: Sequence[str | int],
    targ_mesh: str,
    seed_mesh: str,
    n_grad_components: int,
) -> str:
    """SHA-256 of the canonical cohort definition.

    Stable across machines / filesystem reorderings: subjects and
    sessions are sorted before hashing (cohort identity is set-shaped,
    not order-sensitive). The hash does NOT include artifact paths or
    timestamps — re-running step1 on the same logical cohort yields
    the same run_id.

    ``n_grad_components`` IS part of the hash by design: changing it
    means step2/step3 need a different gradient matrix per subject, so
    a re-run with a new component count must invalidate the prior
    cohort's merged artifacts. Treat it as part of cohort identity,
    not metadata.
    """
    payload = {
        "subjects": sorted(str(s) for s in subjects),
        "sessions": sorted(str(s) for s in sessions),
        "targ": targ_mesh,
        "seed": seed_mesh,
        "n_grad_components": int(n_grad_components),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_path(project_dir: str | Path, path_str: str) -> Path:
    """Resolve a cohort.json path field against ``project_dir``.

    Absolute paths (POSIX ``/foo`` or Windows ``C:\\foo``) are returned
    as-is. Relative paths are joined under ``project_dir``. Relative
    paths in the file use POSIX separators by convention but
    ``pathlib`` handles ``/`` on Windows transparently.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Force POSIX-style interpretation of the relative segment for
    # cross-platform stability; ``Path / PurePosixPath`` on Windows
    # silently coerces separators.
    return Path(project_dir) / PurePosixPath(path_str)


def relpath_or_abs(project_dir: str | Path, path: str | Path) -> str:
    """Encode ``path`` for cohort.json storage.

    Paths under ``project_dir`` are stored relative (POSIX separators);
    paths outside are stored as absolute. The GT builder relies on
    this — its gradient paths point at external mode-A directories and
    must round-trip as absolute.
    """
    p = Path(path)
    base = Path(project_dir).resolve()
    try:
        if p.resolve().is_relative_to(base):
            return p.resolve().relative_to(base).as_posix()
    except OSError:
        # Resolve can fail on non-existent paths on some platforms;
        # fall through to absolute-string fallback.
        pass
    return str(p)


# ─────────────────────────────────────────────────────────────────────
# Read / write
# ─────────────────────────────────────────────────────────────────────
def read_cohort(project_dir: str | Path) -> CohortManifest:
    """Read and validate the ``cohort.json`` at ``project_dir``.

    Raises ``FileNotFoundError`` if the file is missing, ``ValueError``
    on schema-version mismatch or malformed payload.
    """
    p = cohort_json_path(project_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"cohort.json not found at {p}. Run step1 to produce it, "
            f"or build one manually for legacy / GT cohorts."
        )
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _from_dict(data, source=str(p))


def write_cohort(project_dir: str | Path, manifest: CohortManifest) -> Path:
    """Write ``manifest`` to ``project_dir/cohort.json`` (overwriting).

    Atomic: writes to a sibling tempfile, fsyncs, then ``os.replace``
    onto the canonical name. A crash or SIGINT mid-write leaves the
    previous cohort.json intact rather than a truncated/invalid JSON
    that step2/step3 would silently misread.

    Use :func:`write_cohort_partial` if you want field-level merge
    against an existing cohort.json. This function is the "I know the
    full manifest" entry point.
    """
    p = cohort_json_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_dict(manifest)
    # Write to a sibling tempfile in the same directory (so os.replace
    # stays atomic — cross-filesystem rename is not). delete=False keeps
    # the file around after the with-block so we can fsync + replace it.
    fd, tmp_name = tempfile.mkstemp(
        prefix=COHORT_JSON_NAME + ".",
        suffix=".tmp",
        dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # ``ensure_ascii=False`` to keep paths with non-ASCII chars
            # human-readable; ``indent=2`` for diff-friendliness.
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        # Don't leave a half-written tempfile behind on any failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


def write_cohort_partial(
    project_dir: str | Path,
    *,
    subjects: Sequence[CohortSubject],
    mesh: Dict[str, str],
    n_grad_components: int,
    run_id: str,
    group_mat: Optional[str] = None,
    spatial_mask_mat: Optional[str] = None,
    avg_profile_lh_npy: Optional[str] = None,
    avg_profile_rh_npy: Optional[str] = None,
) -> CohortManifest:
    """Merge a partial cohort manifest into the existing one (if any).

    Merge rule:
      * Roster fields (``subjects``, ``mesh``, ``n_grad_components``,
        ``run_id``) are always REPLACED with the new values. If the
        existing roster doesn't match, the previous artifacts referred
        to a different cohort and don't carry over.
      * Top-level artifact fields (``group_mat``, ``spatial_mask_mat``,
        ``avg_profile_*_npy``): kept from the existing manifest when
        the partial passes ``None``; overwritten when the partial
        passes a non-None value.
      * Per-subject artifact fields (``profile_b2nd``, ``gradient_*``):
        same partial-merge rule, keyed on subject id. If the existing
        manifest has a subject not in the new roster, that subject's
        entry is dropped entirely (it's no longer in the cohort).

    The roster-mismatch case (existing cohort differs from new) drops
    all carry-over artifacts so we don't end up with stale paths from
    an unrelated cohort.
    """
    existing: Optional[CohortManifest] = None
    p = cohort_json_path(project_dir)
    if p.exists():
        try:
            existing = read_cohort(project_dir)
        except FileNotFoundError:
            # Race: file vanished between exists() and read_cohort(). Treat
            # as a fresh project — nothing to merge against.
            existing = None
        except ValueError as e:
            # Corrupt or schema-mismatched cohort.json. This is a
            # data-loss event — the previous artifact-path ledger is
            # about to be discarded. Surface it on stderr so the user
            # at least sees it; the rest of the run still proceeds
            # because step1 always writes a fresh roster.
            print(
                f"WARNING: cohort.json at {p} could not be parsed "
                f"({e}); discarding prior artifact paths.",
                file=sys.stderr,
            )
            existing = None

    roster_match = (
        existing is not None
        and existing.run_id == run_id
    )

    # Per-subject merge — only if roster matches.
    merged_subjects: List[CohortSubject] = []
    if roster_match:
        existing_by_id = {s.id: s for s in existing.subjects}  # type: ignore[union-attr]
        for new_s in subjects:
            old_s = existing_by_id.get(new_s.id)
            if old_s is None:
                merged_subjects.append(_copy_subject(new_s))
            else:
                merged_subjects.append(CohortSubject(
                    id=new_s.id,
                    sessions=list(new_s.sessions),
                    profile_b2nd=(new_s.profile_b2nd
                                   if new_s.profile_b2nd is not None
                                   else old_s.profile_b2nd),
                    gradient_lh=(new_s.gradient_lh
                                  if new_s.gradient_lh is not None
                                  else old_s.gradient_lh),
                    gradient_rh=(new_s.gradient_rh
                                  if new_s.gradient_rh is not None
                                  else old_s.gradient_rh),
                ))
    else:
        merged_subjects = [_copy_subject(s) for s in subjects]

    # Top-level merge — only carry over when roster matches.
    def _carry(new_val: Optional[str], old_val: Optional[str]) -> Optional[str]:
        if new_val is not None:
            return new_val
        return old_val if roster_match else None

    merged = CohortManifest(
        subjects=merged_subjects,
        mesh=dict(mesh),
        n_grad_components=int(n_grad_components),
        run_id=run_id,
        group_mat=_carry(group_mat,
                          existing.group_mat if existing else None),
        spatial_mask_mat=_carry(spatial_mask_mat,
                                 existing.spatial_mask_mat if existing else None),
        avg_profile_lh_npy=_carry(avg_profile_lh_npy,
                                    existing.avg_profile_lh_npy if existing else None),
        avg_profile_rh_npy=_carry(avg_profile_rh_npy,
                                    existing.avg_profile_rh_npy if existing else None),
    )
    write_cohort(project_dir, merged)
    return merged


# ─────────────────────────────────────────────────────────────────────
# Internal: serialisation
# ─────────────────────────────────────────────────────────────────────
def _to_dict(m: CohortManifest) -> Dict[str, Any]:
    return {
        "schema_version": m.schema_version,
        "created_by": m.created_by,
        "created_at": m.created_at,
        "run_id": m.run_id,
        "mesh": dict(m.mesh),
        "n_grad_components": int(m.n_grad_components),
        "subjects": [asdict(s) for s in m.subjects],
        "group_mat": m.group_mat,
        "spatial_mask_mat": m.spatial_mask_mat,
        "avg_profile_lh_npy": m.avg_profile_lh_npy,
        "avg_profile_rh_npy": m.avg_profile_rh_npy,
    }


def _from_dict(data: Dict[str, Any], *, source: str) -> CohortManifest:
    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"cohort.json at {source} has schema_version={sv}; "
            f"this reader only supports v{SCHEMA_VERSION}"
        )
    try:
        mesh = data["mesh"]
        if not isinstance(mesh, dict) or "targ" not in mesh or "seed" not in mesh:
            raise ValueError("missing 'mesh.targ' or 'mesh.seed'")
        subjects_raw = data["subjects"]
        if not isinstance(subjects_raw, list):
            raise ValueError("'subjects' must be a list")
        subjects: List[CohortSubject] = []
        for i, sd in enumerate(subjects_raw):
            if not isinstance(sd, dict):
                raise ValueError(f"subjects[{i}] is not an object")
            subjects.append(CohortSubject(
                id=str(sd["id"]),
                sessions=[str(x) for x in sd["sessions"]],
                profile_b2nd=sd.get("profile_b2nd"),
                gradient_lh=sd.get("gradient_lh"),
                gradient_rh=sd.get("gradient_rh"),
            ))
        return CohortManifest(
            schema_version=SCHEMA_VERSION,
            created_by=str(data.get("created_by", DEFAULT_CREATED_BY)),
            created_at=str(data.get("created_at", _utc_now_iso())),
            run_id=str(data["run_id"]),
            mesh={"targ": str(mesh["targ"]), "seed": str(mesh["seed"])},
            n_grad_components=int(data["n_grad_components"]),
            subjects=subjects,
            group_mat=data.get("group_mat"),
            spatial_mask_mat=data.get("spatial_mask_mat"),
            avg_profile_lh_npy=data.get("avg_profile_lh_npy"),
            avg_profile_rh_npy=data.get("avg_profile_rh_npy"),
        )
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(
            f"cohort.json at {source} is malformed: {e}"
        ) from e


def _copy_subject(s: CohortSubject) -> CohortSubject:
    return CohortSubject(
        id=s.id,
        sessions=list(s.sessions),
        profile_b2nd=s.profile_b2nd,
        gradient_lh=s.gradient_lh,
        gradient_rh=s.gradient_rh,
    )


# ─────────────────────────────────────────────────────────────────────
# Backfill helper — for migrating pre-cohort.json project_dirs
# ─────────────────────────────────────────────────────────────────────
def backfill_cohort_json(
    project_dir: str | Path,
    *,
    subjects: Sequence[str | int],
    sessions: Sequence[str | int],
    targ_mesh: str,
    seed_mesh: str,
    n_grad_components: int = 100,
) -> CohortManifest:
    """One-shot migration: build ``cohort.json`` from an existing
    project_dir laid out per the pre-cohort.json convention.

    Probes canonical paths to populate artifact fields:

    * ``profile_b2nd`` ← ``profiles_raw/sub<S>/sub<S>_<targ>_roi<seed>.profile.b2nd``
    * ``gradient_lh`` / ``gradient_rh`` ← ``gradients/sub<S>/{lh,rh}_emb_<N>_distance_matrix.{npy,mat}``
      (prefer .npy; fall back to .mat if only .mat is present)
    * ``group_mat`` ← ``group/group.mat``
    * ``spatial_mask_mat`` ← ``spatial_mask/spatial_mask_<targ>.mat``
    * ``avg_profile_*_npy`` ← ``profiles/avg_profile/{lh,rh}_<targ>_roi<seed>_avg_profile.npy``

    Fields whose probe target doesn't exist are left as ``None``.
    Writes the resulting manifest and returns it.

    Intended for one-time migration of a cohort produced by an older
    step1 build (or by external means). For a fresh step1 run, the
    pipeline writes cohort.json itself — no backfill needed.
    """
    pd = Path(project_dir)
    sub_ids = [str(s) for s in subjects]
    sess_ids = [str(s) for s in sessions]
    nc = int(n_grad_components)

    def _opt(p: Path) -> Optional[str]:
        return relpath_or_abs(pd, p) if p.exists() else None

    cohort_subjects: List[CohortSubject] = []
    for sid in sub_ids:
        b2nd = (pd / "profiles_raw" / f"sub{sid}" /
                f"sub{sid}_{targ_mesh}_roi{seed_mesh}.profile.b2nd")
        sub_dir = pd / "gradients" / f"sub{sid}"
        lh_npy = sub_dir / f"lh_emb_{nc}_distance_matrix.npy"
        lh_mat = sub_dir / f"lh_emb_{nc}_distance_matrix.mat"
        rh_npy = sub_dir / f"rh_emb_{nc}_distance_matrix.npy"
        rh_mat = sub_dir / f"rh_emb_{nc}_distance_matrix.mat"
        lh_grad = lh_npy if lh_npy.exists() else (lh_mat if lh_mat.exists() else None)
        rh_grad = rh_npy if rh_npy.exists() else (rh_mat if rh_mat.exists() else None)
        cohort_subjects.append(CohortSubject(
            id=sid,
            sessions=sess_ids,
            profile_b2nd=_opt(b2nd),
            gradient_lh=relpath_or_abs(pd, lh_grad) if lh_grad else None,
            gradient_rh=relpath_or_abs(pd, rh_grad) if rh_grad else None,
        ))

    group_mat = _opt(pd / "group" / "group.mat")
    spatial_mask_mat = _opt(pd / "spatial_mask" / f"spatial_mask_{targ_mesh}.mat")
    avg_lh = _opt(pd / "profiles" / "avg_profile" /
                   f"lh_{targ_mesh}_roi{seed_mesh}_avg_profile.npy")
    avg_rh = _opt(pd / "profiles" / "avg_profile" /
                   f"rh_{targ_mesh}_roi{seed_mesh}_avg_profile.npy")

    run_id = compute_run_id(sub_ids, sess_ids, targ_mesh, seed_mesh, nc)
    return write_cohort_partial(
        pd,
        subjects=cohort_subjects,
        mesh={"targ": targ_mesh, "seed": seed_mesh},
        n_grad_components=nc,
        run_id=run_id,
        group_mat=group_mat,
        spatial_mask_mat=spatial_mask_mat,
        avg_profile_lh_npy=avg_lh,
        avg_profile_rh_npy=avg_rh,
    )
