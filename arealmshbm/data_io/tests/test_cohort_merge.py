# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Unit tests for the field-level merge semantics in
:func:`arealmshbm.data_io.cohort.write_cohort_partial`.

The merge rules are non-trivial and silent failures here would corrupt
the manifest that step2 / step3 read as their sole discovery mechanism.
We pin the three behaviours that matter:

  1. roster-match (same run_id) → top-level + per-subject artifacts
     carry over when the partial passes ``None``.
  2. roster-mismatch (different run_id) → all carry-over artifacts are
     dropped, because they referred to a different cohort.
  3. corrupt cohort.json → start fresh (no crash, warning on stderr).

Plus a smoke test for atomic write durability (the temp + os.replace
pattern doesn't leave the destination half-written on a writer crash).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arealmshbm.data_io.cohort import (
    CohortSubject,
    cohort_json_path,
    compute_run_id,
    read_cohort,
    write_cohort_partial,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _mkcohort(tmp_path: Path, *, subjects, run_id_kwargs=None, **artifacts):
    """Write a cohort.json into ``tmp_path`` and return the manifest."""
    rk = run_id_kwargs or {}
    rid = compute_run_id(
        subjects=rk.get("subjects", [s.id for s in subjects]),
        sessions=rk.get("sessions", subjects[0].sessions if subjects else []),
        targ_mesh=rk.get("targ_mesh", "fsaverage6"),
        seed_mesh=rk.get("seed_mesh", "fsaverage3"),
        n_grad_components=rk.get("n_grad_components", 100),
    )
    return write_cohort_partial(
        tmp_path,
        subjects=subjects,
        mesh={"targ": "fsaverage6", "seed": "fsaverage3"},
        n_grad_components=100,
        run_id=rid,
        **artifacts,
    )


def _sub(id_, **kw):
    return CohortSubject(
        id=str(id_),
        sessions=kw.get("sessions", ["1", "2"]),
        profile_b2nd=kw.get("profile_b2nd"),
        gradient_lh=kw.get("gradient_lh"),
        gradient_rh=kw.get("gradient_rh"),
    )


# ─────────────────────────────────────────────────────────────────────
# Roster-match: carry-over preserves artifacts on second call
# ─────────────────────────────────────────────────────────────────────
def test_roster_match_carries_top_level_artifacts(tmp_path):
    """Same roster, second call passes None → existing top-level
    artifact paths are preserved."""
    subs = [_sub(1), _sub(2)]
    _mkcohort(
        tmp_path,
        subjects=subs,
        group_mat="group/group.mat",
        spatial_mask_mat="spatial_mask/spatial_mask_fsaverage6.mat",
        avg_profile_lh_npy="profiles/avg_profile/lh_fsaverage6_roifsaverage3_avg_profile.npy",
        avg_profile_rh_npy="profiles/avg_profile/rh_fsaverage6_roifsaverage3_avg_profile.npy",
    )
    # Second call with same roster but no artifacts in the partial.
    _mkcohort(tmp_path, subjects=subs)
    m = read_cohort(tmp_path)
    assert m.group_mat == "group/group.mat"
    assert m.spatial_mask_mat == "spatial_mask/spatial_mask_fsaverage6.mat"
    assert m.avg_profile_lh_npy is not None
    assert m.avg_profile_rh_npy is not None


def test_roster_match_carries_per_subject_artifacts(tmp_path):
    """Same roster, second call passes None on a subject's fields →
    that subject's prior artifact paths are preserved."""
    subs_v1 = [
        _sub(1, profile_b2nd="profiles_raw/sub1/sub1.profile.b2nd",
             gradient_lh="gradients/sub1/lh_emb_100_distance_matrix.npy"),
        _sub(2, profile_b2nd="profiles_raw/sub2/sub2.profile.b2nd"),
    ]
    _mkcohort(tmp_path, subjects=subs_v1)
    # Second call: same IDs, all artifact fields None.
    subs_v2 = [_sub(1), _sub(2)]
    _mkcohort(tmp_path, subjects=subs_v2)
    m = read_cohort(tmp_path)
    s1 = m.subject_by_id("1")
    s2 = m.subject_by_id("2")
    assert s1.profile_b2nd == "profiles_raw/sub1/sub1.profile.b2nd"
    assert s1.gradient_lh == "gradients/sub1/lh_emb_100_distance_matrix.npy"
    assert s2.profile_b2nd == "profiles_raw/sub2/sub2.profile.b2nd"


def test_roster_match_new_partial_overrides_old(tmp_path):
    """Non-None values in the partial WIN over carry-over."""
    subs_v1 = [_sub(1, profile_b2nd="old.b2nd")]
    _mkcohort(tmp_path, subjects=subs_v1, group_mat="group/old.mat")
    subs_v2 = [_sub(1, profile_b2nd="new.b2nd")]
    _mkcohort(tmp_path, subjects=subs_v2, group_mat="group/new.mat")
    m = read_cohort(tmp_path)
    assert m.group_mat == "group/new.mat"
    assert m.subject_by_id("1").profile_b2nd == "new.b2nd"


# ─────────────────────────────────────────────────────────────────────
# Roster-mismatch: drop all carry-over
# ─────────────────────────────────────────────────────────────────────
def test_roster_mismatch_drops_top_level_carry_over(tmp_path):
    """Different run_id → top-level artifacts are NOT carried over."""
    # First cohort: 2 subjects, with artifacts.
    _mkcohort(
        tmp_path,
        subjects=[_sub(1), _sub(2)],
        group_mat="group/group.mat",
        avg_profile_lh_npy="lh.npy",
    )
    # Second cohort: completely different roster (3 subjects).
    _mkcohort(tmp_path, subjects=[_sub(1), _sub(2), _sub(3)])
    m = read_cohort(tmp_path)
    assert m.group_mat is None
    assert m.avg_profile_lh_npy is None


def test_roster_mismatch_drops_per_subject_carry_over(tmp_path):
    """Different run_id → per-subject artifacts dropped, even for
    subjects whose id appears in both rosters."""
    _mkcohort(
        tmp_path,
        subjects=[_sub(1, profile_b2nd="sub1_v1.b2nd"), _sub(2)],
    )
    # Add a third subject → run_id changes → carry-over is forbidden.
    _mkcohort(
        tmp_path,
        subjects=[_sub(1), _sub(2), _sub(3)],
    )
    m = read_cohort(tmp_path)
    assert m.subject_by_id("1").profile_b2nd is None


def test_n_grad_components_change_is_a_roster_mismatch(tmp_path):
    """``n_grad_components`` participates in the run_id, so changing
    it from 100 → 200 with the same subjects/sessions/mesh should
    drop the prior cohort's carry-over artifacts."""
    subs = [_sub(1)]
    _mkcohort(
        tmp_path,
        subjects=subs,
        run_id_kwargs={"n_grad_components": 100},
        group_mat="group/group.mat",
    )
    # Re-run with a different n_grad_components — new gradients =
    # different cohort identity.
    _mkcohort(
        tmp_path,
        subjects=subs,
        run_id_kwargs={"n_grad_components": 200},
    )
    m = read_cohort(tmp_path)
    assert m.group_mat is None
    assert m.n_grad_components == 100  # Roster mismatch keeps the LAST partial's value
    # ^ note: write_cohort_partial always replaces n_grad_components with
    # what was passed; the assert above pins that contract.


# ─────────────────────────────────────────────────────────────────────
# Corrupt cohort.json
# ─────────────────────────────────────────────────────────────────────
def test_corrupt_cohort_json_starts_fresh(tmp_path, capsys):
    """A malformed existing cohort.json must NOT crash the writer; the
    new partial overwrites it and the corruption event surfaces on
    stderr."""
    p = cohort_json_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json", encoding="utf-8")

    _mkcohort(tmp_path, subjects=[_sub(1)], group_mat="group/group.mat")
    captured = capsys.readouterr()
    assert "cohort.json" in captured.err
    assert "could not be parsed" in captured.err

    m = read_cohort(tmp_path)
    assert m.group_mat == "group/group.mat"


def test_missing_cohort_json_starts_silently(tmp_path, capsys):
    """No existing cohort.json → no warning (this is the fresh-project
    path, not a data-loss event)."""
    _mkcohort(tmp_path, subjects=[_sub(1)])
    captured = capsys.readouterr()
    assert captured.err == ""


# ─────────────────────────────────────────────────────────────────────
# Atomic write — destination intact on writer error
# ─────────────────────────────────────────────────────────────────────
def test_schema_version_mismatch_rejected_on_read(tmp_path):
    """A cohort.json with an unknown schema_version must surface as a
    ValueError from read_cohort — the loud path. Silent acceptance
    would let a future v2 manifest corrupt a v1 reader's downstream
    assumptions."""
    p = cohort_json_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema_version": 99,
        "created_by": "test",
        "created_at": "1970-01-01T00:00:00+00:00",
        "run_id": "deadbeef",
        "mesh": {"targ": "fsaverage6", "seed": "fsaverage3"},
        "n_grad_components": 100,
        "subjects": [],
        "group_mat": None, "spatial_mask_mat": None,
        "avg_profile_lh_npy": None, "avg_profile_rh_npy": None,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_cohort(tmp_path)


def test_atomic_write_preserves_old_file_on_writer_failure(tmp_path,
                                                            monkeypatch):
    """If json.dump raises mid-write, the existing cohort.json must
    remain readable (the tempfile pattern guarantees the canonical
    name only swaps after a complete write)."""
    # Establish a valid first cohort.
    _mkcohort(tmp_path, subjects=[_sub(1)], group_mat="group/v1.mat")
    p = cohort_json_path(tmp_path)
    before = p.read_text(encoding="utf-8")

    # Sabotage json.dump on the next write so the temp file write
    # raises. The atomic pattern should leave the destination intact.
    import arealmshbm.data_io.cohort as cohort_mod
    orig_dump = cohort_mod.json.dump

    def _boom(*args, **kw):
        raise RuntimeError("simulated writer failure")

    monkeypatch.setattr(cohort_mod.json, "dump", _boom)
    with pytest.raises(RuntimeError, match="simulated"):
        _mkcohort(tmp_path, subjects=[_sub(1)], group_mat="group/v2.mat")
    monkeypatch.setattr(cohort_mod.json, "dump", orig_dump)

    # Old file still readable + unchanged.
    after = p.read_text(encoding="utf-8")
    assert before == after
    m = read_cohort(tmp_path)
    assert m.group_mat == "group/v1.mat"

    # No leftover .tmp tempfiles in the directory.
    leftover_tmps = list(p.parent.glob("cohort.json.*.tmp"))
    assert leftover_tmps == [], f"tempfiles not cleaned up: {leftover_tmps}"
