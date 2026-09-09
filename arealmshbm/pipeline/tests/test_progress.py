# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""Unit tests for :mod:`arealmshbm.pipeline._progress`.

Covers:
  - JSONL output is one valid JSON object per line, with the expected
    schema (plan / state / iter).
  - Concurrent emits from multiple threads never interleave mid-line
    (Lock serialises the file append).
  - ``enabled=False`` is a true no-op — no file is created.
  - A broken sink (read-only logs dir) doesn't propagate exceptions
    into the caller — the stage thread must never crash on a sink
    failure.
  - The plan-builder produces the expected ``(step, sub_id)`` slot set
    for Mode A and Mode B.
  - The driver's ``Pipeline.__init__`` constructs the emitter pointing
    at the project's logs dir (without touching disk).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from arealmshbm.pipeline._progress import (
    COHORT_SUB_ID,
    COHORT_STEPS,
    ProgressEmitter,
    build_plan_entries,
)


# ─────────────────────────────────────────────────────────────────────
# JSONL output schema
# ─────────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    """Parse ``progress.jsonl``: one JSON object per line, strict."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_emit_plan_writes_one_line(tmp_path: Path) -> None:
    em = ProgressEmitter(tmp_path)
    entries = [
        {"step": "step0", "sub_id": "1"},
        {"step": "step3", "sub_id": "1"},
    ]
    em.emit_plan(entries)
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert len(rows) == 1
    assert rows[0]["type"] == "plan"
    assert rows[0]["entries"] == entries
    assert "ts" in rows[0] and isinstance(rows[0]["ts"], (int, float))
    # No meta key when caller didn't pass one.
    assert "meta" not in rows[0]


def test_emit_plan_carries_meta_dict(tmp_path: Path) -> None:
    """``emit_plan(..., meta=...)`` surfaces the dict verbatim under
    the plan record's ``meta`` key. Driver passes mode/variant/beta/
    dataset_name so the frontend can label the run without re-parsing
    pipeline_config.json or bold_inputs.json."""
    em = ProgressEmitter(tmp_path)
    meta = {
        "mode": "modeB_train_prior",
        "variant": "gMSHBM",
        "beta_scalar": 5,
        "dataset_name": "demo_cohort",
    }
    em.emit_plan([{"step": "step0", "sub_id": "1"}], meta=meta)
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert rows[0]["meta"] == meta


def test_emit_plan_truncates_stale_file(tmp_path: Path) -> None:
    """A previous ``Pipeline.run()`` on the same project may have left
    a populated ``progress.jsonl``; the new run's ``emit_plan`` must
    truncate so a frontend reading from the beginning never sees the
    previous run's ``done``/``failed`` records as if they belonged to
    the current run.

    The reviewer concern (P2 #1): without truncate, append-mode
    preservation + reused ``(step, sub_id)`` keys would let stale
    rows render subjects as complete before this run reaches them.
    """
    # Pre-stage a "previous run" worth of records: plan + a couple of
    # state events on subject ids that the new run will also touch.
    em = ProgressEmitter(tmp_path)
    em.emit_plan([
        {"step": "step0", "sub_id": "1"},
        {"step": "step3", "sub_id": "1"},
    ])
    em.emit_state("step0", "1", "running")
    em.emit_state("step0", "1", "done")
    em.emit_state("step3", "1", "done")
    em.close()

    pre_rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert len(pre_rows) == 4  # 1 plan + 3 state

    # New run: same project, new emitter, new plan. The plan emit must
    # truncate so the file contains ONLY the new run's events.
    em2 = ProgressEmitter(tmp_path)
    em2.emit_plan([
        {"step": "step0", "sub_id": "1"},
        {"step": "step3", "sub_id": "1"},
    ])
    em2.emit_state("step0", "1", "running")
    em2.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert len(rows) == 2  # only new plan + 1 new state
    assert rows[0]["type"] == "plan"
    assert rows[1]["type"] == "state"
    assert rows[1]["state"] == "running"  # NOT the stale "done"


def test_emit_plan_truncates_within_same_emitter(tmp_path: Path) -> None:
    """Same emitter, two ``emit_plan`` calls — the second truncates
    the file (and closes/reopens the active handle so the bytes
    actually drop). Edge case in case a caller restarts a plan inside
    one process (e.g. retry-after-validation flow)."""
    em = ProgressEmitter(tmp_path)
    em.emit_plan([{"step": "step0", "sub_id": "1"}])
    em.emit_state("step0", "1", "running")
    em.emit_state("step0", "1", "done")

    # Second plan, same emitter — must truncate.
    em.emit_plan([{"step": "step0", "sub_id": "2"}])
    em.emit_state("step0", "2", "running")
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert len(rows) == 2  # only the second plan + its one state
    assert rows[0]["entries"] == [{"step": "step0", "sub_id": "2"}]
    assert rows[1]["sub_id"] == "2"


def test_emit_state_writes_running_and_done(tmp_path: Path) -> None:
    em = ProgressEmitter(tmp_path)
    em.emit_state("step0", "1", "running")
    em.emit_state("step0", "1", "done")
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert [r["state"] for r in rows] == ["running", "done"]
    assert all(r["type"] == "state" for r in rows)
    assert all(r["step"] == "step0" and r["sub_id"] == "1" for r in rows)
    assert all("error" not in r for r in rows)


def test_emit_state_failed_carries_error(tmp_path: Path) -> None:
    em = ProgressEmitter(tmp_path)
    em.emit_state("step3", "2", "failed", error="ValueError: bad input")
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert rows[0]["state"] == "failed"
    assert rows[0]["error"] == "ValueError: bad input"


def test_emit_iter_inter_only(tmp_path: Path) -> None:
    em = ProgressEmitter(tmp_path)
    em.emit_iter("step2", iter_inter=3, max_inter=50)
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert rows[0]["type"] == "iter"
    assert rows[0]["step"] == "step2"
    assert rows[0]["iter_inter"] == 3
    assert rows[0]["max_inter"] == 50
    assert "iter_intra" not in rows[0]
    assert "max_intra" not in rows[0]


def test_emit_iter_with_intra(tmp_path: Path) -> None:
    em = ProgressEmitter(tmp_path)
    em.emit_iter(
        "step2", iter_inter=3, max_inter=50,
        iter_intra=7, max_intra=50,
    )
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert rows[0]["iter_intra"] == 7
    assert rows[0]["max_intra"] == 50


# ─────────────────────────────────────────────────────────────────────
# Concurrency — lines must not interleave under multi-thread emit
# ─────────────────────────────────────────────────────────────────────
def test_concurrent_emit_does_not_interleave(tmp_path: Path) -> None:
    em = ProgressEmitter(tmp_path)
    n_threads = 16
    emits_per_thread = 50

    def hammer(tid: int) -> None:
        for k in range(emits_per_thread):
            em.emit_state(f"step{tid % 4}", str(k), "running")

    threads = [threading.Thread(target=hammer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    # Total count is exact (no dropped lines under lock).
    assert len(rows) == n_threads * emits_per_thread
    # Every row parses as valid JSON with the expected schema; if
    # any two writes had interleaved mid-line, json.loads would have
    # raised on at least one line, which _read_jsonl propagates.
    assert all(r["type"] == "state" for r in rows)
    assert all(r["state"] == "running" for r in rows)


# ─────────────────────────────────────────────────────────────────────
# enabled=False — true no-op
# ─────────────────────────────────────────────────────────────────────
def test_disabled_emitter_creates_no_file(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs_should_not_exist"
    em = ProgressEmitter(logs_dir, enabled=False)
    em.emit_plan([{"step": "step0", "sub_id": "1"}])
    em.emit_state("step0", "1", "running")
    em.emit_iter("step2", iter_inter=1, max_inter=10)
    em.close()

    # The logs dir itself must not be created — the disabled emitter
    # is a true no-op, including no mkdir.
    assert not logs_dir.exists()


def test_disabled_classmethod_attribute_parity(tmp_path: Path) -> None:
    """``ProgressEmitter.disabled()`` uses ``cls.__new__(cls)`` to
    bypass ``__init__`` (so it doesn't need a meaningful logs_dir).
    The downside: any future attribute added in ``__init__`` would be
    silently missing on disabled instances, and the first method to
    read it would ``AttributeError``. This parity test pins the
    invariant — every attribute set by ``__init__`` must also be set
    by ``disabled()``.

    Reviewer note: this is the "fragile factory" risk flagged on PR
    #66 R2. The alternative (calling ``cls("", enabled=False)``)
    would not need this test, but loses the ``_logs_dir=None``
    defence-in-depth (an accidental ``_write`` on a disabled emitter
    would silently scribble to ``./progress.jsonl`` instead of
    exploding).
    """
    real_attrs = set(ProgressEmitter(tmp_path).__dict__.keys())
    disabled_attrs = set(ProgressEmitter.disabled().__dict__.keys())
    missing = real_attrs - disabled_attrs
    assert not missing, (
        f"ProgressEmitter.disabled() is missing attributes that "
        f"__init__ sets: {sorted(missing)}. Update the classmethod "
        f"to assign each new attribute, or switch the factory to "
        f"``cls(<path>, enabled=False)`` (and remove the "
        f"``_logs_dir=None`` defence). Either way, keep this test "
        f"green."
    )


def test_disabled_classmethod_is_true_noop(tmp_path: Path) -> None:
    """``ProgressEmitter.disabled()`` is the idiomatic way to get a
    no-op emitter when there's no meaningful logs_dir to hand it (used
    by stage-pipeline fallbacks when the caller didn't pass a real
    emitter). Asserts (a) no constructor argument is needed and (b)
    every emit method is a no-op (no exception, no file, no warning)."""
    em = ProgressEmitter.disabled()
    em.emit_plan([{"step": "step0", "sub_id": "1"}])
    em.emit_state("step0", "1", "running")
    em.emit_state("step0", "1", "done")
    em.emit_iter("step2", iter_inter=1, max_inter=10)
    em.close()
    # No cwd-relative file created — the disabled emitter has
    # ``_logs_dir=None`` so even a buggy _write would explode loudly
    # rather than silently scribble somewhere unexpected.
    assert not (tmp_path / "progress.jsonl").exists()
    assert not Path("progress.jsonl").exists()
    # Disabled state stays honest.
    assert em._enabled is False
    assert em._logs_dir is None


# ─────────────────────────────────────────────────────────────────────
# Sink failure isolation
# ─────────────────────────────────────────────────────────────────────
def test_broken_sink_does_not_raise(tmp_path: Path, capsys) -> None:
    """If the logs path can't be opened (e.g. a file exists where the
    dir should be), the emitter must swallow the error and warn once
    on stderr — NEVER propagate into the calling stage thread."""
    # Create a file at the path where logs_dir would go, so mkdir
    # would succeed but open(progress.jsonl, "a") would fail because
    # the parent is a file, not a dir. Actually mkdir(exist_ok=True)
    # raises FileExistsError when the existing path is a file — that
    # surfaces inside _write and must be swallowed.
    sentinel = tmp_path / "blocked"
    sentinel.write_text("not a directory", encoding="utf-8")

    em = ProgressEmitter(sentinel)
    # Each emit must return normally even though the underlying mkdir
    # raises FileExistsError (parent exists as a file, not a dir).
    em.emit_state("step0", "1", "running")
    em.emit_state("step0", "1", "done")
    em.close()

    # The first failure should have printed a single warning to stderr.
    captured = capsys.readouterr()
    assert "ProgressEmitter write failed" in captured.err


# ─────────────────────────────────────────────────────────────────────
# Plan-builder schema
# ─────────────────────────────────────────────────────────────────────
def test_build_plan_entries_mode_a(tmp_path: Path) -> None:
    """Mode A: no step2 entry. Order is step0×K, step1×K, cohort triple,
    step3×K — matches the driver's actual execution order."""
    entries = build_plan_entries(["1", "2", "3"], include_step2=False)

    steps = [e["step"] for e in entries]
    subs = [e["sub_id"] for e in entries]
    # 3 step0 + 3 step1 + 3 cohort + 3 step3 = 12
    assert len(entries) == 12
    assert steps == [
        "step0", "step0", "step0",
        "step1", "step1", "step1",
        "step1_avg", "step1_ini", "step1_mask",
        "step3", "step3", "step3",
    ]
    assert subs == [
        "1", "2", "3",
        "1", "2", "3",
        COHORT_SUB_ID, COHORT_SUB_ID, COHORT_SUB_ID,
        "1", "2", "3",
    ]
    # No step2 entry in Mode A.
    assert "step2" not in steps


def test_build_plan_entries_mode_b_has_step2() -> None:
    entries = build_plan_entries(["1", "2"], include_step2=True)
    step2_entries = [e for e in entries if e["step"] == "step2"]
    assert len(step2_entries) == 1
    assert step2_entries[0]["sub_id"] == COHORT_SUB_ID
    # step2 sits between the cohort triple and step3×K — frontend
    # renders the rows in this exact order.
    steps = [e["step"] for e in entries]
    assert steps.index("step2") > steps.index("step1_mask")
    assert steps.index("step2") < steps.index("step3")


def test_build_plan_entries_subject_ids_stringified() -> None:
    """``sub_id`` is always a string in the plan, matching the
    state-event side. Integer subject ids from
    ``BoldInputs.subject_ids()`` get stringified here so the frontend
    never sees a type-mismatch between plan and state rows."""
    entries = build_plan_entries([1, 2, 3], include_step2=False)
    sub_ids = [e["sub_id"] for e in entries if e["sub_id"] != COHORT_SUB_ID]
    assert all(isinstance(s, str) for s in sub_ids)
    assert set(sub_ids) == {"1", "2", "3"}


def test_cohort_steps_constant_matches_plan_keys() -> None:
    """``COHORT_STEPS`` lists every cohort-level pseudo-step the
    builder can emit. Keeping the constant in sync with the builder
    means downstream consumers (frontend, log analysers) can rely on
    it as the canonical list."""
    entries = build_plan_entries(["1"], include_step2=True)
    emitted_cohort = {
        e["step"] for e in entries if e["sub_id"] == COHORT_SUB_ID
    }
    assert emitted_cohort == set(COHORT_STEPS)


# ─────────────────────────────────────────────────────────────────────
# Driver wires emitter at __init__ pointing at project logs dir
# ─────────────────────────────────────────────────────────────────────
def _write_minimal_mode_a_project(root: Path) -> Path:
    """Write the smallest valid Mode A single-subject project tree.

    We crib ``pipeline_config.json`` verbatim from the in-tree sample
    so this fixture stays in sync with the schema as it evolves, and
    write a one-subject-one-session ``bold_inputs.json`` with dummy
    ``.gii`` paths (parser checks extension, not existence).

    Just enough for ``Pipeline.__init__`` to parse — we do NOT call
    ``run()`` (which would need real BOLD files, mesh, etc.).
    """
    # Walk up to repo root from this test file: …/arealmshbm/pipeline/tests/<this>.
    repo_root = Path(__file__).resolve().parents[3]
    sample = repo_root / "projects" / "sample_modeA_single"

    project = root / "tiny_project"
    project.mkdir()
    # Reuse the sample's pipeline_config.json verbatim — it's the
    # ground truth for "what a valid schema v2 config looks like".
    (project / "pipeline_config.json").write_text(
        (sample / "pipeline_config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # bold_inputs.json — minimal: 1 subject × 1 session, dummy .gii
    # paths. Parser only validates the .gii suffix at parse time;
    # existence is checked later in _check_min_inputs (we never hit
    # that path because we don't call run()).
    (project / "bold_inputs.json").write_text(json.dumps({
        "schema_version": "1",
        "dataset_name": "test_progress_fixture",
        "targ_mesh": "fsaverage6",
        "seed_mesh": "fsaverage3",
        "subjects": [{"id": "1", "sessions": [
            {"id": "1",
             "lh": str(project / "lh.dummy.func.gii"),
             "rh": str(project / "rh.dummy.func.gii")},
        ]}],
    }), encoding="utf-8")
    return project


# ─────────────────────────────────────────────────────────────────────
# Meta validation — bad meta raises ValueError on driver thread
# ─────────────────────────────────────────────────────────────────────
def test_emit_plan_rejects_non_json_meta(tmp_path: Path) -> None:
    """``emit_plan(meta=...)`` validates the meta dict for JSON-
    serializability BEFORE truncating the file. Bad meta (Path,
    numpy scalar, datetime, custom object) is a programmer bug — it
    should raise ValueError on the driver thread, NOT be swallowed
    by ``_write``'s broad except clause (which would leave the file
    truncated-but-empty and the frontend with no plan to render).

    Reviewer P1 #1 contract pin.
    """
    em = ProgressEmitter(tmp_path)
    # Path objects are NOT json.dumps-able by default.
    bad_meta = {"mode": "modeA_single", "logs_dir": Path("/tmp/x")}

    with pytest.raises(ValueError, match="JSON-serializable"):
        em.emit_plan([{"step": "step0", "sub_id": "1"}], meta=bad_meta)

    # The file must NOT exist (validation runs before truncate) — so
    # if there was a pre-existing progress.jsonl from a prior run,
    # the failed validation didn't destroy it.
    assert not (tmp_path / "progress.jsonl").exists()


def test_emit_plan_rejects_non_json_entries(tmp_path: Path) -> None:
    """Symmetric with meta validation (R3-I4): direct callers
    bypassing ``build_plan_entries`` could hand emit_plan a list with
    non-JSON values (e.g. ``Path`` for sub_id, or a raw
    ``BoldInputSubject`` dataclass). The validation runs eagerly so
    the file truncate doesn't fire on a bad payload.

    The default path is safe: ``build_plan_entries`` stringifies
    sub_ids (pinned by
    ``test_build_plan_entries_subject_ids_stringified``). This test
    pins the direct-caller defence."""
    em = ProgressEmitter(tmp_path)
    bad_entries = [
        {"step": "step0", "sub_id": Path("/x")},  # Path is non-JSON
    ]
    with pytest.raises(ValueError, match="JSON-serializable"):
        em.emit_plan(bad_entries)
    # File untouched.
    assert not (tmp_path / "progress.jsonl").exists()


def test_emit_plan_rejects_non_json_meta_preserves_prior_file(
    tmp_path: Path,
) -> None:
    """Stronger guarantee: if a prior run's progress.jsonl exists,
    a failed meta-validation must not truncate it. The driver thread
    sees the ValueError; the previous run's records remain on disk
    for forensic inspection."""
    # Pre-stage a previous run's file.
    em_prev = ProgressEmitter(tmp_path)
    em_prev.emit_plan([{"step": "step0", "sub_id": "1"}])
    em_prev.emit_state("step0", "1", "done")
    em_prev.close()

    pre_size = (tmp_path / "progress.jsonl").stat().st_size
    pre_rows = _read_jsonl(tmp_path / "progress.jsonl")
    assert pre_size > 0 and len(pre_rows) == 2

    # New emitter, bad meta — must raise without touching the file.
    em = ProgressEmitter(tmp_path)
    with pytest.raises(ValueError, match="JSON-serializable"):
        em.emit_plan(
            [{"step": "step0", "sub_id": "1"}],
            meta={"path": Path("/x")},  # Path is the canonical bad value
        )

    # File still has the prior content; not truncated.
    assert (tmp_path / "progress.jsonl").stat().st_size == pre_size
    assert _read_jsonl(tmp_path / "progress.jsonl") == pre_rows


# ─────────────────────────────────────────────────────────────────────
# Structural pins — each stage pipeline + driver must contain the
# expected emit_state / emit_plan calls. Brittle against renames but
# cheap, and catches outright removal that no unit test would notice
# (reviewer R1-P2-4 / R2 future-friendliness).
# ─────────────────────────────────────────────────────────────────────
def _read_module_source(module_relpath: str) -> str:
    """Read a project-relative module source as text. Used by the
    structural pins to grep for emit_state call patterns."""
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / module_relpath).read_text(encoding="utf-8")


def test_driver_contains_plan_emit() -> None:
    """``Pipeline.run`` must emit the plan at the head of the run.
    A copy-paste refactor removing the call would leave the frontend
    blind to the slot roster.

    Also pins R3-I1: the ``current_phase = "emit_plan"`` sentinel
    sits next to the call, proving emit_plan lives INSIDE the try
    block (so the finally that writes the run log + closes the
    progress sink still runs if emit_plan raises ValueError on a
    bad meta payload). If the assignment ever drifts away from the
    call, the test catches it via the proximity check."""
    src = _read_module_source("arealmshbm/pipeline/driver.py")
    assert "emit_plan(" in src, \
        "Pipeline.run lost its emit_plan call — frontend has no roster"
    assert "build_plan_entries(" in src, \
        "Pipeline.run lost its build_plan_entries call"
    # current_phase sentinel must appear before the emit_plan call so
    # the finally block writes a run log with failed_at_step="emit_plan"
    # on a meta-validation failure rather than skipping the audit
    # trail entirely.
    phase_idx = src.find('current_phase = "emit_plan"')
    emit_idx = src.find("emit_plan(")
    assert phase_idx != -1, (
        "Pipeline.run lost its ``current_phase = \"emit_plan\"`` "
        "sentinel — the emit_plan call should sit inside the try "
        "block so its ValueError doesn't bypass the run-log write "
        "(R3-I1)."
    )
    assert phase_idx < emit_idx, (
        "``current_phase = \"emit_plan\"`` must precede the emit_plan "
        "call so failed_at_step is correctly labelled on a "
        "meta-validation failure (R3-I1)."
    )


def test_step0_stage_pipeline_emits_running_and_terminal() -> None:
    """``_stage_A`` opens the per-subject lifecycle with ``running``
    and ``_stage_D`` closes it with ``done`` / ``failed``. Removing
    either site silently breaks the frontend without failing any
    unit test (the GPU stage pipelines are too coupled to mock
    cheaply)."""
    src = _read_module_source("arealmshbm/pipeline/_step0_stage_pipeline.py")
    assert 'emit_state("step0"' in src, \
        "step0 stage pipeline lost all emit_state calls"
    assert '"running"' in src, "step0 missing running emit"
    assert '"done"' in src, "step0 missing done emit"
    assert '"failed"' in src, "step0 missing failed emit"


def test_step1_stage_pipeline_emits_running_and_terminal() -> None:
    """``_prime_loop`` opens with ``running``; ``_finish`` closure in
    ``_stage_ASSEM`` is the single terminal chokepoint for done /
    failed. The closure is a structural guarantee — but only as long
    as it actually contains both terminal emit branches."""
    src = _read_module_source("arealmshbm/pipeline/_step1_stage_pipeline.py")
    assert 'emit_state("step1"' in src, \
        "step1 stage pipeline lost all emit_state calls"
    assert '"running"' in src, "step1 missing running emit"
    assert '"done"' in src, "step1 missing done emit (in _finish?)"
    assert '"failed"' in src, "step1 missing failed emit (in _finish?)"


def test_step1_stage_pipeline_pops_buffer_on_failure() -> None:
    """Reviewer R3-P3: when a per-session failure arrives for a sub
    that already has a partial bucket in ``buffers``, the bucket
    must be popped so its bitpacked slabs (~4 MB per session on
    fsa6 K=400, unbounded across failures in one run) become
    collectable. Pin the pop call in the failure path."""
    src = _read_module_source("arealmshbm/pipeline/_step1_stage_pipeline.py")
    # The two paths that drop a sub mid-flight: the upstream-failure
    # branch and the assembly-exception branch.
    assert src.count("buffers.pop(work.sub_id, None)") >= 2, (
        "step1 _stage_ASSEM lost one of its buffer.pop calls — the "
        "failure path now leaks the partial slab bucket (R3-P3)"
    )


def test_step1_stage_pipeline_suppresses_failed_then_done() -> None:
    """Reviewer R1-P2-3: a pre-existing buffer-logic edge case
    means a per-session failure followed by later sessions of the
    same subject assembling + writing successfully could emit
    ``failed`` then ``done`` in that order. Pin the "first terminal
    wins" suppression in ``_stage_ASSEM`` so a future refactor
    doesn't silently re-introduce out-of-order terminals."""
    src = _read_module_source("arealmshbm/pipeline/_step1_stage_pipeline.py")
    # Sentinel: the ``terminated`` set is the carrier of the
    # suppression invariant. Removing or renaming it should fail
    # this test and force the refactor to consciously re-derive
    # "first terminal wins."
    assert "terminated: set[str]" in src or "terminated = set()" in src, (
        "step1 _stage_ASSEM lost its per-subject ``terminated`` set — "
        "failed→done ordering is no longer suppressed (R1-P2-3 regression)"
    )
    # _finish's idempotency guard must also be present.
    assert "if sub_id in terminated:" in src, (
        "step1 _finish lost its idempotency guard — duplicate terminal "
        "emits can leak through"
    )


def test_step3_stage_pipeline_emits_running_and_terminal() -> None:
    """``_stage_LOAD`` opens with ``running``; ``_stage_SAVE`` closes
    with done / failed before ``qDone.put``."""
    src = _read_module_source("arealmshbm/pipeline/_step3_stage_pipeline.py")
    assert 'emit_state("step3"' in src, \
        "step3 stage pipeline lost all emit_state calls"
    assert '"running"' in src, "step3 missing running emit"
    assert '"done"' in src, "step3 missing done emit"
    assert '"failed"' in src, "step3 missing failed emit"


# ─────────────────────────────────────────────────────────────────────
# Step2Pipeline contract — must NOT auto-emit ``done`` (driver owns it)
# ─────────────────────────────────────────────────────────────────────
def test_step2_run_emits_running_but_not_done(tmp_path: Path) -> None:
    """Contract pin for reviewer P2 #2.

    ``Step2Pipeline.run()`` emits ``running`` at entry and ``failed``
    on exception but deliberately does NOT emit ``done`` at success.
    The done emit lives in ``Pipeline._run_step2_train_prior`` AFTER
    the ``prior_dst.exists()`` check — otherwise a silent early-exit
    (Step2 returns without writing Params_Final.mat) would mark the
    slot complete on the frontend even though the driver immediately
    raises ``FileNotFoundError``.

    This test runs ``Step2Pipeline.run()`` with its three internal
    methods stubbed so it succeeds without touching real BOLD / mesh
    files, then asserts the emitted state stream contains ``running``
    but not ``done``.
    """
    from arealmshbm.step2_pipeline.pipeline import Step2Pipeline

    # Cheapest valid instance: __new__ + manual attr set so we don't
    # have to satisfy Step2Config (which would pull mesh / cohort).
    # ``SimpleNamespace`` over an ad-hoc class for readability — both
    # AttributeError on missing attrs, so if Step2Pipeline.run grows
    # another ``self.cfg.X`` access this test fails loudly with the
    # missing attr name in the traceback (R3-nit).
    pipe = Step2Pipeline.__new__(Step2Pipeline)
    em = ProgressEmitter(tmp_path)
    pipe.progress = em
    pipe.timings = {}
    # ``backend`` is read by run()'s stage dispatch — 'cpu' keeps the
    # dense triple (the three methods stubbed below).
    pipe.cfg = SimpleNamespace(verbose=False, backend="cpu")

    # Stub the three internal methods so run() succeeds trivially.
    pipe.load_inputs = lambda: object()
    pipe.initialize_params = lambda inputs: {}
    pipe.run_em = lambda Params, inputs: object()

    pipe.run()
    em.close()

    rows = _read_jsonl(tmp_path / "progress.jsonl")
    states = [r for r in rows if r.get("type") == "state"]
    assert any(r["state"] == "running" for r in states), \
        "Step2Pipeline.run must still emit 'running' at entry"
    assert not any(r["state"] == "done" for r in states), (
        "Step2Pipeline.run emitted 'done' — that emit belongs in the "
        "driver wrapper AFTER the prior_dst.exists() check "
        "(reviewer P2 #2). Move it back to "
        "driver._run_step2_train_prior."
    )


def test_pipeline_init_constructs_emitter_pointed_at_logs_dir(tmp_path: Path) -> None:
    """``Pipeline.__init__`` must attach a real :class:`ProgressEmitter`
    targeting ``<project>/logs``. Constructing the emitter alone must
    not touch disk (lazy open) — so the logs dir is still absent after
    ``__init__``."""
    project = _write_minimal_mode_a_project(tmp_path)

    # Build a dummy prior so _check_min_inputs would pass; but we
    # never call run(), so this isn't actually needed here. We just
    # want __init__ to succeed.
    from arealmshbm.pipeline.driver import Pipeline

    try:
        pipe = Pipeline(project)
    except Exception as e:
        # If the minimal project doesn't pass parsing (config schema
        # drift), the test still asserts the contract precisely so a
        # future schema bump tells us to update this fixture rather
        # than silently skipping.
        pytest.fail(f"Pipeline.__init__ failed on minimal project: {e}")

    assert isinstance(pipe._progress, ProgressEmitter)
    # Pointed at the project's logs dir.
    assert pipe._progress._logs_dir == project / "logs"
    # Lazy open — no file written yet.
    assert not (project / "logs" / "progress.jsonl").exists()
