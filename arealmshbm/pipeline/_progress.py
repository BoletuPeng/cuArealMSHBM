"""_progress.py — per-subject progress emitter for frontend consumption.

The frontend (a separate process) renders a progress bar by tailing
``<project>/logs/progress.jsonl``. The driver and each per-step stage
pipeline call :class:`ProgressEmitter` at queue/seam boundaries (NOT
inside any GPU kernel or numba JIT region) to push three kinds of
events into that file:

  * ``type="plan"``  — emitted once at the start of :meth:`Pipeline.run`,
    enumerating every ``(step, sub_id)`` slot the run will visit, plus
    an optional ``meta`` dict (``mode``/``variant``/``beta_scalar``/
    ``dataset_name``) so the frontend can label the run without re-
    reading the project's pipeline_config.json. The frontend uses the
    plan to pre-populate the "pending" rows before any state event
    arrives. **Side effect:** every ``emit_plan`` call truncates the
    output file (open mode ``"w"``) so a fresh ``Pipeline.run()`` on a
    project that has been run before starts with a clean slate. Stale
    ``done``/``failed`` records from a previous run can't pollute the
    frontend's tail; the per-run audit trail still lives in
    ``logs/pipeline_run_<timestamp>.json``. **Caveat:** if the user
    SIGKILLs run N partway through (no clean shutdown → no
    pipeline_run JSON either, since the driver's finally block never
    runs), starting run N+1 will truncate run N's progress.jsonl
    before the user has a chance to inspect it. SIGINT / KeyboardInterrupt
    is safer — the driver's finally clause writes the run log and
    closes the sink, so the JSONL state at interrupt time is
    preserved on disk until N+1 explicitly truncates.
  * ``type="state"`` — ``state ∈ {"running", "done", "failed"}`` for one
    ``(step, sub_id)``. ``"pending"`` is the implicit default declared
    by the plan event; no explicit ``pending`` state event is emitted.
  * ``type="iter"``  — sub-progress inside a long cohort-level step
    (currently only step2's EM loop). Carries
    ``{iter_inter, max_inter, iter_intra?, max_intra?}``.
    **No ``sub_id`` field today** because the only emitter is step2
    (always cohort-level). Per-subject heartbeats (e.g. step3's intra-
    EM iter counter) would need ``sub_id`` added to the iter schema
    before extending — frontends should treat the field as required-
    going-forward when wiring per-subject iter events.

Timestamps
----------
``ts`` is ``time.time()`` (Unix seconds) — wall clock, not monotonic.
NTP step-adjustments can in theory make ``ts`` decrease across two
events emitted seconds apart. The authoritative event order is the
**file order** (append-only, lock-serialised); ``ts`` is a display hint
the frontend should treat as fuzzy. If you need strictly increasing
event timestamps, derive them from line number, not ``ts``.

Known limitations
-----------------
1. **Outer-loop death before first iteration.** If a stage pipeline's
   ``for sub in subjects:`` loop dies before its first iteration
   (e.g. cupy stream creation throws), no per-subject ``running``
   event is emitted for any planned slot — those slots stay
   implicitly ``pending`` until the driver's finally block closes
   the sink.
2. **Mid-loop death after partial progress.** A more common variant:
   the loop emits ``running`` for subs 0..K-1 then dies before any
   reaches the terminal stage. Subs that got their ``running`` emit
   but never reached ``done``/``failed`` stay ``running`` on the
   frontend tail — there is no synthetic ``aborted`` transition.
   This is per-design — the emitter doesn't track which slots have
   open transitions; the source of truth on disconnect / process-
   exit is the per-run JSON log
   (``logs/pipeline_run_<timestamp>.json``) which records
   ``success=False`` + ``failed_at_step``. Frontends should pair the
   JSONL tail with that file to detect "stuck running" slots after
   the driver process exits.
3. **Cross-process emit_plan race.** Two ``Pipeline.run()``
   invocations concurrently against the same project directory
   (rare — would also race on most other artifacts) will truncate
   each other's ``progress.jsonl`` via mode-``"w"`` reopens. The
   contract is one pipeline per project at a time; a file lock is
   not enforced.

Step → sub_id model::

    step0           per real subject id      ("1", "2", …)
    step1           per real subject id      (subject-major generate_profiles)
    step1_avg       sub_id="cohort"          (avg_profiles subgraph)
    step1_ini       sub_id="cohort"          (ini_params subgraph)
    step1_mask      sub_id="cohort"          (radius_mask subgraph)
    step2           sub_id="cohort"          (coupled EM, Mode B only)
    step3           per real subject id

Performance contract
--------------------
Every emit is at most one short line append + one ``flush()`` (so the
tail-reading frontend sees the line immediately). All emit call sites
sit at stage-thread queue seams where the thread is otherwise
``queue.put`` / ``queue.get`` blocked — adding ~10 µs of file I/O on
the GIL-releasing ``write()`` path costs nothing observable against
per-subject walls of ~2.7 s. The hot kernels (cuBLAS sgemm, eigsh,
numba JIT bodies) never see an emit.

Failure isolation
-----------------
Every public method swallows its own exceptions (and prints to stderr
once on first failure) — a broken progress sink must never propagate
into a stage thread and abort a real subject. ``enabled=False``
constructor flag makes every method a true no-op, so old perf scripts
and unit tests that don't care about progress can keep zero file I/O.

Thread safety
-------------
A single :class:`threading.Lock` serialises every write. The underlying
file handle is opened once at first emit (lazy open keeps zero side
effect on construction) and closed by :meth:`close` or the GC. Multiple
stage threads (step0's 4 stages, step1's 4 stages, step3's 3+ stages,
plus the driver thread) call into the same emitter concurrently —
under the lock each line lands atomically and no two events interleave
mid-line.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_PROGRESS_FILENAME = "progress.jsonl"


class ProgressEmitter:
    """Append progress events to ``<logs_dir>/progress.jsonl``.

    Constructor side effects: none. The output file is opened lazily on
    first emit (``mkdir(parents=True, exist_ok=True)`` then
    ``open(mode="a")``) so a never-emitted emitter touches no disk.

    Lifecycle::

        emitter = ProgressEmitter(layout.logs_dir)
        emitter.emit_plan([{"step": "step0", "sub_id": "1"}, ...])
        emitter.emit_state("step0", "1", "running")
        # ...
        emitter.emit_state("step0", "1", "done")
        emitter.close()

    Pass ``enabled=False`` to disable all I/O (every method becomes a
    no-op). Used by pure-CPU perf scripts and unit tests that don't
    want to assert on file output. The :meth:`disabled` classmethod is
    the idiomatic way to construct one when there's no meaningful
    logs_dir to hand it (e.g. stage-pipeline fallback defaults).
    """

    def __init__(self, logs_dir: Path | str, *, enabled: bool = True):
        self._logs_dir = Path(logs_dir)
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._fh: Optional[Any] = None
        # Set once the first emit fails so subsequent failures don't
        # spam stderr. We still try every emit — the failure could be
        # transient (disk full → freed) and the cost of an extra failed
        # open() is tiny relative to a stage wall.
        self._warned: bool = False

    @classmethod
    def disabled(cls) -> "ProgressEmitter":
        """Construct an emitter whose every method is a true no-op.

        Equivalent to ``ProgressEmitter(<any-path>, enabled=False)`` but
        avoids the awkward "pass a dummy path you never write to"
        idiom at stage-pipeline construction sites. Returns an instance
        with ``_logs_dir=None`` so accidentally calling ``_write`` on
        it would raise rather than silently writing to the wrong place
        — defence in depth on top of the ``_enabled`` short-circuit.
        """
        em = cls.__new__(cls)
        em._logs_dir = None  # type: ignore[assignment]
        em._enabled = False
        em._lock = threading.Lock()
        em._fh = None
        em._warned = False
        return em

    # ─────────────────────────────────────────────────────────────────
    # Public emit API
    # ─────────────────────────────────────────────────────────────────
    def emit_plan(self, entries: Sequence[Dict[str, str]],
                  *, meta: Optional[Dict[str, Any]] = None) -> None:
        """Declare the full set of ``(step, sub_id)`` slots this run will
        visit. Should be called exactly once at the start of
        :meth:`Pipeline.run` before any state event.

        ``entries`` items must each have ``step`` and ``sub_id`` keys —
        e.g. ``{"step": "step0", "sub_id": "1"}``. Extra keys are
        preserved verbatim. ``meta`` is an optional flat dict surfaced
        verbatim under the record's ``meta`` field; driver typically
        passes ``{"mode", "variant", "beta_scalar", "dataset_name"}``
        so the frontend can label the run without re-reading
        pipeline_config.json.

        Raises ``ValueError`` if ``meta`` contains a value that
        ``json.dumps`` cannot encode (Path, numpy scalar, datetime,
        custom object, …). The check runs on the driver thread BEFORE
        any file I/O so a programmer-bug payload doesn't leave the
        progress sink in an inconsistent state (truncated-but-empty
        file with no plan line for the frontend to read). State /
        iter events sit on stage worker threads where re-raising
        would crash a stage, so those keep the broad-except path —
        they only carry primitives anyway.

        **Side effect:** this call truncates ``progress.jsonl`` to
        zero bytes before writing the plan line. Each ``Pipeline.run()``
        invocation thereby starts the file fresh so stale ``done``/
        ``failed`` records from a previous run on the same project can't
        confuse a frontend that reads from the beginning.
        """
        if not self._enabled:
            return
        # Fail-fast validation for the plan payload: the driver thread
        # can handle a ValueError cleanly, but a json.dumps explosion
        # inside _write would be swallowed by its broad except and
        # leave the truncated file empty — frontend would see no plan
        # at all. We validate BOTH ``entries`` and ``meta`` here:
        #   * ``entries`` are normally produced by
        #     :func:`build_plan_entries` (which stringifies sub_ids,
        #     pinned by ``test_build_plan_entries_subject_ids_stringified``)
        #     but a direct external caller could bypass that and hand
        #     us non-JSON values.
        #   * ``meta`` is freeform; Path/numpy/datetime are likely
        #     accidents.
        entries_list = list(entries)
        try:
            json.dumps(entries_list, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"ProgressEmitter.emit_plan: entries list is not "
                f"JSON-serializable ({type(e).__name__}: {e}). Pass "
                f"only primitives (str, int, float, bool, None) and "
                f"containers thereof — typically you want "
                f"``build_plan_entries(...)`` which stringifies sub_ids "
                f"for you."
            ) from e
        if meta is not None:
            try:
                json.dumps(meta, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"ProgressEmitter.emit_plan: meta dict is not "
                    f"JSON-serializable ({type(e).__name__}: {e}). "
                    f"Pass only primitives (str, int, float, bool, "
                    f"None) and containers thereof — Path / numpy "
                    f"scalars / datetime / custom objects must be "
                    f"coerced (e.g. str(path), int(np_scalar)) before "
                    f"emit_plan."
                ) from e
        record: Dict[str, Any] = {
            "ts": _ts(),
            "type": "plan",
            "entries": entries_list,
        }
        if meta is not None:
            record["meta"] = dict(meta)
        self._write(record, truncate=True)

    def emit_state(self, step: str, sub_id: str, state: str,
                   *, error: Optional[str] = None) -> None:
        """Transition a ``(step, sub_id)`` slot to ``state``.

        ``state`` is one of ``"running"`` / ``"done"`` / ``"failed"``.
        On ``"failed"``, pass ``error`` as a short single-line summary
        (the stage thread already wrote the full traceback to stderr).
        """
        if not self._enabled:
            return
        record: Dict[str, Any] = {
            "ts": _ts(),
            "type": "state",
            "step": step,
            "sub_id": sub_id,
            "state": state,
        }
        if error is not None:
            record["error"] = error
        self._write(record)

    def emit_iter(self, step: str, *,
                  iter_inter: int, max_inter: int,
                  iter_intra: Optional[int] = None,
                  max_intra: Optional[int] = None) -> None:
        """Sub-progress inside a cohort-level step's iterative body.

        Currently only step2's outer/inner EM loop emits this. The
        frontend may show "step2: outer 3/50, inner 7/50" between the
        step's ``running`` and ``done`` state events.
        """
        if not self._enabled:
            return
        record: Dict[str, Any] = {
            "ts": _ts(),
            "type": "iter",
            "step": step,
            "iter_inter": int(iter_inter),
            "max_inter": int(max_inter),
        }
        if iter_intra is not None:
            record["iter_intra"] = int(iter_intra)
        if max_intra is not None:
            record["max_intra"] = int(max_intra)
        self._write(record)

    def close(self) -> None:
        """Close the underlying file handle if it was opened.

        Safe to call even when ``enabled=False`` or when no emit ever
        opened the file. Errors are swallowed.
        """
        # When disabled the file was never opened — skip the lock
        # acquisition + handle check entirely. Disabled emitters live
        # in stage-pipeline fallbacks that may be constructed in tight
        # loops; the early exit keeps them cost-free.
        if not self._enabled:
            return
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None

    # ─────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────
    def _write(self, record: Dict[str, Any], *, truncate: bool = False) -> None:
        """Serialize one record + flush. Wrapped in try/except so a
        broken sink can never crash a stage thread.

        ``truncate=True`` (passed only by :meth:`emit_plan`) closes any
        existing handle and reopens with mode ``"w"`` so the file
        starts fresh. State / iter emits use ``truncate=False`` and
        either open with mode ``"a"`` on first call or reuse the
        existing handle.
        """
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with self._lock:
                if truncate and self._fh is not None:
                    # Active handle exists (re-emit_plan from same
                    # process) — close it so the mode="w" reopen below
                    # actually truncates the on-disk file rather than
                    # leaving the previous handle's buffered content
                    # in place.
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
                if self._fh is None:
                    self._logs_dir.mkdir(parents=True, exist_ok=True)
                    # Append mode + line-buffering on first state/iter,
                    # write mode on plan (truncates any previous run's
                    # progress.jsonl). The flush below is belt-and-
                    # braces in case the OS gives us a fully buffered
                    # file (rare in append mode but Python is platform-
                    # quirky here).
                    mode = "w" if truncate else "a"
                    self._fh = open(
                        self._logs_dir / _PROGRESS_FILENAME,
                        mode=mode, encoding="utf-8", buffering=1,
                    )
                self._fh.write(line)
                self._fh.flush()
        except Exception as e:
            # Intentional: we do NOT reset ``self._fh`` here. If open()
            # succeeded but a later write() raised (disk full mid-run,
            # remote share dropped, …), the partial handle stays and
            # every subsequent emit will also fail. That's fine —
            # ``_warned`` ensures the user only sees one stderr line,
            # and the per-run JSON log will still record the failure.
            # Recovering the sink mid-run is out of scope; the run
            # continues with progress events silently dropping.
            if not self._warned:
                self._warned = True
                print(
                    f"WARNING: ProgressEmitter write failed "
                    f"({type(e).__name__}: {e}); progress events for "
                    f"this run may be incomplete",
                    file=sys.stderr,
                )

    # Context manager support — convenience for the tests + ad-hoc CLI.
    # The driver owns the emitter for the lifetime of ``Pipeline``, so
    # it uses ``close()`` directly rather than ``with``.
    def __enter__(self) -> "ProgressEmitter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────
# Plan-entry helpers — keep the plan schema in one place so driver and
# tests speak the exact same shape.
# ─────────────────────────────────────────────────────────────────────
COHORT_SUB_ID = "cohort"

#: Cohort-level pseudo-steps. The frontend renders these as ordinary
#: rows alongside per-subject step0/step1/step3 entries. Order here is
#: the order in which the driver actually runs them.
COHORT_STEPS = (
    "step1_avg",
    "step1_ini",
    "step1_mask",
    "step2",
)


def build_plan_entries(
    subject_ids: Sequence[str],
    *,
    include_step2: bool,
) -> List[Dict[str, str]]:
    """Build the full ``entries`` list for one driver run.

    Real per-subject slots: ``step0``, ``step1``, ``step3``, one entry
    per subject id (in input order). Cohort-level slots: ``step1_avg``,
    ``step1_ini``, ``step1_mask`` always; ``step2`` only when
    ``include_step2=True`` (Mode B).
    """
    out: List[Dict[str, str]] = []
    for sid in subject_ids:
        out.append({"step": "step0", "sub_id": str(sid)})
    for sid in subject_ids:
        out.append({"step": "step1", "sub_id": str(sid)})
    out.append({"step": "step1_avg", "sub_id": COHORT_SUB_ID})
    out.append({"step": "step1_ini", "sub_id": COHORT_SUB_ID})
    out.append({"step": "step1_mask", "sub_id": COHORT_SUB_ID})
    if include_step2:
        out.append({"step": "step2", "sub_id": COHORT_SUB_ID})
    for sid in subject_ids:
        out.append({"step": "step3", "sub_id": str(sid)})
    return out


def _ts() -> float:
    """Single-source timestamp for every event. Unix seconds with
    microsecond fraction — JSON-friendly. Wall-clock, not monotonic;
    see the module docstring's "Timestamps" section for the contract."""
    return time.time()
