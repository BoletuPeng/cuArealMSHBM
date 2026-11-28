"""test_step1_stage_pipeline.py — the CPU step-1 coordinator
(:class:`Step1GenerateProfilesStagePipeline`) on a synthetic schedule.

The leaf and the ``.b2nd`` writer are stubbed; the threads, queues and
per-subject assembly are the real ones.

  * Each subject's sessions land in one ``(T, V_lh + V_rh, K)`` array,
    in schedule order, V-major — whatever the subject's session count.
  * The prefetcher is primed once per session, in schedule order.
  * A failing session fails its subject once, the other subjects are
    still written, and ``run()`` raises the failure.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

import arealmshbm.data_io.profile_io as profile_io
import arealmshbm.generate_profiles as generate_profiles
from arealmshbm.pipeline._step1_stage_pipeline import (
    Step1GenerateProfilesStagePipeline,
)

K, V_LH, V_RH = 3, 4, 5


def _leaf_pair(sub: str, sess: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(K, V_h) fp32`` pair whose every cell names its (sub, sess, k, v)."""
    base = 1000.0 * int(sub) + 100.0 * int(sess)
    lh = base + np.arange(K * V_LH, dtype=np.float32).reshape(K, V_LH)
    rh = -base - np.arange(K * V_RH, dtype=np.float32).reshape(K, V_RH)
    return lh.astype(np.float32), rh.astype(np.float32)


class _Prefetcher:
    def __init__(self) -> None:
        self.primed: List[Tuple[str, str]] = []

    def prime(self, sub, sess, lh_paths, rh_paths) -> None:
        self.primed.append((sub, sess))

    def get(self, sub, sess):
        return [(sub, sess)]


class _Progress:
    def __init__(self) -> None:
        self.states: List[Tuple[str, str]] = []

    def emit_state(self, step, sub_id, state, *, error=None) -> None:
        self.states.append((sub_id, state))

    def of(self, sub_id: str) -> List[str]:
        return [s for sid, s in self.states if sid == sub_id]


def _run(monkeypatch, tmp_path: Path, schedule: List[Tuple[str, str]],
         fail: Optional[Tuple[str, str]] = None):
    written: Dict[str, np.ndarray] = {}

    def leaf(out_dir, sub, sess, threshold, bold_runs):
        assert bold_runs == [(sub, sess)]
        if (sub, sess) == fail:
            raise RuntimeError(f"boom sub{sub} sess{sess}")
        return _leaf_pair(sub, sess)

    def write(out_path, arr):
        written[Path(out_path).stem] = arr.copy()

    monkeypatch.setattr(generate_profiles, "compute_profile_arrays", leaf)
    monkeypatch.setattr(profile_io, "write_subject_profile_tnd", write)

    subjects = list(dict.fromkeys(sub for sub, _ in schedule))
    prefetcher, progress = _Prefetcher(), _Progress()
    coord = Step1GenerateProfilesStagePipeline(
        project_dir=tmp_path,
        schedule=schedule,
        bold_paths={key: (["lh"], ["rh"]) for key in schedule},
        out_paths={sub: tmp_path / f"{sub}.b2nd" for sub in subjects},
        threshold=0.1,
        prefetcher=prefetcher,
        verbose=False,
        progress=progress,
    )
    return coord, written, prefetcher, progress


def test_ragged_schedule_assembles_each_subject(monkeypatch, tmp_path) -> None:
    # 4 + 1 + 6 sessions: unequal counts, and more entries than the
    # prime loop's lookahead window.
    counts = {"1": 4, "2": 1, "3": 6}
    schedule = [(sub, str(s)) for sub, n in counts.items()
                for s in range(1, n + 1)]
    coord, written, prefetcher, progress = _run(monkeypatch, tmp_path, schedule)

    out = coord.run()

    assert out == [(sub, tmp_path / f"{sub}.b2nd") for sub in counts]
    assert prefetcher.primed == schedule
    for sub, n in counts.items():
        arr = written[sub]
        assert arr.shape == (n, V_LH + V_RH, K) and arr.dtype == np.float32
        for t in range(n):
            lh, rh = _leaf_pair(sub, str(t + 1))
            assert np.array_equal(arr[t, :V_LH], lh.T)
            assert np.array_equal(arr[t, V_LH:], rh.T)
        assert progress.of(sub) == ["running", "done"]


def test_failed_session_fails_its_subject_once(monkeypatch, tmp_path) -> None:
    schedule = [("1", "1"), ("1", "2"),
                ("2", "1"), ("2", "2"), ("2", "3"),
                ("3", "1")]
    coord, written, _, progress = _run(
        monkeypatch, tmp_path, schedule, fail=("2", "2"))

    with pytest.raises(RuntimeError, match="boom sub2 sess2"):
        coord.run()

    assert sorted(written) == ["1", "3"]
    assert written["1"].shape == (2, V_LH + V_RH, K)
    assert progress.of("1") == ["running", "done"]
    assert progress.of("2") == ["running", "failed"]
    assert progress.of("3") == ["running", "done"]
