"""test_background_write.py - failure semantics of the shared background
write handle (avg-profile ``.npy`` pair, ``group.mat``).

Pins the three things that make a background write safe to hand out:

  * a writer-thread exception is re-raised by EVERY ``wait()``, not
    just the first (a late joiner must not see a failed write as done);
  * the failure is announced at failure time via a ``RuntimeWarning``
    from the done-callback, so a caller that never joins is not left
    with a silently missing file;
  * dropping a handle whose write failed without ever joining warns
    again from the finalizer.

Plus the happy path: no warning, no latched exception, idempotent wait,
``wait()`` returns the ``result`` it was constructed with.

Pure CPU - no cupy, no mesh bundle.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import gc
import time
import warnings
from functools import partial

import numpy as np
import pytest

from arealmshbm.data_io._background_write import (
    BackgroundWriteHandle, submit_background_write,
)


def _start(out_dir, *, make_dirs: bool):
    """Two np.save jobs; ``make_dirs=False`` makes them fail (no dir)."""
    target = out_dir / "sub"
    if make_dirs:
        target.mkdir(parents=True, exist_ok=True)
    a_path, b_path = target / "a.npy", target / "b.npy"
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = a + 10.0
    return submit_background_write(
        (a_path, b_path), "test pair",
        partial(np.save, a_path, a, allow_pickle=False),
        partial(np.save, b_path, b, allow_pickle=False),
    )


def _spin(pred, timeout=15.0):
    end = time.monotonic() + timeout
    while not pred() and time.monotonic() < end:
        time.sleep(0.005)
    return pred()


def test_success_path_is_clean(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = _start(tmp_path, make_dirs=True)
        assert isinstance(h, BackgroundWriteHandle)
        a_path, b_path = h.wait()
        assert h.wait() == (a_path, b_path)          # idempotent
    assert h.exception is None
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []
    assert np.array_equal(np.load(a_path),
                          np.arange(6, dtype=np.float32).reshape(2, 3))
    assert np.array_equal(np.load(b_path),
                          np.arange(6, dtype=np.float32).reshape(2, 3) + 10.0)


def test_every_wait_reraises_not_just_the_first(tmp_path):
    h = _start(tmp_path / "missing", make_dirs=False)
    with pytest.raises(Exception) as first:
        h.wait()
    with pytest.raises(Exception) as second:
        h.wait()
    assert second.value is first.value
    with pytest.raises(Exception):
        h.wait()
    assert h.exception is first.value


def test_failure_warns_at_failure_time_without_wait(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = _start(tmp_path / "missing", make_dirs=False)
        assert _spin(lambda: bool(caught)), "done-callback never warned"
    assert any(issubclass(w.category, RuntimeWarning)
               and "background write failed" in str(w.message) for w in caught)
    assert h.exception is not None
    with pytest.raises(Exception):                 # join so __del__ stays quiet
        h.wait()


def test_dropping_an_unjoined_failed_handle_warns(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = _start(tmp_path / "missing", make_dirs=False)
        assert _spin(lambda: h.exception is not None)
        h = None                      # drop the last reference
        gc.collect()
        assert _spin(lambda: any("dropped without wait()" in str(w.message)
                                 for w in caught))
    hits = [w for w in caught if "dropped without wait()" in str(w.message)]
    assert hits and issubclass(hits[0].category, RuntimeWarning)


def test_no_jobs_is_rejected():
    with pytest.raises(ValueError, match="no jobs"):
        submit_background_write(None, "empty")
