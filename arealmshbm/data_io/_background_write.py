"""_background_write.py

The one join handle behind every step-1 artifact that is written on a
background thread: the avg-profile ``.npy`` pair
(:mod:`arealmshbm.avg_profiles`) and ``group.mat``
(:mod:`arealmshbm.ini_params`). Both writes are pure disk I/O with the
GIL released, so taking them off the caller's critical path is free;
what is not free is losing an error, which is what this handle exists
to prevent.

Failure semantics
-----------------
* A writer-thread exception is latched on the handle by a
  done-callback the moment it happens, and a ``RuntimeWarning`` is
  emitted right then -- so a caller that never joins is not left with
  a silently missing file.
* :meth:`BackgroundWriteHandle.wait` re-raises that exception on
  EVERY call, not only the first: a late joiner cannot mistake a failed
  write for a finished one. The exception ``wait()`` itself raised
  overwrites whatever a done-callback latched, so every call raises the
  same object — with two jobs failing at once the callbacks may
  otherwise latch the *other* job's exception.
* Dropping a handle whose write failed without ever calling
  :meth:`~BackgroundWriteHandle.wait` warns again from the finalizer.

The exception is latched with a plain attribute store, never under the
handle's join lock: ``wait()`` holds that lock across
``pool.shutdown(wait=True)``, and a callback that tried to take it would
deadlock the writer thread against the joiner.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import threading
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


def _note_failure(ref, label: str, fut) -> None:
    """Done-callback: latch a writer-thread exception + warn at once."""
    try:
        exc = fut.exception()
    except BaseException:            # cancelled / interpreter teardown
        return
    if exc is None:
        return
    handle = ref()
    if handle is not None and handle._exc is None:
        handle._exc = exc
    warnings.warn(
        f"{label}: background write failed in the writer thread: {exc!r}",
        RuntimeWarning, stacklevel=2,
    )


class BackgroundWriteHandle(Generic[T]):
    """Join handle for one or more writes running on a private pool.

    ``result`` is what :meth:`wait` returns once every write has
    landed -- the path(s) of the file(s) written. The arrays handed to
    the writer jobs MUST NOT be mutated until :meth:`wait` returns.
    """

    __slots__ = ("result", "label", "_pool", "_futs", "_lock", "_joined",
                 "_exc", "__weakref__")

    def __init__(self, result: T, pool: ThreadPoolExecutor, futs, label: str):
        self.result = result
        self.label = label
        self._pool = pool
        self._futs = list(futs)
        self._lock = threading.Lock()
        self._joined = False
        self._exc: Optional[BaseException] = None
        # weakref: handle -> future -> callback -> handle must not be a
        # reference cycle, or __del__ would only fire at a GC pass.
        ref = weakref.ref(self)
        for f in self._futs:
            f.add_done_callback(partial(_note_failure, ref, label))

    def wait(self) -> T:
        """Block until every write is done; return :attr:`result`.

        Idempotent on success. A writer-thread exception is re-raised
        by every call (see the module docstring).
        """
        with self._lock:
            if not self._joined:
                try:
                    for f in self._futs:
                        f.result()
                except BaseException as exc:
                    # Unconditional: a done-callback may already have
                    # latched a DIFFERENT job's exception, and the one
                    # a later wait() re-raises must be the one this
                    # call raised.
                    self._exc = exc
                    raise
                finally:
                    self._pool.shutdown(wait=True)
                    self._joined = True
            elif self._exc is not None:
                raise self._exc
        return self.result

    @property
    def exception(self) -> Optional[BaseException]:
        """The writer-thread exception, or ``None``. Set by the
        done-callback as soon as a write fails -- no ``wait()`` needed."""
        return self._exc

    def __del__(self):
        # Last-chance notice: the write failed and nobody ever joined.
        try:
            if self._exc is not None and not self._joined:
                warnings.warn(
                    f"{self.label}: handle dropped without wait(); its "
                    f"background write had failed: {self._exc!r}",
                    RuntimeWarning,
                )
        except Exception:            # interpreter teardown / partial init
            pass


def submit_background_write(result: T, label: str,
                            *jobs: Callable[[], object]
                            ) -> BackgroundWriteHandle[T]:
    """Run ``jobs`` (zero-arg callables, one thread each) on a private
    pool and return the join handle immediately."""
    if not jobs:
        raise ValueError("submit_background_write: no jobs")
    pool = ThreadPoolExecutor(max_workers=len(jobs),
                              thread_name_prefix="bgwrite")
    try:
        futs = [pool.submit(job) for job in jobs]
    except BaseException:
        pool.shutdown(wait=False)
        raise
    return BackgroundWriteHandle(result, pool, futs, label)


__all__ = ["BackgroundWriteHandle", "submit_background_write"]
