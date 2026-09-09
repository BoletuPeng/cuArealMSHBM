"""step2_runners.py — step-2 driver-side helpers.

Step 2 has a single super-call (:class:`arealmshbm.step2_pipeline.Step2Pipeline`)
rather than the four subgraphs step 1 exposes, so this module carries
only the piece the driver needs *around* that call: the GPU prewarm.

GPU prewarm
-----------
The ``gpu`` backend compiles one ``cupy.RawModule``
(``step2_em_iter_master/_kernels_gpu.py``) holding every step-2
kernel. That NVRTC compile is process-global and one-shot, and until
this module existed nothing paid it off the timed path — the first
``run_iter`` of a fresh process paid it inside ``em_total``.

:func:`prewarm_step2_gpu` pays it on a daemon thread while step 0 runs
(seconds of cover on any real cohort). It compiles the module and runs
one 32x32 gemm, which pays cuBLAS's process-wide kernel-module load that
the iteration-1 dense pass (K7) would otherwise charge to the first
``run_iter``. cupy's cuBLAS handle is per thread, so that gemm cannot
disturb the handle another thread is using.

It allocates **no pinned staging slots**: those are deliberately left to
the Session ctor, because the driver unconditionally drains both cupy
pools after step 0 (``driver.py``, right after ``Step0StagePipeline.run``),
so any pinned block allocated here would be handed straight back to the
OS before step 2 ever starts.

Idempotent, swallow-all, daemon — same shape as
``step1_runners.prewarm_step1_gpu``: a prewarm failure only means the
run pays the compile itself, so it must never surface as an error, and
a second call joins the in-flight thread instead of compiling twice.
Join it with :func:`join_step2_prewarm` before the process exits, or a
still-running NVRTC compile can outlive cupy's teardown.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import logging
import threading
from typing import Optional


_PREWARM_THREAD: Optional[threading.Thread] = None
_PREWARM_LOCK = threading.Lock()
_log = logging.getLogger(__name__)


def prewarm_step2_gpu(*, background: bool = True) -> Optional[threading.Thread]:
    """Compile the step-2 GPU RawModule off the timed path.

    Returns the daemon thread when backgrounded (``None`` when run
    inline). Never raises.
    """
    global _PREWARM_THREAD

    # A fresh thread defaults to device 0, so the prewarm must be
    # pinned to the device the CALLER is on or it warms the wrong one.
    dev_id = None
    try:
        import cupy as cp
        dev_id = int(cp.cuda.runtime.getDevice())
    except Exception:      # noqa: BLE001 — no cupy / no device
        _log.debug("step2 prewarm: no CUDA device to pin", exc_info=True)

    def _work() -> None:
        try:
            if dev_id is not None:
                import cupy as cp
                cp.cuda.Device(dev_id).use()
            from arealmshbm.step2_em_iter_master import warmup_step2_gpu
            warmup_step2_gpu()
        except Exception:      # noqa: BLE001 — warm-up is best effort
            _log.debug("step2 GPU prewarm failed; the run will pay the "
                       "compile itself", exc_info=True)

    if not background:
        _work()
        return None
    with _PREWARM_LOCK:
        th = _PREWARM_THREAD
        if th is not None and th.is_alive():
            return th
        th = threading.Thread(target=_work, name="step2-prewarm", daemon=True)
        _PREWARM_THREAD = th
        th.start()
        return th


def join_step2_prewarm(timeout: Optional[float] = 30.0) -> None:
    """Join the background thread :func:`prewarm_step2_gpu` started.

    No-op when none is running. A thread still alive after ``timeout``
    is left alone (it is a daemon); that is logged at DEBUG.
    """
    with _PREWARM_LOCK:
        th = _PREWARM_THREAD
    if th is None or not th.is_alive():
        return
    th.join(timeout)
    if th.is_alive():
        _log.debug("step2 GPU prewarm still running after %.1fs; leaving it "
                   "to the daemon-thread teardown", timeout)
