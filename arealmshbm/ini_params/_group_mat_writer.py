"""_group_mat_writer.py

``group.mat`` write plumbing shared by the CPU and GPU ini-params
supercalls: the result container that carries the (optional) background
write handle, and the writer itself.

Why a handle at all: nearly all of the ``savemat`` cost is the dense
``(N_keep, L)`` uint8 ``lambda`` going through zlib. ``lambda`` is
written purely for MATLAB-format parity — :mod:`arealmshbm.step2_io`'s
``load_group_mtc``, the only Python reader of this file, never touches
it — so it must stay, but there is no reason for the caller's critical
path to wait on it. Handing the write to a background thread removes
it from the step-1 wall; the caller joins with ``result.writer.wait()``.

The supercalls always write compressed (the historical on-disk form);
``compress`` stays a parameter of this writer because both settings
round-trip identically through ``scipy.io.loadmat`` / ``load_group_mtc``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Optional

import scipy.io as sio

from ..data_io._background_write import (
    BackgroundWriteHandle, submit_background_write,
)


class IniParamsResult(dict):
    """``group.mat`` field dict, plus the optional async write handle.

    Subclasses ``dict`` so every existing consumer (``out["mtc"]``,
    ``sio.savemat(out)``, ``**out``) keeps working unchanged; the
    ``writer`` attribute is ``None`` unless the supercall was asked to
    save asynchronously.
    """

    writer: Optional[BackgroundWriteHandle] = None


def _savemat_group(out_path: str, payload: dict, compress: bool) -> None:
    """``sio.savemat`` with the project's pinned format options."""
    sio.savemat(out_path, payload, do_compression=compress, format="5")


def write_group_mat(out_dir: str, payload: dict, *,
                    compress: bool = True,
                    background: bool = False
                    ) -> Optional[BackgroundWriteHandle]:
    """Write ``<out_dir>/group/group.mat``.

    ``background=False`` (default) writes synchronously and returns
    ``None``. ``background=True`` submits the write to a private thread
    and returns a :class:`BackgroundWriteHandle` whose ``wait()``
    returns the path — the caller MUST call it before the process
    exits. The payload is snapshotted (shallow ``dict`` copy) so the
    writer never observes a later mutation of the caller's result.
    """
    out_path = os.path.join(out_dir, "group", "group.mat")
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    snapshot = dict(payload)
    if not background:
        _savemat_group(out_path, snapshot, compress)
        return None
    return submit_background_write(
        Path(out_path), f"group.mat write to {out_path}",
        partial(_savemat_group, out_path, snapshot, compress),
    )


__all__ = ["IniParamsResult", "write_group_mat"]
