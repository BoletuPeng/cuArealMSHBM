"""_loaders.py — in-memory stand-ins for the step-2 subject loaders.

Wrap a full ``(S, N, T, D)`` / ``(S, T, N, D_grad)`` array in RAM as a
streaming loader, so the session tests can build synthetic data without
going through disk.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class InMemoryProfileLoader:
    """Adapter that wraps an in-memory ``(S, N, T, D)`` fp32 array as a
    :class:`SubjectProfileLoader`. ``load_into(s, out)`` does one
    ``np.copyto`` from ``arr[s-1]`` into ``out``.
    """

    def __init__(self, bold_SNTD: np.ndarray, num_session: int):
        if bold_SNTD.ndim != 4:
            raise ValueError(
                f"bold_SNTD must be 4D (S, N, T, D); got {bold_SNTD.shape}")
        if bold_SNTD.dtype != np.float32:
            raise ValueError(
                f"bold_SNTD must be fp32; got {bold_SNTD.dtype}")
        self._arr = bold_SNTD
        self.num_sub = int(bold_SNTD.shape[0])
        self.num_session = int(num_session)
        if self.num_session != int(bold_SNTD.shape[2]):
            raise ValueError(
                f"num_session {self.num_session} != bold_SNTD.shape[2] "
                f"{bold_SNTD.shape[2]}")
        self.format = "in_memory"

    def dims(self) -> Tuple[int, int, int]:
        S, N, T, D = self._arr.shape
        return (int(N), int(T), int(D))

    def load(self, s: int) -> np.ndarray:
        return np.ascontiguousarray(self._arr[s - 1])

    def load_into(self, s: int, out: np.ndarray) -> None:
        np.copyto(out, self._arr[s - 1])


class InMemoryGradientLoader:
    """Adapter that wraps an in-memory ``(S, T, N, D_grad)`` fp32 array
    as a :class:`SubjectGradientLoader`. Same contract — one
    ``np.copyto`` per ``load_into`` call.
    """

    def __init__(self, grad_STND: np.ndarray, num_session: int):
        if grad_STND.ndim != 4:
            raise ValueError(
                f"grad_STND must be 4D (S, T, N, D_grad); got {grad_STND.shape}")
        if grad_STND.dtype != np.float32:
            raise ValueError(
                f"grad_STND must be fp32; got {grad_STND.dtype}")
        self._arr = grad_STND
        self.num_sub = int(grad_STND.shape[0])
        self.num_session = int(num_session)
        if self.num_session != int(grad_STND.shape[1]):
            raise ValueError(
                f"num_session {self.num_session} != grad_STND.shape[1] "
                f"{grad_STND.shape[1]}")
        self.format = "in_memory"

    def dims(self) -> Tuple[int, int]:
        S, T, N, D_grad = self._arr.shape
        return (int(N), int(D_grad))

    def load(self, s: int) -> np.ndarray:
        return np.ascontiguousarray(self._arr[s - 1])

    def load_into(self, s: int, out: np.ndarray) -> None:
        np.copyto(out, self._arr[s - 1])
