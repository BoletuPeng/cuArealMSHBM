"""save_parcellation.py

Step-3 output writer. Builds ``lh_labels`` / ``rh_labels`` from the
converged ``Params.s_lambda`` (argmax over parcels at active vertices)
and saves to
``Ind_parcellation_MSHBM_sub<id>_w<W>_MRF<C>_beta<B>.mat`` in MATLAB
v7 format so the existing comparison tools and the MATLAB reference
pipeline can read it.

Public API:
    derive_labels     — argmax(s_lambda) → (lh_labels, rh_labels).
    save_parcellation — derive labels and write the .mat artifact.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def derive_labels(s_lambda: np.ndarray,
                  n_lh: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax labels (1-indexed; 0 at medial wall).

    Mirrors MATLAB step3 lines 345-348.

    Parameters
    ----------
    s_lambda : (N, L) — soft posterior probabilities.
    n_lh     : LH vertex count. Defaults to ``N // 2`` — correct for the
               equal-hemisphere meshes the pipeline currently supports
               (fsaverage6: 40962+40962, fs_LR_32k: 32492+32492). Pass
               explicitly for any future asymmetric-hemisphere mesh, or
               to assert against the mesh's known LH vertex count when
               ``s_lambda`` may have come from a different mesh.

    Returns
    -------
    lh_labels, rh_labels : (n_lh,), (N - n_lh,) int64. Bilateral split.
    """
    sl = np.ascontiguousarray(s_lambda)
    N, L = sl.shape

    labels = np.zeros(N, dtype=np.int64)
    active = sl.sum(axis=1) != 0
    if active.any():
        labels[active] = sl[active].argmax(axis=1) + 1
    if n_lh is None:
        if N % 2 != 0:
            raise ValueError(
                f"derive_labels: N={N} is odd; pass n_lh explicitly for "
                f"asymmetric-hemisphere meshes"
            )
        n_lh = N // 2
    elif not (0 < n_lh < N):
        raise ValueError(
            f"derive_labels: n_lh={n_lh} out of range for N={N}"
        )
    return labels[:n_lh], labels[n_lh:]


def _fmt_num(x: float | int | str) -> str:
    """MATLAB ``num2str`` semantics for filename interpolation: integers
    render as bare digits (``100`` -> ``"100"``); non-integers keep their
    significant digits (``0.5`` -> ``"0.5"``); strings pass through
    unchanged (callers sometimes hand in pre-stringified beta).

    The earlier ``f"{int(w)}"`` form silently truncated non-integer
    values to 0 — ``int(0.5)`` is ``0``, so a ``w=0.5`` config would have
    produced a file named ``..._w0_...`` colliding with ``w=0``. ``:g``
    matches MATLAB's ``num2str`` for the integer + small-float regime
    these knobs use.
    """
    if isinstance(x, str):
        return x
    return f"{float(x):g}"


def save_parcellation(s_lambda: np.ndarray,
                      out_dir: str | Path,
                      subid: str | int,
                      w: float,
                      c: float,
                      beta: float | str | None = None,
                      n_lh: int | None = None,
                      out_path: str | Path | None = None,
                      lh_labels_override: np.ndarray | None = None,
                      rh_labels_override: np.ndarray | None = None,
                      ) -> Path:
    """Write the standard CBIG ``Ind_parcellation_*.mat`` artifact.

    Parameters
    ----------
    s_lambda : (N, L) — converged soft posterior. Used to derive
               ``lh_labels`` / ``rh_labels`` via :func:`derive_labels`,
               unless explicit override arrays are provided.
    out_dir  : target directory; created if missing. Ignored when
               ``out_path`` is given.
    subid    : subject id (string or int) — interpolated into filename.
    w, c, beta : pipeline parameters — interpolated into filename. ``w``
                 and ``c`` are required; ``beta`` may be ``None`` for the
                 dMSHBM variant which has no beta term and writes
                 ``Ind_parcellation_MSHBM_sub<i>_w<w>_MRF<c>.mat`` (no
                 ``_beta`` suffix). Each may be int / float / pre-
                 stringified; non-integers are preserved (no truncation).
                 MATLAB ``num2str`` semantics.
    n_lh     : optional LH vertex count, forwarded to :func:`derive_labels`.
    out_path : optional full destination path. When provided, ``out_dir``
               and ``w/c/beta`` are NOT used to build a filename — the
               file is written exactly to this path (parents created on
               demand). Useful for callers that want a custom filename
               (e.g. ad-hoc diff harnesses that need both candidate and
               reference outputs in the same dir). Both branches write
               IDENTICAL on-disk content; consolidating here keeps the
               two destinations bit-equal if the .mat schema ever
               changes.
    lh_labels_override / rh_labels_override : if both are provided,
               skip the :func:`derive_labels` call on ``s_lambda`` and
               write these directly. Used by cMSHBM where the labels
               undergo :func:`remove_isolated_surface_components`
               post-processing AFTER argmax.

    Returns
    -------
    out_path : the written file path.
    """
    from scipy.io import savemat

    if lh_labels_override is not None and rh_labels_override is not None:
        lh_labels = np.ascontiguousarray(lh_labels_override).ravel()
        rh_labels = np.ascontiguousarray(rh_labels_override).ravel()
    else:
        lh_labels, rh_labels = derive_labels(s_lambda, n_lh=n_lh)

    if out_path is None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if beta is None:
            fname = (
                f"Ind_parcellation_MSHBM_sub{subid}"
                f"_w{_fmt_num(w)}_MRF{_fmt_num(c)}.mat"
            )
        else:
            fname = (
                f"Ind_parcellation_MSHBM_sub{subid}"
                f"_w{_fmt_num(w)}_MRF{_fmt_num(c)}_beta{_fmt_num(beta)}.mat"
            )
        out_path = out_dir / fname
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        str(out_path),
        {"lh_labels": lh_labels.astype(np.float64).reshape(-1, 1),
         "rh_labels": rh_labels.astype(np.float64).reshape(-1, 1)},
        do_compression=False,
    )
    return out_path
