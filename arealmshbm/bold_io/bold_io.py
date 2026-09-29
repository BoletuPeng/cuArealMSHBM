"""bold_io.py

Read a single-hemi surface BOLD ``.func.gii`` and reshape it to the
``(N, T)`` matrix the rest of the step-0 pipeline expects.

The hemi concatenation + NaN→0 + medial-mask drop is the inner content
of the MATLAB scan loop in ``CBIG_SPGrad_RSFC_gradients.m``. We
reproduce its order:

    1. Read lh_curr_data, replace NaN with 0.
    2. Read rh_curr_data, replace NaN with 0.
    3. Vstack [lh; rh].
    4. Delete rows where ``medial_mask == 1``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from arealmshbm.data_io.gifti_io import read_surface_gifti

PathLike = Union[str, Path]


def read_surface_bold(path: PathLike,
                      expected_n: Optional[int] = None) -> np.ndarray:
    """Read a single-hemi surface BOLD ``.func.gii`` and return ``(N, T)`` fp32.

    Thin wrapper around
    :func:`arealmshbm.data_io.gifti_io.read_surface_gifti`. Adds an
    ``expected_n`` guard so a GIFTI writer that
    changed the vertex count is caught at read time rather than
    producing shuffled-but-correctly-sized data downstream.

    Parameters
    ----------
    path : str | Path
        Path to a ``.func.gii`` surface BOLD file with the BOLD
        contract (FLOAT32 / GZipBase64Binary / LittleEndian).
    expected_n : int, optional
        If given, refuse the file when its ``Dim0`` (vertex count)
        disagrees with this number. Default is no check.

    Returns
    -------
    np.ndarray (N, T) fp32
        Vertex-by-time BOLD matrix. ``N`` = ``Dim0`` of the first
        ``<DataArray>``; ``T`` = number of darrays.

    Notes
    -----
    NaN values are NOT replaced here — that is the caller's job
    (matches the MATLAB order: NaN→0 happens after the
    reshape, in the scan loop in
    :func:`concat_hemis_drop_medial`).
    """
    p = Path(path)
    if p.suffix.lower() != ".gii":
        raise ValueError(
            f"read_surface_bold: only ``.gii`` surface BOLD is "
            f"supported (typically ``.func.gii``); got {p}."
        )
    vol = read_surface_gifti(p)
    if expected_n is not None and vol.shape[0] != expected_n:
        raise ValueError(
            f"read_surface_bold: GIFTI {p} has N={vol.shape[0]} "
            f"!= expected_n={expected_n}"
        )
    return vol


def concat_hemis_drop_medial(
    lh: np.ndarray,
    rh: np.ndarray,
    medial_mask: np.ndarray,
) -> np.ndarray:
    """Concatenate lh + rh BOLD, replace NaN with 0, drop medial rows.

    Mirrors the MATLAB sequence in ``CBIG_SPGrad_RSFC_gradients.m``:
        curr_data(isnan(curr_data)) = 0;          % per hemi
        rh_curr_data(isnan(rh_curr_data)) = 0;
        curr_data = [curr_data; rh_curr_data];
        curr_data(medial_mask, :) = [];

    One difference: ``np.nan_to_num`` also clamps ``+inf`` / ``-inf`` to
    the largest / smallest finite fp32, where the MATLAB lines leave an
    infinity in place.

    Parameters
    ----------
    lh, rh : (N_hemi, T) fp32
        Per-hemi BOLD as returned by ``read_surface_bold``.
        **Mutated in place when input is already fp32.** ``np.asarray
        (arr, dtype=np.float32)`` returns a view of the caller's
        buffer when the dtype already matches, and the subsequent
        ``np.nan_to_num(copy=False)`` writes through that view. The
        prefetcher / step-0 worker passes fresh per-session buffers
        so this is safe today; callers reusing buffers across calls
        must clone first.
    medial_mask : (2 * N_hemi,) array-like, truthy=medial
        1 means medial wall (drop), 0 means cortex (keep). Accepts uint8,
        bool, int — anything truthy. Reshape-safe: ``(2*N_hemi, 1)`` is
        squeezed to 1-D.

    Returns
    -------
    np.ndarray (N_cortex, T) fp32
        Concatenated, NaN-cleaned, medial-dropped BOLD ready for the
        FC-similarity stage.
    """
    if lh.ndim != 2 or rh.ndim != 2:
        raise ValueError(
            f"lh and rh must be 2-D (N, T); got lh.shape={lh.shape}, rh.shape={rh.shape}"
        )
    if lh.shape[1] != rh.shape[1]:
        raise ValueError(
            f"lh and rh must have matching T; got {lh.shape[1]} vs {rh.shape[1]}"
        )
    lh32 = np.asarray(lh, dtype=np.float32)
    rh32 = np.asarray(rh, dtype=np.float32)
    # Per-hemi NaN→0, matching MATLAB ordering.
    np.nan_to_num(lh32, copy=False, nan=0.0)
    np.nan_to_num(rh32, copy=False, nan=0.0)
    full = np.vstack([lh32, rh32])

    mask = np.asarray(medial_mask).reshape(-1).astype(bool)
    if mask.shape[0] != full.shape[0]:
        raise ValueError(
            f"medial_mask length {mask.shape[0]} != concatenated rows {full.shape[0]}"
        )
    keep = ~mask
    return full[keep, :]
