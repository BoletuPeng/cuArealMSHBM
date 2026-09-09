"""bold_io.py

Read a single-hemi surface BOLD ``.func.gii`` and reshape it to the
``(N, T)`` matrix the rest of the step-0 pipeline expects.

Format history
--------------
The pipeline originally consumed a "fake 4-D NIFTI" mirror produced
by CBIG's MATLAB driver
(``CBIG_save_data_to_surface_mat.m → MRIwrite``) — a ``(I, J, K, T)``
volume where ``I*J*K = N_hemi``, decompressed with isal igzip and
Fortran-reshaped to ``(N, T)``. That mirror is gone — the offline
converter and its e2e consistency harness were retired alongside the
NIFTI reader (the project's git history before the strip carries
``profile_nifti_io.py`` + ``convert_ys_bold_parallel.py``). The
pipeline now reads the source GIFTI directly via
:func:`arealmshbm.data_io.gifti_io.read_surface_gifti`. End-to-end
bit-equality between the direct-GIFTI route and the converted-NIFTI
route was verified at strip time on real YS sub-001 data (with CPU
step0 to bypass GPU eigsh non-determinism); every numerical artifact
downstream — profile.b2nd, avg_profile_{lh,rh}.npy,
ind_parcellation_*.mat, gradients/* — matched bit-for-bit. The
verification result table lives in the PR #54 description and in
``docs/v2_schema_apple_to_apple_verification.md`` for the v2-schema
refactor that preceded the strip.

The hemi concatenation + NaN→0 + medial-mask drop is the inner content
of the MATLAB scan loop in ``CBIG_SPGrad_RSFC_gradients.m`` lines
264–285. We faithfully reproduce its order:

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
    :func:`arealmshbm.data_io.gifti_io.read_surface_gifti`. Keeps the
    historical ``expected_n`` guard so a future GIFTI writer that
    changed the vertex count is caught at read time rather than
    producing shuffled-but-correctly-sized data downstream.

    Parameters
    ----------
    path : str | Path
        Path to a ``.func.gii`` surface BOLD file with the BOLD
        contract (FLOAT32 / GZipBase64Binary / LittleEndian). NIFTI
        ``.nii.gz`` inputs are no longer supported — convert by
        re-running DeepPrep, or fall back to a git revision before
        the NIFTI strip.
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
    (matches the historical MATLAB order: NaN→0 happens after the
    reshape, in the scan loop in
    :func:`concat_hemis_drop_medial`).
    """
    p = Path(path)
    if p.suffix.lower() != ".gii":
        raise ValueError(
            f"read_surface_bold: only ``.gii`` surface BOLD is "
            f"supported (typically ``.func.gii``); got {p}. The "
            f"historical NIFTI mirror path was removed after the "
            f"GIFTI direct-read landed."
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
