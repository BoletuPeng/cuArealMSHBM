"""fetch_data.py

Step-3 data ingest for one subject (Mode A, fsaverage path). Mode A
always uses fsaverage6 RSFC profiles + diffusion-embedding gradients;
the fs_LR_32k branch is not ported.

Discovery contract: every input path comes from ``cohort.json`` at
``project_dir/cohort.json`` (written by step1). The ``subid`` argument
is a **1-based position** into the cohort's ``subjects`` list (matching
the rest of the step-3 contract).

Input artifacts (resolved against project_dir via cohort.json):
    * ``subjects[subid-1].profile_b2nd`` — bitpacked per-subject .b2nd
      (one file, all T sessions).
    * ``subjects[subid-1].gradient_lh`` / ``gradient_rh`` — per-hemi
      diffusion-embedding (``.npy`` from step0, or legacy ``.mat`` from
      MATLAB step0; reader sniffs the suffix).

Output shape contract:
    data["series"]       : (N, T, ⌈D/8⌉) uint8 — bit-packed bilateral
                           BOLD profile, T sessions, D features. Each
                           consumer (CPU or GPU Session) runs its own
                           unpack + demean + L2-norm step on this
                           buffer; the on-disk packed format flows
                           through this layer untouched (modulo MW
                           zeroing for algorithmic parity with MATLAB).
    data["D_unpacked"]   : int — original cell count along axis -1.
    data["gradient_mat"] : (N, D_grad) fp32 — diffusion embedding.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _read_b2nd_series_packed(b2nd_path: Path,
                              T: int, N: int,
                              lh_mw: np.ndarray, rh_mw: np.ndarray,
                              ) -> Tuple[np.ndarray, int]:
    """Decode a per-subject .b2nd into ``(N, T, ⌈D/8⌉)`` uint8.

    Reads the on-disk packed bytes directly — no fp32 round-trip,
    verifies shape, reorders ``(T, N, D_bytes)`` (on-disk) ->
    ``(N, T, D_bytes)`` (the layout the GPU Session H2Ds), then
    zeros MW rows.

    The MW-zero pass mirrors MATLAB's CBIG_ArealMSHBM step-3 reader
    (``series(medial_mask, :) = 0;`` right after ``CBIG_MSHBM_read_fmri``,
    before demean+normalize) — it is the algorithm's MW-zero-before-
    normalize step, not a defensive guard. Bitpacked .b2nd files
    happen to also be zero at MW on disk because the step1 binarize
    writer zeros MW on the fp32 source before packing, so this pass
    is a no-op in practice, but
    the algorithm convention is stated at the reader for parity with
    MATLAB and so the downstream's "MW is zero" assumption is
    explicit, not implicit.

    Returns ``(packed_NTD_bytes, D_unpacked)``. Raises ``ValueError``
    on shape mismatch.
    """
    from arealmshbm.data_io.profile_io import (
        read_subject_profile_packed_tnd,
    )

    packed_TND_bytes, D = read_subject_profile_packed_tnd(b2nd_path)
    T_disk, N_disk, D_bytes = packed_TND_bytes.shape
    if T_disk != T:
        raise ValueError(
            f"fetch_data: bitpacked .b2nd at {b2nd_path} has T={T_disk}; "
            f"num_session={T} mismatch"
        )
    if N_disk != N:
        raise ValueError(
            f"fetch_data: bitpacked .b2nd at {b2nd_path} has N={N_disk}; "
            f"expected n_lh+n_rh={N}"
        )
    expected_bytes = (int(D) + 7) // 8
    if D_bytes != expected_bytes:
        raise ValueError(
            f"fetch_data: bitpacked .b2nd at {b2nd_path} has D_bytes="
            f"{D_bytes}; expected ceil(D/8)={expected_bytes} for "
            f"D_unpacked={D}"
        )
    # (T, N, D_bytes) -> (N, T, D_bytes). C-contig transpose alloc.
    out = np.ascontiguousarray(np.transpose(packed_TND_bytes, (1, 0, 2)))
    # Algorithmic MW-zero pass (mirrors MATLAB step3 line 509/539).
    n_lh = int(lh_mw.size)
    if lh_mw.any():
        out[:n_lh][lh_mw, :, :] = 0
    if rh_mw.any():
        out[n_lh:][rh_mw, :, :] = 0
    return out, int(D)


def _read_gradient_emb(path: str | Path, n_components: int = 100) -> np.ndarray:
    """Load a CBIG diffusion-embedding gradient ``.mat`` or ``.npy`` →
    ``(N_hemi, n_components)`` fp32 C-contig.

    Auto-detects format from the suffix:

      * ``.npy`` — raw fp32 ndarray written by step0. ~6× faster to
        load than the MATLAB struct path.
      * ``.mat`` — legacy CBIG file with an ``emb`` field. Both v5/v7
        (``scipy.io.loadmat``) and v7.3 (``h5py``) are supported.

    The cohort.json roster records the exact path written by step0 /
    backfilled by :func:`backfill_cohort_json` — there is no runtime
    sibling-suffix probing.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if not p.exists():
        raise FileNotFoundError(f"gradient file not found: {p}")

    if suffix == ".npy":
        emb = np.load(p)
        if emb.ndim != 2:
            raise ValueError(
                f"gradient .npy must be 2-D; got shape {emb.shape} from {p}"
            )
    else:
        try:
            from scipy.io import loadmat
            m = loadmat(p, squeeze_me=False)
            emb = np.asarray(m["emb"])
        except (NotImplementedError, ValueError):
            import h5py
            with h5py.File(p, "r") as f:
                # h5py is transposed wrt MATLAB.
                emb = np.asarray(f["emb"]).T
    if emb.shape[1] < n_components:
        raise ValueError(
            f"gradient emb has {emb.shape[1]} cols; need >= {n_components} "
            f"(from {p})"
        )
    return np.ascontiguousarray(emb[:, :n_components], dtype=np.float32)


def fetch_data(project_dir: str | Path,
               num_session: int,
               subid: int,
               mesh: str,
               lh_mesh: Dict[str, np.ndarray],
               rh_mesh: Dict[str, np.ndarray],
               n_grad_components: int = 100,
               with_gradient: bool = True,
               *,
               precomputed_gradient_mat: Optional[np.ndarray] = None,
               ) -> Dict[str, Any]:
    """Single-subject fsaverage data ingest.

    Discovery: reads ``cohort.json`` at ``project_dir/cohort.json`` and
    looks up ``subjects[subid - 1]`` (1-based positional lookup, matching
    the rest of step-3's contract). All file paths come from the
    manifest — no probing, no .txt manifests.

    BOLD is always returned bit-packed ``(N, T, ⌈D/8⌉) uint8`` plus
    ``D_unpacked``. Each consumer (CPU / gpu_elambda / gpu_full
    :class:`VmfClusteringSession`) runs its own unpack + demean +
    L2-norm step on this buffer — host-side via
    :func:`arealmshbm.data_io.bitpacked_norm.unpack_normalize_packed_NTD_host`
    for CPU paths, on-device via the fused CUDA kernel for gpu_full.

    Parameters
    ----------
    project_dir       : project root containing ``cohort.json``.
    num_session       : T — session count for this subject. Validated
                        against the cohort manifest.
    subid             : 1-based positional index into the cohort's
                        ``subjects`` list.
    mesh              : 'fsaverage6' / 'fsaverage5' / 'fsaverage'.
                        Validated against ``cohort.mesh.targ``.
    lh_mesh, rh_mesh  : output of :func:`load_avg_mesh.load_avg_mesh`
                        (only ``MARS_label`` and vertex count are read).
    n_grad_components : keep the first N gradient cols. Default 100.
    with_gradient     : if True (default, gMSHBM), read the LH+RH
                        diffusion-embedding from the cohort and emit
                        ``gradient_mat``. If False (cMSHBM / dMSHBM),
                        the cohort's gradient fields are not read and
                        ``gradient_mat`` is omitted from the output.
    precomputed_gradient_mat
                        Optional in-memory ``(N, n_grad_components)``
                        fp32 array. When set AND ``with_gradient=True``,
                        the gradient_lh/rh disk reads are skipped and
                        this array is used verbatim — the in-memory
                        pass from step0's output to step3's input.
                        Shape and column count are validated. Cohort
                        gradient paths still exist on disk as cache;
                        this just skips re-reading them.

    Returns
    -------
    data : dict
        ``series``        — (N, T, ⌈D/8⌉) uint8 bit-packed BOLD.
        ``D_unpacked``    — int, original cell count along axis -1.
        ``gradient_mat``  — (N, n_grad_components) fp32, only when
                            ``with_gradient=True``.
    """
    if not (mesh.startswith("fsaverage")):
        raise ValueError(
            f"fetch_data is single-subject fsaverage only; got mesh={mesh!r}. "
            "fs_LR_32k support is left for a future port."
        )
    project_dir = Path(project_dir)
    T = int(num_session)

    n_lh = int(lh_mesh["MARS_label"].shape[0])
    n_rh = int(rh_mesh["MARS_label"].shape[0])
    N = n_lh + n_rh
    lh_mw = (lh_mesh["MARS_label"] == 1)
    rh_mw = (rh_mesh["MARS_label"] == 1)

    # Resolve all paths from cohort.json — the single source of truth.
    from arealmshbm.data_io.cohort import read_cohort, resolve_path
    cohort = read_cohort(project_dir)
    if cohort.mesh.get("targ") != mesh:
        raise ValueError(
            f"fetch_data: cohort mesh.targ={cohort.mesh.get('targ')!r} "
            f"!= caller mesh={mesh!r}"
        )
    idx = int(subid) - 1
    if idx < 0 or idx >= cohort.num_sub:
        raise IndexError(
            f"fetch_data: subid={subid} out of range for cohort "
            f"(num_sub={cohort.num_sub})"
        )
    sub_entry = cohort.subjects[idx]
    if len(sub_entry.sessions) != T:
        raise ValueError(
            f"fetch_data: cohort subjects[{idx}].sessions has "
            f"len={len(sub_entry.sessions)} != num_session={T}"
        )
    if sub_entry.profile_b2nd is None:
        raise ValueError(
            f"fetch_data: cohort subjects[{idx}] (id={sub_entry.id!r}) "
            f"has no profile_b2nd path. Run step1's generate_profiles."
        )
    b2nd_path = resolve_path(project_dir, sub_entry.profile_b2nd)
    if not b2nd_path.exists():
        raise FileNotFoundError(
            f"fetch_data: cohort subjects[{idx}] profile_b2nd missing "
            f"on disk: {b2nd_path}"
        )

    # If a precomputed gradient was provided in-memory, validate it and
    # skip the disk-read planning entirely. The cohort gradient paths
    # are still cache on disk (step0 writes them), but the driver's
    # in-memory pass means we don't re-read for this run.
    use_precomputed_gradient = (
        with_gradient and precomputed_gradient_mat is not None
    )
    lh_grad_path: Optional[Path] = None
    rh_grad_path: Optional[Path] = None
    if with_gradient and not use_precomputed_gradient:
        if sub_entry.gradient_lh is None or sub_entry.gradient_rh is None:
            raise ValueError(
                f"fetch_data: with_gradient=True but cohort "
                f"subjects[{idx}] (id={sub_entry.id!r}) has no "
                f"gradient_lh/gradient_rh path."
            )
        lh_grad_path = resolve_path(project_dir, sub_entry.gradient_lh)
        rh_grad_path = resolve_path(project_dir, sub_entry.gradient_rh)
    elif use_precomputed_gradient:
        g = precomputed_gradient_mat
        if g.ndim != 2 or g.shape[0] != N:
            raise ValueError(
                f"fetch_data: precomputed_gradient_mat shape {g.shape} "
                f"incompatible with N={N} (expected (N, n_grad_components))"
            )
        if g.shape[1] < n_grad_components:
            raise ValueError(
                f"fetch_data: precomputed_gradient_mat has "
                f"{g.shape[1]} cols; need >= {n_grad_components}"
            )

    gradient_mat: Optional[np.ndarray] = None

    # BOLD: always packed; consumer-side unpack+normalize. Gradient
    # reads run concurrently on a 2-worker pool when both are needed.
    if use_precomputed_gradient:
        packed_NTD, D_unpacked = _read_b2nd_series_packed(
            b2nd_path, T, N, lh_mw, rh_mw,
        )
        gradient_mat = np.ascontiguousarray(
            precomputed_gradient_mat[:, :n_grad_components],
            dtype=np.float32,
        )
    elif with_gradient:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_lh = ex.submit(_read_gradient_emb, lh_grad_path,
                              n_grad_components)
            f_rh = ex.submit(_read_gradient_emb, rh_grad_path,
                              n_grad_components)
            packed_NTD, D_unpacked = _read_b2nd_series_packed(
                b2nd_path, T, N, lh_mw, rh_mw,
            )
            lh_emb = f_lh.result()
            rh_emb = f_rh.result()
            gradient_mat = np.concatenate([lh_emb, rh_emb], axis=0)
    else:
        packed_NTD, D_unpacked = _read_b2nd_series_packed(
            b2nd_path, T, N, lh_mw, rh_mw,
        )

    out: Dict[str, Any] = {
        "series": packed_NTD,
        "D_unpacked": D_unpacked,
    }
    if gradient_mat is not None:
        out["gradient_mat"] = gradient_mat
    return out
