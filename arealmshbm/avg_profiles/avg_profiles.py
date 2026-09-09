"""avg_profiles.py

Supercall for averaging RSFC profiles across (subject × session).
Reads per-subject bitpacked ``.b2nd`` profiles ``(T, N, ⌈D/8⌉)`` uint8
written by step-1's generate_profiles, splits the N axis into lh/rh,
accumulates set bits into a fp32 buffer, divides by
``num_sub × num_sess``, and writes the canonical
``profiles/avg_profile/{lh,rh}_<…>_avg_profile.npy`` pair.

Output format: raw ``.npy`` ``(V_h, D)`` fp32 C-contig. A project-level
single-shot output (~192 MB / hemi at fsa6 schaefer-400) consumed only
by :mod:`arealmshbm.ini_params` — no streaming benefit from
chunked blosc2, no external MATLAB consumer in the Mode-A pipeline.

CPU path: numba kernels in :mod:`._kernels` called directly.
Dispatch on ``backend='gpu'`` defers to :mod:`.avg_profiles_gpu`
(CUDA RawKernel + cupy in-place scale). cupy is imported lazily — the
CPU path never touches it.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from arealmshbm.data_io._background_write import (
    BackgroundWriteHandle, submit_background_write,
)


@dataclass(frozen=True)
class AvgProfilesResult:
    """Result of one ``avg_profiles`` (or GPU twin) invocation.

    Carries both the on-disk artifact paths (cache) and the in-memory
    arrays — so the next subgraph (``generate_ini_params``) can consume
    the fp32 means directly instead of re-reading the .npy files.

    ``lh_avg_dev`` / ``rh_avg_dev`` are the device-resident fp32
    ``(V_h, D)`` accumulators (cupy arrays) when the memory-side GPU
    supercall (``avg_profiles_from_packed_gpu``) produced this result
    — ``generate_ini_params_gpu`` consumes them via
    ``precomputed_{lh,rh}_avg_dev`` so the fp32 means never round trip
    through host memory. They are ``None`` on the CPU path AND on the
    disk-reading ``avg_profiles_gpu`` path, which has no device-side
    consumer and drops them so the pool can reclaim ~385 MB.

    ``writer`` is the background ``.npy`` write handle (``None`` when
    ``save=False`` or the write ran synchronously); ``writer.wait()``
    returns ``(lh_path, rh_path)`` and callers MUST call it before the
    process exits.
    """
    lh_path: Path
    rh_path: Path
    lh_avg: np.ndarray   # (V_lh, D) fp32 C-contig
    rh_avg: np.ndarray   # (V_rh, D) fp32 C-contig
    lh_avg_dev: Optional[Any] = None   # (V_lh, D) fp32 cupy, GPU path only
    rh_avg_dev: Optional[Any] = None   # (V_rh, D) fp32 cupy, GPU path only
    writer: Optional[BackgroundWriteHandle] = None

from arealmshbm.data_io.load_avg_mesh import load_avg_mesh
from arealmshbm.data_io.profile_io import (
    profile_path, read_subject_profile_packed_tnd,
)
from ._kernels import (
    _accum_packed_session_inplace_kernel, _scale_inplace_kernel,
)


def _hemi_vert_counts(targ_mesh: str) -> Tuple[int, int]:
    """``(V_lh, V_rh)`` for the given fsaverage* mesh. The b2nd writer
    stacks lh + rh along the N axis, so we need both counts up front
    to slice each per-session chunk back into hemis."""
    lh = load_avg_mesh("lh", targ_mesh, "inflated")
    rh = load_avg_mesh("rh", targ_mesh, "inflated")
    return int(lh["MARS_label"].shape[0]), int(rh["MARS_label"].shape[0])


def _avg_paths(out_dir: Path,
               targ_mesh: str, seed_mesh: str) -> Tuple[Path, Path]:
    base = Path(out_dir) / "profiles" / "avg_profile"
    return (
        base / f"lh_{targ_mesh}_roi{seed_mesh}_avg_profile.npy",
        base / f"rh_{targ_mesh}_roi{seed_mesh}_avg_profile.npy",
    )


def _load_subject_packed_TND(b2nd_p: Path) -> Tuple[np.ndarray, int]:
    """Full-decode one bitpacked subject as ``(T, N, ⌈D/8⌉) uint8`` +
    the original D.

    ~8× smaller in-RAM footprint than the fp32 path; the trailing host
    ``np.unpackbits`` + cast is skipped — averaging happens directly
    over packed bits via :func:`_accum_packed_session_inplace_kernel`.
    """
    return read_subject_profile_packed_tnd(b2nd_p)


def _discover_and_decode_subjects(
    out_dir: Path,
    targ_mesh: str,
    seed_mesh: str,
    num_sub: int,
    num_sess: int,
    V_lh: int,
    V_rh: int,
    verbose: bool,
) -> Tuple[List[Tuple[int, Path]], List[np.ndarray], int, int, int, int]:
    """Discover present per-subject ``.b2nd`` files, parallel-decode
    them as packed ``(T, N, ⌈D/8⌉) uint8``, and validate cross-subject
    shape/D consistency. Both CPU and GPU supercalls consume the same
    disk-side decode pipeline; only the accumulator backend differs.

    Returns ``(sub_paths, packed_arrs, T0, N0, D0_bytes, D0)``.
    """
    sub_paths: List[Tuple[int, Path]] = []
    for sub in range(1, num_sub + 1):
        p = profile_path(out_dir, str(sub), targ_mesh, seed_mesh)
        if p.exists():
            sub_paths.append((sub, p))
        elif verbose:
            print(f"Skip: {p}")
    if not sub_paths:
        raise FileNotFoundError(
            f"avg_profiles: no per-subject .b2nd files found under "
            f"{out_dir}/profiles_raw/ for sub=1..{num_sub}."
        )

    packed_arrs: List[Optional[np.ndarray]] = [None] * len(sub_paths)
    D_vals: List[Optional[int]] = [None] * len(sub_paths)
    with ThreadPoolExecutor(max_workers=min(len(sub_paths), 8)) as pool:
        futures = [pool.submit(_load_subject_packed_TND, p)
                   for _, p in sub_paths]
        for i, fut in enumerate(futures):
            packed_arrs[i], D_vals[i] = fut.result()

    T0, N0, D0_bytes = packed_arrs[0].shape
    D0 = int(D_vals[0])
    if N0 != V_lh + V_rh:
        raise ValueError(
            f"avg_profiles: per-subject .b2nd has N={N0} but mesh "
            f"{targ_mesh!r} yields V_lh+V_rh={V_lh + V_rh}."
        )
    if T0 != num_sess:
        raise ValueError(
            f"avg_profiles: per-subject .b2nd has T={T0} but num_sess={num_sess}."
        )
    for i, arr in enumerate(packed_arrs[1:], 1):
        if arr.shape != (T0, N0, D0_bytes) or int(D_vals[i]) != D0:
            raise ValueError(
                f"avg_profiles: shape/D mismatch — sub idx {i} has "
                f"{arr.shape} D={D_vals[i]}, sub idx 0 has "
                f"{(T0, N0, D0_bytes)} D={D0}"
            )
    return sub_paths, packed_arrs, T0, N0, D0_bytes, D0


def _save_avg_npy_pair(
    out_dir: Path, targ_mesh: str, seed_mesh: str,
    lh_acc: np.ndarray, rh_acc: np.ndarray,
) -> Tuple[Path, Path]:
    """Write the ``(V_h, D)`` fp32 ``.npy`` pair in parallel. Reader
    (:mod:`arealmshbm.ini_params`) reads the same layout back via
    ``np.load``.
    """
    lh_out, rh_out = _avg_paths(out_dir, targ_mesh, seed_mesh)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_lh = pool.submit(np.save, lh_out, lh_acc, allow_pickle=False)
        f_rh = pool.submit(np.save, rh_out, rh_acc, allow_pickle=False)
        f_lh.result()
        f_rh.result()
    return lh_out, rh_out


def _start_avg_npy_pair_async(
    out_dir: Path, targ_mesh: str, seed_mesh: str,
    lh_acc: np.ndarray, rh_acc: np.ndarray,
) -> BackgroundWriteHandle:
    """Kick off the ``.npy`` pair write on two background threads and
    return immediately with a join handle (``wait()`` -> the pair).

    Byte-for-byte the same files as :func:`_save_avg_npy_pair` — the
    same ``np.save`` call, just off the caller's thread (``np.save``
    releases the GIL in ``tofile``, so the caller is not slowed). The
    caller must not mutate ``lh_acc`` / ``rh_acc`` until ``wait()``
    returns.
    """
    lh_out, rh_out = _avg_paths(out_dir, targ_mesh, seed_mesh)
    return submit_background_write(
        (lh_out, rh_out),
        f"avg_profiles .npy pair ({lh_out.name} / {rh_out.name})",
        partial(np.save, lh_out, lh_acc, allow_pickle=False),
        partial(np.save, rh_out, rh_acc, allow_pickle=False),
    )


def avg_profiles(
    seed_mesh: str,
    targ_mesh: str,
    out_dir,
    num_sub,
    num_sess,
    *,
    verbose: bool = True,
    backend: str = "cpu",
) -> AvgProfilesResult:
    """Average RSFC profiles across (subject × session).

    Pipeline: parallel per-subject .b2nd decodes → in-place voxel-wise
    accumulator (per session per subject) → divide by count → parallel
    .npy writes. fp32 throughout — every summand is exactly 0.0f or 1.0f,
    partial sums never exceed ``num_sub × num_sess`` (well under 2^24)
    so fp32 adds are integer-exact and order-independent.

    ``backend='gpu'`` dispatches to :func:`avg_profiles_gpu.avg_profiles_gpu`
    (CUDA RawKernel accumulator on device; .b2nd decode + .npy writes
    stay CPU-side).
    """
    if backend == "gpu":
        from .avg_profiles_gpu import avg_profiles_gpu
        return avg_profiles_gpu(
            seed_mesh=seed_mesh, targ_mesh=targ_mesh, out_dir=out_dir,
            num_sub=num_sub, num_sess=num_sess, verbose=verbose,
        )
    if backend != "cpu":
        raise ValueError(f"avg_profiles: unknown backend {backend!r}")
    if "fsaverage" not in targ_mesh:
        raise ValueError(
            f"avg_profiles: only fsaverage* targ_mesh supported, got {targ_mesh!r}."
        )
    num_sub = int(num_sub)
    num_sess = int(num_sess)
    out_dir = Path(out_dir)
    (out_dir / "profiles" / "avg_profile").mkdir(parents=True, exist_ok=True)

    V_lh, V_rh = _hemi_vert_counts(targ_mesh)

    # Bitpacked path:
    #   * Loader returns (T, N, ⌈D/8⌉) uint8 (~8× smaller than fp32) + D.
    #   * Accumulator is a flat (V_h * D,) fp32 buffer; the fused
    #     popcount-accumulate kernel adds 1.0 for each set bit.
    sub_paths, packed_arrs, T0, _N0, _D0_bytes, D0 = _discover_and_decode_subjects(
        out_dir, targ_mesh, seed_mesh, num_sub, num_sess, V_lh, V_rh, verbose,
    )

    lh_acc_flat = np.zeros(V_lh * D0, dtype=np.float32)
    rh_acc_flat = np.zeros(V_rh * D0, dtype=np.float32)

    n_pairs = 0
    for (sub, _p), arr in zip(sub_paths, packed_arrs):
        for t in range(T0):
            slab_NDb = arr[t]                 # (N, D_bytes) uint8 view
            lh_slab = slab_NDb[:V_lh]         # (V_lh, D_bytes) view
            rh_slab = slab_NDb[V_lh:]         # (V_rh, D_bytes) view
            _accum_packed_session_inplace_kernel(lh_acc_flat, lh_slab, D0)
            _accum_packed_session_inplace_kernel(rh_acc_flat, rh_slab, D0)
            n_pairs += 1
        if verbose:
            print(f"current subject:{sub}")

    inv_n = np.float32(1.0 / float(n_pairs))
    _scale_inplace_kernel(lh_acc_flat, inv_n)
    _scale_inplace_kernel(rh_acc_flat, inv_n)

    lh_acc = lh_acc_flat.reshape(V_lh, D0)
    rh_acc = rh_acc_flat.reshape(V_rh, D0)
    lh_out, rh_out = _save_avg_npy_pair(
        out_dir, targ_mesh, seed_mesh, lh_acc, rh_acc,
    )
    return AvgProfilesResult(
        lh_path=lh_out, rh_path=rh_out,
        lh_avg=lh_acc, rh_avg=rh_acc,
    )
