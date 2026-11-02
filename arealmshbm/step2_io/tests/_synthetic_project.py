"""_synthetic_project.py — a complete synthetic step-2 project on disk.

``make_project`` writes everything ``load_step2_sparse_inputs`` reads —
per-subject bit-packed ``.b2nd`` BOLD, per-subject gradient embeddings,
``group/group.mat``, ``spatial_mask/spatial_mask_<mesh>.mat`` and
``cohort.json`` — for a cohort small enough to drive the CPU master in a
second; ``load_project`` loads it back with ``load_avg_mesh`` standing
in as the tiny fake mesh the project was written for (only
``MARS_label`` is read). Every array is drawn from ``seed`` unless the
caller supplies it, so a test that needs its own bits, mask, ``mtc`` or
gradients writes exactly those and still gets what the loader builds.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from arealmshbm.data_io.cohort import backfill_cohort_json
from arealmshbm.data_io.profile_io import profile_path, write_subject_profile_tnd
from arealmshbm.step2_io import load_step2_sparse_inputs
from arealmshbm.step2_pipeline.config import Step2Config

TARG, SEED = "fsaverage6", "fsaverage3"


def make_project(root: Path, *, S: int = 2, T: int = 2, n_h: int = 9,
                 L_h: int = 3, D: int = 11, n_grad: int = 4,
                 grad_fmt: str = "npy", mw: Sequence[int] = (0, 4),
                 break_mw: bool = False, seed: int = 0,
                 bold: Optional[np.ndarray] = None,
                 boundary: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 mtc: Optional[np.ndarray] = None,
                 grad: Optional[np.ndarray] = None):
    """Write a complete synthetic step-2 project under ``root``; return
    ``(cfg, refs)``.

    Generated from ``seed`` unless given: ``bold`` ``(S, T, N, D)`` in
    {0, 1} (a generated cohort zeroes its medial-wall rows unless
    ``break_mw``; a given one is written verbatim), ``boundary =
    (lh_b, rh_b)`` two ``(n_h, L_h)`` 0/1 masks, ``mtc`` ``(D, L)``,
    ``grad`` ``(S, N, n_grad)`` fp32 (the generated embeddings carry two
    extra columns the loader must slice off). ``cfg`` is the gMSHBM
    ``Step2Config`` of the project; ``refs`` holds what was written.
    """
    rng = np.random.default_rng(seed)
    N, L = 2 * n_h, 2 * L_h
    mw_idx = np.asarray(mw, dtype=np.int64)

    raws = []
    for s in range(1, S + 1):
        if bold is None:
            raw = (rng.random((T, N, D)) < 0.4).astype(np.float32)
            if not break_mw:
                raw[:, mw_idx, :] = 0.0             # LH medial wall
                raw[:, n_h + mw_idx, :] = 0.0       # RH medial wall
        else:
            raw = np.ascontiguousarray(bold[s - 1])
        write_subject_profile_tnd(profile_path(root, str(s), TARG, SEED),
                                  np.ascontiguousarray(raw))
        raws.append(raw)

    grads = []
    for s in range(1, S + 1):
        gdir = root / "gradients" / f"sub{s}"
        gdir.mkdir(parents=True, exist_ok=True)
        if grad is None:
            lh = rng.standard_normal((n_h, n_grad + 2)).astype(np.float32)
            rh = rng.standard_normal((n_h, n_grad + 2)).astype(np.float32)
        else:
            g = np.asarray(grad[s - 1], dtype=np.float32)
            lh, rh = g[:n_h], g[n_h:]
        if grad_fmt == "npy":
            np.save(gdir / f"lh_emb_{n_grad}_distance_matrix.npy", lh)
            np.save(gdir / f"rh_emb_{n_grad}_distance_matrix.npy", rh)
        else:
            sio.savemat(gdir / f"lh_emb_{n_grad}_distance_matrix.mat",
                        {"emb": lh})
            sio.savemat(gdir / f"rh_emb_{n_grad}_distance_matrix.mat",
                        {"emb": rh})
        grads.append(np.concatenate([lh[:, :n_grad], rh[:, :n_grad]], axis=0))

    (root / "group").mkdir(parents=True, exist_ok=True)
    if mtc is None:
        mtc = rng.standard_normal((D, L))
    sio.savemat(root / "group" / "group.mat",
                {"mtc": np.asarray(mtc, dtype=np.float64),
                 "epsil": np.array([[3.0]]),
                 "lh_labels": np.arange(n_h).reshape(-1, 1),
                 "rh_labels": np.arange(n_h).reshape(-1, 1),
                 "lambda": rng.standard_normal((D, 4 * L))},
                do_compression=True, format="5")

    (root / "spatial_mask").mkdir(parents=True, exist_ok=True)
    if boundary is None:
        lh_b = (rng.random((n_h, L_h)) < 0.5).astype(np.float64)
        rh_b = (rng.random((n_h, L_h)) < 0.5).astype(np.float64)
        for m in (lh_b, rh_b):
            m[:, 0] = 0.0
            m[3, 0] = 1.0                            # no empty column
    else:
        lh_b, rh_b = (np.asarray(m, dtype=np.float64) for m in boundary)
    sio.savemat(root / "spatial_mask" / f"spatial_mask_{TARG}.mat",
                {"lh_boundary": sp.csc_matrix(lh_b),
                 "rh_boundary": sp.csc_matrix(rh_b)},
                do_compression=True, format="5")

    backfill_cohort_json(root,
                         subjects=[str(s) for s in range(1, S + 1)],
                         sessions=[str(t) for t in range(1, T + 1)],
                         targ_mesh=TARG, seed_mesh=SEED,
                         n_grad_components=n_grad)

    cfg = Step2Config(project_dir=root, num_sub=S, num_session=T,
                      num_clusters=L, mesh=TARG, seed_mesh=SEED,
                      n_grad_components=n_grad)
    refs = {"raws": raws, "grads": grads, "mtc": mtc,
            "lh_b": lh_b, "rh_b": rh_b, "n_h": n_h, "N": N, "L": L, "D": D,
            "mw": mw_idx}
    return cfg, refs


def load_project(cfg, refs, *, overlap: bool = True):
    """``load_step2_sparse_inputs(cfg)`` on a project ``make_project``
    wrote: for the duration of the call ``load_avg_mesh`` is the mesh
    that project was written for (``refs['n_h']`` vertices per
    hemisphere, medial wall at ``refs['mw']``)."""
    lab = np.full((1, refs["n_h"]), 2, dtype=np.int32)
    lab[0, refs["mw"]] = 1                          # 1 == medial wall
    # ``arealmshbm.data_io.load_avg_mesh`` names the re-exported function
    # in the package namespace, so the dotted-string form of setattr
    # resolves to the function, not the module. Grab the module itself.
    mod = importlib.import_module("arealmshbm.data_io.load_avg_mesh")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod, "load_avg_mesh",
                   lambda hemi, mesh, surf: {"MARS_label": lab})
        return load_step2_sparse_inputs(cfg, overlap=overlap)
