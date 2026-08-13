"""test_subject_loaders.py — synthetic-only self-tests for the per-subject
streaming loaders.

Covers:
  1. SubjectProfileLoader.b2nd path — normalize semantics + load_into
     scratch reuse.
  2. SubjectGradientLoader.npy path — bilateral stack matches manual
     concatenation, replicated across T (gradient is session-invariant
     in this fork).

Synthetic-only — no external data required. Uses
:func:`backfill_cohort_json` to construct a minimal cohort.json after
writing the synthetic fixtures.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange

from arealmshbm.data_io.cohort import (
    backfill_cohort_json, read_cohort,
)
from arealmshbm.data_io.profile_io import (
    profile_path,
    write_subject_profile_tnd,
)
from arealmshbm.step2_io import (
    SubjectGradientLoader,
    SubjectProfileLoader,
)


# ─────────────────────────────────────────────────────────────────────
# Reference normalize impl — test-only.
#
# Production no longer ships a host fp32-input normalize kernel: the
# only in-tree normalize path is the fused bitpacked-input
# ``_widen_normalize_bitpacked_to_f32_NTD_kernel`` in
# ``arealmshbm/step2_io/load_subject_profiles.py``. This local 3-pass
# numba kernel exists ONLY to validate that fused kernel by comparing
# against the equivalent fp32-input demean + L2-norm. fp32 sequential
# summation matches the bit-identical contract documented at the
# production kernel.
# ─────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False, boundscheck=False,
      error_model="numpy", parallel=True)
def _normalize_session_inplace_ref(series_ND):
    N, D = series_ND.shape
    inv_D = np.float32(1.0) / np.float32(D)
    for n in prange(N):
        row = series_ND[n]
        s_sum = np.float32(0.0)
        for d in range(D):
            s_sum += row[d]
        mean = s_sum * inv_D

        all_nonzero = True
        sq_sum = np.float32(0.0)
        for d in range(D):
            v = row[d] - mean
            row[d] = v
            if v == np.float32(0.0):
                all_nonzero = False
            sq_sum += v * v

        if all_nonzero and sq_sum > np.float32(0.0):
            inv = np.float32(1.0) / np.float32(np.sqrt(sq_sum))
            for d in range(D):
                row[d] = row[d] * inv


def _normalize_session_ref(series: np.ndarray) -> np.ndarray:
    """fp32 demean + L2-norm reference impl — see kernel docstring above."""
    series = np.ascontiguousarray(series, dtype=np.float32)
    _normalize_session_inplace_ref(series)
    return series


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _build_b2nd_fixture(project_dir: Path, S: int, T: int, N: int, D: int,
                        *, targ_mesh: str = "fsaverage6",
                        seed_mesh: str = "fsaverage3"):
    """Write S per-subject .b2nd files at the canonical
    ``profiles_raw/sub<S>/...profile.b2nd`` path, build a minimal
    cohort.json, and return ``(raw_arrays, cohort)``.
    """
    rng = np.random.default_rng(7)
    raws = []
    for s in range(1, S + 1):
        raw_TND = (rng.random((T, N, D)) < 0.1).astype(np.float32)
        p = profile_path(project_dir, str(s), targ_mesh, seed_mesh)
        write_subject_profile_tnd(p, np.ascontiguousarray(raw_TND))
        raws.append(raw_TND)
    cohort = backfill_cohort_json(
        project_dir,
        subjects=[str(s) for s in range(1, S + 1)],
        sessions=[str(t) for t in range(1, T + 1)],
        targ_mesh=targ_mesh,
        seed_mesh=seed_mesh,
        n_grad_components=100,
    )
    return raws, cohort


# ─────────────────────────────────────────────────────────────────────
# SubjectProfileLoader — .b2nd path produces normalized rows
# ─────────────────────────────────────────────────────────────────────
def test_profile_loader_b2nd_matches_inline_normalize(tmp_path: Path) -> None:
    """When .b2nd is present, the loader's output equals the in-memory
    normalize of the same raw binary content. Medial-wall rows are NOT
    re-zeroed (b2nd content is pre-medial-zeroed by step1
    generate_profiles)."""
    S, T, N, D = 2, 3, 17, 5
    raws, cohort = _build_b2nd_fixture(tmp_path, S, T, N, D)

    loader = SubjectProfileLoader(
        cohort=cohort,
        project_dir=tmp_path,
        targ_mesh="fsaverage6",
        seed_mesh="fsaverage3",
    )
    assert loader.format == "b2nd"
    assert loader.dims() == (N, T, D)

    for s_1, raw_TND in enumerate(raws, start=1):
        got = loader.load(s_1)
        assert got.dtype == np.float32
        assert got.shape == (N, T, D)

        # Reproduce: per-t, demean + L2-norm via the local reference impl.
        expected = np.empty((N, T, D), dtype=np.float32)
        for t in range(T):
            expected[:, t, :] = _normalize_session_ref(raw_TND[t])
        np.testing.assert_array_equal(got, expected)


# ─────────────────────────────────────────────────────────────────────
# SubjectProfileLoader — load_into reuses caller scratch
# ─────────────────────────────────────────────────────────────────────
def test_profile_loader_load_into_zero_alloc(tmp_path: Path) -> None:
    S, T, N, D = 2, 2, 9, 4
    _, cohort = _build_b2nd_fixture(tmp_path, S, T, N, D)
    loader = SubjectProfileLoader(
        cohort=cohort,
        project_dir=tmp_path,
        targ_mesh="fsaverage6", seed_mesh="fsaverage3",
    )

    scratch = np.empty((N, T, D), dtype=np.float32)
    # Hammer the same scratch slot across subjects; final state must
    # match a fresh .load() of the last subject.
    for s_1 in range(1, S + 1):
        loader.load_into(s_1, scratch)
        np.testing.assert_array_equal(scratch, loader.load(s_1))


# ─────────────────────────────────────────────────────────────────────
# SubjectGradientLoader — .npy path matches manual stack
# ─────────────────────────────────────────────────────────────────────
def test_gradient_loader_npy_round_trip(tmp_path: Path) -> None:
    """When per-hemi .npy files are present at the canonical step0
    output location, ``backfill_cohort_json`` records them in cohort.json
    and the loader serves them. Output is the manually-concatenated
    per-hemi arrays replicated across T."""
    S, T, n_lh, n_rh, D_grad = 2, 2, 6, 4, 7
    N = n_lh + n_rh

    rng = np.random.default_rng(3)
    for s_1 in range(1, S + 1):
        sub_dir = tmp_path / "gradients" / f"sub{s_1}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        lh = rng.standard_normal((n_lh, D_grad)).astype(np.float32)
        rh = rng.standard_normal((n_rh, D_grad)).astype(np.float32)
        np.save(sub_dir / f"lh_emb_{D_grad}_distance_matrix.npy", lh)
        np.save(sub_dir / f"rh_emb_{D_grad}_distance_matrix.npy", rh)

    cohort = backfill_cohort_json(
        tmp_path,
        subjects=[str(s) for s in range(1, S + 1)],
        sessions=[str(t) for t in range(1, T + 1)],
        targ_mesh="fsaverage6", seed_mesh="fsaverage3",
        n_grad_components=D_grad,
    )

    loader = SubjectGradientLoader(
        cohort=cohort,
        project_dir=tmp_path,
        n_components=D_grad,
    )
    assert loader.format == "npy"
    assert loader.dims() == (N, D_grad)

    for s_1 in range(1, S + 1):
        sub_dir = tmp_path / "gradients" / f"sub{s_1}"
        lh = np.load(sub_dir / f"lh_emb_{D_grad}_distance_matrix.npy")
        rh = np.load(sub_dir / f"rh_emb_{D_grad}_distance_matrix.npy")
        stacked = np.concatenate([lh, rh], axis=0).astype(np.float32)
        got = loader.load(s_1)
        for t in range(T):
            np.testing.assert_array_equal(got[t], stacked)
