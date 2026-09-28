# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""End-to-end checks of ``backend='gpu'`` on sub-001 (skipped without
cupy / the profile store), for gMSHBM and cMSHBM:

* run-to-run determinism (labels and ``s_lambda`` bit-identical);
* the EM trajectory's iteration structure (intra rounds, EM iters,
  λ-iters, comp iters, check_connectedness and xyz-prior calls) and the
  comp loop's end state, pinned as the sub-001 values — these are set by
  the data, and the cpu backend produces the same numbers;
* ``save()`` writes the argmax labels (gMSHBM) or their
  ``remove_isolated`` cleanup on the inflated mesh (cMSHBM);
* with ``MSHBM_TEST_CPU_REF=1``, a cross-check against a ~30 s cpu run:
  same iteration structure and label agreement inside the documented
  CPU↔GPU drift band.

cMSHBM additionally pins the comp loop's convergence, which needs the
device ``remove_isolated`` pre-predicate inside ``check_connectedness``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import numpy as np
import pytest

from arealmshbm.vmf_clustering.tests._sub001_fixture import (
    SUB001_DIR, skip_unless_cupy, sub001_available,
)

VARIANTS = ("gMSHBM", "cMSHBM")
ITER_KEYS = ("iter_count_em_total", "iter_count_lambda_total",
             "iter_count_comp_total", "iter_count_check_conn",
             "iter_count_spatial_xyz")

# Measured on sub-001 (w=50, c=10, beta=5, L=300); cpu gives the same.
#   variant: (iter_intra_em, ITER_KEYS values, max_components)
SUB001_STRUCTURE = {
    "gMSHBM": (3, (10, 115, 108, 105, 105), 4.0),
    "cMSHBM": (3, (9, 113, 104, 104, 113), 1.0),
}

# SHA-256 of int64 ``concat(lh, rh)`` from the gpu backend on sub-001.
# These labels are 3 (gMSHBM) / 0 (cMSHBM) vertices from the cpu
# backend's (``test_matches_cpu_reference``). The pin is bound to the
# toolchain: ``m_step_gpu`` compiles with FMA contraction on, so a
# CUDA / cupy / driver upgrade may move a handful of vertices. After
# such an upgrade rerun the MSHBM_TEST_CPU_REF=1 cross-check, and if
# it still holds, re-pin.
SUB001_LABELS_SHA256 = {
    "gMSHBM": "c59477e800340aacb0b3c88286b98e940c1123844753ca6d91e8713333a89060",
    "cMSHBM": "d1ca80eac80bebe3f3e4fe6f8f07c67a267fb2b978444f8ed4c88f3184c65bcd",
}


def _config(backend: str, variant: str):
    from arealmshbm.step3_pipeline import Step3Config
    return Step3Config(
        project_dir=SUB001_DIR, num_session=6, num_clusters=300, subid=1,
        mesh="fsaverage6", w=50.0, c=10.0, beta_scalar=5.0, backend=backend,
        pipeline_type=variant,
    )


class Run(NamedTuple):
    lh: np.ndarray
    rh: np.ndarray
    s_lambda: np.ndarray
    stages: Dict[str, float]
    iter_intra_em: int
    mu: np.ndarray
    mu_is_session_host: bool
    max_components: float
    saved_lh: Optional[np.ndarray] = None       # from ``save()``, when asked for
    saved_rh: Optional[np.ndarray] = None


class Runs(NamedTuple):
    variant: str
    a: Run              # gpu
    b: Run              # gpu, second run


def _run(backend: str, variant: str, save_dir: Optional[Path] = None) -> Run:
    from scipy.io import loadmat
    from arealmshbm.step3_pipeline import Step3Pipeline
    saved_lh = saved_rh = None
    with Step3Pipeline(_config(backend, variant)) as pipe:
        res = pipe.run(time_stages=True)
        stages = dict(pipe.timings.get("per_stage_em", {}))
        sl = np.array(res.Params["s_lambda"], copy=True)   # pinned view → own copy
        lh, rh = res.lh_labels.copy(), res.rh_labels.copy()  # save() rewrites them (cMSHBM)
        mu = res.Params["mu"]
        # the session's own host mu, not a device round trip of it
        mu_is_session_host = mu is getattr(pipe._session, "mu_host", None)
        if save_dir is not None:
            m = loadmat(str(pipe.save(res, out_path=save_dir / "out.mat")))
            saved_lh = m["lh_labels"].ravel().astype(np.int64)
            saved_rh = m["rh_labels"].ravel().astype(np.int64)
    return Run(lh, rh, sl, stages, res.iter_intra_em, mu, mu_is_session_host,
               float(res.Params["max_components"]), saved_lh, saved_rh)


@pytest.fixture(scope="module", params=VARIANTS)
def runs(request, tmp_path_factory) -> Runs:
    skip_unless_cupy()
    if not sub001_available():
        pytest.skip(f"sub-001 profile store not found at {SUB001_DIR}")
    variant = request.param
    if not (SUB001_DIR / "priors" / variant / "beta5" / "Params_Final.mat").exists():
        pytest.skip(f"no {variant} prior staged for sub-001")
    a = _run("gpu", variant, tmp_path_factory.mktemp(variant))
    b = _run("gpu", variant)
    return Runs(variant, a, b)


def _prior(variant: str):
    from arealmshbm.data_io.load_group_prior import load_group_prior
    return load_group_prior(_config("gpu", variant).group_prior_path)


def test_deterministic(runs):
    a, b = runs.a, runs.b
    assert np.array_equal(a.lh, b.lh) and np.array_equal(a.rh, b.rh)
    assert np.array_equal(a.s_lambda, b.s_lambda)


def test_labels_pinned(runs):
    import hashlib
    a = runs.a
    x = np.ascontiguousarray(
        np.concatenate([a.lh.ravel(), a.rh.ravel()]).astype(np.int64))
    assert (hashlib.sha256(x.tobytes()).hexdigest()
            == SUB001_LABELS_SHA256[runs.variant])


def test_iteration_structure(runs):
    a = runs.a
    intra, iters, _ = SUB001_STRUCTURE[runs.variant]
    assert a.iter_intra_em == intra
    assert tuple(int(a.stages[k]) for k in ITER_KEYS) == iters


def test_comp_loop_end_state(runs):
    """``max_components`` at the end of the comp loop is pinned by the
    data. cMSHBM's loop stops only once every parcel is one component
    (threshold 1); without the ``remove_isolated`` pre-predicate it never
    gets there on sub-001 (stays at 3, the EM takes an extra iteration)."""
    a = runs.a
    assert a.max_components == SUB001_STRUCTURE[runs.variant][2]
    if runs.variant == "cMSHBM":
        assert a.max_components == 1.0


def test_support_inside_theta(runs):
    theta = _prior(runs.variant)["theta"]
    sl = runs.a.s_lambda
    assert not np.any(sl[theta == 0])
    lh_sum = sl.sum(axis=1)
    assert np.all(lh_sum[theta.sum(axis=1) == 0] == 0)


def test_mu_is_the_input_mu(runs):
    """mu is read-only for the session — Params carries the host array
    it was built from, not a device round-trip."""
    ref = _prior(runs.variant)["mu"]
    a = runs.a
    assert a.mu.dtype == np.float32 and a.mu.shape == ref.shape
    assert np.array_equal(a.mu, ref)
    assert a.mu_is_session_host, "Params['mu'] is not the session's host mu"


def test_saved_labels(runs):
    """``save()`` on gpu: gMSHBM writes the argmax labels as they are;
    cMSHBM writes their ``remove_isolated`` cleanup (host function on
    the inflated-mesh neighbours — the sparse inputs' ``lh_inflated`` /
    ``rh_inflated`` exist for this one reader)."""
    a = runs.a
    if runs.variant == "cMSHBM":
        from arealmshbm.postprocessing import remove_isolated_surface_components
        from arealmshbm.vmf_clustering.tests._sub001_fixture import load_sub001
        fx = load_sub001()
        thr = int(_config("gpu", "cMSHBM").cMSHBM_isolated_component_min_size)
        exp_lh = remove_isolated_surface_components(a.lh, fx.lh_inflated["vertexNbors"], thr)
        exp_rh = remove_isolated_surface_components(a.rh, fx.rh_inflated["vertexNbors"], thr)
        assert np.any(exp_lh != a.lh) or np.any(exp_rh != a.rh)   # 746 verts on sub-001
    else:
        exp_lh, exp_rh = a.lh, a.rh
    assert np.array_equal(a.saved_lh, exp_lh) and np.array_equal(a.saved_rh, exp_rh)


@pytest.mark.skipif(os.environ.get("MSHBM_TEST_CPU_REF") != "1",
                    reason="set MSHBM_TEST_CPU_REF=1 for the ~30 s CPU reference run")
def test_matches_cpu_reference(runs):
    a = runs.a
    c = _run("cpu", runs.variant)
    assert a.iter_intra_em == c.iter_intra_em
    for k in ITER_KEYS:
        assert a.stages[k] == c.stages[k], k
    assert a.max_components == c.max_components
    dl = int((a.lh != c.lh).sum()); dr = int((a.rh != c.rh).sum())
    print(f"{runs.variant} gpu vs cpu: lh {dl} rh {dr} differing vertices")
    assert dl + dr <= 60         # observed on sub-001: 3 (gMSHBM), 0 (cMSHBM)
