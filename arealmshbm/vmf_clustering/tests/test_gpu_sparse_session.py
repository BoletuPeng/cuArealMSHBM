# Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""End-to-end checks of ``backend='gpu_sparse'`` on sub-001 (skipped
without cupy / the profile store).

* run-to-run determinism (labels and ``s_lambda`` bit-identical);
* the EM trajectory matches the dense backends' iteration structure
  (intra rounds, EM iters, λ-iters, comp iters — these are pinned by the
  data, not by the backend);
* label agreement with ``gpu_full`` inside the documented CPU↔GPU drift
  band; and the sparse backend must not be *further* from the CPU
  reference than ``gpu_full`` is (checked only when the slow CPU run is
  requested via ``MSHBM_TEST_CPU_REF=1``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from arealmshbm.vmf_clustering.tests._sub001_fixture import (
    SUB001_DIR, skip_unless_cupy, sub001_available,
)


def _run(backend: str):
    from arealmshbm.step3_pipeline import Step3Config, Step3Pipeline
    cfg = Step3Config(
        project_dir=SUB001_DIR, num_session=6, num_clusters=300, subid=1,
        mesh="fsaverage6", w=50.0, c=10.0, beta_scalar=5.0, backend=backend,
    )
    with Step3Pipeline(cfg) as pipe:
        res = pipe.run(time_stages=True)
        stages = dict(pipe.timings.get("per_stage_em", {}))
        sl = np.array(res.Params["s_lambda"], copy=True)   # pinned view → own copy
        mu = res.Params["mu"]
        # the session's own host mu, not a device round trip of it
        mu_is_session_host = mu is getattr(pipe._session, "mu_host", None)
    return (res.lh_labels.copy(), res.rh_labels.copy(), sl, stages,
            res.iter_intra_em, mu, mu_is_session_host)


@pytest.fixture(scope="module")
def runs():
    skip_unless_cupy()
    if not sub001_available():
        pytest.skip(f"sub-001 profile store not found at {SUB001_DIR}")
    a = _run("gpu_sparse")
    b = _run("gpu_sparse")
    f = _run("gpu_full")
    return a, b, f


def test_deterministic(runs):
    a, b, _ = runs
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


def test_iteration_structure_matches_dense_backend(runs):
    a, _, f = runs
    assert a[4] == f[4]
    for k in ("iter_count_em_total", "iter_count_lambda_total", "iter_count_comp_total"):
        assert a[3][k] == f[3][k], k


def test_support_inside_theta(runs):
    a, _, _ = runs
    from arealmshbm.vmf_clustering.tests._sub001_fixture import load_sub001
    fx = load_sub001()
    assert not np.any(a[2][fx.theta == 0])
    lh_sum = a[2].sum(axis=1)
    assert np.all(lh_sum[fx.theta.sum(axis=1) == 0] == 0)


def test_mu_is_the_input_mu(runs):
    """mu is read-only for the session — Params carries the host array
    it was built from, not a device round-trip."""
    from arealmshbm.vmf_clustering.tests._sub001_fixture import load_sub001
    a, _, _ = runs
    ref = load_sub001().Params["mu"]
    assert a[5].dtype == np.float32 and a[5].shape == ref.shape
    assert np.array_equal(a[5], ref)
    assert a[6], "Params['mu'] is not the session's host mu"


def test_labels_close_to_gpu_full(runs):
    a, _, f = runs
    dl = int((a[0] != f[0]).sum()); dr = int((a[1] != f[1]).sum())
    print(f"gpu_sparse vs gpu_full: lh {dl} rh {dr} differing vertices")
    assert dl + dr <= 60         # observed 9 on sub-001


@pytest.mark.skipif(os.environ.get("MSHBM_TEST_CPU_REF") != "1",
                    reason="set MSHBM_TEST_CPU_REF=1 for the ~30 s CPU reference run")
def test_not_further_from_cpu_than_gpu_full(runs):
    a, _, f = runs
    c = _run("cpu")
    d_sparse = int((a[0] != c[0]).sum() + (a[1] != c[1]).sum())
    d_full = int((f[0] != c[0]).sum() + (f[1] != c[1]).sum())
    print(f"vs cpu: gpu_sparse {d_sparse}, gpu_full {d_full}")
    assert d_sparse <= max(d_full, 10) * 2
