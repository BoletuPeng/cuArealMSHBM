"""test_gradient_from_disk.py — step 3 takes its gradient from the two
files ``cohort.json`` names.

Those files are the whole step 0 → step 3 hand-off, so both backends'
readers — ``fetch_data`` (cpu) and ``fetch_gradient`` (gpu) — must
return the subject's lh file stacked over its rh file, cut to
``n_grad_components`` columns, in either format step 0 writes.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.data_io import fetch_data
from arealmshbm.step2_io.tests._synthetic_project import make_project
from arealmshbm.step3_pipeline.sparse_inputs import fetch_gradient


@pytest.mark.parametrize("fmt", ["npy", "mat"])
def test_both_readers_return_the_cohort_files(tmp_path, fmt):
    cfg, refs = make_project(tmp_path, S=2, T=2, grad_fmt=fmt)
    lab = np.full(refs["n_h"], 2, dtype=np.int32)
    lab[refs["mw"]] = 1                              # 1 == medial wall
    mesh = {"MARS_label": lab}

    for subid in (1, 2):
        want = refs["grads"][subid - 1]
        got_gpu = fetch_gradient(tmp_path, subid, cfg.n_grad_components)
        got_cpu = fetch_data(
            tmp_path, cfg.num_session, subid, mesh, mesh,
            n_grad_components=cfg.n_grad_components)["gradient_mat"]
        for got in (got_gpu, got_cpu):
            assert got.dtype == np.float32
            assert np.array_equal(got, want)
