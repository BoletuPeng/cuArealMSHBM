"""test_group_mat_writer.py — ``group.mat`` write plumbing.

Pins that the sync and background writers produce a file the step-2
reader accepts, that ``do_compression=False`` round-trips the same
values as the compressed default (only the file size changes), that the
background writer snapshots the payload, and that a ``savemat`` failure
reaches ``wait()``. The handle's own failure semantics are pinned in
``data_io/tests/test_background_write.py``.

Run::

    python -m pytest arealmshbm/ini_params/tests/test_group_mat_writer.py -v

Pure CPU — no cupy, no mesh bundle.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.io import loadmat

from arealmshbm.data_io._background_write import BackgroundWriteHandle
from arealmshbm.ini_params._group_mat_writer import (
    IniParamsResult, write_group_mat,
)
from arealmshbm.step2_io.load_group_mtc import load_group_mtc


def _payload(D=40, L=6, N=200):
    rng = np.random.default_rng(4)
    return {
        "mtc": rng.standard_normal((D, L)),
        "epsil": np.array([[1382.004888945558]], dtype=np.float64),
        "lambda": (rng.random((N, L)) < 0.2).astype(np.uint8),
        "lh_labels": rng.integers(0, L + 1, size=(N, 1)).astype(np.int64),
        "rh_labels": rng.integers(0, L + 1, size=(N, 1)).astype(np.int64),
    }


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("background", [False, True])
def test_round_trips_through_load_group_mtc(tmp_path, compress, background):
    pay = _payload()
    handle = write_group_mat(str(tmp_path), pay, compress=compress,
                             background=background)
    if background:
        assert isinstance(handle, BackgroundWriteHandle)
        assert handle.wait() == tmp_path / "group" / "group.mat"
        handle.wait()                       # idempotent
        assert handle.exception is None
    else:
        assert handle is None

    p = tmp_path / "group" / "group.mat"
    raw = loadmat(str(p))
    for k, v in pay.items():
        assert np.array_equal(raw[k], v), k
    assert set(raw) >= set(pay), "all five keys must survive the write"

    got = load_group_mtc(p)
    assert np.array_equal(got["mtc"], pay["mtc"])
    assert got["epsil"] == float(pay["epsil"].ravel()[0])
    assert np.array_equal(got["lh_labels"], pay["lh_labels"].ravel())
    assert np.array_equal(got["rh_labels"], pay["rh_labels"].ravel())


def test_compressed_and_uncompressed_agree_on_values(tmp_path):
    pay = _payload()
    a, b = tmp_path / "a", tmp_path / "b"
    write_group_mat(str(a), pay, compress=True)
    write_group_mat(str(b), pay, compress=False)
    ma = loadmat(str(a / "group" / "group.mat"))
    mb = loadmat(str(b / "group" / "group.mat"))
    for k in pay:
        assert np.array_equal(ma[k], mb[k]), k
    assert (b / "group" / "group.mat").stat().st_size > \
           (a / "group" / "group.mat").stat().st_size


def test_writer_snapshots_the_payload(tmp_path):
    """The background writer must not observe a later mutation."""
    pay = _payload()
    res = IniParamsResult(pay)
    res.writer = write_group_mat(str(tmp_path), res, compress=False,
                                 background=True)
    res["mtc"] = np.zeros_like(res["mtc"])      # racing mutation
    res.writer.wait()
    got = loadmat(str(tmp_path / "group" / "group.mat"))
    assert np.array_equal(got["mtc"], pay["mtc"])


def _bad_payload():
    return {"mtc": np.zeros((2, 2)), "not serialisable": object()}


def test_wait_reraises_writer_failure(tmp_path):
    h = write_group_mat(str(tmp_path), _bad_payload(), compress=False,
                        background=True)
    with pytest.raises(Exception):
        h.wait()
    assert h.exception is not None


def test_result_is_a_plain_dict_for_consumers():
    res = IniParamsResult(_payload())
    assert isinstance(res, dict)
    assert res.writer is None
    assert set(res) == {"mtc", "epsil", "lambda", "lh_labels", "rh_labels"}
    assert dict(res)["mtc"].shape == (40, 6)
