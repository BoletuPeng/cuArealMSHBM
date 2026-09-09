"""test_load_group_prior.py — the Mode-A ``Params_Final.mat`` reader.

A ``Params_Final.mat`` whose writer chose the sparse container (a
MATLAB-side ``sparse()`` — the in-tree step-2 writer always densifies)
hands this reader a ``csc_array``/``csc_matrix`` out of
``scipy.io.loadmat``.
Without the densify, ``np.ascontiguousarray`` on one of those raises
``ValueError: setting an array element with a sequence``, the reader's
own ``except`` re-routes into the h5py branch, and the run dies with a
misleading ``OSError: file signature not found``. These tests pin the
densify and the ``(N, L)`` fp32 output contract on both containers.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from arealmshbm.data_io.load_group_prior import load_group_prior


N, L, D = 31, 5, 7


def _prior(seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = rng.random((N, L)).astype(np.float32)
    theta[theta < 0.6] = 0.0
    return {
        "mu": rng.standard_normal((D, L)).astype(np.float32),
        "theta": theta,
        "epsil": (rng.random(L) * 100).astype(np.float32).reshape(1, L),
        "sigma": (rng.random(L) * 100).astype(np.float32).reshape(1, L),
    }


def _write(path: Path, prior: dict, *, sparse_theta: bool, compress: bool):
    theta = prior["theta"].astype(np.float64)
    out = {
        "mu": prior["mu"].astype(np.float64),
        "theta": sp.csc_matrix(theta) if sparse_theta else theta,
        "epsil": prior["epsil"].astype(np.float64),
        "sigma": prior["sigma"].astype(np.float64),
    }
    sio.savemat(path, {"Params": out}, do_compression=compress, format="5")


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("sparse_theta", [False, True])
def test_dense_and_sparse_theta_read_identically(tmp_path: Path,
                                                 sparse_theta: bool,
                                                 compress: bool) -> None:
    ref = _prior()
    p = tmp_path / f"Params_Final_{int(sparse_theta)}_{int(compress)}.mat"
    _write(p, ref, sparse_theta=sparse_theta, compress=compress)

    got = load_group_prior(p)
    assert set(got) == {"mu", "theta", "epsil", "sigma"}
    for k in got:
        assert got[k].dtype == np.float32, k
    assert got["theta"].shape == (N, L)
    assert got["theta"].flags["C_CONTIGUOUS"]
    assert got["mu"].shape == (D, L)
    assert got["epsil"].shape == (1, L) and got["sigma"].shape == (1, L)
    for k in ("mu", "theta", "epsil", "sigma"):
        assert np.array_equal(got[k], ref[k]), k


def test_sparse_matches_dense_bit_for_bit(tmp_path: Path) -> None:
    ref = _prior(seed=3)
    d, s = tmp_path / "dense.mat", tmp_path / "sparse.mat"
    _write(d, ref, sparse_theta=False, compress=True)
    _write(s, ref, sparse_theta=True, compress=True)
    a, b = load_group_prior(d), load_group_prior(s)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k


def test_an_all_zero_theta_column_survives(tmp_path: Path) -> None:
    ref = _prior(seed=5)
    ref["theta"][:, 2] = 0.0
    p = tmp_path / "zerocol.mat"
    _write(p, ref, sparse_theta=True, compress=False)
    got = load_group_prior(p)
    assert got["theta"].shape == (N, L)
    assert np.all(got["theta"][:, 2] == 0.0)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_group_prior(tmp_path / "nope.mat")
