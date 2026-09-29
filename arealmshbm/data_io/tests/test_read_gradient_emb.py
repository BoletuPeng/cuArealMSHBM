"""test_read_gradient_emb.py — the one gradient reader for steps 2 and 3.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arealmshbm.data_io.fetch_data import read_gradient_emb


def test_npy_and_mat_agree(tmp_path: Path):
    from scipy.io import savemat
    emb = np.random.default_rng(3).standard_normal((40, 8)).astype(np.float32)
    np.save(tmp_path / "g.npy", emb)
    savemat(tmp_path / "g.mat", {"emb": emb.astype(np.float64)})
    a = read_gradient_emb(tmp_path / "g.npy", 5)
    b = read_gradient_emb(tmp_path / "g.mat", 5)
    assert a.shape == b.shape == (40, 5)
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.array_equal(a, emb[:, :5])


def test_missing_emb_names_the_file(tmp_path: Path):
    from scipy.io import savemat
    p = tmp_path / "g.mat"
    savemat(p, {"not_emb": np.zeros((4, 4))})
    with pytest.raises(KeyError, match=r"'emb'.*g\.mat"):
        read_gradient_emb(p, 2)


def test_missing_emb_v73_names_the_file(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "g73.mat"
    with h5py.File(p, "w", userblock_size=512) as f:
        f.create_dataset("not_emb", data=np.zeros((4, 4)))
    # A MAT v7.3 file is HDF5 with a 512-byte MATLAB user block.
    with open(p, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file".ljust(116, b" ")
                 + b"\x00" * 8 + b"\x00\x02" + b"IM")
    with pytest.raises(KeyError, match=r"'emb'.*g73\.mat"):
        read_gradient_emb(p, 2)


def test_too_few_components(tmp_path: Path):
    np.save(tmp_path / "g.npy", np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError, match=r"need >= 5"):
        read_gradient_emb(tmp_path / "g.npy", 5)
