"""test_save.py — ``Params_Final.mat`` writer (``step2_pipeline/_save.py``).

The saver is the one place the internal Params layout becomes the
external MATLAB one, and the one place a scipy-sparse ``theta`` (what
the ``gpu`` backend exports) becomes the dense block MATLAB CBIG
can read. The bars:

* the file reproduces the pre-2026-09 ``_save_params`` byte semantics
  (``mu`` transposed, float arrays widened to fp64, scalars / lists
  passed through, container compressed);
* a sparse ``theta`` in gives the same bytes out as the equal dense one.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from arealmshbm.data_io.load_group_prior import load_group_prior
from arealmshbm.step2_pipeline._save import save_params_final


N, L, D, S = 40, 6, 7, 2


def _params(seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = rng.random((N, L)).astype(np.float32)
    theta[theta < 0.7] = 0.0                     # ~30 % dense, like the real one
    theta[3, :] = 0.0                            # a fully dead row
    return {
        "ini_val": 403.61102294921875,
        "sigma": (rng.random((1, L)) * 100).astype(np.float32),
        "epsil": (rng.random((1, L)) * 100).astype(np.float32),
        "kappa": (rng.random((1, L)) * 100).astype(np.float32),
        "mu": rng.standard_normal((L, D)).astype(np.float32),
        "theta": theta,
        "cost_em": rng.standard_normal(S),
        "iter_inter": 4,
        "iter_intra": 2,
        "Record": [-1.5, -1.25, -1.2],
        "cost_intra": -1.2,
        "cost_inter": -1.2,
        "mode": "gMSHBM",
    }


def _load(path: Path):
    return sio.loadmat(str(path), squeeze_me=False)["Params"]


def test_field_shapes_and_dtypes(tmp_path: Path) -> None:
    P = _params()
    p = tmp_path / "Params_Final.mat"
    save_params_final(P, p)
    out = _load(p)

    mu = out["mu"][0, 0]
    assert mu.shape == (D, L) and mu.dtype == np.float64
    assert np.array_equal(mu, P["mu"].T.astype(np.float64))

    for k in ("sigma", "epsil", "kappa"):
        v = out[k][0, 0]
        assert v.shape == (1, L) and v.dtype == np.float64
        assert np.array_equal(v, P[k].astype(np.float64))

    assert out["cost_em"][0, 0].dtype == np.float64
    assert np.array_equal(out["cost_em"][0, 0].ravel(), P["cost_em"])
    assert int(out["iter_inter"][0, 0].ravel()[0]) == 4
    assert np.array_equal(out["Record"][0, 0].ravel(), np.asarray(P["Record"]))
    assert float(out["cost_inter"][0, 0].ravel()[0]) == pytest.approx(-1.2)
    assert str(out["mode"][0, 0][0]) == "gMSHBM"

    th = out["theta"][0, 0]
    assert not sp.issparse(th)
    assert th.shape == (N, L) and th.dtype == np.float64


def test_a_sparse_theta_is_densified(tmp_path: Path) -> None:
    """A csc ``theta`` (what ``export_params`` returns) comes out dense.

    MATLAB CBIG's ``CBIG_MSHBM_generate_individual_parcellation.m``
    evaluates ``log(Params.theta)``, which MATLAB will not do on a
    sparse input, so the writer owns the densify.
    """
    P = _params(seed=13)
    dense_theta = P["theta"]
    P["theta"] = sp.csc_matrix(dense_theta.astype(np.float64))
    p = tmp_path / "sparse_in_dense_out.mat"
    save_params_final(P, p)

    th = _load(p)["theta"][0, 0]
    assert not sp.issparse(th)
    assert th.shape == (N, L) and th.dtype == np.float64
    assert np.array_equal(th, dense_theta.astype(np.float64))
    assert np.array_equal(load_group_prior(p)["theta"], dense_theta)


def test_sparse_input_matches_dense_input_byte_for_byte(
        tmp_path: Path) -> None:
    """Same bytes whichever container the caller supplies."""
    P = _params(seed=13)
    dense_theta = P["theta"].astype(np.float64)
    a = tmp_path / "from_dense.mat"
    b = tmp_path / "from_sparse.mat"
    save_params_final({**P, "theta": dense_theta}, a)
    save_params_final({**P, "theta": sp.csc_matrix(dense_theta)}, b)
    # bytes[128:] — the 128-byte MAT header carries a save timestamp.
    assert a.read_bytes()[128:] == b.read_bytes()[128:]


def test_parent_directory_is_created(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "Params_Final.mat"
    save_params_final(_params(), p)
    assert p.exists()


def test_dense_branch_matches_the_legacy_saver(tmp_path: Path) -> None:
    """Byte-for-byte against the pre-``_save.py`` inline implementation."""
    P = _params(seed=11)

    def _legacy(Params, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {}
        for k, v in Params.items():
            if isinstance(v, np.ndarray):
                arr = v
                if k == "mu" and arr.ndim == 2:
                    arr = np.ascontiguousarray(arr.T)
                out[k] = (arr.astype(np.float64, copy=False)
                          if arr.dtype.kind == "f" else arr)
            elif isinstance(v, (int, float, str)):
                out[k] = v
            elif isinstance(v, list):
                out[k] = np.asarray(v)
            else:
                out[k] = v
        sio.savemat(path, {"Params": out}, do_compression=True, format="5")

    a, b = tmp_path / "new.mat", tmp_path / "old.mat"
    save_params_final(P, a)
    _legacy(P, b)

    # The docstring's claim, actually checked: identical on disk, not just
    # equal once loaded (a value comparison survives a do_compression flip,
    # a field reorder, or a non-float dtype change). Compared from byte
    # 128 on — the MAT header carries a save timestamp.
    ba, bb = a.read_bytes()[128:], b.read_bytes()[128:]
    assert hashlib.md5(ba).hexdigest() == hashlib.md5(bb).hexdigest()
    assert ba == bb

    ma, mb = _load(a), _load(b)
    assert sorted(ma.dtype.names) == sorted(mb.dtype.names)
    for k in ma.dtype.names:
        va, vb = ma[k][0, 0], mb[k][0, 0]
        if isinstance(va, np.ndarray) and va.dtype.kind in "fiu":
            assert np.array_equal(va, vb), k
        else:
            assert str(va) == str(vb), k
