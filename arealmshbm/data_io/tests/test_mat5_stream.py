"""test_mat5_stream.py — the fast MAT-v5 walker vs ``scipy.io``.

Most assertions are "the fast path returns exactly what scipy returns":
same dtype, same shape, same bytes. Synthetic fixtures are written by
``scipy.io.savemat`` itself (compressed and uncompressed, dense fp64 /
fp32 / int32, MATLAB sparse, and a struct variable the walker must
name but must not try to materialise).

The one documented divergence — a dense variable whose payload MATLAB
narrowed (mxDOUBLE_CLASS carrying an ``miINT16`` ``pr``) comes back in
the mx CLASS dtype, i.e. ``loadmat(mat_dtype=True)`` semantics — is
pinned by ``test_narrowed_payload_is_widened_to_the_class_dtype``
against a hand-built container, since ``savemat`` never narrows.

``isal`` is optional: one test loads a private copy of the module with
``isal`` masked and one runs the whole fast path on stdlib ``zlib``,
asserting the same bytes come back.

An optional bench-backed test (``MSHBM_STEP2_BENCH_DIR``, default
``testdata/step2_bench``) checks the two real step-2 inputs — the
``group.mat`` whose 22 MB ``lambda`` the walker skips, and the
MATLAB-sparse ``spatial_mask_fsaverage6.mat``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from arealmshbm.data_io import mat5_stream as ms


BENCH_DIR = Path(os.environ.get("MSHBM_STEP2_BENCH_DIR", "testdata/step2_bench"))


def _fixture(path: Path, *, compress: bool) -> dict:
    rng = np.random.default_rng(11)
    data = {
        "a_f64": rng.standard_normal((7, 5)),
        "b_f32": rng.standard_normal((4, 9)).astype(np.float32),
        "c_i32": rng.integers(-1000, 1000, size=(3, 3)).astype(np.int32),
        "row": rng.standard_normal((1, 12)),
        "s_sparse": sp.csc_matrix(
            (rng.random((20, 6)) < 0.25) * rng.standard_normal((20, 6))),
        "st": {"x": np.arange(4.0).reshape(2, 2), "y": 3.0},
    }
    sio.savemat(path, data, do_compression=compress, format="5")
    return data


@pytest.mark.parametrize("compress", [True, False])
def test_walk_names_matches_scipy(tmp_path: Path, compress: bool) -> None:
    p = tmp_path / f"v5_{int(compress)}.mat"
    _fixture(p, compress=compress)
    names = ms.walk_names(p)
    assert ms.LAST_PATH == "fast"
    assert sorted(names) == sorted(nm for nm, _s, _d in sio.whosmat(str(p)))


@pytest.mark.parametrize("compress", [True, False])
def test_read_fields_bitwise_equal_to_scipy(tmp_path: Path,
                                            compress: bool) -> None:
    p = tmp_path / f"v5_{int(compress)}.mat"
    _fixture(p, compress=compress)
    ref = sio.loadmat(str(p), squeeze_me=False)

    want = {"a_f64", "b_f32", "c_i32", "row"}
    got = ms.read_fields(p, want)
    assert ms.LAST_PATH == "fast"
    assert set(got) == want
    for k in sorted(want):
        assert got[k].shape == ref[k].shape, k
        assert got[k].dtype == ref[k].dtype, k
        assert np.array_equal(got[k], ref[k]), k
        # bit-for-bit, not just ==: catches a -0.0 / NaN payload rewrite
        assert got[k].tobytes() == np.ascontiguousarray(ref[k]).tobytes(), k


@pytest.mark.parametrize("compress", [True, False])
def test_read_one_field_skips_the_others(tmp_path: Path,
                                         compress: bool) -> None:
    """Asking for one variable must not need any other to be readable."""
    p = tmp_path / f"one_{int(compress)}.mat"
    _fixture(p, compress=compress)
    got = ms.read_fields(p, {"b_f32"})
    assert ms.LAST_PATH == "fast"
    assert list(got) == ["b_f32"]


@pytest.mark.parametrize("compress", [True, False])
def test_read_sparse_matches_scipy(tmp_path: Path, compress: bool) -> None:
    p = tmp_path / f"sp_{int(compress)}.mat"
    _fixture(p, compress=compress)
    ref = sio.loadmat(str(p), squeeze_me=False)["s_sparse"]

    got = ms.read_sparse(p, "s_sparse")
    assert ms.LAST_PATH == "fast"
    assert sp.issparse(got)
    assert got.shape == ref.shape
    assert got.nnz == ref.nnz
    assert (got != sp.csc_matrix(ref)).nnz == 0
    assert np.array_equal(got.toarray(), np.asarray(ref.todense()))
    assert got.dtype == ref.dtype


# ``read_sparse`` must return the payload's OWN dtype, which is what
# ``scipy.io.loadmat`` gives for a sparse variable — and, unlike the
# dense reader, this walker does not widen a sparse payload to the mx
# class either, so the two agree here.
_SPARSE_DTYPE_CASES = {
    # name              matrix                                 scipy dtype
    "sp_f64":  (lambda A: sp.csc_matrix(A),                    np.float64),
    "sp_f32":  (lambda A: sp.csc_matrix(A.astype(np.float32)), np.float32),
    "sp_bool": (lambda A: sp.csc_matrix(A > 0),                np.uint8),
    "sp_i32":  (lambda A: sp.csc_matrix((A * 100).astype(np.int32)), np.int32),
}


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("case", sorted(_SPARSE_DTYPE_CASES))
def test_read_sparse_dtype_matches_scipy(tmp_path: Path, compress: bool,
                                         case: str) -> None:
    """logical- and single-payload sparse keep scipy's dtype, not fp64/bool."""
    make, want = _SPARSE_DTYPE_CASES[case]
    rng = np.random.default_rng(3)
    A = rng.random((9, 4))
    A[A < 0.6] = 0.0
    m = make(A)
    p = tmp_path / f"{case}_{int(compress)}.mat"
    sio.savemat(str(p), {"x": m}, do_compression=compress, format="5")

    ref = sio.loadmat(str(p), squeeze_me=False)["x"]
    got = ms.read_sparse(p, "x")
    assert ms.LAST_PATH == "fast"
    assert ref.dtype == want, f"scipy baseline moved for {case}"
    assert got.dtype == ref.dtype, case
    assert got.shape == ref.shape and got.nnz == ref.nnz
    assert np.array_equal(got.toarray(), np.asarray(sp.csc_matrix(ref).todense()))


@pytest.mark.parametrize("compress", [True, False])
def test_read_sparse_accepts_a_dense_variable(tmp_path: Path,
                                              compress: bool) -> None:
    p = tmp_path / f"dense_as_sparse_{int(compress)}.mat"
    data = _fixture(p, compress=compress)
    got = ms.read_sparse(p, "a_f64")
    assert ms.LAST_PATH == "fast"
    assert np.array_equal(got.toarray(), data["a_f64"])


def test_struct_variable_is_named_but_not_materialised(tmp_path: Path) -> None:
    p = tmp_path / "with_struct.mat"
    _fixture(p, compress=True)
    assert "st" in ms.walk_names(p)
    # A struct is not a numeric variable: the fast walk bails and the
    # scipy fallback answers instead (with scipy's own struct object).
    got = ms.read_fields(p, {"st"})
    assert ms.LAST_PATH == "scipy"
    assert got["st"].shape == (1, 1)


# ─────────────────────────────────────────────────────────────────────
# The documented dtype divergence: a narrowed payload
# ─────────────────────────────────────────────────────────────────────
def _tag(mi_type: int, payload: bytes) -> bytes:
    """One MAT-v5 data element: 8-byte tag + payload padded to 8."""
    pad = (-len(payload)) % 8
    return struct.pack("<II", mi_type, len(payload)) + payload + b"\x00" * pad


def _handmade_narrowed_double(path: Path, values: np.ndarray) -> None:
    """Write ``x`` as mxDOUBLE_CLASS whose ``pr`` element is ``miINT16``.

    MATLAB does this whenever the narrowing is lossless; ``savemat``
    never does, so the fixture has to be built by hand.
    """
    R, C = values.shape
    body = (
        _tag(6, struct.pack("<II", 6, 0))          # array flags, mxDOUBLE_CLASS
        + _tag(5, struct.pack("<ii", R, C))        # dims, miINT32
        + _tag(1, b"x")                            # name, miINT8
        + _tag(3, np.asfortranarray(values.astype(np.int16)).tobytes(order="F"))
    )
    header = (b" " * 116) + (b"\x00" * 8) + b"\x00\x01" + b"IM"
    path.write_bytes(header + struct.pack("<II", 14, len(body)) + body)


def test_narrowed_payload_is_widened_to_the_class_dtype(tmp_path: Path) -> None:
    """mxDOUBLE_CLASS + miINT16 ``pr`` → float64 here, int16 from loadmat.

    This is the module docstring's dtype contract: the fast walker
    presents the mx CLASS dtype (``mat_dtype=True`` semantics), default
    ``loadmat`` presents the payload dtype. Values agree; dtypes do not.
    """
    vals = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int16)
    p = tmp_path / "narrowed.mat"
    _handmade_narrowed_double(p, vals)

    got = ms.read_fields(p, {"x"})["x"]
    assert ms.LAST_PATH == "fast"
    assert got.dtype == np.float64
    assert got.shape == (2, 3)
    assert np.array_equal(got, vals.astype(np.float64))

    ref = sio.loadmat(str(p), squeeze_me=False)["x"]
    assert ref.dtype == np.int16                  # default loadmat does NOT widen
    assert np.array_equal(got, ref.astype(np.float64))

    wide = sio.loadmat(str(p), squeeze_me=False, mat_dtype=True)["x"]
    assert wide.dtype == got.dtype and np.array_equal(wide, got)


def test_missing_variable_raises_keyerror(tmp_path: Path) -> None:
    p = tmp_path / "missing.mat"
    _fixture(p, compress=True)
    with pytest.raises(KeyError):
        ms.read_fields(p, {"nope"})
    with pytest.raises(KeyError):
        ms.read_sparse(p, "nope")


def test_missing_file_raises(tmp_path: Path) -> None:
    for fn in (lambda: ms.walk_names(tmp_path / "x.mat"),
               lambda: ms.read_fields(tmp_path / "x.mat", {"a"}),
               lambda: ms.read_sparse(tmp_path / "x.mat", "a")):
        with pytest.raises(FileNotFoundError):
            fn()


def test_not_a_mat_file_falls_back(tmp_path: Path) -> None:
    p = tmp_path / "garbage.mat"
    p.write_bytes(b"\x00" * 4096)
    with pytest.raises(Exception):
        ms.read_fields(p, {"a"})
    assert ms.LAST_PATH == "scipy"


# ─────────────────────────────────────────────────────────────────────
# What falls back, and what does not
# ─────────────────────────────────────────────────────────────────────
def test_a_corrupt_zlib_payload_propagates(tmp_path: Path) -> None:
    """A decode error is NOT a "the walker does not handle this" verdict.

    Only ``_Unsupported`` routes to scipy. A bit-flip inside the zlib
    stream must reach the caller as the decompressor's own error, not as
    whatever unrelated thing ``loadmat`` says about the same bytes (and
    certainly not as a silent 40x-slower success).
    """
    p = tmp_path / "corrupt.mat"
    _fixture(p, compress=True)
    raw = bytearray(p.read_bytes())
    typ, _nb = struct.unpack("<II", raw[128:136])
    assert typ == 15                       # miCOMPRESSED
    raw[136] ^= 0xFF                       # the zlib header byte
    p.write_bytes(bytes(raw))

    ms.LAST_PATH = "sentinel"
    with pytest.raises(Exception) as exc:
        ms.read_fields(p, {"a_f64"})
    assert "zlib" in str(exc.value).lower() or "Error -" in str(exc.value)
    assert ms.LAST_PATH == "sentinel"      # never re-routed to scipy


def test_a_truncated_container_is_an_unsupported_verdict(tmp_path: Path) -> None:
    """Cut mid-element -> "runs past EOF", which IS a walker verdict.

    The walker cannot know whether it mis-parsed a layout it does not
    understand or the file is short, so this one legitimately falls back
    and lets scipy have the last word.
    """
    p = tmp_path / "short.mat"
    _fixture(p, compress=True)
    p.write_bytes(p.read_bytes()[:200])
    with pytest.raises(Exception):
        ms.read_fields(p, {"a_f64"})
    assert ms.LAST_PATH == "scipy"


def test_read_sparse_on_a_struct_falls_back(tmp_path: Path) -> None:
    """A valid layout the walker will not materialise still routes to scipy."""
    p = tmp_path / "struct_sparse.mat"
    _fixture(p, compress=True)
    with pytest.raises(Exception):
        ms.read_sparse(p, "st")
    assert ms.LAST_PATH == "scipy"


def test_empty_name_set_is_a_noop(tmp_path: Path) -> None:
    p = tmp_path / "v5.mat"
    _fixture(p, compress=True)
    assert ms.read_fields(p, set()) == {}


# ─────────────────────────────────────────────────────────────────────
# isal is optional: stdlib zlib is the fallback
# ─────────────────────────────────────────────────────────────────────
def test_module_falls_back_to_stdlib_zlib_without_isal(monkeypatch) -> None:
    """A machine with no ``isal`` must still import and read the fast path.

    Loaded as a private copy so the cached module (and the step-3 reader
    that imports its primitives) is untouched.
    """
    import importlib.util
    import sys
    import zlib

    monkeypatch.setitem(sys.modules, "isal", None)   # -> ImportError
    spec = importlib.util.spec_from_file_location("_mat5_no_isal", ms.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._zlib is zlib


@pytest.mark.parametrize("compress", [True, False])
def test_stdlib_zlib_gives_identical_values(tmp_path: Path,
                                            compress: bool,
                                            monkeypatch) -> None:
    """``zlib.decompressobj()`` is a drop-in for the walker's inflate."""
    import zlib

    p = tmp_path / f"nozlib_{int(compress)}.mat"
    _fixture(p, compress=compress)
    want = {"a_f64", "b_f32", "c_i32", "row"}
    ref = ms.read_fields(p, want)
    assert ms.LAST_PATH == "fast"
    ref_names = ms.walk_names(p)
    ref_sparse = ms.read_sparse(p, "s_sparse")

    monkeypatch.setattr(ms, "_zlib", zlib)
    got = ms.read_fields(p, want)
    assert ms.LAST_PATH == "fast"
    assert set(got) == want
    for k in sorted(want):
        assert got[k].dtype == ref[k].dtype, k
        assert got[k].tobytes() == ref[k].tobytes(), k
    assert ms.walk_names(p) == ref_names
    assert ms.LAST_PATH == "fast"
    got_sparse = ms.read_sparse(p, "s_sparse")
    assert ms.LAST_PATH == "fast"
    assert (got_sparse != ref_sparse).nnz == 0


# ─────────────────────────────────────────────────────────────────────
# Bench-backed: the two real step-2 inputs
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not (BENCH_DIR / "proj" / "group" / "group.mat").exists(),
                    reason="step-2 bench project not present")
def test_bench_group_mtc_equals_scipy() -> None:
    p = BENCH_DIR / "proj" / "group" / "group.mat"
    names = ms.walk_names(p)
    assert ms.LAST_PATH == "fast"
    assert {"mtc", "epsil", "lh_labels", "rh_labels"} <= set(names)
    got = ms.read_fields(p, {"mtc"})["mtc"]
    assert ms.LAST_PATH == "fast"
    ref = sio.loadmat(str(p))["mtc"]
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert np.array_equal(got, ref)


@pytest.mark.skipif(
    not (BENCH_DIR / "proj" / "spatial_mask"
         / "spatial_mask_fsaverage6.mat").exists(),
    reason="step-2 bench project not present")
def test_bench_spatial_mask_equals_scipy() -> None:
    p = BENCH_DIR / "proj" / "spatial_mask" / "spatial_mask_fsaverage6.mat"
    ref = sio.loadmat(str(p), squeeze_me=False)
    for nm in ("lh_boundary", "rh_boundary"):
        got = ms.read_sparse(p, nm)
        assert ms.LAST_PATH == "fast"
        r = sp.csc_matrix(ref[nm])
        assert got.shape == r.shape and got.nnz == r.nnz
        assert (got != r).nnz == 0
