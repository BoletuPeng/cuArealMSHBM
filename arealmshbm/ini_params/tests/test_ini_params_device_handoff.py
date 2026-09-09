"""test_ini_params_device_handoff.py — supercall-level contract for the
device-resident ini_params hand-off.

Pins:

  * ``precomputed_{lh,rh}_avg_dev`` (cupy) produces bit-identical
    ``mtc`` / ``epsil`` / ``lambda`` / labels to the host-array
    hand-off and to the ``.npy`` disk read — the device concat + fp64
    widen is a pure relocation of the same arithmetic;
  * ``save_async=True`` writes the same ``group.mat`` as the sync path
    and the file loads through ``step2_io.load_group_mtc``;
  * the CPU backend still accepts ``reduction_dtype=np.float32``, and
    the GPU backend rejects it with a named error instead of failing
    deep inside a RawKernel;
  * the device/host hand-off argument pairing is validated.

Run::

    python -m pytest arealmshbm/ini_params/tests/test_ini_params_device_handoff.py -v

Skips if cupy is unavailable or the shipped ``avg_mesh`` bundle is not
staged in this checkout (see ``arealmshbm/data/README.md``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.ini_params import generate_ini_params
from arealmshbm.ini_params.ini_params import _load_medial_mask
from arealmshbm.ini_params.ini_params_gpu import generate_ini_params_gpu
from arealmshbm.step2_io.load_group_mtc import load_group_mtc

TARG, SEED = "fsaverage6", "fsaverage3"
D = 24
L_H = 6


@pytest.fixture(scope="module")
def fixture_case():
    """Synthetic (labels, lh/rh avg profiles) on the real fsa6 mesh."""
    try:
        mw = _load_medial_mask(TARG)
    except FileNotFoundError as exc:      # pragma: no cover - env-dependent
        pytest.skip(f"avg_mesh bundle not staged: {exc}")
    n = mw.shape[0] // 2
    rng = np.random.default_rng(2026)
    lh_labels = rng.integers(1, L_H + 1, size=n).astype(np.int64)
    rh_labels = rng.integers(1, L_H + 1, size=n).astype(np.int64)
    lh_avg = rng.random((n, D), dtype=np.float32)
    rh_avg = rng.random((n, D), dtype=np.float32)
    # a handful of genuinely all-zero (non-MW) rows -> keep_mask False
    lh_avg[5:9] = 0.0
    return lh_labels, rh_labels, lh_avg, rh_avg


def _dev(a):
    return cp.asarray(a)


def _assert_same(a, b, tag):
    assert np.array_equal(a["mtc"], b["mtc"]), f"{tag}: mtc"
    assert float(a["epsil"].ravel()[0]) == float(b["epsil"].ravel()[0]), \
        f"{tag}: epsil"
    assert np.array_equal(a["lambda"], b["lambda"]), f"{tag}: lambda"
    assert np.array_equal(a["lh_labels"], b["lh_labels"]), f"{tag}: lh_labels"
    assert np.array_equal(a["rh_labels"], b["rh_labels"]), f"{tag}: rh_labels"


def test_device_handoff_matches_host_and_disk(tmp_path, fixture_case):
    lh_labels, rh_labels, lh_avg, rh_avg = fixture_case

    host = generate_ini_params_gpu(
        SEED, TARG, lh_labels, rh_labels, str(tmp_path),
        precomputed_lh_avg=lh_avg, precomputed_rh_avg=rh_avg, save=False,
    )
    dev = generate_ini_params_gpu(
        SEED, TARG, lh_labels, rh_labels, str(tmp_path),
        precomputed_lh_avg_dev=_dev(lh_avg), precomputed_rh_avg_dev=_dev(rh_avg),
        save=False,
    )
    _assert_same(dev, host, "device vs host hand-off")

    base = tmp_path / "profiles" / "avg_profile"
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / f"lh_{TARG}_roi{SEED}_avg_profile.npy", lh_avg)
    np.save(base / f"rh_{TARG}_roi{SEED}_avg_profile.npy", rh_avg)
    disk = generate_ini_params_gpu(
        SEED, TARG, lh_labels, rh_labels, str(tmp_path), save=False,
    )
    _assert_same(dev, disk, "device vs .npy read")


def test_device_handoff_is_deterministic(tmp_path, fixture_case):
    lh_labels, rh_labels, lh_avg, rh_avg = fixture_case
    kw = dict(precomputed_lh_avg_dev=_dev(lh_avg),
              precomputed_rh_avg_dev=_dev(rh_avg), save=False)
    a = generate_ini_params_gpu(SEED, TARG, lh_labels, rh_labels,
                                str(tmp_path), **kw)
    b = generate_ini_params_gpu(SEED, TARG, lh_labels, rh_labels,
                                str(tmp_path), **kw)
    _assert_same(a, b, "run-to-run")


def test_async_save_matches_sync_save(tmp_path, fixture_case):
    lh_labels, rh_labels, lh_avg, rh_avg = fixture_case
    sync_dir, async_dir = tmp_path / "s", tmp_path / "a"
    sync_dir.mkdir(); async_dir.mkdir()

    s = generate_ini_params_gpu(
        SEED, TARG, lh_labels, rh_labels, str(sync_dir),
        precomputed_lh_avg_dev=_dev(lh_avg), precomputed_rh_avg_dev=_dev(rh_avg),
        save=True, save_async=False,
    )
    assert s.writer is None
    a = generate_ini_params_gpu(
        SEED, TARG, lh_labels, rh_labels, str(async_dir),
        precomputed_lh_avg_dev=_dev(lh_avg), precomputed_rh_avg_dev=_dev(rh_avg),
        save=True, save_async=True,
    )
    assert a.writer is not None
    a.writer.wait()
    _assert_same(a, s, "async vs sync result")
    # bytes[128:] - the MAT-5 header's first 128 bytes carry a write
    # timestamp, so only the payload can be compared.
    sync_mat = (sync_dir / "group" / "group.mat").read_bytes()
    async_mat = (async_dir / "group" / "group.mat").read_bytes()
    assert sync_mat[128:] == async_mat[128:]

    got = load_group_mtc(async_dir / "group" / "group.mat")
    assert np.array_equal(got["mtc"], a["mtc"])
    assert got["epsil"] == float(a["epsil"].ravel()[0])


def test_fp32_reduction_dtype(tmp_path, fixture_case):
    """CPU backend must still accept an fp32 reduction; GPU must refuse
    it up front rather than tripping a RawKernel dtype guard."""
    lh_labels, rh_labels, lh_avg, rh_avg = fixture_case
    out = generate_ini_params(
        SEED, TARG, lh_labels, rh_labels, str(tmp_path), backend="cpu",
        reduction_dtype=np.float32,
        precomputed_lh_avg=lh_avg, precomputed_rh_avg=rh_avg, save=False,
    )
    assert out["mtc"].dtype == np.float64      # cast on the way out
    assert np.isfinite(float(out["epsil"].ravel()[0]))

    with pytest.raises(ValueError, match="reduction_dtype must be float64"):
        generate_ini_params(
            SEED, TARG, lh_labels, rh_labels, str(tmp_path), backend="gpu",
            reduction_dtype=np.float32,
            precomputed_lh_avg=lh_avg, precomputed_rh_avg=rh_avg, save=False,
        )


def test_handoff_argument_validation(tmp_path, fixture_case):
    lh_labels, rh_labels, lh_avg, rh_avg = fixture_case
    common = (SEED, TARG, lh_labels, rh_labels, str(tmp_path))

    with pytest.raises(ValueError, match="BOTH precomputed_lh_avg_dev"):
        generate_ini_params_gpu(*common, precomputed_lh_avg_dev=_dev(lh_avg),
                                save=False)
    with pytest.raises(ValueError, match="not both"):
        generate_ini_params_gpu(
            *common, precomputed_lh_avg=lh_avg, precomputed_rh_avg=rh_avg,
            precomputed_lh_avg_dev=_dev(lh_avg),
            precomputed_rh_avg_dev=_dev(rh_avg), save=False,
        )
    with pytest.raises(ValueError, match="must be a cupy ndarray"):
        generate_ini_params_gpu(*common, precomputed_lh_avg_dev=lh_avg,
                                precomputed_rh_avg_dev=rh_avg, save=False)
    with pytest.raises(ValueError, match="labels imply V_lh"):
        generate_ini_params_gpu(
            *common, precomputed_lh_avg_dev=_dev(lh_avg[:-1]),
            precomputed_rh_avg_dev=_dev(rh_avg), save=False,
        )
    with pytest.raises(ValueError, match="only accepted by backend='gpu'"):
        generate_ini_params(*common, backend="cpu",
                            precomputed_lh_avg_dev=_dev(lh_avg),
                            precomputed_rh_avg_dev=_dev(rh_avg), save=False)
