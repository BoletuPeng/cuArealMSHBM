"""test_avg_profiles_from_packed_gpu.py — regression gate for the
memory-side GPU avg-profiles entry point.

What is pinned here:

  * the device packed-accumulator is bit-identical (``np.array_equal``)
    to the numba CPU kernel for host *and* cupy inputs, across several
    subjects/sessions and for D values that are and are not multiples
    of 8 (padding bits in the trailing byte);
  * ``avg_profiles_from_packed_gpu`` produces the same fp32 means as
    the CPU supercall's arithmetic, and the ``.npy`` pair it writes is
    byte-for-byte what ``np.save`` writes for those arrays;
  * ``AvgProfilesResult.lh_avg`` (pinned host buffer) and
    ``lh_avg_dev`` (device) agree bit-for-bit;
  * ``writer.wait()`` is idempotent and the files exist afterwards;
  * every documented precondition raises a named ``ValueError``.

Run::

    python -m pytest arealmshbm/avg_profiles/tests/test_avg_profiles_from_packed_gpu.py -v

Skips if cupy is unavailable; the supercall-level cases additionally
skip when the shipped ``avg_mesh`` bundle is not staged in this
checkout (see ``arealmshbm/data/README.md``).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from arealmshbm.avg_profiles._kernels import (
    _accum_packed_session_inplace_kernel, _scale_inplace_kernel,
)
from arealmshbm.avg_profiles.avg_profiles import AvgProfilesResult
from arealmshbm.avg_profiles.avg_profiles_gpu import (
    _accum_packed_session_device,
    _hemi_vert_counts,
    _pinned_host_buffer,
    avg_profiles_from_packed_gpu,
)

TARG, SEED = "fsaverage6", "fsaverage3"


def _mesh_or_skip():
    """(V_lh, V_rh) for the test mesh, or skip when unstaged."""
    try:
        return _hemi_vert_counts(TARG)
    except FileNotFoundError as exc:      # pragma: no cover - env-dependent
        pytest.skip(f"avg_mesh bundle not staged: {exc}")


def _random_packed(rng, T, N, D):
    """Random binary ``(T, N, D)`` + its LSB-first packed bytes."""
    bits = (rng.random((T, N, D)) < 0.3).astype(np.uint8)
    packed = np.ascontiguousarray(
        np.packbits(bits, axis=-1, bitorder="little")
    )
    return bits, packed


def _cpu_reference(packed_subjects, D, V_lh, V_rh):
    """CPU supercall arithmetic: numba accumulate + scale."""
    lh = np.zeros(V_lh * D, dtype=np.float32)
    rh = np.zeros(V_rh * D, dtype=np.float32)
    n = 0
    for arr in packed_subjects:
        for t in range(arr.shape[0]):
            slab = arr[t]
            _accum_packed_session_inplace_kernel(lh, slab[:V_lh], D)
            _accum_packed_session_inplace_kernel(rh, slab[V_lh:], D)
            n += 1
    _scale_inplace_kernel(lh, np.float32(1.0 / n))
    _scale_inplace_kernel(rh, np.float32(1.0 / n))
    return lh.reshape(V_lh, D), rh.reshape(V_rh, D)


# ─────────────────────────────────────────────────────────────────────
# Kernel level — no mesh needed
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("D", [8, 13, 64, 65, 1175])
@pytest.mark.parametrize("V_h", [1, 7, 300])
def test_device_accum_matches_numba(D, V_h):
    rng = np.random.default_rng(1234 + D * 31 + V_h)
    bits, packed = _random_packed(rng, 3, V_h, D)

    ref = np.zeros(V_h * D, dtype=np.float32)
    for t in range(3):
        _accum_packed_session_inplace_kernel(ref, packed[t], D)

    acc = cp.zeros((V_h, D), dtype=cp.float32)
    packed_dev = cp.asarray(packed)
    for t in range(3):
        _accum_packed_session_device(packed_dev[t], acc, D)

    assert np.array_equal(cp.asnumpy(acc).ravel(), ref)
    # and against the plain unpacked sum, to catch a shared packing bug
    assert np.array_equal(cp.asnumpy(acc), bits.sum(axis=0).astype(np.float32))


def test_device_accum_rejects_bad_inputs():
    acc = cp.zeros((4, 16), dtype=cp.float32)
    good = cp.zeros((4, 2), dtype=cp.uint8)
    with pytest.raises(ValueError, match="must be uint8"):
        _accum_packed_session_device(cp.zeros((4, 2), cp.float32), acc, 16)
    with pytest.raises(ValueError, match="must be fp32"):
        _accum_packed_session_device(good, cp.zeros((4, 16), cp.float64), 16)
    with pytest.raises(ValueError, match="V_h mismatch"):
        _accum_packed_session_device(cp.zeros((5, 2), cp.uint8), acc, 16)
    with pytest.raises(ValueError, match="D mismatch"):
        _accum_packed_session_device(good, acc, 15)


def test_pinned_host_buffer_shape_and_dtype():
    buf = _pinned_host_buffer((3, 5), np.float32)
    assert buf.shape == (3, 5)
    assert buf.dtype == np.float32
    assert buf.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="must be 2-D"):
        _pinned_host_buffer((3,), np.float32)


# ─────────────────────────────────────────────────────────────────────
# Supercall level — needs the mesh bundle
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("D", [13, 32])
@pytest.mark.parametrize("n_sub", [1, 3])
def test_from_packed_matches_cpu_and_npy_bytes(tmp_path, D, n_sub):
    V_lh, V_rh = _mesh_or_skip()
    N = V_lh + V_rh
    rng = np.random.default_rng(99 + D + n_sub)
    subs = [_random_packed(rng, 2, N, D)[1] for _ in range(n_sub)]

    ref_lh, ref_rh = _cpu_reference(subs, D, V_lh, V_rh)

    res = avg_profiles_from_packed_gpu(subs, D, TARG, SEED, tmp_path, save=True)
    assert isinstance(res, AvgProfilesResult)
    assert np.array_equal(res.lh_avg, ref_lh)
    assert np.array_equal(res.rh_avg, ref_rh)
    # host (pinned D2H buffer) and device copies agree bit-for-bit
    assert np.array_equal(res.lh_avg, cp.asnumpy(res.lh_avg_dev))
    assert np.array_equal(res.rh_avg, cp.asnumpy(res.rh_avg_dev))

    assert res.writer.wait() == (res.lh_path, res.rh_path)
    res.writer.wait()          # idempotent
    assert res.writer.exception is None
    assert res.lh_path.exists() and res.rh_path.exists()
    assert np.array_equal(np.load(res.lh_path), ref_lh)
    assert np.array_equal(np.load(res.rh_path), ref_rh)

    # the background write must be byte-identical to a plain np.save
    expect = tmp_path / "expect.npy"
    np.save(expect, ref_lh, allow_pickle=False)
    assert expect.read_bytes() == res.lh_path.read_bytes()


def test_from_packed_accepts_device_input(tmp_path):
    V_lh, V_rh = _mesh_or_skip()
    rng = np.random.default_rng(5)
    D = 17
    host = _random_packed(rng, 2, V_lh + V_rh, D)[1]

    a = avg_profiles_from_packed_gpu([host], D, TARG, SEED, tmp_path, save=False)
    b = avg_profiles_from_packed_gpu(
        [cp.asarray(host)], D, TARG, SEED, tmp_path, save=False,
    )
    assert np.array_equal(a.lh_avg, b.lh_avg)
    assert np.array_equal(a.rh_avg, b.rh_avg)
    assert a.writer is None and b.writer is None
    # save=False writes nothing but still names the canonical paths
    assert not a.lh_path.exists()
    assert a.lh_path.name == f"lh_{TARG}_roi{SEED}_avg_profile.npy"


def test_from_packed_validation(tmp_path):
    V_lh, V_rh = _mesh_or_skip()
    N = V_lh + V_rh
    good = np.zeros((2, N, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="packed_subjects is empty"):
        avg_profiles_from_packed_gpu([], 13, TARG, SEED, tmp_path, save=False)
    with pytest.raises(ValueError, match="D must be > 0"):
        avg_profiles_from_packed_gpu([good], 0, TARG, SEED, tmp_path, save=False)
    with pytest.raises(ValueError, match="only fsaverage"):
        avg_profiles_from_packed_gpu([good], 13, "fs_LR_32k", SEED, tmp_path,
                                     save=False)
    with pytest.raises(ValueError, match="must be uint8"):
        avg_profiles_from_packed_gpu([good.astype(np.float32)], 13, TARG, SEED,
                                     tmp_path, save=False)
    with pytest.raises(ValueError, match="must be 3-D"):
        avg_profiles_from_packed_gpu([good[0]], 13, TARG, SEED, tmp_path,
                                     save=False)
    with pytest.raises(ValueError, match=r"but the mesh yields"):
        avg_profiles_from_packed_gpu([np.zeros((2, N + 1, 2), np.uint8)], 13,
                                     TARG, SEED, tmp_path, save=False)
    with pytest.raises(ValueError, match=r"!= ceil\(D/8\)"):
        avg_profiles_from_packed_gpu([np.zeros((2, N, 3), np.uint8)], 13,
                                     TARG, SEED, tmp_path, save=False)
    with pytest.raises(ValueError, match="subject 1 has T="):
        avg_profiles_from_packed_gpu(
            [good, np.zeros((3, N, 2), np.uint8)], 13, TARG, SEED, tmp_path,
            save=False,
        )
