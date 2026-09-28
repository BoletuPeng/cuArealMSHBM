"""test_avg_profiles_accumulator_gpu.py — regression gate for the device
accumulator behind step 1's GPU average.

What is pinned here:

  * the device packed-accumulate kernel is bit-identical
    (``np.array_equal``) to the numba CPU kernel, across several
    subjects/sessions and for D values that are and are not multiples
    of 8 (padding bits in the trailing byte);
  * ``PackedProfileAccumulator`` fed one device session slab at a time
    (the order the fused leaf feeds it) + ``avg_profiles_from_accumulator``
    produce the same fp32 means as the CPU supercall's arithmetic, and
    the ``.npy`` pair it writes is byte-for-byte what ``np.save`` writes
    for those arrays;
  * ``AvgProfilesResult.lh_avg`` (pinned host buffer) and ``lh_avg_dev``
    (device) agree bit-for-bit, and the device means ARE the
    accumulator's own arrays;
  * the disk-reading ``avg_profiles(backend='gpu')`` — the same
    accumulator fed from decoded ``.b2nd`` files — matches and drops
    the device fields;
  * ``writer.wait()`` is idempotent and the files exist afterwards;
  * the mean is taken once: a second finalize raises and leaves the
    means alone, and ``add`` refuses further sessions;
  * sessions arrive per subject in slot order, and the finalize refuses
    an incomplete subject;
  * the zero-fill is complete before a feeder on another (non-blocking)
    stream can fold a session in;
  * every documented precondition raises a named error.

Run::

    python -m pytest arealmshbm/avg_profiles/tests/test_avg_profiles_accumulator_gpu.py -v

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
from arealmshbm.avg_profiles.avg_profiles import (
    AvgProfilesResult, avg_profiles,
)
from arealmshbm.avg_profiles.avg_profiles_gpu import (
    PackedProfileAccumulator,
    _accum_packed_session_device,
    _hemi_vert_counts,
    _pinned_host_buffer,
    avg_profiles_from_accumulator,
)
from arealmshbm.data_io.profile_io import (
    profile_path, write_subject_profile_tnd,
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


def _accumulate(packed_subjects, V_lh, V_rh, D):
    """Feed the accumulator the way the fused leaf does: one device
    session slab at a time, in (subject, session) order."""
    acc = PackedProfileAccumulator(V_lh, V_rh, D,
                                   int(packed_subjects[0].shape[0]))
    for arr in packed_subjects:
        dev = cp.asarray(arr)
        for t in range(arr.shape[0]):
            acc.add(t, dev[t])
    return acc


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
    with pytest.raises(ValueError, match="C-contiguous"):
        _accum_packed_session_device(cp.zeros((4, 4), cp.uint8)[:, ::2],
                                     acc, 16)
    with pytest.raises(ValueError, match="C-contiguous"):
        _accum_packed_session_device(good, cp.zeros((4, 32), cp.float32)[:, ::2],
                                     16)


def test_pinned_host_buffer_shape_and_dtype():
    buf = _pinned_host_buffer((3, 5), np.float32)
    assert buf.shape == (3, 5)
    assert buf.dtype == np.float32
    assert buf.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="must be 2-D"):
        _pinned_host_buffer((3,), np.float32)


@pytest.mark.parametrize("D", [13, 64])
def test_accumulator_sums_match_numba_per_session(D):
    """Session-at-a-time device folds == the CPU kernel's running sum,
    for a split-hemi slab (the leaf's joint lh|rh layout)."""
    V_lh, V_rh = 5, 9
    N = V_lh + V_rh
    rng = np.random.default_rng(3 + D)
    subs = [_random_packed(rng, 2, N, D)[1] for _ in range(3)]

    acc = _accumulate(subs, V_lh, V_rh, D)
    assert acc.n_pairs == 6

    lh = np.zeros(V_lh * D, dtype=np.float32)
    rh = np.zeros(V_rh * D, dtype=np.float32)
    for arr in subs:
        for t in range(2):
            _accum_packed_session_inplace_kernel(lh, arr[t][:V_lh], D)
            _accum_packed_session_inplace_kernel(rh, arr[t][V_lh:], D)
    assert np.array_equal(cp.asnumpy(acc.lh).ravel(), lh)
    assert np.array_equal(cp.asnumpy(acc.rh).ravel(), rh)


def test_accumulator_add_validation():
    V_lh, V_rh, D = 3, 4, 13
    N = V_lh + V_rh
    with pytest.raises(ValueError, match="must all be positive"):
        PackedProfileAccumulator(V_lh, V_rh, 0, 1)
    with pytest.raises(ValueError, match="must all be positive"):
        PackedProfileAccumulator(V_lh, V_rh, D, 0)
    acc = PackedProfileAccumulator(V_lh, V_rh, D, 2)
    assert acc.D_bytes == 2
    good = cp.zeros((N, 2), dtype=cp.uint8)
    with pytest.raises(ValueError, match="must be a cupy ndarray"):
        acc.add(0, np.zeros((N, 2), np.uint8))
    with pytest.raises(ValueError, match="slab shape"):
        acc.add(0, cp.zeros((N + 1, 2), cp.uint8))
    with pytest.raises(ValueError, match="D_bytes"):
        acc.add(0, cp.zeros((N, 3), cp.uint8))
    with pytest.raises(ValueError, match="must be uint8"):
        acc.add(0, good.astype(cp.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        acc.add(0, cp.zeros((N, 4), cp.uint8)[:, ::2])
    # Sessions arrive per subject in slot order 0..num_sess-1.
    with pytest.raises(ValueError, match="slot 1 arrived where 0 was due"):
        acc.add(1, good)
    assert acc.n_pairs == 0 and acc.n_subjects == 0
    acc.add(0, good)
    with pytest.raises(ValueError, match="slot 0 arrived where 1 was due"):
        acc.add(0, good)
    assert acc.n_pairs == 1 and acc.n_subjects == 0
    acc.add(1, good)
    assert acc.n_pairs == 2 and acc.n_subjects == 1
    # Nothing was folded in by the refused calls.
    assert np.array_equal(cp.asnumpy(acc.lh), np.zeros((V_lh, D), np.float32))


_SPIN_SRC = r"""
extern "C" __global__ void _test_spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
"""


def test_zero_fill_lands_before_a_feeder_on_another_stream_adds():
    """``cp.zeros`` is an asynchronous memset on the constructing stream
    and the leaf folds sessions in on its own non-blocking compute
    stream. Built while the null stream is busy, the accumulator must
    not let a fold on a non-blocking stream land before its zeros do --
    the memset would wipe the fold."""
    V_lh, V_rh, D = 64, 64, 64
    N = V_lh + V_rh
    ones = cp.full((N, D // 8), 0xFF, dtype=cp.uint8)     # every bit set
    spin = cp.RawKernel(_SPIN_SRC, "_test_spin")
    spin((1,), (32,), (np.int64(1),))                      # compile
    # Warm the pool with this accumulator's block sizes and compile the
    # fold kernel: a pool miss (cudaMalloc) or an NVRTC compile would
    # drain the queued spin before the memset ever sat behind it.
    warm = PackedProfileAccumulator(V_lh, V_rh, D, 1)
    warm.add(0, ones)
    del warm
    cp.cuda.Device().synchronize()

    spin((1,), (32,), (np.int64(300_000_000),))           # ~0.1 s, null stream
    acc = PackedProfileAccumulator(V_lh, V_rh, D, 1)       # zeros queue behind it
    with cp.cuda.Stream(non_blocking=True) as s:
        acc.add(0, ones)
        s.synchronize()
    cp.cuda.Device().synchronize()
    assert np.array_equal(cp.asnumpy(acc.lh), np.ones((V_lh, D), np.float32))
    assert np.array_equal(cp.asnumpy(acc.rh), np.ones((V_rh, D), np.float32))


# ─────────────────────────────────────────────────────────────────────
# Supercall level — needs the mesh bundle
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("D", [13, 32])
@pytest.mark.parametrize("n_sub", [1, 3])
def test_from_accumulator_matches_cpu_and_npy_bytes(tmp_path, D, n_sub):
    V_lh, V_rh = _mesh_or_skip()
    N = V_lh + V_rh
    rng = np.random.default_rng(99 + D + n_sub)
    subs = [_random_packed(rng, 2, N, D)[1] for _ in range(n_sub)]

    ref_lh, ref_rh = _cpu_reference(subs, D, V_lh, V_rh)

    acc = _accumulate(subs, V_lh, V_rh, D)
    assert acc.n_pairs == 2 * n_sub
    res = avg_profiles_from_accumulator(acc, TARG, SEED, tmp_path, save=True)
    assert isinstance(res, AvgProfilesResult)
    assert np.array_equal(res.lh_avg, ref_lh)
    assert np.array_equal(res.rh_avg, ref_rh)
    # The device means are the accumulator's own arrays (scaled in
    # place), and the host (pinned D2H buffer) copies agree bit-for-bit.
    assert res.lh_avg_dev is acc.lh and res.rh_avg_dev is acc.rh
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

    # the mean was taken: nothing can be folded in afterwards, and a
    # second finalize (a retry) must not divide the means by n again
    with pytest.raises(RuntimeError, match="already taken"):
        acc.add(0, cp.asarray(subs[0][0]))
    with pytest.raises(RuntimeError, match="already taken"):
        avg_profiles_from_accumulator(acc, TARG, SEED, tmp_path, save=False)
    assert np.array_equal(cp.asnumpy(acc.lh), ref_lh)
    assert np.array_equal(cp.asnumpy(acc.rh), ref_rh)


def test_disk_reading_gpu_supercall_matches_the_accumulator(tmp_path):
    """``avg_profiles(backend='gpu')`` decodes the ``.b2nd`` set into the
    same accumulator: same means as the CPU arithmetic, device fields
    dropped, files joined before it returns."""
    V_lh, V_rh = _mesh_or_skip()
    N, D, T = V_lh + V_rh, 13, 2
    rng = np.random.default_rng(7)
    subs = []
    for s in (1, 2):
        bits, packed = _random_packed(rng, T, N, D)
        write_subject_profile_tnd(
            profile_path(tmp_path, str(s), TARG, SEED), bits)
        subs.append(packed)
    ref_lh, ref_rh = _cpu_reference(subs, D, V_lh, V_rh)

    res = avg_profiles(seed_mesh=SEED, targ_mesh=TARG, out_dir=tmp_path,
                       num_sub=2, num_sess=T, verbose=False, backend="gpu")
    assert res.lh_avg_dev is None and res.rh_avg_dev is None
    assert np.array_equal(res.lh_avg, ref_lh)
    assert np.array_equal(res.rh_avg, ref_rh)
    assert np.array_equal(np.load(res.lh_path), ref_lh)
    assert np.array_equal(np.load(res.rh_path), ref_rh)


def test_from_accumulator_validation(tmp_path):
    V_lh, V_rh = _mesh_or_skip()
    N = V_lh + V_rh
    acc = PackedProfileAccumulator(V_lh, V_rh, 13, 2)
    with pytest.raises(ValueError, match="no session was accumulated"):
        avg_profiles_from_accumulator(acc, TARG, SEED, tmp_path, save=False)
    acc.add(0, cp.zeros((N, 2), dtype=cp.uint8))
    with pytest.raises(ValueError, match="only fsaverage"):
        avg_profiles_from_accumulator(acc, "fs_LR_32k", SEED, tmp_path,
                                      save=False)
    other = PackedProfileAccumulator(V_lh + 1, V_rh, 13, 1)
    other.add(0, cp.zeros((N + 1, 2), dtype=cp.uint8))
    with pytest.raises(ValueError, match="hemi sizes"):
        avg_profiles_from_accumulator(other, TARG, SEED, tmp_path, save=False)
    # one of two sessions folded in: the subject is incomplete
    with pytest.raises(ValueError, match="last subject is incomplete"):
        avg_profiles_from_accumulator(acc, TARG, SEED, tmp_path, save=False)
    acc.add(1, cp.zeros((N, 2), dtype=cp.uint8))

    # save=False writes nothing but still names the canonical paths
    res = avg_profiles_from_accumulator(acc, TARG, SEED, tmp_path, save=False)
    assert res.writer is None
    assert not res.lh_path.exists()
    assert res.lh_path.name == f"lh_{TARG}_roi{SEED}_avg_profile.npy"
    assert np.array_equal(res.lh_avg, np.zeros((V_lh, 13), np.float32))
