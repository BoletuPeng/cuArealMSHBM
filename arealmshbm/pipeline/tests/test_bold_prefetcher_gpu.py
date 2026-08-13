"""test_bold_prefetcher_gpu.py — GPU backend smoke + bit-equality
for ``Step0BoldPrefetcher`` and ``Step1BoldPrefetcher`` (issue #55).

The prefetcher modules import cupy lazily (only when a 'gpu' worker
actually runs), so these tests skip cleanly on CPU-only envs. The
unit-level kernel bit-equality (``concat_hemis_drop_medial_gpu`` vs
CPU) lives in ``arealmshbm/bold_io/tests/test_bold_io_gpu.py`` — here
we pin the prefetcher's end-to-end contract:

  * ``backend='gpu'`` returns device cupy buffers on the current
    cupy device.
  * Output is bit-equal to the CPU prefetcher run on the same
    synthetic GIFTI inputs (so a future device-driver / nvCOMP update
    can't silently drift the numerical output).
  * Bad backend strings raise with a named error.

Synthetic GIFTI generation is shared with
``arealmshbm/data_io/tests/test_gifti_io.py``; we re-implement the
minimal version inline to keep this file independent.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("nvidia.nvcomp")  # GPU reader's required nvCOMP

from arealmshbm.pipeline._step0_bold_prefetcher import Step0BoldPrefetcher
from arealmshbm.pipeline._step1_bold_prefetcher import Step1BoldPrefetcher


def _synth_gifti_bytes(N: int = 32, T: int = 4, seed: int = 0) -> bytes:
    """Minimal GIFTI: T DataArrays, each Dim0=N fp32. Payload is
    deterministic on (seed, t, i) so the per-sub/per-sess synthetic
    files are unique and reproducible.
    """
    rng = np.random.default_rng(seed)
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>\n',
             b'<GIFTI Version="1.0" NumberOfDataArrays="',
             str(T).encode(), b'">\n']
    for t in range(T):
        arr = rng.standard_normal(N).astype(np.float32)
        payload = base64.b64encode(zlib.compress(arr.tobytes())).decode("ascii")
        parts.append(
            f'<DataArray DataType="NIFTI_TYPE_FLOAT32" '
            f'ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" '
            f'Encoding="GZipBase64Binary" Endian="LittleEndian" '
            f'ExternalFileName="" ExternalFileOffset="0" '
            f'Dim0="{N}"><Data>'.encode()
        )
        parts.append(payload.encode("ascii"))
        parts.append(b'</Data></DataArray>\n')
    parts.append(b'</GIFTI>\n')
    return b"".join(parts)


def _write_synth_pair(tmp_path: Path, name: str, N: int, T: int, seed: int):
    """Write a per-hemi GIFTI pair to disk; return (lh_path, rh_path)."""
    lh = tmp_path / f"{name}_lh.func.gii"
    rh = tmp_path / f"{name}_rh.func.gii"
    lh.write_bytes(_synth_gifti_bytes(N=N, T=T, seed=seed))
    rh.write_bytes(_synth_gifti_bytes(N=N, T=T, seed=seed + 1))
    return lh, rh


# ─────────────────────────────────────────────────────────────────────
# Step0BoldPrefetcher — GPU backend
# ─────────────────────────────────────────────────────────────────────
def test_step0_prefetcher_gpu_returns_cupy_on_current_device(tmp_path):
    """A primed (sub_id, sess_idx) under backend='gpu' returns a cupy
    ndarray on the current device — not host numpy. The H2D is gone."""
    n_h = 24
    T = 3
    lh, rh = _write_synth_pair(tmp_path, "s0", N=n_h, T=T, seed=10)
    # Drop the second half of vertices as "medial" so the post-drop
    # row count is exactly n_h on each hemi.
    mask = np.zeros(2 * n_h, dtype=np.uint8)
    mask[n_h // 2:n_h] = 1   # drop second half of lh
    mask[n_h + n_h // 2:] = 1  # drop second half of rh
    expected_rows = 2 * n_h - int(mask.sum())

    with Step0BoldPrefetcher(
        medial_mask=mask, n_lh=n_h, n_rh=n_h, backend="gpu",
    ) as p:
        p.prime_subject("sub-001", [(lh, rh)])
        prov = p.get_provider("sub-001")
        out = prov(1)
        assert isinstance(out, cp.ndarray)
        assert int(out.device.id) == int(cp.cuda.runtime.getDevice())
        assert out.shape == (expected_rows, T)
        assert out.dtype == cp.float32


def test_step0_prefetcher_gpu_bit_equal_to_cpu(tmp_path):
    """Same synthetic input through backend='cpu' and backend='gpu'
    produces bit-equal numerical output. This catches any silent drift
    between stock zlib and nvCOMP Deflate, and any divergence in
    concat_hemis_drop_medial vs concat_hemis_drop_medial_gpu."""
    n_h = 48
    T = 5
    # Two synthetic sessions, distinct seeds.
    s1_lh, s1_rh = _write_synth_pair(tmp_path, "s1", n_h, T, seed=100)
    s2_lh, s2_rh = _write_synth_pair(tmp_path, "s2", n_h, T, seed=200)
    mask = np.zeros(2 * n_h, dtype=np.uint8)
    mask[5:15] = 1
    mask[n_h:n_h + 7] = 1

    sess = [(s1_lh, s1_rh), (s2_lh, s2_rh)]

    with Step0BoldPrefetcher(
        medial_mask=mask, n_lh=n_h, n_rh=n_h, backend="cpu",
    ) as p_cpu:
        p_cpu.prime_subject("sub-001", sess)
        prov_cpu = p_cpu.get_provider("sub-001")
        cpu_1 = prov_cpu(1)
        cpu_2 = prov_cpu(2)

    with Step0BoldPrefetcher(
        medial_mask=mask, n_lh=n_h, n_rh=n_h, backend="gpu",
    ) as p_gpu:
        p_gpu.prime_subject("sub-001", sess)
        prov_gpu = p_gpu.get_provider("sub-001")
        gpu_1 = cp.asnumpy(prov_gpu(1))
        gpu_2 = cp.asnumpy(prov_gpu(2))

    np.testing.assert_array_equal(gpu_1, cpu_1)
    np.testing.assert_array_equal(gpu_2, cpu_2)


def test_step0_prefetcher_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        Step0BoldPrefetcher(
            medial_mask=np.zeros(4, dtype=np.uint8),
            n_lh=2, n_rh=2, backend="cuda",
        )


# ─────────────────────────────────────────────────────────────────────
# Step1BoldPrefetcher — GPU backend
# ─────────────────────────────────────────────────────────────────────
def test_step1_prefetcher_gpu_returns_cupy_tuples(tmp_path):
    """A primed (sub, sess) under backend='gpu' returns a list of
    ``(lh_TxV, rh_TxV)`` cupy tuples — not host numpy. Each buffer is
    C-contig (T, N) fp32 on the current device."""
    n_h = 24
    T = 5
    lh, rh = _write_synth_pair(tmp_path, "s1", N=n_h, T=T, seed=33)

    with Step1BoldPrefetcher(n_workers=2, backend="gpu") as p:
        p.prime("sub-001", "ses-01", [str(lh)], [str(rh)])
        runs = p.get("sub-001", "ses-01")
        assert isinstance(runs, list) and len(runs) == 1
        lh_TxV, rh_TxV = runs[0]
        assert isinstance(lh_TxV, cp.ndarray)
        assert isinstance(rh_TxV, cp.ndarray)
        assert lh_TxV.shape == (T, n_h)
        assert rh_TxV.shape == (T, n_h)
        assert lh_TxV.dtype == cp.float32
        assert rh_TxV.dtype == cp.float32
        assert lh_TxV.flags.c_contiguous
        assert rh_TxV.flags.c_contiguous


def test_step1_prefetcher_gpu_bit_equal_to_cpu(tmp_path):
    """Same per-run input through backend='cpu' and backend='gpu'
    produces bit-equal (T, N) fp32 output per hemi."""
    n_h = 32
    T = 4
    lh, rh = _write_synth_pair(tmp_path, "s1", N=n_h, T=T, seed=77)

    with Step1BoldPrefetcher(n_workers=2, backend="cpu") as p_cpu:
        p_cpu.prime("sub-001", "ses-01", [str(lh)], [str(rh)])
        cpu_runs = p_cpu.get("sub-001", "ses-01")
        cpu_lh, cpu_rh = cpu_runs[0]

    with Step1BoldPrefetcher(n_workers=2, backend="gpu") as p_gpu:
        p_gpu.prime("sub-001", "ses-01", [str(lh)], [str(rh)])
        gpu_runs = p_gpu.get("sub-001", "ses-01")
        gpu_lh = cp.asnumpy(gpu_runs[0][0])
        gpu_rh = cp.asnumpy(gpu_runs[0][1])

    np.testing.assert_array_equal(gpu_lh, cpu_lh)
    np.testing.assert_array_equal(gpu_rh, cpu_rh)


def test_step1_prefetcher_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        Step1BoldPrefetcher(backend="opencl")


def test_compute_profile_arrays_cpu_rejects_cupy_precomputed():
    """compute_profile_arrays(backend='cpu') with cupy-resident
    precomputed_bold_runs must raise a named TypeError, not fail deep
    inside numba. Closes the backend-mismatch trap where a caller
    pairs ``Step1BoldPrefetcher(backend='gpu')`` with the CPU leaf —
    the prefetcher returns cupy buffers but the leaf expects host
    numpy, and the failure would otherwise surface inside the
    ``s_norm.T @ lh_norm`` BLAS dispatch with no diagnostic value."""
    from arealmshbm.generate_profiles import compute_profile_arrays

    n_h = 8
    T = 3
    fake_lh = cp.zeros((T, n_h), dtype=cp.float32)
    fake_rh = cp.zeros((T, n_h), dtype=cp.float32)
    cupy_runs = [(fake_lh, fake_rh)]
    with pytest.raises(TypeError,
                       match=r"compute_profile_arrays\(backend='cpu'\)"):
        compute_profile_arrays(
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            out_dir="/tmp/does-not-matter",
            sub="1", sess="1",
            backend="cpu",
            precomputed_bold_runs=cupy_runs,
        )
