"""test_gifti_io.py — equivalence + contract tests for the GIFTI reader.

The pipeline reads ``.func.gii`` surface BOLD directly. End-to-end
bit-equality with the historical converted-NIFTI mirror was proven
at strip time on real YS sub-001 data (10/10 numerical artifacts
matched under ``backend_step0=cpu``); the e2e harness and the
offline ``convert_ys_bold_parallel.py`` converter were both retired
once the strip landed. See the PR #54 description for the result
table.

What we still pin here:

  1. **CPU vs GPU bit-equality** — the nvCOMP-batched Deflate GPU
     reader produces the exact same ``(N, T) fp32`` as the
     bytescan-isal CPU reader, on both synthetic and real input.
  2. **strict-mode refusal** — a GIFTI that violates the documented
     contract (non-FLOAT32 / non-B64GZ / non-LE) must be refused
     with an explicit error rather than producing garbage.
  3. **out= buffer reuse** — the GPU reader's caller-allocated
     destination path works and rejects wrong-shape buffers.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import zlib
import os
from pathlib import Path

import numpy as np
import pytest

from arealmshbm.data_io.gifti_io import (
    read_surface_gifti,
    read_surface_gifti_gpu,
    read_surface_giftis_gpu_full_pipeline,
)

# GPU-stack availability check. The GPU reader requires both cupy and
# ``nvidia-nvcomp-cu12``; tests that touch the GPU path skip when
# either is missing instead of hard-failing on CPU-only CI.
try:
    import cupy as _cp  # noqa: F401
    import nvidia.nvcomp as _nvcomp  # noqa: F401
    _GPU_AVAILABLE = True
    _GPU_SKIP_REASON = ""
except ImportError as _e:
    _GPU_AVAILABLE = False
    _GPU_SKIP_REASON = f"GPU stack unavailable ({_e})"


# Real-data reference (used by the CPU↔GPU real-data equivalence test).
# Point MSHBM_TEST_BOLD_ROOT at a BOLD mount laid out as
# <root>/sub-XXX/ses-YY/func/*.func.gii; the test skips when unset or
# the mount is offline.
_YS_GII_ROOT = Path(os.environ.get("MSHBM_TEST_BOLD_ROOT",
                                   "testdata/BOLD"))
_GII_PATH = (_YS_GII_ROOT / "sub-001" / "ses-01" / "func"
             / "sub-001_ses-01_task-rest_hemi-L"
               "_space-fsaverage6_bold.func.gii")
_REAL_AVAILABLE = _GII_PATH.exists()
_REAL_SKIP_REASON = (
    f"YS sub-001 reference .func.gii not on disk at {_GII_PATH}"
)


# ─────────────────────────────────────────────────────────────────────
# Strict-mode refusal — synthesize minimal GIFTI bytes
# ─────────────────────────────────────────────────────────────────────
def _make_minimal_gifti(*, datatype: str = "NIFTI_TYPE_FLOAT32",
                        encoding: str = "GZipBase64Binary",
                        endian: str = "LittleEndian",
                        dim0: int = 4, T: int = 2,
                        override_idx: int | None = None,
                        override_datatype: str | None = None,
                        override_encoding: str | None = None,
                        override_endian: str | None = None,
                        override_dim0: int | None = None) -> bytes:
    """Build a tiny but valid GIFTI byte stream with the given attrs.

    The data payload is deterministic — ``np.arange(dim0, dtype=fp32) + t``
    for each darray. We only synthesize the GZipBase64Binary encoding
    (the only one the reader supports); the refusal tests exercise the
    non-GZB64 branches by setting the ``encoding`` header attr to a
    different string while keeping the payload as GZipBase64Binary —
    the reader's header check fires before the decode.

    Per-darray override
    -------------------
    The ``override_idx`` knob targets *one* DataArray (0-based) and
    swaps its DataType / Encoding / Endian / Dim0 attribute(s) to the
    ``override_*`` values. The remaining darrays keep the uniform
    ``datatype`` / ``encoding`` / ``endian`` / ``dim0`` values. Used by
    the heterogeneous-darray rejection tests to verify the parser
    catches contract violations on any darray, not just the first.
    """
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>\n']
    parts.append(b'<GIFTI Version="1.0" NumberOfDataArrays="')
    parts.append(str(T).encode())
    parts.append(b'">\n')
    for t in range(T):
        # Per-darray override resolution.
        is_override = (override_idx is not None and t == override_idx)
        dt_t = (override_datatype if is_override and override_datatype is not None
                else datatype)
        enc_t = (override_encoding if is_override and override_encoding is not None
                 else encoding)
        end_t = (override_endian if is_override and override_endian is not None
                 else endian)
        dim0_t = (override_dim0 if is_override and override_dim0 is not None
                  else dim0)
        # Payload uses dim0_t so the bytes match the overridden header
        # — the parser fires on the header substring check before any
        # decode runs, so payload correctness is incidental.
        arr = (np.arange(dim0_t, dtype=np.float32) + t).tobytes()
        # Payload always GZipBase64Binary for the synth — the refusal
        # tests don't need a "real" alternate-encoding payload.
        payload = base64.b64encode(zlib.compress(arr)).decode("ascii")
        parts.append(
            f'<DataArray Intent="NIFTI_INTENT_TIME_SERIES" '
            f'DataType="{dt_t}" '
            f'ArrayIndexingOrder="RowMajorOrder" '
            f'Dimensionality="1" '
            f'Encoding="{enc_t}" '
            f'Endian="{end_t}" '
            f'ExternalFileName="" ExternalFileOffset="0" '
            f'Dim0="{dim0_t}">'.encode()
        )
        parts.append(b'<Data>')
        parts.append(payload.encode("ascii"))
        parts.append(b'</Data>')
        parts.append(b'</DataArray>\n')
    parts.append(b'</GIFTI>\n')
    return b"".join(parts)


def test_read_surface_gifti_synthetic_round_trip(tmp_path: Path):
    """A minimal hand-built GIFTI decodes back to the expected
    deterministic payload — sanity check that the bytescan + decode
    contract is right independent of any specific real file."""
    p = tmp_path / "synth.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=8, T=3))
    out = read_surface_gifti(p)
    assert out.shape == (8, 3)
    assert out.dtype == np.float32
    expected = np.column_stack([np.arange(8, dtype=np.float32) + t
                                 for t in range(3)])
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("attr,bad_value,marker", [
    ("datatype", "NIFTI_TYPE_INT16", "DataType"),
    ("encoding", "Base64Binary", "Encoding"),
    ("endian",   "BigEndian",     "Endian"),
])
def test_read_surface_gifti_rejects_off_contract(tmp_path: Path,
                                                 attr, bad_value, marker):
    """Strict-mode refusal: the reader bails on any non-FLOAT32 /
    non-GZip-B64 / non-LE header. The error message must name the
    offending attribute so the operator can fix the writer. There is
    no NIFTI fallback — ``.nii.gz`` is rejected at parse time."""
    kwargs = {attr: bad_value}
    p = tmp_path / "bad.func.gii"
    p.write_bytes(_make_minimal_gifti(**kwargs))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=marker):
        read_surface_gifti(p)


def test_read_surface_gifti_missing_dim0(tmp_path: Path):
    """A GIFTI missing the Dim0 attribute is refused with an explicit
    error — not a silent crash later in the buffer-length validation."""
    raw = _make_minimal_gifti(dim0=8, T=2)
    # Strip the Dim0 attribute from the first DataArray header.
    mutated = raw.replace(b' Dim0="8"', b"", 1)
    p = tmp_path / "no_dim0.func.gii"
    p.write_bytes(mutated)
    with pytest.raises(ValueError, match="Dim0"):
        read_surface_gifti(p)


# ─────────────────────────────────────────────────────────────────────
# Per-darray contract validation — a GIFTI whose first <DataArray>
# is contract-compliant but a *later* darray switches DataType /
# Encoding / Endian / Dim0 must be refused at parse time with an
# error that names the offending darray index. Critically, a
# heterogeneous-DataType file (e.g. NIFTI_TYPE_INT32 with the same
# N*4-byte payload as FLOAT32) cannot be caught by any per-chunk
# byte-length check — the per-darray header substring check is the
# only line of defense.
# ─────────────────────────────────────────────────────────────────────
def test_rejects_heterogeneous_datatype_in_later_darray(tmp_path: Path):
    """A second DataArray with ``NIFTI_TYPE_INT32`` is refused even
    though its payload byte-count equals the FLOAT32 byte-count
    (4 bytes per element either way). Length check alone would miss
    this — per-darray header validation is what catches it."""
    p = tmp_path / "het_dtype.func.gii"
    p.write_bytes(_make_minimal_gifti(
        dim0=8, T=3,
        override_idx=1,
        override_datatype="NIFTI_TYPE_INT32",
    ))
    with pytest.raises(ValueError, match=r"DataArray 1.*DataType"):
        read_surface_gifti(p)


def test_rejects_heterogeneous_encoding_in_later_darray(tmp_path: Path):
    """A third DataArray that switches to ``Base64Binary`` is refused
    at parse time before any decode runs."""
    p = tmp_path / "het_encoding.func.gii"
    p.write_bytes(_make_minimal_gifti(
        dim0=8, T=4,
        override_idx=2,
        override_encoding="Base64Binary",
    ))
    with pytest.raises(ValueError, match=r"DataArray 2.*Encoding"):
        read_surface_gifti(p)


def test_rejects_heterogeneous_endian_in_later_darray(tmp_path: Path):
    """A later DataArray that claims ``BigEndian`` is refused — the
    reader does no byteswap and would silently produce garbage."""
    p = tmp_path / "het_endian.func.gii"
    p.write_bytes(_make_minimal_gifti(
        dim0=8, T=3,
        override_idx=2,
        override_endian="BigEndian",
    ))
    with pytest.raises(ValueError, match=r"DataArray 2.*Endian"):
        read_surface_gifti(p)


def test_rejects_heterogeneous_dim0_in_later_darray(tmp_path: Path):
    """A later DataArray whose ``Dim0`` disagrees with the first
    darray's ``N`` is refused at parse time — same-width corruption
    that would otherwise surface deep inside the column-write."""
    p = tmp_path / "het_dim0.func.gii"
    p.write_bytes(_make_minimal_gifti(
        dim0=8, T=3,
        override_idx=1,
        override_dim0=7,
    ))
    with pytest.raises(ValueError, match=r"DataArray 1.*Dim0"):
        read_surface_gifti(p)


# ─────────────────────────────────────────────────────────────────────
# GPU reader equivalence
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_read_surface_gifti_gpu_matches_cpu_synthetic(tmp_path: Path):
    """The nvCOMP-batched GPU decode lands the same bytes as the
    CPU isal_zlib path on a deterministic synthetic GIFTI. Stock
    zlib's deflate output is byte-deterministic, and nvCOMP's
    Deflate decoder under ``BitstreamKind.RAW`` is a conforming
    implementation of RFC 1951, so the two must agree bit-for-bit
    after stripping the zlib wrapper.
    """
    p = tmp_path / "for_gpu.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=64, T=12))
    cpu = read_surface_gifti(p)
    gpu_host = read_surface_gifti_gpu(p, to_host=True)
    np.testing.assert_array_equal(cpu, gpu_host)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_read_surface_gifti_gpu_device_stay_matches_host_out(tmp_path: Path):
    """``to_host=False`` returns a cupy device array; pulling it back
    via ``cp.asnumpy`` must equal the ``to_host=True`` path. Pin the
    contract so a future kernel rewrite of the column-write doesn't
    silently break."""
    import cupy as cp
    p = tmp_path / "devstay.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=64, T=12))
    gpu_dev = read_surface_gifti_gpu(p, to_host=False)
    gpu_host = read_surface_gifti_gpu(p, to_host=True)
    assert isinstance(gpu_dev, cp.ndarray)
    assert gpu_dev.dtype == cp.float32
    np.testing.assert_array_equal(cp.asnumpy(gpu_dev), gpu_host)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_read_surface_gifti_gpu_into_preallocated_out(tmp_path: Path):
    """The ``out=`` parameter lets a prefetcher reuse a single
    ``(N, T) cupy fp32`` buffer across files. Verify the writer
    actually populates the passed buffer and the values match the
    fresh-allocation path."""
    import cupy as cp
    p = tmp_path / "prealloc.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=32, T=4))
    fresh = read_surface_gifti_gpu(p, to_host=False)
    buf = cp.zeros_like(fresh)
    returned = read_surface_gifti_gpu(p, out=buf)
    assert returned is buf  # writes into caller's buffer
    np.testing.assert_array_equal(cp.asnumpy(buf), cp.asnumpy(fresh))


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_read_surface_gifti_gpu_out_shape_mismatch_raises(tmp_path: Path):
    """A wrong-shape ``out`` buffer is rejected with an explicit
    error rather than silently writing garbage."""
    import cupy as cp
    p = tmp_path / "badout.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=32, T=4))
    bad = cp.zeros((32, 99), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        read_surface_gifti_gpu(p, out=bad)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_read_surface_gifti_gpu_out_numpy_rejected(tmp_path: Path):
    """An ``out`` buffer that's a numpy ndarray (not cupy) is rejected
    with a named error. Without the isinstance guard the dtype check
    would silently pass — cupy reuses numpy's fp32 dtype — and the
    subsequent ``out.device.id`` access would AttributeError deep in
    the call, which gives a much worse error to the caller."""
    p = tmp_path / "wrong_buf_kind.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=32, T=4))
    host_buf = np.zeros((32, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="cupy.ndarray"):
        read_surface_gifti_gpu(p, out=host_buf)


@pytest.mark.skipif(not _REAL_AVAILABLE, reason=_REAL_SKIP_REASON)
@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_read_surface_gifti_gpu_matches_cpu_real():
    """GPU GIFTI on a real YS sub-001 .func.gii produces the same
    ``(N, T) fp32`` as the CPU reader. Synthetic equivalence is the
    fast path; this is the real-data backstop, gated on the local YS
    cohort being mounted."""
    cpu = read_surface_gifti(_GII_PATH)
    gpu_host = read_surface_gifti_gpu(_GII_PATH, to_host=True)
    assert cpu.shape == (40962, 242)
    assert cpu.dtype == np.float32
    np.testing.assert_array_equal(cpu, gpu_host)


# ─────────────────────────────────────────────────────────────────────
# Full device-stay GPU pipeline (read_surface_giftis_gpu_full_pipeline)
# ─────────────────────────────────────────────────────────────────────
# What we pin here:
#   * Bit-equality vs the CPU isal_zlib reader on both single files and
#     mixed-size batches — the GPU bytescan + base64 RawKernels and the
#     batched nvCOMP decode must agree byte-for-byte with the host
#     reference.
#   * Order preservation — output[i] corresponds to paths[i] even when
#     the parse pool finishes files out of submission order.
#   * Per-darray header validation surfaces a clear error on
#     heterogeneous-Dim0 GIFTIs instead of crashing later inside
#     ``cp.stack`` with a shape mismatch.
#   * Real-data backstop — synthetic equivalence is the cheap gate; the
#     real-file test catches a regression that only shows up at fsa6 /
#     T=242 sizes (e.g. nvCOMP batch threshold, position-array
#     overflow).


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_matches_cpu_synthetic_single(tmp_path: Path):
    """Single-file batch (the smallest possible call) returns the same
    bytes as :func:`read_surface_gifti`. Pins the GPU bytescan + base64
    against the host isal_zlib reference for one file."""
    p = tmp_path / "single.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=64, T=12))
    cpu = read_surface_gifti(p)
    out = read_surface_giftis_gpu_full_pipeline([p], to_host=True)
    assert isinstance(out, list) and len(out) == 1
    np.testing.assert_array_equal(cpu, out[0])


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_matches_cpu_synthetic_batch(tmp_path: Path):
    """A 4-file batch with mixed (N, T) returns the same bytes per file
    as the CPU reader. Exercises the per-file ``file_ranges`` split of
    the single batched ``codec.decode`` output and the per-file
    ``cp.stack`` + transpose, including files where the chunk count and
    per-chunk byte length both differ."""
    shapes = [(16, 4), (64, 12), (32, 8), (8, 3)]
    paths = []
    for i, (dim0, T) in enumerate(shapes):
        p = tmp_path / f"batch_{i}.func.gii"
        p.write_bytes(_make_minimal_gifti(dim0=dim0, T=T))
        paths.append(p)

    outs = read_surface_giftis_gpu_full_pipeline(paths, to_host=True)
    assert len(outs) == len(paths)
    for i, p in enumerate(paths):
        cpu = read_surface_gifti(p)
        assert outs[i].shape == cpu.shape, (
            f"file {i}: shape {outs[i].shape} != cpu {cpu.shape}"
        )
        np.testing.assert_array_equal(
            cpu, outs[i],
            err_msg=f"file {i} (shape {shapes[i]}) differs",
        )


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_device_stay_matches_host_out(tmp_path: Path):
    """``to_host=False`` returns cupy device arrays; pulling each one
    back via ``cp.asnumpy`` must equal the ``to_host=True`` path. Pins
    the device-stay contract — the step1 GPU prefetcher relies on it to
    skip a redundant H2D round trip."""
    import cupy as cp
    paths = [tmp_path / f"ds_{i}.func.gii" for i in range(2)]
    for p in paths:
        p.write_bytes(_make_minimal_gifti(dim0=32, T=6))

    dev_outs = read_surface_giftis_gpu_full_pipeline(paths, to_host=False)
    host_outs = read_surface_giftis_gpu_full_pipeline(paths, to_host=True)
    assert all(isinstance(o, cp.ndarray) for o in dev_outs)
    assert all(o.dtype == cp.float32 for o in dev_outs)
    for d, h in zip(dev_outs, host_outs):
        np.testing.assert_array_equal(cp.asnumpy(d), h)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_preserves_input_order(tmp_path: Path):
    """``outs[i]`` corresponds to ``paths[i]`` regardless of the order
    the internal parse pool finishes files. Distinguish files by giving
    each a unique ``dim0`` and checking ``outs[i].shape[0] == dim0_i``.
    Without order preservation, the prefetcher's per-(sub, sess) future
    would resolve with the wrong session's BOLD."""
    dim0_seq = [8, 24, 56, 16, 40]
    paths = []
    for i, d in enumerate(dim0_seq):
        p = tmp_path / f"ord_{i}.func.gii"
        p.write_bytes(_make_minimal_gifti(dim0=d, T=4))
        paths.append(p)

    outs = read_surface_giftis_gpu_full_pipeline(paths, to_host=True)
    out_dims = [o.shape[0] for o in outs]
    assert out_dims == dim0_seq, (
        f"order not preserved: got {out_dims}, expected {dim0_seq}"
    )


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_empty_paths_returns_empty():
    """Empty path list short-circuits to an empty list — no pool
    allocation, no codec call. Pins the edge case before the first
    ``parsed[0]`` lookup that would otherwise IndexError."""
    out = read_surface_giftis_gpu_full_pipeline([], to_host=True)
    assert out == []


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_rejects_non_float32_datatype(tmp_path: Path):
    """First-DataArray contract validation surfaces an explicit error
    naming ``DataType`` when the GIFTI advertises a non-FLOAT32 dtype.
    Without it, the downstream nvCOMP decode + ``view(float32)`` would
    silently reinterpret integer bytes as fp32 garbage."""
    p = tmp_path / "wrong_dtype.func.gii"
    p.write_bytes(_make_minimal_gifti(
        dim0=8, T=2, datatype="NIFTI_TYPE_INT32"))
    with pytest.raises(ValueError, match="DataType"):
        read_surface_giftis_gpu_full_pipeline([p], to_host=True)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_rejects_heterogeneous_dim0(tmp_path: Path):
    """A GIFTI whose 3rd DataArray declares a different ``Dim0`` than
    the first is refused at the host-side per-darray validation pass —
    before the GPU base64 + nvCOMP decode would produce a list of
    different-sized chunks that ``cp.stack`` cannot assemble. The
    rejection names the offending darray index and ``Dim0`` so a
    cohort builder can locate the bad file."""
    p = tmp_path / "het_dim0.func.gii"
    p.write_bytes(_make_minimal_gifti(
        dim0=16, T=5, override_idx=2, override_dim0=24))
    with pytest.raises(ValueError, match="DataArray 2"):
        read_surface_giftis_gpu_full_pipeline([p], to_host=True)


@pytest.mark.skipif(not _REAL_AVAILABLE, reason=_REAL_SKIP_REASON)
@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_matches_cpu_real():
    """Real-data backstop: at fsa6 / T=242 the GPU full pipeline (
    bytescan + base64 + batched nvCOMP) lands identical bytes to the
    CPU isal_zlib reader. Catches regressions that only show up at
    production scale — position-array overflow, batched decode chunk-
    boundary mishandling, etc."""
    cpu = read_surface_gifti(_GII_PATH)
    out = read_surface_giftis_gpu_full_pipeline([_GII_PATH], to_host=True)
    assert len(out) == 1
    assert out[0].shape == (40962, 242)
    assert out[0].dtype == np.float32
    np.testing.assert_array_equal(cpu, out[0])


@pytest.mark.skipif(not _REAL_AVAILABLE, reason=_REAL_SKIP_REASON)
@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_matches_cpu_real_device_stay():
    """Real-data device-stay backstop: at fsa6 / T=242 with
    ``to_host=False``, the device-resident output round-trips bit-equal
    to the CPU isal_zlib reader. The synthetic device-stay test pins
    the to_host=True/False equivalence on small N; this test pins the
    same contract at production scale — catches device-side regressions
    (e.g. cp.stack chunk-boundary handling, layout='NxT' transpose
    contiguity) that would only surface under real BOLD geometry."""
    cpu = read_surface_gifti(_GII_PATH)
    out_dev = read_surface_giftis_gpu_full_pipeline(
        [_GII_PATH], to_host=False)
    assert len(out_dev) == 1
    import cupy as cp
    assert isinstance(out_dev[0], cp.ndarray)
    assert out_dev[0].shape == (40962, 242)
    assert out_dev[0].dtype == cp.float32
    np.testing.assert_array_equal(cpu, cp.asnumpy(out_dev[0]))


@pytest.mark.skipif(not _REAL_AVAILABLE, reason=_REAL_SKIP_REASON)
@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_layout_TxN_matches_NxT_transposed():
    """``layout='TxN'`` returns the transpose of the default 'NxT'
    output — pins the layout option's correctness against the existing
    NxT contract. The step1 GPU prefetcher relies on this to skip a
    transpose pair (reader's NxT→.T plus consumer's .T)."""
    out_NT = read_surface_giftis_gpu_full_pipeline(
        [_GII_PATH], to_host=True, layout="NxT")
    out_TN = read_surface_giftis_gpu_full_pipeline(
        [_GII_PATH], to_host=True, layout="TxN")
    assert out_NT[0].shape == (40962, 242)
    assert out_TN[0].shape == (242, 40962)
    np.testing.assert_array_equal(out_NT[0], out_TN[0].T)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_rejects_wrapped_b64(tmp_path: Path):
    """A GIFTI whose ``<Data>`` payload contains line-wrapping
    whitespace (\\n, \\r, space, \\t) is rejected with an explicit
    error. The GPU base64 RawKernel maps non-alphabet bytes to 0 and
    would silently produce wrong bytes; the CPU reader silently strips
    via base64.b64decode. Surface the divergence instead of letting
    the two backends drift."""
    # Build a normal GIFTI then splice four newlines into the first
    # <Data> payload. Four (not one) preserves the multiple-of-4 b64
    # length invariant so the test exercises the whitespace path
    # specifically — a single \n would trip the modulo-4 sanity check
    # first.
    raw = _make_minimal_gifti(dim0=8, T=2)
    p = tmp_path / "wrapped.func.gii"
    data_idx = raw.index(b"<Data>")
    after_open = data_idx + len(b"<Data>")
    spliced = raw[:after_open + 4] + b"\n\n\n\n" + raw[after_open + 4:]
    p.write_bytes(spliced)
    with pytest.raises(ValueError, match="contains whitespace"):
        read_surface_giftis_gpu_full_pipeline([p], to_host=True)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason=_GPU_SKIP_REASON)
def test_full_pipeline_rejects_invalid_layout(tmp_path: Path):
    """Unknown layout strings fail at the boundary, not deep in a
    cp.stack reshape — the kwarg has a tight permitted set."""
    p = tmp_path / "ok.func.gii"
    p.write_bytes(_make_minimal_gifti(dim0=8, T=2))
    with pytest.raises(ValueError, match="layout must be"):
        read_surface_giftis_gpu_full_pipeline([p], layout="bogus")
