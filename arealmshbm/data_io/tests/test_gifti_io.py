"""test_gifti_io.py — equivalence + contract tests for the GIFTI reader.

The pipeline reads ``.func.gii`` surface BOLD directly. End-to-end
bit-equality with the historical converted-NIFTI mirror was proven
at strip time on real YS sub-001 data (10/10 numerical artifacts
matched under ``backend_step0=cpu``); the e2e harness and the
offline ``convert_ys_bold_parallel.py`` converter were both retired
once the strip landed. See the PR #54 description for the result
table.

What we still pin here: **strict-mode refusal** — a GIFTI that
violates the documented contract (non-FLOAT32 / non-B64GZ / non-LE,
or a later darray that disagrees with the first) must be refused with
an explicit error rather than producing garbage.

CPU↔GPU bit-equality for the whole-subject device reader lives in
``test_gifti_readers.py``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import pytest

from arealmshbm.data_io.gifti_io import read_surface_gifti


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
