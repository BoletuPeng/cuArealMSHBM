"""test_bold_io.py — step0 BOLD reader (GIFTI-only).

``arealmshbm.bold_io.read_surface_bold`` is the step0 surface BOLD
entry point. Since the NIFTI mirror was retired it is a thin
``.func.gii``-only wrapper around
:func:`arealmshbm.data_io.gifti_io.read_surface_gifti` plus the
``expected_n`` guard. These tests pin both the contract refusal
(``.nii.gz`` paths must error) and the V-mismatch guard.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import pytest

from arealmshbm.bold_io import read_surface_bold


def _synth_gifti(dim0: int = 4, T: int = 2) -> bytes:
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>\n',
             b'<GIFTI Version="1.0" NumberOfDataArrays="',
             str(T).encode(), b'">\n']
    for t in range(T):
        arr = (np.arange(dim0, dtype=np.float32) + t).tobytes()
        payload = base64.b64encode(zlib.compress(arr)).decode("ascii")
        parts.append(
            f'<DataArray DataType="NIFTI_TYPE_FLOAT32" '
            f'ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" '
            f'Encoding="GZipBase64Binary" Endian="LittleEndian" '
            f'ExternalFileName="" ExternalFileOffset="0" '
            f'Dim0="{dim0}"><Data>'.encode()
        )
        parts.append(payload.encode("ascii"))
        parts.append(b'</Data></DataArray>\n')
    parts.append(b'</GIFTI>\n')
    return b"".join(parts)


def test_read_surface_bold_synthetic_gifti(tmp_path: Path):
    """Synthetic GIFTI → expected deterministic payload via
    ``read_surface_bold``. No real-cohort dependency."""
    p = tmp_path / "s.func.gii"
    p.write_bytes(_synth_gifti(dim0=8, T=3))
    out = read_surface_bold(p, expected_n=8)
    expected = np.column_stack(
        [np.arange(8, dtype=np.float32) + t for t in range(3)]
    )
    np.testing.assert_array_equal(out, expected)


def test_read_surface_bold_expected_n_guard(tmp_path: Path):
    """``expected_n`` mismatch raises with a message that names
    expected vs actual."""
    p = tmp_path / "s.func.gii"
    p.write_bytes(_synth_gifti(dim0=8, T=2))
    with pytest.raises(ValueError, match="expected_n"):
        read_surface_bold(p, expected_n=12345)


def test_read_surface_bold_rejects_nii_gz(tmp_path: Path):
    """The NIFTI path was retired — a ``.nii.gz`` argument must hit
    an explicit refusal naming the suffix, not silently call into
    nibabel or skip the GIFTI parser."""
    p = tmp_path / "fake.nii.gz"
    p.write_bytes(b"\x1f\x8b\x08\x00")  # gzip magic, doesn't matter
    with pytest.raises(ValueError, match=r"\.gii"):
        read_surface_bold(p, expected_n=8)


def test_read_surface_bold_rejects_unknown_suffix(tmp_path: Path):
    p = tmp_path / "bold.mgh"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError, match=r"\.gii"):
        read_surface_bold(p)
