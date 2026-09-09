"""test_step0_bold_prefetcher.py — the CPU-only ``Step0BoldPrefetcher``
contract: what ``prime_subject`` / ``get_provider`` hand the step-0
driver, and the bookkeeping around it.

The worker is ``read_surface_bold`` + ``concat_hemis_drop_medial``, so
the output oracle is those two called inline; what these tests pin is
the futures keying (1-indexed session within a subject), pop-on-consume,
idempotent priming, subjects decoding independently, and ``close()``
dropping unpulled work. No GPU anywhere: decode is host isal, full stop.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np
import pytest

from arealmshbm.bold_io import concat_hemis_drop_medial, read_surface_bold
from arealmshbm.pipeline._step0_bold_prefetcher import Step0BoldPrefetcher


def _synth_gifti_bytes(N: int, T: int, seed: int) -> bytes:
    """Minimal GIFTI: T DataArrays, each Dim0=N fp32, deterministic on
    ``seed`` so every synthetic file is unique and reproducible."""
    rng = np.random.default_rng(seed)
    parts = [b'<?xml version="1.0" encoding="UTF-8"?>\n',
             b'<GIFTI Version="1.0" NumberOfDataArrays="',
             str(T).encode(), b'">\n']
    for _ in range(T):
        arr = rng.standard_normal(N).astype(np.float32)
        payload = base64.b64encode(zlib.compress(arr.tobytes()))
        parts.append(
            f'<DataArray DataType="NIFTI_TYPE_FLOAT32" '
            f'ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" '
            f'Encoding="GZipBase64Binary" Endian="LittleEndian" '
            f'ExternalFileName="" ExternalFileOffset="0" '
            f'Dim0="{N}"><Data>'.encode())
        parts.append(payload)
        parts.append(b'</Data></DataArray>\n')
    parts.append(b'</GIFTI>\n')
    return b"".join(parts)


def _write_pair(tmp_path: Path, name: str, N: int, T: int, seed: int):
    lh = tmp_path / f"{name}_lh.func.gii"
    rh = tmp_path / f"{name}_rh.func.gii"
    lh.write_bytes(_synth_gifti_bytes(N, T, seed))
    rh.write_bytes(_synth_gifti_bytes(N, T, seed + 1))
    return lh, rh


def _oracle(pair, n_h: int, mask: np.ndarray) -> np.ndarray:
    lh, rh = pair
    return concat_hemis_drop_medial(read_surface_bold(lh, expected_n=n_h),
                                    read_surface_bold(rh, expected_n=n_h),
                                    mask)


def test_provider_hands_back_the_decoded_session_and_pops_it(tmp_path):
    n_h, T = 48, 5
    s1 = _write_pair(tmp_path, "s1", n_h, T, seed=100)
    s2 = _write_pair(tmp_path, "s2", n_h, T, seed=200)
    mask = np.zeros(2 * n_h, dtype=np.uint8)
    mask[5:15] = 1
    mask[n_h:n_h + 7] = 1
    n_cortex = 2 * n_h - int(mask.sum())

    with Step0BoldPrefetcher(medial_mask=mask, n_lh=n_h, n_rh=n_h) as p:
        p.prime_subject("sub-001", [s1, s2])
        p.prime_subject("sub-001", [s1, s2])          # idempotent on key
        prov = p.get_provider("sub-001")
        got_2 = prov(2)                                 # any order
        got_1 = prov(1)
        for got, pair in ((got_1, s1), (got_2, s2)):
            assert isinstance(got, np.ndarray)
            assert got.shape == (n_cortex, T) and got.dtype == np.float32
            np.testing.assert_array_equal(got, _oracle(pair, n_h, mask))
        with pytest.raises(KeyError, match="not primed"):
            prov(1)                                     # popped on consume
        with pytest.raises(KeyError, match="not primed"):
            prov(3)


def test_subjects_decode_independently_and_close_drops_the_rest(tmp_path):
    n_h, T = 16, 3
    a = _write_pair(tmp_path, "a", n_h, T, seed=1)
    b = _write_pair(tmp_path, "b", n_h, T, seed=2)
    mask = np.zeros(2 * n_h, dtype=np.uint8)
    p = Step0BoldPrefetcher(medial_mask=mask, n_lh=n_h, n_rh=n_h,
                            n_io_workers=2)
    p.prime_subject("A", [a])
    p.prime_subject("B", [b])
    np.testing.assert_array_equal(p.get_provider("B")(1),
                                  _oracle(b, n_h, mask))
    p.close()
    with pytest.raises(KeyError, match="not primed"):
        p.get_provider("A")(1)
