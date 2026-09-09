"""_gifti_kernels.py

CPU (numba) kernels for the ``.func.gii`` BOLD reader: all
``njit(nogil=True, cache=True)`` over pre-allocated buffers, none
``parallel=True`` (the callers own the thread pool). Nothing here does
floating-point arithmetic — the fp32 payload is only moved.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
from numba import njit

# ── base64 status codes (returned instead of raising: see docstring) ──
B64_ERR_LEN = -2        # input length not a multiple of 4
B64_ERR_ALPHABET = -1   # byte outside A-Za-z0-9+/ (or misplaced '=')

_PAD = 61   # ord('=')
_INVALID = 255

#: 256-entry decode table: alphabet -> 6-bit value, every other byte
#: (whitespace, '=', control, high-bit) -> 255. Passed into the kernels
#: rather than read as a global: numba freezes globals at compile time.
B64_TABLE = np.full(256, _INVALID, dtype=np.uint8)
for _i, _c in enumerate(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
):
    B64_TABLE[_c] = _i


@njit(cache=True, nogil=True, boundscheck=False)
def b64_decode_strict(table, src, in_off, in_len, dst, out_off):
    """Decode base64 ``src[in_off:in_off+in_len]`` into ``dst[out_off:]``.

    ``dst`` must hold at least ``in_len // 4 * 3`` bytes at ``out_off``;
    the kernel is ``boundscheck=False``. Returns bytes written, or
    negative on refusal (:data:`B64_ERR_LEN` / :data:`B64_ERR_ALPHABET`
    — a ``nogil`` kernel cannot raise cheaply).
    Padding is legal only in the final quad.
    """
    if in_len < 0:
        return B64_ERR_LEN
    if (in_len & 3) != 0:
        return B64_ERR_LEN
    n_quads = in_len >> 2
    o = out_off
    for q in range(n_quads):
        s = in_off + (q << 2)
        a = table[src[s]]
        b = table[src[s + 1]]
        c = table[src[s + 2]]
        d = table[src[s + 3]]
        if a == 255 or b == 255:
            return B64_ERR_ALPHABET
        if c == 255:
            # "xx==" — one output byte, only valid as the last quad.
            if src[s + 2] != _PAD or src[s + 3] != _PAD or q != n_quads - 1:
                return B64_ERR_ALPHABET
            dst[o] = np.uint8(((a << 2) | (b >> 4)) & 255)
            return o + 1 - out_off
        if d == 255:
            # "xxx=" — two output bytes, only valid as the last quad.
            if src[s + 3] != _PAD or q != n_quads - 1:
                return B64_ERR_ALPHABET
            dst[o] = np.uint8(((a << 2) | (b >> 4)) & 255)
            dst[o + 1] = np.uint8((((b & 15) << 4) | (c >> 2)) & 255)
            return o + 2 - out_off
        triple = ((np.uint32(a) << 18) | (np.uint32(b) << 12)
                  | (np.uint32(c) << 6) | np.uint32(d))
        dst[o] = np.uint8((triple >> 16) & 255)
        dst[o + 1] = np.uint8((triple >> 8) & 255)
        dst[o + 2] = np.uint8(triple & 255)
        o += 3
    return o - out_off


@njit(cache=True, nogil=True, boundscheck=False)
def copy_row_f32(src, dst, row):
    """``dst[row, :] = src`` without taking the GIL."""
    n = src.shape[0]
    for i in range(n):
        dst[row, i] = src[i]


@njit(cache=True, nogil=True, boundscheck=False)
def transpose_f32(src, dst, n0, n1):
    """Blocked ``dst[n0:n1, :] = src[:, n0:n1].T`` without the GIL."""
    T = src.shape[0]
    B = 64
    for nb in range(n0, n1, B):
        ne = min(nb + B, n1)
        for tb in range(0, T, B):
            te = min(tb + B, T)
            for n in range(nb, ne):
                for t in range(tb, te):
                    dst[n, t] = src[t, n]
