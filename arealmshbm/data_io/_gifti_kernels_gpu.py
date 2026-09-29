"""_gifti_kernels_gpu.py

The two CuPy ``RawKernel``s between "raw ``.func.gii`` bytes are on
device" and "one session's ``(T, n_lh + n_rh)`` fp32 buffer is on
device"; DEFLATE in the middle is nvCOMP's. No arithmetic is done on
the BOLD values here and no float atomics, so the output is
bit-identical to the host reader. One rule a maintainer must not break:
``find_gifti_tags_batch`` emits tag offsets through an ``atomicAdd``
counter, so **the host must sort the positions**, and scanning several
files in one launch is safe only while the caller pads each file's
region with >= ``gifti_bold_gpu._MIN_PAD`` zero bytes.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import threading

import numpy as np

_CACHE: dict = {}
_LOCK = threading.Lock()

_SRC_FIND_TAGS = r"""
extern "C" __global__
void find_gifti_tags_batch(
    const unsigned char* __restrict__ buf,
    long long                         buf_len,
    long long*           __restrict__ out_positions,
    int*                 __restrict__ out_kinds,   // 0=<DataArray 1=<Data> 2=</Data>
    int*                 __restrict__ n_found,
    int                               max_found)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x
                      + (long long)threadIdx.x;
    if (i >= buf_len) return;
    if (buf[i] != (unsigned char)'<') return;
    int kind = -1;
    if (i + 10 <= buf_len &&
        buf[i+1]=='D' && buf[i+2]=='a' && buf[i+3]=='t' && buf[i+4]=='a' &&
        buf[i+5]=='A' && buf[i+6]=='r' && buf[i+7]=='r' && buf[i+8]=='a' &&
        buf[i+9]=='y') {
        kind = 0;
    } else if (i + 6 <= buf_len &&
        buf[i+1]=='D' && buf[i+2]=='a' && buf[i+3]=='t' && buf[i+4]=='a' &&
        buf[i+5]=='>') {
        kind = 1;
    } else if (i + 7 <= buf_len &&
        buf[i+1]=='/' && buf[i+2]=='D' && buf[i+3]=='a' && buf[i+4]=='t' &&
        buf[i+5]=='a' && buf[i+6]=='>') {
        kind = 2;
    } else {
        return;
    }
    const int idx = atomicAdd(n_found, 1);
    if (idx < max_found) {
        out_positions[idx] = i;
        out_kinds[idx]     = kind;
    }
}
"""

_SRC_B64 = r"""
extern "C" __global__
void base64_decode_checked(
    const unsigned char* __restrict__ table,
    const unsigned char* __restrict__ in,
    const long long*     __restrict__ in_offsets,
    const int*           __restrict__ in_lens,
    unsigned char*       __restrict__ out,
    const long long*     __restrict__ out_offsets,
    int*                 __restrict__ err_chunk)
{
    const int chunk = blockIdx.x;
    const int q     = blockIdx.y * blockDim.x + threadIdx.x;
    const int n_quads = in_lens[chunk] >> 2;
    if (q >= n_quads) return;

    const unsigned char* src = in  + in_offsets[chunk]  + ((long long)q << 2);
    unsigned char*       dst = out + out_offsets[chunk] + (long long)q * 3;

    // '=' is in the table (it decodes to a 0 sextet the host then
    // truncates), so the alphabet guard alone would accept it anywhere.
    // It is legal ONLY as the tail padding of the final quad: at
    // position 3, or at 2 and 3 together.
    const bool eq0 = (src[0] == '='), eq1 = (src[1] == '=');
    const bool eq2 = (src[2] == '='), eq3 = (src[3] == '=');
    if (eq0 || eq1 || (eq2 && !eq3) || ((eq2 || eq3) && q != n_quads - 1)) {
        atomicMin(err_chunk, chunk);
        return;
    }

    const unsigned int a = table[src[0]];
    const unsigned int b = table[src[1]];
    const unsigned int c = table[src[2]];
    const unsigned int d = table[src[3]];
    if (a == 255u || b == 255u || c == 255u || d == 255u) {
        atomicMin(err_chunk, chunk);
        return;
    }
    const unsigned int triple = (a << 18) | (b << 12) | (c << 6) | d;
    dst[0] = (unsigned char)((triple >> 16) & 0xffu);
    dst[1] = (unsigned char)((triple >>  8) & 0xffu);
    dst[2] = (unsigned char)( triple        & 0xffu);
}
"""


def get_gifti_gpu_kernels():
    """Build (once) and return the RawKernels + the b64 decode table.

    Returns ``(find_tags, b64_decode, table_dev)``, the last
    being the 256-entry uint8 base64 lookup (255 = outside the
    alphabet). The cache is **not** keyed by device: a multi-GPU port
    must add the device id here. Every caller that can run on a
    background thread pins the caller's device first.
    """
    if "k" in _CACHE:
        return _CACHE["k"]
    with _LOCK:
        if "k" in _CACHE:
            return _CACHE["k"]
        import cupy as cp

        find_tags = cp.RawKernel(_SRC_FIND_TAGS, "find_gifti_tags_batch")
        b64 = cp.RawKernel(_SRC_B64, "base64_decode_checked")

        table = np.full(256, 255, dtype=np.uint8)
        for i, ch in enumerate(
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            b"0123456789+/"
        ):
            table[ch] = i
        # Padding contributes no real byte; the kernel is what
        # enforces that it only ever appears as the tail pad.
        table[ord("=")] = 0
        _CACHE["k"] = (find_tags, b64, cp.asarray(table))
        return _CACHE["k"]
