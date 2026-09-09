"""gifti_bold_gpu.py — one subject's BOLD ingest, straight to device.

:func:`read_subject_bold_gpu` turns one subject's ``(lh, rh)``
``.func.gii`` pairs into one ``(N_cortex, T) fp32`` **device** array
per session, bit-equal to ``concat_hemis_drop_medial(
read_surface_bold(lh), read_surface_bold(rh), medial_mask)``. One
``(T, n_lh + n_rh) fp32`` buffer per session is the batched
decompressor's destination, so on return it already *is*
``vstack([lh, rh]).T``; only the fused epilogue is left. Stream
``prep`` carries H2D, tag scan and base64; ``dec`` carries the Deflate
decode, the epilogue and the returned arrays. CuPy's pool keys free
lists by the allocating stream, so a buffer written on ``prep`` and
read on ``dec`` is held until that group's event fires.

:func:`iter_subject_bold_gpu` is the pipelined variant: it yields each
session as soon as its group's event has fired. ``raw_sessions=True``
skips the epilogue and hands back the ``(T, n_lh + n_rh) fp32`` session
buffer itself (lh in columns ``[0, n_lh)``), which is what step 1's
fused subject leaf wants.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import ctypes
import os
import re
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from arealmshbm.data_io._gifti_kernels_gpu import get_gifti_gpu_kernels
from arealmshbm.data_io._nvcomp_batched import get_deflate_batch

PathLike = "str | Path"

# Contract literals — same values the CPU reader enforces.
_EXPECTED_DTYPE = b"NIFTI_TYPE_FLOAT32"
_EXPECTED_ENC = b"GZipBase64Binary"
_EXPECTED_END = b"LittleEndian"
_RE_DIM0 = re.compile(rb'Dim0="(\d+)"')
_RE_DATATYPE = re.compile(rb'DataType="([^"]+)"')
_RE_ENCODING = re.compile(rb'Encoding="([^"]+)"')
_RE_ENDIAN = re.compile(rb'Endian="([^"]+)"')

#: Alignment + minimum zero-pad between concatenated files; the pad is
#: what makes the single-launch batched bytescan safe.
_FILE_ALIGN = 512
_MIN_PAD = 16

#: Groups that may keep their cross-stream base64 buffer alive at once;
#: 2 is the double-buffer minimum that still overlaps prep and decode.
_MAX_INFLIGHT = 2

#: Poison for the nvCOMP per-chunk report arrays, so ``_verify`` can
#: never pass on memory nvCOMP did not write.
_BAD_LEN = (1 << 64) - 1
_BAD_STATUS = -1

#: ``(event, keep)`` pairs an ABANDONED generator may still be decoding
#: out of. Each entry carries the event **its own** group was recorded
#: on, so draining is correct no matter which stream the next call runs
#: on (syncing "the current call's stream" says nothing about the
#: stream the parked buffers were used on).
_PENDING_KEEPS: list = []
_PENDING_LOCK = threading.Lock()

_STREAM_CACHE: dict = {}

#: Reusable pinned staging buffer, so a cohort pays ``cudaHostAlloc``
#: once. ``busy`` makes a concurrent second caller fall back to a
#: private allocation instead of corrupting the shared one.
_STAGING: dict = {"mem": None, "size": 0, "busy": False}
_STAGING_LOCK = threading.Lock()


def _drain_pending():
    """Wait out every parked in-flight group and release it."""
    with _PENDING_LOCK:
        pend = list(_PENDING_KEEPS)
        del _PENDING_KEEPS[:]
    if not pend:
        return
    for evt, _keep in pend:
        if evt is not None:
            evt.synchronize()
    if any(evt is None for evt, _ in pend):
        # Parked by an error path before its event was recorded — we
        # don't know which stream is reading, so sync the device.
        import cupy as cp
        cp.cuda.Device().synchronize()


def _acquire_staging(nbytes: int):
    """Claim the shared pinned staging buffer (or a private one).

    Returns ``(pinned_memory, cached)``; ``cached=True`` obliges the
    caller to call :func:`_release_staging`, ``False`` does not.
    """
    import cupy as cp
    with _STAGING_LOCK:
        if _STAGING["busy"]:
            return cp.cuda.alloc_pinned_memory(nbytes), False
        _STAGING["busy"] = True
        mem, size = _STAGING["mem"], _STAGING["size"]
        if mem is not None and size >= nbytes:
            return mem, True
        # Drop EVERY reference to the old block — the dict's and this
        # frame's — before allocating, or cupy's pinned pool cannot
        # recycle it into the larger request and both stay resident.
        _STAGING["mem"] = None
        _STAGING["size"] = 0
        mem = None
    try:
        mem = cp.cuda.alloc_pinned_memory(nbytes)
    except Exception:
        with _STAGING_LOCK:
            _STAGING["busy"] = False
        raise
    with _STAGING_LOCK:
        _STAGING["mem"] = mem
        # The pool rounds up to the next power of two; record what we
        # actually got so a slightly larger next request reuses it.
        _STAGING["size"] = int(mem.mem.size)
    return mem, True


def _release_staging():
    with _STAGING_LOCK:
        _STAGING["busy"] = False


def release_staging_buffer():
    """Drop the cached pinned staging buffer; the next ingest re-allocates."""
    with _STAGING_LOCK:
        if _STAGING["busy"]:
            raise RuntimeError(
                "release_staging_buffer(): an ingest is in flight")
        _STAGING["mem"] = None
        _STAGING["size"] = 0


def prewarm_subject_bold_gpu(session_paths=None, *, nbytes=None):
    """Pay the NVRTC JIT, ctypes load and ``cudaHostAlloc`` up front.

    Synchronous: it claims and releases the shared staging buffer, so
    running it concurrently with an ingest would make that ingest fall
    back to a private whole-subject pinned block. Callers serialise
    through ``generate_profiles.profiles_subject_gpu._PREWARM_LOCK``.

    Parameters
    ----------
    session_paths : sequence of (lh, rh), optional
        Used only to size the staging buffer; the files are ``stat``-ed,
        never read.
    nbytes : int, optional
        Explicit staging size when the paths are not known yet;
        ``max(nbytes, sum of the padded file sizes)`` is allocated.
    """
    want = int(nbytes or 0)
    if session_paths:
        paths = [p for pair in session_paths for p in pair]
        tot = 0
        for p in paths:
            s = os.path.getsize(p)
            tot += ((s + _MIN_PAD + _FILE_ALIGN - 1)
                    // _FILE_ALIGN) * _FILE_ALIGN
        want = max(want, tot)
    if want <= 0:
        raise ValueError("prewarm_subject_bold_gpu: pass session_paths "
                         "or nbytes")

    # Runs on the caller's thread, so the base64 lookup table is cached
    # on the caller's device — which is the one the ingest will use.
    import cupy as cp
    get_gifti_gpu_kernels()
    get_deflate_batch()
    # The first CuPy reduction the reader runs pays a module load; a
    # tiny ``flatnonzero`` warms the same code path.
    int(cp.flatnonzero(cp.zeros(8, dtype=cp.bool_)).shape[0])
    _mem, cached = _acquire_staging(want)
    if cached:
        _release_staging()


def _default_stream():
    """Module-owned non-blocking decode stream, created once."""
    import cupy as cp
    st = _STREAM_CACHE.get("s")
    if st is None:
        st = cp.cuda.Stream(non_blocking=True)
        _STREAM_CACHE["s"] = st
    return st


def _prep_stream():
    """Module-owned stream for the prep half.

    Separate from the decode stream so prep's two host stalls never
    wait on the previous group's nvCOMP decompression.
    """
    import cupy as cp
    st = _STREAM_CACHE.get("prep")
    if st is None:
        st = cp.cuda.Stream(non_blocking=True)
        _STREAM_CACHE["prep"] = st
    return st


def _readinto(path, host_view, off, nbytes):
    """``f.readinto`` one file into the pinned staging buffer.

    ``readinto`` skips the intermediate ``bytes`` of ``f.read()`` and
    releases the GIL, so a pool of these actually overlaps.
    """
    with open(path, "rb") as f:
        got = f.readinto(memoryview(host_view)[off:off + nbytes])
    if got != nbytes:
        raise ValueError(
            f"GIFTI {path}: short read ({got} of {nbytes} bytes) — file "
            f"changed size between stat and read?"
        )


def _validate_headers(host_buf, da_starts, data_starts, N, path):
    """Per-``<DataArray>`` contract check against the staged host bytes.

    Same contract ``gifti_io._scan_gifti_spans`` enforces; the first
    darray goes through regex so the error names the attribute.
    """
    first = bytes(host_buf[int(da_starts[0]):int(data_starts[0])])
    dt = _RE_DATATYPE.search(first)
    if not dt or dt.group(1) != _EXPECTED_DTYPE:
        raise ValueError(
            f"GIFTI {path}: DataType="
            f"{dt.group(1).decode() if dt else 'MISSING'!r}; "
            f"requires {_EXPECTED_DTYPE.decode()!r}")
    enc = _RE_ENCODING.search(first)
    if not enc or enc.group(1) != _EXPECTED_ENC:
        raise ValueError(
            f"GIFTI {path}: Encoding="
            f"{enc.group(1).decode() if enc else 'MISSING'!r}; "
            f"requires {_EXPECTED_ENC.decode()!r}")
    en = _RE_ENDIAN.search(first)
    if not en or en.group(1) != _EXPECTED_END:
        raise ValueError(
            f"GIFTI {path}: Endian="
            f"{en.group(1).decode() if en else 'MISSING'!r}; "
            f"requires {_EXPECTED_END.decode()!r}")
    m = _RE_DIM0.search(first)
    if not m:
        raise ValueError(f"GIFTI {path}: missing Dim0 attribute")
    n_dim0 = int(m.group(1))
    if N is not None and n_dim0 != N:
        raise ValueError(
            f"GIFTI {path}: Dim0={n_dim0} != expected {N}")

    dtype_needle = b'DataType="' + _EXPECTED_DTYPE + b'"'
    enc_needle = b'Encoding="' + _EXPECTED_ENC + b'"'
    end_needle = b'Endian="' + _EXPECTED_END + b'"'
    dim0_needle = b'Dim0="' + str(n_dim0).encode() + b'"'
    for j in range(1, len(da_starts)):
        hdr = bytes(host_buf[int(da_starts[j]):int(data_starts[j])])
        if (dtype_needle not in hdr or enc_needle not in hdr
                or end_needle not in hdr or dim0_needle not in hdr):
            raise ValueError(
                f"GIFTI {path}: DataArray {j} contract violation "
                f"(DataType/Encoding/Endian/Dim0)")
    return n_dim0


def _cortex_index(medial_mask_d, n_full):
    """``(N_cortex,) int32`` device ``flatnonzero(~mask)``: kept rows."""
    import cupy as cp
    if not isinstance(medial_mask_d, cp.ndarray):
        raise TypeError(
            "read_subject_bold_gpu: medial_mask_d must be a cupy.ndarray "
            "(upload once with ``cp.asarray(host_mask)``); got "
            f"{type(medial_mask_d).__name__}")
    mask = medial_mask_d.reshape(-1)
    if int(mask.shape[0]) != int(n_full):
        raise ValueError(
            f"read_subject_bold_gpu: medial_mask length {int(mask.shape[0])}"
            f" != n_lh + n_rh = {int(n_full)}")
    return cp.flatnonzero(~mask.astype(cp.bool_)).astype(cp.int32)


#: Shared pool for the ``readinto`` stage. Its tasks never block on the
#: pool, so no nested-pool guard is needed.
_READ_POOL: dict = {}
_READ_POOL_LOCK = threading.Lock()


def _read_pool():
    """Module-owned ``ThreadPoolExecutor``, created on first ingest."""
    pool = _READ_POOL.get("p")
    if pool is not None:
        return pool
    with _READ_POOL_LOCK:
        pool = _READ_POOL.get("p")
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor
            pool = _READ_POOL["p"] = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="gifti-ingest")
        return pool


def _ingest_groups(pairs, *, medial_mask_d=None, n_lh, n_rh,
                   prep_stream, dec_stream, group_sessions,
                   raw_sessions=False):
    """Decode one subject group by group.

    Yields ``(sess_indices, arrays, event, verify)``: the sessions of
    one group, the event recorded on ``dec_stream`` after it, and the
    decompressed-length check, which must run *after* that event.

    ``raw_sessions=True`` hands back the **pre-epilogue** per-session
    buffer instead: ``(T, n_lh + n_rh) fp32`` C-contig on device, lh in
    columns ``[0, n_lh)`` and rh in ``[n_lh, n_full)``, with no medial
    drop, no transpose and no NaN cleaning. ``medial_mask_d`` is then
    unused (and may be ``None``).

    Whatever is still in flight when the generator finishes — or when
    an abandoned generator is closed — is parked in
    :data:`_PENDING_KEEPS` **with its own event**, so a buffer the
    decode stream may still be reading is released against the right
    stream by :func:`_drain_pending`.
    """
    import cupy as cp

    _drain_pending()

    paths = [p for pair in pairs for p in pair]
    n_full = int(n_lh) + int(n_rh)
    n_sess = len(pairs)
    gs = n_sess if group_sessions is None else max(1, int(group_sessions))
    expected_n = [int(n_lh), int(n_rh)] * n_sess

    find_tags, b64_kernel, epilogue, b64_table = get_gifti_gpu_kernels()
    nvbatch = get_deflate_batch()
    read_pool = _read_pool()

    # ── pinned staging: every file's raw bytes, 512 B-aligned ──
    sizes = [os.path.getsize(p) for p in paths]
    padded = [((s + _MIN_PAD + _FILE_ALIGN - 1) // _FILE_ALIGN) * _FILE_ALIGN
              for s in sizes]
    offsets = []
    cur = 0
    for ps in padded:
        offsets.append(cur)
        cur += ps
    total = cur

    pinned = None
    host = None
    staging_cached = False
    read_futs: list = []
    inflight: list = []     # [(event, [cross-stream keeps])], bounded
    try:
        if total:
            # The staging block covers the WHOLE subject, so an OOM here
            # is a host-pinned limit, not a VRAM one; cudaHostAlloc's own
            # error reads exactly like a GPU OOM.
            try:
                pinned, staging_cached = _acquire_staging(total)
            except cp.cuda.runtime.CUDARuntimeError as exc:
                raise RuntimeError(
                    f"read_subject_bold_gpu: could not pin {total} bytes of "
                    f"host staging for this subject's {len(paths)} "
                    f"``.func.gii`` files ({exc}). The whole subject is "
                    f"staged at once; stage fewer runs per subject, or run "
                    f"step 1 with ``backend_step1='cpu'``."
                ) from exc
            host = np.frombuffer(pinned, dtype=np.uint8, count=total)
            # The gaps MUST be zeroed: cupy's pinned pool hands back
            # recycled memory, so a previous call's bytes would make the
            # batched tag scan report a tag belonging to no file here.
            ends = offsets[1:] + [total]
            for off, s, nxt in zip(offsets, sizes, ends):
                host[off + s:nxt] = 0
            read_futs = [read_pool.submit(_readinto, p, host, o, s)
                         for p, o, s in zip(paths, offsets, sizes)]

        if raw_sessions:
            # No epilogue -> no gather map, and no mask required.
            cidx = None
            n_cortex = 0
        else:
            with dec_stream:
                # Allocated *and* consumed on ``dec`` (the epilogue
                # reads it), so it never crosses streams and needs no
                # keepalive.
                cidx = _cortex_index(medial_mask_d, n_full)
            n_cortex = int(cidx.shape[0])
        t_subject = None

        for g0 in range(0, n_sess, gs):
            g1 = min(g0 + gs, n_sess)
            f0, f1 = 2 * g0, 2 * g1
            n_gf = f1 - f0          # files in this group (2 per session)

            # Bound the cross-stream retention to a double buffer.
            while len(inflight) >= _MAX_INFLIGHT:
                old_evt, _old = inflight.pop(0)
                old_evt.synchronize()

            decoded = None
            T0 = None
            out_off = real = None

            for fu in read_futs[f0:f1]:
                fu.result()
            byte0 = offsets[f0]
            byte1 = offsets[f1 - 1] + padded[f1 - 1]
            span = byte1 - byte0

            with prep_stream:
                # ── H2D of this group's byte span (async, pinned) ──
                dev = cp.empty(span, dtype=cp.uint8)
                dev.data.copy_from_host_async(
                    ctypes.c_void_p(pinned.ptr + byte0), span,
                    prep_stream)

                # ── one tag scan for the whole group ──
                # On overflow the kernel's ``atomicAdd`` counter still
                # counts every tag, so one retry at the exact count fits.
                max_found = 4096 * n_gf
                for _attempt in range(2):
                    pos_d = cp.empty(max_found, dtype=cp.int64)
                    kind_d = cp.empty(max_found, dtype=cp.int32)
                    n_found_d = cp.zeros(1, dtype=cp.int32)
                    block = 256
                    find_tags(((span + block - 1) // block,), (block,),
                              (dev, np.int64(span), pos_d, kind_d,
                               n_found_d, np.int32(max_found)))
                    n_found = int(cp.asnumpy(n_found_d)[0])
                    if n_found <= max_found:
                        break
                    pos_d = kind_d = None
                    max_found = n_found
                else:
                    raise ValueError(
                        f"read_subject_bold_gpu: tag scan found {n_found} "
                        f"tags > cap {max_found} even after resizing")
                positions = cp.asnumpy(pos_d[:n_found])
                kinds = cp.asnumpy(kind_d[:n_found])
                prep_stream.synchronize()

                order = np.argsort(positions, kind="stable")
                positions = positions[order]
                kinds = kinds[order]

                # ── host bookkeeping, vectorized ──
                local_starts = (np.asarray(offsets[f0:f1], dtype=np.int64)
                                - byte0)
                owner = np.searchsorted(local_starts, positions,
                                        side="right") - 1
                sel = [(positions[kinds == k], owner[kinds == k])
                       for k in (0, 1, 2)]
                chunk_starts, chunk_lens, per_file_T = [], [], []
                hv = host[byte0:byte1]
                for j in range(n_gf):
                    da = sel[0][0][sel[0][1] == j]
                    ds = sel[1][0][sel[1][1] == j]
                    dc = sel[2][0][sel[2][1] == j]
                    T = int(ds.shape[0])
                    p = paths[f0 + j]
                    if T == 0:
                        raise ValueError(
                            f"GIFTI {p}: no <DataArray>/<Data> pair found")
                    if da.shape[0] != T or dc.shape[0] != T:
                        raise ValueError(
                            f"GIFTI {p}: tag counts mismatch — "
                            f"DataArray={da.shape[0]} Data={T} "
                            f"/Data={dc.shape[0]}")
                    _validate_headers(hv, da, ds, expected_n[f0 + j], p)
                    cs = ds.astype(np.int64) + 6      # len('<Data>')
                    cl = dc.astype(np.int64) - cs
                    if np.any(cl < 0):
                        bad = int(np.flatnonzero(cl < 0)[0])
                        raise ValueError(
                            f"GIFTI {p}: chunk {bad} has negative b64 "
                            f"length — malformed <Data>/</Data> pairing")
                    if np.any(cl % 4 != 0):
                        bad = int(np.flatnonzero(cl % 4 != 0)[0])
                        raise ValueError(
                            f"GIFTI {p}: chunk {bad} b64 length "
                            f"{int(cl[bad])} not a multiple of 4")
                    chunk_starts.append(cs)
                    chunk_lens.append(cl)
                    per_file_T.append(T)

                T0 = per_file_T[0]
                if any(t != T0 for t in per_file_T):
                    raise ValueError(
                        f"read_subject_bold_gpu: all files of a subject "
                        f"must share T; got {per_file_T}")

                starts_all = np.concatenate(chunk_starts)
                lens_all = np.concatenate(chunk_lens)
                n_chunks = int(starts_all.shape[0])

                # '=' padding → real decoded length.
                cends = starts_all + lens_all
                pad = (hv[cends - 1] == 61).astype(np.int64)
                pad += pad * (hv[cends - 2] == 61)
                aligned = (lens_all >> 2) * 3
                real = aligned - pad
                if np.any(real < 6):
                    bad = int(np.flatnonzero(real < 6)[0])
                    raise ValueError(
                        f"GIFTI {paths[f0 + bad // T0]}: DataArray "
                        f"{bad % T0} <Data> decodes to {int(real[bad])} "
                        f"bytes, too short to be a zlib stream (2-byte "
                        f"header + 4-byte adler32)")
                # Not a tight cumsum: nvCOMP wants each chunk's start
                # ADDRESS (``offset + 2``, past the zlib header) 4-byte
                # aligned, which the planner's <= 3 B pad guarantees.
                out_off, decoded_bytes = nvbatch.plan(aligned,
                                                      header_skip=2)

                # ── batched base64 + alphabet validation ──
                decoded = cp.empty(decoded_bytes, dtype=cp.uint8)
                err_d = cp.full(1, np.iinfo(np.int32).max, dtype=cp.int32)
                bblock = 256
                grid_y = ((int((lens_all.max() + 3) >> 2) + bblock - 1)
                          // bblock)
                b64_kernel((n_chunks, grid_y), (bblock,),
                           (b64_table, dev, cp.asarray(starts_all),
                            cp.asarray(lens_all.astype(np.int32)),
                            decoded, cp.asarray(out_off), err_d))
                err = int(cp.asnumpy(err_d)[0])
                if err != np.iinfo(np.int32).max:
                    raise ValueError(
                        f"GIFTI {paths[f0 + err // T0]}: DataArray "
                        f"{err % T0} <Data> payload contains a byte "
                        f"outside the base64 alphabet (wrapped / "
                        f"whitespace-padded b64), or a '=' pad outside "
                        f"the final quad. This reader requires "
                        f"unwrapped, single-line GZipBase64Binary; "
                        f"re-run step 1 with ``backend_step1='cpu'`` "
                        f"(its reader accepts wrapped payloads), or "
                        f"unwrap the file.")
                # zlib FLG bit 5 (FDICT): a 4-byte DICTID would follow
                # the 2-byte header, and the "+2" skip below would hand
                # it to nvCOMP as DEFLATE data. No writer here sets it;
                # refuse by name rather than misparse.
                fdict = cp.asnumpy(decoded[cp.asarray(out_off + 1)] & 0x20)
                if fdict.any():
                    j = int(np.flatnonzero(fdict)[0])
                    raise ValueError(
                        f"GIFTI {paths[f0 + j // T0]}: DataArray {j % T0} "
                        f"zlib stream has FDICT set (preset dictionary); "
                        f"this reader inflates bare DEFLATE streams and "
                        f"cannot apply one.")
                del dev
                prep_evt = prep_stream.record()

            dec_stream.wait_event(prep_evt)

            if t_subject is None:
                t_subject = T0
            elif T0 != t_subject:
                raise ValueError(
                    f"read_subject_bold_gpu: all files of a subject must "
                    f"share T; group {g0}..{g1 - 1} has T={T0}, earlier "
                    f"groups had T={t_subject}")

            with dec_stream:
                sess_bufs = [cp.empty((T0, n_full), dtype=cp.float32)
                             for _ in range(g1 - g0)]
                keeps: list = []
                inflight.append((None, keeps))   # event filled in below
                lh_bytes = int(n_lh) * 4
                # (global file index, session slot, hemisphere) per file.
                gpu_slots = [(fi, fi // 2 - g0, fi % 2)
                             for fi in range(f0, f1)]

                # ── nvCOMP straight into the session buffers ──
                # The batched C entry point takes DEVICE arrays of
                # pointers and sizes, so the launch is async.
                base = int(decoded.data.ptr)
                n_ch = n_gf * T0
                dst_h = np.empty(n_ch, dtype=np.uint64)
                cap_h = np.empty(n_ch, dtype=np.uint64)
                row_off = (np.arange(T0, dtype=np.int64)
                           * (n_full * 4)).astype(np.uint64)
                for j, (fi, si, hemi) in enumerate(gpu_slots):
                    o0 = 0 if hemi == 0 else lh_bytes
                    sl = slice(j * T0, (j + 1) * T0)
                    dst_h[sl] = (np.uint64(int(sess_bufs[si].data.ptr) + o0)
                                 + row_off)
                    cap_h[sl] = np.uint64(expected_n[fi] * 4)
                # +2 skips the zlib CMF/FLG header, -6 also drops the
                # adler32 trailer: nvCOMP's RAW kind wants bare DEFLATE.
                src_ptrs = cp.asarray((out_off + 2 + base).astype(np.uint64))
                src_bytes = cp.asarray((real - 6).astype(np.uint64))
                dst_ptrs = cp.asarray(dst_h)
                dst_cap = cp.asarray(cap_h)
                # Poisoned, not ``cp.empty``: recycled pool memory can
                # legitimately read as "correct length, status 0".
                actual = cp.full(n_ch, _BAD_LEN, dtype=cp.uint64)
                statuses = cp.full(n_ch, _BAD_STATUS, dtype=cp.int32)
                tbytes = nvbatch.temp_bytes(n_ch, int(cap_h.max()),
                                            int(cap_h.sum()))
                temp = cp.empty(tbytes, dtype=cp.uint8) if tbytes else None
                # ``decoded`` crosses streams, so it is registered
                # BEFORE the launch, not after.
                keeps.append(decoded)
                keeps.extend((src_ptrs, src_bytes, dst_ptrs, dst_cap,
                              temp))
                nvbatch.decompress_async(
                    src_ptrs=src_ptrs, src_bytes=src_bytes,
                    dst_ptrs=dst_ptrs, dst_capacity=dst_cap,
                    actual_bytes=actual, statuses=statuses,
                    temp=temp, stream=dec_stream)
                decoded = None
                check = (actual, statuses, dst_cap, tuple(gpu_slots))

                def _verify(check=check, T0=T0):
                    """Enforce the decompressed-length contract exactly.

                    GPU twin of the CPU reader's ``arr.shape[0] != N``.
                    Must run after the group's event; it does not
                    synchronise on its own account.
                    """
                    act, stat, cap, slots = check
                    bad = cp.flatnonzero((act != cap) | (stat != 0))
                    if int(bad.shape[0]) == 0:
                        return
                    idx = int(cp.asnumpy(bad[:1])[0])
                    j, t = divmod(idx, T0)
                    fi = slots[j][0]
                    got = int(cp.asnumpy(act[idx:idx + 1])[0])
                    code = int(cp.asnumpy(stat[idx:idx + 1])[0])
                    want = int(expected_n[fi]) * 4
                    raise ValueError(
                        f"GIFTI {paths[fi]}: DataArray {t} decompressed to "
                        f"{got} bytes != expected {want} "
                        f"({expected_n[fi]} x fp32) [nvCOMP status "
                        f"{nvbatch.status_name(code)}]; Dim0 disagrees with "
                        f"the decoded byte count "
                        f"({int(bad.shape[0])} chunk(s) affected)")

                if raw_sessions:
                    # Hand back the session buffers themselves: they
                    # already ARE (T, lh|rh) fp32 C-contig on ``dec``.
                    arrays = list(sess_bufs)
                else:
                    # ── fused NaN-clean + medial gather + transpose ──
                    arrays = []
                    gx = (n_cortex + 31) // 32
                    gy = (T0 + 31) // 32
                    for b in sess_bufs:
                        dst_a = cp.empty((n_cortex, T0), dtype=cp.float32)
                        epilogue((gx, gy), (32, 8),
                                 (b, np.int32(T0), np.int32(n_full), cidx,
                                  np.int32(n_cortex), dst_a))
                        arrays.append(dst_a)
                dec_evt = dec_stream.record()
                inflight[-1] = (dec_evt, keeps)

            yield list(range(g0, g1)), arrays, dec_evt, _verify
    finally:
        # Every outstanding ``readinto`` writes into the SHARED pinned
        # staging buffer, so none may outlive this call.
        for fu in read_futs:
            fu.cancel()
        for fu in read_futs:
            if not fu.cancelled():
                try:
                    fu.result()
                except Exception:
                    pass
        # Anything still decoding is parked with ITS OWN event, so an
        # abandoned generator can never free a buffer the decode stream
        # is still reading; :func:`_drain_pending` releases it against
        # the right stream.
        if inflight:
            with _PENDING_LOCK:
                _PENDING_KEEPS.extend(inflight)
            del inflight[:]
        if staging_cached:
            _release_staging()


def read_subject_bold_gpu(session_paths,
                          *,
                          medial_mask_d=None,
                          n_lh: int,
                          n_rh: int,
                          stream=None,
                          group_sessions: Optional[int] = 2,
                          raw_sessions: bool = False):
    """Decode one subject's sessions straight to device ``(N_cortex, T)``.

    Parameters
    ----------
    session_paths : sequence of (lh_path, rh_path)
        One ``.func.gii`` pair per session, in session order. Payloads
        must be unwrapped single-line ``GZipBase64Binary``; a wrapped
        one raises (run step 1 with ``backend_step1='cpu'``, whose
        reader accepts it).
    medial_mask_d : cupy.ndarray, shape ``(n_lh + n_rh,)``
        Truthy = medial wall (dropped); must already live on device.
        Unused (and optional) when ``raw_sessions=True``.
    n_lh, n_rh : int
        Expected per-hemi vertex counts, enforced against each file's
        ``Dim0`` and against the decompressed byte count.
    stream : cupy.cuda.Stream, optional
        Stream the **decode** half runs on and the returned arrays are
        produced on; defaults to a module-owned non-blocking stream.
    group_sessions : int or None, optional
        Sessions per pipeline group; ``None`` = one group, no overlap.
    raw_sessions : bool, optional
        Return the **pre-epilogue** session buffer instead:
        ``(T, n_lh + n_rh) fp32`` C-contig on device, lh in columns
        ``[0, n_lh)`` and rh in ``[n_lh, n_full)``, no medial drop, no
        transpose, no NaN cleaning.

    Returns
    -------
    list of cupy.ndarray, ``(N_cortex, T) fp32``
        One per session, in input order (``(T, n_lh + n_rh)`` with
        ``raw_sessions=True``). Host-synchronous, so the arrays are
        usable from any stream on return.
    """
    pairs = [(Path(a), Path(b)) for a, b in session_paths]
    if not pairs:
        return []

    dec_stream = stream if stream is not None else _default_stream()
    results: List = [None] * len(pairs)
    verifies: list = []
    last_evt = None
    for idxs, arrays, evt, verify in _ingest_groups(
            pairs, medial_mask_d=medial_mask_d, n_lh=n_lh, n_rh=n_rh,
            prep_stream=_prep_stream(), dec_stream=dec_stream,
            group_sessions=group_sessions, raw_sessions=raw_sessions):
        for i, a in zip(idxs, arrays):
            results[i] = a
        verifies.append(verify)
        last_evt = evt
    last_evt.synchronize()
    for v in verifies:
        v()
    # Decodes are complete, so the last groups' cross-stream buffers
    # can go back to the pool now.
    _drain_pending()
    return results


def iter_subject_bold_gpu(session_paths,
                          *,
                          medial_mask_d=None,
                          n_lh: int,
                          n_rh: int,
                          stream=None,
                          group_sessions: int = 1,
                          raw_sessions: bool = False):
    """Pipelined variant — yield sessions as soon as each group lands.

    Yields ``(sess_idx, dev_array)``, 0-based and in input order. The
    generator waits only on the current group's completion event while
    the remaining groups are already in flight on the two streams, so
    the first session is available long before a whole-subject call
    would return. Each array is yielded only after its group's event
    has been synchronised, so the consumer may use it from any stream.

    Transients are bounded at :data:`_MAX_INFLIGHT` groups, so the live
    device set is *outputs the consumer still holds* plus at most two
    groups' base64 keeps, independent of the session count. Output
    lifetime is the consumer's problem: each array is allocated on the
    decode stream, so hold it until the kernels that read it have
    completed.

    Parameters are as :func:`read_subject_bold_gpu`, including
    ``raw_sessions``.
    """
    pairs = [(Path(a), Path(b)) for a, b in session_paths]
    if not pairs:
        return

    dec_stream = stream if stream is not None else _default_stream()
    for idxs, arrays, evt, verify in _ingest_groups(
            pairs, medial_mask_d=medial_mask_d, n_lh=n_lh, n_rh=n_rh,
            prep_stream=_prep_stream(), dec_stream=dec_stream,
            group_sessions=group_sessions, raw_sessions=raw_sessions):
        evt.synchronize()
        verify()
        for i, a in zip(idxs, arrays):
            yield i, a
    _drain_pending()
