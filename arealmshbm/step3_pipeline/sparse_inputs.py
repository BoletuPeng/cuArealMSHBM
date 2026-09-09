"""sparse_inputs.py

Fast, sparse-aware input path for the step-3 ``gpu_sparse`` backend.

The dense loader (:meth:`arealmshbm.step3_pipeline.pipeline.Step3Pipeline.load_inputs`)
materialises three (N, L) fp32/fp64 arrays — θ (98 MB), the boundary
mask (197 MB fp64) and ``spatial_xyz_vmf`` — of which ~1 % is nonzero,
plus a (T, N, D_b) → (N, T, D_b) host transpose of the packed BOLD.
``gpu_sparse`` wants none of that: it consumes a
:class:`~arealmshbm.vmf_clustering.sparse_layout.CandidateLayout` over
the ``P = nnz(θ)`` candidate cells and the on-disk ``(T, N, D_b)``
packed BOLD.

This module reads exactly the same bytes and produces bit-identical
numbers, without ever forming an (N, L) array:

    load_group_prior_csr        Params_Final.mat → mu/epsil/sigma + θ as CSR.
                                Fast path walks the MAT Level-5 container
                                with :mod:`arealmshbm.data_io.mat5_stream`'s
                                primitives (streaming inflate + a numba
                                column-major nonzero scan that emits CSC);
                                falls back to ``scipy.io.loadmat`` +
                                dense→CSR for anything it does not handle
                                (v7.3, uncompressed, unexpected classes).
    load_spatial_mask_csr       spatial_mask_<mesh>.mat → two csr_matrix.
    build_candidate_layout_fast CSR θ + CSR masks → CandidateLayout,
                                equal (``layouts_equal``) to
                                ``build_candidate_layout_dense``.
    fetch_packed_bold_TND       cohort discovery → (T, N, D_b) uint8 as
                                stored (RAW: MW rows not zeroed), plus
                                the MW row index lists.
    fetch_gradient              cohort discovery → (N, Dg) fp32.
    load_step3_sparse_cohort    prior + layout — what is constant across a
                                cohort — as one shareable object.
    load_step3_sparse_inputs    the whole thing, mirroring
                                ``Step3Pipeline.load_inputs`` knob for knob;
                                takes an optional cohort object so a
                                multi-subject run loads it once.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numba import njit

# The MAT Level-5 container primitives live in the shared leaf; this
# module only adds the Params-struct walk and the streaming θ scan.
from arealmshbm.data_io.mat5_stream import (
    _MI_DTYPE, _InflateReader, _Unsupported, _as_array, _miCOMPRESSED,
    _miDOUBLE, _miMATRIX, _miSINGLE, _mxDOUBLE_CLASS, _mxSINGLE_CLASS,
    _mxSTRUCT_CLASS, _pad8, _parse_tag, _read_element,
)
from arealmshbm.vmf_clustering.sparse_layout import MAX_D_BYTES, CandidateLayout

# Which path the last ``load_group_prior_csr`` call took — 'fast' or
# 'scipy'. Purely informational (tests / timing reports read it).
LAST_PRIOR_PATH: str = ""


# ─────────────────────────────────────────────────────────────────────
# numba kernels
# ─────────────────────────────────────────────────────────────────────
@njit(nogil=True, cache=True)
def _scan_f32_colmajor(vals, start, nrows, out_row, out_val, colcnt, cnt):
    """Emit CSC entries for the nonzeros of a column-major fp32 slice."""
    for i in range(vals.size):
        v = vals[i]
        if v != np.float32(0.0):
            g = start + i
            out_row[cnt] = g % nrows
            out_val[cnt] = v
            colcnt[g // nrows] += 1
            cnt += 1
    return cnt


@njit(nogil=True, cache=True)
def _scan_f64_colmajor(vals, start, nrows, out_row, out_val, colcnt, cnt):
    """Same, for an fp64 payload — the fp32 cast happens *before* the
    zero test so the result matches ``dense.astype(np.float32) != 0``."""
    for i in range(vals.size):
        v = np.float32(vals[i])
        if v != np.float32(0.0):
            g = start + i
            out_row[cnt] = g % nrows
            out_val[cnt] = v
            colcnt[g // nrows] += 1
            cnt += 1
    return cnt


@njit(nogil=True, cache=True)
def _csc_to_csr(rows, cols, vals, row_ptr, out_col, out_val):
    """Stable CSC→CSR. Input entries are in (col, row) order, so each
    output row comes out with ascending columns."""
    pos = row_ptr[:-1].copy()
    for i in range(rows.size):
        r = rows[i]
        k = pos[r]
        out_col[k] = cols[i]
        out_val[k] = vals[i]
        pos[r] = k + 1


@njit(nogil=True, cache=True)
def _counting_sort_by_col(col, L, out_order):
    """Stable sort of CSR entry indices by column → CSC permutation."""
    cnt = np.zeros(L + 1, dtype=np.int64)
    for i in range(col.size):
        cnt[col[i] + 1] += 1
    for l in range(L):
        cnt[l + 1] += cnt[l]
    for i in range(col.size):
        c = col[i]
        out_order[cnt[c]] = i
        cnt[c] += 1


@njit(nogil=True, cache=True)
def _gather_bm(t_row_ptr, t_col, m_row_ptr, m_col, m_val, out):
    """For every θ entry of active row m, pick up the boundary-mask value
    at the same (row, col) by merging the two ascending column lists.
    Returns 0 on success, or 1 + the offending θ entry index."""
    M = t_row_ptr.size - 1
    for m in range(M):
        a, a_end = t_row_ptr[m], t_row_ptr[m + 1]
        b, b_end = m_row_ptr[m], m_row_ptr[m + 1]
        while a < a_end:
            c = t_col[a]
            while b < b_end and m_col[b] < c:
                b += 1
            if b >= b_end or m_col[b] != c:
                return 1 + a
            out[a] = np.float32(m_val[b])
            a += 1
    return 0


# ─────────────────────────────────────────────────────────────────────
# 1. group prior → CSR
# ─────────────────────────────────────────────────────────────────────
def _stream_theta_csc(rd: _InflateReader, mi_type: int, nbytes: int,
                      N: int, L: int):
    """Scan θ's column-major payload straight out of the inflate stream."""
    itemsize = 4 if mi_type == _miSINGLE else 8
    n_elem = nbytes // itemsize
    if n_elem != N * L:
        raise _Unsupported("theta payload size does not match dims")
    kernel = _scan_f32_colmajor if mi_type == _miSINGLE else _scan_f64_colmajor
    dt = np.float32 if mi_type == _miSINGLE else np.float64

    cap = max(1 << 16, n_elem // 64)
    out_row = np.empty(cap, dtype=np.int32)
    out_val = np.empty(cap, dtype=np.float32)
    colcnt = np.zeros(L, dtype=np.int64)
    state = {"cnt": 0, "start": 0, "row": out_row, "val": out_val, "cap": cap}

    def _consume(arr) -> None:
        k = arr.size
        if state["cnt"] + k > state["cap"]:
            new_cap = max(state["cap"] * 2, state["cnt"] + k)
            grown_row = np.empty(new_cap, dtype=np.int32)
            grown_val = np.empty(new_cap, dtype=np.float32)
            grown_row[:state["cnt"]] = state["row"][:state["cnt"]]
            grown_val[:state["cnt"]] = state["val"][:state["cnt"]]
            state["row"], state["val"], state["cap"] = (
                grown_row, grown_val, new_cap)
        state["cnt"] = kernel(arr, state["start"], N, state["row"],
                              state["val"], colcnt, state["cnt"])
        state["start"] += k

    # Scan straight out of the inflate buffers. Views are element-
    # aligned within the stream but not necessarily within the host
    # buffer; numpy/numba handle the unaligned loads.
    carry = b""
    for mv in rd.iter_raw(nbytes):
        if carry:
            need = itemsize - len(carry)
            if len(mv) < need:
                carry += bytes(mv)
                continue
            carry += bytes(mv[:need])
            _consume(np.frombuffer(carry, dtype=dt, count=1))
            carry = b""
            mv = mv[need:]
        nb = len(mv)
        rem = nb % itemsize
        if rem:
            carry = bytes(mv[nb - rem:])
            mv = mv[:nb - rem]
        k = (nb - rem) // itemsize
        if k:
            _consume(np.frombuffer(mv, dtype=dt, count=k))
    if carry:
        raise _Unsupported("theta payload is not a whole number of elements")
    rd.skip(_pad8(nbytes) - nbytes)
    cnt = state["cnt"]
    return state["row"][:cnt].copy(), state["val"][:cnt].copy(), colcnt


def _csc_arrays_to_csr(rows, vals, colcnt, N: int, L: int):
    """(CSC rows/vals/per-column counts) → (row_ptr, col, val) CSR."""
    P = int(rows.size)
    cols = np.repeat(np.arange(L, dtype=np.int32), colcnt)
    counts = np.bincount(rows, minlength=N)
    row_ptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    out_col = np.empty(P, dtype=np.int32)
    out_val = np.empty(P, dtype=np.float32)
    _csc_to_csr(rows, cols, vals, row_ptr, out_col, out_val)
    return row_ptr.astype(np.int32), out_col, out_val


def _theta_dense_to_csr(theta: np.ndarray):
    """Fallback: dense (N, L) fp32 → CSR over all rows."""
    th = np.ascontiguousarray(theta, dtype=np.float32)
    N, L = th.shape
    r, c = np.nonzero(th)                       # row-major ⇒ cols ascending
    counts = np.bincount(r, minlength=N)
    row_ptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return (row_ptr.astype(np.int32),
            np.ascontiguousarray(c, dtype=np.int32),
            np.ascontiguousarray(th[r, c], dtype=np.float32))


def _load_group_prior_csr_fast(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 136:
        raise _Unsupported("file too small")
    if raw[124:126] != b"\x00\x01" or raw[126:128] != b"IM":
        # subsys offset / endian indicator; 'IM' little-endian only.
        raise _Unsupported("not a little-endian MAT v5 container")

    off = 128
    fields: Dict[str, Any] = {}
    want = {"mu", "epsil", "sigma", "theta"}
    while off + 8 <= len(raw) and want - set(fields):
        typ, nb = struct.unpack("<II", raw[off:off + 8])
        if typ >> 16:
            raise _Unsupported("small-data element at container level")
        if typ != _miCOMPRESSED:
            raise _Unsupported(f"top-level element type {typ} unsupported")
        rd = _InflateReader(memoryview(raw)[off + 8:off + 8 + nb])
        _walk_top_matrix(rd, fields, want)
        off += 8 + _pad8(nb)

    missing = want - set(fields)
    if missing:
        raise _Unsupported(f"missing prior fields: {sorted(missing)}")
    return fields


def _walk_top_matrix(rd: _InflateReader, fields: Dict[str, Any],
                     want: set) -> None:
    """Walk one inflated top-level element, filling the wanted fields."""
    typ, nb, small = _parse_tag(rd.read(8))
    if small is not None or typ != _miMATRIX:
        raise _Unsupported("top-level element is not a miMATRIX")

    flags_t, flags, _ = _read_element(rd)
    fa = _as_array(flags_t, flags).astype(np.uint32)
    cls = int(fa[0] & 0xFF)
    complex_flag = bool(fa[0] & 0x0800)
    if cls != _mxSTRUCT_CLASS or complex_flag:
        raise _Unsupported("top-level variable is not a real struct")
    _read_element(rd)                                    # dims
    name_t, name, _ = _read_element(rd)
    if bytes(name) != b"Params":
        raise _Unsupported(f"unexpected struct name {bytes(name)!r}")
    _, fnl, _ = _read_element(rd)
    field_len = int(np.frombuffer(fnl, dtype=np.int32)[0])
    _, fnames_buf, _ = _read_element(rd)
    n_fields = len(fnames_buf) // field_len
    names = [bytes(fnames_buf[i * field_len:(i + 1) * field_len]).split(b"\0")[0]
             .decode("ascii") for i in range(n_fields)]

    for nm in names:
        ftyp, fnb, fsmall = _parse_tag(rd.read(8))
        if fsmall is not None:
            raise _Unsupported("struct field stored as small-data element")
        if ftyp != _miMATRIX:
            raise _Unsupported(f"struct field {nm!r} is not a miMATRIX")
        if nm not in want or nm in fields:
            rd.skip(_pad8(fnb))
            if not (want - set(fields)):
                return
            continue
        _read_struct_field(rd, nm, fnb, fields)
        if not (want - set(fields)):
            return


def _read_struct_field(rd: _InflateReader, nm: str, fnb: int,
                       fields: Dict[str, Any]) -> None:
    """Parse one wanted numeric field; θ is streamed, the rest read whole."""
    ftyp, fdata, consumed = _read_element(rd)
    fa = _as_array(ftyp, fdata).astype(np.uint32)
    cls = int(fa[0] & 0xFF)
    if int(fa[0] & 0x0800):
        raise _Unsupported(f"{nm}: complex arrays unsupported")
    if cls not in (_mxDOUBLE_CLASS, _mxSINGLE_CLASS):
        raise _Unsupported(f"{nm}: class {cls} unsupported")

    dtyp, ddata, used = _read_element(rd)
    consumed += used
    dims = _as_array(dtyp, ddata).astype(np.int64)
    if dims.size != 2:
        raise _Unsupported(f"{nm}: {dims.size}-D array unsupported")

    _, _, used = _read_element(rd)                       # field name (empty)
    consumed += used

    ptyp, pnb, psmall = _parse_tag(rd.read(8))
    consumed += 8
    if psmall is not None:
        payload = bytes(psmall)
        pnb_real = len(psmall)
    else:
        pnb_real = pnb
        payload = None
    if ptyp not in (_miSINGLE, _miDOUBLE):
        raise _Unsupported(f"{nm}: payload type {ptyp} unsupported")

    R, C = int(dims[0]), int(dims[1])
    if nm == "theta":
        if payload is not None:
            raise _Unsupported("theta stored as small-data element")
        rows, vals, colcnt = _stream_theta_csc(rd, ptyp, pnb_real, R, C)
        fields["theta"] = ("csc", rows, vals, colcnt, R, C)
        consumed += _pad8(pnb_real)
    else:
        if payload is None:
            payload = rd.read(pnb_real)
            rd.skip(_pad8(pnb_real) - pnb_real)
        arr = np.frombuffer(payload, dtype=_MI_DTYPE[ptyp])
        if arr.size != R * C:
            raise _Unsupported(f"{nm}: payload size mismatch")
        fields[nm] = np.asfortranarray(arr.reshape(R, C, order="F"))
        consumed += _pad8(pnb_real)
    if consumed > fnb:
        raise _Unsupported(f"{nm}: over-read the field element")
    rd.skip(fnb - consumed)


def load_group_prior_csr(prior_path: str | Path) -> Dict[str, Any]:
    """Read ``Params_Final.mat`` keeping θ sparse.

    Returns
    -------
    dict with keys
        ``mu``        — (D, L) fp32 C-contiguous
        ``epsil``     — (L,) fp32
        ``sigma``     — (L,) fp32
        ``theta_csr`` — ``(row_ptr (N+1,) int32, col (P,) int32,
                        val (P,) fp32)``; CSR over **all** N rows,
                        columns ascending inside each row
        ``N``, ``L``  — ints

    Values are bit-identical to :func:`data_io.load_group_prior` — the
    fp32 cast happens before the nonzero test, exactly as
    ``dense.astype(np.float32) != 0`` would.
    """
    global LAST_PRIOR_PATH
    p = Path(prior_path)
    if not p.exists():
        raise FileNotFoundError(f"group prior not found: {p}")

    try:
        f = _load_group_prior_csr_fast(p)
        _, rows, vals, colcnt, N, L = f["theta"]
        row_ptr, col, val = _csc_arrays_to_csr(rows, vals, colcnt, N, L)
        mu = np.ascontiguousarray(f["mu"], dtype=np.float32)
        epsil = np.asarray(f["epsil"]).reshape(-1).astype(np.float32)
        sigma = np.asarray(f["sigma"]).reshape(-1).astype(np.float32)
        if mu.shape[1] != L or epsil.size != L or sigma.size != L:
            raise _Unsupported("prior field shapes disagree")
        LAST_PRIOR_PATH = "fast"
        return {"mu": mu, "epsil": epsil, "sigma": sigma,
                "theta_csr": (row_ptr, col, val), "N": int(N), "L": int(L)}
    except _Unsupported:
        # Only "the walker does not model this layout" routes to scipy.
        # A corrupt payload, a MemoryError or a walker bug propagates
        # rather than silently succeeding on the 6x-slower dense path.
        LAST_PRIOR_PATH = "scipy"

    from arealmshbm.data_io import load_group_prior
    prior = load_group_prior(p)
    theta = prior["theta"]
    N, L = theta.shape
    return {
        "mu": np.ascontiguousarray(prior["mu"], dtype=np.float32),
        "epsil": np.asarray(prior["epsil"]).reshape(-1).astype(np.float32),
        "sigma": np.asarray(prior["sigma"]).reshape(-1).astype(np.float32),
        "theta_csr": _theta_dense_to_csr(theta),
        "N": int(N), "L": int(L),
    }


def theta_csr_to_dense(theta_csr, N: int, L: int) -> np.ndarray:
    """Densify a ``theta_csr`` triple — tests / debugging only."""
    row_ptr, col, val = theta_csr
    out = np.zeros((N, L), dtype=np.float32)
    rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(row_ptr))
    out[rows, col] = val
    return out


# ─────────────────────────────────────────────────────────────────────
# 2. spatial mask → CSR
# ─────────────────────────────────────────────────────────────────────
def load_spatial_mask_csr(mask_path: str | Path):
    """Read ``spatial_mask_<mesh>.mat`` as two fp64 ``csr_matrix``.

    Same values as :func:`data_io.load_spatial_mask`, never densified.
    Explicit stored zeros are dropped so ``nonzero`` support matches the
    dense loader's ``mask != 0``.
    """
    import scipy.sparse as sp

    p = Path(mask_path)
    if not p.exists():
        raise FileNotFoundError(f"spatial mask not found: {p}")

    def _to_csr(x):
        if sp.issparse(x):
            m = x.tocsr()
        else:
            m = sp.csr_matrix(np.asarray(x, dtype=np.float64))
        m = m.astype(np.float64)
        m.eliminate_zeros()
        m.sort_indices()
        return m

    try:
        from scipy.io import loadmat
        m = loadmat(p, squeeze_me=False)
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(p, "r") as f:
            return (_to_csr(_read_v73_boundary_sparse(f, "lh_boundary")),
                    _to_csr(_read_v73_boundary_sparse(f, "rh_boundary")))

    return _to_csr(m["lh_boundary"]), _to_csr(m["rh_boundary"])


def _read_v73_boundary_sparse(f, key: str):
    """v7.3 branch of :func:`data_io.load_spatial_mask._read_v73_boundary`,
    stopping at the sparse matrix instead of densifying."""
    import h5py
    import scipy.sparse as sp

    obj = f[key]
    if isinstance(obj, h5py.Group):
        data = np.asarray(obj["data"]).ravel()
        indices = np.asarray(obj["ir"]).ravel()
        indptr = np.asarray(obj["jc"]).ravel()
        n_rows_attr = obj.attrs.get("MATLAB_sparse")
        if n_rows_attr is None:
            raise ValueError(
                f"v7.3 sparse {key!r}: missing MATLAB_sparse attribute")
        n_rows = int(np.asarray(n_rows_attr).ravel()[0])
        n_cols = int(indptr.shape[0]) - 1
        return sp.csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))
    return sp.csr_matrix(np.ascontiguousarray(np.asarray(obj).T,
                                              dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────
# 3. candidate layout, from sparse inputs only
# ─────────────────────────────────────────────────────────────────────
def _build_neighborhood_active(lh_vertex_nbors, rh_vertex_nbors,
                               active: np.ndarray) -> np.ndarray:
    """``V_lambda.setup.build_neighborhood`` with the dense ``s_lambda``
    replaced by its per-row active flag. Identical output."""
    N_total = int(active.size)
    N_hemi = N_total // 2

    lh_keep = np.flatnonzero(active[:N_hemi])
    rh_keep = np.flatnonzero(active[N_hemi:])

    lh_remap = np.zeros(N_hemi + 1, dtype=np.int64)
    lh_remap[lh_keep + 1] = np.arange(1, lh_keep.size + 1, dtype=np.int64)
    rh_remap = np.zeros(N_hemi + 1, dtype=np.int64)
    rh_remap[rh_keep + 1] = np.arange(1, rh_keep.size + 1, dtype=np.int64)

    lh_nbh_keep = np.ascontiguousarray(lh_vertex_nbors[:, lh_keep], dtype=np.int64)
    rh_nbh_keep = np.ascontiguousarray(rh_vertex_nbors[:, rh_keep], dtype=np.int64)

    lh_remapped = lh_remap[np.clip(lh_nbh_keep, 0, N_hemi)]
    rh_remapped = rh_remap[np.clip(rh_nbh_keep, 0, N_hemi)]

    n_lh_active = lh_keep.size
    rh_remapped_global = np.where(rh_remapped > 0, rh_remapped + n_lh_active, 0)

    return np.ascontiguousarray(
        np.concatenate([lh_remapped, rh_remapped_global], axis=1),
        dtype=np.int64)


def build_candidate_layout_fast(theta_csr,
                                lh_mask_csr,
                                rh_mask_csr,
                                lh_vertex_nbors: np.ndarray,
                                rh_vertex_nbors: np.ndarray,
                                ) -> CandidateLayout:
    """Build a :class:`CandidateLayout` from CSR θ + per-hemi CSR masks.

    Equal (``layouts_equal``) to
    ``build_candidate_layout_dense(theta_dense, boundary_mask_dense, ...)``.
    """
    row_ptr_full, col_full, val_full = theta_csr
    row_ptr_full = np.asarray(row_ptr_full, dtype=np.int64)
    col_full = np.ascontiguousarray(col_full, dtype=np.int32)
    val_full = np.ascontiguousarray(val_full, dtype=np.float32)

    N = int(row_ptr_full.size) - 1
    n_lh, l_lh = lh_mask_csr.shape
    n_rh, l_rh = rh_mask_csr.shape
    if n_lh != n_rh or l_lh != l_rh:
        raise ValueError("lh/rh spatial masks must have the same shape")
    if n_lh + n_rh != N:
        raise ValueError(f"mask rows {n_lh}+{n_rh} != theta rows {N}")
    L = int(l_lh + l_rh)
    if N % 2 or L % 2:
        raise ValueError("N and L must be even (bilateral layout)")
    if val_full.size and (val_full < 0).any():
        raise ValueError("theta must be non-negative")

    counts = np.diff(row_ptr_full)
    active = counts != 0
    row_idx_active = np.flatnonzero(active).astype(np.int32)
    M = int(row_idx_active.size)
    inv_active = np.full(N, -1, dtype=np.int32)
    inv_active[row_idx_active] = np.arange(M, dtype=np.int32)

    act_counts = counts[active]
    row_ptr = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(act_counts, out=row_ptr[1:])
    P = int(row_ptr[-1])

    # Gather the active rows' θ entries (rows are ascending, so a plain
    # index-range gather preserves CSR order).
    take = np.repeat(row_ptr_full[:-1][active] - row_ptr[:-1], act_counts)
    take = np.arange(P, dtype=np.int64) + take
    col = np.ascontiguousarray(col_full[take], dtype=np.int32)
    theta_P = np.ascontiguousarray(val_full[take], dtype=np.float32)

    n_full = row_idx_active[np.repeat(np.arange(M, dtype=np.int64), act_counts)]
    if (((n_full < n_lh) & (col >= l_lh)) |
            ((n_full >= n_lh) & (col < l_lh))).any():
        raise ValueError("theta has cross-hemisphere candidates")

    # Boundary-mask support over the active rows, bilateral coords.
    lh_act = row_idx_active[row_idx_active < n_lh]
    rh_act = row_idx_active[row_idx_active >= n_lh] - n_lh
    lh_sub = lh_mask_csr[lh_act]
    rh_sub = rh_mask_csr[rh_act]
    lh_sub.sort_indices()
    rh_sub.sort_indices()
    bm_row_ptr = np.concatenate([
        np.asarray(lh_sub.indptr, dtype=np.int64),
        np.asarray(rh_sub.indptr, dtype=np.int64)[1:] + lh_sub.nnz,
    ])
    bm_col = np.concatenate([
        np.asarray(lh_sub.indices, dtype=np.int32),
        np.asarray(rh_sub.indices, dtype=np.int32) + np.int32(l_lh),
    ]).astype(np.int32)
    bm_val = np.concatenate([
        np.asarray(lh_sub.data, dtype=np.float64),
        np.asarray(rh_sub.data, dtype=np.float64),
    ])

    bm_P = np.empty(P, dtype=np.float32)
    rc = _gather_bm(row_ptr, col, bm_row_ptr, bm_col, bm_val, bm_P)
    if rc != 0:
        raise ValueError("supp(theta) must be inside supp(boundary_mask)")
    if (bm_P == 0).any():
        raise ValueError("supp(theta) must be inside supp(boundary_mask)")

    # CSC: entries sorted by (col, n). A stable counting sort by column
    # over the CSR order reproduces ``np.lexsort((n_full, col))``.
    order = np.empty(P, dtype=np.int64)
    _counting_sort_by_col(col, L, order)
    order = order.astype(np.int32)
    col_counts = np.bincount(col, minlength=L)
    col_ptr = np.zeros(L + 1, dtype=np.int64)
    np.cumsum(col_counts, out=col_ptr[1:])
    csc_row = np.ascontiguousarray(n_full[order], dtype=np.int32)
    csc_pidx = np.ascontiguousarray(order, dtype=np.int32)

    nbh_M1xM = _build_neighborhood_active(lh_vertex_nbors, rh_vertex_nbors,
                                          active)
    if nbh_M1xM.shape[1] != M:
        raise ValueError("neighborhood active count mismatch")
    neighborhood = np.ascontiguousarray(nbh_M1xM.T, dtype=np.int32)

    return CandidateLayout(
        N=N, L=L, M_active=M, P=P,
        row_idx_active=row_idx_active, inv_active=inv_active,
        row_ptr=row_ptr.astype(np.int32), col=col, theta=theta_P, bm=bm_P,
        col_ptr=col_ptr.astype(np.int32), csc_row=csc_row, csc_pidx=csc_pidx,
        neighborhood=neighborhood,
        bm_row_ptr=bm_row_ptr.astype(np.int32), bm_col=bm_col,
    )


# ─────────────────────────────────────────────────────────────────────
# 4. BOLD + gradient
# ─────────────────────────────────────────────────────────────────────
def _cohort_entry(project_dir: Path, subid: int, mesh: str):
    from arealmshbm.data_io.cohort import read_cohort
    cohort = read_cohort(project_dir)
    if cohort.mesh.get("targ") != mesh:
        raise ValueError(
            f"fetch: cohort mesh.targ={cohort.mesh.get('targ')!r} "
            f"!= caller mesh={mesh!r}")
    idx = int(subid) - 1
    if idx < 0 or idx >= cohort.num_sub:
        raise IndexError(
            f"fetch: subid={subid} out of range for cohort "
            f"(num_sub={cohort.num_sub})")
    return cohort.subjects[idx], idx


def fetch_packed_bold_TND(project_dir: str | Path,
                          num_session: int,
                          subid: int,
                          mesh: str,
                          lh_mesh: Dict[str, np.ndarray],
                          rh_mesh: Dict[str, np.ndarray],
                          ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Packed BOLD in the **on-disk** ``(T, N, ⌈D/8⌉)`` layout.

    Same discovery + validation as :func:`data_io.fetch_data`, but with
    no host transpose and no MW-zeroing pass — the device backend zeroes
    the MW rows itself from the returned index lists.

    Returns ``(packed_TND, D_unpacked, mw_lh_idx, mw_rh_idx)``; the MW
    index lists are 0-based into their own hemisphere's row block.
    """
    if not mesh.startswith("fsaverage"):
        raise ValueError(
            f"fetch_packed_bold_TND is fsaverage only; got mesh={mesh!r}")
    project_dir = Path(project_dir)
    T = int(num_session)

    n_lh = int(lh_mesh["MARS_label"].shape[0])
    n_rh = int(rh_mesh["MARS_label"].shape[0])
    N = n_lh + n_rh
    mw_lh_idx = np.flatnonzero(lh_mesh["MARS_label"] == 1).astype(np.int32)
    mw_rh_idx = np.flatnonzero(rh_mesh["MARS_label"] == 1).astype(np.int32)

    from arealmshbm.data_io.cohort import resolve_path
    from arealmshbm.data_io.profile_io import read_subject_profile_packed_tnd

    sub_entry, idx = _cohort_entry(project_dir, subid, mesh)
    if len(sub_entry.sessions) != T:
        raise ValueError(
            f"fetch_packed_bold_TND: cohort subjects[{idx}].sessions has "
            f"len={len(sub_entry.sessions)} != num_session={T}")
    if sub_entry.profile_b2nd is None:
        raise ValueError(
            f"fetch_packed_bold_TND: cohort subjects[{idx}] "
            f"(id={sub_entry.id!r}) has no profile_b2nd path.")
    b2nd_path = resolve_path(project_dir, sub_entry.profile_b2nd)
    if not b2nd_path.exists():
        raise FileNotFoundError(
            f"fetch_packed_bold_TND: profile_b2nd missing on disk: {b2nd_path}")

    packed, D = read_subject_profile_packed_tnd(b2nd_path)
    T_disk, N_disk, D_bytes = packed.shape
    if T_disk != T:
        raise ValueError(
            f"fetch_packed_bold_TND: .b2nd at {b2nd_path} has T={T_disk}; "
            f"num_session={T} mismatch")
    if N_disk != N:
        raise ValueError(
            f"fetch_packed_bold_TND: .b2nd at {b2nd_path} has N={N_disk}; "
            f"expected n_lh+n_rh={N}")
    expected_bytes = (int(D) + 7) // 8
    if D_bytes != expected_bytes:
        raise ValueError(
            f"fetch_packed_bold_TND: .b2nd at {b2nd_path} has "
            f"D_bytes={D_bytes}; expected ceil(D/8)={expected_bytes} for "
            f"D_unpacked={D}")
    if D_bytes > MAX_D_BYTES:
        raise ValueError(
            f"fetch_packed_bold_TND: profile dimension D={D} "
            f"(ceil(D/8)={D_bytes} > {MAX_D_BYTES} bytes) exceeds the "
            f"gpu_sparse kernels' limit of {MAX_D_BYTES * 8}; use "
            f"backend='gpu_full' or 'cpu'.")
    # The device ``acc_bits`` kernel folds all 8 bits of every byte with
    # no ``d < D`` clamp, so a set padding bit in the last byte would be
    # counted (and, on the final row, read past the allocation). Both
    # in-tree writers zero the padding; verify it once per subject —
    # a (T, N)-byte pass.
    pad = (-int(D)) % 8
    if pad and bool((packed[..., -1] >> np.uint8(8 - pad)).any()):
        raise ValueError(
            f"fetch_packed_bold_TND: {b2nd_path} (subject id="
            f"{sub_entry.id!r}) has nonzero padding bits in the last "
            f"packed byte (D={D}, {pad} pad bits); the gpu_sparse "
            f"kernels require them zeroed.")
    return (np.ascontiguousarray(packed), int(D), mw_lh_idx, mw_rh_idx)


def fetch_gradient(project_dir: str | Path,
                   subid: int,
                   mesh: str,
                   N: int,
                   n_grad_components: int = 100,
                   *,
                   precomputed_gradient_mat: Optional[np.ndarray] = None,
                   ) -> np.ndarray:
    """``fetch_data``'s ``gradient_mat``, with the same passthrough rule."""
    from arealmshbm.data_io.cohort import resolve_path
    from arealmshbm.data_io.fetch_data import _read_gradient_emb

    if precomputed_gradient_mat is not None:
        g = precomputed_gradient_mat
        if g.ndim != 2 or g.shape[0] != N:
            raise ValueError(
                f"fetch_gradient: precomputed_gradient_mat shape {g.shape} "
                f"incompatible with N={N}")
        if g.shape[1] < n_grad_components:
            raise ValueError(
                f"fetch_gradient: precomputed_gradient_mat has "
                f"{g.shape[1]} cols; need >= {n_grad_components}")
        return np.ascontiguousarray(g[:, :n_grad_components], dtype=np.float32)

    project_dir = Path(project_dir)
    sub_entry, idx = _cohort_entry(project_dir, subid, mesh)
    if sub_entry.gradient_lh is None or sub_entry.gradient_rh is None:
        raise ValueError(
            f"fetch_gradient: cohort subjects[{idx}] (id={sub_entry.id!r}) "
            f"has no gradient_lh/gradient_rh path.")
    lh_p = resolve_path(project_dir, sub_entry.gradient_lh)
    rh_p = resolve_path(project_dir, sub_entry.gradient_rh)
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_lh = ex.submit(_read_gradient_emb, lh_p, n_grad_components)
        f_rh = ex.submit(_read_gradient_emb, rh_p, n_grad_components)
        return np.concatenate([f_lh.result(), f_rh.result()], axis=0)


# ─────────────────────────────────────────────────────────────────────
# 5. the bundle
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Step3SparseInputs:
    """Everything the ``gpu_sparse`` backend needs, with no dense (N, L)."""
    layout: CandidateLayout
    packed_TND: np.ndarray                 # (T, N, ⌈D/8⌉) uint8, RAW on
    #                                        disk: MW rows are NOT zeroed
    #                                        (the session zeroes on device)
    D: int
    mw_lh_idx: np.ndarray
    mw_rh_idx: np.ndarray
    mu: np.ndarray                         # (D_prof, L) fp32
    epsil: np.ndarray                      # (L,) fp32
    sigma: np.ndarray                      # (L,) fp32
    theta_csr: Tuple[np.ndarray, np.ndarray, np.ndarray]
    gradient_mat: Optional[np.ndarray]
    lh_inflated: Dict[str, np.ndarray]
    rh_inflated: Dict[str, np.ndarray]
    lh_sphere: Dict[str, np.ndarray]
    rh_sphere: Dict[str, np.ndarray]
    sphere_xyz_bilateral: np.ndarray
    ini_val: float
    setting_params: Dict[str, Any]
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Step3SparseCohort:
    """The cohort-constant half of :class:`Step3SparseInputs`.

    The group prior, the two spatial masks and the derived
    :class:`CandidateLayout` depend only on ``cfg.mesh``,
    ``cfg.group_prior_path``, ``cfg.spatial_mask_path`` and
    ``cfg.num_clusters`` — identical for every subject of one cohort.
    Loaded once by :func:`load_step3_sparse_cohort` and shared
    read-only: every consumer copies out of ``prior`` / ``layout``
    (``cp.asarray`` / ``np.asarray``), none writes into them —
    ``prior['mu']`` is additionally handed back unchanged as each
    subject's ``Step3Result.Params['mu']``. The two spatial masks are
    not kept — the layout is the only thing built from them.
    """
    mesh: str
    group_prior_path: str
    spatial_mask_path: str
    num_clusters: int
    prior: Dict[str, Any]                  # mu/epsil/sigma/theta_csr/N/L
    layout: CandidateLayout
    timings: Dict[str, float]

    def check(self, cfg) -> None:
        """Raise ``ValueError`` naming the first field ``cfg`` disagrees on."""
        for name, mine, theirs in (
            ("mesh", self.mesh, str(cfg.mesh)),
            ("group_prior_path", self.group_prior_path,
             str(cfg.group_prior_path)),
            ("spatial_mask_path", self.spatial_mask_path,
             str(cfg.spatial_mask_path)),
            ("num_clusters", self.num_clusters, int(cfg.num_clusters)),
        ):
            if mine != theirs:
                raise ValueError(
                    f"Step3SparseCohort: {name} mismatch — cohort was built "
                    f"with {mine!r}, cfg has {theirs!r}")


def load_step3_sparse_cohort(cfg, *, overlap: bool = True,
                             ) -> Step3SparseCohort:
    """Load the prior + spatial masks + candidate layout for one cohort.

    ``overlap=False`` reads the prior and the masks serially (the
    timing report attributes per-piece cost that way).
    """
    from arealmshbm.data_io import load_avg_mesh

    timings: Dict[str, float] = {}
    lh_inflated = load_avg_mesh("lh", cfg.mesh, "inflated")
    rh_inflated = load_avg_mesh("rh", cfg.mesh, "inflated")
    N_expect = (int(lh_inflated["MARS_label"].shape[0])
                + int(rh_inflated["MARS_label"].shape[0]))

    def _prior():
        t = time.perf_counter()
        r = load_group_prior_csr(cfg.group_prior_path)
        timings["load_group_prior_csr"] = time.perf_counter() - t
        return r

    def _mask():
        t = time.perf_counter()
        r = load_spatial_mask_csr(cfg.spatial_mask_path)
        timings["load_spatial_mask_csr"] = time.perf_counter() - t
        return r

    if overlap:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_prior = ex.submit(_prior)
            f_mask = ex.submit(_mask)
            prior, masks = f_prior.result(), f_mask.result()
    else:
        prior = _prior()
        masks = _mask()

    lh_mask_csr, rh_mask_csr = masks
    N = int(prior["N"])
    L = int(prior["L"])
    if N != N_expect:
        raise ValueError(f"prior N={N} != mesh vertex count {N_expect}")
    if L != int(cfg.num_clusters):
        raise ValueError(f"prior L={L} != cfg.num_clusters={cfg.num_clusters}")

    t0 = time.perf_counter()
    layout = build_candidate_layout_fast(
        prior["theta_csr"], lh_mask_csr, rh_mask_csr,
        lh_inflated["vertexNbors"], rh_inflated["vertexNbors"])
    timings["build_candidate_layout"] = time.perf_counter() - t0

    return Step3SparseCohort(
        mesh=str(cfg.mesh),
        group_prior_path=str(cfg.group_prior_path),
        spatial_mask_path=str(cfg.spatial_mask_path),
        num_clusters=int(cfg.num_clusters),
        prior=prior, layout=layout, timings=timings,
    )


def load_step3_sparse_inputs(cfg,
                             precomputed_gradient_mat: Optional[np.ndarray] = None,
                             *,
                             overlap: bool = True,
                             cohort: Optional[Step3SparseCohort] = None,
                             ) -> Step3SparseInputs:
    """Sparse-path mirror of :meth:`Step3Pipeline.load_inputs`.

    Same config knobs, same numbers — no dense (N, L) array is ever
    allocated. Independent reads (cohort / BOLD / gradient) run on a
    small thread pool by default; ``overlap=False`` runs them serially
    (used by the timing report to attribute per-piece cost).

    ``cohort`` is a :class:`Step3SparseCohort` already loaded for this
    cohort — the stage pipeline loads one for the whole subject list
    and passes it to every subject. Its arrays are used as-is (shared
    read-only) and its timings are not merged into this subject's.
    When it is ``None`` the cohort is loaded here, on the same pool as
    the BOLD / gradient reads.
    """
    from arealmshbm.data_io import load_avg_mesh
    from arealmshbm.initialize_concentration import initialize_concentration

    build_cohort = cohort is None
    if not build_cohort:
        cohort.check(cfg)

    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    lh_inflated = load_avg_mesh("lh", cfg.mesh, "inflated")
    rh_inflated = load_avg_mesh("rh", cfg.mesh, "inflated")
    lh_sphere = load_avg_mesh("lh", cfg.mesh, "sphere")
    rh_sphere = load_avg_mesh("rh", cfg.mesh, "sphere")
    timings["load_avg_mesh"] = time.perf_counter() - t0

    sphere_xyz_bilateral = np.concatenate(
        [lh_sphere["vertices"].T, rh_sphere["vertices"].T], axis=0)

    n_lh = int(lh_inflated["MARS_label"].shape[0])
    N_expect = n_lh + int(rh_inflated["MARS_label"].shape[0])
    want_grad = bool(cfg.variant.use_connect_prior)

    def _cohort():
        return load_step3_sparse_cohort(cfg, overlap=overlap)

    def _bold():
        t = time.perf_counter()
        r = fetch_packed_bold_TND(cfg.project_dir, cfg.num_session, cfg.subid,
                                  cfg.mesh, lh_inflated, rh_inflated)
        timings["fetch_packed_bold"] = time.perf_counter() - t
        return r

    def _grad():
        if not want_grad:
            return None
        t = time.perf_counter()
        r = fetch_gradient(cfg.project_dir, cfg.subid, cfg.mesh, N_expect,
                           cfg.n_grad_components,
                           precomputed_gradient_mat=precomputed_gradient_mat)
        timings["fetch_gradient"] = time.perf_counter() - t
        return r

    if overlap:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_cohort = ex.submit(_cohort) if build_cohort else None
            f_bold = ex.submit(_bold)
            f_grad = ex.submit(_grad)
            if f_cohort is not None:
                cohort = f_cohort.result()
            packed_TND, D, mw_lh, mw_rh = f_bold.result()
            gradient_mat = f_grad.result()
    else:
        if build_cohort:
            cohort = _cohort()
        packed_TND, D, mw_lh, mw_rh = _bold()
        gradient_mat = _grad()

    if build_cohort:
        timings.update(cohort.timings)
    prior = cohort.prior
    L = int(prior["L"])
    T = int(packed_TND.shape[0])
    if T != int(cfg.num_session):
        raise ValueError(
            f"BOLD has {T} sessions; cfg.num_session={cfg.num_session}")

    dim = int(D) - 1
    t0 = time.perf_counter()
    ini_val = 650.0 if dim == 1482 else float(initialize_concentration(dim))
    timings["initialize_concentration"] = time.perf_counter() - t0

    setting_params: Dict[str, Any] = {
        "mesh": cfg.mesh,
        "num_session": T,
        "num_clusters": L,
        "subid": int(cfg.subid),
        "w": float(cfg.w),
        "c": float(cfg.c),
        "beta": np.full(L, cfg.beta_internal, dtype=np.float64),
        "epsilon": float(cfg.epsilon),
        "connect_th": float(cfg.connect_th),
        "dim": dim,
        "num_verts": int(prior["N"]),
    }

    timings["total"] = time.perf_counter() - t_total
    return Step3SparseInputs(
        layout=cohort.layout, packed_TND=packed_TND, D=int(D),
        mw_lh_idx=mw_lh, mw_rh_idx=mw_rh,
        mu=prior["mu"], epsil=prior["epsil"], sigma=prior["sigma"],
        theta_csr=prior["theta_csr"], gradient_mat=gradient_mat,
        lh_inflated=lh_inflated, rh_inflated=rh_inflated,
        lh_sphere=lh_sphere, rh_sphere=rh_sphere,
        sphere_xyz_bilateral=sphere_xyz_bilateral,
        ini_val=ini_val, setting_params=setting_params, timings=timings,
    )


__all__ = [
    "load_group_prior_csr", "theta_csr_to_dense", "load_spatial_mask_csr",
    "build_candidate_layout_fast", "fetch_packed_bold_TND", "fetch_gradient",
    "Step3SparseCohort", "load_step3_sparse_cohort",
    "Step3SparseInputs", "load_step3_sparse_inputs",
]
