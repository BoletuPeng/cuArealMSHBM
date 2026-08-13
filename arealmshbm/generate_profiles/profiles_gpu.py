"""profiles_gpu.py

GPU supercall for per-session FC-profile generation. BOLD gzip
decompression stays CPU (no GPU gzip path); once decompressed, all
arithmetic happens on device end-to-end:

    H2D run pair → seed-mask gather → fused zscore (RawKernel) →
    fp32 cuBLAS sgemm → NaN→0 → running accumulator on device →
    scale + threshold-by-percentile + binarize on device →
    D2H binary masks.

Per (sub, sess) device residency: one (T, V_h) per hemi per run for
the duration of one run's accumulator update, then released; the two
(K, V_h) running sums live for the whole supercall. Threshold value
is computed via ``cp.partition`` on the concatenated (K, V_lh+V_rh)
correlation sum.

cupy import lives at module top — this file is only loaded via the
``backend == 'gpu'`` dispatch branch in :mod:`.profiles`, so the CPU
path never pays the cupy import.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from ._kernels_gpu import (
    binarize_mwzero_pack_cupy,
    zscore_unit_norm_columns_cupy,
)
from .profiles import _load_session_inputs


def _threshold_top_fraction_cupy(combined_corr_dev: cp.ndarray,
                                  fraction: float) -> float:
    """Value at rank ``round(N·fraction)`` (1-indexed) in descending
    order — the cutoff that admits the top ``fraction`` of entries.

    Uses ``cp.partition`` (GPU introselect). For our shape
    (K × 2V ≈ 80M), this avoids a 320 MB D2H of the combined array.
    Returns a host fp32 scalar — only the threshold value, no copy
    of the partitioned array, leaves the GPU.
    """
    flat = combined_corr_dev.reshape(-1)
    numel = int(flat.size)
    raw = numel * float(fraction)
    if raw >= 0.0:
        idx_1based = int(np.floor(raw + 0.5))
    else:
        idx_1based = -int(np.floor(-raw + 0.5))
    idx_1based = max(1, min(idx_1based, numel))
    kth = numel - idx_1based   # k-th smallest = (numel-k)-th largest in ascending
    # cp.partition returns a new partitioned array; pick the pivot value.
    partitioned = cp.partition(flat, kth)
    return float(partitioned[kth].get())


def compute_profile_arrays_gpu(seed_mesh: str,
                                targ_mesh: str,
                                out_dir,
                                sub: str,
                                sess: str,
                                split_flag: str = "0",
                                threshold="0.1",
                                precomputed_bold_runs=None,
                                ):
    """GPU port of :func:`profiles.compute_profile_arrays`.

    Parameters
    ----------
    precomputed_bold_runs : list of (lh_TxV, rh_TxV), optional
        When set, skip the internal gzip-decode ``ThreadPoolExecutor``
        and use the pre-decoded BOLD buffers. Driver-level
        :class:`Step1BoldPrefetcher` hoists gzip out of the serial
        (sub, sess) GPU loop.

    Returns
    -------
    (lh_packed_VxDb, rh_packed_VxDb, K_unpacked)
        ``lh_packed_VxDb`` / ``rh_packed_VxDb`` are host
        ``(V_h, ⌈K/8⌉) uint8`` arrays — the result of fusing
        binarize + MW-zero + transpose + packbits-along-K into a
        single CUDA RawKernel on device. ``K_unpacked`` is the
        original seed-axis size (K), needed by the
        ``write_subject_profile_tnd`` writer to record
        ``D_unpacked`` in the .b2nd vlmeta.

        Bit convention matches ``numpy.packbits(bitorder='little')``:
        cell k -> bit (k & 7) of byte (k >> 3). Padding bits past K
        in the last byte are zero; MW rows in the output are entirely
        zero bytes.

        Only the packed bytes (6 MB / hemi at fsa6 + Schaefer-300)
        are D2H'd — a 32x shrink vs the legacy fp32 D2H path.
    """
    inputs = _load_session_inputs(
        seed_mesh=seed_mesh, targ_mesh=targ_mesh, out_dir=out_dir,
        sub=sub, sess=sess, split_flag=split_flag, threshold=threshold,
        precomputed_bold_runs=precomputed_bold_runs,
    )
    lh_runs = inputs.lh_runs
    rh_runs = inputs.rh_runs
    n_runs = inputs.n_runs
    censor_runs = inputs.censor_runs
    lh_mars = inputs.lh_mars
    rh_mars = inputs.rh_mars
    threshold_f = inputs.threshold_f

    V_lh = int(lh_runs[0].shape[1])
    V_rh = int(rh_runs[0].shape[1])
    if lh_mars.shape[0] < inputs.n_seed_lh_verts or rh_mars.shape[0] < inputs.n_seed_rh_verts:
        raise ValueError("MARS_label arrays shorter than seed-vert count")
    # Seed-mask indices on host once, then H2D as int64 gather indices.
    lh_seed_mask = np.zeros(V_lh, dtype=bool)
    rh_seed_mask = np.zeros(V_rh, dtype=bool)
    lh_seed_mask[:inputs.n_seed_lh_verts] = (lh_mars[:inputs.n_seed_lh_verts] == 2)
    rh_seed_mask[:inputs.n_seed_rh_verts] = (rh_mars[:inputs.n_seed_rh_verts] == 2)
    lh_seed_idx_dev = cp.asarray(np.flatnonzero(lh_seed_mask).astype(np.int64))
    rh_seed_idx_dev = cp.asarray(np.flatnonzero(rh_seed_mask).astype(np.int64))
    K_lh = int(lh_seed_idx_dev.size)
    K_rh = int(rh_seed_idx_dev.size)
    K = K_lh + K_rh

    # Running accumulators stay on device for the whole loop.
    lh_corr_sum_dev = cp.zeros((K, V_lh), dtype=cp.float32)
    rh_corr_sum_dev = cp.zeros((K, V_rh), dtype=cp.float32)

    for r in range(n_runs):
        lh_TxV = lh_runs[r]
        rh_TxV = rh_runs[r]
        if lh_TxV.shape[0] != rh_TxV.shape[0]:
            raise ValueError(
                f"run {r}: lh/rh time-axis mismatch: "
                f"{lh_TxV.shape[0]} vs {rh_TxV.shape[0]}"
            )
        # Issue #55: ``precomputed_bold_runs`` may now hand the leaf
        # device-resident cupy arrays (Step1BoldPrefetcher's GPU
        # backend skips the H2D round trip per (sub, sess)). Detect
        # by type so the censor branch — which applies a bool mask —
        # uses the right array module for the input. The internal
        # gzip-decode path (no precompute) always lands on host
        # numpy.
        on_device = isinstance(lh_TxV, cp.ndarray)
        if censor_runs is not None and censor_runs[r] is not None:
            keep_host = (censor_runs[r] == 1)
            if keep_host.shape[0] != lh_TxV.shape[0]:
                raise ValueError(
                    f"run {r}: censor length {keep_host.shape[0]} != "
                    f"T={lh_TxV.shape[0]}"
                )
            if on_device:
                keep_dev = cp.asarray(keep_host)
                lh_TxV = cp.ascontiguousarray(lh_TxV[keep_dev], dtype=cp.float32)
                rh_TxV = cp.ascontiguousarray(rh_TxV[keep_dev], dtype=cp.float32)
            else:
                lh_TxV = np.ascontiguousarray(lh_TxV[keep_host], dtype=np.float32)
                rh_TxV = np.ascontiguousarray(rh_TxV[keep_host], dtype=np.float32)

        # H2D this run's BOLD; ``cp.asarray(cp.ndarray)`` is a zero-copy
        # view on already-device input, so the GPU prefetcher's device
        # buffer flows straight to the kernels without round-tripping.
        # Freed at the end of the loop iter via pool.
        lh_dev = cp.asarray(lh_TxV)
        rh_dev = cp.asarray(rh_TxV)

        # Seed gather: build (T, K) by indexing columns. lh_dev[:, lh_seed_idx]
        # is a (T, K_lh) fancy-index (copy); cheaper than masking + concat.
        T_ = lh_dev.shape[0]
        s_series_dev = cp.empty((T_, K), dtype=cp.float32)
        s_series_dev[:, :K_lh] = lh_dev[:, lh_seed_idx_dev]
        s_series_dev[:, K_lh:] = rh_dev[:, rh_seed_idx_dev]

        # Fused per-column zscore.
        s_norm_dev = cp.empty_like(s_series_dev)
        lh_norm_dev = cp.empty_like(lh_dev)
        rh_norm_dev = cp.empty_like(rh_dev)
        zscore_unit_norm_columns_cupy(s_series_dev, s_norm_dev)
        zscore_unit_norm_columns_cupy(lh_dev, lh_norm_dev)
        zscore_unit_norm_columns_cupy(rh_dev, rh_norm_dev)

        # cuBLAS sgemm: (K, T) @ (T, V_h) → (K, V_h).
        lh_corr = s_norm_dev.T @ lh_norm_dev
        rh_corr = s_norm_dev.T @ rh_norm_dev

        # NaN/Inf → 0 (zero-variance columns leak ±inf through (v - mean) * inv_norm).
        cp.nan_to_num(lh_corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cp.nan_to_num(rh_corr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        lh_corr_sum_dev += lh_corr
        rh_corr_sum_dev += rh_corr
        # Free the temporaries we don't need anymore so the pool reuses
        # their blocks for the next run iter.
        del lh_dev, rh_dev, s_series_dev, s_norm_dev, lh_norm_dev, rh_norm_dev
        del lh_corr, rh_corr

    inv_n = cp.float32(1.0 / n_runs)
    lh_corr_sum_dev *= inv_n
    rh_corr_sum_dev *= inv_n

    # Top-fraction threshold on the joint (lh + rh) sum.
    combined_dev = cp.concatenate([lh_corr_sum_dev, rh_corr_sum_dev], axis=1)
    t = _threshold_top_fraction_cupy(combined_dev, threshold_f)
    del combined_dev

    # Fused binarize + MW-zero + transpose-pack-along-K on device. This
    # replaces the legacy chain
    #     (corr >= t).astype(fp32) -> D2H (385 MB / sess) -> host
    #     _apply_mw_zero -> POST stage transpose -> WRITE packbits
    # with one RawKernel that writes (V_h, ⌈K/8⌉) uint8 packed bytes
    # directly. The K-axis ordering of the output matches what
    # numpy.packbits(axis=-1, bitorder='little') would have produced
    # downstream, so the .b2nd byte stream is bit-exact. MW rows
    # (MARS_label == 1 on the target mesh) come out as zero bytes,
    # enforcing the writer-side contract in packed form.
    V_lh_out = int(lh_corr_sum_dev.shape[1])
    V_rh_out = int(rh_corr_sum_dev.shape[1])
    if lh_mars.shape[0] != V_lh_out:
        raise ValueError(
            f"lh MARS_label.size ({lh_mars.shape[0]}) != V_lh ({V_lh_out})"
        )
    if rh_mars.shape[0] != V_rh_out:
        raise ValueError(
            f"rh MARS_label.size ({rh_mars.shape[0]}) != V_rh ({V_rh_out})"
        )
    D_bytes = (K + 7) // 8

    lh_mw_dev = cp.asarray(
        (np.asarray(lh_mars).ravel() == 1).astype(np.uint8))
    rh_mw_dev = cp.asarray(
        (np.asarray(rh_mars).ravel() == 1).astype(np.uint8))
    lh_packed_dev = cp.empty((V_lh_out, D_bytes), dtype=cp.uint8)
    rh_packed_dev = cp.empty((V_rh_out, D_bytes), dtype=cp.uint8)

    binarize_mwzero_pack_cupy(
        lh_corr_sum_dev, lh_mw_dev, float(t), lh_packed_dev)
    binarize_mwzero_pack_cupy(
        rh_corr_sum_dev, rh_mw_dev, float(t), rh_packed_dev)

    # D2H — 6 MB / hemi at fsa6 + Schaefer-300 (vs 192 MB for the legacy
    # fp32 binary). Both hemis together take ~3 ms on the legacy default
    # stream; the equivalent legacy D2H + host MW-zero pair took ~125 ms.
    lh_packed_VxDb = cp.asnumpy(lh_packed_dev)
    rh_packed_VxDb = cp.asnumpy(rh_packed_dev)

    return lh_packed_VxDb, rh_packed_VxDb, int(K)
