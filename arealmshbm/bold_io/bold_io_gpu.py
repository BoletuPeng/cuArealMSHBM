"""bold_io_gpu.py — device-side counterpart of :mod:`bold_io`.

Mirrors :func:`bold_io.concat_hemis_drop_medial` on the GPU so the
step-0 BOLD prefetcher's ``backend='gpu'`` path can keep BOLD
device-resident from the GIFTI decode all the way through to the
step-0 ``_subgraph_A`` consumer. The CPU path stays untouched.

Numerical contract is bit-equal to the CPU path. The MATLAB-parity
ordering (per-hemi NaN→0 BEFORE vstack, then drop medial rows) is
preserved — same semantics on numpy/host and cupy/device.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as _cupy_type_check  # noqa: F401


def concat_hemis_drop_medial_gpu(lh, rh, medial_mask_dev):
    """Device-side ``concat + nan_to_num + drop medial``.

    Mirrors the CPU sequence in
    :func:`arealmshbm.bold_io.concat_hemis_drop_medial` exactly:

        1. Per-hemi ``cp.nan_to_num`` in place.
        2. ``cp.vstack([lh, rh])`` → ``(N_full, T)``.
        3. Drop rows where ``medial_mask_dev`` is truthy.

    Parameters
    ----------
    lh, rh : cupy.ndarray (N_hemi, T) fp32
        Per-hemi BOLD on device — typically the output of
        :func:`arealmshbm.data_io.gifti_io.read_surface_gifti_gpu` with
        ``to_host=False``. Non-fp32 inputs are coerced via
        ``astype(cp.float32, copy=False)`` to mirror the CPU path's
        ``np.asarray(..., dtype=np.float32)`` semantics — no silent
        precision divergence vs CPU. Both hemis must live on the same
        cupy device.

        **Mutated in place when input is already fp32.** ``astype
        (copy=False)`` returns a view of the caller's buffer when the
        dtype already matches, and the subsequent ``cp.nan_to_num
        (copy=False)`` writes through that view. Both worker call
        sites pass fresh per-session buffers so this is safe today;
        callers reusing buffers across calls must clone first. (The
        CPU path :func:`bold_io.concat_hemis_drop_medial` has the
        same gotcha.)
    medial_mask_dev : cupy.ndarray (2 * N_hemi,) truthy=medial
        Medial mask, **must already live on device** — the prefetcher
        uploads it once at ``__init__`` via ``cp.asarray(host_mask)``
        and reuses across every session. Accepts bool / uint8 / int
        (anything truthy). Must live on the same cupy device as
        ``lh`` / ``rh``. The ``(N_full, 1)``→``(N_full,)`` ravel is
        done here so the CPU prefetcher's reshape-safe semantics
        carry over.

    Returns
    -------
    cupy.ndarray (N_cortex, T) fp32
        Concatenated, NaN-cleaned, medial-dropped BOLD on the same
        device as ``lh``/``rh``. Ready for direct hand-off to step-0
        ``_subgraph_A``'s GPU compute (``cp.asarray(curr_data)`` on a
        cupy.ndarray is a zero-copy view).

    Notes
    -----
    The boolean-index drop ``full[keep, :]`` triggers a single
    device-side gather kernel — no host round trip. ``keep = ~mask``
    is also evaluated on device.

    Raises explicit errors for: host-numpy hemis, device-mismatched
    inputs (lh/rh/mask on different cupy devices), mask length
    mismatch. All caught early to keep failure modes named at the
    call site rather than deep in cupy's gather kernel.
    """
    import cupy as cp

    if not isinstance(lh, cp.ndarray) or not isinstance(rh, cp.ndarray):
        raise TypeError(
            f"concat_hemis_drop_medial_gpu: lh and rh must be cupy.ndarray; "
            f"got {type(lh).__name__}, {type(rh).__name__}"
        )
    # Mask strict check — symmetric with lh/rh. The pre-fix permissive
    # ``cp.asarray(host_mask)`` path would silently H2D the mask on
    # every call; the prefetcher always uploads the mask at init, so
    # there's no legitimate caller passing host here.
    if not isinstance(medial_mask_dev, cp.ndarray):
        raise TypeError(
            f"concat_hemis_drop_medial_gpu: medial_mask_dev must be "
            f"cupy.ndarray (upload via ``cp.asarray(host_mask)`` once at "
            f"prefetcher init); got {type(medial_mask_dev).__name__}"
        )
    if lh.ndim != 2 or rh.ndim != 2:
        raise ValueError(
            f"lh and rh must be 2-D (N, T); got lh.shape={lh.shape}, "
            f"rh.shape={rh.shape}"
        )
    if lh.shape[1] != rh.shape[1]:
        raise ValueError(
            f"lh and rh must have matching T; got {lh.shape[1]} vs "
            f"{rh.shape[1]}"
        )
    if int(lh.device.id) != int(rh.device.id):
        raise ValueError(
            f"concat_hemis_drop_medial_gpu: lh on cupy device "
            f"{int(lh.device.id)} but rh on device {int(rh.device.id)}; "
            f"both hemis must live on the same device"
        )
    if int(medial_mask_dev.device.id) != int(lh.device.id):
        raise ValueError(
            f"concat_hemis_drop_medial_gpu: medial_mask_dev on cupy "
            f"device {int(medial_mask_dev.device.id)} but lh/rh on "
            f"device {int(lh.device.id)}; all three must live on the "
            f"same device (capture device id at prefetcher init and "
            f"``with cp.cuda.Device(captured):`` in workers)"
        )

    # Mirror CPU's ``np.asarray(lh, dtype=np.float32)`` exactly. On
    # already-fp32 input ``astype(copy=False)`` is a no-op view; on
    # other dtypes it allocates a converted buffer — same semantics
    # as numpy. Without this coercion a future non-fp32 caller would
    # silently diverge from the CPU path.
    lh = lh.astype(cp.float32, copy=False)
    rh = rh.astype(cp.float32, copy=False)

    # cp.nan_to_num supports in-place via copy=False; same call signature
    # as the CPU path so the MATLAB-parity ordering is preserved
    # (per-hemi NaN→0 *before* the vstack).
    cp.nan_to_num(lh, copy=False, nan=0.0)
    cp.nan_to_num(rh, copy=False, nan=0.0)
    full = cp.vstack([lh, rh])

    # Mask reshape: accept (N_full,) or (N_full, 1) cupy mask. ``astype``
    # on already-bool is a view; on uint8/int it converts on device.
    mask = medial_mask_dev.reshape(-1).astype(cp.bool_)
    if mask.shape[0] != full.shape[0]:
        raise ValueError(
            f"medial_mask length {mask.shape[0]} != concatenated rows "
            f"{full.shape[0]}"
        )
    keep = ~mask
    return full[keep, :]
