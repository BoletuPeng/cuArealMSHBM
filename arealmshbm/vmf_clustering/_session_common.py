"""_session_common.py — ctor validators and Params staging helpers for
the step-3 sessions.

  * ``validate_variant_requirements`` — the variant's required input
    slots. Shared by the cpu :class:`VmfClusteringSession` and the gpu
    :class:`VmfClusteringSessionSparseCUDA`, so both backends accept the
    same variant inputs.
  * ``validate_packed_bold_shape`` — the packed ``(N, T, ⌈D/8⌉)`` BOLD
    contract of the cpu session.
  * ``_stage_1d`` / ``_stage_2d`` / ``_stage_3d`` — shape-check +
    dtype-cast the EM Params dict fields before the cpu session's
    ``run()`` consumes them.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from arealmshbm.step3_pipeline.variant import VariantSpec


def validate_variant_requirements(
    variant_spec: VariantSpec,
    grad_data: Any,                                       # (N, D_grad) fp32 or None
    sphere_xyz: Any,                                      # (3, N) or (N, 3) or None
    lh_sphere_mesh: Optional[Dict[str, np.ndarray]],
    rh_sphere_mesh: Optional[Dict[str, np.ndarray]],
) -> None:
    """Raise if the variant's required input slots are missing.

    * ``use_connect_prior`` → ``grad_data`` must not be None.
    * ``use_check_connectedness`` → all three of (sphere_xyz, LH sphere
      mesh, RH sphere mesh) must not be None; the variant needs them
      for both the connectedness BFS and the spatial xyz prior.
    """
    if variant_spec.use_connect_prior and grad_data is None:
        raise ValueError(
            f"variant {variant_spec.name!r} needs grad_data; got None"
        )
    if variant_spec.use_check_connectedness and (
        lh_sphere_mesh is None or rh_sphere_mesh is None or sphere_xyz is None
    ):
        raise ValueError(
            f"variant {variant_spec.name!r} needs sphere_xyz + LH/RH "
            f"sphere meshes (for check_connectedness + xyz_prior); "
            f"got sphere_xyz={sphere_xyz is not None}, "
            f"lh_sphere={lh_sphere_mesh is not None}, "
            f"rh_sphere={rh_sphere_mesh is not None}"
        )


def validate_packed_bold_shape(
    arr: Any,                                  # (N, T, D_bytes) uint8 packed
    num_session: int,
    D_unpacked: int,
) -> Tuple[int, int, int]:
    """Validate the packed BOLD input contract; return ``(N, T, D_bytes)``.

    The caller then feeds ``arr`` to
    :func:`arealmshbm.data_io.bitpacked_norm.unpack_normalize_packed_NTD_host`.

    Accepted contract:
      * dtype == uint8
      * ndim == 3
      * shape == ``(N, T, ⌈D_unpacked/8⌉)``
      * T == num_session

    A pure shape/dtype validator; the unpack + normalize runs
    downstream.
    """
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        raise ValueError(
            f"data_series_NTD must be uint8 (bit-packed); got {a.dtype}"
        )
    if a.ndim != 3:
        raise ValueError(
            f"data_series_NTD must be 3D (N, T, ⌈D/8⌉); got {a.shape}"
        )
    N, T, D_bytes = a.shape
    if T != int(num_session):
        raise ValueError(
            f"data_series_NTD T={T} != num_session={num_session}"
        )
    expected_bytes = (int(D_unpacked) + 7) // 8
    if D_bytes != expected_bytes:
        raise ValueError(
            f"data_series_NTD D_bytes ({D_bytes}) != ceil(D/8) "
            f"({expected_bytes}) for D_unpacked={D_unpacked}"
        )
    return int(N), int(T), int(D_bytes)


# ─────────────────────────────────────────────────────────────────────
# Run-time Params staging helpers.
#
# The cpu Session's ``run()`` accepts a Params dict at the per-call
# boundary and stages each field through one of these three shape-
# checked dtype-cast helpers before the EM math touches it.
# ─────────────────────────────────────────────────────────────────────
def _stage_2d(arr: Any, N: int, L: int, name: str) -> np.ndarray:
    """Stage a (N, L) Params field as fp32 C-contig. Strict shape check —
    single-subject only, so (N, L, 1) and other extra-axis shapes are
    rejected at the boundary."""
    a = np.asarray(arr)
    if a.shape != (N, L):
        raise ValueError(f"{name}: expected (N={N}, L={L}); got {arr.shape}")
    return np.ascontiguousarray(a, dtype=np.float32)


def _stage_3d(arr: Any, D: int, L: int, T: int, name: str) -> np.ndarray:
    """Stage a (D, L, T) Params field as fp32 C-contig. Strict shape check."""
    a = np.asarray(arr)
    if a.shape != (D, L, T):
        raise ValueError(
            f"{name}: expected (D={D}, L={L}, T={T}); got {arr.shape}"
        )
    return np.ascontiguousarray(a, dtype=np.float32)


def _stage_1d(arr: Any, L: int, name: str, dtype) -> np.ndarray:
    """Stage a (L,) Params field, accepting (L,) (preferred) or (1, L)
    (MATLAB GT shape readers produce this) and flattening either way.
    Internal storage is always (L,) flat.

    Pre-ravel shape check: anything with L total elements would pass
    a post-ravel ``shape == (L,)`` test (e.g. (2, L//2), (L, 1),
    arbitrary 4-D reshapes). Validate the original shape first so those
    cases fail loudly at the boundary rather than feeding silently into
    downstream kernels.
    """
    a = np.asarray(arr)
    if a.shape != (L,) and a.shape != (1, L):
        raise ValueError(
            f"{name}: expected (L={L},) or (1, L); got {a.shape}"
        )
    return np.ascontiguousarray(a.ravel(), dtype=dtype)
