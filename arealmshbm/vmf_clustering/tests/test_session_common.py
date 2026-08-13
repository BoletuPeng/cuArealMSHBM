"""test_session_common.py — pin the two shared validators that the CPU
:class:`VmfClusteringSession` and GPU :class:`VmfClusteringSessionCUDA`
constructors both call.

Both validators live in :mod:`arealmshbm.vmf_clustering._session_common`
and form the cross-backend "contract surface": if these accept an
input on one backend, they must accept it on the other. A regression
on either backend that drifts the contract gets caught here, not at
the next E2E.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.step3_pipeline.variant import VariantSpec
from arealmshbm.vmf_clustering._session_common import (
    validate_packed_bold_shape,
    validate_variant_requirements,
)


# ─────────────────────────────────────────────────────────────────────
# validate_packed_bold_shape
# ─────────────────────────────────────────────────────────────────────
class TestValidatePackedBoldShape:
    def test_accepts_valid_packed_buffer(self) -> None:
        # D=17 → ceil(17/8) = 3 bytes/row
        arr = np.zeros((5, 6, 3), dtype=np.uint8)
        N, T, D_bytes = validate_packed_bold_shape(
            arr, num_session=6, D_unpacked=17,
        )
        assert (N, T, D_bytes) == (5, 6, 3)

    def test_rejects_fp32_dtype(self) -> None:
        arr = np.zeros((5, 6, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            validate_packed_bold_shape(arr, num_session=6, D_unpacked=17)

    def test_rejects_2d(self) -> None:
        arr = np.zeros((5, 18), dtype=np.uint8)
        with pytest.raises(ValueError, match="3D"):
            validate_packed_bold_shape(arr, num_session=1, D_unpacked=144)

    def test_rejects_T_mismatch(self) -> None:
        arr = np.zeros((5, 6, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"T=6 != num_session=7"):
            validate_packed_bold_shape(arr, num_session=7, D_unpacked=17)

    def test_rejects_d_bytes_mismatch(self) -> None:
        # D=17 needs 3 bytes; pass 4.
        arr = np.zeros((5, 6, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"D_bytes"):
            validate_packed_bold_shape(arr, num_session=6, D_unpacked=17)


# ─────────────────────────────────────────────────────────────────────
# validate_variant_requirements
# ─────────────────────────────────────────────────────────────────────
class TestValidateVariantRequirements:
    @staticmethod
    def _mesh_stub() -> dict:
        return {"vertices": np.zeros((3, 4), dtype=np.float32),
                "vertexNbors": np.zeros((6, 4), dtype=np.int32)}

    def test_gMSHBM_needs_grad_data(self) -> None:
        # gMSHBM uses use_connect_prior + use_check_connectedness; both
        # grad_data and spheres are required.
        v = VariantSpec.from_pipeline_type("gMSHBM")
        with pytest.raises(ValueError, match="grad_data"):
            validate_variant_requirements(
                v, None,
                np.zeros((3, 4), dtype=np.float32),
                self._mesh_stub(), self._mesh_stub(),
            )

    def test_gMSHBM_needs_sphere_xyz_and_meshes(self) -> None:
        v = VariantSpec.from_pipeline_type("gMSHBM")
        grad = np.zeros((4, 100), dtype=np.float32)
        # sphere_xyz missing
        with pytest.raises(ValueError, match="sphere_xyz"):
            validate_variant_requirements(
                v, grad, None, self._mesh_stub(), self._mesh_stub(),
            )
        # LH sphere mesh missing
        with pytest.raises(ValueError, match="sphere_xyz"):
            validate_variant_requirements(
                v, grad, np.zeros((3, 4)), None, self._mesh_stub(),
            )
        # RH sphere mesh missing
        with pytest.raises(ValueError, match="sphere_xyz"):
            validate_variant_requirements(
                v, grad, np.zeros((3, 4)), self._mesh_stub(), None,
            )

    def test_gMSHBM_accepts_complete_inputs(self) -> None:
        v = VariantSpec.from_pipeline_type("gMSHBM")
        # No raise → return None.
        validate_variant_requirements(
            v,
            np.zeros((4, 100), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
            self._mesh_stub(), self._mesh_stub(),
        )

    def test_cMSHBM_only_needs_spheres_not_grad(self) -> None:
        """cMSHBM has use_xyz_prior + use_check_connectedness but
        use_connect_prior=False — grad_data not required, but spheres
        still are (for the xyz prior + connectedness BFS)."""
        v = VariantSpec.from_pipeline_type("cMSHBM")
        # grad_data None is fine for cMSHBM.
        validate_variant_requirements(
            v, None,
            np.zeros((3, 4), dtype=np.float32),
            self._mesh_stub(), self._mesh_stub(),
        )
        # ...but a missing sphere mesh still raises.
        with pytest.raises(ValueError, match="sphere_xyz"):
            validate_variant_requirements(
                v, None, np.zeros((3, 4)), None, self._mesh_stub(),
            )

    def test_dMSHBM_accepts_all_nones(self) -> None:
        """dMSHBM has use_connect_prior=False AND
        use_check_connectedness=False, so neither grad_data nor any
        sphere input is required."""
        v = VariantSpec.from_pipeline_type("dMSHBM")
        # All None is legal for dMSHBM — no raise.
        validate_variant_requirements(v, None, None, None, None)
