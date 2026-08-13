"""test_session_bold_ingest.py — pin the BOLD-ingestion seam at the
CPU :class:`VmfClusteringSession` boundary.

This is the Session-level integration test the unit-level pair
(:mod:`test_bitpacked_norm` + :mod:`test_session_common`) does not
catch: those test the two pieces in isolation, but neither verifies
that ``VmfClusteringSession.__init__`` wires
``validate_packed_bold_shape`` → ``unpack_normalize_packed_NTD_host``
correctly. If a future refactor changes ``D_unpacked`` threading or
mismatches the shape contract between the two, the per-piece tests
would still pass while the Session ctor would silently produce wrong
``ds_NTD``.

Also pins the **MW-zero contract at the Session boundary**: a packed
input with MW rows zeroed (per the upstream ``fetch_data`` invariant)
must produce zero fp32 rows on the Session's BOLD attribute. The
unit-level ``test_all_zero_packed_row_stays_zero`` pins the kernel;
this pins the wiring from caller-facing packed buffer to
Session-owned fp32 buffer.

Uses the dMSHBM variant (use_connect_prior / use_xyz_prior /
use_check_connectedness all False) so the fixture avoids the
gMSHBM/cMSHBM sphere + gradient inputs.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.step3_pipeline.variant import VariantSpec
from arealmshbm.vmf_clustering import VmfClusteringSession


# ─────────────────────────────────────────────────────────────────────
# fixture
# ─────────────────────────────────────────────────────────────────────
def _build_minimal_session(
    *,
    N: int = 8,
    T: int = 2,
    L: int = 2,
    D: int = 17,
    mw_indices: tuple = (0, 1, 2, 3),
    seed: int = 20260605,
) -> tuple[VmfClusteringSession, np.ndarray]:
    """Construct a minimal-valid dMSHBM Session, returning the Session
    and the packed BOLD that fed it. MW rows are zero-packed; non-MW
    rows carry random bits — exactly the upstream contract that
    ``fetch_data._read_b2nd_series_packed`` produces."""
    D_bytes = (int(D) + 7) // 8
    rng = np.random.default_rng(seed)
    packed = np.zeros((N, T, D_bytes), dtype=np.uint8)
    mw_set = set(mw_indices)
    for n in range(N):
        if n in mw_set:
            continue
        for t in range(T):
            bits = rng.integers(0, 2, size=D, dtype=np.uint8)
            packed[n, t] = np.packbits(bits, bitorder="little")

    # theta: uniform; row_active = True for all rows (matches MATLAB).
    theta = np.ones((N, L), dtype=np.float32)
    # boundary_mask: must be zero at cross-hemi cells (LH-vert × RH-parcel
    # and RH-vert × LH-parcel). Session ctor enforces this.
    n_lh, L_lh = N // 2, L // 2
    boundary_mask = np.ones((N, L), dtype=np.float32)
    boundary_mask[:n_lh, L_lh:] = 0.0
    boundary_mask[n_lh:, :L_lh] = 0.0
    # V_lambda: M1 ~ 6 (range-checked [1, 16]). Potts weights (V_same=0,
    # V_diff=1) are baked into the kernel — no edge-weight matrices.
    M1 = 6
    neighborhood = np.zeros((M1, N), dtype=np.int64)
    # Minimal candidate set (1 entry per parcel).
    row_idx = np.array([0, 1], dtype=np.int64)
    col_idx = np.array([0, 1], dtype=np.int64)
    beta = np.ones(L, dtype=np.float64)

    sess = VmfClusteringSession(
        data_series_NTD=packed,
        grad_data=None,
        sphere_xyz=None,
        lh_sphere_mesh=None,
        rh_sphere_mesh=None,
        theta=theta,
        boundary_mask=boundary_mask,
        neighborhood=neighborhood,
        row_idx=row_idx, col_idx=col_idx,
        dim=2, num_clusters=L, num_session=T,
        w=50.0, c=10.0, beta=beta,
        backend="cpu", D_unpacked=D,
        variant_spec=VariantSpec.from_pipeline_type("dMSHBM"),
    )
    return sess, packed


# ─────────────────────────────────────────────────────────────────────
# Wiring: validate + unpack reach the Session attr
# ─────────────────────────────────────────────────────────────────────
def test_session_ctor_produces_unpacked_fp32_bold() -> None:
    """The Session ctor wires ``validate_packed_bold_shape`` →
    ``unpack_normalize_packed_NTD_host`` such that the BOLD reaches
    ``self.m_step_session.data_series_NTD`` as a ``(N, T, D)`` fp32
    buffer."""
    sess, _ = _build_minimal_session(N=8, T=2, L=2, D=17)
    bold = sess.m_step_session.data_series_NTD
    assert bold.shape == (8, 2, 17), (
        f"Session ctor produced wrong BOLD shape: {bold.shape}; expected (8, 2, 17)"
    )
    assert bold.dtype == np.float32, (
        f"Session ctor produced wrong BOLD dtype: {bold.dtype}; expected float32"
    )


# ─────────────────────────────────────────────────────────────────────
# MW-zero contract at the Session boundary
# ─────────────────────────────────────────────────────────────────────
def test_session_preserves_mw_zero_contract() -> None:
    """MW-zero contract: rows the caller zeroed in the packed input
    (per the ``fetch_data._read_b2nd_series_packed`` invariant) MUST
    arrive at the Session's BOLD attribute as exact-zero fp32 rows.

    This is the "round-trip integration test" the reviewer asked for —
    pinning the contract at the consumer boundary, not the producer
    boundary (``fetch_data``). If a future refactor moves the Session
    to skip ``unpack_normalize_packed_NTD_host`` or feeds it the wrong
    ``D_unpacked``, the kernel's ``has_zero`` gate could fail to fire
    and MW rows would leak non-zero data. This test catches that
    drift faster than the next E2E.
    """
    mw_indices = (0, 1, 2, 3)
    sess, _ = _build_minimal_session(N=8, T=2, L=2, D=17, mw_indices=mw_indices)
    bold = sess.m_step_session.data_series_NTD
    for n in mw_indices:
        assert np.all(bold[n] == 0.0), (
            f"MW row {n} leaked non-zero data through the Session ctor: "
            f"max |bold[{n}]| = {np.max(np.abs(bold[n]))}"
        )
    # And the non-MW rows should NOT all be zero (otherwise the test
    # passes trivially regardless of wiring).
    non_mw = [n for n in range(8) if n not in mw_indices]
    assert any(not np.all(bold[n] == 0.0) for n in non_mw), (
        "fixture too degenerate: all non-MW rows ended up zero too"
    )


# ─────────────────────────────────────────────────────────────────────
# Invalid-input gates at the Session boundary
# ─────────────────────────────────────────────────────────────────────
def test_session_rejects_fp32_bold() -> None:
    """A caller that hands fp32 BOLD instead of bit-packed uint8 must
    fail at the validator, not later in the ctor. Pins that
    ``validate_packed_bold_shape`` runs *first* and that the error
    message points at the dtype mismatch."""
    N, T, D, L = 8, 2, 17, 2
    fp32_bold = np.zeros((N, T, D), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        VmfClusteringSession(
            data_series_NTD=fp32_bold,
            grad_data=None, sphere_xyz=None,
            lh_sphere_mesh=None, rh_sphere_mesh=None,
            theta=np.ones((N, L), dtype=np.float32),
            boundary_mask=np.ones((N, L), dtype=np.float32),
            neighborhood=np.zeros((6, N), dtype=np.int64),
            row_idx=np.array([0, 1], dtype=np.int64),
            col_idx=np.array([0, 1], dtype=np.int64),
            dim=2, num_clusters=L, num_session=T,
            w=50.0, c=10.0, beta=np.ones(L, dtype=np.float64),
            backend="cpu", D_unpacked=D,
            variant_spec=VariantSpec.from_pipeline_type("dMSHBM"),
        )


def test_session_rejects_T_mismatch() -> None:
    """Packed buffer with T axis disagreeing with num_session is
    rejected at the validator (not later)."""
    N, T_packed, T_arg, D, L = 8, 2, 3, 17, 2
    D_bytes = (D + 7) // 8
    packed = np.zeros((N, T_packed, D_bytes), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"T=\d+ != num_session=\d+"):
        VmfClusteringSession(
            data_series_NTD=packed,
            grad_data=None, sphere_xyz=None,
            lh_sphere_mesh=None, rh_sphere_mesh=None,
            theta=np.ones((N, L), dtype=np.float32),
            boundary_mask=np.ones((N, L), dtype=np.float32),
            neighborhood=np.zeros((6, N), dtype=np.int64),
            row_idx=np.array([0, 1], dtype=np.int64),
            col_idx=np.array([0, 1], dtype=np.int64),
            dim=2, num_clusters=L, num_session=T_arg,
            w=50.0, c=10.0, beta=np.ones(L, dtype=np.float64),
            backend="cpu", D_unpacked=D,
            variant_spec=VariantSpec.from_pipeline_type("dMSHBM"),
        )
