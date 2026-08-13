"""test_step2_em_outer.py

Synthetic tests for the three outer-EM closed-form leaves under
``arealmshbm.step2_em_outer``. All tests here are self-contained (no
external data dependency). A set of MATLAB-GT validation functions
(``test_L16/L17/L18_against_matlab_gt``) used to live alongside these;
they were retired when the pipeline decoupled from MATLAB.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.step2_em_outer import (
    intra_subject_var_loop,
    inter_subject_var,
    intra_em_cost_step2,
)

# Default vMF dim for this fork's Mode-A subjects.
_DIM = 1174


# ---------------------------------------------------------------------------
# Synthetic fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture
def small_shapes():
    """Small (D=5, L=4, S=3, T=2) random tensors with reasonable scales.
    Arrays are in INTERNAL layout (S, T, L, D) / (S, L, D) / (L, D) —
    matches what the production leaves expect.
    """
    rng = np.random.default_rng(seed=42)
    D, L, S, T = 5, 4, 3, 2

    # s_t_nu (S, T, L, D): per-(s, t, l) unit-norm column over D.
    nu = rng.standard_normal(size=(S, T, L, D)).astype(np.float32)
    nu /= np.sqrt((nu * nu).sum(axis=-1, keepdims=True))

    # s_psi (S, L, D): per-(s, l) unit-norm column over D.
    psi = rng.standard_normal(size=(S, L, D)).astype(np.float32)
    psi /= np.sqrt((psi * psi).sum(axis=-1, keepdims=True))

    # mu (L, D): per-l unit-norm.
    mu = rng.standard_normal(size=(L, D)).astype(np.float32)
    mu /= np.sqrt((mu * mu).sum(axis=-1, keepdims=True))

    sigma = np.full((1, L), 500.0, dtype=np.float32)
    epsil = np.full((1, L), 300.0, dtype=np.float32)
    cost_em = rng.standard_normal(size=(S,)).astype(np.float32)

    return dict(
        D=D, L=L, S=S, T=T,
        s_t_nu=nu, s_psi=psi, mu=mu,
        sigma=sigma, epsil=epsil,
        cost_em=cost_em,
    )


# ---------------------------------------------------------------------------
# L17 — intra_subject_var_loop synthetic.
# ---------------------------------------------------------------------------

class TestIntraSubjectVarLoop:

    def test_shapes_and_dtypes(self, small_shapes):
        sh = small_shapes
        s_psi_new, sigma_new, flag_psi = intra_subject_var_loop(
            sh["s_t_nu"], sh["s_psi"], sh["sigma"], sh["epsil"], sh["mu"],
            dim=_DIM, ini_val=1.0, epsilon=1e-4, max_iter=20,
        )
        assert s_psi_new.shape == (sh["S"], sh["L"], sh["D"])
        assert s_psi_new.dtype == np.float32
        assert sigma_new.shape == (1, sh["L"])
        assert sigma_new.dtype == np.float32
        assert flag_psi.shape == (sh["S"],)
        assert flag_psi.dtype in (np.int32, np.int64)

    def test_s_psi_columns_unit_norm(self, small_shapes):
        sh = small_shapes
        s_psi_new, _, _ = intra_subject_var_loop(
            sh["s_t_nu"], sh["s_psi"], sh["sigma"], sh["epsil"], sh["mu"],
            dim=_DIM, ini_val=1.0,
        )
        # Each (D,) row at (s, l) should be unit-norm (or all-zero if
        # the upstream sum was zero, which doesn't happen on these inputs).
        col_norms = np.sqrt((s_psi_new * s_psi_new).sum(axis=-1))   # (S, L)
        np.testing.assert_allclose(col_norms, 1.0, atol=1e-5)

    def test_convergence_quick_on_repeated_input(self, small_shapes):
        """When the inputs already satisfy the update fixed-point closely
        enough, the loop should terminate well before ``max_iter`` and
        every subject should be flagged."""
        sh = small_shapes
        # Run once to get a near-fixed-point.
        s_psi_1, sigma_1, _ = intra_subject_var_loop(
            sh["s_t_nu"], sh["s_psi"], sh["sigma"], sh["epsil"], sh["mu"],
            dim=_DIM, ini_val=1.0, epsilon=1e-2, max_iter=20,
        )
        # Feed it back in — should converge immediately (flag_psi all 1).
        _, _, flag_psi = intra_subject_var_loop(
            sh["s_t_nu"], s_psi_1, sigma_1, sh["epsil"], sh["mu"],
            dim=_DIM, ini_val=1.0, epsilon=1e-2, max_iter=20,
        )
        assert flag_psi.sum() == sh["S"], (
            f"flag_psi={flag_psi}, expected all 1 after warm-restart"
        )

    def test_ini_val_floor_respected(self, small_shapes):
        """sigma_new should never go below ini_val."""
        sh = small_shapes
        ini_val = 50.0
        _, sigma_new, _ = intra_subject_var_loop(
            sh["s_t_nu"], sh["s_psi"], sh["sigma"], sh["epsil"], sh["mu"],
            dim=_DIM, ini_val=ini_val,
        )
        assert (sigma_new >= ini_val - 1e-3).all(), (
            f"sigma_new={sigma_new.ravel()} dipped below ini_val={ini_val}"
        )

    def test_max_iter_one(self, small_shapes):
        """A single iteration must still produce valid outputs."""
        sh = small_shapes
        s_psi_new, sigma_new, flag_psi = intra_subject_var_loop(
            sh["s_t_nu"], sh["s_psi"], sh["sigma"], sh["epsil"], sh["mu"],
            dim=_DIM, ini_val=1.0, max_iter=1,
        )
        assert np.isfinite(s_psi_new).all()
        assert np.isfinite(sigma_new).all()


# ---------------------------------------------------------------------------
# L18 — inter_subject_var synthetic.
# ---------------------------------------------------------------------------

class TestInterSubjectVar:

    def test_shapes_and_dtypes(self, small_shapes):
        sh = small_shapes
        mu_new, epsil_new = inter_subject_var(
            sh["s_psi"], sh["mu"], sh["epsil"], dim=_DIM, ini_val=1.0,
        )
        assert mu_new.shape == (sh["L"], sh["D"])
        assert mu_new.dtype == np.float32
        assert epsil_new.shape == (1, sh["L"])
        assert epsil_new.dtype == np.float32

    def test_mu_columns_unit_norm(self, small_shapes):
        sh = small_shapes
        mu_new, _ = inter_subject_var(
            sh["s_psi"], sh["mu"], sh["epsil"], dim=_DIM, ini_val=1.0,
        )
        col_norms = np.sqrt((mu_new * mu_new).sum(axis=-1))   # (L,)
        np.testing.assert_allclose(col_norms, 1.0, atol=1e-5)

    def test_zero_column_fallback(self, small_shapes):
        """When the Σ_s s_psi column is exactly zero, mu_new falls back
        to prev_mu at that slot — not an L2-of-zero NaN."""
        sh = small_shapes
        S = sh["S"]
        assert S >= 2
        # Anti-symmetrize the first parcel across two subjects (internal
        # (S, L, D) layout) so its subject-sum is zero.
        psi = sh["s_psi"].copy()
        psi[1, 0, :] = -sh["s_psi"][0, 0, :]
        psi[2:, 0, :] = 0.0
        mu_new, _ = inter_subject_var(
            psi, sh["mu"], sh["epsil"], dim=_DIM, ini_val=1.0,
        )
        # The first row of mu_new (parcel 0) must equal prev_mu's first row.
        np.testing.assert_allclose(mu_new[0, :], sh["mu"][0, :], atol=1e-6)

    def test_ini_val_floor_respected(self, small_shapes):
        sh = small_shapes
        ini_val = 25.0
        _, epsil_new = inter_subject_var(
            sh["s_psi"], sh["mu"], sh["epsil"], dim=_DIM, ini_val=ini_val,
        )
        assert (epsil_new >= ini_val - 1e-3).all()


# ---------------------------------------------------------------------------
# L16 — intra_em_cost_step2 synthetic.
# ---------------------------------------------------------------------------

class TestIntraEmCostStep2:

    def test_returns_finite_scalar(self, small_shapes):
        sh = small_shapes
        params = {
            "s_psi": sh["s_psi"],
            "s_t_nu": sh["s_t_nu"],
            "mu": sh["mu"],
            "sigma": sh["sigma"],
            "epsil": sh["epsil"],
            "cost_em": sh["cost_em"],
        }
        cost = intra_em_cost_step2(params, dim=_DIM)
        assert isinstance(cost, float)
        assert np.isfinite(cost), f"cost={cost} is not finite"

    def test_cost_em_additive(self, small_shapes):
        """Adding a constant ``c`` to every entry of cost_em should shift
        the result by ``S * c`` exactly."""
        sh = small_shapes
        params = {
            "s_psi": sh["s_psi"],
            "s_t_nu": sh["s_t_nu"],
            "mu": sh["mu"],
            "sigma": sh["sigma"],
            "epsil": sh["epsil"],
            "cost_em": sh["cost_em"],
        }
        c0 = intra_em_cost_step2(params, dim=_DIM)
        params2 = dict(params)
        params2["cost_em"] = sh["cost_em"] + np.float32(2.5)
        c1 = intra_em_cost_step2(params2, dim=_DIM)
        np.testing.assert_allclose(c1 - c0, 2.5 * sh["S"], atol=1e-3)

    def test_matches_literal_reference_form(self, small_shapes):
        """Reference: compute the cost via the literal 4D algebraic chain
        (the same formula the MATLAB code implements) in numpy and compare
        to the vectorized form in L16. A pure-numerics consistency check —
        no external data. Inputs are in internal layout (S, T, L, D) /
        (S, L, D) / (L, D); the reference computation uses the same.
        """
        from arealmshbm.em_stop_criterion._cdln import cdln_general_to_f32

        sh = small_shapes
        D, L, S, T = sh["D"], sh["L"], sh["S"], sh["T"]
        psi = sh["s_psi"].astype(np.float64)        # (S, L, D)
        nu = sh["s_t_nu"].astype(np.float64)        # (S, T, L, D)
        mu = sh["mu"].astype(np.float64)            # (L, D)
        sigma = sh["sigma"].astype(np.float64).reshape(1, L)
        epsil = sh["epsil"].astype(np.float64).reshape(1, L)

        # term1: Σ_{s,t,l,d} σ[l] · ψ[s,l,d] · ν[s,t,l,d]  +  S·T·Σ_l Cdln(σ_l)
        # Broadcast ψ across t: psi (S, L, D) → (S, 1, L, D); ν (S, T, L, D).
        A = psi[:, None, :, :] * nu                                 # (S, T, L, D)
        A = (sigma.reshape(1, 1, L, 1) * A).sum(axis=-1)            # (S, T, L)
        cdln_sigma = np.empty(L, dtype=np.float32)
        cdln_general_to_f32(sigma.ravel(), _DIM, cdln_sigma)
        A = A + cdln_sigma.astype(np.float64).reshape(1, 1, L)      # (S, T, L)
        term1 = A.sum()

        # term2: Σ_{s,l,d} ε[l] · μ[l,d] · ψ[s,l,d]  +  S·Σ_l Cdln(ε_l)
        B = mu[None, :, :] * psi                                    # (S, L, D)
        B = (epsil.reshape(1, L, 1) * B).sum(axis=-1)               # (S, L)
        cdln_epsil = np.empty(L, dtype=np.float32)
        cdln_general_to_f32(epsil.ravel(), _DIM, cdln_epsil)
        B = B + cdln_epsil.astype(np.float64).reshape(1, L)         # (S, L)
        term2 = B.sum()

        ref_cost = float(term1 + term2 + float(sh["cost_em"].astype(np.float64).sum()))

        params = {
            "s_psi": sh["s_psi"],
            "s_t_nu": sh["s_t_nu"],
            "mu": sh["mu"],
            "sigma": sh["sigma"],
            "epsil": sh["epsil"],
            "cost_em": sh["cost_em"],
        }
        cost = intra_em_cost_step2(params, dim=_DIM)
        # Relative tolerance — Cdln output is fp32, magnitudes are large.
        np.testing.assert_allclose(cost, ref_cost, rtol=1e-5, atol=1e-3)
