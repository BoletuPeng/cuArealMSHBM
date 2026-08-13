"""test_alias_invariant.py — pin the run_iter alias contract.

``Step2EmIterSession.run_iter`` requires that ``Params['s_t_nu' /
's_lambda' / 'theta']`` ARE the Session-owned scratch buffers (the
alias established by :meth:`upload_initial_state`). The kernel writes
in place into those scratch slots; if a caller reassigns any Params
slot to a fresh array between calls, the kernel would silently read
stale bytes from the unrelated buffer. ``run_iter`` catches this with
an ``is`` identity check at the top.

These tests reassign each slot post-upload and assert the loud
ValueError fires. They lock the contract so a future refactor that
makes the alias optional has to consciously update the tests.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import numpy as np
import pytest

from arealmshbm.step2_em_iter_master import (
    Step2EmIterSession, warmup_em_iter_master,
)
from arealmshbm.step2_io import (
    InMemoryGradientLoader,
    InMemoryProfileLoader,
)


@pytest.fixture(scope="module", autouse=True)
def _warmup():
    warmup_em_iter_master()


def _build_session():
    """Construct a minimal dMSHBM Session + initial Params dict.

    Same shape conventions as the GPU parity test; small enough to keep
    the construction wall sub-second."""
    S, T, N, D, L, D_grad = 2, 2, 32, 16, 4, 8
    n_lh = N // 2
    L_lh = L // 2
    rng = np.random.default_rng(0)

    dp = rng.standard_normal((S, N, T, D)).astype(np.float32)
    bm = np.zeros((N, L), dtype=np.float32)
    bm[:n_lh, :L_lh] = 1.0
    bm[n_lh:, L_lh:] = 1.0

    s_lambda = rng.uniform(0.1, 1.0, size=(S, N, L)).astype(np.float32)
    s_lambda *= bm[None, :, :]
    rs = s_lambda.sum(axis=-1, keepdims=True)
    s_lambda /= np.where(rs > 0, rs, 1.0)

    s_t_nu = rng.standard_normal((S, T, L, D)).astype(np.float32)
    nu_norms = np.linalg.norm(s_t_nu, axis=-1, keepdims=True)
    s_t_nu /= np.where(nu_norms > 0, nu_norms, 1.0)

    s_psi = rng.standard_normal((S, L, D)).astype(np.float32)
    psi_norms = np.linalg.norm(s_psi, axis=-1, keepdims=True)
    s_psi /= np.where(psi_norms > 0, psi_norms, 1.0)

    Params = {
        "s_lambda": s_lambda,
        "s_t_nu": s_t_nu,
        "theta": s_lambda.mean(axis=0).astype(np.float32),
        "kappa": np.full((1, L), 100.0, dtype=np.float32),
        "sigma": np.full((1, L), 0.5, dtype=np.float32),
        "s_psi": s_psi,
    }

    sess = Step2EmIterSession(
        bold_loader=InMemoryProfileLoader(dp, num_session=T),
        grad_loader=None,
        num_sub=S, N=N, T=T, D=D, D_grad=0,
        boundary_mask=bm, s_psi=s_psi, sigma=Params["sigma"],
        mode="dMSHBM", dim=D, num_clusters=L,
        ini_val=30.0, beta_internal=0.0,
        n_lh=n_lh, eps_m_step=1e-4, max_iter_m=5,
    )
    return sess, Params


@pytest.mark.parametrize("key", ["s_t_nu", "s_lambda", "theta"])
def test_run_iter_rejects_reassigned_slot(key):
    """After upload_initial_state aliases Params[key] to Session-owned
    scratch, reassigning Params[key] to a fresh copy breaks the alias
    invariant and run_iter must raise.

    Locks the contract for s_t_nu / s_lambda / theta. The kernel writes
    in place into the Session scratch; a stale Params alias would let
    the next outer-EM iter feed an unrelated buffer to the master
    kernel, silently producing wrong outputs."""
    sess, Params = _build_session()
    sess.upload_initial_state(Params)
    # Sanity: post-upload the alias holds.
    sess.run_iter(Params)
    # Break the alias on this one slot only — fresh copy with the same
    # shape/dtype/contig so only the identity differs.
    Params[key] = Params[key].copy()
    with pytest.raises(ValueError, match=f"Params\\[{key!r}\\]"):
        sess.run_iter(Params)


def test_run_iter_passes_when_alias_held():
    """Two back-to-back run_iter calls succeed when the caller never
    reassigns the aliased slots. Mirrors the production loop in
    Step2Pipeline.run_em."""
    sess, Params = _build_session()
    sess.upload_initial_state(Params)
    sess.run_iter(Params)
    sess.run_iter(Params)  # second call still satisfies the alias
