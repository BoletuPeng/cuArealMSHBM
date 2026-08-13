"""diffusion_map.py

Diffusion-map embedding of a precomputed (dis)similarity matrix.

Direct port of ``mapalign.embed.compute_diffusion_map`` with the
parameter set used by CBIG step-0
(``apply_diffusion_on_individul_distance_matrix.py``):

    embed.compute_diffusion_map(distance_mat, alpha=0.5,
        n_components=100, return_result=True,
        skip_checks=True, overwrite=True)

That call shows the upstream ``mapalign`` signature; this port exposes
only ``dist`` / ``alpha`` / ``n_components`` and always returns just the
embedding array (the ``return_result`` / ``skip_checks`` / ``overwrite``
knobs are not surfaced).

Algorithm (mapalign, ``skip_checks=True, overwrite=True`` branch):

  1. Treat the input ``L`` as the affinity kernel directly. mapalign
     does **not** apply ``exp(-D/D.max())`` inside this function;
     CBIG simply hands the distance matrix in and lets it run.
  2. Alpha-normalize::
         d        = L.sum(axis=1)
         d_alpha  = d ** -alpha
         L_alpha  = d_alpha[:, None] * L * d_alpha[None, :]
  3. Row-normalize to a row-stochastic transition matrix::
         d2 = L_alpha.sum(axis=1)
         M  = d2[:, None]**-1 * L_alpha
  4. Eigendecomposition of ``M`` keeping the top ``n_components+1``
     eigenpairs sorted by descending eigenvalue. ``M`` is similar to
     the symmetric matrix
         S = D2^{-1/2} L_alpha D2^{-1/2}
     so we form ``S`` and call ``scipy.sparse.linalg.eigsh`` (Lanczos)
     on it for stability + speed, then map eigenvectors back to ``M``
     via ``psi = D2^{-1/2} phi``. This yields the same eigenpairs that
     mapalign's ``eigs(M)`` call would return, but in a numerically
     better-conditioned symmetric form. Eigenvectors are L2-normalized
     in the symmetric basis; that does not affect the final embedding
     because step 5 below renormalizes by the trivial eigenvector.
  5. Renormalize: ``psi[:, k] /= psi[:, 0]``. ``psi[:, 0]`` is the
     constant stationary eigenvector. (mapalign computes this
     unconditionally — including dividing the trivial column by
     itself, which yields all-ones.)
  6. Apply the ``diffusion_time=0`` rescaling of the eigenvalues::
         lambdas = lambdas[1:] / (1 - lambdas[1:])
  7. Embed::
         emb = psi[:, 1:n_components+1] * lambdas[:n_components][None, :]

Precision policy:
    Internal arithmetic is fp64. The input dist is upcast on entry.
    The returned ``emb`` is fp32 (MATLAB GT is fp32).

Connectedness precondition:
    ``dist`` must correspond to an irreducible Markov chain — i.e. a
    connected mesh component with strictly positive row sums after
    alpha-normalization. The renormalization step ``psi /= psi[:, 0]``
    and the eigenvalue rescaling ``lambdas / (1 - lambdas)`` both
    silently produce ``inf`` / ``nan`` on a disconnected input (zero
    in ``d2`` or a non-trivial unit eigenvalue). The CBIG step-0
    caller satisfies this — the gradient distance matrix comes from a
    single icosphere component — so we don't guard at runtime.

Memory note:
    For N=12962 the dense fp64 affinity is ~1.35 GB. With
    ``overwrite=True`` semantics we modify the upcast working copy in
    place. The original input is left untouched (we always upcast,
    and the upcast itself produces a new array regardless of input
    dtype).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh


def compute_diffusion_map(
    dist: np.ndarray,
    *,
    alpha: float = 0.5,
    n_components: int = 100,
) -> np.ndarray:
    """Diffusion-map embedding of a dense (dis)similarity matrix.

    Parameters
    ----------
    dist : (N, N) ndarray
        Symmetric, non-negative kernel matrix. In the CBIG step-0
        caller this is the gradient distance matrix in [0, 1]; the
        function treats whatever it gets as the kernel ``L``
        directly (mirrors mapalign).
    alpha : float, default 0.5
        Anisotropic diffusion exponent.
    n_components : int, default 100
        Number of non-trivial diffusion components to return.

    Returns
    -------
    emb : (N, n_components) fp32
        Diffusion embedding (eigenvectors rescaled by their
        eigenvalues, trivial eigenpair dropped).
    """
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(
            f"dist must be a square 2D matrix, got shape {dist.shape}"
        )
    ndim = dist.shape[0]
    if n_components is None:
        raise ValueError("n_components must be a positive int")
    if n_components < 1 or n_components >= ndim:
        raise ValueError(
            f"n_components={n_components} out of range for N={ndim}"
        )

    # 1. Upcast to fp64 working copy. We always copy so the caller's
    #    input is never mutated; this is the closest match to
    #    mapalign's `overwrite=True` branch given that we need fp64
    #    precision for the eigensolve regardless of input dtype.
    L_alpha = np.asarray(dist, dtype=np.float64)
    if L_alpha is dist:  # already fp64; force a private copy
        L_alpha = L_alpha.copy()

    # 2. Alpha-normalization (mirrors mapalign's dense branch).
    if alpha > 0:
        d = L_alpha.sum(axis=1)
        d_alpha = np.power(d, -alpha)
        # L_alpha = d_alpha[:, None] * L_alpha; L_alpha = L_alpha * d_alpha[None, :]
        L_alpha *= d_alpha[:, np.newaxis]
        L_alpha *= d_alpha[np.newaxis, :]

    # 3. Row-stochastic normalization. mapalign computes
    #    `d2 = L_alpha.sum(axis=1)` then forms M = D2^-1 L_alpha. We
    #    keep d2 around so we can build the symmetric similar matrix
    #    S = D2^-1/2 L_alpha D2^-1/2 instead.
    d2 = L_alpha.sum(axis=1)
    inv_sqrt_d2 = 1.0 / np.sqrt(d2)
    # Form S in place: S = D2^-1/2 L_alpha D2^-1/2.
    L_alpha *= inv_sqrt_d2[:, np.newaxis]
    L_alpha *= inv_sqrt_d2[np.newaxis, :]
    # Numerical hygiene: ensure exact symmetry. The two scalings are
    # the same operation transposed, so any drift is at the rounding
    # level — symmetrize anyway so eigsh sees a Hermitian operator.
    L_alpha = 0.5 * (L_alpha + L_alpha.T)
    S = L_alpha  # alias

    # 4. Top (n_components + 1) eigenpairs of S via Lanczos.
    #    eigsh on a symmetric matrix returns real eigenvalues; we ask
    #    for the largest-magnitude (which='LM'). The top eigenvalues
    #    of an alpha-normalized symmetric Markov operator are
    #    non-negative (largest = 1, stationary), so which='LM' and
    #    which='LA' yield the same set here. We use LM because that's
    #    what mapalign's `eigs(M)` defaults to.
    k = n_components + 1
    v0 = np.ones(ndim, dtype=np.float64)
    eigvals, phi = eigsh(S, k=k, which="LM", v0=v0)
    # eigsh returns eigenvalues in ascending order; flip to descending.
    eigvals = eigvals[::-1]
    phi = phi[:, ::-1]

    # Map back from the symmetric basis to the row-stochastic basis:
    #   M psi = lambda psi    with psi = D2^-1/2 phi
    vectors = phi * inv_sqrt_d2[:, np.newaxis]

    # 5. Renormalize eigenvectors by the trivial (constant) eigenvector.
    # Defensive: the stationary eigenvector should be strictly positive
    # on a connected mesh (graph_distance already raises on disconnection,
    # so this is double-defense for a near-irreducible regime where one
    # entry could underflow toward zero).
    psi0_min = float(np.min(np.abs(vectors[:, 0])))
    if psi0_min < 1e-30:
        raise RuntimeError(
            f"diffusion_map: stationary eigenvector has a near-zero entry "
            f"(min |psi[:, 0]| = {psi0_min:.3e}); renormalization would emit "
            f"inf/nan into the embedding. Check connectedness of the "
            f"underlying gradient graph."
        )
    psi = vectors / vectors[:, [0]]

    # 6. diffusion_time = 0 rescaling of eigenvalues (mirrors _step_5).
    lambdas_rescaled = eigvals[1:] / (1.0 - eigvals[1:])

    # 7. Embed: drop the trivial eigenpair and scale.
    embedding = psi[:, 1 : n_components + 1] * lambdas_rescaled[:n_components][np.newaxis, :]
    embedding_f32 = embedding.astype(np.float32, copy=False)

    return embedding_f32
