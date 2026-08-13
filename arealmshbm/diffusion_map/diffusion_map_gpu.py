"""diffusion_map_gpu.py

CuPy mirror of :mod:`arealmshbm.diffusion_map.diffusion_map`.
Same algorithm class — alpha-normalized random-walk Lanczos for top-k
diffusion components — executed on the GPU via ``cupy.linalg`` and
``cupyx.scipy.sparse.linalg.eigsh``.

Precision: fp32 by default. The on-disk distance matrix is fp32, the
affinity matrix ``exp(-D/D.max())`` lives in (0, 1] (bounded dynamic
range), and the alpha-normalised symmetric similar matrix is also
bounded. Empirically on the fsa6 down-sphere (N=12962, k=101), every
one of the top 100 diffusion components matches the fp64 reference to
|cos| = 1.0000. ``working_dtype=cp.float64`` is still selectable for
archaeology / audit; fp64 is ~5× slower because the algorithm is
fp64-bandwidth bound on the dense (N, N) GEMV inside Lanczos.

CPU vs. GPU equivalence (orthogonal to the precision axis above)
----------------------------------------------------------------
The CPU/GPU paths use different partial-Lanczos implementations
(scipy/ARPACK vs cupyx). They are **algorithm-class equivalent**, not
bit-identical:

  * eigenvalues agree to ~6 dp,
  * eigenvectors agree up to a per-component sign flip,
  * degenerate eigenspaces may be returned in a different basis
    orientation.

Downstream the CBIG step-0 caller is robust to all three (psi/psi[:,0]
absorbs the sign; the parcellation downstream depends on the invariant
subspace, not on its basis orientation), but anything that consumes
the raw embedding should not assume byte equality between backends.
See ``validate_cpu_gpu.py`` for the smoke that asserts the actual
contract (|cos| per eigvec near 1; final parcellation labels agree).

Public API:
    compute_diffusion_map_gpu(affinity, alpha, n_components,
                              working_dtype=cp.float32) -> emb_f32

Wall on RTX 5090 Laptop (N=12962, k=101): ~0.3 s/hemi fp32 warm,
~1.7 s/hemi fp64 warm (host affinity dominates fp64; device-side
affinity is in compute_diffusion_map_gpu_repaired).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations


def compute_diffusion_map_gpu(
    affinity,
    *,
    alpha: float = 0.5,
    n_components: int = 100,
    working_dtype=None,
):
    """GPU partial-Lanczos diffusion-map embedding of a precomputed
    AFFINITY matrix (not distance — see
    :func:`compute_diffusion_map_gpu_repaired` for the distance →
    affinity wrapper).

    Parameters
    ----------
    affinity : (N, N) ndarray or cupy.ndarray
        Symmetric, non-negative affinity matrix. Host arrays are
        copied to device; device arrays are aliased if the dtype
        already matches ``working_dtype`` (no extra copy).
    alpha : float
        α-normalization power.
    n_components : int
        Number of diffusion components to return.
    working_dtype : numpy / cupy dtype, optional
        Device working precision. Defaults to fp32 (the production
        choice — see module docstring). Pass ``cp.float64`` for the
        legacy fp64 path.

    Returns
    -------
    emb : (N, n_components) fp32 ndarray (always returned in fp32 to
          match the MATLAB GT artifact, independent of working_dtype).
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh

    if working_dtype is None:
        working_dtype = cp.float32

    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"affinity must be square 2D, got {affinity.shape}")
    ndim = affinity.shape[0]
    if n_components < 1 or n_components >= ndim:
        raise ValueError(f"n_components={n_components} out of range for N={ndim}")

    # 1. Push to device at working precision (aliases if already there + matching dtype).
    L_alpha = cp.asarray(affinity, dtype=working_dtype)

    # 2. Alpha-normalization.
    if alpha > 0:
        d = L_alpha.sum(axis=1)
        d_alpha = cp.power(d, -alpha).astype(working_dtype)
        L_alpha *= d_alpha[:, cp.newaxis]
        L_alpha *= d_alpha[cp.newaxis, :]

    # 3. Symmetric similar matrix S = D2^-1/2 L_alpha D2^-1/2.
    d2 = L_alpha.sum(axis=1)
    inv_sqrt_d2 = (1.0 / cp.sqrt(d2)).astype(working_dtype)
    L_alpha *= inv_sqrt_d2[:, cp.newaxis]
    L_alpha *= inv_sqrt_d2[cp.newaxis, :]
    L_alpha = 0.5 * (L_alpha + L_alpha.T)        # exact symmetrize

    # 4. Top (n_components + 1) eigenpairs via Lanczos on device.
    k = n_components + 1
    v0 = cp.ones(ndim, dtype=working_dtype)
    eigvals, phi = eigsh(L_alpha, k=k, which="LM", v0=v0)
    # ascending -> descending
    eigvals = eigvals[::-1]
    phi = phi[:, ::-1]

    # Back to row-stochastic basis: psi = D2^-1/2 phi
    vectors = phi * inv_sqrt_d2[:, cp.newaxis]

    # 5. Renormalize by the trivial (constant) eigenvector.
    # Mirrors the CPU guard at diffusion_map.py:160-170 — see there
    # for why the connectedness invariant makes this double-defense.
    psi0_min = float(cp.min(cp.abs(vectors[:, 0])))
    if psi0_min < 1e-30:
        raise RuntimeError(
            f"diffusion_map_gpu: stationary eigenvector has a near-zero "
            f"entry (min |psi[:, 0]| = {psi0_min:.3e})."
        )
    psi = vectors / vectors[:, [0]]

    # 6. diffusion_time = 0 eigenvalue rescaling.
    lambdas_rescaled = eigvals[1:] / (1.0 - eigvals[1:])

    # 7. Embed.
    embedding = psi[:, 1 : n_components + 1] * lambdas_rescaled[:n_components][cp.newaxis, :]
    emb_host = cp.asnumpy(embedding.astype(cp.float32))

    return emb_host
