"""ini_params

Step-1 leaf — vMF initial parameters from a group-level parcellation
+ averaged RSFC profile.

For each parcel l ∈ 1..L: average the L2-normalized profile rows over
all vertices with that label (column-group sum via one-hot λ exploit),
column-renormalize → ``mtc``. Solve ``A_d(ε) = mean inner-product of
each kept row with its parcel's mtc column`` for ε via the Bessel-ratio
root finder ``invAd`` (D-1 ≈ 1174). Save mtc, ε, dense one-hot λ, and
both label vectors.

Backends (dispatched by the ``backend=`` kwarg on the supercall):
    cpu — numba ``@njit(parallel=True)`` kernels + numpy/MKL BLAS.
    gpu — CuPy RawKernels end to end: row-demean/L2, a parcel-CSR
          fp64 ``groupsum`` in the CPU kernel's exact summation order
          (bit-exact ``mtc``), an in-place column L2-renorm in numpy's
          ``sum(axis=0)`` order, and a fused per-row dot for the
          epsilon input (replacing the ``profile @ mtc`` dgemm and its
          (N, L) intermediate). invAd (scipy Bessel root solve, ~ms on
          a single scalar) stays CPU. cupy imported lazily.

Public API:
    generate_ini_params(seed_mesh, targ_mesh, lh_labels, rh_labels, out_dir,
                         …, backend='cpu' | 'gpu')
        Optional hand-offs: ``precomputed_{lh,rh}_avg`` (host arrays),
        ``precomputed_{lh,rh}_avg_dev`` (cupy arrays from
        ``avg_profiles_from_packed_gpu``; GPU backend only),
        ``save_async`` (background ``group.mat`` write — join via
        ``result.writer.wait()``).

Reads:
    <out_dir>/profiles/avg_profile/{lh,rh}_<targ>_roi<seed>_avg_profile.npy
    (raw fp32 ``(V_h, D)`` C-contig, as written by :mod:`avg_profiles`)

Writes:
    <out_dir>/group/group.mat with keys
        ``mtc``       — (D, L) fp64 column-unit-norm cluster means
        ``epsil``     — scalar fp64 vMF concentration (invAd output)
        ``lambda``    — (N_keep, L) uint8 dense one-hot assignments
        ``lh_labels`` — (V_lh, 1) input lh labels (unmodified)
        ``rh_labels`` — (V_rh, 1) rh labels offset by ``max(lh_labels)``
                        on non-zero entries.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from ._group_mat_writer import IniParamsResult
from .ini_params import generate_ini_params

__all__ = ["IniParamsResult", "generate_ini_params"]
