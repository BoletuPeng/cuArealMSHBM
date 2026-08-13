"""generate_profiles

Step-1 leaf — per-session functional-connectivity profile (binary,
top-fraction thresholded) on the fsaverage* path.

For each (sub, sess): for every BOLD run, build a (K, V) seed-to-target
correlation matrix (fp32 GEMM after a fused per-column zscore+L2-norm),
NaN→0, average across runs, threshold the joint lh+rh sum at the top
``threshold`` fraction (selection, not full sort), binarize, and
return the per-hemi arrays (layout is backend-dependent — see
below). The step-1 pipeline stacks
the per-session arrays into a per-subject ``(T, N, D)`` block and
writes one ``.b2nd`` per subject via
:func:`arealmshbm.data_io.profile_io.write_subject_profile_tnd`.

Backends (dispatched by the ``backend=`` kwarg on the supercall):
    cpu — numba ``@njit(parallel=True)`` kernels + numpy/MKL BLAS.
    gpu — CuPy RawKernel zscore + cuBLAS sgemm + ufunc threshold /
          binarize. cupy is imported lazily — the cpu path never
          touches it.

Public API:
    compute_profile_arrays(seed_mesh, targ_mesh, out_dir, sub, sess, …,
                            backend='cpu' | 'gpu')
        — pure compute, returns ``(lh_arr, rh_arr, K_unpacked)``.
          cpu: ``(K, V_h) fp32`` binary arrays + ``K_unpacked=None``;
          the writer does the host transpose + packbits.
          gpu: ``(V_h, ⌈K/8⌉) uint8`` pre-packed bytes + integer
          ``K_unpacked`` — binarize + MW-zero + transpose + packbits
          fused into one device RawKernel.

Reads:
    <out_dir>/data_list/fMRI_list/{lh,rh}_sub<sub>_sess<sess>.txt

Writer (caller-side, in the step-1 pipeline):
    <out_dir>/profiles_raw/sub<sub>/sub<sub>_<targ>_roi<seed>.profile.b2nd

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .profiles import compute_profile_arrays

__all__ = ["compute_profile_arrays"]
