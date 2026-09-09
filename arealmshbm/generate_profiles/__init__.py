"""generate_profiles

Step-1 leaf — functional-connectivity profile (binary, top-fraction
thresholded) on the fsaverage* path.

For each (sub, sess): for every BOLD run, build a (K, V) seed-to-target
correlation matrix (fp32 GEMM after a fused per-column zscore+L2-norm),
average across runs, threshold the joint lh+rh sum at the top
``threshold`` fraction (selection, not full sort), binarize and zero
the medial wall. The step-1 pipeline stacks the per-session results
into a per-subject ``(T, N, D)`` block and writes one ``.b2nd`` per
subject via :mod:`arealmshbm.data_io.profile_io`.

The two paths are shaped differently, not backend-switched: the CPU
one is per-session and returns arrays, the GPU one owns the whole
subject and writes the file itself.

Public API:
    compute_profile_arrays(seed_mesh, targ_mesh, out_dir, sub, sess, …)
        — CPU only. numba ``@njit(parallel=True)`` kernels + numpy/MKL
          BLAS. Pure compute, one (sub, sess) at a time, returning
          ``(lh_KxV_fp32, rh_KxV_fp32)``; the writer still owes the
          transpose + packbits.

    generate_subject_profiles_gpu(project_dir, sub_id, sess_ids, ...)
    compute_subject_profiles_gpu(sessions, ...)
    prewarm_generate_profiles_gpu(bold_paths=None)
        — the GPU path, WHOLE-SUBJECT, in :mod:`.profiles_subject_gpu`.
          Ingest of session g+1 overlaps the compute of session g, all
          sessions share one device packed buffer, and the subject
          leaves the GPU in a single pinned D2H; the .b2nd write runs
          on a background thread. Ingest is nvCOMP-batched when
          ``nvidia-nvcomp-cu12`` is installed and the CPU GIFTI reader
          otherwise — same bytes either way, so the compute and the
          artifacts are identical. Import it directly: the package
          ``__init__`` must stay cupy-free for the CPU path.

Reads:
    <out_dir>/data_list/fMRI_list/{lh,rh}_sub<sub>_sess<sess>.txt

Writer (caller-side, in the step-1 pipeline):
    <out_dir>/profiles_raw/sub<sub>/sub<sub>_<targ>_roi<seed>.profile.b2nd

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .profiles import compute_profile_arrays

__all__ = ["compute_profile_arrays"]
