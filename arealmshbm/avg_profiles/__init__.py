"""avg_profiles

Step-1 leaf — average RSFC profiles across (subject × session).
Reads per-subject ``.b2nd`` profiles (T, N, D) written by step-1's
generate_profiles, splits the N axis into lh/rh, accumulates, divides
by ``num_sub × num_sess``, and writes the canonical avg_profile pair.

Backends (dispatched by the ``backend=`` kwarg on the supercall):
    cpu — numba in-place accum + scale kernels.
    gpu — cupy in-place ufuncs on a device-resident accumulator;
          cupy imported lazily.

Public API:
    avg_profiles(seed_mesh, targ_mesh, out_dir, num_sub, num_sess,
                  backend='cpu' | 'gpu')

Reads:
    <out_dir>/profiles_raw/sub<S>/sub<S>_<targ>_roi<seed>.profile.b2nd

Writes:
    <out_dir>/profiles/avg_profile/{lh,rh}_<targ>_roi<seed>_avg_profile.npy
    (raw fp32 ``(V_h, D)`` C-contig — consumed only by
    :mod:`arealmshbm.ini_params`)

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .avg_profiles import AvgProfilesResult, avg_profiles

__all__ = ["AvgProfilesResult", "avg_profiles"]
