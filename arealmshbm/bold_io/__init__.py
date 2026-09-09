"""bold_io

BOLD ingest for surface-space fMRI.

Public API:
    read_surface_bold           — read one ``.func.gii`` hemisphere and
                                  return an ``(N, T) fp32`` matrix.
                                  Thin wrapper over
                                  :func:`arealmshbm.data_io.gifti_io.read_surface_gifti`
                                  that adds the optional ``expected_n``
                                  guard.
    concat_hemis_drop_medial    — vstack lh + rh, replace NaN with 0
                                  (per hemi, matching MATLAB ordering),
                                  then drop rows flagged as medial wall.

Precision policy:
    Storage and arithmetic are fp32 throughout; this matches MATLAB's
    ``single`` everywhere downstream of MRIread. End-to-end
    bit-equality vs the historical converted-NIFTI route was verified
    on real YS sub-001 data before the NIFTI mirror was retired —
    every numerical pipeline artifact (profile.b2nd, avg_profile,
    gradients, parcellation labels) matched bit-for-bit when step0
    runs on CPU (GPU eigsh has intrinsic non-determinism that affects
    both BOLD sources identically).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .bold_io import read_surface_bold, concat_hemis_drop_medial

__all__ = [
    "read_surface_bold",
    "concat_hemis_drop_medial",
]
