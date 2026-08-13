"""data_io

Step-3 file I/O leaves (Mode A).

Public API:
    fetch_data        — load per-subject bitpacked profile (``.b2nd``
                        via blosc2) + diffusion-embedding gradient
                        (``.npy`` canonical / ``.mat`` legacy) for one
                        subject. The historical "BOLD profile NIFTI"
                        reader was retired with the b2nd format switch.
    load_group_prior  — read Params_Final.mat (mu, theta, epsil, sigma).
    load_spatial_mask — read spatial_mask_<mesh>.mat → (lh, rh) boundary.
    load_avg_mesh     — fsaverage* mesh loader (vertices, vertexNbors,
                        MARS_label) for either inflated or sphere surface.
    save_parcellation — argmax labels + save Ind_parcellation_*.mat.
    derive_labels     — argmax(s_lambda) → (lh_labels, rh_labels).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .fetch_data import fetch_data
from .load_group_prior import load_group_prior
from .load_spatial_mask import load_spatial_mask
from .load_avg_mesh import load_avg_mesh
from .save_parcellation import save_parcellation, derive_labels
from .gifti_io import (
    read_surface_gifti, read_surface_gifti_gpu,
    read_surface_giftis_gpu_full_pipeline,
)
from .cohort import (
    CohortManifest, CohortSubject,
    backfill_cohort_json, cohort_json_path, compute_run_id,
    read_cohort, resolve_path, relpath_or_abs,
    write_cohort, write_cohort_partial,
)

__all__ = [
    "fetch_data",
    "load_group_prior",
    "load_spatial_mask",
    "load_avg_mesh",
    "save_parcellation",
    "derive_labels",
    "read_surface_gifti",
    "read_surface_gifti_gpu",
    "read_surface_giftis_gpu_full_pipeline",
    "CohortManifest",
    "CohortSubject",
    "backfill_cohort_json",
    "cohort_json_path",
    "compute_run_id",
    "read_cohort",
    "resolve_path",
    "relpath_or_abs",
    "write_cohort",
    "write_cohort_partial",
]
