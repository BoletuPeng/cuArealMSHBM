"""data_io

Step-3 file I/O leaves (Mode A).

Public API:
    fetch_data        — load per-subject bitpacked profile (``.b2nd``
                        via blosc2) + diffusion-embedding gradient
                        (``.npy`` canonical / ``.mat`` legacy) for one
                        subject.
    load_group_prior  — read Params_Final.mat (mu, theta, epsil, sigma).
    load_spatial_mask — read spatial_mask_<mesh>.mat → (lh, rh) boundary.
    load_avg_mesh     — fsaverage* mesh loader (vertices, vertexNbors,
                        MARS_label) for either inflated or sphere surface.
    read_surface_gifti — ``.func.gii`` BOLD reader (see ``gifti_io``).
    save_parcellation — argmax labels + save Ind_parcellation_*.mat.
    derive_labels     — argmax(s_lambda) → (lh_labels, rh_labels).

``gifti_bold_gpu.read_subject_bold_gpu`` decodes one subject's sessions
straight to device; cupy/nvcomp are soft deps, so it is not re-exported.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .fetch_data import fetch_data
from .load_group_prior import load_group_prior
from .load_spatial_mask import load_spatial_mask
from .load_avg_mesh import load_avg_mesh
from .save_parcellation import save_parcellation, derive_labels

# ``gifti_io`` is re-exported lazily (PEP 562): it pulls
# ``._gifti_kernels``, whose ``@njit`` decorators drag in the whole
# numba/llvmlite stack, which a *GPU* BOLD ingest never touches. Only
# this name may be deferred — every other export shares a name with
# its defining submodule, and an earlier ``import data_io.<name>`` binds
# that module as the attribute, which never reaches ``__getattr__``.
_LAZY_EXPORTS = {
    "read_surface_gifti": "gifti_io",
}


def __getattr__(name):
    """Resolve a lazily re-exported reader, or a submodule, on first access."""
    from importlib import import_module
    sub = _LAZY_EXPORTS.get(name)
    if sub is not None:
        value = getattr(import_module(f"{__name__}.{sub}"), name)
        globals()[name] = value
        return value
    if not name.startswith("_"):
        from importlib.util import find_spec
        try:
            found = find_spec(f"{__name__}.{name}") is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        if found:
            return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals()) + __all__))

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
