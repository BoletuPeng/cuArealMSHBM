"""config.py

Configuration dataclass for the step-3 super-call. Bundles every knob
the Python pipeline accepts; constructor is keyword-only.

Mode-A only — fsaverage / fsaverage4 / fsaverage5 / fsaverage6.

Public API:
    Step3Config — all parameters for one single-subject parcellation run.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .variant import VariantSpec, VALID_PIPELINE_TYPES


_SUPPORTED_MESHES = frozenset({
    "fsaverage", "fsaverage4", "fsaverage5", "fsaverage6",
})


@dataclass
class Step3Config:
    """All parameters for one step-3 single-subject parcellation run.

    Required fields (no defaults; caller must set):
    -------------------------------------------------
    project_dir : path to the project directory produced by step1:
                    project_dir/
                      cohort.json                                (roster + artifacts)
                      profiles_raw/sub<S>/sub<S>_*.profile.b2nd  (per-subject BOLD)
                      gradients/sub<S>/{lh,rh}_emb_*.npy|.mat    (per-subject gradient)
                      priors/gMSHBM/beta<B>/Params_Final.mat     (group prior)
                      spatial_mask/spatial_mask_<mesh>.mat       (spatial prior)
    num_session : T — number of sessions for this subject. Validated
                  against the cohort manifest.
    num_clusters : L — number of cortical parcels (e.g. 300).

    Optional, with defaults matching the MATLAB driver's example case:
    ------------------------------------------------------------------
    subid : 1-based positional index into the cohort's ``subjects``
            list (i.e. the n-th subject step1 was invoked with).
    mesh : surface space; 'fsaverage6' / 'fsaverage5' / 'fsaverage'.
    w : weight on the group spatial prior log(theta).
    c : weight on the MRF Potts smoothness prior.
    beta_scalar : raw beta (e.g. 5). The internal beta vector replicated
                  to (L,) is ``beta_scalar * 1000`` per the MATLAB rule
                  (``setting_params.beta = repmat(str2double(beta)*1000,
                  1, num_clusters)``). Keep both representations so a
                  future GUI binds to the user-facing scalar without
                  losing the MATLAB-equivalent internal weight.
    backend : 'cpu' | 'gpu_elambda' | 'gpu_full'. See
              :class:`arealmshbm.vmf_clustering.VmfClusteringSession`.
    connect_th : connectedness threshold for the distributed-parcel test.
    epsilon : convergence tolerance shared across the four nested loops.
    max_iter_intra_em : outer loop cap (MATLAB hardcodes 50).
    max_iter_em : inner EM cap (MATLAB hardcodes 100).
    max_iter_lambda : λ-loop cap (MATLAB hardcodes 50).
    max_iter_m : M-step cap (MATLAB hardcodes 50).
    max_iter_comp : comp_iter cap (MATLAB hardcodes 15).
    n_grad_components : keep first N gradient components (MATLAB hardcodes 100).

    Output routing:
    ---------------
    out_dir : where to write Ind_parcellation_*.mat. Default:
                project_dir/ind_parcellation_gMSHBM/<T>_sess/beta<B>/
    out_subid : the id string interpolated into the filename. Default is
                str(subid) — kept separate so a multi-subject driver
                can save under "sub-002" while the cohort.json data
                ingest still uses subid=1 (Mode A's typical layout has
                one-subject-per-cohort).
    """

    # ── required ──
    project_dir: str | Path = ""
    num_session: int = 0
    num_clusters: int = 0

    # ── core knobs (defaults match driver-example) ──
    subid: int = 1
    mesh: str = "fsaverage6"
    w: float = 50.0
    c: float = 10.0
    beta_scalar: float = 5.0
    # ``pipeline_type`` selects between the three CBIG variants:
    #   * 'gMSHBM' — gradient-based areal (default; current production).
    #   * 'cMSHBM' — strictly contiguous areal (xyz prior + RemoveIsolated).
    #   * 'dMSHBM' — distributed areal (no spatial priors, no connectedness).
    # See ``docs/pipeline_variants.md`` and :class:`VariantSpec`.
    pipeline_type: str = "gMSHBM"

    # ── runtime ──
    backend: str = "cpu"
    # ``connect_th`` is the "distributed parcel" euclidean-distance gate.
    # gMSHBM = 15 mm; cMSHBM = 0 mm (any disconnected component triggers);
    # dMSHBM unused. Leave as ``None`` (default) to pick the variant
    # default automatically; pass an explicit float to override.
    connect_th: Optional[float] = None
    epsilon: float = 1e-4
    max_iter_intra_em: int = 50
    max_iter_em: int = 100
    max_iter_lambda: int = 50
    max_iter_m: int = 50
    max_iter_comp: int = 15
    n_grad_components: int = 100
    # cMSHBM-only: minimum size (in surface vertices) for parcels to
    # survive ``remove_isolated_surface_components``. Used at two sites:
    #   * inner ``check_connectedness`` pre-predicate inside the EM body
    #     (vmf_clustering.py / vmf_clustering_gpu.py).
    #   * outer post-EM final cleanup before save (Step3Pipeline.save).
    # gMSHBM / dMSHBM never read this. Default 5 matches the CBIG
    # MATLAB driver (the only value the reference implementation uses).
    cMSHBM_isolated_component_min_size: int = 5

    # ── output routing ──
    out_dir: Optional[str | Path] = None
    out_subid: Optional[str] = None
    # Optional override for the group-prior file path. When None
    # (default), :attr:`group_prior_path` is built from ``pipeline_type``
    # and ``beta_scalar`` per the CBIG layout. Set explicitly when the
    # native variant prior is unavailable (e.g. cMSHBM/dMSHBM validation
    # against MATLAB GT borrows the gMSHBM ``Params_Final.mat`` — see
    # ``docs/pipeline_variants.md`` §4.4).
    group_prior_path_override: Optional[str | Path] = None

    # ── frozen / derived (filled in __post_init__) ──
    _resolved: bool = field(default=False, init=False, repr=False)
    _variant: Optional[VariantSpec] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.project_dir:
            raise ValueError("Step3Config: project_dir is required")
        if self.num_session <= 0:
            raise ValueError(
                f"Step3Config: num_session must be positive (got {self.num_session})"
            )
        if self.num_clusters <= 0:
            raise ValueError(
                f"Step3Config: num_clusters must be positive (got {self.num_clusters})"
            )
        if self.subid < 1:
            raise ValueError(
                f"Step3Config: subid must be >= 1 (got {self.subid})"
            )
        if self.backend not in ("cpu", "gpu_elambda", "gpu_full"):
            raise ValueError(
                f"Step3Config: backend must be one of "
                f"'cpu' / 'gpu_elambda' / 'gpu_full' (got {self.backend!r})"
            )
        if self.pipeline_type not in VALID_PIPELINE_TYPES:
            raise ValueError(
                f"Step3Config: pipeline_type must be one of "
                f"{VALID_PIPELINE_TYPES} (got {self.pipeline_type!r})"
            )
        # Resolve VariantSpec early — every default-derivation below
        # branches on it.
        self._variant = VariantSpec.from_pipeline_type(self.pipeline_type)
        if self.mesh not in _SUPPORTED_MESHES:
            raise ValueError(
                f"Step3Config: mesh must be one of {sorted(_SUPPORTED_MESHES)} "
                f"(got {self.mesh!r}). fs_LR_32k is left for a later port."
            )
        # Numeric guards. The hot path doesn't validate these; a negative
        # epsilon or zero iter cap would silently produce wrong / never-
        # terminating runs. ``w/c/beta_scalar`` are non-negative weights
        # (0 disables the corresponding prior term — meaningful, kept).
        if self.epsilon <= 0.0:
            raise ValueError(
                f"Step3Config: epsilon must be positive (got {self.epsilon})"
            )
        # connect_th is only consulted when ``use_check_connectedness``
        # fires (gMSHBM, cMSHBM). dMSHBM never reads it. Default per
        # variant (gMSHBM=15, cMSHBM=0); explicit float overrides.
        if self.connect_th is None:
            if self.pipeline_type == "gMSHBM":
                self.connect_th = 15.0
            elif self.pipeline_type == "cMSHBM":
                self.connect_th = 0.0
            else:
                self.connect_th = 0.0    # dMSHBM: unused, value irrelevant
        if self._variant.use_check_connectedness and self.connect_th < 0.0:
            raise ValueError(
                f"Step3Config: connect_th must be >= 0 (got {self.connect_th})"
            )
        if self.w < 0.0:
            raise ValueError(f"Step3Config: w must be >= 0 (got {self.w})")
        if self.c < 0.0:
            raise ValueError(f"Step3Config: c must be >= 0 (got {self.c})")
        if self.beta_scalar < 0.0:
            raise ValueError(
                f"Step3Config: beta_scalar must be >= 0 (got {self.beta_scalar})"
            )
        for name in ("max_iter_intra_em", "max_iter_em", "max_iter_lambda",
                     "max_iter_m", "max_iter_comp"):
            v = int(getattr(self, name))
            if v <= 0:
                raise ValueError(
                    f"Step3Config: {name} must be positive (got {v})"
                )
        if int(self.n_grad_components) <= 0:
            raise ValueError(
                f"Step3Config: n_grad_components must be positive "
                f"(got {self.n_grad_components})"
            )
        if int(self.cMSHBM_isolated_component_min_size) <= 0:
            raise ValueError(
                f"Step3Config: cMSHBM_isolated_component_min_size must be "
                f"positive (got {self.cMSHBM_isolated_component_min_size})"
            )
        self.project_dir = Path(self.project_dir)
        if self.out_dir is None:
            base = (
                self.project_dir
                / self._variant.output_subdir
                / f"{int(self.num_session)}_sess"
            )
            self.out_dir = (
                base / f"beta{_fmt_num(self.beta_scalar)}"
                if self._variant.output_subdir_includes_beta
                else base
            )
        else:
            self.out_dir = Path(self.out_dir)
        if self.out_subid is None:
            self.out_subid = str(int(self.subid))
        if self.group_prior_path_override is not None:
            self.group_prior_path_override = Path(self.group_prior_path_override)
        self._resolved = True

    # ── derived accessors ──
    @property
    def variant(self) -> VariantSpec:
        """The :class:`VariantSpec` for this run's ``pipeline_type``."""
        assert self._variant is not None, "Step3Config.__post_init__ did not run"
        return self._variant

    @property
    def beta_internal(self) -> float:
        """Internal beta vector multiplier per variant.

        gMSHBM: ``beta_scalar * 1000`` (matches MATLAB
        ``setting_params.beta = repmat(beta*1000, ...)``).
        cMSHBM: ``beta_scalar`` (no rescale).
        dMSHBM: ``0`` (no beta term in EM body).
        """
        return float(self.beta_scalar) * float(self.variant.beta_internal_scale)

    @property
    def group_prior_path(self) -> Path:
        """Resolved ``Params_Final.mat`` path.

        Returns ``group_prior_path_override`` when set; otherwise builds
        the per-variant default
        ``project_dir/priors/<variant>/[beta<B>/]Params_Final.mat``.
        dMSHBM omits the ``beta<B>`` segment.
        """
        if self.group_prior_path_override is not None:
            return Path(self.group_prior_path_override)
        base = self.project_dir / "priors" / self.variant.name
        if self.variant.prior_path_includes_beta:
            base = base / f"beta{_fmt_num(self.beta_scalar)}"
        return base / "Params_Final.mat"

    @property
    def spatial_mask_path(self) -> Path:
        return self.project_dir / "spatial_mask" / f"spatial_mask_{self.mesh}.mat"


def _fmt_num(x: float | int | str) -> str:
    """Match MATLAB ``num2str`` for the integer / small-float regime used
    by these knobs (``5`` -> ``"5"``, ``0.5`` -> ``"0.5"``).
    Strings pass through unchanged.
    """
    if isinstance(x, str):
        return x
    return f"{float(x):g}"
