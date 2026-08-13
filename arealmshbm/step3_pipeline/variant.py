"""variant.py

Pipeline-variant specification (gMSHBM / cMSHBM / dMSHBM). The three
CBIG ArealMSHBM step-3 drivers share ~95% of their EM body; this module
captures the deltas as a small immutable struct so the super-call can
branch on data, not subclasses.

Single source of truth: every variant-sensitive site in the Python
pipeline reads its decision from a :class:`VariantSpec` field. No
``if pipeline_type == ...`` branches outside this module's
:meth:`VariantSpec.from_pipeline_type` factory.

See ``docs/pipeline_variants.md`` for the canonical delta table.

Public API:
    VariantSpec          — frozen capability spec for one variant.
    VALID_PIPELINE_TYPES — tuple of accepted ``name`` values.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass


VALID_PIPELINE_TYPES = ("gMSHBM", "cMSHBM", "dMSHBM")


@dataclass(frozen=True)
class VariantSpec:
    """Frozen capability spec for one pipeline variant.

    Field meanings (all are direct gates on the EM body):

    name : pipeline type label.
    use_connect_prior : build :class:`ConnectSession` and update the
        gradient-embedding-distance ``spatial_connect_vmf`` term per EM
        iter. gMSHBM only.
    use_xyz_prior : build :class:`XyzSession` and emit a
        ``spatial_xyz_vmf`` term. cMSHBM (always-on) + gMSHBM (only when
        ``check_connectedness`` flagged distributed parcels).
    use_check_connectedness : run the per-comp_iter connectedness step
        which mutates ``xyz_gamma`` (the multiplier driving the xyz
        prior's strength).
    first_em_iter_with_conn_xyz : ``check_connectedness`` + xyz update
        run only when ``iter_em >= this``. gMSHBM gates ``iter_em > 1``
        (so this is 2). cMSHBM runs from iter 1 (so this is 1). dMSHBM
        unused.
    components_threshold : the "distributed parcel" predicate is
        ``parcel_components > components_threshold``. gMSHBM uses 3,
        cMSHBM uses 1.
    pre_predicate_remove_isolated : cMSHBM's ``check_connectedness``
        applies :func:`remove_isolated_surface_components` to the argmax
        labels BEFORE the components/distance test (so small isolated
        blobs don't count as "distributed"). The size cutoff is
        configurable via ``Step3Config.cMSHBM_isolated_component_min_size``
        (default 5).
    wrap_comp_iter : run the inner ``while not stop_comp`` wrap around
        the λ-loop. gMSHBM/cMSHBM yes; dMSHBM has just a flat λ-loop
        per EM iter (no comp_iter).
    em_stop_uses_beta_scv : the EM-stop cost integration includes the
        ``+ Σ beta · s_lambda · spatial_connect_vmf`` term. gMSHBM only.
    final_remove_isolated : after pipeline argmax, apply
        :func:`remove_isolated_surface_components` on
        ``lh_labels`` / ``rh_labels``. cMSHBM only. Size cutoff is
        configurable via ``Step3Config.cMSHBM_isolated_component_min_size``
        (default 5).
    beta_internal_scale : the multiplier from user-facing ``beta_scalar``
        to the EM-internal ``beta`` vector (entered into log_vmf as
        ``beta[l] · scv[n,l]`` or ``beta[l] · sxv[n,l]``). gMSHBM uses
        1000.0 (matches ``setting_params.beta = repmat(beta*1000, ...)``).
        cMSHBM uses 1.0. dMSHBM has no beta term — internal beta is 0.
    output_subdir : leading folder for ``Step3Config.out_dir``
        (``ind_parcellation_<variant>``).
    output_subdir_includes_beta : whether the auto-built ``out_dir``
        nests under a ``beta<B>`` subfolder. dMSHBM omits it.
    filename_includes_beta : whether ``Ind_parcellation_*.mat``
        filename interpolates ``_beta<B>``. dMSHBM omits it.
    prior_path_includes_beta : whether the default
        ``priors/<variant>/[beta<B>/]Params_Final.mat`` carries a
        ``beta<B>`` segment. dMSHBM omits it.
    """

    name: str
    use_connect_prior: bool
    use_xyz_prior: bool
    use_check_connectedness: bool
    first_em_iter_with_conn_xyz: int
    components_threshold: int
    pre_predicate_remove_isolated: bool
    wrap_comp_iter: bool
    em_stop_uses_beta_scv: bool
    final_remove_isolated: bool
    beta_internal_scale: float
    output_subdir: str
    output_subdir_includes_beta: bool
    filename_includes_beta: bool
    prior_path_includes_beta: bool

    @classmethod
    def from_pipeline_type(cls, name: str) -> "VariantSpec":
        if name == "gMSHBM":
            return cls(
                name="gMSHBM",
                use_connect_prior=True,
                use_xyz_prior=True,
                use_check_connectedness=True,
                first_em_iter_with_conn_xyz=2,
                components_threshold=3,
                pre_predicate_remove_isolated=False,
                wrap_comp_iter=True,
                em_stop_uses_beta_scv=True,
                final_remove_isolated=False,
                beta_internal_scale=1000.0,
                output_subdir="ind_parcellation_gMSHBM",
                output_subdir_includes_beta=True,
                filename_includes_beta=True,
                prior_path_includes_beta=True,
            )
        if name == "cMSHBM":
            return cls(
                name="cMSHBM",
                use_connect_prior=False,
                use_xyz_prior=True,
                use_check_connectedness=True,
                first_em_iter_with_conn_xyz=1,
                components_threshold=1,
                pre_predicate_remove_isolated=True,
                wrap_comp_iter=True,
                em_stop_uses_beta_scv=False,
                final_remove_isolated=True,
                beta_internal_scale=1.0,
                output_subdir="ind_parcellation_cMSHBM",
                output_subdir_includes_beta=True,
                filename_includes_beta=True,
                prior_path_includes_beta=True,
            )
        if name == "dMSHBM":
            return cls(
                name="dMSHBM",
                use_connect_prior=False,
                use_xyz_prior=False,
                use_check_connectedness=False,
                first_em_iter_with_conn_xyz=10**9,    # never
                components_threshold=0,
                pre_predicate_remove_isolated=False,
                wrap_comp_iter=False,
                em_stop_uses_beta_scv=False,
                final_remove_isolated=False,
                beta_internal_scale=0.0,
                output_subdir="ind_parcellation_dMSHBM",
                output_subdir_includes_beta=False,
                filename_includes_beta=False,
                prior_path_includes_beta=False,
            )
        raise ValueError(
            f"VariantSpec.from_pipeline_type: unknown name {name!r}; "
            f"expected one of {VALID_PIPELINE_TYPES}"
        )


__all__ = ["VariantSpec", "VALID_PIPELINE_TYPES"]
