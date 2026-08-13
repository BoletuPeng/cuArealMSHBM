"""layout.py — single source of truth for paths inside a project.

Every cross-module question about "where does X live inside the project
dir" routes through :class:`ProjectLayout`. Existing per-step pipelines
(Step0/1/2/3) already use these path conventions; this class makes them
explicit and centralised so future moves only touch one file.

Path conventions::

    projects/<name>/
      bold_inputs.json                # user input
      pipeline_config.json            # user input
      cohort.json                     # driver-maintained cache record
      data_list/fMRI_list/{lh,rh}_sub<id>_sess<id>.txt
      gradients/sub<id>/{lh,rh}_emb_<N>_distance_matrix.npy
      profiles_raw/sub<id>/sub<id>_<targ>_roi<seed>.profile.b2nd
      profiles/avg_profile/{lh,rh}_<targ>_roi<seed>_avg_profile.npy
      group/group.mat
      spatial_mask/spatial_mask_<mesh>.mat
      priors/<variant>/beta<X>/Params_Final.mat
      ind_parcellation_<variant>/<T>_sess/beta<B>/sub<id>/Ind_parcellation_*.mat
      logs/pipeline_run_<timestamp>.json

(The ``ind_parcellation_*`` segment is owned by ``Step3Config`` —
the layout helper does not synthesize it.)

``<id>`` is the per-subject 1-based positional integer from
``bold_inputs.json`` (so ``sub1``, ``sub2``, ...) — keeps step1/2/3's
existing on-disk naming.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path


class ProjectLayout:
    """Path conventions inside one project directory.

    Constructed with the project root; every method returns a ``Path``
    relative to that root. Does NOT create directories — callers create
    them at write time.
    """

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir)

    # ── user inputs ──
    @property
    def bold_inputs_path(self) -> Path:
        return self.project_dir / "bold_inputs.json"

    @property
    def pipeline_config_path(self) -> Path:
        return self.project_dir / "pipeline_config.json"

    # ── driver-maintained cache record ──
    @property
    def cohort_json_path(self) -> Path:
        return self.project_dir / "cohort.json"

    # ── step0 outputs ──
    def gradient_dir(self, sub_id: str) -> Path:
        return self.project_dir / "gradients" / f"sub{sub_id}"

    def gradient_path(self, sub_id: str, hemi: str, n_components: int) -> Path:
        return self.gradient_dir(sub_id) / (
            f"{hemi}_emb_{n_components}_distance_matrix.npy"
        )

    # ── step1 outputs ──
    def profile_b2nd_path(self, sub_id: str, targ_mesh: str, seed_mesh: str) -> Path:
        # Mirrors arealmshbm.data_io.profile_io.profile_path.
        return (self.project_dir / "profiles_raw" / f"sub{sub_id}"
                / f"sub{sub_id}_{targ_mesh}_roi{seed_mesh}.profile.b2nd")

    @property
    def group_mat_path(self) -> Path:
        return self.project_dir / "group" / "group.mat"

    def spatial_mask_path(self, targ_mesh: str) -> Path:
        return self.project_dir / "spatial_mask" / f"spatial_mask_{targ_mesh}.mat"

    # ── group prior (step2 output for Mode B; creator-staged for Mode A) ──
    # ``include_beta`` mirrors the step3 VariantSpec.prior_path_includes_beta
    # policy: gMSHBM/cMSHBM carry a ``beta<X>`` segment, dMSHBM does NOT.
    # Callers that know the variant's policy (the driver) must pass it so
    # this matches where step2 writes (Step2Config.out_dir) and where step3
    # reads (Step3Config.group_prior_path). Defaults True for the common
    # gMSHBM case.
    def prior_dir(self, variant: str, beta_scalar: int,
                  include_beta: bool = True) -> Path:
        base = self.project_dir / "priors" / variant
        return base / f"beta{beta_scalar}" if include_beta else base

    def prior_path(self, variant: str, beta_scalar: int,
                   include_beta: bool = True) -> Path:
        return self.prior_dir(variant, beta_scalar, include_beta) / "Params_Final.mat"

    # ── logs ──
    @property
    def logs_dir(self) -> Path:
        return self.project_dir / "logs"

    def run_log_path(self, timestamp: str) -> Path:
        return self.logs_dir / f"pipeline_run_{timestamp}.json"
