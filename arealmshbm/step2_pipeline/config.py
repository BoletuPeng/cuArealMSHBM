"""config.py

Configuration dataclass for the step-2 super-call (group prior
estimation). Mode-B style: a coupled multi-subject EM estimates
``Params = (mu, theta, epsil, sigma, kappa)`` from a training cohort.

Dispatch on ``mode ∈ {'gMSHBM', 'dMSHBM'}``. cMSHBM is recognised
but raises ``NotImplementedError`` — no xyz-vMF path in the master
kernel.

Public API:
    Step2Config — all parameters for one step-2 group-prior estimation.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


_SUPPORTED_MESHES = frozenset({
    "fsaverage", "fsaverage4", "fsaverage5", "fsaverage6",
})

_VALID_MODES = frozenset({"gMSHBM", "dMSHBM"})
# cMSHBM is not wired — the master kernel has no xyz-vMF (s_muc / gamma /
# log_xyz_term) path. Passing mode='cMSHBM' raises in __post_init__.

_VALID_BACKENDS = frozenset({"cpu", "gpu"})


@dataclass
class Step2Config:
    """All parameters for one step-2 group-prior estimation run.

    Required fields:
    ----------------
    project_dir : path containing the Mode-B layout produced by step1:
        project_dir/
          cohort.json                                    (roster + artifact ledger)
          profiles_raw/sub<S>/sub<S>_<…>.profile.b2nd    (per-subject BOLD)
          gradients/sub<S>/{lh,rh}_emb_<N>_*.npy|.mat    (gMSHBM only)
          group/group.mat
          spatial_mask/spatial_mask_<mesh>.mat
    num_sub : S — number of subjects in the training cohort.
                  Must match cohort.json's roster size.
    num_session : T — number of sessions per subject.
                      Must match cohort.json's per-subject sessions.
    num_clusters : L — number of cortical parcels.
    mode : 'gMSHBM' | 'dMSHBM'.  (cMSHBM deferred — see module note above.)

    Mode-dependent:
    ---------------
    beta_scalar : raw beta. gMSHBM requires it; dMSHBM ignores it.
                  Internal beta = ``beta_scalar * 1000`` for gMSHBM,
                  ``0`` for dMSHBM.

    Loop caps + tolerances (defaults match MATLAB):
    -----------------------------------------------
    max_iter_inter : outer EM cap. MATLAB default 10.
    max_iter_intra_em : intra-EM loop cap inside each outer iter. MATLAB hard cap 15.
    max_iter_em : vMF clustering EM body cap. MATLAB hard cap 100.
    max_iter_m : M-step inner loop cap. MATLAB hard cap 50.
    max_iter_intra_var : intra_subject_var inner loop cap. MATLAB hard cap 20.
    epsilon : per-cluster convergence tolerance on cosine / rel-diff. MATLAB 1e-4.
    intra_em_convergence_eps : rel-diff threshold on intra-EM cost. 1e-4.
    inter_convergence_eps : rel-diff threshold on outer-EM cost. 1e-5.
    em_convergence_eps : per-subject rel-diff on EM body cost. 1e-4.

    Output routing:
    ---------------
    out_dir : where to write Params_Final.mat. Default per CBIG:
        gMSHBM:  project_dir/priors/gMSHBM/beta<B>/Params_Final.mat
        dMSHBM:  project_dir/priors/dMSHBM/Params_Final.mat

    Misc:
    -----
    verbose : print per-stage / per-iter timing logs.
    """

    # ── required ──
    project_dir: str | Path = ""
    num_sub: int = 0
    num_session: int = 0
    num_clusters: int = 0
    mode: Literal["gMSHBM", "dMSHBM"] = "gMSHBM"

    # ── mode-dependent ──
    beta_scalar: float = 5.0

    # ── backend ──
    # 'cpu' — numba master kernel (default).
    # 'gpu' — CuPy port of the master; outer-EM closure leaves (L16/L17/L18)
    #          stay on CPU since they run only I·J ≈ 30 times per pipeline.
    #          State lives device-resident across all I·J·K outer-EM iters;
    #          only s_t_nu (~11 MB), kappa (1.6 KB) and cost_em (24 B)
    #          round-trip per vmf_clustering_batch call.
    backend: Literal["cpu", "gpu"] = "cpu"

    # GPU-only knob — how the per-subject BOLD reaches the device.
    # 'auto'             — picks ``eager_bitpacked`` when it fits in
    #                       free device memory minus
    #                       ``gpu_cache_safety_margin_gb`` headroom;
    #                       otherwise falls through to ``stream``.
    #                       Default.
    # 'eager_bitpacked'  — preload (S, N, T, ⌈D/8⌉) uint8 packed cache
    #                       on device (1 bit/cell); per-iter BOLD comes
    #                       from a device-side bitpacked→fp32 widen+
    #                       normalize RawKernel. Footprint: ~72 MB at
    #                       S=3 fsa6 T=2 D=1175; ~14.5 GB at S=200
    #                       fsa6 T=6. Note: this only sizes the BOLD
    #                       cache — other on-device state (notably the
    #                       (S, N, L) fp32 s_lambda buffer at ~19.7 GB
    #                       for S=200 L=300, ~26 GB for L=400) is the
    #                       real ceiling on a 24 GB card; eager_bitpacked
    #                       enables the largest cohort the buffer
    #                       budget admits, not arbitrary S=200.
    # 'stream'           — per-iter disk decode + pinned-host H2D. Safe
    #                       fallback for cohorts that don't fit the
    #                       packed cache; ~10-15× slower em_total than
    #                       eager_bitpacked.
    #
    # The legacy ``'eager'`` (unpacked uint8 device cache, ~8× larger
    # than bitpacked) mode was removed in 2026-06 — bitpacked subsumes
    # it (same fp32 contract, 8× less device read traffic, ~15% faster
    # kernel wall).
    bold_cache_mode: Literal[
        "auto", "eager_bitpacked", "stream"
    ] = "auto"
    # Headroom (GB) reserved on device when ``bold_cache_mode='auto'``
    # decides whether eager fits. Should cover sgemm scratch + softmax
    # + the rest of step-2's session state (~1-2 GB at S=3, more at
    # large L); 4 GB is conservative for fsa6 / L=400.
    gpu_cache_safety_margin_gb: float = 4.0

    # ── mesh ──
    mesh: str = "fsaverage6"
    # Seed mesh used by step-1's profile writer; baked into the .b2nd
    # canonical filename (``sub<S>_<targ>_roi<seed>.profile.b2nd``).
    # Step-2's BOLD loader uses this to find the on-disk b2nd files.
    seed_mesh: str = "fsaverage3"
    # Gradient embedding components per hemi (used only in gMSHBM mode).
    # Matches step-3's Step3Config.n_grad_components default.
    n_grad_components: int = 100

    # ── loop caps + tolerances ──
    max_iter_inter: int = 10
    max_iter_intra_em: int = 15
    max_iter_em: int = 100
    max_iter_m: int = 50
    max_iter_intra_var: int = 20
    epsilon: float = 1e-4
    intra_em_convergence_eps: float = 1e-4
    inter_convergence_eps: float = 1e-5
    em_convergence_eps: float = 1e-4

    # ── output ──
    out_dir: Optional[str | Path] = None  # auto-derived if None

    # ── env ──
    verbose: bool = True

    def __post_init__(self) -> None:
        if not self.project_dir:
            raise ValueError("Step2Config: project_dir is required")
        if self.mesh not in _SUPPORTED_MESHES:
            raise ValueError(
                f"Step2Config: mesh must be one of "
                f"{sorted(_SUPPORTED_MESHES)} (got {self.mesh!r})"
            )
        # cMSHBM is recognised but not implemented — raise the
        # semantically-correct NotImplementedError (matches
        # vmf_clustering_batch + Step2EmIterSession).
        if self.mode == "cMSHBM":
            raise NotImplementedError(
                "Step2Config: mode='cMSHBM' is not wired (no xyz-vMF "
                "path in the master kernel). Use 'gMSHBM' or 'dMSHBM'."
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Step2Config: mode must be one of {sorted(_VALID_MODES)} "
                f"(got {self.mode!r})"
            )
        if self.num_sub <= 0:
            raise ValueError(f"Step2Config: num_sub must be positive (got {self.num_sub})")
        if self.num_session <= 0:
            raise ValueError(f"Step2Config: num_session must be positive (got {self.num_session})")
        if self.num_clusters <= 0:
            raise ValueError(f"Step2Config: num_clusters must be positive (got {self.num_clusters})")
        if self.mode == "gMSHBM" and self.beta_scalar is None:
            raise ValueError(f"Step2Config: beta_scalar required for gMSHBM")
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Step2Config: backend must be one of {sorted(_VALID_BACKENDS)} "
                f"(got {self.backend!r})"
            )
        # Loop-cap / tolerance / GPU-headroom range checks. These used
        # to live in a duplicate parser-side ``_validate_step2`` —
        # consolidated here as the single owner so future range edits
        # only touch one site (parser-side validator dropped 2026-06).
        for name in ("max_iter_inter", "max_iter_intra_em", "max_iter_em",
                     "max_iter_m", "max_iter_intra_var"):
            v = int(getattr(self, name))
            if v <= 0:
                raise ValueError(
                    f"Step2Config: {name} must be positive (got {v})"
                )
        for name in ("epsilon", "intra_em_convergence_eps",
                     "inter_convergence_eps", "em_convergence_eps"):
            v = float(getattr(self, name))
            if v <= 0.0:
                raise ValueError(
                    f"Step2Config: {name} must be positive (got {v})"
                )
        if self.gpu_cache_safety_margin_gb < 0.0:
            raise ValueError(
                f"Step2Config: gpu_cache_safety_margin_gb must be >= 0 "
                f"(got {self.gpu_cache_safety_margin_gb})"
            )

        self.project_dir = Path(self.project_dir)

        if self.out_dir is None:
            self.out_dir = self._default_out_dir()
        else:
            self.out_dir = Path(self.out_dir)

    # ── derived ──
    @property
    def beta_internal(self) -> float:
        """Internal beta the MATLAB code uses inside the E-step."""
        if self.mode == "gMSHBM":
            return float(self.beta_scalar) * 1000.0
        return 0.0  # dMSHBM — no spatial prior

    @property
    def beta_str(self) -> str:
        """The ``betaN`` string used in CBIG output paths."""
        b = self.beta_scalar
        return f"{int(b)}" if float(b).is_integer() else f"{b}"

    def _default_out_dir(self) -> Path:
        base = self.project_dir / "priors"
        if self.mode == "dMSHBM":
            return base / "dMSHBM"
        return base / self.mode / f"beta{self.beta_str}"
