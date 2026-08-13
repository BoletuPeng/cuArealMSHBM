"""config.py

Configuration dataclass for the step-0 super-call. Mode-A only —
fsaverage6 (the canonical CBIG validation mesh).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple


_SUPPORTED_MESHES = frozenset({"fsaverage6"})


@dataclass
class Step0Config:
    """All parameters for one step-0 single-subject run.

    Required:
        project_dir : Mode-A project root with ``converted_bold/``
                      populated; outputs land under
                      ``project_dir/gradients/sub<out_subid>/``.
        sub_id      : 'sub-001' / 'sub-002' / ... — drives BOLD lookup.
        sess_list   : tuple of session IDs.

    Defaults match the CBIG canonical call (sub_FC=100, sub_verts=200,
    block_a=3, block_b=10, smooth_sigma=2.55 mm, K_hop=3,
    watershed_steps=50, downsample=3.2, num_components=100).
    """

    project_dir: Path
    sub_id: str
    sess_list: Tuple[str, ...]

    out_subid: str = "1"
    mesh: str = "fsaverage6"
    sub_FC: int = 100
    sub_verts: int = 200
    block_a: int = 3
    block_b: int = 10
    smooth_sigma: float = 2.55
    K_hop: int = 3
    watershed_steps: int = 50
    watershed_frac: float = 1.0
    downsample: float = 3.2
    num_components: int = 100

    cbig_code_dir: Optional[Path] = None
    bold_root: Optional[Path] = None
    save_artifacts: bool = True

    # Explicit (lh, rh) path tuples per session, in 1-indexed order. When
    # set, ``bold_paths(sess_idx)`` returns directly from here and ignores
    # ``bold_root`` / ``sub_id`` / ``mesh``-derived naming. Used by the
    # unified pipeline driver (``arealmshbm.pipeline``) so the BOLD
    # list in ``bold_inputs.json`` is the single source of truth without
    # imposing a filename convention on the user.
    bold_paths_override: Optional[Tuple[Tuple[Path, Path], ...]] = None

    # External per-session BOLD getter. When non-None, ``Step0Pipeline.run``
    # skips its internal ThreadPoolExecutor and pulls ``(N_cortex, T)``
    # fp32 arrays straight from this callable. Used by the unified
    # pipeline driver to share one persistent worker pool across all
    # subjects — sessions of subject K+lookahead are decompressed in
    # background threads while subject K is still on the GPU, hiding
    # the per-subject cold-start latency. ``sess_idx`` is 1-indexed.
    #
    # Excluded from ``repr`` so dataclass logging stays clean.
    bold_provider: Optional[Callable[[int], Any]] = field(
        default=None, repr=False
    )

    # ── emb output format ──
    # The 100-component embedding ``emb`` is the per-subject artifact
    # downstream step-2 reads once per (subject, session) pair. .npy is
    # ~6× faster to read than scipy.io.loadmat on the same dense fp32
    # array (no MATLAB header parse, no struct overhead) and is
    # self-describing about shape/dtype.
    #   'npy'  — write {lh,rh}_emb_<nc>_distance_matrix.npy. Default.
    #   'mat'  — write the legacy CBIG .mat (key 'emb'). Kept for
    #            consumers that haven't migrated yet.
    #   'both' — write both. Useful when a downstream tool still
    #            expects the legacy .mat alongside the new .npy.
    emb_output_format: Literal["npy", "mat", "both"] = "npy"

    # 'cpu' = pure-numba CPU path (default).
    # 'gpu' = CuPy backend for the dominant hotspot (subgraph C
    #         eigsh / partial Lanczos). All other leaves stay on CPU
    #         numba; gpu mode requires `cupy` installed.
    backend: str = "cpu"

    # ── geodesic distance matrix dump ──
    # The (N_down, N_down) fp32 geodesic distance matrices
    # ``{lh,rh}_gradient_distance_matrix.npy`` are step0 *intermediate*
    # outputs (subgraph B → C handoff). They are large (~672 MB per
    # hemi per subject, ~53.6 GB for a 40-subject cohort on fsaverage6)
    # and no production downstream step reads them — step3's fetch_data
    # consumes only the much smaller ``{lh,rh}_emb_<nc>_distance_matrix``
    # embedding, and the driver passes the embedding in memory anyway
    # (``precomputed_gradient_mat`` keyword).
    #
    # Default OFF since 2026-05. No in-tree consumer; toggle on for
    # ad-hoc archaeological probes (cohort-level CPU/GPU drift comparison
    # via the internal step-0 CPU/GPU diff harness).
    save_geodesic_distance: bool = False

    # ── edge_density.npy dump ──
    # ``edge_density.npy`` is a step-0 *intermediate* (subgraph A →
    # subgraph B handoff): subgraph A produces it from
    # watershed → edge_count and subgraph B consumes it in-memory to
    # seed the gradient-geodesic-distance Dijkstra. No production
    # downstream step reads the disk artifact — step3's pipeline
    # consumes only the embeddings, and step1/2 never look at it. The
    # only consumer is the internal step-0 CPU/GPU bit-equality
    # diff harness.
    #
    # Default OFF since 2026-06. Set True only when running that
    # comparison script (or any future tool that wants the on-disk
    # ``edge_density.npy``). Independent of ``save_geodesic_distance``
    # — gates a different intermediate.
    save_edge_density: bool = False

    def __post_init__(self):
        if self.mesh not in _SUPPORTED_MESHES:
            raise ValueError(
                f"step-0 currently supports only mesh in {sorted(_SUPPORTED_MESHES)}; "
                f"got {self.mesh!r}"
            )
        if self.backend not in ("cpu", "gpu"):
            raise ValueError(
                f"backend must be 'cpu' or 'gpu' (got {self.backend!r})"
            )
        if self.backend == "gpu":
            # Fail at config time on a missing GPU stack rather than ~30 s
            # into the pipeline at the first eigsh call.
            try:
                import cupy  # noqa: F401
                from cupyx.scipy.sparse.linalg import eigsh  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "backend='gpu' requires cupy + cupyx.scipy.sparse.linalg; "
                    f"import failed ({e}). Use backend='cpu' or install cupy."
                ) from e
        self.project_dir = Path(self.project_dir)
        if self.cbig_code_dir is not None:
            self.cbig_code_dir = Path(self.cbig_code_dir)
        if self.bold_root is not None:
            self.bold_root = Path(self.bold_root)
        if not isinstance(self.sess_list, tuple):
            self.sess_list = tuple(self.sess_list)
        if len(self.sess_list) == 0:
            raise ValueError("sess_list must be non-empty")
        for k in ("sub_FC", "sub_verts", "block_a", "block_b",
                  "K_hop", "watershed_steps", "num_components"):
            v = getattr(self, k)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{k} must be a positive int; got {v!r}")
        for k in ("smooth_sigma", "watershed_frac", "downsample"):
            v = getattr(self, k)
            if not (isinstance(v, (int, float)) and v > 0):
                raise ValueError(f"{k} must be a positive number; got {v!r}")

    @property
    def num_sess(self) -> int:
        return len(self.sess_list)

    @property
    def gradients_out_dir(self) -> Path:
        return self.project_dir / "gradients" / f"sub{self.out_subid}"

    def bold_paths(self, sess_idx: int) -> Tuple[Path, Path]:
        """Resolve (lh, rh) BOLD ``.func.gii`` paths for one session
        (1-indexed).

        Priority:
          1. ``bold_paths_override`` (driver-injected explicit list);
          2. ``bold_root`` + ``sub_id`` + ``sess`` + ``mesh`` naming —
             synthesizes the canonical DeepPrep / fmriprep filename
             ``{sub}_{ses}_task-rest_hemi-{L,R}_space-{mesh}_bold.func.gii``.
             The historical ``.nii.gz`` mirror was retired together with
             the offline ``convert_ys_bold_parallel.py`` script; the
             pipeline now reads GIFTI source directly via
             :func:`arealmshbm.bold_io.read_surface_bold`, which hard-
             rejects any non-``.gii`` suffix.
        """
        if self.bold_paths_override is not None:
            return self.bold_paths_override[sess_idx - 1]
        sess = self.sess_list[sess_idx - 1]
        root = self.bold_root if self.bold_root is not None else (
            self.project_dir / "converted_bold"
        )
        lh = root / (
            f"{self.sub_id}_{sess}_task-rest_hemi-L_"
            f"space-{self.mesh}_bold.func.gii"
        )
        rh = root / (
            f"{self.sub_id}_{sess}_task-rest_hemi-R_"
            f"space-{self.mesh}_bold.func.gii"
        )
        return lh, rh
