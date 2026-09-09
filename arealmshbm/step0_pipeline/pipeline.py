"""pipeline.py

End-to-end Python step-0 super-call. One entry point that consumes a
CBIG-style ``project_dir`` (with ``converted_bold/`` populated) and
writes the diffusion-embedding gradient matrices that step-3 reads as
its ``gradient`` spatial prior.

Composes the four subgraphs of CBIG step 0 (see
``docs/step0_flow_and_subgraphs.md`` for the full call graph):

  Subgraph A — RSFC gradients
      bold_io → subsampling → fc_similarity → surface_gradient (per
      scan, per block-a) → per-block accumulator → surface_smoothing →
      local_minima → watershed → edge_density.
  Subgraph B — gradient distance (per hemi)
      icosphere → mesh_topology → interpolate_sphere (downsample) →
      graph_distance.
  Subgraph C — diffusion embedding (per hemi)
      diffusion_map. CPU / GPU dispatched on ``cfg.backend``.
  Subgraph D — upsample (per hemi)
      interpolate_sphere, source/target swapped.

Hot-path kernels are ``@njit`` numba CPU; tech-stack exceptions
delegate to BLAS sgemm (``numpy @``) and Lanczos partial eigh
(``scipy.sparse.linalg.eigsh`` or ``cupyx.scipy.sparse.linalg.eigsh``).

Public API:
    Step0Inputs   — meshes / neighbors / smoothing prep / medial mask
                    (subject-independent for fsaverage6).
    Step0Result   — per-hemi emb_up plus per-stage timings.
    Step0Pipeline — single-subject load → run → save lifecycle, with
                    ``with``-block context-manager teardown.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

# All mesh / neighbor / smoothing-prep imports moved to
# ``arealmshbm.precompute.step0_inputs_builder`` (it owns the heavy
# precomputation; the production path here reads its output from disk).
from arealmshbm.icosphere import make_icosphere

from arealmshbm.bold_io import (
    read_surface_bold,
    concat_hemis_drop_medial,
)
from arealmshbm.subsampling import set_downsample_params
from arealmshbm.fc_similarity import (
    compute_t_series,
    compute_FC_simi_block,
)
from arealmshbm.surface_gradient import cifti_gradient
from arealmshbm.surface_smoothing import cifti_smoothing
from arealmshbm.local_minima import find_minima
from arealmshbm.watershed import watershed_algorithm_repaired

from arealmshbm.interpolate_sphere import linear_interpolate_sphere
from arealmshbm.graph_distance import gradient_geodesic_distance
from arealmshbm.diffusion_map import (
    compute_diffusion_map_repaired,
    compute_diffusion_map_gpu_repaired,
)

from .config import Step0Config


# MATLAB parity (``CBIG_SPGrad_RSFC_gradients.m:436``):
#     num_grads_across_scan_block = grads > 10^-10;
# i.e. only gradient magnitudes above 1e-10 contribute to the
# block-averaged "positive gradient" count. Kept as a named constant
# so future tuning is explicit, not hidden in a literal.
_POS_GRAD_THRESHOLD = 1e-10


# ─────────────────────────────────────────────────────────────────────
# Process-wide cache for the subject-invariant ``load_inputs`` output.
# Keyed on the fields of cfg that actually determine the result:
# mesh + cbig_code_dir + smooth_sigma + K_hop. A batch driver running
# K subjects with the same cfg.mesh pays the ~3 s load_inputs cost
# exactly once instead of K times. The cached value is the CPU portion
# of Step0Inputs (GPU gathers are NOT cached — ``Step0Pipeline.close()``
# calls ``cp.get_default_memory_pool().free_all_blocks()`` on the GPU
# backend, which would invalidate any device-resident cache anyway,
# and the GPU gather H2D itself is ~10 ms — cheap to rebuild).
#
# The cache is unbounded and grows linearly in the number of distinct
# (mesh, cbig_code_dir, smooth_sigma, K_hop) tuples seen by the
# process. The current single-mesh batch usage keeps it at one entry;
# test suites or pipelines that iterate meshes should call
# :func:`clear_inputs_cache` between configs.
_INPUTS_CACHE: Dict[Tuple, "Step0Inputs"] = {}


def _inputs_cache_key(cfg) -> Tuple:
    """Cache key for the subject-invariant portion of load_inputs."""
    cbig = str(cfg.cbig_code_dir) if cfg.cbig_code_dir is not None else None
    return (str(cfg.mesh), cbig, float(cfg.smooth_sigma), int(cfg.K_hop))


def clear_inputs_cache() -> None:
    """Drop the process-wide load_inputs cache.

    Useful in test suites and when switching meshes mid-process; not
    needed in a normal batch run. The cache also auto-evicts the CPU
    side memory only — GPU memory is governed by Step0Pipeline.close()
    teardown.
    """
    _INPUTS_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────
# GPU-mirror builder. Called twice (once on cache-hit, once on
# fresh-load) — factored out so the two paths share one source of truth
# for what the GPU mirrors are and how to build them.
# ─────────────────────────────────────────────────────────────────────
def _build_gpu_mirrors(cpu: "Step0Inputs", backend: str) -> Dict[str, Any]:
    """Build the 6 GPU-mirror fields from a CPU-only :class:`Step0Inputs`.

    Returns a dict keyed on the dataclass field names so callers can
    splat it into ``dataclasses.replace(cpu, **mirrors)``.

    On ``backend != 'gpu'`` every entry is ``None`` (the dataclass
    defaults, made explicit here so the call site stays uniform).
    On ``backend == 'gpu'`` each pair of (lh, rh) values is built once
    via the leaf-level ``prepare_*_gpu`` helpers — ~10 ms total H2D.
    """
    if backend != "gpu":
        return dict(
            lh_smooth_gather_gpu=None, rh_smooth_gather_gpu=None,
            lh_grad_verts_gpu=None, rh_grad_verts_gpu=None,
            lh_grad_mesh_gpu=None, rh_grad_mesh_gpu=None,
        )
    # Lazy imports keep cupy a soft dependency on the CPU backend.
    from arealmshbm.surface_smoothing.surface_smoothing_gpu import (
        prepare_smoothing_gather_gpu,
    )
    from arealmshbm.surface_gradient.surface_gradient_gpu import (
        prepare_gradient_mesh_gpu,
    )
    import cupy as cp
    return dict(
        lh_smooth_gather_gpu=prepare_smoothing_gather_gpu(cpu.lh_smooth_gather),
        rh_smooth_gather_gpu=prepare_smoothing_gather_gpu(cpu.rh_smooth_gather),
        lh_grad_verts_gpu=cp.asarray(cpu.lh_mid_verts, dtype=cp.float32),
        rh_grad_verts_gpu=cp.asarray(cpu.rh_mid_verts, dtype=cp.float32),
        lh_grad_mesh_gpu=prepare_gradient_mesh_gpu(cpu.lh_grad_prep),
        rh_grad_mesh_gpu=prepare_gradient_mesh_gpu(cpu.rh_grad_prep),
    )


# ─────────────────────────────────────────────────────────────────────
# Stage outputs
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Step0Inputs:
    """Subject-independent loaded inputs for a step-0 run.

    Keyed by ``(cfg.mesh, cfg.smooth_sigma, cfg.K_hop)``, so a
    multi-subject driver can hoist the load once. Cached load reads
    raw ``.npy`` files and needs no atlas; cold rebuild via
    :mod:`arealmshbm.precompute.step0_inputs_builder` reads a CBIG
    checkout (``cfg.cbig_code_dir`` / ``$CBIG_CODE_DIR``) for the
    midthickness atlas + FreeSurfer source meshes.
    """
    medial_mask: np.ndarray             # (N_full,) bool — lh first, rh second
    n_lh: int
    n_rh: int
    n_full: int                         # = n_lh + n_rh
    n_cortex: int                       # = sum(~medial_mask)
    # full-resolution sphere meshes (for subgraph B downsample interp)
    lh_sphere_verts: np.ndarray         # (N_lh, 3) fp32
    lh_sphere_faces: np.ndarray         # (Flh, 3) int32
    lh_sphere_vertex_faces: np.ndarray  # (N_lh, max_faces) int32 1-indexed
    rh_sphere_verts: np.ndarray
    rh_sphere_faces: np.ndarray
    rh_sphere_vertex_faces: np.ndarray
    # midthickness meshes (for surface_gradient + smoothing)
    lh_mid_verts: np.ndarray            # (N_lh, 3) fp32
    lh_mid_faces: np.ndarray            # (Flh, 3) int32
    rh_mid_verts: np.ndarray
    rh_mid_faces: np.ndarray
    # surface_smoothing mesh-only precomputation (per hemi). Each is a
    # 7-tuple (va, n1_indptr, n1_idx, n1_dist, n2_indptr, n2_idx,
    # n2_dist) — built once at load_inputs by prepare_smoothing_mesh.
    lh_smooth_prep: Tuple[np.ndarray, ...]
    rh_smooth_prep: Tuple[np.ndarray, ...]
    # surface_smoothing gather cache (per hemi). The pair
    # (gather_W: csr_matrix, inv_weight_sum: ndarray) depends only on
    # mesh + roi + cfg.smooth_sigma — all subject-invariant on a fixed
    # mesh — so we run the per-source Dijkstra exactly once at
    # load_inputs and reuse the CSR across every iter_a call.
    lh_smooth_gather: Any
    rh_smooth_gather: Any
    # surface_gradient mesh-only precomputation (per hemi). Each is a
    # 4-tuple (deg, nbors, vn, va) from prepare_gradient_mesh — also
    # subject-independent.
    lh_grad_prep: Tuple[np.ndarray, ...]
    rh_grad_prep: Tuple[np.ndarray, ...]
    # neighbor tables on the cortex-only space
    neighbors_table: np.ndarray         # (N_cortex, 7) fp64 (col 0 = self 1-idx; NaN pad)
    K_neighbors: np.ndarray             # (N_cortex, M) fp64
    # GPU mirror of the smoothing gather — populated only when
    # cfg.backend == 'gpu'. Each is a (cupyx.sparse.csr_matrix,
    # cp.ndarray) fp32 bundle, derived from the CPU gather at
    # load_inputs by ``prepare_smoothing_gather_gpu``. Cached so every
    # iter_a call lands a pure SpMM on cuSPARSE without re-uploading the
    # (N_full, ~few hundred-nnz/row) CSR. Default-None keeps the CPU
    # backend's construction call unchanged.
    lh_smooth_gather_gpu: Any = None
    rh_smooth_gather_gpu: Any = None
    # GPU mirror of the gradient mesh prep + midthickness verts —
    # populated only when cfg.backend == 'gpu'. The kernel reads
    # (verts, deg, nbors, vn, va) as device arrays for the per-vertex
    # tangent-plane LS solve; cached here so the iter_a × num_sess hot
    # loop pays only the per-call H2D of the (N_cortex, K) data slab.
    lh_grad_verts_gpu: Any = None
    rh_grad_verts_gpu: Any = None
    lh_grad_mesh_gpu: Any = None    # (deg_d, nbors_d, vn_d, va_d) tuple
    rh_grad_mesh_gpu: Any = None


@dataclass
class Step0Result:
    """Outputs of one ``Pipeline.run()``."""
    edge_density: np.ndarray                  # (N_cortex,) fp32
    # lh_dist / rh_dist are populated only when ``cfg.save_geodesic_distance``
    # is True (subgraph C's optional D2H of the 670 MB / hemi distance
    # matrix). Default-False production runs leave these as ``None`` —
    # the geodesic dist is a step-0 intermediate consumed in-memory by
    # subgraph C and has no production downstream reader on disk, so the
    # 1.34 GB / subject D2H is pure waste in that mode. See
    # :class:`Step0Config.save_geodesic_distance` for the gate.
    lh_dist: Optional[np.ndarray]             # (N_down, N_down) fp32 or None
    rh_dist: Optional[np.ndarray]             # (N_down, N_down) fp32 or None
    lh_emb_down: np.ndarray                   # (N_down, num_components) fp32
    rh_emb_down: np.ndarray                   # (N_down, num_components) fp32
    lh_emb_up: np.ndarray                     # (N_lh, num_components) fp32
    rh_emb_up: np.ndarray                     # (N_rh, num_components) fp32
    out_paths: Dict[str, Optional[Path]] = field(default_factory=dict)
    timings: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# BOLD session getter (cross-subject prefetch hook).
#
# Two implementations behind one ``sess_idx -> (N_cortex, T)`` contract:
#
#   * External — ``cfg.bold_provider`` set by the unified pipeline
#     driver. The driver owns one persistent ``ThreadPoolExecutor`` for
#     the full step0 phase and primes ``lookahead`` subjects ahead of
#     the current compute target. Decode for subject K+1 happens
#     concurrently with subject K's GPU compute, so ``bold_provider``
#     pulls return near-instantly.
#
#   * Internal — fallback for standalone ``Step0Pipeline`` usage
#     (ad-hoc single-subject runs, leaf tests). Spins up a per-subject
#     bounded sliding-window pool inside this contextmanager; behaviour
#     matches the prior in-pipeline ``ThreadPoolExecutor``. The
#     cold-start latency that the external path hides is unavoidable
#     here because there's no driver to manage cross-subject overlap.
#
# The 8-thread cap matches the isal_zlib saturation plateau on this
# hardware (~1.6 GB/s aggregate at nw=8; diminishing returns past).
# The GIFTI bytescan + per-DataArray base64 + zlib decompress all
# drop the GIL, so threads (not processes) give real parallelism.
#
# At ``num_sess <= n_io_workers + 1`` (true for YS at num_sess=6) the
# bounded-window structure degenerates to "submit all upfront, consume
# in order" — the post-consume refill never fires. The bounded shape
# is dormant insurance for batch / dense-session workloads (num_sess
# > 9), preserving steady-state peak ≤ ``prefetch_depth × 72 MB``.
# ─────────────────────────────────────────────────────────────────────
def _prefetch_session_bold(
    cfg: Step0Config, ti: "Step0Inputs", sess_idx: int,
) -> np.ndarray:
    """Worker body for the internal BOLD pool.

    Lifted to module scope (vs. the prior closure in ``run()``) so the
    same body can be reached from both the in-pipeline pool and (via
    ``_step0_bold_prefetcher._load_session_bold_cpu``) the driver
    pool — one source of truth for "what a session load is".

    Always CPU IO regardless of ``cfg.backend`` — same as the unified
    driver's Step0BoldPrefetcher: parallel CPU decode hides fully
    behind the GPU compute via GIL-released isal_zlib, and
    ``_subgraph_A``'s ``cp.asarray(curr_data)`` H2D is the right
    seam — cheap, fully overlapped.
    """
    lh_path, rh_path = cfg.bold_paths(sess_idx)
    # expected_n guards against a silent reshape mis-layout if a
    # future BOLD converter changes the (I, J, K) decomposition.
    lh_bold = read_surface_bold(lh_path, expected_n=ti.n_lh)
    rh_bold = read_surface_bold(rh_path, expected_n=ti.n_rh)
    return concat_hemis_drop_medial(lh_bold, rh_bold, ti.medial_mask)


@contextmanager
def _bold_session_getter(
    cfg: Step0Config, ti: "Step0Inputs",
) -> Iterator[Callable[[int], np.ndarray]]:
    """Yield a 1-indexed ``sess_idx -> (N_cortex, T) fp32`` getter.

    If ``cfg.bold_provider`` is non-None, yield it directly (driver
    owns the worker pool). Otherwise spin up a per-subject pool with
    the same bounded sliding-window semantics as the prior in-line
    code, and shut it down on exit.
    """
    if cfg.bold_provider is not None:
        yield cfg.bold_provider
        return

    n_io_workers = max(1, min(cfg.num_sess, 8))
    pool = ThreadPoolExecutor(
        max_workers=n_io_workers, thread_name_prefix="bold-io")
    try:
        prefetch_depth = min(n_io_workers + 1, cfg.num_sess)
        sess_futures: Dict[int, Any] = {}
        next_submit = [1]   # box so the closure can mutate
        # Prime the window.
        while (next_submit[0] <= cfg.num_sess
                 and len(sess_futures) < prefetch_depth):
            sess_futures[next_submit[0]] = pool.submit(
                _prefetch_session_bold, cfg, ti, next_submit[0])
            next_submit[0] += 1

        def _get(sess_idx: int) -> np.ndarray:
            result = sess_futures.pop(sess_idx).result()
            # Refill: now that we've freed a slot, submit the next
            # session (if any). Keeps prefetch depth constant
            # regardless of ``cfg.num_sess``.
            if next_submit[0] <= cfg.num_sess:
                sess_futures[next_submit[0]] = pool.submit(
                    _prefetch_session_bold, cfg, ti, next_submit[0])
                next_submit[0] += 1
            return result

        yield _get
    finally:
        pool.shutdown(wait=True)


def _new_per_stage_dict() -> Dict[str, float]:
    """Fresh per-subject timing accumulator dict — used by both ``run()``
    and the pipeline-parallel coordinator. Each subgraph adds to its
    own keys; never clears across subgraphs so the same dict can flow
    through the 4-stage pipeline.
    """
    return {
        "subgraph_A_total": 0.0,
        "bold_io_wait": 0.0,
        "fc_similarity": 0.0,
        "surface_gradient": 0.0,
        "surface_smoothing": 0.0,
        "local_minima": 0.0,
        "watershed": 0.0,
        "subgraph_B_total": 0.0,
        "subgraph_C_total": 0.0,
        "subgraph_D_total": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────
class Step0Pipeline:
    """Single-subject step-0 pipeline. Stateful — cache once, run once.

    Lifecycle:
      * ``__init__(cfg)``                       — store config; no IO.
      * ``load_inputs() -> Step0Inputs``         — read fsaverage6 meshes + neighbors.
      * ``run(inputs=None) -> Step0Result``      — execute subgraphs A→B→C→D.
      * ``save(result) -> dict[str, Path]``      — write .mat / .npy artifacts.
      * ``close()``                              — drop refs (no GPU pool here).

    Pipeline is also a context manager.
    """

    def __init__(self, cfg: Step0Config):
        self.cfg = cfg
        self._inputs: Optional[Step0Inputs] = None
        self.timings: Dict[str, Any] = {}

    # ---- context manager -------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        self._inputs = None
        # Release the CuPy memory pool in GPU mode so multi-subject
        # batch loops don't keep ~3 GB of fp64 affinity / Lanczos
        # workspace on the device between subjects. We catch + warn
        # rather than swallow so a real cupy regression (e.g. a driver
        # mismatch) is at least observable; teardown failure should not
        # mask the primary computation result.
        if self.cfg.backend == "gpu":
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                warnings.warn(
                    f"Step0Pipeline.close(): cupy teardown failed "
                    f"({type(e).__name__}: {e}); device memory may not be "
                    f"released for this subject.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # =================================================================
    # Stage 1: load inputs (subject-independent)
    # =================================================================
    def load_inputs(self) -> Step0Inputs:
        """Resolve the subject-invariant inputs in three tiers, fastest first.

        1. **Process cache** (``_INPUTS_CACHE``) — ~36 ms for batch hits.
        2. **On-disk .npy cache** — ~30-100 ms cold-start vs ~3.0 s for
           the live rebuild path. Keyed on
           ``(mesh, smooth_sigma, K_hop)``. Built by
           :mod:`arealmshbm.precompute.step0_inputs_builder`.
        3. **Live rebuild** (``build_step0_inputs``) — only on first
           cold start, after which the on-disk cache is written so
           subsequent processes hit tier 2.

        GPU mirrors are populated via :func:`_build_gpu_mirrors`
        regardless of which tier resolves the CPU-side blob.
        """
        cfg = self.cfg
        t0 = time.perf_counter()

        # ── Tier 1: process cache ──────────────────────────────────────
        key = _inputs_cache_key(cfg)
        cached = _INPUTS_CACHE.get(key)
        if cached is not None:
            inputs = replace(cached, **_build_gpu_mirrors(cached, cfg.backend))
            self.timings["load_inputs"] = time.perf_counter() - t0
            self._inputs = inputs
            return inputs

        # ── Tier 2: on-disk .npy cache ─────────────────────────────────
        from .inputs_cache import (
            cache_dir_for, cache_exists, load_cached_inputs, save_inputs,
        )
        cache_dir = cache_dir_for(cfg)
        if cache_exists(cache_dir):
            cached_cpu = load_cached_inputs(cache_dir)
        else:
            # ── Tier 3: live rebuild (only on first cold start) ────────
            # All heavy precomputation lives in
            # arealmshbm/precompute/step0_inputs_builder.py — moved out
            # of the production path. We call it here once on first run
            # and write the result to disk so tier 2 wins next time.
            from arealmshbm.precompute.step0_inputs_builder import (
                build_step0_inputs,
            )
            cached_cpu = build_step0_inputs(cfg)
            try:
                save_inputs(cached_cpu, cache_dir, cfg)
            except OSError as e:
                warnings.warn(
                    f"step0 inputs cache write failed at {cache_dir} "
                    f"({type(e).__name__}: {e}); will rebuild next process.",
                    RuntimeWarning, stacklevel=2)

        # Stash in process cache so any further Step0Pipeline in this
        # process hits tier 1.
        _INPUTS_CACHE[key] = cached_cpu

        self._inputs = replace(
            cached_cpu, **_build_gpu_mirrors(cached_cpu, cfg.backend))
        self.timings["load_inputs"] = time.perf_counter() - t0
        return self._inputs

    # =================================================================
    # Stage 2: run the four subgraphs
    # =================================================================
    def run(self, inputs: Optional[Step0Inputs] = None,
            *, time_stages: bool = True) -> Step0Result:
        """Sequential composition of the four subgraphs.

        Each subgraph is also exposed as ``_subgraph_{A,B,C,D}`` so the
        unified pipeline driver's stage-pipeline coordinator can call
        them on separate threads / streams. The driver's pipeline-
        parallel path produces bit-identical output to this sequential
        path (same arithmetic, different schedule).
        """
        cfg = self.cfg
        if inputs is None:
            inputs = self._inputs if self._inputs is not None else self.load_inputs()
        ti = inputs

        per_stage = _new_per_stage_dict()

        with _bold_session_getter(cfg, ti) as get_session_bold:
            a_out = self._subgraph_A(ti, get_session_bold, per_stage)

        b_out = self._subgraph_B(ti, a_out["edge_density"], per_stage)
        c_out = self._subgraph_C(b_out["lh_dist"], b_out["rh_dist"], per_stage)
        d_out = self._subgraph_D(
            ti,
            c_out["lh_emb_down"], c_out["rh_emb_down"],
            b_out["down_v"], b_out["down_f"], b_out["down_vf"],
            per_stage,
        )

        timings: Dict[str, Any] = {
            "load_inputs": self.timings.get("load_inputs", 0.0),
            "per_stage": per_stage if time_stages else {},
            "num_sample_S": a_out["num_sample_S"],
            "iter_a": a_out["iter_a"],
            "block_size_a": a_out["block_size_a"],
            "n_cortex": ti.n_cortex,
            "n_down_sphere": int(b_out["down_v"].shape[0]),
        }
        return Step0Result(
            edge_density=a_out["edge_density"],
            lh_dist=c_out["lh_dist_h"], rh_dist=c_out["rh_dist_h"],
            lh_emb_down=c_out["lh_emb_down"], rh_emb_down=c_out["rh_emb_down"],
            lh_emb_up=d_out["lh_emb_up"], rh_emb_up=d_out["rh_emb_up"],
            timings=timings,
        )

    # =================================================================
    # Subgraph stages — extracted as independently callable methods so
    # the pipeline-parallel coordinator can schedule them on separate
    # threads / streams. Each method takes its inputs explicitly and
    # returns a dict of outputs; ``ti`` (Step0Inputs) and ``self.cfg``
    # are read-only shared state.
    # =================================================================
    def _subgraph_A(
        self, ti: Step0Inputs,
        get_session_bold: Callable[[int], np.ndarray],
        per_stage: Dict[str, float],
    ) -> Dict[str, Any]:
        """RSFC gradients → ``edge_density``.

        Per-session: fc_similarity (cuBLAS-heavy) → surface_gradient
        (RawKernel) → per-iter_a accumulator. Per-block: average →
        smoothing (cuSPARSE SpMM) → find_minima (CPU numba) →
        watershed (RawKernel) → edge count.

        Returns dict with: edge_density (N_cortex,) fp32, iter_a,
        num_sample_S, block_size_a.
        """
        cfg = self.cfg
        num_sample_S = int(round(ti.n_cortex / cfg.sub_verts))
        iter_a = cfg.block_a + (1 if num_sample_S % cfg.block_a != 0 else 0)
        block_size_a = num_sample_S // cfg.block_a

        if cfg.backend == "gpu":
            from arealmshbm.fc_similarity.fc_similarity_gpu import (
                compute_t_series_gpu,
                compute_FC_simi_block_gpu,
            )
            import cupy as cp
        ta = time.perf_counter()
        # Per-block accumulators of (sum_grads, count_grads>_POS_GRAD_THRESHOLD).
        # On the GPU backend these slots hold cp.ndarray (device-stay
        # accumulator chain), not np.ndarray — widened type covers both.
        block_sums: List[Optional["Union[np.ndarray, cp.ndarray]"]] = [None] * iter_a
        block_counts: List[Optional["Union[np.ndarray, cp.ndarray]"]] = [None] * iter_a

        # ---- per-session loop ----
        # ``get_session_bold`` is supplied by the caller (either
        # ``run()`` via ``_bold_session_getter``, or the pipeline-
        # parallel coordinator via its own BOLD prefetcher).
        for sess_idx in range(1, cfg.num_sess + 1):
            t_io = time.perf_counter()
            curr_data = get_session_bold(sess_idx)
            per_stage["bold_io_wait"] += time.perf_counter() - t_io

            randinds_verts, randinds_FC = set_downsample_params(
                num_vertices=ti.n_cortex,
                sub_verts=cfg.sub_verts,
                sub_FC=cfg.sub_FC,
                scan_idx=sess_idx,
            )

            if cfg.backend == "gpu":
                t_fc = time.perf_counter()
                curr_data_d = cp.asarray(curr_data)
                t_series_d, mag_t_d = compute_t_series_gpu(
                    curr_data_d, randinds_FC)
                cp.cuda.get_current_stream().synchronize()
                per_stage["fc_similarity"] += time.perf_counter() - t_fc
            else:
                t_fc = time.perf_counter()
                t_series, mag_t = compute_t_series(curr_data, randinds_FC)
                per_stage["fc_similarity"] += time.perf_counter() - t_fc

            for i in range(iter_a):
                t_fc = time.perf_counter()
                if cfg.backend == "gpu":
                    FC_simi_block = compute_FC_simi_block_gpu(
                        curr_data_d=curr_data_d,
                        t_series_d=t_series_d,
                        mag_t_d=mag_t_d,
                        randinds_verts=randinds_verts,
                        block_a_index=i,
                        num_blocks_a=cfg.block_a,
                        num_blocks_b=cfg.block_b,
                    )
                else:
                    FC_simi_block = compute_FC_simi_block(
                        curr_data=curr_data,
                        t_series=t_series,
                        mag_t=mag_t,
                        randinds_verts=randinds_verts,
                        block_a_index=i,
                        num_blocks_a=cfg.block_a,
                        num_blocks_b=cfg.block_b,
                    )
                per_stage["fc_similarity"] += time.perf_counter() - t_fc

                t_g = time.perf_counter()
                if cfg.backend == "gpu":
                    from arealmshbm.surface_gradient.surface_gradient_gpu import (
                        cifti_gradient_gpu,
                    )
                    grads = cifti_gradient_gpu(
                        FC_simi_block,
                        medial_mask=ti.medial_mask,
                        lh_verts_d=ti.lh_grad_verts_gpu,
                        lh_mesh_gpu=ti.lh_grad_mesh_gpu,
                        rh_verts_d=ti.rh_grad_verts_gpu,
                        rh_mesh_gpu=ti.rh_grad_mesh_gpu,
                    )
                else:
                    lh_d, lh_nb, lh_vn, lh_va = ti.lh_grad_prep
                    rh_d, rh_nb, rh_vn, rh_va = ti.rh_grad_prep
                    grads = cifti_gradient(
                        FC_simi_block,
                        ti.lh_mid_verts, lh_d, lh_nb, lh_vn, lh_va,
                        ti.rh_mid_verts, rh_d, rh_nb, rh_vn, rh_va,
                        ti.medial_mask,
                    )
                per_stage["surface_gradient"] += time.perf_counter() - t_g

                # Device-stay accumulator (PR #56-pattern refactor for
                # subgraph A): on GPU ``grads`` is cp.ndarray, so the
                # comparison + add stay on device. block_sums/counts
                # slots are cp.ndarray throughout the iter_a × num_sess
                # loop — no per-iter D2H/H2D ping-pong.
                pos = grads > _POS_GRAD_THRESHOLD
                if block_sums[i] is None:
                    block_sums[i] = grads.copy()
                    block_counts[i] = pos.astype(np.int32)
                else:
                    block_sums[i] += grads
                    block_counts[i] += pos

            if cfg.backend == "gpu":
                del curr_data_d, t_series_d, mag_t_d
            del curr_data

        # ---- per-block: avg → smooth → minima → watershed ----
        edge_count = np.zeros(ti.n_cortex, dtype=np.int64)
        neighbors_int_d = None
        K_neighbors_int_d = None
        if cfg.backend == "gpu":
            from arealmshbm.watershed import prepare_neighbors_for_gpu
            from arealmshbm.local_minima.local_minima_gpu import (
                find_minima_gpu, prepare_K_neighbors_for_gpu,
            )
            neighbors_int_d = prepare_neighbors_for_gpu(ti.neighbors_table)
            # Same hoist pattern as watershed's neighbors prep: K_neighbors
            # is subject-invariant on a fixed mesh, so we pay the int32
            # encode + H2D once across all iter_a in this subject.
            K_neighbors_int_d = prepare_K_neighbors_for_gpu(ti.K_neighbors)
        for i in range(iter_a):
            if cfg.backend == "gpu":
                # ``cp`` already in local scope from the gpu-branch
                # import at function head (line ~567).
                # Device-stay avg: block_counts/sums are cp.ndarray
                # already (see device-stay accumulator above). cp.where
                # produces a cp.ndarray; no NEP-18 dispatch ambiguity,
                # no host materialisation.
                avg_grads = cp.where(
                    block_counts[i] > 0,
                    block_sums[i] / block_counts[i].astype(cp.float32),
                    0.0,
                ).astype(cp.float32)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    avg_grads = np.where(
                        block_counts[i] > 0,
                        block_sums[i] / block_counts[i].astype(np.float32),
                        0.0,
                    ).astype(np.float32)

            t_s = time.perf_counter()
            if cfg.backend == "gpu":
                from arealmshbm.surface_smoothing.surface_smoothing_gpu import (
                    cifti_smoothing_gpu,
                )
                smoothed = cifti_smoothing_gpu(
                    avg_grads,
                    ti.medial_mask,
                    lh_gather_gpu=ti.lh_smooth_gather_gpu,
                    rh_gather_gpu=ti.rh_smooth_gather_gpu,
                )
            else:
                smoothed = cifti_smoothing(
                    avg_grads,
                    ti.medial_mask,
                    lh_gather=ti.lh_smooth_gather,
                    rh_gather=ti.rh_smooth_gather,
                )
            per_stage["surface_smoothing"] += time.perf_counter() - t_s

            t_m = time.perf_counter()
            if cfg.backend == "gpu":
                # find_minima_gpu consumes the device-resident ``smoothed``
                # directly — no D2H/H2D pair around the CPU find_minima.
                minima = find_minima_gpu(smoothed, K_neighbors_int_d)
            else:
                minima = find_minima(smoothed, ti.K_neighbors)
            per_stage["local_minima"] += time.perf_counter() - t_m

            t_w = time.perf_counter()
            if cfg.backend == "gpu":
                from arealmshbm.watershed import watershed_edge_count_gpu
                # smoothed + minima are already device-resident — pass
                # them through; watershed_edge_count_gpu's entry H2D
                # becomes a no-op alias on cp.ndarray fp32.
                ec_local = watershed_edge_count_gpu(
                    edge_metrics=smoothed,
                    minima=minima,
                    neighbors=None,
                    stepnum=cfg.watershed_steps,
                    fracmaxh=cfg.watershed_frac,
                    neighbors_int_d=neighbors_int_d,
                )
                per_stage["watershed"] += time.perf_counter() - t_w
                edge_count += ec_local
            else:
                edges = watershed_algorithm_repaired(
                    edge_metrics=smoothed,
                    minima=minima,
                    neighbors=ti.neighbors_table,
                    stepnum=cfg.watershed_steps,
                    fracmaxh=cfg.watershed_frac,
                )
                per_stage["watershed"] += time.perf_counter() - t_w
                edge_count += (edges == 0).sum(axis=1).astype(np.int64)

        edge_density = (edge_count.astype(np.float32) / float(num_sample_S))
        per_stage["subgraph_A_total"] = time.perf_counter() - ta
        return {
            "edge_density": edge_density,
            "iter_a": iter_a,
            "num_sample_S": num_sample_S,
            "block_size_a": block_size_a,
        }

    def _subgraph_B(
        self, ti: Step0Inputs,
        edge_density: np.ndarray,
        per_stage: Dict[str, float],
    ) -> Dict[str, Any]:
        """Per-hemi graph geodesic distance on the downsampled sphere.

        On GPU, ``lh_dist`` / ``rh_dist`` stay device-resident so
        :meth:`_subgraph_C` can consume them without an extra round
        trip; the D2H to numpy happens at the end of subgraph C.
        """
        cfg = self.cfg
        tb = time.perf_counter()
        # Scatter cortex-only edge_density to full mesh (medial = 0).
        full_grad_edge = np.zeros(ti.n_full, dtype=np.float32)
        full_grad_edge[~ti.medial_mask] = edge_density
        lh_grad_edge_full = full_grad_edge[:ti.n_lh]
        rh_grad_edge_full = full_grad_edge[ti.n_lh:]

        # Downsample sphere via icosphere. Both hemis share the same
        # icosphere triangulation; only per-vertex *data values* differ
        # downstream — so verts/faces/incidence are aliased, not copied.
        n_target = int(round(ti.n_full / (2.0 * cfg.downsample)))
        down_v, down_f, down_vf, down_vn = self._build_down_sphere(n_target)

        # Per-hemi: interpolate edge density to the down sphere.
        lh_grad_edge_down = linear_interpolate_sphere(
            target_points=down_v,
            source_verts=ti.lh_sphere_verts,
            source_faces=ti.lh_sphere_faces,
            source_vertex_faces=ti.lh_sphere_vertex_faces,
            data=lh_grad_edge_full,
        ).astype(np.float32)
        rh_grad_edge_down = linear_interpolate_sphere(
            target_points=down_v,
            source_verts=ti.rh_sphere_verts,
            source_faces=ti.rh_sphere_faces,
            source_vertex_faces=ti.rh_sphere_vertex_faces,
            data=rh_grad_edge_full,
        ).astype(np.float32)

        if cfg.backend == "gpu":
            from arealmshbm.graph_distance import (
                gradient_geodesic_distance_gpu_device,
            )
            import cupy as cp
            # vertex_nbors layout is identical for lh and rh on a shared
            # icosphere triangulation — upload once, reuse for both.
            vn_d = cp.asarray(down_vn, dtype=cp.int32)
            lh_grad_d = cp.asarray(
                lh_grad_edge_down.ravel(), dtype=cp.float32)
            lh_dist = gradient_geodesic_distance_gpu_device(vn_d, lh_grad_d)
            rh_grad_d = cp.asarray(
                rh_grad_edge_down.ravel(), dtype=cp.float32)
            rh_dist = gradient_geodesic_distance_gpu_device(vn_d, rh_grad_d)
            del vn_d, lh_grad_d, rh_grad_d
        else:
            lh_dist = gradient_geodesic_distance(
                verts=down_v,
                vertex_nbors=down_vn,
                grad_data=lh_grad_edge_down.ravel(),
            )
            rh_dist = gradient_geodesic_distance(
                verts=down_v,
                vertex_nbors=down_vn,
                grad_data=rh_grad_edge_down.ravel(),
            )
        per_stage["subgraph_B_total"] = time.perf_counter() - tb
        return {
            "lh_dist": lh_dist,
            "rh_dist": rh_dist,
            "down_v": down_v,
            "down_f": down_f,
            "down_vf": down_vf,
            "down_vn": down_vn,
        }

    def _subgraph_C(
        self,
        lh_dist: Any, rh_dist: Any,
        per_stage: Dict[str, float],
    ) -> Dict[str, Any]:
        """Diffusion embedding (per hemi) — wraps mapalign's Lanczos.

        On GPU, ``lh_dist`` / ``rh_dist`` arrive as cupy ndarrays;
        diffmap runs eigsh on cuSOLVER. The embeddings are always
        materialised to host (consumed by subgraph D); the distance
        matrices are D2H'd **only when ``cfg.save_geodesic_distance``
        is True** — that flag is False by default since 2026-05 because
        no production downstream step reads the on-disk dist artifact,
        so the legacy unconditional D2H of (N_down, N_down) fp32 ×2
        hemi (~1.34 GB / subject) was pure waste in the default mode.
        """
        cfg = self.cfg
        tc = time.perf_counter()
        diffmap = (compute_diffusion_map_gpu_repaired
                   if cfg.backend == "gpu" else compute_diffusion_map_repaired)
        # Gate the dist preservation + D2H on save_geodesic_distance.
        # When False (production default), we let diffmap consume the
        # dist buffers in place: skips both the 670 MB / hemi device
        # .copy() AND the matching D2H downstream — see Step0Result
        # docstring for the contract.
        need_dist_host = bool(cfg.save_geodesic_distance)
        if cfg.backend == "gpu":
            # diffmap rewrites its ``dist`` argument in place to
            # ``exp(-D / D.max())``. When we need the original
            # distance later (for the D2H below), .copy() first;
            # otherwise hand the buffer over and skip the copy.
            lh_in = lh_dist.copy() if need_dist_host else lh_dist
            rh_in = rh_dist.copy() if need_dist_host else rh_dist
            lh_emb_down = diffmap(
                lh_in, alpha=0.5, n_components=cfg.num_components,
            ).astype(np.float32)
            rh_emb_down = diffmap(
                rh_in, alpha=0.5, n_components=cfg.num_components,
            ).astype(np.float32)
        else:
            # CPU diffmap does NOT need a matching ``.copy()`` — its
            # ``_distance_to_affinity`` opens with
            # ``D = np.asarray(dist, dtype=np.float64)`` which forces a
            # fresh fp64 allocation (the input dist is fp32 from
            # gradient_geodesic_distance, so the dtype change copies).
            # Subsequent ``np.exp(-D / m)`` allocates again. The original
            # fp32 ``lh_dist`` / ``rh_dist`` are read-only on this path,
            # so the host D2H ``need_dist_host`` branch below sees the
            # untouched distances.
            lh_emb_down = diffmap(
                lh_dist, alpha=0.5, n_components=cfg.num_components,
            ).astype(np.float32)
            rh_emb_down = diffmap(
                rh_dist, alpha=0.5, n_components=cfg.num_components,
            ).astype(np.float32)
        if need_dist_host:
            if cfg.backend == "gpu":
                import cupy as cp
                lh_dist_h = cp.asnumpy(lh_dist)
                rh_dist_h = cp.asnumpy(rh_dist)
            else:
                lh_dist_h = lh_dist
                rh_dist_h = rh_dist
        else:
            # Skip 1.34 GB / subject of pointless D2H — Step0Result
            # carries None and save() respects save_geodesic_distance=False
            # to skip the dump.
            lh_dist_h = None
            rh_dist_h = None
        per_stage["subgraph_C_total"] = time.perf_counter() - tc
        return {
            "lh_dist_h": lh_dist_h,
            "rh_dist_h": rh_dist_h,
            "lh_emb_down": lh_emb_down,
            "rh_emb_down": rh_emb_down,
        }

    def _subgraph_D(
        self, ti: Step0Inputs,
        lh_emb_down: np.ndarray, rh_emb_down: np.ndarray,
        down_v: np.ndarray, down_f: np.ndarray, down_vf: np.ndarray,
        per_stage: Dict[str, float],
    ) -> Dict[str, Any]:
        """Upsample the down-sphere embedding back to full fsaverage.

        Pure CPU (numba ``linear_interpolate_sphere``); runs
        concurrently with later subjects' GPU stages under the
        pipeline-parallel coordinator.
        """
        td = time.perf_counter()
        lh_emb_up = linear_interpolate_sphere(
            target_points=ti.lh_sphere_verts,
            source_verts=down_v,
            source_faces=down_f,
            source_vertex_faces=down_vf,
            data=lh_emb_down.T,
        ).T.astype(np.float32)
        rh_emb_up = linear_interpolate_sphere(
            target_points=ti.rh_sphere_verts,
            source_verts=down_v,
            source_faces=down_f,
            source_vertex_faces=down_vf,
            data=rh_emb_down.T,
        ).T.astype(np.float32)

        # Zero medial verts (matches MATLAB).
        lh_emb_up[ti.medial_mask[:ti.n_lh], :] = 0
        rh_emb_up[ti.medial_mask[ti.n_lh:], :] = 0
        per_stage["subgraph_D_total"] = time.perf_counter() - td
        return {
            "lh_emb_up": lh_emb_up,
            "rh_emb_up": rh_emb_up,
        }

    # =================================================================
    # Stage 3: persist
    # =================================================================
    def save(self, result: Step0Result,
             *, out_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Write the artifacts step-3 reads.

        Outputs (under ``out_dir`` or ``cfg.gradients_out_dir``):
            edge_density.npy                                 (N_cortex,) fp32 — opt-in
            {lh,rh}_gradient_distance_matrix.npy             (N_down, N_down) fp32  — opt-in
            {lh,rh}_emb_<num_comp>_distance_matrix.npy       (N_full, num_comp) fp32
            {lh,rh}_emb_<num_comp>_distance_matrix.mat       (N_full, num_comp) fp32 — key 'emb'

        Which emb format(s) are written is controlled by
        ``cfg.emb_output_format`` (``'npy'`` default, ``'mat'`` legacy,
        ``'both'`` for validation).

        The ``gradient_distance_matrix.npy`` pair is gated by
        ``cfg.save_geodesic_distance`` (default False) — these are
        step0 intermediate outputs (~672 MB per hemi per subject) that
        no production downstream step reads. Skipping the dump removes
        ~53.6 GB of writes on a 40-subject fsa6 run and cuts step0
        wall noticeably (the writes were physically flushing for tens
        of seconds after step0 finished, on top of the actual step0
        compute). Toggle on for ad-hoc cohort-comparison archaeology
        (via the internal step-0 CPU/GPU diff harness).
        """
        out_dir = Path(out_dir) if out_dir is not None else self.cfg.gradients_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        nc = self.cfg.num_components

        paths: Dict[str, Path] = {}
        # edge_density.npy is a step-0 intermediate with no production
        # downstream consumer (subgraph B reads it in-memory; the disk
        # artifact is consumed only by the internal CPU/GPU diff harness).
        # Gated by ``save_edge_density`` (default False since 2026-06) —
        # symmetric to ``save_geodesic_distance`` which gates the other
        # intermediate dump.
        if self.cfg.save_edge_density:
            np.save(out_dir / "edge_density.npy", result.edge_density)
            paths["edge_density"] = out_dir / "edge_density.npy"
        if self.cfg.save_geodesic_distance:
            np.save(out_dir / "lh_gradient_distance_matrix.npy", result.lh_dist)
            paths["lh_dist"] = out_dir / "lh_gradient_distance_matrix.npy"
            np.save(out_dir / "rh_gradient_distance_matrix.npy", result.rh_dist)
            paths["rh_dist"] = out_dir / "rh_gradient_distance_matrix.npy"

        emb_fmt = self.cfg.emb_output_format
        if emb_fmt in ("npy", "both"):
            lh_npy = out_dir / f"lh_emb_{nc}_distance_matrix.npy"
            rh_npy = out_dir / f"rh_emb_{nc}_distance_matrix.npy"
            np.save(lh_npy, result.lh_emb_up, allow_pickle=False)
            np.save(rh_npy, result.rh_emb_up, allow_pickle=False)
            paths["lh_emb_npy"] = lh_npy
            paths["rh_emb_npy"] = rh_npy
            # Stable aliases — ``lh_emb`` / ``rh_emb`` always point at the
            # primary output (the format the new readers prefer).
            paths.setdefault("lh_emb", lh_npy)
            paths.setdefault("rh_emb", rh_npy)
        if emb_fmt in ("mat", "both"):
            from scipy.io import savemat
            lh_mat = out_dir / f"lh_emb_{nc}_distance_matrix.mat"
            rh_mat = out_dir / f"rh_emb_{nc}_distance_matrix.mat"
            savemat(lh_mat, {"emb": result.lh_emb_up}, do_compression=False)
            savemat(rh_mat, {"emb": result.rh_emb_up}, do_compression=False)
            paths["lh_emb_mat"] = lh_mat
            paths["rh_emb_mat"] = rh_mat
            if emb_fmt == "mat":
                paths["lh_emb"] = lh_mat
                paths["rh_emb"] = rh_mat
        result.out_paths = paths
        return paths

    def run_and_save(self) -> Step0Result:
        result = self.run()
        if self.cfg.save_artifacts:
            self.save(result)
        return result

    # =================================================================
    # Internals
    # =================================================================
    def _build_down_sphere(self, n_target: int):
        """Build one downsampled icosphere + its topology.

        Both hemispheres share this triangulation — only per-vertex
        data values differ downstream — so the caller aliases the
        same arrays into the lh and rh slots.

        Returns
        -------
        (verts, faces, vertex_faces, vertex_nbors)
        """
        from arealmshbm.mesh_topology import compute_topology
        v, f = make_icosphere(n_target, radius=100.0)
        vn, vf = compute_topology(v, f)
        return v, f, vf, vn
