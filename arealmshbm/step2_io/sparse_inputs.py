"""sparse_inputs.py — inputs of the step-2 ``gpu`` backend.

Contract: ``docs/step2_sparse_design.md`` §5. The dataclass below is the
hand-off between the loader (``load_step2_sparse_inputs``) and the device
Session (``step2_em_iter_master.session_gpu.Step2SparseSession``);
``Step2SparseInputs.from_arrays`` builds the same object from in-memory
arrays for tests and benches.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .sparse_layout import Step2Layout, build_step2_layout


# Type of the per-subject fill callbacks. ``s_1`` is the 1-based subject
# index (positional in cohort.json); ``out`` is a caller-owned C-contiguous
# buffer the callback must fill completely. The Session may call a
# BoldReader from up to ``_INGEST_DEPTH`` decode-pool worker threads at
# once, always for distinct subjects and distinct ``out`` buffers, so a
# BoldReader must hold no mutable state shared between calls. It calls a
# GradReader serially, on the thread that constructs the Session.
BoldReader = Callable[[int, np.ndarray], None]      # out: (T, N, Db) uint8
GradReader = Callable[[int, np.ndarray], None]      # out: (N, D_grad) fp32


@dataclass
class Step2SparseInputs:
    """Everything the sparse Session needs, host-side.

    Attributes
    ----------
    layout      : static P-layout (``P = nnz(boundary_mask)``).
    S, T, N, D, D_grad, n_lh, n_rh : dims. ``D`` is the profile length
                  (``mtc.shape[0]``, 1175 at fsaverage3 seeds); ``dim = D - 1``
                  is the vMF dimension fed to ``Cdln`` / ``invAd``.
    mtc         : (D, L) fp64 — ``group.mat['mtc']`` verbatim.
    bold_reader : fills ``(T, N, Db)`` uint8 for subject ``s_1`` — the on-disk
                  bit-packed layout (LSB-first along D, padding bits zero).
                  One ``.b2nd`` decode per call, decompressed straight into
                  the caller's buffer (~47 ms at fsaverage6 / T=6); where
                  the subjects then live — the device cache, or a host
                  cache in stream mode — is the Session's decision.
    grad_reader : fills ``(N, D_grad)`` fp32 for subject ``s_1`` (gMSHBM);
                  ``None`` for dMSHBM.
    timings     : per-item load timings (seconds).
    """

    layout: Step2Layout
    S: int
    T: int
    N: int
    D: int
    D_grad: int
    n_lh: int
    n_rh: int
    mtc: np.ndarray
    dim: int
    bold_reader: BoldReader
    grad_reader: Optional[GradReader] = None
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def D_bytes(self) -> int:
        """``ceil(D / 8)`` — the packed profile's byte width."""
        return (int(self.D) + 7) // 8

    @property
    def L(self) -> int:
        """Parcel count, taken from the layout."""
        return int(self.layout.L)

    def __post_init__(self) -> None:
        if self.layout.N != self.N:
            raise ValueError(f"layout.N={self.layout.N} != N={self.N}")
        if self.n_lh + self.n_rh != self.N or self.layout.n_lh != self.n_lh:
            raise ValueError("n_lh / n_rh inconsistent with N / layout")
        mtc = np.asarray(self.mtc)
        if mtc.ndim != 2 or mtc.shape[0] != self.D or mtc.shape[1] != self.layout.L:
            raise ValueError(
                f"mtc shape {mtc.shape} != (D={self.D}, L={self.layout.L})")
        if self.dim != self.D - 1:
            raise ValueError(f"dim={self.dim} must equal D-1={self.D - 1}")

    # ── in-memory constructor (tests / benches) ──
    @classmethod
    def from_arrays(
        cls,
        layout: Step2Layout,
        packed_STNDb: np.ndarray,
        mtc: np.ndarray,
        *,
        D: Optional[int] = None,
        grad_SNDg: Optional[np.ndarray] = None,
    ) -> "Step2SparseInputs":
        """Build inputs from a ``(S, T, N, Db)`` uint8 packed stack (+ optional
        ``(S, N, D_grad)`` fp32 gradients). ``D`` defaults to ``mtc.shape[0]``.
        """
        pk = np.ascontiguousarray(packed_STNDb, dtype=np.uint8)
        if pk.ndim != 4:
            raise ValueError("packed_STNDb must be (S, T, N, Db)")
        S, T, N, Db = (int(x) for x in pk.shape)
        mtc = np.ascontiguousarray(mtc, dtype=np.float64)
        D = int(mtc.shape[0] if D is None else D)
        if (D + 7) // 8 != Db:
            raise ValueError(f"ceil(D/8)={(D + 7) // 8} != Db={Db}")

        def _bold(s_1: int, out: np.ndarray) -> None:
            np.copyto(out, pk[s_1 - 1])

        grad_reader: Optional[GradReader] = None
        D_grad = 0
        if grad_SNDg is not None:
            g = np.ascontiguousarray(grad_SNDg, dtype=np.float32)
            if g.ndim != 3 or g.shape[0] != S or g.shape[1] != N:
                raise ValueError("grad_SNDg must be (S, N, D_grad)")
            D_grad = int(g.shape[2])

            def _grad(s_1: int, out: np.ndarray) -> None:
                np.copyto(out, g[s_1 - 1])

            grad_reader = _grad

        return cls(
            layout=layout, S=S, T=T, N=N, D=D, D_grad=D_grad,
            n_lh=layout.n_lh, n_rh=N - layout.n_lh, mtc=mtc, dim=D - 1,
            bold_reader=_bold, grad_reader=grad_reader,
        )


# ─────────────────────────────────────────────────────────────────────
# Disk loader
# ─────────────────────────────────────────────────────────────────────
#
# The same reads ``Step2Pipeline.load_inputs`` does (cohort, mesh,
# profiles, gradients, group.mat, spatial mask) with three differences:
#
#   * the spatial mask never becomes a dense (N, L) fp32 array — it goes
#     straight from MATLAB's ir/jc/pr into ``Step2Layout``;
#   * ``group.mat`` is read through ``mat5_stream`` so the 22 MB
#     ``lambda`` field is skipped by element size;
#   * the BOLD is handed to the Session as a per-subject packed reader:
#     one ``.b2nd`` decode per call, decompressed by blosc2 straight
#     into the caller's buffer (``decode_subject_profile_packed_into``,
#     no intermediate array). The Session decides where the subjects
#     live (its device cache, or a host cache in stream mode) and pulls
#     them through the reader from its constructor — there is no
#     ``(N, T, D)`` fp32 slab anywhere on this path.
#
# The four disk reads (BOLD header, gradient peek, group.mat, mask)
# overlap on a 4-worker pool: the inflate releases the GIL.


def _mars_masks(mesh: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(lh_mars, rh_mars)`` bool arrays — True at medial-wall vertices."""
    from arealmshbm.data_io.load_avg_mesh import load_avg_mesh

    lh_inf = load_avg_mesh("lh", mesh, "inflated")
    rh_inf = load_avg_mesh("rh", mesh, "inflated")
    lh = np.asarray(lh_inf["MARS_label"]).ravel() == 1
    rh = np.asarray(rh_inf["MARS_label"]).ravel() == 1
    return lh, rh


def _verify_mw_zero(packed_ND: np.ndarray, lh_mars: np.ndarray,
                    rh_mars: np.ndarray, path: Any) -> None:
    """Assert every packed byte at a medial-wall vertex is zero.

    The same one-shot contract check ``SubjectProfileLoader.__init__``
    runs: step-1's binarize is supposed to write zero MW rows, and the
    sparse backend's row statistics (design §1.1) rely on it — a nonzero
    MW row would silently become a live vertex.
    """
    n_lh = int(lh_mars.size)
    lh_rows = packed_ND[:n_lh][lh_mars]
    rh_rows = packed_ND[n_lh:][rh_mars]
    if (lh_rows.size and np.any(lh_rows != 0)) or \
       (rh_rows.size and np.any(rh_rows != 0)):
        raise ValueError(
            f"load_step2_sparse_inputs: .b2nd at {path} has non-zero "
            f"values at MW vertices in sub1/sess1. step1's binarize "
            f"stage should produce zero MW rows; this dataset violates "
            f"that contract."
        )


def _check_subject_header(view, shape: Tuple[int, int, int], D: int,
                          path: Path, s_1: int) -> None:
    """Subject ``s_1``'s ``.b2nd`` header must match subject 1's.

    Two checks, on every subject, on every ``bold_reader`` call -- so
    they raise from the Session's constructor or a stream-mode visit,
    not from the loader (the CPU loader,
    ``subject_loaders.SubjectProfileLoader``, runs the same pair):

    * packed shape ``== (T, N, ceil(D/8))``. ``np.copyto`` broadcasts, so
      a subject holding a single session would otherwise be replicated
      into all ``T`` slots instead of raising;
    * ``D_unpacked == D``, which the shape does not pin: 1174 and 1175
      share ``Db = 147``, so a cohort mixing the two would be normalised
      and unpacked with subject 1's ``D``.
    """
    got = tuple(int(x) for x in view.shape)
    want = tuple(int(x) for x in shape)
    if got != want:
        raise ValueError(
            f"bold_reader: sub {s_1} packed shape {got} != "
            f"expected (T, N, Db)={want} ({path})")
    D_disk = int(view.D_unpacked)
    if D_disk != D:
        raise ValueError(
            f"bold_reader: sub {s_1} on-disk D_unpacked="
            f"{D_disk} != D={D} from sub 1 ({path})")


def load_step2_sparse_inputs(cfg: Any, *,
                             overlap: bool = True) -> Step2SparseInputs:
    """Disk loader for the step-2 ``gpu`` backend.

    Reads ``cohort.json``, subject 1's bit-packed ``.b2nd`` header and
    one chunk of it, subject 1's gradient dims (gMSHBM),
    ``group/group.mat`` and ``spatial_mask/spatial_mask_<mesh>.mat``,
    and returns the :class:`Step2SparseInputs` the sparse Session
    consumes, whose ``bold_reader`` / ``grad_reader`` read each subject
    when the Session pulls it. See ``docs/step2_sparse_design.md`` §5.

    Parameters
    ----------
    cfg : :class:`~arealmshbm.step2_pipeline.config.Step2Config` (or any
        object carrying ``project_dir``, ``num_sub``, ``num_session``,
        ``num_clusters``, ``mode``, ``mesh``, ``seed_mesh``,
        ``n_grad_components``).
    overlap : run the four disk reads on a 4-worker pool. ``False``
        serialises them, which is what the per-item timings mean when
        you want to attribute cost.

    Notes
    -----
    Only subject 1's header (dims, ``D_unpacked``) and one chunk of it
    (the medial-wall contract) are read here; every subject's bytes are
    decoded through ``bold_reader`` when the Session pulls them.
    """
    from arealmshbm.data_io.cohort import read_cohort, resolve_path

    from .subject_loaders import SubjectGradientLoader

    t_all = time.perf_counter()
    timings: Dict[str, float] = {}
    project_dir = Path(cfg.project_dir)
    S = int(cfg.num_sub)
    T = int(cfg.num_session)
    L_cfg = int(cfg.num_clusters)
    mode = str(cfg.mode)

    # ── cohort.json — the sole discovery mechanism ──
    t0 = time.perf_counter()
    cohort = read_cohort(project_dir)
    if cohort.num_sub != S:
        raise ValueError(
            f"load_step2_sparse_inputs: cfg.num_sub={S} != "
            f"cohort.json num_sub={cohort.num_sub}")
    if cohort.num_session != T:
        raise ValueError(
            f"load_step2_sparse_inputs: cfg.num_session={T} != "
            f"cohort.json num_session={cohort.num_session}")
    if cohort.mesh.get("targ") != cfg.mesh:
        raise ValueError(
            f"load_step2_sparse_inputs: cfg.mesh={cfg.mesh!r} != "
            f"cohort.json mesh.targ={cohort.mesh.get('targ')!r}")
    if cohort.mesh.get("seed") != cfg.seed_mesh:
        raise ValueError(
            f"load_step2_sparse_inputs: cfg.seed_mesh={cfg.seed_mesh!r} != "
            f"cohort.json mesh.seed={cohort.mesh.get('seed')!r}")

    grad_loader: Optional[SubjectGradientLoader] = None
    if mode == "gMSHBM":
        grad_loader = SubjectGradientLoader(
            cohort=cohort, project_dir=project_dir,
            n_components=int(cfg.n_grad_components))

    b2nd_paths: List[Path] = []
    for i, sub in enumerate(cohort.subjects):
        if sub.profile_b2nd is None:
            raise ValueError(
                f"load_step2_sparse_inputs: cohort.subjects[{i}] "
                f"(id={sub.id!r}) has no profile_b2nd path. Run step1's "
                f"generate_profiles subgraph for this subject.")
        p = resolve_path(project_dir, sub.profile_b2nd)
        if not p.exists():
            raise FileNotFoundError(
                f"load_step2_sparse_inputs: cohort.subjects[{i}] "
                f"(id={sub.id!r}) profile_b2nd not found on disk: {p}")
        b2nd_paths.append(p)
    timings["cohort"] = time.perf_counter() - t0

    # ── mesh (MARS labels) — lru_cached .npz, effectively free warm ──
    t0 = time.perf_counter()
    lh_mars, rh_mars = _mars_masks(str(cfg.mesh))
    n_lh, n_rh = int(lh_mars.size), int(rh_mars.size)
    N = n_lh + n_rh
    timings["mesh"] = time.perf_counter() - t0

    # ── the four disk reads ──
    state: Dict[str, Any] = {}

    def _task_bold() -> None:
        from arealmshbm.data_io.profile_io import (
            open_subject_profile_packed_tnd,
        )
        t = time.perf_counter()
        a = open_subject_profile_packed_tnd(b2nd_paths[0])
        try:
            # Chunk header + vlmeta only — no cell decode.
            T_disk, N_disk, Db = (int(x) for x in a.shape)
            D = int(a.D_unpacked)
            if T_disk != T:
                raise ValueError(
                    f"load_step2_sparse_inputs: .b2nd at {b2nd_paths[0]} "
                    f"has T={T_disk} but cfg.num_session={T}")
            if N_disk != N:
                raise ValueError(
                    f"load_step2_sparse_inputs: .b2nd N={N_disk} != "
                    f"n_lh+n_rh={N}")
            if (D + 7) // 8 != Db:
                raise ValueError(
                    f"load_step2_sparse_inputs: vlmeta D_unpacked={D} "
                    f"disagrees with the packed width Db={Db}")
            # MW contract: one chunk decode of subject 1 / session 1.
            _verify_mw_zero(np.asarray(a[0]), lh_mars, rh_mars,
                            b2nd_paths[0])
        finally:
            del a
        state["D"], state["Db"] = D, Db
        timings["profiles"] = time.perf_counter() - t

    def _task_grad() -> None:
        t = time.perf_counter()
        if grad_loader is None:
            state["D_grad"] = 0
            timings["gradients"] = time.perf_counter() - t
            return
        N_g, D_grad = grad_loader.dims()
        if N_g != N:
            raise ValueError(
                f"load_step2_sparse_inputs: gradient N={N_g} != "
                f"n_lh+n_rh={N}")
        state["D_grad"] = int(D_grad)
        timings["gradients"] = time.perf_counter() - t

    def _task_group() -> None:
        from arealmshbm.data_io import mat5_stream
        t = time.perf_counter()
        gp = (resolve_path(project_dir, cohort.group_mat)
              if cohort.group_mat else project_dir / "group" / "group.mat")
        if not gp.exists():
            raise FileNotFoundError(f"group.mat not found: {gp}")
        # The same four-key contract ``load_group_mtc`` enforces; only
        # 'mtc' is materialised, the rest is a header walk.
        names = set(mat5_stream.walk_names(gp))
        missing = {"mtc", "epsil", "lh_labels", "rh_labels"} - names
        if missing:
            raise KeyError(
                f"group.mat missing {sorted(missing)}: keys={sorted(names)}")
        mtc = np.asarray(mat5_stream.read_fields(gp, {"mtc"})["mtc"])
        if mtc.ndim != 2:
            raise ValueError(
                f"group.mat 'mtc' must be 2-D; got shape {mtc.shape}")
        if int(mtc.shape[1]) != L_cfg:
            raise ValueError(
                f"group.mtc has L={mtc.shape[1]} but "
                f"cfg.num_clusters={L_cfg}")
        state["mtc"] = np.ascontiguousarray(mtc, dtype=np.float64)
        timings["group_mtc"] = time.perf_counter() - t

    def _task_mask() -> None:
        # NOTE: do NOT stash ``mat5_stream.LAST_PATH`` in ``state`` here (or
        # in ``_task_group``). It is a module-level global and, with the
        # default ``overlap=True``, these tasks run concurrently on the
        # thread pool below — each would read whichever route the other
        # task wrote last. If a per-file 'fast'/'scipy' diagnostic is ever
        # wanted, mat5_stream must return it, not stash it.
        from arealmshbm.data_io import mat5_stream
        t = time.perf_counter()
        mp = (resolve_path(project_dir, cohort.spatial_mask_mat)
              if cohort.spatial_mask_mat
              else project_dir / "spatial_mask" /
              f"spatial_mask_{cfg.mesh}.mat")
        if not mp.exists():
            raise FileNotFoundError(f"spatial mask not found: {mp}")
        lh_b = mat5_stream.read_sparse(mp, "lh_boundary")
        rh_b = mat5_stream.read_sparse(mp, "rh_boundary")
        state["layout"] = build_step2_layout(lh_b, rh_b)
        timings["boundary"] = time.perf_counter() - t

    tasks = (_task_bold, _task_grad, _task_group, _task_mask)
    if overlap:
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            for fut in [ex.submit(fn) for fn in tasks]:
                fut.result()
    else:
        for fn in tasks:
            fn()

    # ── stitch ──
    layout: Step2Layout = state["layout"]
    D = int(state["D"])
    Db = int(state["Db"])
    D_grad = int(state["D_grad"])
    mtc = state["mtc"]
    if layout.N != N:
        raise ValueError(
            f"load_step2_sparse_inputs: spatial mask N={layout.N} != "
            f"n_lh+n_rh={N}")
    if layout.n_lh != n_lh:
        raise ValueError(
            f"load_step2_sparse_inputs: spatial mask n_lh={layout.n_lh} != "
            f"mesh n_lh={n_lh}")
    if layout.L != L_cfg:
        raise ValueError(
            f"load_step2_sparse_inputs: spatial mask L={layout.L} != "
            f"cfg.num_clusters={L_cfg}")
    if int(mtc.shape[0]) != D:
        raise ValueError(
            f"load_step2_sparse_inputs: group.mtc has {mtc.shape[0]} rows "
            f"but the profiles carry D={D}")

    def _bold_reader(s_1: int, out: np.ndarray) -> None:
        if out.shape != (T, N, Db) or out.dtype != np.uint8:
            raise ValueError(
                f"bold_reader: out must be uint8 (T,N,Db)=({T},{N},{Db}); "
                f"got {out.dtype} {out.shape}")
        from arealmshbm.data_io.profile_io import (
            decode_subject_profile_packed_into,
            open_subject_profile_packed_tnd,
        )
        h = open_subject_profile_packed_tnd(b2nd_paths[s_1 - 1])
        try:
            _check_subject_header(h, (T, N, Db), D, b2nd_paths[s_1 - 1], s_1)
            decode_subject_profile_packed_into(h, out)
        finally:
            del h

    grad_reader: Optional[GradReader] = None
    if grad_loader is not None:
        def _grad_reader(s_1: int, out: np.ndarray) -> None:
            if out.shape != (N, D_grad) or out.dtype != np.float32:
                raise ValueError(
                    f"grad_reader: out must be fp32 (N,Dg)=({N},{D_grad}); "
                    f"got {out.dtype} {out.shape}")
            grad_loader.read_stacked_into(s_1, out)

        grad_reader = _grad_reader

    timings["total"] = time.perf_counter() - t_all
    return Step2SparseInputs(
        layout=layout, S=S, T=T, N=N, D=D, D_grad=D_grad,
        n_lh=n_lh, n_rh=n_rh, mtc=mtc, dim=D - 1,
        bold_reader=_bold_reader, grad_reader=grad_reader, timings=timings,
    )
