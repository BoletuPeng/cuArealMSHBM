"""subject_loaders.py — per-subject streaming loaders for step-2 EM.

The master kernel's per-EM-iter loop reads each subject's BOLD twice
(Phase A.3 + Phase D) and grad once (Phase C); these loaders decode
**one subject at a time** from disk on demand, capping peak BOLD RAM
at one subject's slab (~770 MB at fsa6/T=6) and grad RAM at ~30 MB
regardless of cohort size S. At S=200 the eager-mega-buffer
alternative would be 231 GB BOLD + 39 GB grad — incompatible with any
reasonable RAM budget.

Discovery contract
------------------

Both loaders are constructed from a :class:`CohortManifest` (read from
``project_dir/cohort.json`` by the caller) plus the ``project_dir`` to
resolve relative paths. There is **no** auto-detect / probe / fallback
— the cohort manifest is the single source of truth for which subject
maps to which file. If a subject's ``profile_b2nd`` is missing from
the manifest, the loader raises at construction.

BOLD: per-subject bitpacked ``.b2nd`` (only on-disk format supported).

Gradient: per-hemi ``.npy`` (preferred, step0 output) or per-hemi
``.mat`` (legacy CBIG output); the loader sniffs the suffix on the
cohort.json path and routes accordingly.

Both loaders return per-subject **internal-layout** arrays matching
what the step-2 master kernel expects:

  * BOLD: ``(N, T, D)`` fp32 C-contig — per-session demean +
    L2-row-norm applied. MW rows trusted to be zero on disk (verified
    one-shot at construction when mars masks are passed).
  * grad: ``(T, N, D_grad)`` fp32 C-contig — the per-(s, t) slice the
    kernel's Phase C reads (gradient is session-invariant in this
    fork; the same ``(N, D_grad)`` is replicated across T sessions).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from arealmshbm.data_io.cohort import (
    CohortManifest, resolve_path,
)
from arealmshbm.data_io.profile_io import open_subject_profile_packed_tnd

from .load_subject_profiles import (
    _widen_normalize_bitpacked_to_f32_NTD_kernel,
)
from .load_subject_gradient import _read_emb_100


# ─────────────────────────────────────────────────────────────────────
# BOLD profile per-subject loader
# ─────────────────────────────────────────────────────────────────────
class SubjectProfileLoader:
    """Per-subject BOLD profile loader.

    Constructed from a :class:`CohortManifest` (read from
    ``project_dir/cohort.json``) plus the ``project_dir`` to resolve
    relative paths. The manifest is the authoritative roster — every
    subject must have a non-None ``profile_b2nd``.

    Subject indexing is **1-based by position in the manifest's
    ``subjects`` list**, matching the rest of step-2's contract.

    Methods
    -------
    load(s) -> ndarray
        Returns ``(N, T, D)`` fp32 C-contig for subject ``s`` (1-based),
        with per-session demean + L2-row-norm applied.

    load_into(s, out) -> None
        Decodes subject ``s`` directly into ``out`` (a pre-allocated
        ``(N, T, D)`` fp32 ndarray). Lets the caller hold one scratch
        slot and re-use it across all S subject visits.

    dims() -> (N, T, D)
        Peeks subject 1 to learn the dims (one chunk decode).

    Attributes
    ----------
    format : str
        Always ``"b2nd"`` for the disk-backed loader. Preserved for
        compatibility with code that branches on loader type.
    """

    def __init__(
        self,
        cohort: CohortManifest,
        project_dir: Path,
        *,
        targ_mesh: str,
        seed_mesh: str,
        lh_mars: Optional[np.ndarray] = None,
        rh_mars: Optional[np.ndarray] = None,
        cache_mode: str = "stream",
    ):
        # cache_mode:
        #   'stream' (default) — per-call disk decode + normalize.
        #   'eager_bitpacked' — lazy-build a host (N, T, ⌈D/8⌉) uint8
        #                       bit-packed cache per subject on first
        #                       load_into; subsequent calls feed a
        #                       numba widen+normalize kernel from the
        #                       cache, eliminating the disk decode
        #                       (~250 ms/subject) on every hit.
        if cache_mode not in ("stream", "eager_bitpacked"):
            raise ValueError(
                f"SubjectProfileLoader.cache_mode must be 'stream' or "
                f"'eager_bitpacked' (got {cache_mode!r})"
            )
        self.cache_mode = cache_mode
        self._packed_cache: dict = {}  # s -> (N, T, D_bytes) uint8 host ndarray
        self.project_dir = Path(project_dir)
        self.num_sub = cohort.num_sub
        self.num_session = cohort.num_session

        # Mesh-config consistency check: the caller's expected
        # (targ_mesh, seed_mesh) must agree with what cohort.json
        # records. Mismatch means the cohort was produced for a
        # different surface space and the .b2nd filenames don't line up
        # with the consumer's mesh-derived state.
        if cohort.mesh.get("targ") != targ_mesh:
            raise ValueError(
                f"SubjectProfileLoader: cohort targ_mesh="
                f"{cohort.mesh.get('targ')!r} != expected {targ_mesh!r}"
            )
        if cohort.mesh.get("seed") != seed_mesh:
            raise ValueError(
                f"SubjectProfileLoader: cohort seed_mesh="
                f"{cohort.mesh.get('seed')!r} != expected {seed_mesh!r}"
            )

        self._lh_mars = (None if lh_mars is None
                         else np.asarray(lh_mars).ravel())
        self._rh_mars = (None if rh_mars is None
                         else np.asarray(rh_mars).ravel())

        # Resolve per-subject .b2nd paths from the manifest. The
        # loader's 1-based subject index matches the manifest's
        # ``subjects`` list order; the caller's ``s`` is positional, not
        # an id lookup. We still record the id for error messages.
        self._b2nd_paths: List[Path] = []
        self._subject_ids: List[str] = []
        for i, sub in enumerate(cohort.subjects):
            if sub.profile_b2nd is None:
                raise ValueError(
                    f"SubjectProfileLoader: cohort.subjects[{i}] "
                    f"(id={sub.id!r}) has no profile_b2nd path. Run "
                    f"step1's generate_profiles subgraph for this "
                    f"subject."
                )
            self._b2nd_paths.append(
                resolve_path(self.project_dir, sub.profile_b2nd)
            )
            self._subject_ids.append(sub.id)
        for i, p in enumerate(self._b2nd_paths):
            if not p.exists():
                raise FileNotFoundError(
                    f"SubjectProfileLoader: cohort.subjects[{i}] "
                    f"(id={self._subject_ids[i]!r}) profile_b2nd not "
                    f"found on disk: {p}"
                )

        self.format = "b2nd"

        # Validate dims from sub 1, one-shot. Cheap (one chunk-header
        # read on the bitpacked handle; ``D_unpacked`` comes from
        # vlmeta — no per-cell decode at construction).
        a = self._open_b2nd(1)
        T_disk = int(a.shape[0])
        N_disk = int(a.shape[1])
        D_unpacked = int(a.D_unpacked)

        # Trust-contract verification: the loader does NOT zero MW
        # rows at read time; it assumes step1's threshold→binarize
        # already produced zero MW rows on disk. One-time cost; only
        # runs when mars masks are provided. If mars are NOT provided,
        # the contract is silently trusted — emit a warning so direct
        # callers see the gap. Verification decodes one packed chunk
        # for sess 1 and checks that every packed byte at MW rows is
        # zero (which is equivalent to every unpacked bit at MW being
        # zero — no fp32 unpack needed).
        if self._lh_mars is None or self._rh_mars is None:
            warnings.warn(
                "SubjectProfileLoader: constructed without lh_mars / "
                "rh_mars - medial-wall contract (step1 writer producing "
                "zero MW rows) is trusted but NOT verified.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            packed_ND_check = np.asarray(a[0])
            n_lh_check = int(self._lh_mars.size)
            lh_mw_rows = packed_ND_check[:n_lh_check][self._lh_mars == 1]
            rh_mw_rows = packed_ND_check[n_lh_check:][self._rh_mars == 1]
            if (lh_mw_rows.size and np.any(lh_mw_rows != 0)) or \
               (rh_mw_rows.size and np.any(rh_mw_rows != 0)):
                del a
                raise ValueError(
                    f"SubjectProfileLoader: .b2nd at {self._b2nd_paths[0]} "
                    f"has non-zero values at MW vertices in sub1/sess1. "
                    f"step1's binarize stage should produce zero MW "
                    f"rows; this dataset violates that contract."
                )
        del a
        if T_disk != self.num_session:
            raise ValueError(
                f"SubjectProfileLoader: .b2nd at {self._b2nd_paths[0]} "
                f"has T={T_disk} but cohort num_session={self.num_session}"
            )

        # Cached dims (N/T/D filled now; all three sourced from the
        # single chunk-header read above — no second handle open).
        self._dims_cache: Tuple[int, int, int] = (
            N_disk, T_disk, D_unpacked,
        )

    # ── internal: open one .b2nd packed handle ──
    def _open_b2nd(self, s_1: int):
        return open_subject_profile_packed_tnd(self._b2nd_paths[s_1 - 1])

    # ── public: dims ──
    def dims(self) -> Tuple[int, int, int]:
        """Return ``(N, T, D)`` — cached at construction from subject 1."""
        return self._dims_cache

    # ── public: load one subject ──
    def load(self, s: int) -> np.ndarray:
        """Return ``(N, T, D)`` fp32 C-contig for subject ``s`` (1-based)."""
        N, T, D = self.dims()
        out = np.empty((N, T, D), dtype=np.float32)
        self.load_into(s, out)
        return out

    def load_into(self, s: int, out: np.ndarray) -> None:
        """Decode subject ``s`` into the caller-allocated ``out`` slot.

        ``out`` must be ``(N, T, D)`` fp32 C-contig. Output bytes:
        per-session demean + L2-row-norm.

        Both cache modes read the bitpacked bytes off disk and run
        :func:`_widen_normalize_bitpacked_to_f32_NTD_kernel` (numba)
        to do bit-unpack + demean + L2-norm in one fused pass per
        (n, t) row — no fp32 chunk-view intermediate.

        Cache dispatch — when ``cache_mode='eager_bitpacked'``, the
        first ``load_into(s)`` for each subject pays the disk decode
        cost (~250 ms); subsequent calls skip disk and read directly
        from the host packed cache. When ``cache_mode='stream'``,
        every call re-reads the packed bytes from disk.
        """
        if out.dtype != np.float32:
            raise ValueError(f"out must be fp32; got {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        N, T, D = self.dims()
        if out.shape != (N, T, D):
            raise ValueError(
                f"out shape {out.shape} != ({N}, {T}, {D})")

        if self.cache_mode == "eager_bitpacked":
            self._load_into_from_packed_cache(s, out, N, T, D)
            return

        self._load_into_b2nd(s, out, N, T, D)

    # ── eager bitpacked host cache path ──
    def _load_into_from_packed_cache(self, s: int, out: np.ndarray,
                                       N: int, T: int, D: int) -> None:
        """Bit-packed cache fast-path. Reads packed bytes from disk on
        first call, then runs the widen+normalize numba kernel from
        the cached bytes on subsequent calls."""
        packed = self._packed_cache.get(s)
        if packed is None:
            packed = self._read_packed_from_b2nd(s, N, T, D)
            self._packed_cache[s] = packed
        _widen_normalize_bitpacked_to_f32_NTD_kernel(packed, out, np.int64(D))

    def _read_packed_from_b2nd(self, s: int, N: int, T: int, D: int) -> np.ndarray:
        """Disk → ``(N, T, ⌈D/8⌉) uint8`` packed cache slab, no fp32.

        Reads the per-session ``(N, ⌈D/8⌉)`` packed chunks from disk and
        strides them into the cache's ``(N, T, ⌈D/8⌉)`` layout.
        """
        D_bytes = (D + 7) // 8
        pview = open_subject_profile_packed_tnd(self._b2nd_paths[s - 1])
        try:
            if pview.D_unpacked != D:
                raise ValueError(
                    f"SubjectProfileLoader: sub {s} on-disk D_unpacked"
                    f"={pview.D_unpacked} != cached D={D} from sub 1"
                )
            T_disk, N_disk, D_b_disk = (int(x) for x in pview.shape)
            if T_disk != T or N_disk != N or D_b_disk != D_bytes:
                raise ValueError(
                    f"SubjectProfileLoader: sub {s} packed shape "
                    f"({T_disk}, {N_disk}, {D_b_disk}) != expected "
                    f"({T}, {N}, {D_bytes})"
                )
            packed = np.empty((N, T, D_bytes), dtype=np.uint8)
            for t in range(T):
                # ``pview[t]`` decodes one chunk → fresh (N, D_bytes) uint8.
                packed[:, t, :] = np.asarray(pview[t])
        finally:
            del pview
        return packed

    # ── public: prefetch packed bytes into cache (parallel disk pump) ──
    def prefetch_packed(self, s: int) -> None:
        """Read subject ``s``'s packed bytes from disk into the internal
        ``_packed_cache`` slot (no fp32 widen, no normalize).

        Only valid in ``cache_mode='eager_bitpacked'``. Idempotent — a
        second call for the same ``s`` is a no-op. Designed for use from
        a :class:`concurrent.futures.ThreadPoolExecutor` worker pool to
        overlap disk reads across subjects with the main thread's
        widen+normalize numba kernel work — blosc2 chunk decoding
        releases the GIL during decompression, so concurrent prefetch
        from N worker threads achieves ~Nx disk-bandwidth scaling on a
        cold cohort (and ~free idle on a hot OS-page-cache cohort).

        Thread-safety: each call opens its own b2nd handle on a
        different ``s``; the only shared mutable state is the
        ``_packed_cache`` dict, and CPython dict ``__setitem__`` on
        distinct keys is GIL-protected per single bytecode op.
        """
        if self.cache_mode != "eager_bitpacked":
            raise RuntimeError(
                "prefetch_packed only valid in cache_mode='eager_bitpacked' "
                f"(got {self.cache_mode!r}); switch cache_mode to "
                "'eager_bitpacked' or construct a fresh loader."
            )
        if s in self._packed_cache:
            return
        N, T, D = self.dims()
        packed = self._read_packed_from_b2nd(s, N, T, D)
        # Dict assignment on distinct keys is GIL-atomic in CPython.
        self._packed_cache[s] = packed

    # ── b2nd streaming path (bitpacked → fp32 fused) ──
    def _load_into_b2nd(self, s: int, out: np.ndarray,
                         N: int, T: int, D: int) -> None:
        """Read all T sessions of subject s's packed bytes off disk and
        run the fused bit-unpack + demean + L2-row-norm numba kernel
        straight into ``out``. Bit-identical to a hypothetical "unpack
        to fp32 + standard per-row normalize" reference on binary
        {0, 1} input (see :func:`_widen_normalize_bitpacked_to_f32_NTD_kernel`'s
        docstring for the fp32 round-off contract proof).

        The .b2nd content holds the BINARY 0/1 profile and is
        pre-medial-zeroed (step1's threshold→binarize already produces
        zero rows at medial vertices); the kernel preserves that
        invariant — MW rows trip the ``has_zero`` gate and stay at
        their demeaned value (which is 0 for an all-zero row).
        """
        D_bytes = (D + 7) // 8
        packed_NTDb = np.empty((N, T, D_bytes), dtype=np.uint8)
        pview = self._open_b2nd(s)
        try:
            for t in range(T):
                # (N, D_bytes) uint8 chunk decode — zero fp32 cost.
                packed_NTDb[:, t, :] = np.asarray(pview[t])
        finally:
            del pview
        _widen_normalize_bitpacked_to_f32_NTD_kernel(
            packed_NTDb, out, np.int64(D),
        )


# ─────────────────────────────────────────────────────────────────────
# Gradient (emb) per-subject loader
# ─────────────────────────────────────────────────────────────────────
class SubjectGradientLoader:
    """Per-subject diffusion-embedding loader.

    The gradient is **session-invariant** in this fork — one
    ``(N, D_grad)`` matrix per subject, replicated across the T axis
    when materialized into the kernel-expected ``(T, N, D_grad)`` shape.

    Constructed from a :class:`CohortManifest` + ``project_dir``. Each
    subject's gradient paths come from ``gradient_lh`` / ``gradient_rh``
    fields in the cohort. The loader sniffs the suffix to pick a
    reader: ``.npy`` (preferred, step0 output) reads via ``np.load``;
    ``.mat`` (legacy CBIG output) reads via ``_read_emb_100``.

    Subject indexing is **1-based by position** in the manifest's
    ``subjects`` list.
    """

    def __init__(
        self,
        cohort: CohortManifest,
        project_dir: Path,
        *,
        n_components: int = 100,
    ):
        self.project_dir = Path(project_dir)
        self.num_sub = cohort.num_sub
        self.num_session = cohort.num_session
        self.n_components = int(n_components)

        # Resolve per-subject gradient paths from the manifest.
        self._grad_paths: List[Tuple[Path, Path]] = []
        self._subject_ids: List[str] = []
        for i, sub in enumerate(cohort.subjects):
            if sub.gradient_lh is None or sub.gradient_rh is None:
                raise ValueError(
                    f"SubjectGradientLoader: cohort.subjects[{i}] "
                    f"(id={sub.id!r}) has no gradient_lh/gradient_rh "
                    f"path. gMSHBM mode requires gradient embeddings."
                )
            lh = resolve_path(self.project_dir, sub.gradient_lh)
            rh = resolve_path(self.project_dir, sub.gradient_rh)
            self._grad_paths.append((lh, rh))
            self._subject_ids.append(sub.id)
        for i, (lh, rh) in enumerate(self._grad_paths):
            if not lh.exists() or not rh.exists():
                raise FileNotFoundError(
                    f"SubjectGradientLoader: cohort.subjects[{i}] "
                    f"(id={self._subject_ids[i]!r}) gradient files "
                    f"missing on disk: lh={lh} rh={rh}"
                )

        # ``format`` is determined by sub 1's suffix; all subjects must
        # agree (no mixed-format cohorts).
        first_suffix = self._grad_paths[0][0].suffix.lower()
        for i, (lh, rh) in enumerate(self._grad_paths):
            if lh.suffix.lower() != first_suffix or rh.suffix.lower() != first_suffix:
                raise ValueError(
                    f"SubjectGradientLoader: cohort.subjects[{i}] "
                    f"has mixed gradient suffix ({lh.suffix}, "
                    f"{rh.suffix}) vs first-subject {first_suffix!r}; "
                    f"all subjects must use the same gradient format."
                )
        if first_suffix == ".npy":
            self.format = "npy"
        elif first_suffix == ".mat":
            self.format = "mat"
        else:
            raise ValueError(
                f"SubjectGradientLoader: unsupported gradient suffix "
                f"{first_suffix!r}; expected .npy or .mat."
            )

        self._dims_cache: Optional[Tuple[int, int]] = None

    def dims(self) -> Tuple[int, int]:
        """Return ``(N, D_grad)`` by peeking subject 1."""
        if self._dims_cache is not None:
            return self._dims_cache
        lh_p, rh_p = self._grad_paths[0]
        if self.format == "npy":
            # mmap header read avoids decompressing the full payload.
            lh = np.load(lh_p, mmap_mode="r")
            rh = np.load(rh_p, mmap_mode="r")
            N_d = int(lh.shape[0] + rh.shape[0])
            D_d = int(lh.shape[1])
            if D_d < self.n_components or rh.shape[1] < self.n_components:
                raise ValueError(
                    f"gradient .npy has D_grad < {self.n_components}"
                )
            self._dims_cache = (N_d, self.n_components)
            return self._dims_cache
        # .mat
        lh = _read_emb_100(lh_p, self.n_components)
        rh = _read_emb_100(rh_p, self.n_components)
        self._dims_cache = (int(lh.shape[0] + rh.shape[0]), self.n_components)
        return self._dims_cache

    def load(self, s: int) -> np.ndarray:
        """Return ``(T, N, D_grad)`` fp32 C-contig for subject ``s``
        (1-based). The same ``(N, D_grad)`` matrix is replicated across
        the T axis.
        """
        N, D_grad = self.dims()
        T = self.num_session
        out = np.empty((T, N, D_grad), dtype=np.float32)
        self.load_into(s, out)
        return out

    def load_into(self, s: int, out: np.ndarray) -> None:
        """Decode subject ``s`` into the caller-allocated
        ``(T, N, D_grad)`` fp32 C-contig ``out`` slot.
        """
        if out.dtype != np.float32:
            raise ValueError(f"out must be fp32; got {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        N, D_grad = self.dims()
        T = self.num_session
        if out.shape != (T, N, D_grad):
            raise ValueError(
                f"out shape {out.shape} != ({T}, {N}, {D_grad})")

        lh_p, rh_p = self._grad_paths[s - 1]
        if self.format == "npy":
            lh = np.ascontiguousarray(
                np.load(lh_p)[:, :D_grad], dtype=np.float32
            )
            rh = np.ascontiguousarray(
                np.load(rh_p)[:, :D_grad], dtype=np.float32
            )
        else:
            lh = _read_emb_100(lh_p, D_grad)
            rh = _read_emb_100(rh_p, D_grad)
        if lh.shape[0] + rh.shape[0] != N:
            raise ValueError(
                f"sub {s}: lh+rh rows {lh.shape[0]+rh.shape[0]} != N={N}"
            )
        stacked = np.concatenate([lh, rh], axis=0)  # (N, D_grad)

        # Replicate across the T axis. T memcopies at ~20 GB/s; ~30 ms
        # at fsa6/T=6 (negligible vs the GEMM cost downstream).
        for t in range(T):
            out[t] = stacked


# ─────────────────────────────────────────────────────────────────────
# In-memory loader adapters — wrap a full (S, N, T, D) / (S, T, N, D_grad)
# array in RAM as a streaming loader. Used by tests / benches that build
# synthetic data in RAM without going through disk.
# ─────────────────────────────────────────────────────────────────────
class InMemoryProfileLoader:
    """Adapter that wraps an in-memory ``(S, N, T, D)`` fp32 array as a
    :class:`SubjectProfileLoader`. ``load_into(s, out)`` does one
    ``np.copyto`` from ``arr[s-1]`` into ``out``. Useful for unit tests
    + the numerical-match harness that already build a synthetic
    BOLD stack in RAM.
    """

    def __init__(self, bold_SNTD: np.ndarray, num_session: int):
        if bold_SNTD.ndim != 4:
            raise ValueError(
                f"bold_SNTD must be 4D (S, N, T, D); got {bold_SNTD.shape}")
        if bold_SNTD.dtype != np.float32:
            raise ValueError(
                f"bold_SNTD must be fp32; got {bold_SNTD.dtype}")
        self._arr = bold_SNTD
        self.num_sub = int(bold_SNTD.shape[0])
        self.num_session = int(num_session)
        if self.num_session != int(bold_SNTD.shape[2]):
            raise ValueError(
                f"num_session {self.num_session} != bold_SNTD.shape[2] "
                f"{bold_SNTD.shape[2]}")
        self.format = "in_memory"

    def dims(self) -> Tuple[int, int, int]:
        S, N, T, D = self._arr.shape
        return (int(N), int(T), int(D))

    def load(self, s: int) -> np.ndarray:
        return np.ascontiguousarray(self._arr[s - 1])

    def load_into(self, s: int, out: np.ndarray) -> None:
        np.copyto(out, self._arr[s - 1])


class InMemoryGradientLoader:
    """Adapter that wraps an in-memory ``(S, T, N, D_grad)`` fp32 array
    as a :class:`SubjectGradientLoader`. Same contract — one
    ``np.copyto`` per ``load_into`` call.
    """

    def __init__(self, grad_STND: np.ndarray, num_session: int):
        if grad_STND.ndim != 4:
            raise ValueError(
                f"grad_STND must be 4D (S, T, N, D_grad); got {grad_STND.shape}")
        if grad_STND.dtype != np.float32:
            raise ValueError(
                f"grad_STND must be fp32; got {grad_STND.dtype}")
        self._arr = grad_STND
        self.num_sub = int(grad_STND.shape[0])
        self.num_session = int(num_session)
        if self.num_session != int(grad_STND.shape[1]):
            raise ValueError(
                f"num_session {self.num_session} != grad_STND.shape[1] "
                f"{grad_STND.shape[1]}")
        self.format = "in_memory"

    def dims(self) -> Tuple[int, int]:
        S, T, N, D_grad = self._arr.shape
        return (int(N), int(D_grad))

    def load(self, s: int) -> np.ndarray:
        return np.ascontiguousarray(self._arr[s - 1])

    def load_into(self, s: int, out: np.ndarray) -> None:
        np.copyto(out, self._arr[s - 1])
