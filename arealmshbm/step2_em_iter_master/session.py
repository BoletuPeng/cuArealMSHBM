"""session.py — Python wrapper around the EM-iter master kernel.

Owns all scratch buffers and dispatches the master kernel. The
Session takes per-subject **loaders** (BOLD + gradient) and decodes
one subject at a time into pre-allocated scratch (``_bold_scratch_NTD``
/ ``_grad_scratch_TND``) inside :meth:`run_iter` — peak BOLD/grad RAM
stays at one subject's slab regardless of cohort size S.

Params dicts hold ``(S, T, L, D)``-family arrays natively; ``run_iter``
does zero transposes / promotes per call.

Internal layout:
    BOLD per subject  : (N, T, D)         fp32 C-contig  ← streamed scratch (NTD)
    grad per subject  : (T, N, D_grad)    fp32 C-contig (gMSHBM only)
    s_t_nu_STLD       : (S, T, L, D)      fp32 C-contig (+ ping-pong buffer)
    s_lambda_SNL_f32  : (S, N, L)         fp32 C-contig (storage)
    s_lambda_NL_f64_scratch : (N, L)      fp64 C-contig (per-subject softmax intermediate)
    s_psi_SLD         : (S, L, D)         fp32 C-contig
    boundary_mask_NL  : (N, L)            fp32 C-contig
    theta_NL          : (N, L)            fp32 C-contig
    mu_LD             : (L, D)            fp32 C-contig
    sigma_L / epsil_L : (L,)              fp32

The external (D-leading) layout — (D, L, T, S) / (N, L, S) / (D, L, S)
/ (D, L) — only appears at the ``.mat`` save_params boundary, where
:meth:`Step2Pipeline._save_params` transposes.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from arealmshbm.step2_io import (
    SubjectGradientLoader,
    SubjectProfileLoader,
)

from ._kernels import (
    em_iter_master_kernel_streaming,
)


def _validate_session_ctor_args(
    *,
    cls_name: str,
    mode: str,
    bold_loader,
    grad_loader,
    num_sub: int,
    N: int,
    T: int,
    D: int,
    D_grad: int,
    num_clusters: int,
    n_lh: int,
    boundary_mask: np.ndarray,
    s_psi: np.ndarray,
    sigma: np.ndarray,
) -> None:
    """Shared up-front validation between :class:`Step2EmIterSession` (CPU)
    and :class:`Step2EmIterSessionCUDA` (GPU). Both backends accept the same
    ctor contract — only what they DO with the validated state differs
    (host scratch vs device cache). Raises on the first mismatch; never
    mutates inputs.

    ``cls_name`` threads the calling class's name into the cMSHBM
    NotImplementedError message — preserved from the per-class ctors so
    the user still sees which Session refused the mode.

    Layout / dtype coercion (``np.ascontiguousarray`` / ``astype``) stays
    in the per-class ctor: the CPU path lands the bytes in a host
    scratch, the GPU path H2D's into a device cache, and the validator
    has no business deciding which.
    """
    if mode == "cMSHBM":
        raise NotImplementedError(
            f"{cls_name}: mode='cMSHBM' is not wired (no "
            "xyz-vMF path in the master kernel)."
        )
    if mode not in ("gMSHBM", "dMSHBM"):
        raise ValueError(
            f"unknown mode {mode!r}; supported: 'gMSHBM', 'dMSHBM'."
        )
    if bold_loader is None:
        raise ValueError("bold_loader is required")

    S_i = int(num_sub)
    N_i = int(N)
    T_i = int(T)
    D_i = int(D)
    L_i = int(num_clusters)
    n_lh_i = int(n_lh)

    if N_i != 2 * n_lh_i:
        raise ValueError(f"n_lh={n_lh} != N/2={N_i // 2}")
    if int(bold_loader.num_sub) != S_i:
        raise ValueError(
            f"bold_loader.num_sub={bold_loader.num_sub} != num_sub={S_i}"
        )
    bld_N, bld_T, bld_D = bold_loader.dims()
    if (bld_N, bld_T, bld_D) != (N_i, T_i, D_i):
        raise ValueError(
            f"bold_loader.dims() {(bld_N, bld_T, bld_D)} != "
            f"({N_i}, {T_i}, {D_i})"
        )

    if mode == "gMSHBM":
        if grad_loader is None:
            raise ValueError("mode='gMSHBM' requires grad_loader")
        # Mirror the per-class ctor's coercion exactly: gate on the raw
        # ``D_grad > 0`` (not ``int(D_grad) > 0``) so fractional inputs
        # produce the same effective value here as in the ctor's
        # ``self.D_grad`` assignment. Production passes positive ints so
        # the two gates agree, but mismatched gates would let a 0.5-like
        # input validate against a different effective D_grad than the
        # one the ctor would store.
        D_grad_int = int(D_grad)
        D_grad_i = D_grad_int if D_grad > 0 else 1
        grd_N, grd_D = grad_loader.dims()
        if grd_N != N_i:
            raise ValueError(
                f"grad_loader.dims() N={grd_N} != N={N_i}"
            )
        if grd_D != D_grad_i:
            raise ValueError(
                f"grad_loader.dims() D_grad={grd_D} != D_grad={D_grad_i}"
            )

    if s_psi.shape != (S_i, L_i, D_i):
        raise ValueError(
            f"s_psi shape {s_psi.shape} != ({S_i}, {L_i}, {D_i}); "
            "expected internal (S, L, D) layout"
        )

    sigma_size = np.asarray(sigma).reshape(-1).size
    if sigma_size != L_i:
        raise ValueError(f"sigma size {sigma_size} != L={L_i}")

    bm_shape = np.asarray(boundary_mask).shape
    if bm_shape != (N_i, L_i):
        raise ValueError(
            f"boundary_mask shape {bm_shape} != ({N_i}, {L_i})"
        )


def _validate_initial_state_params(Params: Dict[str, np.ndarray],
                                    S: int, T: int, N: int, L: int, D: int,
                                    ) -> None:
    """Validate Params['s_t_nu'/'s_lambda'/'theta'] against the master
    kernel's shape + dtype contract. Shared between :class:`Step2EmIterSession`
    (CPU) and :class:`Step2EmIterSessionCUDA` (GPU); both backends impose
    identical bit-for-bit ingress requirements — only the downstream
    "where the buffer lands" differs.

    C-contiguity is intentionally NOT validated here: the caller
    (``upload_initial_state``) np.copyto's into a Session-owned scratch
    that was allocated via ``np.empty(..., dtype=np.float32)`` (C-contig
    by construction) and then rebinds ``Params[key]`` to that scratch.
    The downstream ``run_iter`` identity check (``Params[key] is
    _scratch``) therefore guarantees C-contig at every kernel call. If
    a future change makes ``upload_initial_state`` use the caller's
    buffer directly (skipping the copyto), restore a per-key
    ``flags['C_CONTIGUOUS']`` check here.
    """
    expected = {
        "s_t_nu":   (S, T, L, D),
        "s_lambda": (S, N, L),
        "theta":    (N, L),
    }
    for key, exp_shape in expected.items():
        arr = Params[key]
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"upload_initial_state: Params[{key!r}] must be ndarray; "
                f"got {type(arr).__name__}"
            )
        if arr.shape != exp_shape:
            raise ValueError(
                f"upload_initial_state: Params[{key!r}] shape {arr.shape} "
                f"!= expected {exp_shape}"
            )
        if arr.dtype != np.float32:
            raise ValueError(
                f"upload_initial_state: Params[{key!r}] must be fp32; "
                f"got {arr.dtype}"
            )


class Step2EmIterSession:
    """Holds scratch + caches for the EM-iter master kernel.

    Construct ONCE per ``vmf_clustering_batch`` call (i.e., once per
    intra-EM outer iter). Per ``run_iter`` invocation: ONE outer-EM
    iter of the inner-iter while-loop, spatial prior, fused E-step,
    normalize, and theta update.

    The caller (vmf_clustering_batch) handles the EM-convergence test
    in Python between ``run_iter`` calls.

    Construction-time work (per Session, once):
      * BOLD: takes caller's pre-stacked ``(S, N, T, D)`` fp32 C-contig
        buffer **by reference** — zero copy, zero alloc. Caller
        (Step2Pipeline.load_inputs) is responsible for landing data in
        NTD per-sub layout straight from disk. Phase A.3 and Phase D
        consume ``BOLD_SNTD[s]: (N, T, D)`` via zero-copy reshape to
        ``(N, T·D)`` for the per-subject big sgemms.
      * grad: same zero-copy reference for the ``(S, T, N, D_grad)``
        buffer (gMSHBM only). Gradients keep TND layout — Phase C's
        block-diagonal sgemms consume the per-t ``(N, D_grad)``
        C-contig slice directly.
      * Copy caller's ``s_psi`` (S, L, D) into Session-owned buffer.
        ~5.6 MB. No transpose — Params hold internal layout natively.

    Per ``run_iter`` work (called once per outer EM iter):
      * Verify that Params["s_t_nu"/"s_lambda"/"theta"] still ARE the
        Session-owned buffers (alias invariant established by
        :meth:`upload_initial_state`; both production via
        ``vmf_clustering_batch`` and the CPU↔GPU parity test in
        ``test_numerical_match_gpu.py`` honour this contract).
      * Invoke the master kernel — operates on the Session-owned
        buffers directly.

    Memory budget (fsaverage6 / S=3 / L=400 / D=1174 / T=6 / D_grad=100):
      * BOLD stack (caller-owned, by reference)
                              : 6.93 GB
      * grad stack (gMSHBM)   : 590 MB
      * s_lambda (fp64)       : 786 MB
      * s_t_nu (A + B)        : 67.6 MB × 2 = 135 MB
      * X_dot_sl + sigma_psi  : 33.8 + 5.6 = 39 MB
      * theta + bm + log_*    : ~525 MB
      * tmp_idx + flag_ST     : ~131 MB
      * Phase D scratch       : 3.76 MB (s_t_nu_TDL transpose for big sgemm B-operand)
      * Spatial scratch       : ~few MB
    """

    __slots__ = (
        # Shape constants.
        "S", "T", "N", "D", "L", "D_grad", "n_lh", "L_lh",
        # Mode / iter caps.
        "has_spatial", "max_iter_m", "ini_val", "beta_internal",
        "dim", "eps_m_step",
        # Per-subject streaming loaders.
        "_bold_loader", "_grad_loader",
        # Per-subject scratch slots — re-used across the S iter of run_iter.
        "_bold_scratch_NTD",      # (N, T, D)   fp32 C-contig — loader fills per s
        "_grad_scratch_TND",      # (T, N, D_g) fp32 C-contig — loader fills per s
        # Static caches (s_psi/sigma staged once; refreshed across intra iters).
        "_s_psi_SLD", "_sigma_L",
        # Per-iter mutable internal-layout buffers.
        "_s_t_nu_A_STLD", "_s_t_nu_B_STLD",
        "_s_lambda_SNL_f32",            # (S, N, L) fp32 storage
        "_s_lambda_NL_f64_scratch",     # (N, L) fp64 — per-subject Phase D softmax intermediate
        "_theta_NL",
        "_boundary_mask_NL",
        # Scratch.
        "_X_dot_sl_STLD", "_sigma_psi_SLD",
        "_log_connect_NL_buf", "_log_vmf_NL_buf",
        "_tmp_idx_NL", "_flag_ST_acc", "_converge_STL",
        "_scratch_u_LD", "_scratch_uupd_DL", "_scratch_sumlam_L",
        "_scratch_grad_sq", "_scratch_u_sq", "_scratch_cross_NL",
        "_s_t_nu_TDL_scratch",   # Phase D B-operand transpose
        "_log_lambda_scratch_NL",  # (N, L) fp32 — Phase D log_lambda pre-softmax
        "_n_alive_count_N",        # (N,)   int32 — Phase D per-vertex alive count
        # Output scalar from last run_iter (per-subject cost).
        "_cost_S",
        # Host mtc cache (for reset_s_t_nu_from_mtc — the GPU mirror
        # caches on device; on CPU we hold a host reference).
        "_mtc_LD_cached",
    )

    def __init__(
        self,
        bold_loader: SubjectProfileLoader,
        grad_loader: Optional[SubjectGradientLoader],
        num_sub: int,
        N: int,
        T: int,
        D: int,
        D_grad: int,
        boundary_mask: np.ndarray,         # (N, L) fp32
        s_psi: np.ndarray,                 # (S, L, D) fp32 — internal layout
        sigma: np.ndarray,                 # (L,) or (1, L) fp32
        *,
        mode: str,
        dim: int,
        num_clusters: int,
        ini_val: float,
        beta_internal: float,
        n_lh: int,
        eps_m_step: float = 1e-4,
        max_iter_m: int = 50,
    ):
        _validate_session_ctor_args(
            cls_name="Step2EmIterSession",
            mode=mode,
            bold_loader=bold_loader, grad_loader=grad_loader,
            num_sub=num_sub, N=N, T=T, D=D, D_grad=D_grad,
            num_clusters=num_clusters, n_lh=n_lh,
            boundary_mask=boundary_mask, s_psi=s_psi, sigma=sigma,
        )
        self.S = int(num_sub)
        self.N = int(N)
        self.T = int(T)
        self.D = int(D)
        self.L = int(num_clusters)
        self.D_grad = int(D_grad) if D_grad > 0 else 1
        self.n_lh = int(n_lh)
        self.L_lh = self.L // 2
        self.dim = int(dim)
        self.has_spatial = 1 if mode == "gMSHBM" else 0
        self.max_iter_m = int(max_iter_m)
        self.ini_val = float(ini_val)
        self.beta_internal = float(beta_internal)
        self.eps_m_step = float(eps_m_step)

        # ── BOLD per-subject scratch: ONE (N, T, D) fp32 slot, re-used ──
        # The loader fills this on each subject visit (Phase A.3 and
        # again at Phase D); the bytes are evictable between visits.
        # At fsa6/T=6/D=1175 this is ~770 MB regardless of S.
        self._bold_loader = bold_loader
        self._bold_scratch_NTD = np.empty(
            (self.N, self.T, self.D), dtype=np.float32
        )

        # ── grad per-subject scratch (gMSHBM only) ──
        # Same pattern. ~200 MB at fsa6/T=6/D_grad=100 regardless of S.
        # grad_loader presence + dims already verified by
        # _validate_session_ctor_args above.
        if self.has_spatial == 1:
            self._grad_loader = grad_loader
            self._grad_scratch_TND = np.empty(
                (self.T, self.N, self.D_grad), dtype=np.float32
            )
        else:
            # Dummy (1, N, 1) for typing — the kernel skips reads when
            # has_spatial=0 so we don't pay grad RAM on dMSHBM.
            self._grad_loader = None
            self._grad_scratch_TND = np.zeros(
                (self.T, self.N, 1), dtype=np.float32
            )

        # ── Stage s_psi / sigma / boundary_mask in host scratch ──
        # Shapes already validated by _validate_session_ctor_args.
        self._s_psi_SLD = np.ascontiguousarray(s_psi, dtype=np.float32).copy()
        self._sigma_L = np.ascontiguousarray(
            np.asarray(sigma).reshape(-1), dtype=np.float32
        )
        self._boundary_mask_NL = np.ascontiguousarray(
            boundary_mask, dtype=np.float32
        )

        # ── Allocate per-iter buffers (zeros / empties; populated in run_iter) ──
        self._s_t_nu_A_STLD = np.empty(
            (self.S, self.T, self.L, self.D), dtype=np.float32
        )
        self._s_t_nu_B_STLD = np.empty_like(self._s_t_nu_A_STLD)
        # fp32 storage; ~26 GB at S=200/fsa6/L=400. fp64 precision is
        # kept ONLY inside Phase D's softmax exp() intermediate
        # (subnormal-safe at β=5000) — see _fused_estep_per_subject_NTD.
        self._s_lambda_SNL_f32 = np.empty(
            (self.S, self.N, self.L), dtype=np.float32
        )
        # Per-subject (N, L) fp64 scratch — Phase D's output lands
        # here in fp64, then a fused Phase E.1 kernel normalizes in
        # fp64 and casts to fp32 storage in one pass. Keeps the
        # multiply+sum+divide chain in fp64 precision. ~262 MB at
        # fsa6/L=400 (single slot, reused across all S subjects per iter).
        self._s_lambda_NL_f64_scratch = np.empty(
            (self.N, self.L), dtype=np.float64
        )
        self._theta_NL = np.empty((self.N, self.L), dtype=np.float32)

        # Per-iter scratch.
        self._X_dot_sl_STLD = np.empty(
            (self.S, self.T, self.L, self.D), dtype=np.float32
        )
        self._sigma_psi_SLD = np.empty(
            (self.S, self.L, self.D), dtype=np.float32
        )
        self._log_connect_NL_buf = np.empty(
            (self.N, self.L), dtype=np.float32
        )
        self._log_vmf_NL_buf = np.empty((self.N, self.L), dtype=np.float32)
        self._tmp_idx_NL = np.zeros((self.N, self.L), dtype=np.bool_)
        self._flag_ST_acc = np.zeros((self.S, self.T), dtype=np.int8)
        self._converge_STL = np.zeros((self.S, self.T, self.L), dtype=np.int8)

        # Spatial scratch (allocated even for non-gMSHBM, sized minimally).
        D_grad = self.D_grad
        self._scratch_u_LD = np.empty((self.L, D_grad), dtype=np.float32)
        self._scratch_uupd_DL = np.empty((D_grad, self.L), dtype=np.float32)
        self._scratch_sumlam_L = np.empty(self.L, dtype=np.float32)
        self._scratch_grad_sq = np.empty(self.N, dtype=np.float32)
        self._scratch_u_sq = np.empty(self.L, dtype=np.float32)
        self._scratch_cross_NL = np.empty((self.N, self.L), dtype=np.float32)

        # Phase D scratch — (T, D, L) fp32 buffer that holds the
        # transpose of s_t_nu_TLD (T, L, D) → (T, D, L) for the big
        # E-step sgemm's right operand. ~3.76 MB at production.
        self._s_t_nu_TDL_scratch = np.empty(
            (self.T, self.D, self.L), dtype=np.float32
        )

        # Per-subject E-step scratch — promoted out of
        # ``_fused_estep_per_subject_NTD`` to avoid per-call alloc churn.
        # At fsa6 / S=200 / I·J·K = 1500 outer-EM iters this would
        # otherwise be ~450k × (131 MB + 328 KB) of allocator traffic.
        # Single Session-owned slot reused across all subjects per iter.
        self._log_lambda_scratch_NL = np.empty(
            (self.N, self.L), dtype=np.float32
        )
        self._n_alive_count_N = np.empty(self.N, dtype=np.int32)

        # Output scratch.
        self._cost_S = np.zeros(self.S, dtype=np.float64)

        # Host mtc cache for reset_s_t_nu_from_mtc — populated by
        # cache_mtc(). Kept as a host ndarray reference (no copy).
        self._mtc_LD_cached: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────
    # Unified API — methods that the pipeline calls on either Session.
    # The GPU mirror (Step2EmIterSessionCUDA) implements the same names;
    # both classes are duck-typed by ``Step2Pipeline.run_em``.
    # ─────────────────────────────────────────────────────────────────
    def upload_initial_state(self, Params: Dict[str, np.ndarray]) -> None:
        """Stage initial Params into Session-owned scratch + rebind aliases.

        On CPU this is the existing zero-copy aliasing: after this call
        ``Params['s_t_nu']``, ``Params['s_lambda']``, and
        ``Params['theta']`` ARE the Session-owned buffers (no per-iter
        memcpy). The GPU mirror does an H2D instead.
        """
        _validate_initial_state_params(
            Params, self.S, self.T, self.N, self.L, self.D,
        )
        np.copyto(self._s_lambda_SNL_f32, Params["s_lambda"])
        Params["s_lambda"] = self._s_lambda_SNL_f32
        np.copyto(self._theta_NL, Params["theta"])
        Params["theta"] = self._theta_NL
        np.copyto(self._s_t_nu_A_STLD, Params["s_t_nu"])
        Params["s_t_nu"] = self._s_t_nu_A_STLD

    def cache_mtc(self, mtc_LD: np.ndarray) -> None:
        """Cache the host ``mtc`` (L, D) used by :meth:`reset_s_t_nu_from_mtc`.

        The CPU path holds a host reference (no copy); the GPU mirror
        H2Ds the array to a device cache once. Same method name on both
        classes; the pipeline calls it once after Session construction.
        """
        mtc = np.ascontiguousarray(mtc_LD, dtype=np.float32)
        if mtc.shape != (self.L, self.D):
            raise ValueError(f"mtc shape {mtc.shape} != ({self.L}, {self.D})")
        self._mtc_LD_cached = mtc

    def reset_s_t_nu_from_mtc(self) -> None:
        """In-place broadcast of cached mtc into ``s_t_nu``.

        On CPU this writes the Session-owned ``_s_t_nu_A_STLD`` buffer
        (= aliased ``Params['s_t_nu']``). The GPU mirror does the same
        broadcast on device. Must call :meth:`cache_mtc` first.
        """
        if self._mtc_LD_cached is None:
            raise RuntimeError(
                "reset_s_t_nu_from_mtc: cache_mtc must be called first"
            )
        # Reuse the existing numba kernel — broadcasts (L, D) into
        # (S, T, L, D) in parallel.
        from arealmshbm.step2_em_outer import reset_s_t_nu_from_mtc_STLD
        reset_s_t_nu_from_mtc_STLD(self._s_t_nu_A_STLD, self._mtc_LD_cached)

    def sync_to_host(
        self,
        Params: Dict[str, np.ndarray],
        fields: Optional[Iterable[str]] = None,
    ) -> None:
        """No-op on CPU: ``Params`` arrays are aliased to Session buffers.

        Present for API parity with :class:`Step2EmIterSessionCUDA`. The
        GPU mirror does D2H copies here.
        """
        # The only host-side update needed here is cost_em (the CPU
        # vmf_clustering_batch already sets it directly, but allow
        # callers to request it explicitly for symmetry).
        if fields is None:
            return
        for f in fields:
            if f == "cost_em":
                Params["cost_em"] = self._cost_S.copy()
            # s_t_nu / s_lambda / theta — already aliased; no-op.
            # kappa — already updated in run_iter; no-op.

    # ─────────────────────────────────────────────────────────────────
    # Cross-call refresh — reused Session across intra-EM iters.
    # ─────────────────────────────────────────────────────────────────
    def refresh_s_psi_sigma(self, s_psi: np.ndarray, sigma: np.ndarray) -> None:
        """Re-stage ``s_psi`` ((S, L, D) internal layout) and ``sigma``."""
        if s_psi.shape != (self.S, self.L, self.D):
            raise ValueError(
                f"s_psi shape {s_psi.shape} != ({self.S}, {self.L}, {self.D})"
            )
        if s_psi is not self._s_psi_SLD:
            np.copyto(self._s_psi_SLD,
                      np.ascontiguousarray(s_psi, dtype=np.float32))
        sigma_flat = np.asarray(sigma).reshape(-1)
        if sigma_flat.size != self.L:
            raise ValueError(
                f"sigma size {sigma_flat.size} != L={self.L}"
            )
        np.copyto(self._sigma_L, sigma_flat.astype(np.float32, copy=False))

    # ─────────────────────────────────────────────────────────────────
    # Public API — one outer-EM-iter master call.
    # ─────────────────────────────────────────────────────────────────
    def run_iter(
        self,
        Params: Dict[str, np.ndarray],
    ) -> Tuple[int, float]:
        """Run ONE outer EM iter. Updates Params in place.

        Contract — :meth:`upload_initial_state` must have aliased
        Params["s_t_nu"/"s_lambda"/"theta"] to the Session-owned
        scratch buffers; this invariant is verified by identity here.
        A user who reassigns one of these slots between calls breaks
        the aliasing and the master kernel would silently read stale
        bytes — the check fails loudly instead.

        The identity check also encodes the C-contig + fp32 + shape
        invariants: the Session-owned scratch is allocated via
        ``np.empty(shape, dtype=np.float32)`` in ``__init__`` and never
        mutates its layout, so ``Params[key] is _scratch`` implies the
        master kernel sees a C-contig fp32 buffer of the expected shape.
        """
        if Params["s_t_nu"] is not self._s_t_nu_A_STLD:
            raise ValueError(
                "Step2EmIterSession.run_iter: Params['s_t_nu'] is not "
                "the Session-owned buffer; call upload_initial_state "
                "first and don't reassign Params['s_t_nu'] between "
                "run_iter calls."
            )
        if Params["s_lambda"] is not self._s_lambda_SNL_f32:
            raise ValueError(
                "Step2EmIterSession.run_iter: Params['s_lambda'] is not "
                "the Session-owned buffer; same contract as 's_t_nu'."
            )
        if Params["theta"] is not self._theta_NL:
            raise ValueError(
                "Step2EmIterSession.run_iter: Params['theta'] is not "
                "the Session-owned buffer; same contract as 's_t_nu'."
            )
        kappa_flat = np.asarray(Params["kappa"]).reshape(-1)
        if kappa_flat.size != self.L:
            raise ValueError(
                f"kappa size {kappa_flat.size} != L={self.L}"
            )
        kappa_init_f64 = float(kappa_flat[0])
        if not np.all(kappa_flat == kappa_flat[0]):
            raise ValueError(
                "kappa must be uniform across L (M-step kappa-as-scalar "
                "invariant). Got non-uniform input."
            )

        # Closures over the bold/grad loaders. Numba kernels see the
        # per-subject scratch slot directly; the Python orchestrator
        # calls these between phases.
        bold_loader = self._bold_loader
        grad_loader = self._grad_loader

        def _load_bold_into(s_1: int, out: np.ndarray) -> None:
            bold_loader.load_into(s_1, out)

        if grad_loader is None:
            def _load_grad_into(s_1: int, out: np.ndarray) -> None:
                # gMSHBM-only — never called when has_spatial=0. Stub
                # exists so the kernel signature is uniform.
                pass
        else:
            def _load_grad_into(s_1: int, out: np.ndarray) -> None:
                grad_loader.load_into(s_1, out)

        # Invoke the streaming master kernel.
        m_iters, kappa_new = em_iter_master_kernel_streaming(
            self._bold_scratch_NTD,
            self._grad_scratch_TND,
            _load_bold_into,
            _load_grad_into,
            self._boundary_mask_NL,
            np.int64(self.has_spatial),
            self._s_lambda_SNL_f32,
            self._s_t_nu_A_STLD, self._s_t_nu_B_STLD,
            self._theta_NL,
            self._cost_S,
            self._s_lambda_NL_f64_scratch,
            self._X_dot_sl_STLD, self._sigma_psi_SLD,
            self._log_connect_NL_buf, self._log_vmf_NL_buf,
            self._tmp_idx_NL, self._flag_ST_acc,
            self._converge_STL,
            self._scratch_u_LD, self._scratch_uupd_DL, self._scratch_sumlam_L,
            self._scratch_grad_sq, self._scratch_u_sq, self._scratch_cross_NL,
            self._s_t_nu_TDL_scratch,
            self._log_lambda_scratch_NL,
            self._n_alive_count_N,
            self._sigma_L, self._s_psi_SLD,
            np.int64(self.dim),
            np.float64(kappa_init_f64),
            np.float64(self.ini_val),
            np.float64(self.beta_internal),
            np.float32(self.eps_m_step),
            np.int64(self.max_iter_m),
            np.int64(self.n_lh),
            np.int64(self.L_lh),
        )

        # No stage-out needed — Params["s_t_nu" / "s_lambda" / "theta"]
        # ARE the Session-owned buffers (alias invariant checked above),
        # so the kernel's in-place writes already populated Params.
        Params["kappa"] = np.full((1, self.L), kappa_new, dtype=np.float32)
        cost_total = float(self._cost_S.sum())
        return int(m_iters), cost_total

    def get_cost_S(self) -> np.ndarray:
        """Return the per-subject cost from the last ``run_iter`` (copy)."""
        return self._cost_S.copy()
