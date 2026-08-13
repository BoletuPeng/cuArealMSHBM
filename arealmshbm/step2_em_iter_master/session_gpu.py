"""session_gpu.py — full-device GPU wrapper around the EM-iter master.

Mirrors :class:`Step2EmIterSession`'s public API exactly so
``vmf_clustering_batch`` can hold either Session class via duck typing.

Streaming model: in pure ``stream`` mode the per-subject BOLD/grad are
H2D'd into reused device scratch slots (pinned-host staging path), so
peak device memory stays at one subject's slab regardless of cohort
size S, and ``s_lambda_SNL`` is the only multi-subject device buffer
(~393 MB at fsa6/S=3/L=400; 26 GB at S=200 — out of scope for v1).

Production override: the default ``eager_bitpacked`` + gMSHBM path also
holds cohort-wide caches that scale with S — ``_grad_cache_SND_dev``
(S·N·D_grad fp32, always allocated for gMSHBM) and, in eager mode,
``_bold_cache_SNTD_bp_dev`` (S·N·T·⌈D/8⌉ uint8). See ``_kernels_gpu.py``
for the full device budget.

State invariant: between :meth:`run_iter` calls, the device buffers
hold the canonical state. The CPU Params dict is **stale** during the
inner EM-iter loop. The caller (``vmf_clustering_batch``) calls
:meth:`sync_to_host` once at the end of each batch to D2H s_t_nu /
cost_em / theta into Params; outer-EM closure leaves (L16/L17/L18) then
consume the host-side values, and :meth:`refresh_s_psi_sigma` H2Ds the
updated s_psi/sigma back at the start of the next batch.

Initial state: :meth:`upload_initial_state` H2Ds s_lambda / s_t_nu /
theta / kappa exactly once, when ``Step2Pipeline.run_em`` builds the
Session. This is the only ~530 MB H2D in the pipeline; every subsequent
crossing is bounded to ~17 MB (s_t_nu D2H + s_psi/sigma H2D per outer
EM iter).

Reset operations between intra/inter EM iters happen on-device via
:meth:`reset_s_t_nu_from_mtc` and :meth:`reset_kappa_to_ini_val`; the
caller does not have to re-upload the broadcasted mtc each time.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cupy as cp
import numpy as np

from arealmshbm.step2_io import (
    SubjectGradientLoader,
    SubjectProfileLoader,
)

from ._kernels_gpu import (
    em_iter_master_kernel_streaming_cupy,
    _widen_normalize_bold_bitpacked_to_f32_cupy,
)
from .session import _validate_initial_state_params, _validate_session_ctor_args


class Step2EmIterSessionCUDA:
    """Holds device-resident scratch + caches for the EM-iter master kernel.

    Construct ONCE per ``Step2Pipeline.run_em`` call (lives across all
    I·J·K outer-EM iters). Same lifecycle as the CPU
    :class:`Step2EmIterSession`.

    The CPU Session caches use ``__slots__``; we don't bother on the GPU
    class — there are ~30 attributes and CuPy ndarrays already dominate
    the per-Session footprint.
    """

    def __init__(
        self,
        bold_loader: SubjectProfileLoader,
        grad_loader: Optional[SubjectGradientLoader],
        num_sub: int,
        N: int,
        T: int,
        D: int,
        D_grad: int,
        boundary_mask: np.ndarray,
        s_psi: np.ndarray,
        sigma: np.ndarray,
        *,
        mode: str,
        dim: int,
        num_clusters: int,
        ini_val: float,
        beta_internal: float,
        n_lh: int,
        eps_m_step: float = 1e-4,
        max_iter_m: int = 50,
        bold_cache_mode: str = "auto",
        bold_cache_safety_margin_gb: float = 4.0,
    ):
        _validate_session_ctor_args(
            cls_name="Step2EmIterSessionCUDA",
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

        # ── Loaders + BOLD cache mode dispatch ──
        # 'auto' (default): prefer eager_bitpacked → stream. Picks the
        #                   first that fits in free device memory minus
        #                   a safety margin.
        # 'eager_bitpacked': (S, N, T, ⌈D/8⌉) uint8 cache; per-iter BOLD
        #                    comes from a device-side bitpacked→fp32
        #                    widen+normalize kernel. ~1 bit/cell on device,
        #                    the ONLY viable eager path at S=200+ cohorts.
        # 'stream': per-iter disk decode + pinned-H2D path. Safe
        #           fallback when the device packed cache won't fit.
        #
        # The legacy 'eager' (unpacked uint8, ~8x larger on device)
        # mode was removed in 2026-06 — bitpacked subsumes it on every
        # axis (same fp32 contract, 8x less device read traffic,
        # 8x smaller cache footprint, ~15% faster kernel wall).
        _valid_modes = ("auto", "eager_bitpacked", "stream")
        if bold_cache_mode not in _valid_modes:
            raise ValueError(
                f"bold_cache_mode must be one of {_valid_modes} "
                f"(got {bold_cache_mode!r})"
            )
        self._bold_loader = bold_loader

        # Always allocate the per-subject fp32 scratch (target of both
        # the streaming H2D and the eager-cache widen+normalize kernel).
        self._bold_scratch_NTD_dev = cp.empty(
            (self.N, self.T, self.D), dtype=cp.float32
        )

        # Cache-size bookkeeping. bitpacked cache = (S, N, T, ⌈D/8⌉).
        self._D_bytes = (self.D + 7) // 8
        cache_bytes_bp = self.S * self.N * self.T * self._D_bytes

        # Decide cache mode. 'auto' picks eager_bitpacked when it fits
        # within the safety margin, otherwise falls through to stream.
        # We remember whether the user explicitly asked for eager_bitpacked
        # (hard contract) vs. auto-resolved into it (best-effort, will
        # silently degrade if the loader can't supply packed bytes).
        _user_requested_eager = (bold_cache_mode == "eager_bitpacked")
        self._bold_cache_mode = bold_cache_mode
        if bold_cache_mode == "auto":
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            margin_bytes = int(bold_cache_safety_margin_gb * (1 << 30))
            if cache_bytes_bp + margin_bytes < free_bytes:
                self._bold_cache_mode = "eager_bitpacked"
            else:
                self._bold_cache_mode = "stream"

        # Initialize cache handle; populate when eager.
        self._bold_cache_SNTD_bp_dev: Optional[cp.ndarray] = None

        if self._bold_cache_mode == "eager_bitpacked":
            # Loader-capability gate. ``auto`` callers (the production
            # default) want best-effort behavior — silently fall through
            # to stream when the loader can't supply packed bytes (e.g.
            # the in-memory test adapter). Explicit ``eager_bitpacked``
            # callers asked for a specific contract; honor it as a hard
            # error so the mismatch is loud.
            if not hasattr(bold_loader, "load_packed_into"):
                if _user_requested_eager:
                    raise TypeError(
                        f"bold_cache_mode='eager_bitpacked' requires a "
                        f"loader that implements ``load_packed_into`` "
                        f"(got {type(bold_loader).__name__}). Pass a "
                        f"SubjectProfileLoader, or set "
                        f"bold_cache_mode='stream'."
                    )
                self._bold_cache_mode = "stream"
                import warnings
                warnings.warn(
                    f"Step2EmIterSessionCUDA: bold_cache_mode='auto' "
                    f"resolved to eager_bitpacked but loader "
                    f"({type(bold_loader).__name__}) has no "
                    f"``load_packed_into`` — falling back to stream.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self._bold_cache_mode == "eager_bitpacked":
            # Allocate the device packed cache. If the device alloc
            # OOMs (free memory probe vs actual alloc disagree), fall
            # back to streaming. We warn unconditionally — even an
            # explicit ``eager_bitpacked`` caller wants to know their
            # contract degraded under runtime memory pressure, since
            # OOM isn't predictable from caller-visible state. (We
            # don't raise on explicit-OOM the way we do on missing
            # ``load_packed_into`` — a missing method is a config
            # error the caller can fix; OOM is a runtime resource
            # constraint where "warn + degrade" is more useful than
            # "crash".)
            try:
                self._bold_cache_SNTD_bp_dev = cp.empty(
                    (self.S, self.N, self.T, self._D_bytes), dtype=cp.uint8
                )
            except cp.cuda.memory.OutOfMemoryError as e:
                self._bold_cache_SNTD_bp_dev = None
                self._bold_cache_mode = "stream"
                import warnings
                warnings.warn(
                    f"Step2EmIterSessionCUDA: eager_bitpacked device "
                    f"cache allocation OOM'd "
                    f"(S={self.S} N={self.N} T={self.T} D_bytes={self._D_bytes} "
                    f"= {self.S * self.N * self.T * self._D_bytes / (1 << 20):.1f} MB"
                    f"); falling back to stream. ({e})",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Populate the cache via pinned-host staging. The loader's
            # ``load_packed_into`` reads bitpacked bytes straight from
            # the on-disk .b2nd — no host fp32 round-trip.
            if self._bold_cache_mode == "eager_bitpacked":
                try:
                    stage_host = _alloc_pinned(
                        (self.N, self.T, self._D_bytes), np.uint8
                    )
                    try:
                        # CuPy 13.x: .set() is async on the current
                        # stream — sync before reusing host buffer.
                        for s in range(1, self.S + 1):
                            bold_loader.load_packed_into(s, stage_host)
                            self._bold_cache_SNTD_bp_dev[s - 1].set(stage_host)
                            cp.cuda.get_current_stream().synchronize()
                    finally:
                        del stage_host
                except (NotImplementedError, ValueError) as e:
                    # load_packed_into unavailable (in-memory adapter
                    # path) or the dataset isn't 0/1 binary. Drop the
                    # cache and fall back to streaming.
                    self._bold_cache_SNTD_bp_dev = None
                    self._bold_cache_mode = "stream"
                    import warnings
                    warnings.warn(
                        f"Step2EmIterSessionCUDA: eager_bitpacked BOLD "
                        f"cache disabled — {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if self._bold_cache_mode == "stream":
            # Streaming path needs the pinned-host fp32 staging buffer
            # the loader's ``load_into`` writes through.
            self._bold_scratch_NTD_host = _alloc_pinned(
                (self.N, self.T, self.D), np.float32
            )
        else:
            # Eager path: no pinned-host fp32 buffer (we never decode
            # fp32 host-side again).
            self._bold_scratch_NTD_host = None

        if self.has_spatial == 1:
            # grad_loader presence + dims already verified by
            # _validate_session_ctor_args above.
            self._grad_loader = grad_loader

            # ── Eager grad cache: (S, N, D_grad) fp32 device-resident ──
            # gradient is session-invariant in this fork — load_into
            # returns (T, N, D_grad) where all T entries hold the same
            # (N, D_grad). We keep just one (N, D_grad) per sub on device,
            # and the master kernel's spatial_connect hoists the T-fold
            # accumulation. Memory: S·N·D_grad·4 B = 1.31 GB at S=40 /
            # fsa6 / D_grad=100; ~6.55 GB at S=200.
            #
            # Population path: load via the existing TND interface (one-
            # time host pinned buf reused across S), slice [0] to get the
            # session-invariant (N, D_grad), H2D into per-sub cache slot.
            # Cache pump — ASYNC H2D race fix below: CuPy's ``.set()``
            # queues a cudaMemcpyAsync from the pinned host buffer on
            # the current stream; if the next ``load_into`` overwrites
            # the host buffer before the queued copy has actually DMAed
            # the bytes off it, the device sees stale data. We sync the
            # current stream after each ``.set()`` to guarantee the host
            # buffer can be reused for the next sub. The cost is negligible
            # (one device sync per sub × S; the H2D itself dominates wall).
            #
            # NOTE: this is a once-per-Session pump (NOT a per-iter
            # path), so the per-sync overhead is amortized over the
            # full EM run.
            self._grad_cache_SND_dev = cp.empty(
                (self.S, self.N, self.D_grad), dtype=cp.float32
            )
            host_scratch_TND = _alloc_pinned(
                (self.T, self.N, self.D_grad), np.float32
            )
            for s_idx in range(self.S):
                grad_loader.load_into(s_idx + 1, host_scratch_TND)
                # All T slices identical (replicated by the loader). Take
                # [0] to get the (N, D_grad) session-invariant block.
                self._grad_cache_SND_dev[s_idx].set(host_scratch_TND[0])
                cp.cuda.get_current_stream().synchronize()
            del host_scratch_TND
        else:
            self._grad_loader = None
            # dMSHBM mode — no spatial term, master kernel never reads
            # grad_cache. Tiny (1, 1, 1) dummy keeps the signature happy
            # without holding meaningful device memory.
            self._grad_cache_SND_dev = cp.zeros(
                (1, 1, 1), dtype=cp.float32
            )

        # ── s_psi / sigma / boundary_mask H2D ──
        # Shapes already validated by _validate_session_ctor_args.
        self._s_psi_SLD_dev = cp.asarray(
            np.ascontiguousarray(s_psi, dtype=np.float32)
        )
        sigma_flat = np.asarray(sigma).reshape(-1)
        self._sigma_L_dev = cp.asarray(
            np.ascontiguousarray(sigma_flat, dtype=np.float32)
        )
        bm = np.ascontiguousarray(boundary_mask, dtype=np.float32)
        self._boundary_mask_NL_dev = cp.asarray(bm)

        # ── Per-iter mutable internal-layout buffers ──
        # Single in-place s_t_nu state buffer — the M-step renormalizes in
        # place (see _mstep_inner_loop_step2_cupy's factored-cosine path),
        # so there is no longer a persistent ping-pong B buffer. Saves one
        # full (S, T, L, D) fp32 slab of device residency (~0.43 GB at
        # S=40/T=6/L=400, ~2.1 GB at S=200). The s_t_nu update is
        # bit-identical to the old two-buffer form; only the convergence
        # cosine is reformulated (~1 fp32 ULP, within the CPU↔GPU parity
        # bar).
        self._s_t_nu_STLD_dev = cp.empty(
            (self.S, self.T, self.L, self.D), dtype=cp.float32
        )
        self._s_lambda_SNL_f32_dev = cp.empty(
            (self.S, self.N, self.L), dtype=cp.float32
        )
        # GPU softmax + E.1 normalize use fp32 throughout — the prior
        # fp64 contract was inherited from the CPU master (which has a
        # documented "catastrophic subnormal flushing" cliff at β=5000
        # on x86 fp32). On GPU, the softmax shift-by-rmax already
        # bounds every cell at exp(0)=1 max and exp(-large)→0 min;
        # fp32 underflow at ~e^-87 ≈ 1e-38 is functionally equivalent
        # to fp64 underflow at ~e^-708 in the argmax+normalize chain
        # (sub-1e-7 contributions vanish under fp32 sum-by-fp32-max
        # arithmetic anyway, both at fp32 and fp64).
        # Empirically verified: cost trajectory matches the fp64 path
        # within 1e-5 rel through 10 outer iters on the S=40 reference cohort.
        self._s_lambda_NL_f64_scratch_dev = cp.empty(
            (self.N, self.L), dtype=cp.float32
        )
        self._theta_NL_dev = cp.empty((self.N, self.L), dtype=cp.float32)

        # Per-iter scratch.
        self._X_dot_sl_STLD_dev = cp.empty(
            (self.S, self.T, self.L, self.D), dtype=cp.float32
        )
        self._sigma_psi_SLD_dev = cp.empty(
            (self.S, self.L, self.D), dtype=cp.float32
        )
        self._log_connect_NL_buf_dev = cp.empty(
            (self.N, self.L), dtype=cp.float32
        )
        self._log_vmf_NL_buf_dev = cp.empty(
            (self.N, self.L), dtype=cp.float32
        )
        self._tmp_idx_NL_dev = cp.zeros((self.N, self.L), dtype=cp.bool_)

        # mtc broadcast scratch (L, D) — populated on first reset call.
        # Cached so the broadcast doesn't re-H2D mtc every intra-EM iter.
        self._mtc_LD_dev: Optional[cp.ndarray] = None

        # Output scratch.
        self._cost_S_dev = cp.zeros(self.S, dtype=cp.float64)
        self._cost_S_host = np.zeros(self.S, dtype=np.float64)

        # Last-iter kappa cache (host scalar — needed to feed the next
        # run_iter's M-step init and to write back to Params["kappa"]).
        self._last_kappa_f64: float = float(ini_val)

        # ── BOLD loader callback closing over pinned-host + device scratch ──
        # The loader's load_into() writes to a host ndarray; we then
        # ``set()`` it to the device buffer. Pinned host gives a
        # synchronous H2D bandwidth ~12 GB/s; the (N, T, D) fp32 = 770 MB
        # subject slab uploads in ~64 ms on PCIe 4.0. (gradient has no
        # callback — it lives in the eager (S, N, D_grad) device cache
        # built above.)
        self._bold_loader_call = self._make_bold_into_cb()

    # ─────────────────────────────────────────────────────────────────
    # Streaming loader callbacks — H2D one subject's slab per call.
    # ─────────────────────────────────────────────────────────────────
    def _make_bold_into_cb(self):
        if self._bold_cache_mode == "eager_bitpacked":
            cache_dev = self._bold_cache_SNTD_bp_dev
            D = self.D

            def _cb(s_1: int, dev_out: cp.ndarray) -> None:
                # Device-side bitpacked → fp32 widen + demean + L2
                # row-norm. ~0.85 ms wall on RTX 5090 at fsa6/T=2/D=1175.
                _widen_normalize_bold_bitpacked_to_f32_cupy(
                    cache_dev[s_1 - 1], dev_out, D,
                )

            return _cb

        # Streaming fallback.
        bl = self._bold_loader
        host_buf = self._bold_scratch_NTD_host

        def _cb(s_1: int, dev_out: cp.ndarray) -> None:
            # Disk → pinned host scratch (CPU normalize) → H2D.
            bl.load_into(s_1, host_buf)
            dev_out.set(host_buf)
            # CuPy 13.x: .set() is async on the current stream — sync before reusing host_buf.
            cp.cuda.get_current_stream().synchronize()

        return _cb

    # ─────────────────────────────────────────────────────────────────
    # Cross-call refresh — reused Session across intra-EM iters.
    # ─────────────────────────────────────────────────────────────────
    def refresh_s_psi_sigma(self, s_psi: np.ndarray, sigma: np.ndarray) -> None:
        """Re-stage ``s_psi`` ((S, L, D) internal layout) and ``sigma`` from host.

        Called at the start of each intra-EM iter — L17 may have updated
        s_psi + sigma. H2D ~5.6 MB + 1.6 KB.
        """
        if s_psi.shape != (self.S, self.L, self.D):
            raise ValueError(
                f"s_psi shape {s_psi.shape} != ({self.S}, {self.L}, {self.D})"
            )
        self._s_psi_SLD_dev.set(
            np.ascontiguousarray(s_psi, dtype=np.float32)
        )
        sigma_flat = np.asarray(sigma).reshape(-1)
        if sigma_flat.size != self.L:
            raise ValueError(f"sigma size {sigma_flat.size} != L={self.L}")
        self._sigma_L_dev.set(
            np.ascontiguousarray(sigma_flat, dtype=np.float32)
        )

    # ─────────────────────────────────────────────────────────────────
    # Initial state upload — called once per Step2Pipeline.run_em.
    # ─────────────────────────────────────────────────────────────────
    def upload_initial_state(self, Params: Dict[str, np.ndarray]) -> None:
        """H2D s_lambda / s_t_nu / theta / kappa from the host Params dict.

        Validates shapes / dtypes against the master kernel's contract.
        After this call, the device buffers hold the canonical state;
        subsequent ``run_iter`` calls operate device-side until
        :meth:`sync_to_host` is invoked.
        """
        _validate_initial_state_params(
            Params, self.S, self.T, self.N, self.L, self.D,
        )

        self._s_t_nu_STLD_dev.set(
            np.ascontiguousarray(Params["s_t_nu"], dtype=np.float32)
        )
        self._s_lambda_SNL_f32_dev.set(
            np.ascontiguousarray(Params["s_lambda"], dtype=np.float32)
        )
        self._theta_NL_dev.set(
            np.ascontiguousarray(Params["theta"], dtype=np.float32)
        )

        kappa_flat = np.asarray(Params["kappa"]).reshape(-1)
        if kappa_flat.size != self.L:
            raise ValueError(f"kappa size {kappa_flat.size} != L={self.L}")
        if not np.all(kappa_flat == kappa_flat[0]):
            raise ValueError(
                "kappa must be uniform across L (M-step kappa-as-scalar invariant)"
            )
        self._last_kappa_f64 = float(kappa_flat[0])

    # ─────────────────────────────────────────────────────────────────
    # Device-side resets — between intra-EM / inter-EM iters.
    # ─────────────────────────────────────────────────────────────────
    def cache_mtc(self, mtc_LD: np.ndarray) -> None:
        """Cache the ``mtc`` matrix (L, D) on device so reset_s_t_nu / s_psi
        don't re-H2D it each intra/inter iter."""
        mtc = np.ascontiguousarray(mtc_LD, dtype=np.float32)
        if mtc.shape != (self.L, self.D):
            raise ValueError(
                f"mtc shape {mtc.shape} != ({self.L}, {self.D})"
            )
        if self._mtc_LD_dev is None:
            self._mtc_LD_dev = cp.asarray(mtc)
        else:
            self._mtc_LD_dev.set(mtc)

    def reset_s_t_nu_from_mtc(self) -> None:
        """Device-side broadcast of cached mtc (L, D) into s_t_nu (S, T, L, D).

        Must be called after :meth:`cache_mtc`. Reset happens IN-PLACE on
        the single s_t_nu state buffer; the next ``run_iter`` reads it so
        it picks up the reset.
        """
        if self._mtc_LD_dev is None:
            raise RuntimeError(
                "reset_s_t_nu_from_mtc: cache_mtc must be called first"
            )
        # broadcast (L, D) → (S, T, L, D) via CuPy broadcasting.
        self._s_t_nu_STLD_dev[...] = self._mtc_LD_dev[None, None, :, :]

    # ─────────────────────────────────────────────────────────────────
    # D2H sync — called at the seam of vmf_clustering_batch.
    # ─────────────────────────────────────────────────────────────────
    def sync_to_host(
        self,
        Params: Dict[str, np.ndarray],
        fields: Optional[Sequence[str]] = None,
    ) -> None:
        """D2H named fields back into ``Params``.

        ``fields`` defaults to ``('s_t_nu', 'cost_em', 'kappa', 'theta')``
        — the set vmf_clustering_batch / outer-EM closure needs.
        s_lambda is NOT in the default set (Params_Final.mat doesn't save
        it; intra_em_cost_step2 / inter_subject_var don't read it).
        """
        if fields is None:
            fields = ("s_t_nu", "cost_em", "kappa", "theta")
        for f in fields:
            if f == "s_t_nu":
                Params["s_t_nu"] = cp.asnumpy(self._s_t_nu_STLD_dev)
            elif f == "s_lambda":
                Params["s_lambda"] = cp.asnumpy(self._s_lambda_SNL_f32_dev)
            elif f == "theta":
                Params["theta"] = cp.asnumpy(self._theta_NL_dev)
            elif f == "cost_em":
                # Tiny D2H — single fp64 vector of size S.
                self._cost_S_dev.get(out=self._cost_S_host)
                Params["cost_em"] = self._cost_S_host.copy()
            elif f == "kappa":
                Params["kappa"] = np.full(
                    (1, self.L), self._last_kappa_f64, dtype=np.float32,
                )
            else:
                raise ValueError(f"sync_to_host: unknown field {f!r}")

    # ─────────────────────────────────────────────────────────────────
    # Public API — one outer-EM-iter master call.
    # ─────────────────────────────────────────────────────────────────
    def run_iter(
        self,
        Params: Dict[str, np.ndarray],
    ) -> Tuple[int, float]:
        """Run ONE outer EM iter on device.

        Reads ``Params['kappa'][0]`` as the M-step init scalar (host
        side; no D2H/H2D of bulk state). Writes back ``Params['kappa']``
        with the new uniform scalar. Bulk state stays device-resident;
        the caller must invoke :meth:`sync_to_host` at the end of the
        vmf_clustering_batch loop to D2H s_t_nu / cost_em / theta.

        Returns ``(m_iters, cost_total)`` where ``cost_total`` is the
        sum over subjects of the per-subject cost.
        """
        kappa_flat = np.asarray(Params["kappa"]).reshape(-1)
        if kappa_flat.size != self.L:
            raise ValueError(f"kappa size {kappa_flat.size} != L={self.L}")
        if not np.all(kappa_flat == kappa_flat[0]):
            raise ValueError(
                "kappa must be uniform across L (M-step kappa-as-scalar invariant)"
            )
        kappa_init_f64 = float(kappa_flat[0])

        m_iters, kappa_new = em_iter_master_kernel_streaming_cupy(
            self._bold_scratch_NTD_dev,
            self._bold_loader_call,
            self._grad_cache_SND_dev,
            self._boundary_mask_NL_dev,
            int(self.has_spatial),
            self._s_lambda_SNL_f32_dev,
            self._s_t_nu_STLD_dev,
            self._theta_NL_dev,
            self._cost_S_dev,
            self._s_lambda_NL_f64_scratch_dev,
            self._X_dot_sl_STLD_dev,
            self._sigma_psi_SLD_dev,
            self._log_connect_NL_buf_dev,
            self._log_vmf_NL_buf_dev,
            self._tmp_idx_NL_dev,
            self._sigma_L_dev, self._s_psi_SLD_dev,
            int(self.dim),
            float(kappa_init_f64),
            float(self.ini_val),
            float(self.beta_internal),
            float(self.eps_m_step),
            int(self.max_iter_m),
            int(self.n_lh),
            int(self.L_lh),
        )

        # Cache new kappa (host scalar); update Params["kappa"].
        self._last_kappa_f64 = float(kappa_new)
        Params["kappa"] = np.full(
            (1, self.L), self._last_kappa_f64, dtype=np.float32,
        )

        # Sync the (small) per-subject cost array off device for the
        # caller's convergence test. One PCIe round-trip of S * 8 bytes
        # (24 B at S=3) — trivial.
        self._cost_S_dev.get(out=self._cost_S_host)
        cost_total = float(self._cost_S_host.sum())
        return int(m_iters), cost_total

    def get_cost_S(self) -> np.ndarray:
        """Return per-subject cost from the last ``run_iter`` (host copy)."""
        return self._cost_S_host.copy()


# ─────────────────────────────────────────────────────────────────────
# Pinned-host allocation helper.
# ─────────────────────────────────────────────────────────────────────
def _alloc_pinned(shape: Tuple[int, ...], dtype) -> np.ndarray:
    """Allocate a pinned-host ndarray for fast H2D transfers.

    CuPy exposes pinned memory via ``cupy.cuda.alloc_pinned_memory``.
    Wrapping it as a numpy ndarray lets the disk loaders write directly
    to pinned pages, eliminating the driver staging copy on H2D.
    """
    arr = np.empty(shape, dtype=dtype)
    nbytes = int(arr.nbytes)
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    out = np.frombuffer(mem, dtype=dtype, count=arr.size).reshape(shape)
    return out
