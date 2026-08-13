"""step2_em_iter_master — fused EM-iter master kernel.

Pulls every leaf in the step-2 outer-EM iter body
(``CBIG_ArealMSHBM_cdgMSHBM_estimate_group_priors_sequential.m`` lines
425-676) into a single numba ``@njit`` pipeline, calling per-phase
``@njit`` sub-kernels that numba inlines into one fused master:

* Phase A — Precompute ``X_dot_sl[s, t, l, d] = Σ_n s_lambda[s, n, l]
  · BOLD[s, n, t, d]`` via numba ``np.dot`` (BLAS sgemm dispatch).
  ONE big sgemm per subject (``(L, N) × (N, T·D) → (L, T·D)``, then
  transposed to (T, L, D) into ``X_dot_sl_STLD[s]``). Same FLOP count
  as the pre-fusion S·T smaller (L, N)×(N, D) sgemms, hoisted out of
  the M-step inner while-loop (s_lambda is constant within the inner
  loop). Per-s collapse is required by the NTD BOLD layout — numba's
  ``np.dot`` does not dispatch 2-D strided slices through cblas_sgemm.
* Phase B — Multi-subject M-step inner while-loop.
  Sgemm-free given X_dot_sl: ``kappa_sum`` is a fp64 reduction over
  (S, T, L, D); per-(s, t, l) update is unit-stride d. Adapted from
  the single-subject (T, L, D) layout of an earlier fused-kernel
  prototype to the multi-subject (S, T, L, D) layout.
* Phase C — Per-subject spatial_connect prior (gMSHBM only).
  Block-diagonal gemms + assemble in numba (``np.dot`` × 4 per session
  + ``-||grad||² + 2·g·u - ||u||²`` numba loop). gradients keep the
  legacy (S, T, N, D_grad) TND layout — the per-t (N, D_grad) C-contig
  slice is the natural sgemm input.
* Phase D — Per-subject fused E-step (NTD variant). Big sgemm
  ``(N, T·D) × (T·D, L) → (N, L)`` folds both the per-t loop AND the
  d-contraction in a single BLAS call; ``log_vmf[n, l] = κ · lv_sum[n,
  l] + count_alive[n] · cdln_val``, where ``count_alive[n]`` comes
  from a per-(t, n) nonzero-BOLD scan (the medial-wall test that the
  legacy per-t ``alive`` flag was implementing in disguise).
* Phase E — Phase E.1 per-subject normalize (``s_lambda *=
  boundary_mask`` → row-normalize → zero degenerate rows, fp64 scratch
  → fp32 storage on cast), fused inside the per-subject loop with
  Phase C+D. Phase E.2 (``theta = mean(s_lambda, axis=S)``) runs once
  after the loop.
* Convergence — per-subject cost rel-diff (Python-side, between master
  calls — one master call covers ONE outer EM iter).

Layout invariant (callers MUST honor at the Session boundary):

* BOLD per subject : ``(N, T, D)`` fp32 C-contig — NTD layout,
  matching step 0 / 1 / 3 across the fork. Streamed: decoded by the
  ``SubjectProfileLoader.load_into`` callback into one re-used scratch
  slot per outer-EM iter (peak RAM independent of S).
* gradient per sub : ``(T, N, D_grad)`` fp32 C-contig (gMSHBM only —
  Phase C's per-t block-diagonal sgemms read the per-t (N, D_grad)
  slice directly). Same streamed pattern as BOLD via
  ``SubjectGradientLoader.load_into``.
* s_t_nu           : ``(S, T, L, D)`` fp32 C-contig (UNIFIED layout —
  M-step indexes (s, t, l, d) unit-stride; E-step internally
  transposes to (T, D, L) into a session-owned 3.76 MB scratch buffer
  for the big sgemm's right operand). No transpose between M-step and
  E-step within an iter.
* s_lambda         : ``(S, N, L)`` fp32 storage. The E-step's softmax
  exp intermediate keeps fp64 inside a per-subject (N, L) scratch
  through Phase D → Phase E.1; cast to fp32 storage happens once at
  the end of Phase E.1 per subject.
* s_psi            : ``(S, L, D)`` fp32 C-contig (sigma * s_psi
  precomputed once per Session prepare; unit-stride d).
* sigma            : ``(L,)`` fp32.
* theta            : ``(N, L)`` fp32.
* boundary_mask    : ``(N, L)`` fp32.

dtype contract (single fixed production path — no precision interface):

* Storage: fp32 throughout for BOLD, s_t_nu, s_psi, theta, sigma,
  s_lambda.
* fp64 precision for the E-step's β=5000 softmax exp() is preserved
  inside a per-subject (N, L) scratch through Phase D → Phase E.1
  (see _phase_e1_normalize_per_subject_kernel).
* Reduction accumulators: mixed fp32 / fp64 per the per-site policy
  table in ``docs/step2_em_iter_master_kernel.md §2``. Three sites
  MUST stay fp64 (M1 kappa_sum, E row-normalize, D softmax exp at
  β=5000 subnormal); the rest are fp32-safe.
* kappa scalar: fp64 (Bessel territory). Cast to fp32 inside the
  M-step inner kernel once per iter_m.

Public API:

* :class:`Step2EmIterSession` — owns scratch buffers + loader handles,
  exposes ``run_iter(...)`` for one outer-EM iter.
* :func:`em_iter_master_kernel_streaming` — the per-subject-streaming
  master (Python orchestrator + numba sub-kernels).
* :func:`warmup_em_iter_master` — JIT-compile every kernel once at
  process start.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from ._kernels import (
    em_iter_master_kernel_streaming,
    _mstep_inner_loop_master_step2,
    _spatial_connect_per_subject_numba,
    warmup as warmup_em_iter_master,
)
from .session import Step2EmIterSession

__all__ = [
    "em_iter_master_kernel_streaming",
    "_mstep_inner_loop_master_step2",
    "_spatial_connect_per_subject_numba",
    "warmup_em_iter_master",
    "Step2EmIterSession",
]


def _import_gpu():
    """Lazy import of the CuPy-backed Session + kernels.

    Kept off the module-level import path so ``import
    arealmshbm.step2_em_iter_master`` works on CPU-only machines
    where ``import cupy`` would fail.
    """
    from ._kernels_gpu import (
        em_iter_master_kernel_streaming_cupy,
        warmup_step2_gpu,
    )
    from .session_gpu import Step2EmIterSessionCUDA
    return (
        em_iter_master_kernel_streaming_cupy,
        warmup_step2_gpu,
        Step2EmIterSessionCUDA,
    )


def __getattr__(name):
    """PEP 562 lazy attribute access for GPU symbols.

    ``from arealmshbm.step2_em_iter_master import Step2EmIterSessionCUDA``
    triggers a cupy import; on a CPU-only host the user sees the cupy
    ImportError, which is the correct failure mode.
    """
    if name in {
        "em_iter_master_kernel_streaming_cupy",
        "warmup_step2_gpu",
        "Step2EmIterSessionCUDA",
    }:
        eim_cupy, warm, sess_cuda = _import_gpu()
        mapping = {
            "em_iter_master_kernel_streaming_cupy": eim_cupy,
            "warmup_step2_gpu": warm,
            "Step2EmIterSessionCUDA": sess_cuda,
        }
        return mapping[name]
    raise AttributeError(
        f"module 'arealmshbm.step2_em_iter_master' has no attribute {name!r}"
    )
