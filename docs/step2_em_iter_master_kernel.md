# Step-2 EM-iter master kernel

The step-2 EM-iter loop body is one numba nopython master kernel:
`arealmshbm.step2_em_iter_master.em_iter_master_kernel_streaming`.
One call executes one outer EM iter — Phase A (precompute `X_dot_sl`)
→ Phase B (M-step inner while-loop) → Phase C (per-subject
`spatial_connect`, gMSHBM only) → Phase D (per-subject fused E-step)
→ Phase E (`normalize` + `theta`). Convergence (per-subject cost
rel-diff) is checked Python-side between master calls.

`Step2EmIterSession.run_iter` is the production wrapper — owns every
scratch buffer, stages caller-facing arrays at the external/internal
layout boundary, and dispatches the master.

This document covers:

1. [Layout invariant](#1-layout-invariant) — what callers MUST honour.
2. [Precision contract](#2-precision-contract) — dtypes and per-site
   accumulators.
3. [spatial_connect 0/0 rule](#3-spatial_connect-00-rule) — load-bearing
   numpy `error_model` choice.


## 1. Layout invariant

The kernel ASSUMES every tensor is C-contig fp32 (or fp64 where noted)
in the unified `(S, T, L, D)` family. Callers MUST repack at the
boundary if they hold a different layout — `Step2EmIterSession` handles
this once at construction (BOLD, gradients, `s_psi`) and once per
`run_iter` call (`s_t_nu`, `s_lambda`, `theta`, `kappa`).

| Tensor          | External (legacy CBIG shape)     | Internal (this kernel)        | Notes                                    |
|-----------------|----------------------------------|-------------------------------|------------------------------------------|
| BOLD            | `(N, D, T)` per subject (list)   | `(S, T, N, D)` fp32 C-contig | one stacked transpose at Session ctor    |
| gradient        | `(N, D_grad, T)` per subject     | `(S, T, N, D_grad)` fp32     | gMSHBM only; same transpose at ctor      |
| `s_t_nu`        | `(D, L, T, S)`                   | `(S, T, L, D)` fp32          | UNIFIED — M-step + E-step share this    |
| `s_lambda`      | `(N, L, S)`                      | `(S, N, L)` fp32             | fp32 storage; fp64 only on the per-subject Phase D / E.1 scratch |
| `s_psi`         | `(D, L, S)`                      | `(S, L, D)` fp32             | precomputed → `sigma_psi` once / iter    |
| `sigma`         | `(1, L)` or `(L,)`               | `(L,)` fp32                  | unchanged                                |
| `theta`         | `(N, L)`                         | `(N, L)` fp32                | unchanged                                |
| `boundary_mask` | `(N, L)`                         | `(N, L)` fp32                | unchanged                                |
| `kappa`         | `(1, L)` or `(L,)`               | scalar fp64                  | uniform-κ invariant (M-step ψ-broadcast) |

### Why `(S, T, L, D)` and not `(S, T, D, L)`?

The M-step inner kernel's per-`(s, t, l)` update loops over `d` as the
innermost axis (col-norm + cosine). Unit-stride access on the d axis
lets numba's SIMD autovectoriser emit packed FMAs. A d-strided inner
loop loses ~3× throughput.

### How does the E-step keep `(S, T, L, D)` without a transpose?

The fused E-step's per-session sgemm needs `(N, D) @ (D, L) → (N, L)`.
With `s_t_nu_TLD[t]` shape `(L, D)`, the kernel slots `s_t_nu_TLD[t].T`
as the right operand — numba's `np.dot` dispatches the `.T` view to
cblas_sgemm with the transB flag (no materialised transpose, no copy).
The strided sgemm path has ~5 % overhead vs C-contig both sides on
fp32 at this size.

**Result: zero per-iter transpose copies.** `s_t_nu` is owned in
`(S, T, L, D)` for the entire EM-iter loop; both M-step and E-step
read it with their preferred indexing without touching a single byte
of duplicated layout.

### Contiguity contract for the master kernel

If callers bypass `Step2EmIterSession` and call
`em_iter_master_kernel_streaming` directly, **every array argument
MUST be C-contig** at the dtype indicated above. The kernel does NOT
validate contiguity (numba's `np.dot` would silently fall through to a
slower generic path on non-contig inputs).


## 2. Precision contract

### Storage dtypes (fixed)

| Tensor          | Storage | Why                                          |
|-----------------|---------|----------------------------------------------|
| BOLD, gradients | fp32    | matches pipeline-wide convention; memory-bound |
| `s_t_nu`, `s_psi`, `sigma`, `theta` | fp32 | matches pipeline |
| `boundary_mask`  | fp32    | 0/1 mask                                     |
| `s_lambda`       | **fp32** | Storage is fp32. The E-step softmax's subnormal-tier values (~1e-242 at β=5000) require fp64 precision; this is preserved on a per-subject (N, L) fp64 scratch buffer inside Phase D + Phase E.1 that gets row-normalized then cast down to the fp32 storage slot. fp32 storage halves the s_lambda footprint (52 GB → 26 GB at S=200). |
| `kappa` scalar   | fp64    | Bessel territory; cast to fp32 once per iter_m |

### Reduction-accumulator dtypes (per-site policy)

The master kernel uses a mixed accumulator policy — each site was
chosen independently for the lowest dtype that preserves end-to-end
correctness within the 1e-4 EM-convergence bar.

| Phase | Reduction                                  | Acc dtype | Rationale                                                          |
|-------|--------------------------------------------|-----------|---------------------------------------------------------------------|
| B     | `Σ s_t_nu · X_dot_sl` (kappa_sum)          | **fp64**  | fp32 produces 5e-3 κ rel-diff on the 5.6 M-term reduction.          |
| B     | `Σ s_lambda` (denom)                       | **fp64**  | fp32 serial acc stagnates past `2^23` (`S·N ≈ 1.3e7` at `S=200` fsa6) — running sum stops growing once ULP exceeds the `~1/L` per-element value; biases denom by ≥8%, propagates 1:1 into `rbar = kappa_sum / denom` and `κ`. The original "<1e-9" cost-rel claim was measured at `S=2–3`. |
| B     | col-norm `Σ col²`                          | fp32      | safe — cost rel ≤ 1.6e-9.                                           |
| B     | cosine drift `Σ new · old`                 | fp32      | bit-identical to fp64 at D=1175.                                    |
| D     | log_vmf pre-softmax composition            | fp32      | log_vmf + log(θ) + β·log_connect all in fp32; cast to fp64 only at exp(). |
| D     | E-step softmax `exp(log_lambda − row_max)` | **fp64**  | **Mandatory** for β=5000 subnormal safety.                          |
| D     | per-subject cost                            | fp32      | only feeds the EM convergence test (1e-4 bar).                      |
| E     | row-normalize `rs`                          | **fp64**  | fp32 catastrophically zeros subnormal-positive rows — **DO NOT downgrade**. |
| E     | theta = mean over S                         | fp32      | output is fp32 anyway.                                              |

Four sites must stay fp64 (kappa_sum, denom, softmax exp,
row-normalise); the other five are fp32-safe. The CPU master is memory-bound, so fp32
SIMD doesn't accelerate it — the policy is shaped to be GPU-port-ready
(RTX 5090 has a 1:64 fp64:fp32 throughput ratio, so the policy directly
sizes the cost of a future GPU port).

The CuPy GPU port (`step2_em_iter_master/_kernels_gpu.py`) honors this
precision contract verbatim. CuPy's parallel-tree fp32 reductions
differ from numba's serial accumulators at fp32-ULP; bit equality
across backends is NOT a goal. The numerical-equivalence spec is
5e-3 max-rel-diff after 1 outer-EM iter on the random-init test cohort
(see `arealmshbm/step2_em_iter_master/tests/test_numerical_match_gpu.py`);
at production scale + many iters, gMSHBM is additionally chaotic
because β=5000 amplifies any tiny `log_connect` drift into different
argmax flips at parcel boundaries.


## 3. spatial_connect 0/0 rule

Inside `_spatial_connect_per_subject_numba`, the per-parcel
`u[l, d] = u_update[d, l] / sum_lambda[l]` divide must use
`inv = 1.0 / sl` and let numpy's `error_model='numpy'` propagate
`0/0 → NaN`. The NaN cascades through `u_sq → vmf → softmax →
dead-column-zero`, correctly **zeroing the entire empty-parcel column**
in the final `s_lambda`.

A "safety guard" of the form `inv = sl != 0 ? 1/sl : 0` is
**incorrect**: with β=5000 it produces finite ~-0.02 to -0.1
`log_connect` values at empty-parcel cells, which scale into 200-unit
log_lambda contributions and flip argmax at parcel boundaries. The
NaN-propagation form is the only fix — `error_model='numpy'` on the
relevant `@njit` decorator is load-bearing.

dMSHBM is unaffected (β=0, no log_connect term).


## See also

* `arealmshbm.step2_em_iter_master._kernels` — kernel source.
* `arealmshbm.step2_em_iter_master.session.Step2EmIterSession` —
  the wrapper that owns scratch and stages layout.
* [`step2_flow_and_subgraphs.md`](step2_flow_and_subgraphs.md) — where
  this kernel sits in the larger step-2 call graph.
