# Step 3 `gpu` backend — design contract

Working spec for the candidate-set (P-layout) step-3 GPU backend
(`backend_step3='gpu'`, the single-subject EM). Everything below is the
contract shared by the kernel modules; the numerical reference is always
the **CPU backend** (`backend='cpu'`, numba kernels under
`arealmshbm/{vmf_clustering,V_lambda,m_step,spatial_priors,
em_stop_criterion,check_connectedness}`). `gpu` requires seed_mesh
fsaverage3 (`ceil(D/8) ≤ 256`); any other seed mesh is rejected at
config / driver validation — select `backend_step3='cpu'` for it.
Every backend requires w > 0 (`w·log θ` is NaN outside supp(θ) at
w = 0).

## 0. Why

Measured 2026-09-03 on sub-001 (fsaverage6, T=6, L=300, D=1175) on the
dense CuPy EM (`gpu_full`, every (N, L) buffer dense, fp32 BOLD):

| stage | wall | calls | per call |
|---|---:|---:|---:|
| e_step_lambda_loop | 2.18 s | 115 λ-iters | 19 ms |
| m_step | 0.50 s | 90 iter_m | 5.5 ms |
| check_connectedness (CPU) | 0.46 s | 108 | 4.3 ms |
| spatial_xyz_prior | 0.41 s | 108 | 3.8 ms |
| em_stop + connect | 0.19 s | 10 | — |
| **EM total** | **4.0–4.3 s** | | |
| load_inputs + session_init | 1.43 s | | |

Structure: `N=81924, L=300`, active rows `M=74947`, `P = nnz(θ) = 251 502`
(≤ 9 per row), `nnz(boundary_mask) = 586 522`. `supp(s_lambda) ⊆ supp(θ)`
holds for the whole EM (proof in `sparse_layout.py`), so every (N, L)
buffer is 99 % zeros. BOLD is binary (bit density ≈ 9.5 %), so
`X · v` is a popcount-style bit-sum, not a 7050-wide fp32 dot.

## 1. Device-resident state

Layout arrays: see `arealmshbm/vmf_clustering/sparse_layout.py`
(`CandidateLayout`). Additional static state:

| name | shape / dtype | notes |
|---|---|---|
| `bold_packed` | `(T, N, D_bytes)` uint8 | **on-disk layout, no transpose**; MW rows zeroed; bit `d` ↔ bit `(d & 7)` of byte `d >> 3` (LSB-first). `D_bytes = ceil(D/8) ≤ MAX_D_BYTES = 256` (`sparse_layout.py`; `acc_bits` gives its 32 lanes `ACC_MAXB = 8` bytes each) — checked by `fetch_packed_bold_TND`, the `acc_bits` wrapper and the `x_dot_sl_bits` wrapper (`m_step_gpu.py`) |
| `row_mean` | `(T, N)` fp32 | `fp32(fp64(popcount) / fp64(D))` |
| `row_inv` | `(T, N)` fp32 | `fp32(1 / sqrt(pop - D·mean²))` computed in fp64 (same as the CPU `_normalize_session_bitpacked_numba`, `data_io/bitpacked_norm.py`); **0 when `pop ∈ {0, D}`** (the `has_zero` gate ⇒ the row is identically zero) |
| `grad` | `(N, Dg)` fp32, `grad_sq (N,)` fp32 | gMSHBM |
| `sphere_unit` | `(N, 3)` fp32 | `compute_unit_sphere_xyz` |
| `lh_nbors`, `rh_nbors` | `(M1, n_hemi)` int32 | 1-indexed, 0 = absent (mesh `vertexNbors`) |
| `log_theta` | `(P,)` fp32 | `log(θ)` |
| `log_theta_cost` | same values (θ > 0 on P) | |

Per-EM state on P (CSR order): `s_lambda` (ping-pong ×2), `V_temp`,
`acc`, `scv`, `sxv`. Per-(t,l,d): `s_t_nu` ping-pong ×2 and `X_dot_sl`
in **`(T, L, D)` layout** (D contiguous), `sigma_psi (L, D)`.

`X[n, t, d] = (bit − row_mean[t, n]) · row_inv[t, n]` is never
materialised.

### 1.1 Cohort constants (host side)

The group prior (`Params_Final.mat` → mu / epsil / sigma / CSR θ) and
the `CandidateLayout` built from it and the two spatial masks depend
only on `cfg.mesh`, `cfg.group_prior_path`, `cfg.spatial_mask_path` and
`cfg.num_clusters` — they are identical for every subject of one
cohort. `Step3SparseCohort` (`step3_pipeline/sparse_inputs.py`) bundles
them with that identity; `load_step3_sparse_cohort` builds one and
`Step3SparseCohort.check(cfg)` rejects a config it was not built from.
The masks themselves are not kept past the layout build.

The step-3 stage pipeline loads one per run, before its subject loop,
and passes it to every `Step3Pipeline` (`sparse_cohort=`); each
subject's LOAD is then only its packed BOLD + gradient. The arrays are
shared **read-only** — the session copies them out (`cp.asarray`,
`np.log(np.asarray(...))`), so no consumer writes into cohort state
(the layout builder's `sort_indices()` runs on fresh row-sliced copies
of the masks, before the cohort object exists). The one array that
escapes unchanged is `prior['mu']`: the session keeps it as its host
`mu` and hands it straight back in every subject's
`Step3Result.Params['mu']`, so a caller must not write into that
either. A single-subject run (or a standalone
`load_step3_sparse_inputs` call) passes no cohort and builds one
in-line, on the same thread pool as the BOLD / gradient reads.

Measured warm on sub-001 (fsaverage6, T=6, L=300): per-subject
`load_step3_sparse_inputs` **163 ms → 84 ms** (prior 64 + masks 25 +
layout 22 ms lifted out; what remains is the BOLD + gradient read), for
a one-off 93 ms cohort load. Ten back-to-back
`Step3Pipeline.load_inputs()`: 1.61 s → 0.85 s.

## 2. Kernel contracts (math identical to the CPU kernels; reduction
order may differ; **no floating-point atomics** — every reduction is a
fixed tree so the backend is run-to-run bit-reproducible)

### 2.1 `acc` (E-step prepare, once per em_iter)

`acc[p] = Σ_t Σ_d X[n,t,d] · s_t_nu[t, l, d]` for `p = (n, l) ∈ P`.
Bit form: with `S[t,l] = Σ_d s_t_nu[t,l,d]` (fp64),
`acc = Σ_t row_inv · (Σ_{d: bit} s_t_nu[t,l,d] − row_mean · S[t,l])`.
The bit term accumulates in **fp32** (per-lane serial, then a fixed warp
tree); the `row_mean · S` correction accumulates in fp64; the two are
subtracted in fp64 and stored fp32. Also `col_zero[l] = all(s_t_nu[:, l, :] == 0)`
(stands in for the CPU's `all(acc[:, l] == 0)`) and
`col_nan[l] = any(isnan(s_t_nu[:, l, :]))`.

### 2.2 E-step λ-iteration (per active row m, candidates ascending)

```
row_sum[j]   = Σ_{k ∈ P(j)} lam[j,k]                    (ascending k)  -- kernel 1
nbr_sum[m]   = Σ_{n<M1, j=nbh[m,n]>0} row_sum[j-1]       (ascending n)
V[m,k]       = nbr_sum[m] − Σ_{n, j>0} lam[j-1, k]       (ascending n; 0 if (j-1,k) ∉ P)
v            = acc·kappa[l] + (0 if col_zero[l] else cdln_T[l])
v           += w·log_theta ; v −= 2c·V ; v += beta[l]·scv ; v += sxv
rmax         = max over P(m) of v            (fp32; NaN never wins a '>' test)
ev           = fp32(exp(fp64(v − rmax))) · bm  ; rsum = Σ ev (ascending)
out          = ev · (1/rsum)  [col_zero ⇒ 0]  if rsum > 0 and rsum == rsum
               else 0 for the whole row
drift       += Σ |out − curr|  (fp64)         (else Σ curr)
V_temp[p]    = V[m,k]
```
Row poisoning: if `col_nan[l]` for any `l` in the **boundary-mask
support** of row m (not just P), the dense CPU kernel produces NaN `rsum`
⇒ whole row 0. Mirror it via `bm_row_ptr/bm_col` (only evaluated when
`any(col_nan)`). `checklam = drift / (N·L)`; converged when
`|checklam_new − checklam_old| ≤ ε`. V_lambda is bit-identical to the
CPU kernel; `exp` is evaluated in fp64 like numba's `math.exp`.

### 2.3 M-step (per em_iter; contract of `MStepSession.run`)

```
X_dot_sl[t,l,d] = Σ_{n ∈ members(l)} sl[n,l] · X[n,t,d]
                = Σ_{n: bit} w_nt − Σ_n w_nt·row_mean[t,n],  w_nt = sl·row_inv[t,n]
denom           = T · Σ_P sl                                  (fp64)
loop iter_m:
  kappa_sum = Σ_{t,l,d} s_t_nu_old · X_dot_sl                (fp64)
  rbar = kappa_sum/denom ; kappa = invad(dim, rbar)           (host, scipy)
  col[d] = fp32(kappa)·X_dot_sl + sigma_psi[l,d] ; cn = sqrt(Σ col² (fp64))
  inv_cn = fp32(1/cn) ; new = col·inv_cn ; cos[t,l] = Σ new·old (fp64) → fp32
  flag_T[t] |= all_l ((1 − cos[t,l]) < ε)      (monotone OR, NaN ⇒ not converged)
  kappa_drift = |kappa_prev − kappa| / kappa_prev
  stop when (all flag_T and kappa_drift < ε) or iter_m > max_iter_m
```
Empty parcels: `cn = 0 ⇒ inv_cn = inf ⇒ new = NaN` (IEEE, no trapping).

### 2.4 spatial_connect (gMSHBM, once per em_iter, before the comp loop)

`u[l,:] = (Σ_{n∈members(l)} sl·grad[n,:]) / Σ sl` (NaN if empty),
`scv[p] = 2·(grad[n]·u[l]) − grad_sq[n] − Σ_d u[l,d]²`, NaN → −Inf.

### 2.5 spatial_xyz (after every check_connectedness; cMSHBM also once per em_iter before the comp loop)

`lambda_X[:,l] = Σ_{n∈members(l)} sl·sphere[n,:]`, `s_muc = lambda_X / ‖·‖`
(NaN if 0), `cdln3(γ)` = closed form of Cdln(γ, 3),
log γ − γ + ½log 2π − log1p(−e^{−2γ}) (NaN at γ ≤ 0), `sxv[p] = cdln3[l] + γ[l]·(sphere[n]·s_muc[l])`, NaN → 0.

### 2.6 check_connectedness (after every λ-loop when `iter_em ≥ first_em_iter_with_conn_xyz`: 2 for gMSHBM, 1 for cMSHBM)

`labels[n] = 1 + argmax_{l∈P(n)} sl` (first max), 0 if the row sums to 0.
cMSHBM first applies `remove_isolated_surface_components` to those
labels on device (`ConnectednessGPU.remove_isolated`: hook-to-min
components of the label-restricted graph, members threaded into per-root
linked lists, one block per component smaller than
`cMSHBM_isolated_component_min_size` histograms its neighbours' labels
excluding 0 and its own and takes the most frequent, smallest on ties;
every vote reads the entry-state labels and the counts are integers, so
the result is order-invariant and bit-identical to the host function
applied per hemisphere, at a cost flat in the threshold; the cleaned
copy stays internal, as on the CPU chain).
Then exactly `compute_components_general` + `component_distance`
(`arealmshbm/check_connectedness`), producing `parcel_components (L,)
fp64 (NaN if empty)` and `eucli (L,) fp32`; then
`distrib = (eucli > connect_th) | (pc > components_threshold)`,
`γ[distrib] += 1000`, `max_conn = max(eucli[distrib])` (0 if none),
`max_comp = max(pc[distrib & finite])` (`components_threshold` if none).
Comp loop stops when `max_conn ≤ connect_th and max_comp ≤ threshold`,
or `comp_iter ≥ max_iter_comp`.

### 2.7 EM stop (once per em_iter)

```
llp   = fp32(T·cdln[l] + kappa[l]·acc)        (fp32 ops, then fp64)
scv   = LOG_EPS_POW20 if !finite(scv) else scv   (written back)
log_sl= fp64 log(sl) if sl > 0 else LOG_EPS_POW20
cost += sl · (llp + log_theta_cost − w·log_sl − c·V_temp + beta·scv)   (fp64)
```
`cost = NaN` if `any(col_nan)` (dense `0·NaN`). Convergence:
`matlab_ratio_converged(update_cost, cost)`; hard stop at `iter_em > 100`.

### 2.8 intra_em outer loop (Step3Pipeline.run semantics, on device)

Per round: `kappa ← ini_val`, `s_t_nu[t] ← mu`, run EM, then
`intra_subject_var` and `intra_em_cost` (fp64 sums, cdln of σ/ε cached),
`matlab_ratio_converged`, cap `max_iter_intra_em`. `intra_subject_var`
is bit-identical to the CPU reference only for **T < 8**: the kernel
sums the T sessions serially, which matches numpy's axis-0 reduction
only below the length at which its pairwise summation starts blocking.

## 3. Validation

* Unit: each kernel vs its CPU counterpart on sub-001 data
  (`arealmshbm/vmf_clustering/tests/_sub001_fixture.py`).
* End-to-end: labels of `gpu` vs `cpu` on sub-001…003; the reference
  drift is the one measured in `docs/step3_flow_and_subgraphs.md`
  § `gpu` (3 / 35 / 1334 / 122 vertices on sub-001 / YS 1–3).
* Determinism: two `gpu` runs must be `np.array_equal` on labels
  and on `s_lambda`.

The ground-truth records below are dated; arms named `gpu_full` are the
dense CuPy EM measured alongside `gpu` on the same inputs. Pooled over
cohorts the two are fidelity-equivalent, not uniformly ordered (of the
cells below, cMSHBM YS is the one where the dense arm is ahead).

* Cohort ground truth: on an internal 10-subject cohort, `gpu`
  labels agree with the stock-MATLAB reference (produced by an
  internal MATLAB-alignment oracle) at **gMSHBM 95.19 %** and
  **dMSHBM 99.988 %** cortex-only (measured 2026-09-04 on this tree,
  GPU step 0/1). Same run, `gpu_full`: 95.12 % / 99.984 %; clean
  8cb68a0 `gpu_full`: 95.13 % / 99.984 %. Label delta `gpu` vs
  `gpu_full`: dMSHBM 50 vertices over 10 subjects; gMSHBM 3 115, of
  which 2 074 sit in the one subject that also dominates the step-0
  GPU run-to-run band (2 297 of 2 614 between two clean-HEAD runs).
  On a Mode-B prior trained by step-2 `gpu`, the GT agreement
  is 74.90 % vs 74.92 % for `gpu_full` on the same prior.
  `lib/hyperparameters/step3.json` points here for it.
* cMSHBM (2026-09-20, the device `remove_isolated` pre-predicate; same
  step-0/1 caches for every arm, scored by the internal cohort-comparison
  harness against the stock-MATLAB reference on the three 10-subject
  cohorts; stock cMSHBM itself crashes
  on 3 of the 30 subjects, so those cells are 9 / 8): cortex agreement
  `gpu` **99.35 / 98.42 / 98.13 %** (YS / ds004466 / ds000221) vs
  `gpu_full` 99.42 / 98.26 / 98.08 % and `cpu` 99.25 / 98.28 / 98.17 %.
  Label delta `gpu` vs `cpu`: 2 093 / 3 820 / 4 542 vertices over
  the 10 subjects (`gpu_full` vs `cpu` on the same inputs: 2 264 / 2 309 /
  5 079); `gpu` vs `gpu_full`: 1 883 / 3 155 / 1 584. Sub-001 of
  YS and ds004466 is identical on all three backends. The device
  pre-pass itself is bit-identical to the host function
  (`check_connectedness/tests/test_connectedness_gpu.py`, 4 thresholds on
  grids + 9 sub-001 label fields with up to 1 643 relabelled vertices).

## 4. Results (sub-001, RTX 5090 Laptop; 2026-09-03 unless dated)

| | dense GPU EM (`gpu_full`) | `gpu` |
|---|---:|---:|
| load_inputs | 0.57 s | 0.16–0.20 s (streaming MAT parse via `data_io/mat5_stream.py`, θ→CSR, no dense mask, no BOLD transpose) |
| session_init (warm) | 0.86 s | 0.02–0.04 s |
| EM | 4.0 s | 0.10 s |
| λ-iteration | 19 ms | 0.15 ms |
| check_connectedness | 4.3 ms | 0.16 ms |
| check_connectedness, cMSHBM (with the `remove_isolated` pre-pass; 2026-09-24) | 18-27 ms (host pre-pass 14-23 ms) | 0.27 ms (pre-pass 0.12 ms at 400-1 600 small components) |
| M-step call (incl. X·s_λ; mean per `MStepGPU.run` over a full EM run; 2026-09-03, staged `x_dot_sl_bits`) | 50 ms | 2.5 ms |

`x_dot_sl_bits` (2026-09-20): the kernel text is the one shared with step 2
(`m_step/_xdot_kernel.py` — member bytes are read straight from global memory
with a 4-deep `__ldg` prefetch instead of being staged in shared memory;
member weights (still) and row ids (new) are staged per 64-member chunk in
`XDOT_CHUNK * 8` B of dynamic shared memory, with the same two `__syncthreads`
per chunk; still 128 threads × 2 bytes here, which keeps the fp64 B-term tree
unchanged), uint32-identical to the shared-memory-staged form, which is the
oracle in `m_step/tests/test_m_step_gpu.py`, instantiated at both
consumers' geometries (the B tree spans `blockDim.x` lanes, so 128 × 2 and
256 × 1 are not bit-comparable to each other; a cancellation fixture there
makes the B-term order visible, which the sub-001 inputs do not). Same
session on sub-001: `x_dot_sl_bits` 1.55 → 0.80 ms per launch; standalone
`MStepGPU.run` calls on the fixture inputs (θ as s_λ, μ as s_ψ and as the
initial s_t_ν; mean of 5 warm calls after a warm-up) 3.45 → 2.73 ms. That is a
different workload (inner-iteration count) from the table's per-call EM mean
above, which was not re-measured, so the two are not comparable.

Validation: kernel tests (`vmf_clustering/tests/test_kernels_gpu.py`,
`m_step/tests/test_m_step_gpu.py`,
`check_connectedness/tests/test_connectedness_gpu.py`,
`step3_pipeline/tests/test_sparse_inputs.py`) + the end-to-end
`vmf_clustering/tests/test_session_gpu.py`; backend A/B via an
internal step-3 comparison harness. Two gotchas found on the
way: cupy's fp32 elementwise kernels (`log`, even `astype`) flush
denormals — θ carries a few 1e-45 cells, so `log(θ)` is taken on the
host in fp64; and the on-disk MW rows of sub-001's `.b2nd` dev store
are *not* zero (it predates the 2026-05-18 writer-side MW clamp), so
the device MW-zero pass is load-bearing there.
