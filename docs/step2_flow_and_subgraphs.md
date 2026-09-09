# Step 2 — call graph + module map

Structural reference for the production Python step-2 super-call (Mode-B
group prior estimation). Maps each call site in the EM body to the
Python module that owns it.

For the migration conventions drawn from this design (validate /
profile pattern, leaf module shape, backend dispatch), see
[`strategy.md`](strategy.md). For Mode-A vs Mode-B semantics, see
[`pipeline_modes.md`](pipeline_modes.md).

## Variants in scope

`Step2Pipeline` supports **gMSHBM** and **dMSHBM**.
`Step2Config(mode='cMSHBM')` raises `NotImplementedError` — the cMSHBM
xyz-vMF prior is not wired into the master kernel.

## Backends in scope

`Step2Config.backend ∈ {'cpu', 'gpu'}`, and `PipelineConfig.backend_step2`
offers the same two values.

| value | what it is | numerical status |
|---|---|---|
| `'cpu'` | numba master + host outer-EM leaves | **the reference** |
| `'gpu'` | P-layout / bit-packed CuPy backend (`Step2SparseSession`) | CPU semantics on device; contract in [`step2_sparse_design.md`](step2_sparse_design.md). Requires `seed_mesh='fsaverage3'` (⌈D/8⌉ ≤ 256), `num_clusters ≤ 512`, `n_grad_components ≤ 11772` |

The dense CuPy port that held `'gpu'` until 2026-09 (fp32 flush-to-zero
`exp`, wrong at S=1, nondeterministic) was removed; `'gpu_sparse'`, the
name this backend shipped under beside it, is rejected by both configs.

### `Params_Final.mat` format — unchanged, on every backend

`Step2Pipeline._save_params` calls `_save.save_params_final` on both
backends, so the file keeps the contract it has
always had: `theta` a dense fp64 `(N, L)` block, container
`do_compression=True`, byte-identical to the pre-`_save.py` inline saver
(`test_save.py::test_dense_branch_matches_the_legacy_saver` compares
`bytes[128:]`). The `gpu` backend's `export_params()` returns a
scipy `csc_matrix` and the writer densifies it.

* **MATLAB CBIG consumers** are the reason dense is the only form:
  `CBIG_MSHBM_generate_individual_parcellation.m` does
  `log(Params.theta)`, which MATLAB will not evaluate on a sparse input.
  An out-of-tree MATLAB checkout, kept as the cross-implementation
  alignment oracle, reads these files directly.
* Every in-tree reader is agnostic either way:
  `data_io/load_group_prior.py` densifies on `hasattr(x, 'toarray')` and
  returns the same dense fp32 `(N, L)` / `(D, L)` arrays from both
  forms, so a MATLAB-side `sparse()` prior is still a valid Mode-A input.

### `'cpu'` — the host path

`Step2Pipeline.run()` chains `load_inputs` → `initialize_params` →
`run_em`. `run_em` builds one `Step2EmIterSession`
(`step2_em_iter_master/session.py`, kernels in `_kernels.py`) whose
`upload_initial_state` aliases `Params['s_lambda' / 'theta' / 's_t_nu']`
to Session-owned scratch, and drives it through `vmf_clustering_batch`;
the three outer-EM closure leaves (L16 `intra_em_cost_step2`, L17
`intra_subject_var_loop`, L18 `inter_subject_var`) run on the host
between batches.

### `'gpu'` — the P-layout path

A different three-stage chain, selected as a set in `run()` because
stage 1 returns a different dataclass:

```
Step2Pipeline.run()  [backend == 'gpu']
├── load_inputs_sparse()        → Step2SparseInputs   step2_io/sparse_inputs
├── initialize_params_sparse()  → {ini_val, iter_inter, Record}  (host only)
└── _run_em_sparse()            → Step2SparseSession  step2_em_iter_master/session_gpu
    ├── initialize_state()                     (device K1 init; replaces compose_init_state)
    ├── per inter iter:  reset_inter()         (sigma ← ini_val, s_psi ← mtc)
    │   ├── per intra iter: reset_intra()      (kappa seed ← ini_val, s_t_nu ← mtc)
    │   │   ├── em_body_sparse() → run_iter() ×K   step2_pipeline/vmf_clustering_batch
    │   │   └── intra_closure()                (L17 then L16, on device)
    │   └── inter_closure()                    (L18, on device)
    └── export_params()                        (the only bulk D2H of the run)
```

State lives on the `P = nnz(boundary_mask)` support (2.4 % of `N·L`),
not the dense `(N, L)` grid; BOLD stays bit-packed on device and every
contraction with it is a bit-sum plus a rank-1 term. All three outer-EM
closure leaves and both resets moved on-device too, so the only host
traffic per EM iter is the `(S,)` fp64 `cost_S`, and a `'gpu'` run
never imports `step2_em_outer` at all (the pipeline's leaf imports are
lazy — worth ~0.2 s of eager numba at import time).

`theta` reaches the writer as a scipy `csc_matrix` straight out of
`export_params()`; the densify is the writer's job, not this path's, so
the file is the same shape as every other backend's. See
*`Params_Final.mat` format* above.

`Step2Result.timings` gains `init_device` and `closure_total` on this
path, and loses `initialize_params.compose` (there is no host compose).

Profiling / A/B tooling:
`python -m arealmshbm.step2_pipeline.profile --project <p> --backend gpu --runs 5`;
cross-backend A/B is done with an internal step-2 comparison harness.

### BOLD cache modes

`bold_cache_mode` (default `'auto'`) applies to **both** backends with
mode-specific semantics:

#### CPU backend

The loader (`SubjectProfileLoader`) owns a host-side bit-packed cache.
`'auto'` / `'eager_bitpacked'` both map to the loader's
`'eager_bitpacked'` mode (the only host cache mode — there is no
unpacked uint8 / fp32 host cache). `'stream'` keeps per-call disk
decode (also bit-packed end-to-end — the packed bytes flow straight
into the fused widen+normalize numba kernel).

* `'eager_bitpacked'` first `load_into(s)` reads packed bytes from
  `.b2nd` and caches them as `(N, T, ⌈D/8⌉)` uint8 (~250 ms decode).
  Subsequent calls run a numba `_widen_normalize_bitpacked_to_f32`
  kernel directly from the host cache (~25 ms per subject).
* `'stream'` runs the same `_widen_normalize_bitpacked_to_f32` numba
  kernel but reads the packed bytes off disk every call. No fp32 chunk-
  view intermediate is ever materialized.

On binary 0/1 input the bit-packed kernel is **bit-identical** to the
host fp32 `_normalize_session_inplace_kernel` (same fp32 accumulator
order, integer popcount → fp32 mean is exact for D ≤ 2^24).

Wall numbers: [`## Wall`](#wall) below.

#### GPU backend

`bold_cache_mode` / `gpu_cache_safety_margin_gb` size the
`(S, T, N, ⌈D/8⌉)` uint8 packed slab against the resident device state
(`Step2SparseSession._resolve_cache_mode`, design §2.2): `'auto'`
(default) picks `'eager_bitpacked'` when packed cache + gradient +
`(S, T, L, D)` fp32 `s_t_nu` / `X_dot_sl` + `(S, P)` `s_lambda` + the
iteration-1 scratch + the margin fit in free device memory, otherwise
`'stream'`; an explicit `'eager_bitpacked'` that does not fit raises,
and `'stream'` refills one pinned `(T, N, ⌈D/8⌉)` slot from a pageable
host cache built once in `load_inputs_sparse` (two H2D visits per
subject per EM iteration). There is no `(S, N, L)` `s_lambda` on this
backend, so the dense port's real VRAM ceiling is gone.

Bit-packing convention: LSB-first along D (cell `d` → bit `(d & 7)` of
byte `(d >> 3)`), matching `numpy.packbits` with `bitorder='little'`.
Padding bits past `D` in the last byte are zero.

## The whole flow

```
Step2Pipeline.run(cfg)                                                [arealmshbm/step2_pipeline]
│
├── load_inputs()
│   ├── load_avg_mesh(lh/rh, inflated)                                data_io/load_avg_mesh
│   ├── read_cohort(project_dir/cohort.json)                          data_io/cohort
│   ├── SubjectProfileLoader (cohort.subjects[].profile_b2nd)         step2_io/subject_loaders
│   ├── SubjectGradientLoader (cohort.subjects[].gradient_*)          step2_io/subject_loaders
│   │   (gMSHBM only; sniffs .npy / .mat by suffix)
│   ├── load_group_mtc(group.mat)                                     step2_io/load_group_mtc
│   ├── load_spatial_mask                                             data_io/load_spatial_mask
│   ├── build_step2_boundary_mask(mode)                               step2_init/boundary_mask
│   └── compose_init_state(bold_loader)                               step2_init/compose_init_state
│       (fused per-sub argmax + theta + (S, N, L) fp32 s_lambda;
│        streams one subject's BOLD at a time)
│
├── initialize_params()                                               inline + initialize_concentration
│   └── initialize_concentration(D)                                   initialize_concentration
│
├── outer EM loop (≤ max_iter_inter, default 10)
│   ├── reset sigma + s_psi                                           inline
│   ├── intra-EM loop (≤ 15)
│   │   ├── reset kappa + s_t_nu                                      inline
│   │   ├── ⟨A⟩ vmf_clustering_batch                                  step2_pipeline/vmf_clustering_batch
│   │   ├── ⟨B⟩ intra_subject_var_loop                                step2_em_outer/intra_subject_var_loop
│   │   ├── ⟨C⟩ intra_em_cost_step2                                   step2_em_outer/intra_em_cost
│   │   └── convergence test                                          inline
│   ├── ⟨D⟩ inter_subject_var                                         step2_em_outer/inter_subject_var
│   └── outer convergence test                                        inline
└── save Params_Final.mat                                             inline


⟨A⟩ vmf_clustering_batch — the EM body
│   step2_pipeline/vmf_clustering_batch.py refreshes s_psi/sigma on
│   the caller-owned Step2EmIterSession (built ONCE per run_em) and
│   loops max_iter_em times, calling sess.run_iter(Params) and doing
│   the per-subject cost rel-diff convergence test in Python.
│
└── outer EM iter loop (≤ max_iter_em, default 100)
    └── Step2EmIterSession.run_iter(Params)                           step2_em_iter_master/session.py
        └── em_iter_master_kernel_streaming                           step2_em_iter_master/_kernels.py
            │   Python orchestrator + @njit sub-kernels. The two
            │   per-subject loops (Phase A.3 and Phase C+D+E.1) are
            │   driven from Python; loader callbacks decode one
            │   subject's BOLD/grad into reused (N, T, D) / (T, N, D_grad)
            │   scratch slots between phases.
            │
            ├── Phase A.1 — sigma_psi precompute                      _sigma_psi_SLD_compute
            ├── Phase A.3 — streamed per-s X_dot_sl                   _compute_X_dot_sl_s_NTD (1 BLAS sgemm / sub)
            ├── Phase B   — multi-subject M-step inner while-loop     _mstep_inner_loop_master_step2
            │   │   (BLAS-free given X_dot_sl_STLD)
            │   ├── kappa_sum reduce over (S, T, L, D) → invad closed _kappa_sum_reduce_STLD + invad_numba
            │   └── fused per-(s, t, l) col-norm + cosine + flag      _mstep_fused_update_STLD
            ├── per-subject loop (S iterations):
            │     ├── Phase C — spatial_connect_per_subject  (gMSHBM) _spatial_connect_per_subject_numba
            │     ├── Phase D — fused E-step (fp64 scratch out)       _fused_estep_per_subject_NTD
            │     └── Phase E.1 — per-sub normalize → fp32 storage    _phase_e1_normalize_per_subject_kernel
            └── Phase E.2 — theta = mean over S                       _phase_e2_theta_only_kernel

    └── EM convergence test (per-subject rel-diff on cost)            inline (Python, between master calls)
```

Iteration counts on the (3-subj, 2-sess, fsaverage6, L=400, β=5)
reference cohort: outer EM converges in ~10–20 iters per intra-EM
iter; intra-EM converges in 3–6 iters per inter iter; inter converges
in 2–5 iters total. See
[`step2_em_iter_master_kernel.md`](step2_em_iter_master_kernel.md)
for the master kernel's layout invariant and precision contract.

## Component nesting table

Tabular companion to the tree above. Each row is one node in the
call graph, classified by **kind** (driver / phase / loop / subgraph
/ leaf / kernel / inline), with its owner module and how many times
it executes per `Step2Pipeline.run()`.

Multiplicities are expressed as symbolic loop products. On the
reference (3-subj, 2-sess, L=400, β=5) cohort:

- **I** = inter-EM iters (`max_iter_inter`, default 10; typical 2–5)
- **J** = intra-EM iters per inter (`max_iter_intra_em`, ≤ 15; typical 3–6)
- **K** = EM-iters inside `vmf_clustering_batch` (`max_iter_em`, ≤ 100; typical 10–20)
- **M** = M-step inner iters (`max_iter_m`, ≤ 50; typical 5–10)
- **S** = subjects · **T** = sessions · **L** = parcels

Indentation in the **Component** column mirrors call depth (`├` / `│`
/ `└` like the ASCII tree above).

| # | Component | Kind | Module | Mult. per `run()` |
|--:|---|---|---|---|
|  1 | `Step2Pipeline.run` | driver | [step2_pipeline/pipeline.py](../arealmshbm/step2_pipeline/pipeline.py) | 1 |
|  2 | `├ load_inputs` | phase | [pipeline.py](../arealmshbm/step2_pipeline/pipeline.py) | 1 |
|  3 | `│  ├ load_avg_mesh` (lh, rh, inflated) | leaf | [data_io/load_avg_mesh.py](../arealmshbm/data_io/load_avg_mesh.py) | 2 |
|  5 | `│  ├ read_cohort(project_dir/cohort.json)` | leaf | [data_io/cohort.py](../arealmshbm/data_io/cohort.py) | 1 |
|  7 | `│  ├ SubjectProfileLoader` (cohort.subjects[].profile_b2nd) | leaf | [step2_io/subject_loaders.py](../arealmshbm/step2_io/subject_loaders.py) | S (streamed) |
|  8 | `│  ├ SubjectGradientLoader` (cohort.subjects[].gradient_*) | leaf | [step2_io/subject_loaders.py](../arealmshbm/step2_io/subject_loaders.py) | S (streamed, gMSHBM) |
|  9 | `│  ├ load_group_mtc` | leaf | [step2_io/load_group_mtc.py](../arealmshbm/step2_io/load_group_mtc.py) | 1 |
| 10 | `│  ├ load_spatial_mask` | leaf | [data_io/load_spatial_mask.py](../arealmshbm/data_io/load_spatial_mask.py) | 1 |
| 11 | `│  ├ build_step2_boundary_mask` | leaf | [step2_init/boundary_mask.py](../arealmshbm/step2_init/boundary_mask.py) | 1 |
| 12 | `│  └ compose_init_state` (fused per-sub argmax+theta+s_lambda) | leaf | [step2_init/compose_init_state.py](../arealmshbm/step2_init/compose_init_state.py) | 1 |
| 14 | `├ initialize_params` | phase | pipeline.py | 1 |
| 15 | `│  └ initialize_concentration` | leaf | [initialize_concentration/](../arealmshbm/initialize_concentration/) | 1 |
| 16 | `├ outer-inter loop` (≤ I) | loop | pipeline.py | 1 |
| 17 | `│  ├ reset σ + s_psi` | inline | pipeline.py | I |
| 18 | `│  ├ intra-EM loop` (≤ J) | loop | pipeline.py | I |
| 19 | `│  │  ├ reset κ + s_t_nu` | inline | pipeline.py | I·J |
| 20 | `│  │  ├ ⟨A⟩ vmf_clustering_batch` | thin wrapper | [vmf_clustering_batch.py](../arealmshbm/step2_pipeline/vmf_clustering_batch.py) | I·J |
| 21 | `│  │  │  └ EM-iter loop` (≤ K) | loop | vmf_clustering_batch.py | I·J |
| 22 | `│  │  │     └ Step2EmIterSession.run_iter` | subgraph entry | [step2_em_iter_master/session.py](../arealmshbm/step2_em_iter_master/session.py) | I·J·K |
| 23 | `│  │  │        └ em_iter_master_kernel_streaming` | Python orchestrator + @njit subs | [step2_em_iter_master/_kernels.py](../arealmshbm/step2_em_iter_master/_kernels.py) | I·J·K |
| 24 | `│  │  │           ├ Phase A.3 — per-s streamed X_dot_sl[s]` | kernel | `_compute_X_dot_sl_s_NTD` (BLAS sgemm) | I·J·K·S |
| 25 | `│  │  │           ├ Phase B — M-step inner while-loop` (≤ M) | numba loop | `_mstep_inner_loop_master_step2` | I·J·K |
| 26 | `│  │  │           │  ├ kappa_sum reduce (S, T, L, D) | kernel | `_kappa_sum_reduce_STLD` (numba) | I·J·K·M |
| 27 | `│  │  │           │  ├ invad closed-form | kernel | `invad_numba` (numba) | I·J·K·M |
| 28 | `│  │  │           │  └ fused col-norm + cosine + flag | kernel | `_mstep_fused_update_STLD` (numba) | I·J·K·M·S·T·L |
| 29 | `│  │  │           ├ per-subject loop (Python, ×S):` | orchestrator | | I·J·K·S |
| 30 | `│  │  │           │  ├ Phase C — spatial_connect` (gMSHBM) | kernel | `_spatial_connect_per_subject_numba` | I·J·K·S |
| 31 | `│  │  │           │  ├ Phase D — fused E-step → fp64 scratch | kernel | `_fused_estep_per_subject_NTD` (numba) | I·J·K·S |
| 32 | `│  │  │           │  └ Phase E.1 — per-sub normalize + fp32 cast | kernel | `_phase_e1_normalize_per_subject_kernel` (numba prange) | I·J·K·S |
| 34 | `│  │  │           └ Phase E.2 — theta = mean over S | kernel | `_phase_e2_theta_only_kernel` (numba prange) | I·J·K |
| 43 | `│  │  ├ ⟨B⟩ intra_subject_var_loop` | leaf | [step2_em_outer/intra_subject_var_loop.py](../arealmshbm/step2_em_outer/intra_subject_var_loop.py) | I·J |
| 44 | `│  │  │  └ intra_subject_var` (step-3 reuse) | kernel | [intra_em/intra_subject_var.py](../arealmshbm/intra_em/intra_subject_var.py) | I·J·~5–20 (inner while-loop) |
| 45 | `│  │  ├ ⟨C⟩ intra_em_cost_step2` | leaf | [step2_em_outer/intra_em_cost.py](../arealmshbm/step2_em_outer/intra_em_cost.py) | I·J |
| 46 | `│  │  └ intra-EM convergence test` | inline | pipeline.py | I·J |
| 47 | `│  ├ ⟨D⟩ inter_subject_var` | leaf | [step2_em_outer/inter_subject_var.py](../arealmshbm/step2_em_outer/inter_subject_var.py) | I |
| 48 | `│  └ outer convergence test` | inline | pipeline.py | I |
| 49 | `└ save Params_Final.mat` | phase | pipeline.py | 1 |

### Reading the table

- **Rows 20–34** are the EM-iter body — the streaming master kernel.
  Rows 22-34 execute one Python orchestrator call
  (`em_iter_master_kernel_streaming`) per outer EM iter; the two
  per-subject loops (Phase A.3 and the C+D+E.1 trio) run from Python
  with loader callbacks so the working set never exceeds one subject's
  slabs regardless of S. The intermediate phases (B, E.2) are pure
  `@njit` numba. See
  [`step2_em_iter_master_kernel.md`](step2_em_iter_master_kernel.md)
  for the layout invariant and precision contract.
- Rows 43-48 are the outer-EM closure (intra_subject_var, intra_em_cost,
  inter_subject_var) — closed-form updates outside the master kernel.

## Module / file map

```
arealmshbm/
├── step2_pipeline/                        # top-level Python entry
│   ├── config.py                          Step2Config dataclass (g/dMSHBM only)
│   ├── pipeline.py                        Step2Pipeline lifecycle (g/dMSHBM)
│   └── vmf_clustering_batch.py            ⟨A⟩ thin wrapper around the
│                                          fused EM-iter master; loops
│                                          max_iter_em iters + EM-conv test
│
├── step2_em_iter_master/                  # streaming EM-iter body (fused master kernel)
│   ├── __init__.py
│   ├── _kernels.py                        Python orchestrator + @njit subs:
│   │                                       _compute_X_dot_sl_s_NTD, _sigma_psi_SLD_compute,
│   │                                       _mstep_inner_loop_master_step2,
│   │                                       _spatial_connect_per_subject_numba,
│   │                                       _fused_estep_per_subject_NTD,
│   │                                       _phase_e1_normalize_per_subject_kernel,
│   │                                       _phase_e2_theta_only_kernel,
│   │                                       invad_numba,
│   │                                       em_iter_master_kernel_streaming
│   │                                       (single fixed precision profile)
│   ├── session.py                         Step2EmIterSession — scratch +
│   │                                      loader handles + boundary staging
│   ├── _kernels_gpu.py                    the 'gpu' backend's cupy.RawModule
│   │                                       (P-layout kernels K0-K14, design §3)
│   ├── session_gpu.py                     Step2SparseSession — device-resident
│   │                                      state + the §4 Session API
│   └── tests/                             per-kernel parity, Session vs CPU
│                                          master, alias invariant
│
├── step2_io/                              # data I/O leaves
│   ├── load_group_mtc.py                  group.mat (mtc / epsil / labels)
│   ├── load_subject_profiles.py           per-subject (N, D, T) BOLD profiles
│   ├── load_subject_gradient.py           per-subject (N, 100, T) diffusion emb
│   ├── sparse_layout.py                   Step2Layout — CSR/CSC over P ('gpu')
│   └── sparse_inputs.py                   Step2SparseInputs + its loader ('gpu')
│
├── step2_init/                            # init-phase leaves
│   ├── init_s_lambda.py                   L8 — per-subject one-hot init
│   ├── boundary_mask.py                   L9 — block-diagonal (g/dMSHBM)
│   └── init_theta.py                      post-init theta computation
│
└── step2_em_outer/                        # outer-EM closed-form leaves
    ├── intra_subject_var_loop.py          L17 — s_psi + sigma while-loop
    ├── inter_subject_var.py               L18 — mu + epsil closed form
    └── intra_em_cost.py                   L16 — intra-EM convergence cost
```

**Per-leaf spec bars** (the leaves NOT folded into the EM-iter master
kernel — i.e., the three above):
- L16 `intra_em_cost_step2`: rel-diff ≤ 1e-4
- L17 `intra_subject_var_loop`: max-abs(s_psi) ≤ 1e-5
- L18 `inter_subject_var`: max-abs(μ) ≤ 1e-5

The EM body (kappa update, s_t_nu update, spatial_connect, fused E-step,
normalise, theta) lives inside the master kernel; the end-to-end bar
for that block is the per-subject cost rel-diff (≤ 1e-4).

Mode dispatch happens at the `Step2Config.mode` ∈ `{gMSHBM, dMSHBM}`
knob plus a derived `beta_internal` that resolves to `5 × 1000 = 5000`
(gMSHBM) or `0` (dMSHBM). Inside the EM body, gMSHBM enables the
gradient-prior path in `_spatial_connect_per_subject_numba`; dMSHBM
skips spatial entirely.

## Wall

### `'gpu'` backend — measured

Bench `testdata/step2_bench/proj` (S=1) and `proj2` (S=2): sub-001 (+002),
fsaverage6 `N=81924`, `T=6`, `D=1175`, `L=300`, gMSHBM `beta=5`.
RTX 5090 Laptop (82 SMs, 24 GB), cupy 13.6 / NVRTC 12.9, warm =
in-process runs 2+ after a prewarm. The instrumented numbers below were
taken with the per-kernel cuda-event timers the Session carried at the
time (~8-12 % of `em_total`, since retired — see the uninstrumented
range alongside).

| | `'gpu'` | retired dense port | `'cpu'` | speedup vs dense / cpu |
|---|---:|---:|---:|---:|
| S=1 warm `run()` | **0.32-0.36 s** (0.320 [0.311, 0.329] n=7 in one session; 0.357 [0.330, 0.392] in a later, warmer one; uninstrumented 0.32-0.37) | 6.9 s | 71 s | ×19-21 / ×200 |
| S=2 warm `run()` | **0.59-0.64 s** (0.589 [0.580, 0.599] and 0.640 [0.634, 0.646], n=5 each, two sessions) | 14.0 s | 103 s | ×22-24 / ×160-175 |

(The dense-port / `cpu` columns are the pre-rewrite baselines recorded
in [`step2_sparse_design.md`](step2_sparse_design.md) §0; the sparse
column was re-measured 2026-09-03 on the final HEAD, after the F3 kernel
pass — K3/K6 0.80/0.77 → 0.34/0.35 ms per launch, `init_hard_labels`
22.9 → 8.5 ms, K7 39 → 25 ms, every change bit-identical in output.)

Per-stage, warm, S=1 → S=2: `load_inputs` 0.100 → 0.174 s ·
`session_ctor` 0.058 → 0.086 s · `init_device` 0.009 → 0.017 s ·
`em_total` 0.118 → 0.234 s · `closure_total` 0.019 → 0.063 s ·
`save` 0.006-0.010 s.

EM iterations: S=1 `inter=10`, 42 `run_iter` calls, **~3.1 ms/EM-iter**
(≈ 2.5 ms excluding the one-shot iteration-1 K7 dense pass, ~25 ms,
which `em_total` carries); S=2 `inter=10`, 42 calls, ~5.6 ms/EM-iter.
Per-launch means (S=1): K3 `x_dot_sl_bits` 0.34 ms, K6 `acc_P` 0.35,
K8-K10 E-step rows 0.40, M-step 0.17 per `iter_m` (×2.4), K5 connect
0.15, K12 theta 0.22.

Parity vs the CPU reference (`compare_step2_backends.py`, §7.3 bars):
S=1 theta argmax flips **0** (bar ≤ 500), support-diff rows 1, dead-alive
rows **1 303 vs 1 303**, `kappa` max-rel **0.0** (bit-identical),
first-inter `Record` rel **3.99e-6** (bar ≤ 5e-3), `iter_inter` 10 == 10.
S=2 argmax flips **0**, dead-alive **168 vs 168**, `kappa` **0.0**,
first-inter `Record` rel **1.38e-6**. Two `gpu` runs are `np.array_equal`
on every saved field at both S (theta compared as its csc triplet).
Downstream: step 3 (`gpu_full`, sub-001) run from the sparse-GPU prior
produces **exactly the same labels** as from the CPU prior (0 of 73 641
cortex vertices differ); the retired dense port's prior differed on 4 758
(93.5 % agreement). `epsil` max-rel ~1.0 vs CPU at S=1/S=2 is the
ill-conditioned `invAd(dim, R→1)` and has no downstream effect.

Reproduce:

```bash
# per-stage profile
python -m arealmshbm.step2_pipeline.profile     --project testdata/step2_bench/proj --backend gpu --runs 5 --prewarm

```

The A/B against the frozen CPU reference (and the run-to-run
determinism check) runs through an internal step-2 comparison
harness that is not shipped here.

### Production-cohort walls (S=40, T=6, L=300)

CPU backend, per-variant `em_total` at `max_iter_inter=2` (2026-05-20,
internal step-2 profile run): gMSHBM 778.33 s, dMSHBM 562.12 s
(`load_inputs` 0.62 s, `initialize_params.compose` 11.17 s). The
retired dense CuPy port measured 101.49 s / 75.93 s on the same runs,
and 373.94 s for the gMSHBM step at production `inter=10` in the
40-subject Mode-B end-to-end log. The `'gpu'` backend on this tree:
16.5 s for the 40-subject Mode-B step 2 at `inter=10` (2026-09-07
end-to-end run, all-GPU backends), and 4.1 s for the 10-subject cohort
vs 62 s dense / 538 s `cpu`
([`step2_sparse_design.md`](step2_sparse_design.md) §9).

`inter=2` is the short-run benchmark cap; production runs use the
default `max_iter_inter=10`. The step-2-only CPU-vs-GPU sweep across
variants runs through an internal profiling harness not shipped here.

## Precision contract

| Boundary | dtype | Notes |
|---|---|---|
| BOLD profiles on disk | bitpacked uint8 `.b2nd` (1 bit/cell) | step 1 writer; see [`docs/profile_disk_format.md`](profile_disk_format.md) |
| Gradients on disk | fp32 `.npy` per hemi | step 0 writer |
| group.mat `mtc` on disk | fp64 | written by step 1's ini_params |
| In-memory `Params.s_t_nu`, `Params.s_psi`, `Params.mu`, `Params.theta` | fp32 | matches step 1 / step 3 convention |
| `Params.kappa`, `Params.sigma`, `Params.epsil` | fp32 (1, L) | scalars per parcel |
| `Cdln`, `invAd` reductions | fp64 internally | reuses existing helpers |
| `Params.s_lambda` storage | **fp32** | Storage is fp32 (halves the s_lambda footprint vs the legacy fp64 design — 52 GB → 26 GB at S=200). The fp64 precision needed for the E-step softmax's β=5000 subnormal tier is preserved on a per-subject (N, L) fp64 scratch buffer inside Phase D + E.1 only; the row-normalized result casts back down to fp32 storage. |

The fp32 / fp64 boundary at the softmax seam is implemented inside
[`arealmshbm/step2_em_iter_master/_kernels.py`](../arealmshbm/step2_em_iter_master/_kernels.py)
in `_fused_estep_per_subject_NTD` — the kernel reads the fp32
``log_vmf_NL`` accumulator and immediately promotes to fp64 via
``np.float64(...)`` (note: numba's ``float(fp32)`` is a no-op
type-preserving cast; the explicit ``np.float64`` is necessary for
the fp32→fp64 promotion to actually occur).

## Reuse from earlier steps

| Module | Reused by | Notes |
|---|---|---|
| `em_stop_criterion/_cdln.py` (`_cdln_single`) | E-step phase D of the EM-iter master | Cdln, general D, 5-term Debye |
| `m_step/_invad.py` (`invad`) | L17, L18 (outer-EM closure) | invAd with Banerjee + Newton. Inside the EM-iter master we use a numba inline of the Banerjee asymptotic branch (`invad_numba`) — for D ≥ 200 the bessel polish never activates. |
| `initialize_concentration/` | init Params | Bessel-root κ init |
| `data_io/load_avg_mesh.py` | mesh | MARS_label |
| `data_io/load_spatial_mask.py` | boundary_mask build | already returns Tuple |
| `pipeline_setup/build_boundary_mask` (step 3's) | could be used by L9 | step 2 reimplemented L9 for clarity; the block-diag is identical on bilateral meshes |
| `intra_em/intra_subject_var.py` (step 3's) | L17 | step 2 wraps the same s_psi closed-form in a multi-subject + sigma while-loop |

## Open items

- **cMSHBM**: not wired. Re-introducing it requires extending the
  master kernel to compute an xyz-vMF (`s_muc`, `gamma`,
  `log_xyz_term`) inside Phase D, plus a `Step2Config.mode` dispatch.
