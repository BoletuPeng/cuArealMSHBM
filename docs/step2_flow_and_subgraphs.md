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

`Step2Config.backend ∈ {'cpu', 'gpu'}`. The GPU port (CuPy) covers the
EM-iter master kernel only — the three outer-EM closure leaves
(L16 `intra_em_cost_step2`, L17 `intra_subject_var_loop`,
L18 `inter_subject_var`) stay on CPU. The pipeline syncs ``s_t_nu``
host-side at the seam of each ``vmf_clustering_batch`` call
(~11 MB D2H) and re-uploads the updated ``s_psi`` + ``sigma``
(~5.6 MB H2D) at the start of the next batch — ~100 ms total per
pipeline. Outer-EM closure runs ~30× per pipeline; porting it to GPU
would deliver no measurable speedup at this iteration count.

Each backend's GPU mirror lives next to the CPU module:

| CPU                                | GPU                                            |
|------------------------------------|------------------------------------------------|
| ``step2_em_iter_master/_kernels``  | ``step2_em_iter_master/_kernels_gpu``          |
| ``step2_em_iter_master/session``   | ``step2_em_iter_master/session_gpu``           |
| ``em_iter_master_kernel_streaming``| ``em_iter_master_kernel_streaming_cupy``       |
| ``Step2EmIterSession``             | ``Step2EmIterSessionCUDA``                     |

Both Session classes implement the same duck-typed API
(``upload_initial_state`` / ``cache_mtc`` / ``reset_s_t_nu_from_mtc`` /
``refresh_s_psi_sigma`` / ``run_iter`` / ``sync_to_host``), so the
pipeline doesn't branch on backend below construction.

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

For per-mode wall numbers on the production cohort (S=40, gMSHBM /
dMSHBM, CPU vs GPU eager_bitpacked) see [`## Wall`](#wall) below.

#### GPU backend

The GPU backend takes the same knob but routes it to the device cache
(default `'auto'`):

* `'eager_bitpacked'` — at Session ctor, preload all S subjects' raw
  0/1 BOLD as an `(S, N, T, ⌈D/8⌉)` **bit-packed uint8** cache on
  device (`~69 MB` at S=3 fsa6 T=2 D=1175; `~14.5 GB` at S=200 T=6).
  Each per-iter BOLD visit becomes a device-side `bit-unpack → fp32
  widen + demean + L2-row-norm` RawKernel (~0.85 ms per subject).
  The most-compact eager mode — **the only viable eager path for
  S≥~140 production cohorts on 24 GB cards**.
* `'stream'` — per-iter `.b2nd` disk decode + pinned-host fp32 H2D
  (`~770 MB` per subject visit, ~250 ms decode + ~64 ms PCIe). Safe
  fallback when the packed device cache won't fit.
* `'auto'` (default) — query `cp.cuda.runtime.memGetInfo()`; pick
  `eager_bitpacked` when it fits within free device memory minus
  `gpu_cache_safety_margin_gb` (default 4 GB), otherwise fall through
  to `stream`.

The legacy `'eager'` (unpacked uint8, `(S, N, T, D)` device cache)
mode was removed 2026-06 — bitpacked subsumed it on every axis (same
fp32 contract, 8× less device read traffic, 8× smaller footprint,
~15% faster kernel wall).

Bit-packing convention: LSB-first along D (cell `d` → bit `(d & 7)` of
byte `(d >> 3)`), matching `numpy.packbits` with `bitorder='little'`.
Padding bits past `D` in the last byte are zero.

Falls back to `stream` automatically if `load_packed_into` is
unavailable (e.g., the test-mode `InMemoryProfileLoader`) or the
allocation fails. The fallback emits a `RuntimeWarning`.

Wall numbers on the production S=40 cohort live in
[`## Wall`](#wall) below.

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
│   └── tests/                             1-iter / 3-iter numerical match
│                                          (legacy fresh-Session vs reused-Session)
│
├── step2_io/                              # data I/O leaves
│   ├── load_group_mtc.py                  group.mat (mtc / epsil / labels)
│   ├── load_subject_profiles.py           per-subject (N, D, T) BOLD profiles
│   └── load_subject_gradient.py           per-subject (N, 100, T) diffusion emb
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

Mode-B end-to-end on the YS cohort (**40 subjects × 6 sessions ×
fsaverage6 × L=300, gMSHBM**). RTX 5090 Laptop, 24 GB. 2026-05-20.
Source: internal 40-subject × 6-session cohort run; raw timing log
archived internally.

End-to-end wall on the production GPU path: step 0 — 227.41 s ·
step 1 — 223.85 s · step 2 — **373.94 s (gMSHBM, inter=10)** ·
step 3 — 129.35 s · **total 954.84 s (15.91 min)**. Step 2 is the
single biggest step (39 % of end-to-end) — the EM-iter master kernel
runs `inter × intra × em ≈ 10 × 2 × 2 = 40` outer-loop iters, each
sweeping all 40 subjects.

### Per-variant em_total (S=40, T=6, L=300, max_iter_inter=2)

| variant | CPU em_total (s) | GPU em_total (s) | GPU speedup |
|---|---:|---:|---:|
| gMSHBM | 778.33 | **101.49** | **7.67×** |
| dMSHBM | 562.12 |  **75.93** | **7.40×** |

inter=2 is the standard short-run benchmark cap. Production end-to-end
runs use the default `max_iter_inter=10`; the e2e log above captures gMSHBM GPU at inter=10
as 373.94 s (~4× the inter=2 wall — `em_total` scales linearly in
inter iters; the residual gap vs the strict 5× extrapolation reflects
laptop GPU run-to-run noise between the standalone benchmark and the
e2e production run).

### Per-section breakdown (gMSHBM at inter=2)

| section | CPU (s) | GPU (s) |
|---|---:|---:|
| `load_inputs`               |  0.62 |  0.34 |
| `initialize_params.compose` | 11.17 | 48.73 |
| `session_ctor`              |  0.01 |  3.41 |
| `em_total`                  | 778.33 | 101.49 |
| **TOTAL**                   | **793.98** | **156.74** |

The GPU's `initialize_params.compose` is **slower than CPU**
(48.7 vs 11.2 s) at S=40: `compose_init_state` iterates per-subject
sequentially with a per-subject H2D; the launch + transfer overhead
dominates the per-subject compute win at S=40. `session_ctor` is
3.4 s on GPU because the eager bitpacked BOLD cache
(S=40 × N × T × ⌈D/8⌉ ≈ 1.7 GB) is allocated and populated once.
Both costs are one-shot per pipeline and disappear into the noise at
production inter=10.

### Memory-mode resolution at S=40

`bold_cache_mode='auto'` resolves to `'eager_bitpacked'` for both
variants: S=40 × N(81924) × T(6) × ⌈D(1175)/8⌉ ≈ **1.45 GB** packed
on device, well inside the 4 GB safety margin. The `(S, N, L)` fp32
`s_lambda` buffer (40 × 81924 × 300 × 4 ≈ **3.93 GB**) plus session
scratch (sgemm + softmax + intermediates) keeps peak usage at ~9 GB
of the 24 GB card. (For context: the unpacked-uint8 eager mode would
have run at ~11.6 GB just for BOLD; it was removed 2026-06 because
bitpacked subsumed it on every axis — see also
[`docs/profile_disk_format.md`](profile_disk_format.md).)

### Reproduce

```bash
# end-to-end Mode B (all steps)
python -m arealmshbm.pipeline projects/<name>
```

Per-step wall timing is recorded in the driver's per-run JSON log
under `<project>/logs/`.

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
