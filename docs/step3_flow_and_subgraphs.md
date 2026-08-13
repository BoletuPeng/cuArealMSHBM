# Step 3 — call graph + module map

Structural reference for the production Python step-3 super-call.
Maps each call site in the EM body to the Python module that owns it.

For the migration conventions drawn from this design (validate /
profile pattern, leaf module shape, backend dispatch), see
[`strategy.md`](strategy.md).

## The whole flow

```
Step3Pipeline.run(cfg)                                         [arealmshbm/step3_pipeline]
│
├── load_inputs()
│   ├── load_group_prior(Params_Final.mat)                     data_io/load_group_prior
│   ├── load_spatial_mask(spatial_mask_<mesh>.mat)             data_io/load_spatial_mask
│   ├── load_avg_mesh(lh/rh, inflated)                         data_io/load_avg_mesh
│   ├── load_avg_mesh(lh/rh, sphere)                           data_io/load_avg_mesh
│   ├── fetch_data(profiles + gradient)                        data_io/fetch_data
│   ├── initialize_concentration(D)                            initialize_concentration
│   ├── initialize_params(group_prior, ini_val, T, L, N)       initialize_params
│   └── build_pipeline_setup(theta, ...)                       pipeline_setup
│
├── build_session(inputs)
│   └── VmfClusteringSession(...)                              vmf_clustering
│       (cpu | gpu_elambda | gpu_full backend dispatch)
│
├── intra_em loop (~3 iters)
│   ├── reset Params.kappa, Params.s_t_nu                      inline (small)
│   ├── VmfClusteringSession.run(Params)                       vmf_clustering          ← THE EM BODY
│   ├── intra_subject_var(s_t_nu, sigma, epsil, mu)            intra_em/intra_subject_var
│   ├── intra_em_cost(s_psi, s_t_nu, mu, sigma, ...)           intra_em/intra_em_cost
│   └── convergence test                                       inline (small)
│
├── derive_labels(s_lambda)                                    data_io/save_parcellation
└── save_parcellation(...)                                     data_io/save_parcellation


VmfClusteringSession.run(Params, ...)
└── outer EM-iter loop (~3-5 per intra_em iter)
    │
    ├── M-step (~3 inner iters)                                m_step/MStepSession
    │   ├── batched X·s_t_nu sgemm                             inside m_step (numpy/cupy BLAS)
    │   ├── invAd(D, kappa_mean_resultant)                     m_step/_invad
    │   └── per-(t) X·s_lambda + sigma·s_psi normalize         inside m_step
    │
    ├── ELambdaSession.prepare_em_iter(s_t_nu, kappa)          vmf_clustering/e_step_lambda
    │   (caches X @ s_t_nu sgemm + Cdln per em_iter)
    │
    ├── spatial_connect_prior                                  spatial_priors/ConnectSession
    │
    ├── comp_iter loop (~10 per outer iter)
    │   ├── E-step λ-loop                                      vmf_clustering/e_step_lambda
    │   │   ├── data term + V_lambda assembly                  inside e_step_lambda
    │   │   ├── V_lambda close-form                            V_lambda/Session
    │   │   ├── Cdln(kappa, dim)                               em_stop_criterion/_cdln
    │   │   ├── log_vmf assemble + softmax + boundary mask     inside e_step_lambda
    │   │   └── λ-loop convergence test                        inside e_step_lambda
    │   ├── check_connectedness                                check_connectedness
    │   │   ├── compute_components_general (BFS)               check_connectedness/_kernels
    │   │   └── component_distance                             check_connectedness/component_distance
    │   └── spatial_xyz_prior                                  spatial_priors/XyzSession
    │
    └── EM stop criterion                                      em_stop_criterion/EMStopSession
        ├── log_lambda_prop = T·Cdln + κ·X·s_t_nu              inside em_stop_criterion
        ├── log_theta_cost / log_s_lambda_cost                 inside em_stop_criterion
        ├── 5-term cost reduction                              em_stop_criterion/_kernels
        └── convergence test                                   em_stop_criterion
```

Iteration counts on sub-001 (6 sessions, fsaverage6, 300 ROIs):
~3 intra_em iters × ~3-5 EM iters × ~10 comp_iter × ~3 λ-iter →
~90-150 λ-loop calls per parcellation.

## Module / file map

```
arealmshbm/
├── step3_pipeline/                         # top-level Python entry
│   ├── config.py                           Step3Config dataclass
│   ├── pipeline.py                         Step3Pipeline lifecycle
│   └── profile.py                          per-stage wall-time report
│
├── data_io/                                # disk I/O leaves
│   ├── fetch_data.py                       BOLD profile + diffusion gradient
│   ├── load_group_prior.py                 Params_Final.mat
│   ├── load_spatial_mask.py                spatial_mask_<mesh>.mat
│   ├── load_avg_mesh.py                    fsaverage* mesh (inflated/sphere)
│   └── save_parcellation.py                argmax + Ind_parcellation_*.mat
│
├── initialize_concentration/               vMF kappa init (Debye expansion)
├── initialize_params/                      Params struct init
├── pipeline_setup/                         boundary_mask + neighborhood + ...
├── intra_em/                               outer-loop math
│   ├── intra_subject_var.py                s_psi update
│   └── intra_em_cost.py                    convergence cost
│
├── V_lambda/                               # MRF Potts close-form kernel
│   ├── _kernels.py                         numba close-form
│   ├── setup.py                            neighborhood / candidate-index helpers
│   └── v_lambda.py                         V_lambda Session
│
├── check_connectedness/                    # connectedness BFS + distance
│   ├── _kernels.py                         numba BFS + cdist
│   └── component_distance.py               public functions
│
├── spatial_priors/                         # xyz + connect priors
│   ├── _cdln.py                            Cdln d=3 closed form
│   ├── _kernels.py                         numba kernels
│   ├── spatial_xyz.py                      XyzSession
│   └── spatial_connect.py                  ConnectSession
│
├── m_step/                                 # M-step
│   ├── _invad.py                           inverse-A_d Bessel root solver
│   ├── _kernels.py                         numba kernels
│   └── m_step.py                           MStepSession
│
├── em_stop_criterion/                      # EM stop block
│   ├── _cdln.py                            Cdln general d via Debye
│   ├── _kernels.py                         numba kernels
│   └── em_stop_criterion.py                EMStopSession
│
└── vmf_clustering/                         # the EM body super-call
    ├── _kernels.py                         numba kernels (CPU)
    ├── _kernels_gpu.py                     CuPy kernels (GPU)
    ├── e_step_lambda.py                    ELambdaSession (CPU)
    ├── e_step_lambda_gpu.py                ELambdaSessionCUDA
    ├── vmf_clustering.py                   VmfClusteringSession (CPU)
    └── vmf_clustering_gpu.py               VmfClusteringSessionCUDA
```

Backend selection happens at `VmfClusteringSession(... backend=...)`.
`backend='gpu_full'` redirects construction to `VmfClusteringSessionCUDA`
via `__new__`, keeping the same external API.

## Wall

Mode-B end-to-end on the YS cohort (**40 subjects × 6 sessions ×
fsaverage6 × L=300, gMSHBM**). RTX 5090 Laptop, 24 GB. 2026-05-20.
Source: internal 40-subject × 6-session cohort run; raw timing log
archived internally.

End-to-end wall on the production GPU path: step 0 — 227.41 s ·
step 1 — 223.85 s · step 2 — 373.94 s · step 3 — **129.35 s** ·
**total 954.84 s (15.91 min)**. Step 3 is 14 % of the end-to-end
total — per-subject independent, 40 subjects run sequentially through
the driver at ~3.23 s/sub on `gpu_full`.

### Per-subject wall (3 representative subjects, gMSHBM)

| sub        | CPU wall (s) | GPU wall (s) | GPU EM only (s) | GPU speedup |
|------------|-------------:|-------------:|----------------:|------------:|
| sub-001    |   24.73      |    3.33      |    2.38         |    7.4×     |
| sub-002    |   22.96      |    3.45      |    2.50         |    6.7×     |
| sub-003    |   23.47      |    3.31      |    2.40         |    7.1×     |
| **median** | **23.47**    | **3.33**     | **2.40**        | **7.1×**    |

Setup wall (`load_total` + `session_init`) is ~3.7 s on CPU /
~0.9 s on GPU per subject; the bulk of CPU's overhead is fp32 weight
matrix construction + numba JIT warmup on the first subject. The EM
body wall is the meaningful comparison.

### Per-EM-stage breakdown (median across 3 subjects)

| EM stage              | CPU (s) | GPU (s) | GPU speedup |
|-----------------------|--------:|--------:|------------:|
| `m_step`              |   3.24  |   0.24  |   13.5×     |
| `spatial_connect_prior` |  0.31  |   0.04  |    7.8×     |
| `e_step_lambda_loop`  |   9.69  |   1.05  |    9.2×     |
| `check_connectedness` |   3.43  |   0.31  |   11.1×     |
| `spatial_xyz_prior`   |   1.23  |   0.21  |    5.9×     |
| `em_stop_criterion`   |   2.57  |   0.05  |   51.4×     |
| **EM total**          | **20.47** | **1.90** | **10.8×** |

`e_step_lambda_loop` is the dominant EM stage on CPU (~half the EM
wall) and benefits ~9× on GPU via the fused E-step kernel + cuBLAS
sgemm on `X @ s_t_nu`. `m_step` wins 13× from cuBLAS + RawKernel-based
M-step inner-loop. `em_stop_criterion`'s 51× isolated speedup is real
(fp64 cost-reduction sums + Cdln vectorise well on GPU) but it runs
only once per EM iter, so its share of total is small.

Iteration counts at convergence on these subjects:
`iter_intra_em = 2–3`, `iter_count_em_total = 8–9`,
`iter_count_lambda_total ≈ 101`, `iter_count_m_total = 32–127` (CPU)
or `34–75` (GPU; the M-step inner loop convergence test diverges
slightly between backends — see precision contract notes upstream).

### Cohort scaling

End-to-end step 3 wall on GPU is **129.35 s for S=40** (~3.23 s/sub),
matching the per-subject profile median (3.33 s) within 4 %. The
extra 0.13 s/sub in the e2e run is `Step3Pipeline` construct/teardown
+ `.mat` write per subject. Extrapolating from the median CPU
per-subject wall: 40 × 23.47 ≈ **939 s (~15.6 min) if step 3 ran CPU**.

### Pipeline-parallel by phase (GPU, K ≥ 2)

The per-subject pipeline runs three phases sequentially — LOAD
(disk + setup + H2D mirrors), EM (intra_em loop), SAVE (.mat write).
On `gpu_full` the EM body (~2.4 s) gates the per-subject wall; LOAD
(~0.9 s) and SAVE (~0.05 s) are off-GPU work that the wall would
otherwise serialize behind EM. The driver pipelines by phase across
subjects: one stage thread per phase, with cupy streams on LOAD and
EM (cross-stream event sync on the LOAD→EM device-buffer handoff),
so subject K+1's LOAD overlaps subject K's EM and subject K-1's SAVE
overlaps both.

Steady-state per-subject wall becomes `max(t_LOAD, t_EM, t_SAVE)`
instead of the sum. On a 3-subject Mode A smoke
(sub-001…sub-003, gMSHBM, gpu_full) the stage-pipelined driver
delivered **2.86 s/sub** (8.57 s for K=3) against the serial baseline
of ~3.33 s/sub — a 14 % step-3 reduction. The relative win grows with
K because the LOAD/SAVE → EM overlap stabilises after the first
subject's pipeline-fill cost.

Memory budget: with queues capped at 2 in-flight, ~3 subjects can be
on GPU at once (peak ~9.5 GB on the YS profile reference) — well
within the 24 GB device budget. CPU backend and `K=1` runs stay on
the serial loop (no overlap to amortise). See
[`arealmshbm/pipeline/_step3_stage_pipeline.py`](../arealmshbm/pipeline/_step3_stage_pipeline.py).

### Reproduce

```bash
# end-to-end Mode B (all steps)
python -m arealmshbm.pipeline projects/<name>
```

Per-step wall timing is recorded in the driver's per-run JSON log
under `<project>/logs/`.
