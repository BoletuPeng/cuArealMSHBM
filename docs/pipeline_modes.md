# Pipeline Modes — A vs B

Two distinct usage modes that differ in **where the group prior comes
from** and consequently which pipeline steps run.

Both are dispatched by the unified driver
([`arealmshbm/pipeline/`](../arealmshbm/pipeline/)) via
`pipeline_config.json:mode`. Mode A is the routine personalisation
path (per-subject parallel, no step 2). Mode B trains a group prior on
the local cohort via the step-2 super-call (gMSHBM / dMSHBM) — see
[`step2_flow_and_subgraphs.md`](step2_flow_and_subgraphs.md) for the
call graph.

## At a glance

| Aspect | Mode A — HCP prior | Mode B — self-trained prior |
|---|---|---|
| Group prior source | Pre-trained, shipped (40 HCP subjects) | Estimated on your own dataset |
| Steps run | step 0 + step 1 + step 3 | step 0 + step 1 + step 2 + step 3 |
| Step 2 (group EM) | skipped | required (heavy coupled EM, ~hours) |
| Subject coupling | none — each subject independent | all subjects coupled in step 2 |
| Parallelization | trivial per-subject parallel | step 2 is one global EM; only step 3 is per-subject |
| Wall time per subject | ~17 min (measured, see profiling) | ~hours (step 2 once) + ~17 min/subject |
| Use case | routine personalization, deployment | new dataset where HCP normative doesn't generalize |
| `pipeline_config.json:mode` | `modeA_single` (K=1) / `modeA_batch` (K≥1) | `modeB_train_prior` (K≥2) |

## What runs in each mode

### Mode A — "personalize this subject against the HCP atlas"

```
input: BOLD time series for one subject, N sessions
        ↓
[step 0]  arealmshbm.step0_pipeline
          → diffusion embedding (gradients/sub<S>/{lh,rh}_emb_100_distance_matrix.npy)
        ↓
[step 1]  arealmshbm.pipeline.step1_runners.run_generate_profiles  (×N sessions)
          → RSFC profile (correlation with fsaverage3 seeds, packed (N, T, ⌈D/8⌉) per subject)
        ↓
[prior]   group prior must already sit at
          project_dir/priors/gMSHBM/beta<X>/Params_Final.mat
          (project creator staged it; no estimation — replaces step 2)
        ↓
[step 3]  arealmshbm.step3_pipeline
          → run EM with the loaded prior held FIXED
          → output: lh_labels, rh_labels (per-vertex parcel assignment)
```

The unified driver chains all of this; each subject runs fully
independently in Mode A.

### Mode B — "build a prior on this dataset, then personalize"

```
inputs: BOLD time series for K subjects (K=40 in our fork), N sessions each
        ↓
[step 0]  arealmshbm.step0_pipeline × K subjects
        ↓
[step 1]  arealmshbm.pipeline.step1_runners.run_generate_profiles × K × N
        ↓
[step 2]  arealmshbm.step2_pipeline
          → run a coupled EM across ALL K subjects to estimate
            mu, theta, epsil, sigma, kappa
          → output: project_dir/priors/gMSHBM/beta<X>/Params_Final.mat
        ↓
[step 3]  arealmshbm.step3_pipeline × K
          → same as Mode A's step 3, but using the freshly-estimated prior
```

The driver orchestrates all four steps; step 2 is the heavy coupled
EM. cMSHBM is not wired for step 2 (raises `NotImplementedError`);
gMSHBM and dMSHBM both ship.

## What the prior actually is

A `Params_Final.mat` contains a `Params` struct with fields:

| Field | Shape | Meaning |
|---|---|---|
| `mu` | D × L | Group cluster centroids — typical RSFC profile of each parcel |
| `theta` | N × L | Spatial prior — P(parcel l at vertex n) |
| `epsil` | 1 × L | Inter-subject vMF concentration |
| `sigma` | 1 × L | Intra-subject vMF concentration |
| `kappa` | 1 × L | Inter-region vMF concentration |
| `s_psi`, `s_t_nu`, `s_lambda` | (training-only state) | Empty in the shipped HCP prior |

For fsaverage6 with 100 ROIs: `mu = 1175×100`, `theta = 81924×100`. D
depends on the seed mesh (fsaverage3 cortex vertex count); N = 2 hemis
× 40962 vertices.

In Mode A, only `mu`, `theta`, `epsil`, `sigma` (and `kappa` reset to a
default) are loaded — they enter step 3 as fixed parameters. The
subject-level state (`s_psi`, `s_t_nu`, `s_lambda`) is initialized
fresh and re-estimated by step 3's EM.

## Where the prior lives

At run time the driver reads the Mode A prior from **one place only**:

```
<project>/priors/<variant>/beta<beta_scalar>/Params_Final.mat
```

Putting it there is the project creator's job. The pipeline has no
`prior_source` knob and never reaches outside the project — no HCP/CBIG
fallback, no cross-project lookup. Mode B writes this same slot itself
via step 2.

Sources a creator can copy from (toolkits, never read at run time):

| Source | What | Coverage |
|---|---|---|
| `arealmshbm/data/group_priors/HCP_{fsaverage6,fs_LR_32k}_40sub/100/...` | Shipped in this fork | K=100 example only |
| `$CBIG_CODE_DIR/.../Kong2022_ArealMSHBM/lib/group_priors/` | CBIG checkout (creator-side; not read at runtime) | full K=100…1000 |
| another project's `priors/<variant>/beta<X>/Params_Final.mat` | a prior `modeB_train_prior` run | whatever it trained |

`cp` the chosen `Params_Final.mat` into the project slot above before
running Mode A (and stage the matching spatial mask similarly).

## Decision matrix

| Situation | Use |
|---|---|
| ≥1 subject, want individual parcellation, HCP prior exists at your ROI count | Mode A |
| New species / pediatric / clinical population where HCP normative may not generalize | Mode B (estimate on a representative subset, then Mode A on the rest) |
| <40 subjects of your own | Mode A — step 2 needs ≥ several dozen subjects |
