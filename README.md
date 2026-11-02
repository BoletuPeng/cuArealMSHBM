# cuArealMSHBM

<a id="english"></a>
**English** | [简体中文](#chinese)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21922832.svg)](https://doi.org/10.5281/zenodo.21922832)

**CPU/GPU-accelerated Areal-MSHBM** — individual-specific cortical
parcellation from resting-state fMRI, ported end-to-end to Python with
numba (CPU) and CuPy (GPU) backends. Validated at large scale (300+
subjects, 1,800+ fMRI sessions) against the original MATLAB
implementation. **v2.2.0** gives step 2 the reference implementation's
initialisation (the soft masked posterior with `theta_num`, on both
backends): against the reference's own step 2 on the same inputs, the
trained gMSHBM prior's agreement rises from 78.9 % to 94.1 % (dMSHBM:
71.9 % to 81.2 %), at no cost in speed or memory. Since v2.1.0 the
candidate-set session is the one GPU backend for steps 2 and 3 (step 3:
all three variants), BOLD is ingested on the GPU through nvCOMP only,
and — when the packed cohort fits on the device — no host copy of it is
kept: a single subject goes from raw BOLD to parcellation in
**6.2 s** end to end on one laptop GPU, and a 40-subject
cohort goes from raw BOLD through group-prior training to all 40
parcellations in **74.0–75.6 s** (step 2, the
group-prior training, alone: 16.5 s).

---

## Why Areal-MSHBM

The **Areal multi-session hierarchical Bayesian model**
([Kong et al., 2021, *Cerebral Cortex*](https://doi.org/10.1093/cercor/bhab101);
[CBIG reference implementation](https://github.com/ThomasYeoLab/CBIG))
is an excellent individualized parcellation approach: it pools
information across sessions and subjects in a principled generative
model, yielding individual-specific areal-level parcellations with
homogeneity and behavior-prediction advantages over group atlases.

In practice, however, the reference pipeline is slow, memory-hungry,
and engineered around Linux HPC clusters (MATLAB + compiled MEX,
cluster job submission, per-step intermediate dumps). A single-subject
parcellation takes tens of minutes and ~11 GB of RAM on a workstation;
cohort-scale prior training is out of reach without a cluster. That
deployment profile has been a real barrier to the method's adoption —
which its scientific quality does not deserve.

## Our vision

Our goal is to bring
modern GPU-acceleration engineering to promising neuroimaging methods
like this one, so that they can be evaluated — and used — in detail at
a scale that was previously impractical. A method that once required a
compute cluster should run on one laptop GPU; an evaluation that once
sampled a handful of subjects should sweep whole cohorts.

## What this project does

cuArealMSHBM re-implements all four pipeline steps (RSFC gradient
embedding → connectivity profiles + vMF initialization → group-prior
EM training → per-subject parcellation EM) in Python, with
interchangeable per-step backends:

- **CPU backend** — numba-JIT kernels, no GPU required.
- **GPU backend** — CuPy + hand-written RawKernels.
- **Candidate-set GPU backends** (`backend_step2` / `backend_step3` =
  `gpu`) — the EM runs on the candidate set implied by the group
  prior's support instead of the dense (N × L) posterior, so an entire
  cohort stays resident on the device for the whole run when it fits in
  VRAM (step 2 streams subjects otherwise).

Engineering highlights:

- **Faster, more accurate numerics for Bessel-type functions.** The
  vMF normalization constants and concentration updates are computed
  via closed-form and asymptotic expansions of `log I_ν` instead of
  the reference's generic Bessel route. Audited against mpmath at
  50-digit precision: the d=3 vMF log-normalizer is accurate to
  0.31 × fp32 eps — up to **19× more accurate** than the MATLAB
  original at κ = 1000 — while being branch-free and JIT/GPU-friendly.
- **Bit-packed VRAM compression.** Binarized BOLD profiles live on
  device as bit-packed `uint8 (N, T, ⌈D/8⌉)` tensors — an 8×
  compression that keeps a 40-subject × 6-session cohort's BOLD at
  2.9 GB of VRAM (vs ~23 GB unpacked), with popcount-based kernels
  consuming the packed form directly.
- **Kernel fusion and operator tuning** across the bottleneck steps:
  fused profile-generation kernels, fused E-step softmax chains with
  per-subject fp64 scratch (fp32 storage, fp64 where precision
  demands it), tuned reductions, and stream-pipelined BOLD decode that
  hides I/O behind GPU compute.

The result: on the heaviest step the speedup over the MATLAB
reference reaches **three orders of magnitude** on a single Blackwell
GPU (step-3 parcellation EM: 763.1 s → 0.55 s, ~1,400×; step-0 gradient
embedding: 233.6 s → 2.3 s, ~100×), and the full single-subject
pipeline runs **~170× faster** end-to-end.

## Status

**This is the fourth release (v2.2.0)**, containing the CPU and GPU
backends; the GPU backends for steps 2 and 3 are the candidate-set
sessions, and step 3's covers all three variants. In large-scale validation — over **300 subjects
and 1,800 fMRI sessions** to date — it shows alignment with the
original implementation's outputs at the level the numerics permit
(~98% per-vertex agreement; bit-exactness across BLAS/reduction-order
boundaries is mathematically unattainable), together with dramatic
speed and memory savings (tables below).

The project is still being polished. **All precomputed data** (mesh
bundles, step-0 caches, pre-trained HCP group priors, spatial masks)
**will be deployed by the installer of the GUI distribution, which is
under development.** In this source release those assets are
deliberately absent — see
[`arealmshbm/data/README.md`](arealmshbm/data/README.md) for the
layout the installer stages.

## Benchmarks

All numbers measured on one Windows 11 laptop: RTX 5090 Laptop GPU
(Blackwell, 24 GB), Python 3.13, numba 0.63.1, CuPy 13.6.0. The
MATLAB baseline is the CBIG reference pipeline **with its compiled MEX
hot paths** (`mtimesx`), run on the same machine — a stronger baseline
than a typical cluster node. Every cuArealMSHBM **wall-time** figure
below was re-measured on the v2.2.0 tree (2026-11-02), and each "total" is
end-to-end driver wall time on a project with no caches (interpreter
start and input validation included), with the numba and CuPy kernel
caches warm on disk. The label-agreement rows are carried over from the
v1.0.0 measurements. The memory rows are measured on the v2.2.0 tree:
host is the peak resident set of the driver process (except where a row
names another process), device is the peak `nvidia-smi` usage above the
idle desktop. Numerically, v2.2.0 reproduces v2.1.0 (itself bit-identical
to v2.0.0) through step 1: on the 40-subject cohort below, steps 1–3 run on
the v2.2.0 tree from the same step-0 output reproduce the packed profiles,
cohort averages, spatial mask and `group.mat` bit for bit; step 2's prior
differs by design (the new initialisation — see § Agreement below), and
downstream of it 9.4 % of the cortical vertices across the 40
label files change label (per subject 7.0–12.6 %).

### Mode A — individual parcellation under a pre-trained prior

Single subject × 6 sessions, fsaverage6 (74,947 cortical vertices),
K = 300 parcels, gMSHBM, β = 5, w = 50, c = 10:

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 0 — gradient embedding | 233.6 s | 25.9 s | 2.3 s |
| step 1 — profiles + init | 47.2 s | 10.2 s | 1.0 s |
| step 3 — parcellation EM | 763.1 s | 23.0 s | **0.55 s** |
| **total (end-to-end wall)** | **1073.0 s** | **60.5 s (18×)** | **6.2 s (173×)** |
| per-vertex agreement vs MATLAB labels¹ | — | 98.33 % | 98.05 % |
| peak host memory (whole run, driver process) | ~11 GB RAM (step 3) | 6.5 GiB | **2.7 GiB (4× less)** |

GPU column: `backend_step0` / `backend_step1` / `backend_step3` = `gpu`.
The MATLAB path additionally required
a one-off 1,527 s GIFTI→NIfTI conversion pass; cuArealMSHBM ingests
`.func.gii` natively.

Against v1.0.0 on the same machine and config, GPU step 0–3 compute
drops **15.8 s → 3.9 s (4.1×)**: the step-0 Δ-stepping SSSP rewrite,
the fused whole-subject step-1 GPU leaf, and the step-3 candidate-set
backend (all v2.0.0); v2.1.0 holds that wall while step 1, and step 2
in its device-resident cache mode, stop keeping host copies of the
packed profiles, and v2.2.0 leaves Mode A's code path untouched. The
6.2 s wall adds ~1.0 s of one-time package import
(the Mode A min-input check imports the step-3 package to resolve the
prior path) and ~0.6 s of interpreter start — fixed costs that do not grow with the cohort. Repeat runs in fresh processes land at
6.1–6.3 s. A process whose numba cache has been deleted pays
another ~7 s of JIT compilation; one whose CuPy kernel cache is
also gone compiles every GPU kernel first (31 s wall, once per
machine).

¹ Cortical vertices, medial wall excluded. The two Python backends
agree with each other at 98.21 %. Bit-exact MATLAB↔Python equality is
not attainable in the first place: the residual disagreement is the
combined effect of several benign sources — BLAS reduction-order
differences between numerical libraries (every backend has its own
numerical noise band, and the GPU additionally shows small run-to-run
variation), different numerical routes for the Bessel-type functions
(ours evaluates `log I_ν` through closed-form and asymptotic expansions,
audited against mpmath: `arealmshbm/em_stop_criterion/_cdln.py` on the
CPU, the d = 3 closed form on the device, pinned bit-equal in fp32 by
`arealmshbm/em_stop_criterion/tests/test_cdln_d3_oracle.py`), and a few small, deliberate
implementation choices where the port diverges from the reference.
The four-layer framework we use to judge whether differences of this
kind are functionally meaningful is described in
`docs/precision_impact_analysis.md`.

### Mode B — group-prior training on a local cohort

40 subjects × 6 sessions, fsaverage6, K = 300, gMSHBM:

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 2 — group-prior EM (production `max_iter_inter=10`; 48 EM iterations on both backends) | not feasible on a workstation² | 2,573.5 s | **16.5 s (156×)** |
| full pipeline steps 0–3, whole cohort (production run) | not feasible on a workstation² | — | **74.0–75.6 s (≈1.2 min)** |
| device memory, step 2, whole 40-subject cohort | **≈ 370–490 GB RAM** (estimated²) | — | **peak 5.6 GiB of 24 GB VRAM** |
| host memory, step 2 alone (standalone `arealmshbm.step2_pipeline.profile` process), whole 40-subject cohort | (the same ≈ 370–490 GB²) | 11.1 GiB | **peak 1.1 GiB** |

The step-2 GPU column is the `gpu` backend, the candidate-set session
(since v2.0.0; the dense CuPy port that held the name in v1.0.0 is gone).
Measured as a single run in a fresh process (a warm in-process repeat is
15.2 s). The CPU figure is likewise a single run in a fresh process, 48 EM iterations (the v2.1.0 tree measured 2,945.6 s and the v2.0.0 tree 2,638.7 s on the same shape; the CPU EM kernel is the same code in all three, so the spread is run-to-run variation of a 43–49-minute all-core run on a laptop). The initialisation change costs step 2 nothing: the v2.1.0 tree, run in the same session, repeats the standalone GPU step 2 in 15.0–15.1 s warm against 15.0–15.4 s on v2.2.0, with 48 EM iterations on both; the 15.8 s / 14.1 s published for v2.1.0 were this laptop on another day. The full-pipeline row is three production runs (74.0, 75.0 and 75.6 s); in the fastest
one step 2 took 14.6 s beside step 0 40.4 s, step 1 8.4 s and step 3
8.1 s; step 0 accounts for 0.8 s of the 1.5 s between the fastest and the slowest run, steps 1 and 3 for 0.3 s each, step 2 for none. The host-memory row
was added in v2.1.0 and is taken in a standalone step-2 process: in its
device-resident cache mode (`eager_bitpacked`, which `bold_cache_mode:
"auto"` picks whenever the packed cohort, the session's resident state
and a safety margin fit in free VRAM) the session keeps no host copy of the
packed cohort — each subject is decoded straight into a two-slot
page-locked ring and uploaded — so the host working set stops growing
with the cohort; inside the end-to-end driver the process peak during
step 2 is 2.7 GiB, most of it still held from steps 0–1. The `stream`
mode for cohorts that do not fit in VRAM keeps one subject on the device
and, host RAM permitting, a host cache of the packed cohort. On
`backend_step1: "gpu"` step 1 keeps no host copy of the packed cohort
either, whatever its size: each session is folded into device sums as it
is packed. The `cpu` backend's average reads every subject's `.b2nd`
back into host memory.

**Against v1.0.0, on this same machine.** v1.0.0 published a step-2 pair of
794.0 s (CPU) / 156.7 s (dense GPU); those were taken at `max_iter_inter=2`,
the short-run benchmark cap, not the production default of 10 — see
§ Production-cohort walls in
[`docs/step2_flow_and_subgraphs.md`](docs/step2_flow_and_subgraphs.md).
Re-running that exact shape on the v2.0.0 tree puts the `gpu` backend at
**11.1 s**: **14×** faster than v1.0.0's GPU figure and **72×** faster than
its CPU figure. Its measured device peak is **6.7 GiB at both
`max_iter_inter=2` and `=10`** — flat in the outer-iteration count, because
the candidate-set session allocates once and reuses. v1.0.0's dense GPU
backend measured **21 GiB** on the same shape and spilled into host memory
at `inter=10`; v2.0.0 removed it, for that and because its flush-to-zero
fp32 E-step was the less accurate of the two.

² Estimated from the reference source
(`CBIG_ArealMSHBM_gMSHBM_estimate_group_priors_{parent,child}.m`), not
measured — because it *cannot* be measured on a workstation. The
reference's step 2 is architected as **1 parent + S child cluster
jobs** (its own header: *"user should submit 1 parent job and num_sub
child jobs"*) that exchange `.mat` files through the filesystem at
every inner M-step iteration. Children block mid-iteration waiting for
the parent's reply, so subjects cannot be processed serially: all 40
children must be resident **concurrently**. Each child, at this
benchmark shape (N = 81,924 vertices, D = 1,175 profile dims, T = 6
sessions, L = 300 parcels, all fp64), holds the profile store
`data_series` (N × D × T, 4.6 GB) plus the further full-size copies
the code materializes in *different layouts* — per-session N × D slice
copies, N × L × T κ-update products from `mtimesx`, transposed
L × N × T log-vMF stacks, an (N, L, T) spatial-prior stack —
for a working set of **≈ 9–12 GB per subject**; the parent holds the
(N, L, S) posterior (7.9 GB at S = 40) plus same-shape temporaries.
Total: ≈ 370–490 GB of simultaneously-resident RAM across 41
processes, plus per-iteration `.mat` traffic through shared storage —
a compute cluster is a hard requirement, not an optimization.

cuArealMSHBM collapses all of this onto one device: each binarized
profile exists in VRAM exactly **once**, as a bit-packed
`uint8 (N, T, ⌈D/8⌉)` tensor (72 MB per subject, 2.9 GB for the
entire cohort — vs ~23 GB unpacked, and vs the reference's ~4.6 GB
of fp64 per subject *before* its extra copies), shared by reference
across all kernels. The posterior is stored fp32 instead of fp64, and
only on the spatial mask's support — an (S, P) array with P = 586,522
(vertex, parcel) cells, 2.4 % of N × L (about 94 MB for all 40
subjects, where a dense (S, N, L) layout would take 3.93 GB; fp64
scratch only where the softmax demands it) — and kernel fusion
means the N × L × T intermediates the
reference round-trips through RAM and disk are never materialized at
all. Measured peak device usage on the `gpu` backend: **5.6 GiB** — **the
whole 40-subject cohort fits in less memory than the reference needs for
a single subject**, with most of a 24 GB card left over. How far the
cohort can grow before the (S, N, ·) terms dominate has not been
measured.

## Agreement with the reference implementation

### Mode A — individual parcellation under the same prior

Measured against **stock CBIG as published**, on 30 subjects across three
cohorts — K = 300, β = 5, w = 50, c = 10, fsaverage6 target /
fsaverage3 seed, with the same group-prior file loaded on both sides.
Backends: `backend_step0` / `backend_step1` / `backend_step3` = `gpu` —
step 3 on the candidate-set session for all three variants. The cMSHBM
column was re-measured on it for v2.1.0; the gMSHBM and dMSHBM columns are
the v2.0.0 measurement, whose step-3 backend is this same session under
its earlier name. Mode A does not run step 2, so v2.2.0 leaves this table
unchanged.

Per-vertex label agreement over cortical vertices, medial wall excluded —
it is label 0 on both sides and would otherwise count as free agreement.
All-vertex figures in parentheses.

| cohort | subjects × sessions | gMSHBM | cMSHBM | dMSHBM |
|---|---|---|---|---|
| in-house | 10 × 6 | 95.12 % (95.52 %) | 99.35 % (99.40 %) | **99.99 %** (99.98 %) |
| OpenNeuro `ds004466` | 10 × 4 | 93.20 % (93.63 %) | 98.42 % (98.49 %) ᵃ | **99.84 %** (99.69 %) |
| OpenNeuro `ds000221` | 10 × 4 | 93.07 % (93.12 %) | 98.13 % (98.24 %) ᵇ | **99.67 %** (99.12 %) |

ᵃ n = 9 · ᵇ n = 8 — the reference implementation itself
fails on three cMSHBM subjects (`sub2ind` index-out-of-range inside
`CBIG_ArealMSHBM_component_distance`, reproducible on retry), so no reference
labels exist for those cells. cuArealMSHBM completes all three. Every other
cell is over the full 10 subjects.

**Why gMSHBM sits lower than cMSHBM and dMSHBM.** cMSHBM and dMSHBM never
read step 0's gradient output; gMSHBM does. Step 0 is where the port makes a
few deliberately different implementation choices from the reference —
most visibly in how the watershed driver chunks its sample columns, and in
the transform applied to the gradient distance matrix before the diffusion
embedding. A controlled experiment that reverts both to the reference
behaviour recovers 0.7–1.4 percentage points of gMSHBM's gap, so those
choices account for part of it, not all. If closer agreement with the
original turns out to matter for downstream use, we will revisit them.

The 98.33 % / 98.05 % row in the Mode A table above answers a different
question: it is measured against a MATLAB reference whose step 0 carries the
same two changes the port does. Same port, different reference — the two
sets of numbers are not in conflict.

### Mode B — step 2 (group-prior training) against the reference implementation

Mode B has no end-to-end MATLAB baseline on a workstation (² above), so
step 2 is checked on its own. The reference implementation's step 2 was run
unmodified on 8 subjects × 2 sessions of the in-house cohort (K = 300,
10 outer iterations; gMSHBM β = 5 and dMSHBM), and cuArealMSHBM's step 2
(`gpu` backend) was run on the same inputs — the reference's own step-0/1
outputs. Agreement is the per-vertex argmax of the trained `theta` over the
74,947 cortical vertices, counted directly / with exact ties allowed
(10–13 % of the rows carry tied `theta` values, multiples of 1/8).

| trained prior vs the reference's | gMSHBM β = 5 | dMSHBM |
|---|---|---|
| v2.1.0 (one-hot argmax initialisation) | 78.89 % / 89.51 % | 71.86 % / 84.91 % |
| **v2.2.0 (the reference's initialisation)** | **94.07 % / 98.16 %** | **81.19 % / 92.38 %** |
| v2.2.0 with single-precision E-step normalisation (experiment, not shipped) | 99.90 % / 99.97 % | 98.82 % / 99.69 % |

v2.2.0's initialisation is the reference's — `ini_s_lambda` in the child
script, the mask / normalise / `theta_num` block in the parent — evaluated
in single precision as the reference does; the `cpu` and `gpu` backends
produce it bit-identically. What remains is one precision choice: the
reference normalises the E-step's fp32 `exp` values, so a vertex whose
in-mask cells all underflow is zeroed (0/0), whereas cuArealMSHBM
normalises in fp64 and keeps those rows. Without a spatial term that
matters far more — the reference's first E-step after this initialisation
zeroes 23 % of the (vertex, subject) rows in dMSHBM — which is why the
dMSHBM figure trails gMSHBM's. Reproducing that underflow (last row) is
deliberately not shipped: it is a precision artefact of the reference, not
part of the model. On the 40-subject HCP training set (K = 400, 4 runs),
the gMSHBM prior trained by v2.2.0 matches CBIG's published 40-subject
prior on 92.2 % of vertices (v2.1.0: 89.8 %); what remains of that gap lies
upstream of step 2.

## On the original paper's claims

A large-scale stability analysis and an independent third-party
evaluation of Areal-MSHBM will be published in the near future. Our
preliminary results indicate that the original paper's claims about
the method's advantages and improvements are **honest, reliable, and
reproducible — and that they replicate on datasets beyond those used
in the original publication.**

## Quick start

A *project* is a directory under `projects/` with one BOLD manifest
and one config:

```
projects/<name>/
  bold_inputs.json        # per-(subject, session, hemi) BOLD .func.gii paths
  pipeline_config.json    # mode + variant + hyperparameters (schema v2)
```

Templates: [`projects/sample_modeA_single/`](projects/sample_modeA_single/),
[`projects/sample_modeA_batch/`](projects/sample_modeA_batch/),
[`projects/sample_modeB/`](projects/sample_modeB/).

### Before the first run

Two things live outside the repository and must be staged once.

**1. Precomputed assets** — published as release assets under the
[`assets-v1`](https://github.com/BoletuPeng/cuArealMSHBM/releases/tag/assets-v1) tag (versioned independently of the
code releases). Extract into `arealmshbm/data/precomputed/`:

```bash
curl -L -O https://github.com/BoletuPeng/cuArealMSHBM/releases/download/assets-v1/avg_mesh-fsaverage6.tar.gz
curl -L -O https://github.com/BoletuPeng/cuArealMSHBM/releases/download/assets-v1/step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz
tar -xzf avg_mesh-fsaverage6.tar.gz -C arealmshbm/data/precomputed/
tar -xzf step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz -C arealmshbm/data/precomputed/
```

Both are **required**. `avg_mesh` (4.3 MB) is where all mesh geometry
comes from — it is never rebuilt at runtime.
`step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz` (82.6 MB) is the step-0
input cache: the pipeline does contain a builder for it
(`arealmshbm.precompute.step0_inputs_builder`), but that reads raw
FreeSurfer sphere surfaces from a CBIG checkout at a layout current CBIG
revisions do not use, so treat the download as the supported path.

Extract both into `arealmshbm/data/precomputed/`. `$MSHBM_PRECOMPUTED_ROOT`
relocates the `avg_mesh/` bundles only — the step-0 cache is always read
from inside the package — so do not use it to move the pair elsewhere: the
run would fail inside step 0 with an unrelated-looking message about a
missing CBIG checkout.

**2. Atlas directory** — set `MSHBM_ATLAS_DIR` to a directory whose
`<targ_mesh>/label/` holds the aparc and Schaefer2018 Kong2022 `.annot`
files (staged from a CBIG checkout). Step 1 reads them live; there is no
fallback.

Group priors and spatial masks are **not** shipped here — they are
byte-identical to CBIG's and are taken from a CBIG checkout when needed.
Full layout and provenance:
[`arealmshbm/data/README.md`](arealmshbm/data/README.md).

Run the full pipeline:

```bash
python -m arealmshbm.pipeline projects/<name>
```

or programmatically:

```python
from arealmshbm.pipeline import Pipeline
result = Pipeline("projects/<name>").run()
```

Parcellations are written to
`projects/<name>/ind_parcellation_<variant>/<N>_sess/beta<B>/` as
`Ind_parcellation_MSHBM_sub<S>_w<W>_MRF<C>_beta<B>.mat` (for dMSHBM the
`beta<B>/` directory and the `_beta<B>` suffix are absent).

### Operating modes

| mode | step path | group prior |
|---|---|---|
| `modeA_single` (1 subject) | 0 → 1 → 3 | pre-staged in the project |
| `modeA_batch` (N subjects) | 0 → 1 → 3 | pre-staged in the project |
| `modeB_train_prior` (cohort) | 0 → 1 → 2 → 3 | trained on your cohort by step 2 |

Mode A reads `<project>/priors/<variant>/beta<B>/Params_Final.mat`
(dMSHBM: `priors/dMSHBM/Params_Final.mat`, no `beta<B>/` segment);
staging it there is the project creator's job. Variants: gMSHBM and
dMSHBM ship; cMSHBM step 2 is not wired (its step 3 is). See
[`docs/pipeline_modes.md`](docs/pipeline_modes.md) and
[`docs/pipeline_variants.md`](docs/pipeline_variants.md).

### Choosing a backend

Each step picks its backend independently in `pipeline_config.json`
(`backend_step0` … `backend_step3`). `cpu` always works; the GPU
backends carry the preconditions below:

| key | values | notes |
|---|---|---|
| `backend_step0` | `cpu`, `gpu` | GPU step 0 is not bit-reproducible run to run — the measured band is in [`docs/step0_flow_and_subgraphs.md`](docs/step0_flow_and_subgraphs.md) |
| `backend_step1` | `cpu`, `gpu` | packed profiles, averages, spatial mask and `group.mat` labels are bit-identical either way; `group.mat`'s `mtc` / `epsil` agree to within a few fp64 ULP (fp64 reduction order); `gpu` ingests BOLD through nvCOMP (`nvidia-nvcomp-cu12` required) and needs `step1.reduction_dtype: "float64"` (anything else is rejected at validation) |
| `backend_step2` | `cpu`, `gpu` | `gpu` is the candidate-set session: it needs `seed_mesh: "fsaverage3"`, K ≤ 512 and, for gMSHBM, ≤ 11,772 gradient components — any other shape is rejected at validation, so select `cpu` for it; `psutil` required |
| `backend_step3` | `cpu`, `gpu` | `gpu` is the candidate-set session for gMSHBM, cMSHBM and dMSHBM: it needs `seed_mesh: "fsaverage3"` (else `cpu`) and, for gMSHBM / cMSHBM, a CUDA toolkit at `CUDA_PATH` plus a device that supports cooperative launch |

Design notes and the measured backend-vs-backend agreement are in
[`docs/step2_sparse_design.md`](docs/step2_sparse_design.md) and
[`docs/step3_sparse_design.md`](docs/step3_sparse_design.md).

**Upgrading a v2.1.0 project.** No config key changes. Step 2's
initialisation changed, so a Mode B project re-run on v2.2.0 trains a
different group prior than v2.1.0 did, and the individual parcellations
downstream of it move with it (9.4 % of cortical vertices on the
40-subject cohort above). Mode A runs under a given prior file are
unchanged: steps 1 and 3 are the same code.

**Upgrading a v2.0.0 project config.** `backend_step3` accepts `cpu` and
`gpu` only: write `"gpu"` where a config said `"gpu_sparse"` (the same
session), and choose `"gpu"` or `"cpu"` where it said `"gpu_full"` or
`"gpu_elambda"` (the dense GPU EM is gone; against the reference labels the
candidate-set session ties or leads it). `w` must be ≥ 1 on every backend:
at `w = 0` the E-step's `w · log θ` term is undefined outside the prior's
support, in the reference implementation as well. Four things v2.0.0
used only when it found them are requirements: `python-isal` (every
backend), `nvidia-nvcomp-cu12` (`backend_step1: "gpu"`), `psutil`
(`backend_step2: "gpu"`), and a CUDA toolkit at `CUDA_PATH` plus a device
that supports cooperative launch (`backend_step3: "gpu"` with gMSHBM /
cMSHBM) — select `"cpu"` for a step whose requirement the machine does
not meet.

**Upgrading a v1.0.0 project config.** The v2.0.0 note above applies.
In addition, one `step2` key is rejected at parse time: `enable_tf32`
(delete it; it only ever scoped the dense port's sgemms, and
`step0.enable_tf32` is unaffected). A config that already said
`backend_step2: "gpu"` still parses, but trains its prior on the
candidate-set session — which matches the `cpu` reference, where the dense
port did not.

## Repository layout

```
arealmshbm/                 — the Python package
  pipeline/                 — unified project-driven driver (preferred entry)
  step{0,2,3}_pipeline/     — per-step super-calls (step 1: pipeline/step1_runners.py)
  data/                     — precomputed-asset mount point (installer-staged)
  precompute/               — offline cache builders (dev-only)
  <leaf>/                   — one folder per algorithmic primitive
    _kernels.py             — numba CPU kernel
    _kernels_gpu.py         — CuPy / RawKernel GPU kernel (where they diverge)
    tests/                  — pytest leaf-level checks
projects/                   — sample project templates
docs/                       — per-step flow docs, disk formats, precision framework
lib/hyperparameters/        — schema-v2 config definitions
original_paper.md           — algorithm reference (Kong et al. 2021)
```

## Requirements

- Python 3.13; numba 0.63.1, numpy 2.2.6, scipy 1.16.0. numba must run
  on a thread-safe threading layer (OpenMP or TBB, which it selects when
  either runtime is present): the pipeline launches parallel kernels
  from several threads, and numba's `workqueue` fallback layer aborts
  the process on that
- `python-isal` (pip name `isal`; isal_zlib inflate in the GIFTI reader
  and the streaming MAT reader) and
  `blosc2` (`.b2nd` profile files); `h5py` for inputs saved as MAT v7.3
  (a group prior, a spatial mask or a gradient file)
- For the GPU backends: CuPy 13.6.0 (all of them);
  `nvidia-nvcomp-cu12` (`backend_step1='gpu'` — its BOLD ingest is
  nvCOMP-only; a missing or unloadable install is an `ImportError`
  before step 0); `psutil` (`backend_step2='gpu'`); a CUDA toolkit at
  `CUDA_PATH` and a device that supports cooperative launch
  (`backend_step3='gpu'` with gMSHBM / cMSHBM — `cudadevrt` +
  `cooperative_groups.h` for the connected-components kernel; dMSHBM
  does not use it). The driver checks CuPy (import, CUDA device,
  NVRTC), nvCOMP, psutil and the CUDA toolkit / cooperative launch
  before step 0. The
  step-2 and step-3 `gpu` backends require `seed_mesh` `fsaverage3`.
- Staged precomputed assets (installer-deployed; see
  [`arealmshbm/data/README.md`](arealmshbm/data/README.md)) and
  `MSHBM_ATLAS_DIR` pointing at a directory whose
  `<targ_mesh>/label/` holds the Schaefer2018 + aparc `.annot` files
  (the atlas dir's only runtime role)

BOLD input is GIFTI `.func.gii` only.

Tests:

```bash
python -m pytest arealmshbm/
```

## License

MIT — see [`LICENSE`](LICENSE). The license covers the source code in
this repository. Precomputed data staged by the installer (e.g. the
HCP-derived group priors) is not part of this repository and remains
subject to its own data-use terms.

## Citing

If you use cuArealMSHBM, please cite this repository (see
[`CITATION.cff`](CITATION.cff)) **and** the original method:

> Kong R, Yang Q, Gordon E, et al. *Individual-Specific Areal-Level
> Parcellations Improve Functional Connectivity Prediction of
> Behavior.* Cerebral Cortex, 2021;31(10):4477–4500.
> doi:10.1093/cercor/bhab101

All versions of this software are archived on Zenodo — cite via
[doi:10.5281/zenodo.21922832](https://doi.org/10.5281/zenodo.21922832)
(resolves to the latest release).

## Acknowledgments

To the **Computational Brain Imaging Group (CBIG)** at the National
University of Singapore, whose Areal-MSHBM created this line of work
and whose open, carefully engineered reference implementation made a
faithful port possible at all: for this project's contribution to the
field, we extend our sincere greetings and deepest respect.

---

<a id="chinese"></a>
# cuArealMSHBM（中文）

[English](#english) | **简体中文**

**CPU/GPU 加速的 Areal-MSHBM** —— 基于静息态 fMRI 的个体化皮层分区，以 numba（CPU）与 CuPy（GPU）双后端完成端到端的 Python 移植。已在大规模数据（300+ 名被试、1,800+ 个 fMRI session）上与原版 MATLAB 实现完成对照校验。**v2.2.0** 让 step 2 采用参考实现的初始化（带 `theta_num` 的软掩膜后验，两个后端相同）：在同一份输入上与参考实现自身的 step 2 对照，训练出的 gMSHBM 组先验一致率从 78.9 % 升至 94.1 %（dMSHBM：71.9 % → 81.2 %），速度与内存不变。自 v2.1.0 起，候选集 session 是 step 2 / step 3 唯一的 GPU 后端（step 3 覆盖全部三个变体），GPU 上的 BOLD 读入只走 nvCOMP，并且当位打包队列放得进显存时不在主机内存中保留它的副本：单被试从原始 BOLD 到分区结果，在一张笔记本 GPU 上端到端只需 **6.2 秒**；40 名被试的队列从原始 BOLD 经组先验训练到全部 40 份分区结果 **74.0–75.6 秒**完成（其中 step 2 组先验训练单独运行 16.5 秒）。

---

## 为什么是 Areal-MSHBM

**Areal 多 session 层级贝叶斯模型**
（[Kong et al., 2021, *Cerebral Cortex*](https://doi.org/10.1093/cercor/bhab101)；[CBIG 参考实现](https://github.com/ThomasYeoLab/CBIG)）是一个非常出色的个体化分区方案：它以严谨的生成式模型在 session 与被试两个层级间汇聚信息，得到的个体化 areal 级分区在同质性与行为预测上均优于群体图谱。

然而在实践中，参考流水线速度慢、内存占用大，且专为 Linux 大规模计算集群设计（MATLAB + 编译 MEX、集群作业提交、逐步骤中间结果落盘）。在工作站上完成单个被试的分区需要数十分钟和约 11 GB 内存；队列级的先验训练更是离开集群便无从谈起。这样的部署门槛切实阻碍了该方法的普及——而这与它的科学价值并不相称。

## 我们的愿景

我们的目标是将先进的 GPU 加速工程引入这类有潜力的神经影像方法，使它们能够以前所未有的规模得到详细的评估与使用。曾经需要计算集群的方法，应当能在一块笔记本 GPU 上运行；曾经只能抽样少数被试的评估，应当能横扫整个队列。

## 本项目做了什么

cuArealMSHBM 用 Python 重新实现了全部四个流水线步骤（RSFC 梯度嵌入 →连接 profile 与 vMF 初始化 → 组先验 EM 训练 → 逐被试分区 EM），提供可按步骤自由切换的后端：

- **CPU 后端** —— numba-JIT 内核，无需 GPU。
- **GPU 后端** —— CuPy + 手写 RawKernel。
- **候选集 GPU 后端**（`backend_step2` / `backend_step3` = `gpu`）—— EM 只在组先验支撑集所隐含的候选集上运行，而不是稠密的 (N × L) 后验，因此整个队列在放得进显存时可以在整轮运行期间常驻显存（否则 step 2 逐被试流式载入）。

工程亮点：

- **更快、更精确的贝塞尔类函数数值计算。** vMF 归一化常数与浓度参数更新均改用 `log I_ν` 的闭式解与渐近展开，而非参考实现的通用贝塞尔路径。经 mpmath 50 位精度审计：d=3 的 vMF 对数归一化常数精确到 0.31 × fp32 eps —— 在 κ = 1000 处比 MATLAB 原版**精确至多 19 倍**，且无分支、对 JIT/GPU 友好。
- **位打包的显存压缩。** 二值化 BOLD profile 在设备上以位打包的 `uint8 (N, T, ⌈D/8⌉)` 张量存放 —— 8 倍压缩，使 40 被试 × 6 session 队列的 BOLD 仅占 2.9 GB 显存（未打包约 23 GB），并由基于 popcount 的内核直接消费打包形式。
- **瓶颈步骤上的算子融合与算子调优**：融合的 profile 生成内核、带逐被试 fp64 暂存的融合 E-step softmax 链（fp32 存储，精度关键处fp64）、调优的归约，以及将 I/O 隐藏在 GPU 计算之后的流水线化 BOLD 解码。

结果：在最重的步骤上，相对 MATLAB 参考实现的加速在单张 Blackwell GPU 上达到**三个数量级**（step-3 分区 EM：763.1 s → 0.55 s，约 1,400 倍；step-0 梯度嵌入：233.6 s → 2.3 s，约 100 倍），单被试全流程端到端提速**约 170 倍**。

## 项目状态

**这是第四个发布版本（v2.2.0）**，包含 CPU 与 GPU 后端；step 2 与 step 3 的 GPU 后端即候选集 session，其中 step 3 覆盖全部三个变体。在迄今超过 **300 名被试、1,800 个 fMRI session** 的大规模校验中，它与原版实现的输出在数值所允许的极限水平上保持一致（逐顶点一致率约 98%；跨 BLAS/归约顺序边界的位级一致在数学上不可达），同时带来显著的速度与内存节约（见下方表格）。

项目仍在持续完善中。**所有预计算数据**（网格资产包、step-0 缓存、预训练 HCP 组先验、空间掩膜）**将由带 GUI 的发行版（开发中）的安装程序部署。** 本源码版本刻意不包含这些资产 —— 安装程序部署的目录布局见 [`arealmshbm/data/README.md`](arealmshbm/data/README.md)。

## 基准测试

所有数字均在同一台 Windows 11 笔记本上测得：RTX 5090 Laptop GPU（Blackwell，24 GB）、Python 3.13、numba 0.63.1、CuPy 13.6.0。MATLAB基线为 CBIG 参考流水线**并启用其编译 MEX 热路径**（`mtimesx`），在同一台机器上运行 —— 比典型集群节点更强的基线。下表中 cuArealMSHBM 的每一个**耗时**数字都在 v2.2.0 代码上重新测得（2026-11-02），每一项"总计"均为无缓存项目上的端到端驱动器墙钟时间（含解释器启动与输入校验），numba 与 CuPy 的内核缓存在磁盘上已预热；标签一致率各行沿用 v1.0.0 的测量结果。内存各行在 v2.2.0 代码上实测：主机为驱动器进程的常驻内存峰值（表中另注进程者除外），设备为 `nvidia-smi` 相对空闲桌面的用量峰值。数值上，v2.2.0 到 step 1 为止与 v2.1.0（后者与 v2.0.0 逐位相同）完全一致：在下文的 40 被试队列上，用 v2.2.0 代码从同一份 step-0 输出起运行 step 1–3，得到的位打包 profile、队列均值、空间掩膜与 `group.mat` 逐位相同；step 2 的组先验按设计有所不同（新的初始化，见下文"与参考实现的一致性"），其下游 40 份标签文件中有 9.4 % 的皮层顶点标签改变（逐被试 7.0–12.6 %）。

### Mode A —— 基于预训练先验的个体化分区

单被试 × 6 session，fsaverage6（74,947 个皮层顶点），K = 300 分区，
gMSHBM，β = 5，w = 50，c = 10：

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 0 —— 梯度嵌入 | 233.6 s | 25.9 s | 2.3 s |
| step 1 —— profile + 初始化 | 47.2 s | 10.2 s | 1.0 s |
| step 3 —— 分区 EM | 763.1 s | 23.0 s | **0.55 s** |
| **总计（端到端墙钟）** | **1073.0 s** | **60.5 s（18×）** | **6.2 s（173×）** |
| 对 MATLAB 标签的逐顶点一致率¹ | — | 98.33 % | 98.05 % |
| 主机内存峰值（整次运行，驱动器进程） | ~11 GB RAM（step 3） | 6.5 GiB | **2.7 GiB（省 4×）** |

GPU 一列所用后端：`backend_step0` / `backend_step1` / `backend_step3` = `gpu`。MATLAB 路径还额外需要一次 1,527 s 的 GIFTI→NIfTI 一次性格式转换；cuArealMSHBM 原生读取 `.func.gii`。

与 v1.0.0 相比（同一台机器、同一配置），GPU 上 step 0–3 的计算耗时从 **15.8 s 降至 3.9 s（4.1×）**，来自 step-0 的 Δ-stepping SSSP 重写、step-1 融合的整被试 GPU 叶子算子，以及 step-3 的候选集后端（均为 v2.0.0 引入）；v2.1.0 在保持这一耗时的同时去掉了 step 1 与 step 2（显存常驻缓存模式）在主机内存中保留的位打包 profile 副本，v2.2.0 未触及 Mode A 的代码路径。6.2 s 的墙钟时间还额外包含约 1.0 s 的一次性包导入（Mode A 的最小输入检查为解析先验路径而载入 step-3 包）与约 0.6 s 的解释器启动 —— 这些是不随队列规模增长的固定开销。新进程中的重复运行稳定在 6.1–6.3 s；若 numba 缓存被删除，首次运行要多付约 7 s 的 JIT 编译；若 CuPy 内核缓存也被删除，则要先编译全部 GPU 内核（墙钟 31 s，每台机器一次）。

¹ 皮层顶点，剔除内侧壁。两个 Python 后端彼此的一致率为 98.21%。MATLAB↔Python 的位级一致本就不可达：残余分歧是多个良性来源叠加的结果——不同数值库之间的 BLAS 归约顺序差异（每个后端各有自己的数值噪声带，GPU 还存在小幅的逐次运行波动）、贝塞尔类函数所采用的不同数值路径（我们通过 `log I_ν` 的闭式解与渐近展开求值，经 mpmath 审计：CPU 上是 `arealmshbm/em_stop_criterion/_cdln.py`，设备上是 d = 3 的闭式解，两者的 fp32 输出由 `arealmshbm/em_stop_criterion/tests/test_cdln_d3_oracle.py` 钉为逐位相同），以及移植中少数几处与参考实现有意为之的细小实现差异。我们用于判断此类差异是否具有功能意义的四层评估框架见 `docs/precision_impact_analysis.md`。

### Mode B —— 在本地队列上训练组先验

40 被试 × 6 session，fsaverage6，K = 300，gMSHBM：

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 2 —— 组先验 EM（生产配置 `max_iter_inter=10`；两个后端均为 48 次 EM 迭代） | 工作站上不可实施² | 2,573.5 s | **16.5 s（156×）** |
| 全流程 step 0–3，整个队列（生产运行） | 工作站上不可实施² | — | **74.0–75.6 s（≈1.2 分钟）** |
| step 2 在 40 被试全队列下的设备内存 | **≈ 370–490 GB RAM**（预估²） | — | **峰值 5.6 GiB / 24 GB 显存** |
| step 2 单独运行（独立的 `arealmshbm.step2_pipeline.profile` 进程）在 40 被试全队列下的主机内存 | （同上，≈ 370–490 GB²） | 11.1 GiB | **峰值 1.1 GiB** |

step 2 的 GPU 一列取自 `gpu` 后端，即候选集 session（自 v2.0.0 起；v1.0.0 中占用该名字的稠密 CuPy 移植已不存在）。该数字为新进程中的单次运行（进程内的预热重复运行为 15.2 s）。CPU 一列同样是新进程中的单次运行、48 次 EM 迭代（v2.1.0 代码在同一形状上测得 2,945.6 s，v2.0.0 代码 2,638.7 s；三者的 CPU EM 内核代码相同，差异属于笔记本上一次 43–49 分钟全核运行的逐次波动）。初始化的改动不增加 step 2 的耗时：同一会话中用 v2.1.0 代码重跑独立的 GPU step 2，热运行为 15.0–15.1 s，v2.2.0 为 15.0–15.4 s，两者均为 48 次 EM 迭代；v2.1.0 发布时的 15.8 s / 14.1 s 是这台笔记本另一天的状态。全流程一行是三次生产运行（74.0、75.0 与 75.6 s）；最快的一次中 step 2 为 14.6 s，另有 step 0 40.4 s、step 1 8.4 s、step 3 8.1 s；最快与最慢两次相差 1.5 s，其中 step 0 占 0.8 s，step 1 与 step 3 各占 0.3 s，step 2 没有差别。主机内存一行自 v2.1.0 起列出，在独立的 step-2 进程中测得：在显存常驻缓存模式下（`eager_bitpacked`；`bold_cache_mode` 为 `auto` 时，只要位打包队列、session 的常驻状态与一段安全余量放得进空闲显存就选用它），session 不再在主机内存中保留位打包队列的副本 —— 每名被试直接解码进一个两槽页锁定环形缓冲并上传 —— 主机工作集因此不再随队列规模增长；在端到端驱动器运行中，step 2 期间的进程峰值为 2.7 GiB，其中大部分是 step 0–1 遗留的内存。放不进显存的队列走 `stream` 模式：设备上只保留一名被试，并在主机内存允许时在主机内存中缓存整个位打包队列。在 `backend_step1: "gpu"` 下，step 1 同样不在主机内存中保留位打包队列的副本，且与队列规模无关：每个 session 一打包就累加进设备端的求和。`cpu` 后端求平均时会把每名被试的 `.b2nd` 读回主机内存。

**与 v1.0.0 的同机对比。** v1.0.0 公布的 step 2 数字是 794.0 s（CPU）/ 156.7 s（稠密 GPU），但那是在 `max_iter_inter=2` —— 短跑基准的迭代上限 —— 下测得的，并非生产默认值 10，见 [`docs/step2_flow_and_subgraphs.md`](docs/step2_flow_and_subgraphs.md) 的 § Production-cohort walls。在 v2.0.0 代码上重跑同样的形状，`gpu` 后端为 **11.1 s**：比 v1.0.0 的 GPU 数字快 **14×**，比其 CPU 数字快 **72×**。其实测显存峰值在 `max_iter_inter=2` 与 `=10` 下**同为 6.7 GiB** —— 不随外层迭代数增长，因为候选集 session 只分配一次并全程复用。v1.0.0 的稠密 GPU 后端在同一形状下实测为 **21 GiB**，在 `inter=10` 时已溢出到主机内存；v2.0.0 将其移除，除了显存原因，也因为它的 fp32 清零（flush-to-zero）E-step 是两者中精度较低的一个。

² 由参考实现源码（`CBIG_ArealMSHBM_gMSHBM_estimate_group_priors_{parent,child}.m`）推算，而非实测 —— 因为它在工作站上*无法*实测。参考实现的 step 2被架构为 **1 个 parent + S 个 child 集群作业**（其源码头注释原文：*"user should submit 1 parent job and num_sub child jobs"*），在**每个 M-step 内层迭代**都通过文件系统交换 `.mat` 文件。child 会阻塞在迭代中途等待 parent 的回复，因此被试无法串行处理：40 个 child 必须**同时**常驻。在本基准形状下（N = 81,924 顶点、D = 1,175 profile 维、T = 6 session、L = 300 分区、全 fp64），每个 child 持有 profile 主存储`data_series`（N × D × T，4.6 GB），外加代码以*不同内存布局*物化的多份全尺寸拷贝 —— 每 session 的 N × D 切片副本、`mtimesx` 产出的N × L × T κ 更新乘积、转置的 L × N × T log-vMF 堆栈、(N, L, T) 空间先验堆栈 —— 工作集达**每被试约 9–12 GB**；parent 另持有 (N, L, S) 后验（S = 40 时 7.9 GB）及同尺寸临时量。合计：41 个进程约 370–490 GB 的同时常驻内存，外加每迭代经共享存储的 `.mat` 流量 —— 计算集群是硬性前提，而非可选优化。

cuArealMSHBM 将这一切收拢到单个设备上：每份二值化 profile 在显存中**只存在一次**，以位打包的 `uint8 (N, T, ⌈D/8⌉)` 张量存放（每被试 72 MB，整个队列 2.9 GB —— 对比未打包的约 23 GB，以及参考实现*尚未计入额外拷贝*的每被试约 4.6 GB fp64），并在所有内核间按引用共享。后验以 fp32 而非 fp64 存储，且只存放在空间掩膜的支撑集上，即一个 (S, P) 数组（P = 586,522 个（顶点, 分区）单元，占 N × L 的 2.4 %；40 被试共约 94 MB，而稠密的 (S, N, L) 布局需要 3.93 GB；仅在 softmax 需要处使用 fp64 暂存），算子融合则使参考实现在内存与磁盘间往返的 N × L × T中间量根本不被物化。`gpu` 后端上实测的设备峰值占用为 **5.6 GiB** —— **整个 40 被试队列所需的显存比参考实现单个被试所需的内存还少**，且 24 GB 显卡还剩下大半。队列还能扩到多大才会被 (S, N, ·) 项主导，尚未实测。

## 与参考实现的一致性

### Mode A —— 同一先验下的个体化分区

以**未经修改的 CBIG 原版**为基准，在三个队列共 30 名被试上实测 —— K = 300、β = 5、w = 50、c = 10，目标网格 fsaverage6 / 种子网格 fsaverage3，两侧载入同一份组先验文件。后端：`backend_step0` / `backend_step1` / `backend_step3` = `gpu` —— step 3 在全部三个变体上都运行候选集 session。cMSHBM 一列已在该后端上为 v2.1.0 重新测得；gMSHBM 与 dMSHBM 两列沿用 v2.0.0 的测量结果，其 step-3 后端就是这同一个 session（当时名字不同）。Mode A 不运行 step 2，因此 v2.2.0 不改变这张表。

逐顶点标签一致率，仅计皮层顶点、剔除内侧壁 —— 内侧壁两侧同为标签 0，计入会白送一致率。括号内为全顶点口径。

| 队列 | 被试 × session | gMSHBM | cMSHBM | dMSHBM |
|---|---|---|---|---|
| 自有队列 | 10 × 6 | 95.12 %（95.52 %） | 99.35 %（99.40 %） | **99.99 %**（99.98 %） |
| OpenNeuro `ds004466` | 10 × 4 | 93.20 %（93.63 %） | 98.42 %（98.49 %）ᵃ | **99.84 %**（99.69 %） |
| OpenNeuro `ds000221` | 10 × 4 | 93.07 %（93.12 %） | 98.13 %（98.24 %）ᵇ | **99.67 %**（99.12 %） |

ᵃ n = 9 · ᵇ n = 8 —— 参考实现自身在三名被试的 cMSHBM 上失败（`CBIG_ArealMSHBM_component_distance` 内 `sub2ind` 下标越界，重试可复现），这几格没有参考标签可比。cuArealMSHBM 这三名被试均正常完成。其余各格均为完整 10 名被试。

**为什么 gMSHBM 低于 cMSHBM 与 dMSHBM。** cMSHBM 与 dMSHBM 完全不读取 step 0 的梯度输出，gMSHBM 读取。step 0 正是本移植与参考实现有意采取不同实现的地方 —— 最明显的两处是 watershed 驱动对采样列的分块方式，以及在扩散嵌入之前施加于梯度距离矩阵的变换。一项将两处都还原为参考行为的对照实验可以收回 gMSHBM 差距中的 0.7–1.4 个百分点，因此它们解释的是其中一部分，而非全部。若后续发现与原版更高度的一致性对下游使用确有必要，我们会重新考虑这些选择。

上文 Mode A 表中的 98.33 % / 98.05 % 回答的是另一个问题：它是针对一份 step 0 已带有与本移植相同的那两处改动的 MATLAB 参考测得的。同一个移植，不同的参考基准 —— 两组数字并不矛盾。

### Mode B —— step 2（组先验训练）与参考实现的对照

Mode B 在工作站上没有端到端的 MATLAB 基线（见上文 ²），因此单独校验 step 2：在自有队列的 8 名被试 × 2 session 上（K = 300，10 次外层迭代；gMSHBM β = 5 与 dMSHBM）原样运行参考实现的 step 2，再让 cuArealMSHBM 的 step 2（`gpu` 后端）在完全相同的输入 —— 参考实现自己的 step-0/1 输出 —— 上运行。一致率为训练所得 `theta` 在 74,947 个皮层顶点上的逐顶点 argmax 一致率，分别按直接计数 / 允许并列计数给出（10–13 % 的行存在并列的 `theta` 值，均为 1/8 的整数倍）。

| 训练所得先验 vs 参考实现 | gMSHBM β = 5 | dMSHBM |
|---|---|---|
| v2.1.0（one-hot argmax 初始化） | 78.89 % / 89.51 % | 71.86 % / 84.91 % |
| **v2.2.0（参考实现的初始化）** | **94.07 % / 98.16 %** | **81.19 % / 92.38 %** |
| v2.2.0 + 单精度 E-step 归一化（实验，未发布） | 99.90 % / 99.97 % | 98.82 % / 99.69 % |

v2.2.0 的初始化就是参考实现的 —— child 脚本中的 `ini_s_lambda`，加上 parent 中的掩膜 / 归一化 / `theta_num` 一段 —— 并按参考实现的做法以单精度求值；`cpu` 与 `gpu` 两个后端得到的初始化逐位相同。剩下的差距来自一个精度取舍：参考实现对 E-step 的 fp32 `exp` 值做归一化，于是掩膜内各格全部下溢的顶点会被置零（0/0），而 cuArealMSHBM 以 fp64 归一化并保留这些行。没有空间项时这一点影响大得多 —— 在 dMSHBM 中，参考实现在该初始化之后的第一次 E-step 就清零了 23 % 的（顶点, 被试）行 —— 这就是 dMSHBM 一格落后于 gMSHBM 的原因。复现这一下溢（最后一行）是有意不发布的：它是参考实现的精度副产物，不属于模型本身。在 40 被试的 HCP 训练集上（K = 400，4 个 run），v2.2.0 训练出的 gMSHBM 先验与 CBIG 公开发布的 40 被试先验在 92.2 % 的顶点上一致（v2.1.0：89.8 %）；剩余的差距在 step 2 的上游。

## 关于原论文的声明

Areal-MSHBM 的大规模稳定性分析与独立第三方评估将于近期公布。我们的先期结果表明：原论文关于该方法优势与改进的声明是**诚实、可靠、可复现的 —— 并且能够在原发表所用数据集之外的其它数据集上复现。**

## 快速开始

一个*项目*就是 `projects/` 下的一个目录，含一份 BOLD 清单和一份配置：

```
projects/<name>/
  bold_inputs.json        # 每 (被试, session, 半球) 的 BOLD .func.gii 路径
  pipeline_config.json    # 模式 + 变体 + 超参数（schema v2）
```

模板：[`projects/sample_modeA_single/`](projects/sample_modeA_single/)、
[`projects/sample_modeA_batch/`](projects/sample_modeA_batch/)、
[`projects/sample_modeB/`](projects/sample_modeB/)。

### 首次运行前

有两样东西不在仓库里，需要先就位一次。

**1、预计算资产** —— 以 release asset 形式发布在 [`assets-v1`](https://github.com/BoletuPeng/cuArealMSHBM/releases/tag/assets-v1) tag 下（与代码版本独立编号）。解压到 `arealmshbm/data/precomputed/`：

```bash
curl -L -O https://github.com/BoletuPeng/cuArealMSHBM/releases/download/assets-v1/avg_mesh-fsaverage6.tar.gz
curl -L -O https://github.com/BoletuPeng/cuArealMSHBM/releases/download/assets-v1/step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz
tar -xzf avg_mesh-fsaverage6.tar.gz -C arealmshbm/data/precomputed/
tar -xzf step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz -C arealmshbm/data/precomputed/
```

两个都是**必需的**。`avg_mesh`（4.3 MB）是所有网格几何的唯一来源，运行时不会重建。`step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz`（82.6 MB）是 step 0 的输入缓存：仓库里确实带了它的构建器（`arealmshbm.precompute.step0_inputs_builder`），但它需要从 CBIG checkout 读原始 FreeSurfer 球面，而其期待的目录布局与当前 CBIG 版本不一致，因此请以下载为准。两个都请解压到 `arealmshbm/data/precomputed/`。`$MSHBM_PRECOMPUTED_ROOT` 只能重定向 `avg_mesh/` bundle —— step 0 的缓存始终从包内读取 —— 所以不要用它把两者一起搬走：那样会在 step 0 里以一条看上去不相干的、抱怨 CBIG checkout 缺失的错误失败。

**2、Atlas 目录** —— 把 `MSHBM_ATLAS_DIR` 指向一个目录，其 `<targ_mesh>/label/` 下存放 aparc 与 Schaefer2018 Kong2022 的 `.annot` 文件（从 CBIG checkout 里准备）。step 1 会实时读取它们，没有降级路径。

组先验与空间掩码**不**在此发布 —— 它们与 CBIG 的文件逐字节相同，需要时从 CBIG checkout 取用。完整布局与出处见
[`arealmshbm/data/README.md`](arealmshbm/data/README.md)。

运行完整流水线：

```bash
python -m arealmshbm.pipeline projects/<name>
```

或以编程方式：

```python
from arealmshbm.pipeline import Pipeline
result = Pipeline("projects/<name>").run()
```

分区结果写入
`projects/<name>/ind_parcellation_<variant>/<N>_sess/beta<B>/`，文件名为 `Ind_parcellation_MSHBM_sub<S>_w<W>_MRF<C>_beta<B>.mat`（dMSHBM 没有 `beta<B>/` 目录，文件名也没有 `_beta<B>` 后缀）。

### 运行模式

| 模式 | 步骤路径 | 组先验 |
|---|---|---|
| `modeA_single`（1 被试） | 0 → 1 → 3 | 预置于项目内 |
| `modeA_batch`（N 被试） | 0 → 1 → 3 | 预置于项目内 |
| `modeB_train_prior`（队列） | 0 → 1 → 2 → 3 | 由 step 2 在你的队列上训练 |

Mode A 读取 `<project>/priors/<variant>/beta<B>/Params_Final.mat`（dMSHBM 为 `priors/dMSHBM/Params_Final.mat`，没有 `beta<B>/` 一级）；将先验放到该位置是项目创建者的职责。变体：gMSHBM 与 dMSHBM 已提供；cMSHBM 的 step 2 未接入（其 step 3 已接入）。参见[`docs/pipeline_modes.md`](docs/pipeline_modes.md) 与[`docs/pipeline_variants.md`](docs/pipeline_variants.md)。

### 如何选择后端

每个步骤在 `pipeline_config.json` 中独立选择后端（`backend_step0` … `backend_step3`）。`cpu` 始终可用；GPU 后端的前置条件如下表：

| 配置项 | 可选值 | 说明 |
|---|---|---|
| `backend_step0` | `cpu`、`gpu` | GPU 版 step 0 不保证逐次运行的位级可复现，实测波动范围见 [`docs/step0_flow_and_subgraphs.md`](docs/step0_flow_and_subgraphs.md) |
| `backend_step1` | `cpu`、`gpu` | 位打包 profile、平均 profile、空间掩膜与 `group.mat` 的标签在两个后端上逐位一致；`group.mat` 的 `mtc` / `epsil` 一致到几个 fp64 ULP 以内（fp64 归约顺序）；`gpu` 通过 nvCOMP 读入 BOLD（需要 `nvidia-nvcomp-cu12`），并要求 `step1.reduction_dtype: "float64"`（其他取值在校验阶段即被拒绝） |
| `backend_step2` | `cpu`、`gpu` | `gpu` 即候选集 session：要求 `seed_mesh: "fsaverage3"`、K ≤ 512、gMSHBM 还要求梯度分量数 ≤ 11,772；其他形状在校验阶段即被拒绝，请为其选择 `cpu`；需要 `psutil` |
| `backend_step3` | `cpu`、`gpu` | `gpu` 即候选集 session，覆盖 gMSHBM、cMSHBM 与 dMSHBM：要求 `seed_mesh: "fsaverage3"`（否则选 `cpu`）；gMSHBM / cMSHBM 还需要 `CUDA_PATH` 指向的 CUDA 工具包以及支持 cooperative launch 的设备 |

设计说明与后端之间实测的一致性数据见 [`docs/step2_sparse_design.md`](docs/step2_sparse_design.md) 与 [`docs/step3_sparse_design.md`](docs/step3_sparse_design.md)。

**从 v2.1.0 升级项目。** 配置键没有变化。step 2 的初始化变了，因此同一个 Mode B 项目在 v2.2.0 上重跑会训练出与 v2.1.0 不同的组先验，其下游的个体化分区也随之改变（在上文的 40 被试队列上为 9.4 % 的皮层顶点）。Mode A 在给定先验文件下的结果不变：step 1 与 step 3 的代码相同。

**从 v2.0.0 升级项目配置。** `backend_step3` 只接受 `cpu` 与 `gpu`：原来写 `"gpu_sparse"` 的请改写为 `"gpu"`（同一个 session）；原来写 `"gpu_full"` 或 `"gpu_elambda"` 的请改为 `"gpu"` 或 `"cpu"`（稠密 GPU EM 已不存在；对照参考实现的标签，候选集 session 与它持平或更优）。`w` 在所有后端上都必须 ≥ 1：`w = 0` 时 E-step 的 `w · log θ` 项在先验支撑集之外没有定义，参考实现同样如此。v2.0.0 中"找得到才用"的四项依赖是硬性要求：`python-isal`（所有后端）、`nvidia-nvcomp-cu12`（`backend_step1: "gpu"`）、`psutil`（`backend_step2: "gpu"`），以及 `CUDA_PATH` 指向的 CUDA 工具包加上支持 cooperative launch 的设备（`backend_step3: "gpu"` 搭配 gMSHBM / cMSHBM）—— 机器不满足某一步的要求时，请为该步选择 `"cpu"`。

**从 v1.0.0 升级项目配置。** 上面 v2.0.0 的说明同样适用。此外，`step2` 块中有一个键会在解析阶段被拒绝：`enable_tf32`（直接删除；它从来只作用于稠密移植的 sgemm，`step0.enable_tf32` 不受影响）。原本就写 `backend_step2: "gpu"` 的配置仍可解析，但其组先验由候选集 session 训练 —— 后者与 `cpu` 参考实现一致，而稠密移植并不一致。

## 仓库结构

```
arealmshbm/                 — Python 包
  pipeline/                 — 统一的项目驱动入口（首选）
  step{0,2,3}_pipeline/     — 逐步骤超调用（step 1 在 pipeline/step1_runners.py）
  data/                     — 预计算资产挂载点（由安装程序部署）
  precompute/               — 离线缓存构建器（仅开发用）
  <leaf>/                   — 每个算法原语一个目录
    _kernels.py             — numba CPU 内核
    _kernels_gpu.py         — CuPy / RawKernel GPU 内核（与 CPU 分叉处）
    tests/                  — pytest 叶级测试
projects/                   — 示例项目模板
docs/                       — 逐步骤流程文档、磁盘格式、精度框架
lib/hyperparameters/        — schema-v2 配置定义
original_paper.md           — 算法出处（Kong et al. 2021）
```

## 环境要求

- Python 3.13；numba 0.63.1、numpy 2.2.6、scipy 1.16.0。numba 必须运行在线程安全的线程层上（OpenMP 或 TBB；两者任一运行时存在时 numba 会自动选用）：流水线会从多个线程发起并行内核，numba 的后备线程层 `workqueue` 遇到这种情况会直接中止进程
- `python-isal`（pip 包名为 `isal`；GIFTI 读取器与流式 MAT 读取器中的 isal_zlib 解压）与 `blosc2`（`.b2nd` profile 文件）；以 MAT v7.3 保存的输入（组先验、空间掩膜或梯度文件）需要 `h5py`
- GPU 后端：CuPy 13.6.0（全部 GPU 后端）；`nvidia-nvcomp-cu12`（`backend_step1='gpu'` —— 其 BOLD 读入只走 nvCOMP，未安装或无法加载会在 step 0 之前以 `ImportError` 报错）；`psutil`（`backend_step2='gpu'`）；`CUDA_PATH` 指向的 CUDA 工具包以及支持 cooperative launch 的设备（`backend_step3='gpu'` 搭配 gMSHBM / cMSHBM —— 连通分量内核需要 `cudadevrt` 与 `cooperative_groups.h`；dMSHBM 不使用它）。驱动器会在 step 0 之前检查 CuPy（导入、CUDA 设备、NVRTC）、nvCOMP、psutil 以及 CUDA 工具包 / cooperative launch。step 2 与 step 3 的 `gpu` 后端要求 `seed_mesh` 为 `fsaverage3`。
- 已部署的预计算资产（由安装程序部署；见[`arealmshbm/data/README.md`](arealmshbm/data/README.md)），以及指向 `<targ_mesh>/label/` 下含 Schaefer2018 + aparc `.annot` 文件目录的 `MSHBM_ATLAS_DIR`（图谱目录在运行时的唯一用途）

BOLD 输入仅支持 GIFTI `.func.gii`。

测试：

```bash
python -m pytest arealmshbm/
```

## 许可证

MIT —— 见 [`LICENSE`](LICENSE)。许可证覆盖本仓库中的源代码。由安装程序部署的预计算数据（如 HCP 派生的组先验）不属于本仓库，仍受其自身数据使用条款约束。

## 引用

若您使用了 cuArealMSHBM，请引用本仓库（见[`CITATION.cff`](CITATION.cff)），**并**引用原方法：

> Kong R, Yang Q, Gordon E, et al. *Individual-Specific Areal-Level
> Parcellations Improve Functional Connectivity Prediction of
> Behavior.* Cerebral Cortex, 2021;31(10):4477–4500.
> doi:10.1093/cercor/bhab101

本软件的所有版本均存档于 Zenodo——可通过 [doi:10.5281/zenodo.21922832](https://doi.org/10.5281/zenodo.21922832) 引用（自动解析到最新版本）。

## 致谢

谨向新加坡国立大学的 **Computational Brain Imaging Group（CBIG）** 致意：Areal-MSHBM 开创了这一研究方向，其开放且精心打磨的参考实现使忠实的移植成为可能。为其对领域的贡献，我们致以诚挚的问候与最深的敬意。
