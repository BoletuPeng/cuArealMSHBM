# Step 1 — call graph + module map

Structural reference for the production Python step-1 super-call.
Maps each call site to the Python module that owns it.

For the design conventions drawn from this layout (validate / profile
pattern, leaf module shape, CPU/GPU sibling files), see
[`strategy.md`](strategy.md).

Step 1 is driven from the unified pipeline driver
([`arealmshbm/pipeline/`](../arealmshbm/pipeline/)). The
four subgraphs live as module functions in
[`pipeline/step1_runners.py`](../arealmshbm/pipeline/step1_runners.py)
and the cohort.json write is performed by
[`pipeline/cohort_writer.py`](../arealmshbm/pipeline/cohort_writer.py)
at the end of the step-1 phase. The per-stage wall-time report at
[`step1_pipeline/profile.py`](../arealmshbm/step1_pipeline/profile.py)
calls those same module functions directly. (The MATLAB-GT
`validate.py` gate was retired when the pipeline decoupled from
MATLAB; regression now rides on the out-of-tree ICC scorer.)

## The whole flow

```
pipeline.Pipeline._run_step1_all_subjects()                    [arealmshbm/pipeline/driver.py]
│   backend ∈ {'cpu', 'gpu'} dispatched at each leaf supercall.
│
├── run_generate_profiles(seed_mesh, targ_mesh, sub, sess, …)  pipeline/step1_runners.py → generate_profiles
│   ├── parallel BOLD reads (.func.gii, bytescan+isal_zlib)    data_io/gifti_io.read_surface_gifti
│   ├── seed_select (MARS_label==2 mask)                       inline (numpy / cupy fancy-index)
│   ├── cbig_corr per hemi
│   │   ├── zscore + L2-unit-norm columns                      _kernels[_gpu] (numba / RawKernel)
│   │   └── (T, K).T @ (T, V) BLAS GEMM                        numpy MKL / cuBLAS
│   ├── nan_to_zero + per-run sum                              _kernels[_gpu] (numba / cp.nan_to_num)
│   ├── threshold_top_fraction (introselect)                   np.partition / cp.partition
│   ├── threshold-binarize (K-major out)                       _kernels[_gpu] (numba / ufunc)
│   └── per-subject .b2nd writer (one file, T sessions stacked) data_io/profile_io.write_subject_profile_tnd
│
├── run_avg_profiles(seed_mesh, targ_mesh, num_sub, num_sess)  pipeline/step1_runners.py → avg_profiles
│   ├── parallel per-subject .b2nd packed reads                data_io/profile_io.read_subject_profile_packed_tnd
│   ├── accum_inplace (per arrival)                            _kernels[_gpu] (numba / cp.add)
│   ├── scale_inplace (1/n)                                    _kernels[_gpu] (numba / cp.multiply)
│   ├── parallel `np.save` writes (.npy per hemi)              numpy
│   └── returns AvgProfilesResult (paths + in-memory lh_avg/rh_avg arrays)
│
├── run_ini_params(seed_mesh, targ_mesh, lh_labels, rh_labels) pipeline/step1_runners.py → ini_params
│   │   driver threads lh_avg/rh_avg in-memory from the prior subgraph,
│   │   skipping the .npy re-read on the hot path.
│   ├── zero_mw + detect_nonzero                               _kernels[_gpu] (numba / cupy primitives)
│   ├── demean + L2-unit-norm rows                             _kernels[_gpu] (numba / RawKernel)
│   ├── groupsum  (mtc, one-hot λ exploit)                     _kernels (numba) / _kernels_gpu (cuBLAS via dense one-hot)
│   ├── (N, D) @ (D, L) BLAS GEMM                              numpy MKL / cuBLAS
│   ├── epsil_input gather-sum                                 _kernels[_gpu] (numba / cupy gather + sum)
│   └── invAd(D-1, rbar)                                       _invad (scipy Bessel, CPU only)
│
├── run_radius_mask(lh_labels, rh_labels, mesh, radius)        pipeline/step1_runners.py → radius_mask
│   ├── _build_mesh_csr (MARS area-rescale + edge dedup)       inline (numpy + scipy.sparse)
│   ├── build_parcel_csr                                       _kernels (numba; CPU on both backends)
│   ├── classify_central_relevance                             _kernels (numba; CPU on both backends)
│   ├── per-parcel-mean geodesic to central sulcus
│   │   ├── cpu: per-source Dijkstra, prange over sources      _kernels.central_sulcus_kernel
│   │   └── gpu: batched pull-based Bellman-Ford on (V, Nsrc)  _kernels_gpu.bellman_ford_bounded_cupy
│   ├── per-parcel bounded radius mask
│   │   ├── cpu: per-parcel Dijkstra, prange over parcels      _kernels.add_spatial_constraint_kernel
│   │   └── gpu: batched pull-based BF bounded by radius       _kernels_gpu.bellman_ford_bounded_cupy
│   └── truncate (anatomical correction)                       _kernels (numba; CPU on both backends)
│
└── write_cohort_manifest                                       pipeline/cohort_writer.py
    └── project_dir/cohort.json                                 roster + artifact ledger (subjects, mesh, run_id,
                                                                profile_b2nd / gradient_{lh,rh} per subject,
                                                                top-level group.mat / spatial_mask /
                                                                avg_profile_{lh,rh}.npy)
```

The cohort.json write runs at the end of the driver's step-1 phase.
Field-level merge against the existing manifest preserves artifacts
produced by prior step1 runs on the same cohort, so users can run
subgraphs incrementally (e.g. generate_profiles in one invocation,
avg_profiles in another) without losing earlier paths.

Subgraph coverage by mode:

* **Mode A** (HCP-prior, production path) consumes step 1's
  ``generate_profiles`` output (per-subject .b2nd) via cohort.json.
  The spatial mask is typically copied in from CBIG's shipped HCP set,
  so ``radius_mask`` is optional; ``avg_profiles`` + ``ini_params``
  are not needed because the group prior comes pre-trained.
* **Mode B** (self-trained prior) consumes all four subgraphs:
  ``generate_profiles`` per (subject, session), then ``avg_profiles``
  + ``ini_params`` to seed step 2's EM, ``radius_mask`` for the
  spatial prior. cohort.json points step 2 at all of them.

## Module / file map

```
arealmshbm/
├── pipeline/                                # unified driver, drives step 1 directly
│   ├── driver.py                            Pipeline._run_step1_all_subjects(): sequences the four runners,
│   │                                        threads avg-profile arrays in-memory subgraph 2 → 3
│   ├── step1_runners.py                     run_generate_profiles / run_avg_profiles / run_ini_params /
│   │                                        run_radius_mask + resolve_group_labels (single source of truth)
│   └── cohort_writer.py                     write_cohort_manifest: roster + artifact ledger
│
├── step1_pipeline/                          # standalone harness for step 1 (no orchestration)
│   └── profile.py                           per-stage wall-time report (--backend cpu|gpu)
│
├── generate_profiles/                       # subgraph 1 — per-session FC profile
│   ├── _kernels.py                          numba CPU kernels
│   ├── _kernels_gpu.py                      CuPy RawKernel: fused zscore (shared-mem reduce)
│   ├── profiles.py                          CPU supercall + dispatch shim
│   └── profiles_gpu.py                      GPU supercall (cuBLAS sgemm + ufuncs)
│
├── avg_profiles/                            # subgraph 2 — sum/divide across sub × sess
│   ├── _kernels.py                          numba CPU: accum_inplace, scale_inplace
│   ├── avg_profiles.py                      CPU supercall + dispatch shim (returns AvgProfilesResult)
│   └── avg_profiles_gpu.py                  GPU supercall (cupy in-place ufuncs)
│
├── ini_params/                              # subgraph 3 — vMF init from group labels
│   ├── _kernels.py                          numba CPU kernels
│   ├── _kernels_gpu.py                      CuPy RawKernel row-demean+L2; one-hot mtc helpers
│   ├── _invad.py                            Bessel-ratio root solver (fp64, CPU only)
│   ├── ini_params.py                        CPU supercall + dispatch shim
│   └── ini_params_gpu.py                    GPU supercall (cuBLAS dgemm via dense one-hot)
│
├── radius_mask/                             # subgraph 4 — geodesic + per-parcel mask
│   ├── _kernels.py                          numba CPU kernels (Dijkstra + classify + truncate)
│   ├── _kernels_gpu.py                      CuPy RawKernel: batched pull-based Bellman-Ford
│   ├── radius_mask.py                       CPU supercall + dispatch shim
│   └── radius_mask_gpu.py                   GPU supercall (BF SSSP; classify+truncate stay CPU)
│
└── data_io/
    ├── profile_io.py                        per-subject .b2nd writer/reader (blosc2 LZ4+bitshuffle)
    │                                        — used by subgraph 1 to write, subgraph 2 to read.
    └── gifti_io.py                          read_surface_gifti / read_surface_gifti_gpu
                                             (subgraph 1 BOLD ingest; bytescan + isal_zlib /
                                             nvCOMP Deflate; bit-equal across CPU/GPU).
```

## What we replicate, end to end

| # | Subgraph | Inputs → Outputs |
|---|---|---|
| 1 | `generate_profiles/` | per-session BOLD `.func.gii` (lh, rh) → one per-subject `.b2nd` (T sessions stacked) |
| 2 | `avg_profiles/` | per-subject `.b2nd` files → `{lh,rh}_…_avg_profile.npy` |
| 3 | `ini_params/` | avg profile + group labels → `group.mat` (mtc, epsil, lambda, …) |
| 4 | `radius_mask/` | group labels + mesh + radius → sparse `lh_boundary`, `rh_boundary` |
| + | `cohort_writer.write_cohort_manifest` (always, end of step-1 phase) | cohort + step1 outputs → `cohort.json` (roster + artifact ledger; merged with prior runs) |

fsaverage* meshes only — `fs_LR_32k` is intentionally not ported
(Mode A is fsaverage6, and the rest of the pipeline is gated to
fsaverage6 too).

## Backend dispatch

Each runner takes a ``backend ∈ {'cpu', 'gpu'}`` keyword and passes it
straight through to the leaf supercall. Dispatch happens at the leaf
entry — a single `if backend == 'gpu':` branch lazily imports the
`<leaf>_gpu.py` sibling and tail-calls it. cupy is never imported on
the CPU path.

GPU paths inline locally (no per-stage H↔D round-trips within a
supercall):

* `compute_profile_arrays_gpu`: BOLD H2D once per run → zscore (RawKernel)
  → cuBLAS sgemm → nan_to_zero → running fp32 accumulator → percentile
  threshold (`cp.partition`) → binarize → D2H once. The pipeline then
  stacks the per-session arrays and writes one per-subject .b2nd.
* `avg_profiles_gpu`: device-resident (V, T) accumulator; per-pair
  `cp.add` in-place; final scale; single D2H.
* `ini_params_gpu`: profile H2D once → row-demean+L2 (RawKernel) →
  dense one-hot scatter → cuBLAS dgemm for `mtc` and `inner = profile @ mtc`
  → gather-sum → scalar D2H for `invAd` (scipy Bessel stays CPU).
* `radius_mask_gpu`: mesh CSR + labels + aparc H2D once per hemi →
  batched pull-based Bellman-Ford on a (V, K) distance matrix for
  both the radius mask (K = L_h, bounded by radius) and the central-
  sulcus mean distance (K = Nsrc, unbounded). One BF iter = one
  RawKernel pass over V·K cells with neighbour pull-relax; loop until
  the `changed` flag stays 0. Classify / truncate (heavy control flow,
  ~1 ms) stay CPU.

GPU memory-pool release between subjects in batch loops is the
caller's responsibility. The unified driver releases the pool at the
end of each per-subject step; `step1_pipeline/profile.py` releases
between subgraph sections via the `_release_cupy_pool()` helper.

## CPU ↔ GPU consistency

The GPU paths match CPU at sub-ULP across the four numeric subgraphs.
Subgraph 3's ε computes via a different fp64 reduction order between
numba and cupy, producing a ~4.7e-16 rel-diff that round-trips
through ``invAd`` without visible effect. Beyond that, CPU and GPU
outputs are bit-exact on the binary-mask and avg-profile paths.

## Wall

Mode-B end-to-end on the YS cohort (**40 subjects × 6 sessions ×
fsaverage6 × L=300, gMSHBM**). RTX 5090 Laptop, 24 GB. 2026-05-20.
Source: internal 40-subject × 6-session cohort run; raw timing log
archived internally.

End-to-end wall on the production GPU path: step 0 — 227.41 s ·
step 1 — **223.85 s** · step 2 — 373.94 s · step 3 — 129.35 s ·
**total 954.84 s (15.91 min)**. Step 1 is 23 % of the end-to-end
total. The standalone profile harness measures 219.83 s on the same
cohort (the unified driver's extra ~4 s is `cohort.json` write +
data_list staging + intermediate cleanup).

### Per-subgraph breakdown (S=40, T=6, fsaverage6, Schaefer=300)

| Subgraph | CPU (s) | GPU (s) | GPU speedup |
|---|---:|---:|---:|
| generate_profiles      | 323.12 | 212.72 |  1.52×  |
| avg_profiles           |   4.79 |   3.22 |  1.49×  |
| resolve_group_labels   |   0.01 |   0.01 |  ~1×    |
| ini_params             |   2.19 |   1.00 |  2.19×  |
| radius_mask            |   4.88 |   2.75 |  1.78×  |
| **TOTAL**              | **335.30** | **219.83** | **1.53×** |

`generate_profiles` dominates both backends (~96 % of total). The
per-(sub, sess) breakdown on the GPU path is heavily CPU-side: gzip
decode of `.nii.gz` ≈ 263 ms, `.b2nd` write (bitpack + LZ4) ≈ 295 ms
amortized, D2H + MW-zero ≈ 122 ms, vs only ~30 ms of actual GPU work.
The GPU sits idle ~95 % of the time on the raw per-iter wall.

To collapse this we run `generate_profiles` as a step0-style
phase-pipeline. Four stages, each on its own worker thread, with
`queue.Queue(maxsize=2)` backpressure between them:

```
[GZIP pool] → qIN → [Stage GPU] → qPOST → [Stage POST] → qASSEM → [Stage ASSEM] → [WRITE pool]
 8 workers          1 thread             1 thread            1 thread          1 worker
 ~30 ms / sess      ~180 ms / sess       ~200 ms / sess      buffers 6 sess    295 ms / sess
                    H2D + sgemm + D2H    (K,V)→(V,K) copy    per sub           bitpack + LZ4
```

* **`Step1BoldPrefetcher`** ([`_step1_bold_prefetcher.py`](../arealmshbm/pipeline/_step1_bold_prefetcher.py))
  — persistent 8-worker pool decodes BOLD ahead of the GPU consumer
  with an 8-session sliding lookahead. base64 + isal_zlib release the
  GIL on the per-darray decode, so the 240-session phase saturates the
  codec in parallel.
* **`Step1GenerateProfilesStagePipeline`** ([`_step1_stage_pipeline.py`](../arealmshbm/pipeline/_step1_stage_pipeline.py))
  — coordinator that owns the GPU / POST / ASSEM worker threads and
  the WRITE pool. The prime loop maintains the prefetcher's sliding
  window from its own thread, throttled naturally by qIN's
  backpressure.

Steady-state per-(sub, sess) wall is `max(stage_walls)` instead of
`sum` — the WRITE pool (LZ4 + bitpack) at 295 ms / sess amortized
becomes the gating stage. All earlier work (gzip + GPU + `.T` copy
+ assembly) overlaps inside that window.

Combined effect on the YS cohort:

| Variant | `generate_profiles` wall | Speedup vs baseline |
|---|---:|---:|
| Baseline (serial main-thread loop) | 212.72 s | 1.00× |
| + `Step1BoldPrefetcher` + bg WRITE pool (1 stage) | 123.09 s | 1.73× |
| + `Step1GenerateProfilesStagePipeline` (4 stage) | **68.35 s** | **3.11×** |

Per-(sub, sess) mean drops 575 ms → 285 ms (-50 %). Correctness
preserved across both refactors — bit-exact / sub-ULP outputs vs the
serial-loop baseline (measured with an internal per-subgraph
probe harness).

End-to-end driver wall (internal cohort run, 2026-05-22):
step 0 — 118.18 s · step 1 — **78.83 s** · step 2 — 254.58 s · step 3 — 128.45 s ·
**total 580.32 s (9.67 min)**. The driver's step 1 (78.83 s) is the
68.35 s standalone `generate_profiles` plus ~10 s of `avg_profiles` +
`ini_params` + `radius_mask` + cohort write + data_list staging. End-
to-end total drops **22.6 %** vs the pre-step1-perf E2E (749 s →
580 s); step 1 alone accounts for 149 s of that.

Data bit-exactness: the on-disk `.b2nd` blosc2 frame hashes differ
across runs (container metadata is non-deterministic), but the packed
uint8 payload AND the unpacked fp32 binary masks are 100 % bit-exact —
verified on all 40 subjects by check-out-and-recompile against the
pre-step1-perf baseline commit. The stage pipeline is a pure
scheduling change with zero semantic delta.

The thin subgraphs (`avg_profiles`, `ini_params`, `radius_mask`) all
win 1.5–2.2× on GPU. `radius_mask` benefits from batched pull-based
Bellman-Ford replacing 600 serial Dijkstra calls; `ini_params` from
cuBLAS sgemm on the `(N, D) @ (D, L)` projection.

The cohort_manifest writer emits a single `cohort.json` at the
project root in sub-millisecond — not broken out above.

### Reproduce

```bash
# end-to-end Mode B (all steps)
python -m arealmshbm.pipeline projects/<name>
```

Per-step wall timing is recorded in the driver's per-run JSON log
under `<project>/logs/`.
