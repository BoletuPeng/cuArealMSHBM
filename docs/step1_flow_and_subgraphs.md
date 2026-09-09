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
│   gpu: prewarm_step1_gpu() on a daemon thread (started before
│        step 0 in the driver); subgraphs chained in memory / on
│        device; disk writes on background threads, joined once
│        at the end (step1_runners.join_step1_writers).
│
├── resolve_group_labels(targ_mesh, schaefer_resolution)       data_io/annot_io.read_annot_labels (numpy-only .annot)
│
├── run_generate_profiles(seed_mesh, targ_mesh, sub, sess, …)  pipeline/step1_runners.py → generate_profiles
│   ├── gpu: generate_subject_profiles_gpu (one call per subject) generate_profiles/profiles_subject_gpu.py
│   │   ├── batched nvCOMP GIFTI ingest, raw (T, n_lh+n_rh)    data_io/gifti_bold_gpu.iter_subject_bold_gpu(raw_sessions=True)
│   │   │   (pinned readinto → H2D → GPU bytescan + base64 →
│   │   │    one Deflate batch per 2-session group, own streams)
│   │   ├── per session, all on device:
│   │   │   ├── zscore + L2-unit-norm, ALL n_full columns      _kernels_gpu.zscore_unit_norm_columns_zerovar_cupy
│   │   │   │   (zero-variance ⇒ exact 0.0f, so no nan_to_num)
│   │   │   ├── seed gather (global column idx)                cupy fancy-index
│   │   │   ├── ONE sgemm (K, T)@(T, n_full) → (K, n_full)      cuBLAS (= the joint lh|rh array, no concat)
│   │   │   ├── exact k-th order statistic (radix histogram)   _kernels_gpu.exact_kth_smallest_cupy (3 streaming passes)
│   │   │   └── binarize + MW-zero + pack → slab t of          _kernels_gpu.binarize_mwzero_pack_cupy
│   │   │       device (n_sess, N, ⌈K/8⌉) uint8
│   │   ├── per-session pinned D2H of slab t (event-gated)
│   │   └── .b2nd streamed one chunk (= session) at a time on  data_io/profile_io.SubjectProfileStreamWriter
│   │       a background thread; .wait() joins the last slab
│   │   (no nvCOMP ⇒ same leaf, CPU-reader ingest: read_surface_gifti
│   │    (time_major, strict b64) on an 8-worker pool → pinned
│   │    (T, n_lh+n_rh) → H2D. Same accepted files, same bytes, so the
│   │    artifacts are identical by construction; the verbose line
│   │    says which ingest ran)
│   └── cpu: stage pipeline (unchanged)                        pipeline/_step1_stage_pipeline.py
│       ├── parallel BOLD reads (.func.gii, numba b64 + isal)  data_io/gifti_io.read_surface_gifti
│       ├── seed_select, zscore, MKL GEMM, nan_to_zero, sum    generate_profiles/_kernels.py (numba)
│       ├── threshold_top_fraction (introselect), binarize     np.partition / numba
│       └── per-subject .b2nd writer                           data_io/profile_io.write_subject_profile_tnd
│
├── run_avg_profiles(…, packed_subjects, D)                     pipeline/step1_runners.py → avg_profiles
│   ├── gpu: avg_profiles_from_packed_gpu                      avg_profiles/avg_profiles_gpu.py
│   │   ├── packed slabs from ``packed_sink`` (no .b2nd re-read) H2D once per subject
│   │   ├── fused bit-unpack + accumulate (no atomics), scale   RawKernel accum_packed_session_NhDb
│   │   ├── one pinned D2H shared with the writer               AvgProfilesResult.lh_avg / rh_avg (host)
│   │   └── .npy pair write on a background thread              AvgProfilesResult.writer.wait() joins
│   │   returns lh_avg_dev / rh_avg_dev (device fp32) for subgraph 3
│   └── cpu: per-subject .b2nd packed reads → numba accum → np.save
│
├── run_ini_params(…, precomputed_{lh,rh}_avg_dev, save_async)  pipeline/step1_runners.py → ini_params
│   ├── gpu: concat + fp64 widen ON DEVICE (no host round trip) ini_params/ini_params_gpu.py
│   ├── zero_mw + detect_nonzero                               _kernels[_gpu] (numba / cupy primitives)
│   ├── demean + L2-unit-norm rows                             _kernels[_gpu] (numba / RawKernel)
│   ├── groupsum (mtc): per-parcel row sums in ascending-i     _kernels (numba) / _kernels_gpu.groupsum_csr_cupy
│   │   order — GPU bit-exact vs the numba kernel               (parcel CSR; replaces the dense one-hot dgemm)
│   ├── column L2-renorm in numpy's sum(axis=0) order          _kernels_gpu.colnorm_scale_cupy
│   ├── ε input: Σ_i ⟨profile[i], mtc[:, l_i]⟩ (row-dot)       _kernels (numba GEMM+gather) / _kernels_gpu.epsil_input_rowdot_cupy
│   ├── invAd(D-1, rbar)                                       _invad (scipy Bessel, CPU only)
│   └── group.mat write, compressed as before                  ini_params/_group_mat_writer.py
│       (gpu: on a background thread — same bytes, off the wall)
│
├── run_radius_mask(lh_labels, rh_labels, mesh, radius)        pipeline/step1_runners.py → radius_mask
│   ├── _build_mesh_csr (MARS rescale + vertexNbors + checks)  _common (numpy)
│   ├── build_parcel_csr                                       _kernels (numba; CPU on both backends)
│   ├── classify_central_relevance                             _kernels (numba; CPU on both backends)
│   ├── per-parcel-mean geodesic to central sulcus
│   │   ├── cpu: per-source Dijkstra, prange over sources      _kernels.central_sulcus_kernel
│   │   └── gpu: delta-stepping frontier SSSP, 32 src per CTA  _kernels_gpu.central_sulcus_distances_cupy
│   ├── per-parcel bounded radius mask
│   │   ├── cpu: per-parcel Dijkstra, prange over parcels      _kernels.add_spatial_constraint_kernel
│   │   └── gpu: batched pull-based BF bounded by radius       _kernels_gpu.bellman_ford_bounded_cupy
│   ├── truncate (anatomical correction)                       _kernels (numba; CPU on both backends)
│   └── mask → csc + savemat(compressed, as before)            _common._mask_to_csc + scipy.io
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

The driver runs all four subgraphs in both modes — Mode A simply has
no consumer for two of the artifacts. cohort.json therefore always
names ``group.mat`` and the avg-profile ``.npy`` pair.

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
│   ├── _kernels_gpu.py                      CuPy RawKernels: zero-variance zscore (shared-mem
│   │                                        reduce), exact radix k-th select,
│   │                                        binarize+MW-zero+pack
│   ├── profiles.py                          CPU supercall (per session)
│   └── profiles_subject_gpu.py              fused WHOLE-SUBJECT GPU leaf (the GPU path;
│                                            owns its ingest, nvCOMP or CPU reader)
│
├── avg_profiles/                            # subgraph 2 — sum/divide across sub × sess
│   ├── _kernels.py                          numba CPU: accum_inplace, scale_inplace
│   ├── avg_profiles.py                      CPU supercall + dispatch shim (returns AvgProfilesResult)
│   └── avg_profiles_gpu.py                  GPU supercall: disk-reading avg_profiles_gpu +
│                                            memory-side avg_profiles_from_packed_gpu
│
├── ini_params/                              # subgraph 3 — vMF init from group labels
│   ├── _kernels.py                          numba CPU kernels
│   ├── _kernels_gpu.py                      CuPy RawKernels: row-demean+L2, CSR groupsum,
│   │                                        column L2-renorm, fused row-dot ε reduction
│   ├── _invad.py                            Bessel-ratio root solver (fp64, CPU only)
│   ├── _group_mat_writer.py                 group.mat writer (sync or background thread)
│   ├── ini_params.py                        CPU supercall + dispatch shim
│   └── ini_params_gpu.py                    GPU supercall (no GEMM; ordered reductions)
│
├── radius_mask/                             # subgraph 4 — geodesic + per-parcel mask
│   ├── _kernels.py                          numba CPU kernels (Dijkstra + classify + truncate)
│   ├── _kernels_gpu.py                      CuPy RawKernels: bounded pull-based Bellman-Ford
│   │                                        (radius mask) + sssp_batch32 (central sulcus)
│   ├── _common.py                           mesh CSR from vertexNbors, aparc read, mask→csc
│   ├── radius_mask.py                       CPU supercall + dispatch shim
│   └── radius_mask_gpu.py                   GPU supercall (classify+truncate stay CPU)
│
└── data_io/
    ├── profile_io.py                        per-subject .b2nd writer/reader (blosc2 LZ4+bitshuffle,
    │                                        clevel 5) — whole-subject and per-session streaming
    │                                        writers; subgraph 1 writes, subgraph 2 reads.
    ├── annot_io.py                           numpy-only FreeSurfer .annot label reader
    ├── _background_write.py                  join handle for off-thread artifact writes
    ├── gifti_bold_gpu.py                     batched nvCOMP whole-subject BOLD ingest
    └── gifti_io.py                          read_surface_gifti (CPU BOLD ingest: bytescan +
                                             isal_zlib; bit-equal with gifti_bold_gpu).
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

Each runner takes a ``backend ∈ {'cpu', 'gpu'}`` keyword. For
subgraphs 2–4 it passes it straight through to the leaf supercall,
where a single `if backend == 'gpu':` branch lazily imports the
`<leaf>_gpu.py` sibling and tail-calls it. Subgraph 1 dispatches in
the runner instead: the two backends have different shapes (a
per-session CPU leaf fed by the stage pipeline vs. one whole-subject
GPU call), so `run_generate_profiles` calls
`generate_subject_profiles_gpu` directly and the CPU leaf
`compute_profile_arrays` has no `backend` keyword. cupy is never
imported on the CPU path.

GPU paths inline locally (no per-stage H↔D round-trips within a
supercall):

* `generate_subject_profiles_gpu` (the GPU entry, one call per
  subject): raw-session ingest → per session one zscore launch over
  all `n_lh + n_rh` columns (a zero-variance column is written as
  exact `0.0f`, which is what `nan_to_num` produced after the retired
  `+inf` normaliser — so that pass is gone) → one cuBLAS sgemm
  `(K, T) @ (T, n_full)` → exact k-th order statistic by a three-pass
  radix histogram over an order-preserving uint32 key
  (`np.partition`-identical, and with no permutation buffer, unlike
  `cp.partition`) → fused binarize + MW-zero + pack into slab `t` of a
  device `(n_sess, N, ⌈K/8⌉)` buffer → per-session pinned D2H →
  background `.b2nd` write. The packed payload is bit-identical to the
  per-session GPU leaf this replaced, whose algorithm now lives only
  as the oracle in `test_subject_profiles_gpu.py` — an empirical fact
  at the production shape (two per-hemi sgemms plus `nan_to_num`
  against one whole-subject sgemm), which is why it stays pinned. The
  path is fp32 only, so a non-`float32` `profile_dtype_reduce` is
  rejected rather than ignored.

  **Ingest** is the leaf's own choice, made once from
  `_nvcomp_batched.nvcomp_available()` and reported as `.ingest` on
  the result (the runner prints it in the per-subject line): batched
  nvCOMP device decode when the library is installed,
  `read_surface_gifti(time_major=True, allow_wrapped_b64=False)` on
  an 8-worker pool (up to 4 pairs ahead, ramped from one; assembled
  into a pooled pinned block; H2D on the producer thread's cached
  stream) when it is not. The strict flag makes the two ingests accept
  the same files — a line-wrapped payload is refused on both, not
  read on one machine and rejected on another — and
  `test_gifti_readers.py` pins the two readers byte-equal at the
  `(T, n_lh+n_rh)` boundary, so the compute cannot tell them apart
  and the artifacts are identical by construction — no fallback
  numerics, no dropped memory hand-off. The CPU-reader ingest still
  pays a per-subject start-up bubble (one file's decode before the
  first session lands, ~0.13 s at fsaverage6/T=242); a cross-subject
  prefetch would remove it but would restructure the runner for the
  no-nvCOMP case only, so it is not done.
* `avg_profiles_from_packed_gpu`: takes the packed slabs straight from
  `run_generate_profiles(packed_sink=…)`, accumulates on device
  (`accum_packed_session_NhDb`, integer-exact fp32), does one pinned
  D2H shared with the background `.npy` writer, and hands the device
  means (`lh_avg_dev` / `rh_avg_dev`) to subgraph 3. The disk-reading
  `avg_profiles_gpu` remains for cohorts run subgraph by subgraph, and
  clears the device fields so the pool can reclaim them.
* `ini_params_gpu`: device means → concat + fp64 widen on device →
  row-demean+L2 (RawKernel) → `groupsum_csr_cupy` (parcel CSR, rows
  summed in ascending order — bit-exact vs the numba kernel; the dense
  one-hot dgemm is gone) → `colnorm_scale_cupy` (numpy's `sum(axis=0)`
  order) → `epsil_input_rowdot_cupy` (one dot per row against its own
  parcel column; replaces the second dgemm plus its `(N, L)`
  intermediate and the gather) → scalar D2H for `invAd` (scipy Bessel
  stays CPU) → `group.mat` on a background thread. The file itself is
  unchanged: still `do_compression=True`, still the same fields.
* `radius_mask_gpu`: mesh CSR + labels + aparc H2D once per hemi.
  The **radius mask** (K = L_h ≈ 150 problems, bounded at 30 mm) stays
  on the batched pull-based Bellman-Ford: one RawKernel pass over V·K
  cells with neighbour pull-relax, looping until the `changed` flag
  stays 0. The **central-sulcus mean** (Nsrc ≈ 5000 problems,
  unbounded, run to full settle) needs hundreds of such sweeps of the
  full (V, Nsrc) matrix and is memory-roofline-bound, so it runs
  `sssp_batch32` instead: one CTA per 32-source column block, one warp
  per frontier vertex with `lane` = source (a vertex's 32 distances
  are one 128 B transaction), delta-stepping over a ring of
  shared-memory frontier bitmasks, gathering only the relevant-vertex
  rows. Distances are bit-identical to the BF solver. The frontier
  bitmask is static shared memory, so this backend needs V ≤ 90080 —
  fsaverage6 / fsaverage5 are fine, `fsaverage` is CPU-only for this
  leaf. Classify / truncate (heavy control flow, ~1 ms) stay CPU.

GPU memory-pool release is the caller's responsibility. The unified
driver releases the device pool at the end of step 1, after the
background writers are joined; `step1_pipeline/profile.py` releases
the device pool after each run and the pinned pool only at the very
end, so the pinned avg-profile D2H blocks stay page-locked across
runs exactly as the driver keeps them warm across a cohort.

## CPU ↔ GPU consistency

The GPU paths match CPU at sub-ULP across the four numeric subgraphs.
Subgraph 3's ε computes via a different fp64 reduction order between
numba and cupy, producing a ~4.7e-16 rel-diff that round-trips
through ``invAd`` without visible effect. Beyond that, CPU and GPU
outputs are bit-exact on the binary-mask and avg-profile paths.

## Single-subject wall (2026-09-04)

Bench: sub-001, 6 sessions × fsaverage6 (T=242), fsaverage3 seed
(K=1175), Schaefer-300, radius 30 mm; RTX 5090 Laptop. The bench
project and the frozen reference outputs live at `testdata/step1_bench/`
(`baseline_{gpu,cpu}/` were produced by HEAD 8cb68a0). Harness:
`python -m arealmshbm.step1_pipeline.profile --source-project
testdata/step1_bench/proj --temp-out <out> --schaefer-resolution 300
--backend {gpu,cpu} --runs N [--prewarm]`.

Every artifact matches the reference: `.b2nd` decoded packed bytes
`np.array_equal` (and the same file size — the codec is unchanged),
avg `.npy` pair byte-identical, `group.mat` `lambda` / `lh_labels` /
`rh_labels` identical, spatial mask decoded identical and still
compressed (637 KB). `mtc` differs by ≤ 7.8e-16 absolute (2.4e-15
relative to `max|mtc|`) and ε by 4.5e-13 absolute / 3.3e-16 relative —
fp64 reduction order, the same class of delta the leaf already
documented. The CPU backend is byte-identical throughout, `mtc` and ε
included. `mtc` and ε are the only step-1 artifacts step 2 reads, so a
Mode-B run whose step 1 ran on GPU is **not** bit-identical to a
clean-8cb68a0 GPU run: compare a GPU candidate against a GPU baseline,
or pin `backend_step1='cpu'` when a byte-exact A/B is what is
wanted.

| Subgraph (GPU, warm critical path) | 8cb68a0 | now |
|---|---:|---:|
| generate_profiles | 0.61 s | 0.18 s |
| avg_profiles | 0.32 s | 0.04 s |
| ini_params | 0.93 s | 0.04 s |
| radius_mask | 2.74 s | 0.40 s |
| join of background writes | — | 0.001 s |
| **TOTAL** | **4.71 s** | **0.66 s** |

Cold start in a fresh process is 3.2 s; with `prewarm_step1_gpu`
(RawKernel NVRTC compiles, cuBLAS handle, nvCOMP load, pinned
staging — 1.9 s) taken off the timed path the first run is 1.2 s. The
driver starts that prewarm on a daemon thread before step 0 and joins
it in `run()`'s `finally`.

CPU backend on the same data, cold single run: 13.9 s
(generate_profiles 7.7, avg 0.30, ini 1.43, radius_mask 4.49). It
picks up the kernel-only wins (mesh CSR from `vertexNbors`,
`_mask_to_csc`, the numpy-only `.annot` reader) but keeps the
disk-mediated flow.

## Wall

**Pre-2026-09 cohort numbers.** They predate the fused GPU chain
measured in the single-subject table above; the per-subgraph GPU
column here is the 8cb68a0 flow, kept for the cohort-scale shape.

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

`radius_mask` has since been re-cut (see the single-subject table
above: 2.74 s → 0.40 s on GPU, 4.49 s on CPU). The central-sulcus
family moved off the pull-based Bellman-Ford onto `sssp_batch32`; the
mesh CSR is built from the bundle's `vertexNbors` table (with exact
no-self-loop / no-duplicate / symmetry checks) instead of dedup'ing
the face edges; and the boundary mask goes to `csc_matrix` through a
transposed `flatnonzero` instead of a dense fp64 scan. All four
combinations (cpu/gpu × before/after) decode to byte-identical masks,
and the `.mat` is still written with `do_compression=True`.

Dead ends behind `sssp_batch32`, not kept in the tree: the textbook
near/far delta-stepping schedule (min-reduce the far pile, then rescan
it to promote) measured ~3× slower than the fixed 4-slot ring at
delta = 10 × mean weight and ~60× slower at delta = 3 ×, because every
bucket advance rescans the whole far pile; and warp-aggregating the
histogram atomics in `generate_profiles`' radix select was a small
regression, since that pass already runs at the card's read bandwidth.

The cohort_manifest writer emits a single `cohort.json` at the
project root in sub-millisecond — not broken out above.

### Reproduce

```bash
# end-to-end Mode B (all steps)
python -m arealmshbm.pipeline projects/<name>
```

Per-step wall timing is recorded in the driver's per-run JSON log
under `<project>/logs/`.
