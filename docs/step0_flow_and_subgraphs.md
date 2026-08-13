# Step 0 — call graph + module map

Structural reference for the production Python step-0 super-call.
Maps each call site of CBIG `Kong2022_ArealMSHBM` step 0 to the
Python module that owns it.

For the migration conventions drawn from this design (validate /
profile pattern, leaf module shape, backend dispatch), see
[`strategy.md`](strategy.md).

## The whole flow

```
Step0Pipeline.run(cfg)                                         [arealmshbm/step0_pipeline]
│
├── load_inputs()
│   ├── load_avg_mesh(lh/rh, sphere)                          data_io/load_avg_mesh
│   ├── read_surface_mesh(lh/rh midthickness)                 surface_io
│   ├── compute_topology(lh/rh sphere)                        mesh_topology
│   ├── prepare_smoothing_mesh(lh/rh midthickness)            surface_smoothing/_geodesic_kernels
│   │     (vertex areas + Layer 1/2 polyhedral-geodesic CSR)
│   ├── prepare_smoothing_gather(prep, roi, sigma)            surface_smoothing
│   │     (74k per-source Dijkstra → gather CSR + inv weight-sum;
│   │      depends only on mesh + roi + cfg.smooth_sigma, so it's
│   │      hoisted here and reused for every iter_a smoothing call)
│   └── neighbors_exclude_medial + find_K_neighbors            step0_neighbors
│
├── subgraph A — RSFC gradients
│   ├── for each scan:
│   │   ├── read_surface_bold(lh/rh)                          bold_io
│   │   ├── concat_hemis_drop_medial                          bold_io
│   │   ├── set_downsample_params(scan_idx)                   subsampling
│   │   ├── compute_t_series(curr_data, randinds_FC)          fc_similarity
│   │   └── for each block-a (~3 iters):
│   │       ├── compute_FC_simi_block(...)                    fc_similarity
│   │       ├── cifti_gradient(FC_simi_block, midthickness)   surface_gradient
│   │       └── accumulate per-block (sum, count)
│   └── for each block-a:
│       ├── avg_grads = sum / count                           inline
│       ├── cifti_smoothing(avg_grads, lh/rh_smooth_gather)   surface_smoothing
│       │     (SpMV + per-row normalize on the cached gather)
│       ├── find_minima(smoothed, K_neighbors)                local_minima
│       └── watershed_algorithm_repaired(smoothed, ...)       watershed
│           (per-column threshold sweep; processes all K cols)
│
├── subgraph B — gradient distance (per hemi)
│   ├── make_icosphere(n_target)                              icosphere
│   ├── compute_topology(down_sphere)                         mesh_topology
│   ├── linear_interpolate_sphere(downsample edge_density)    interpolate_sphere
│   └── gradient_geodesic_distance(down_mesh, grad_edge_down) graph_distance
│
├── subgraph C — diffusion embedding (per hemi)
│   └── compute_diffusion_map_repaired(dist, alpha=0.5, n_components)  diffusion_map
│       (A = exp(-D/D.max()) → alpha-norm → partial Lanczos eigsh;
│        cpu | gpu backend dispatch — scipy / cupyx)
│
├── subgraph D — upsample emb (per hemi)
│   └── linear_interpolate_sphere(emb_down → full sphere)     interpolate_sphere
│
└── save()
    ├── edge_density.npy
    ├── {lh,rh}_gradient_distance_matrix.npy
    └── {lh,rh}_emb_<num_comp>_distance_matrix.npy   (preferred; .mat legacy via cfg.emb_output_format)
```

For `mesh = 'fsaverage6'`, `sub_FC=100`, `sub_verts=200`, `downsample=3.2`:

| Quantity | Value |
|---|---|
| `N` (full mesh per hemi) | 40962 |
| `N` (concat L+R) | 81924 |
| medial wall mask (lh+rh) | ~3592 verts → cortex ≈ 78332 |
| `num_sample_FC` (N2) | round(78332/100) = 783 |
| `num_sample_S` (N1) | round(78332/200) = 392 |
| `iter_a` (FC_A blocks) | 4 |
| `iter_b` (FC_B blocks) | 11 |
| downsample sphere verts | round(81924 / 6.4) = 12801 → icosphere: **12962** |
| diffusion components | 100 |

## Module / file map

```
arealmshbm/
├── step0_pipeline/                         # top-level Python entry
│   ├── config.py                           Step0Config dataclass
│   ├── pipeline.py                         Step0Pipeline lifecycle (single subject)
│   └── profile.py                          per-stage wall-time report
│       # Multi-subject batch runs go through arealmshbm/pipeline/
│       # (unified driver, mode='modeA_batch').
│
├── data_io/                                # shared with step 3
│   └── load_avg_mesh.py                    fsaverage* mesh
│
├── surface_io/                             FreeSurfer .sphere + GIFTI .surf.gii
├── bold_io/                                FreeSurfer surface .func.gii BOLD (GIFTI only)
├── mesh_topology/                          vertex_nbors / vertex_faces
├── step0_neighbors/                        neighbors_exclude_medial + K-hop
├── subsampling/                            FC speed-up randinds (legacy np.random)
├── fc_similarity/                          block-wise FC-similarity matrix
│                                           (BLAS sgemm + fused numba kernels)
├── surface_gradient/                       per-vertex tangent-plane LS
│                                           (replaces wb -cifti-gradient)
├── surface_smoothing/                      geodesic Gaussian (numba prange)
│   └── _geodesic_kernels.py                polyhedral-geodesic Dijkstra +
│                                           mesh-only precomputation
│                                           (replaces wb -cifti-smoothing)
├── local_minima/                           K-hop local-min mask
├── watershed/                              watershed_algorithm_all_par_cifti
│   └── watershed_repaired.py               production path (all K cols;
│                                           K-mod-4 column-drop repair)
├── icosphere/                              icosahedral subdivision
│                                           (replaces wb -surface-create-sphere)
├── interpolate_sphere/                     barycentric-on-sphere
│                                           (replaces MARS_linearInterpolate)
├── graph_distance/                         numba prange all-pairs Dijkstra
└── diffusion_map/                          distance → affinity + mapalign Lanczos top-k
    ├── diffusion_map_repaired.py           CPU production wrapper: D → A then delegate
    ├── diffusion_map_gpu_repaired.py       GPU production wrapper: D → A then delegate
    ├── diffusion_map.py                    CPU mapalign-class Lanczos body
    │                                       (called by diffusion_map_repaired)
    └── diffusion_map_gpu.py                GPU mapalign-class Lanczos body
                                            (called by diffusion_map_gpu_repaired)
```

Backend selection happens at `Step0Config.backend ∈ {'cpu', 'gpu'}`.
`backend='gpu'` routes two leaves to CuPy + cuBLAS:

  * **`fc_similarity`** (`fc_similarity_gpu.py`) — `curr_data`,
    `t_series`, `mag_t`, `FC_A` stay device-resident across one
    session's iter_a / iter_b loop; only the per-iter_a
    `FC_simi_block` is D2H-copied because the downstream
    `cifti_gradient` is still numba CPU.
  * **`diffusion_map`** (`diffusion_map_gpu_repaired.py`) — partial
    Lanczos eigsh on the (N_down × N_down) affinity via cuSPARSE.

Every other leaf stays on the CPU numba path. The remaining
GPU-port candidates (surface_gradient, surface_smoothing's per-call
SpMV, subgraph B's all-pairs Dijkstra) are bandwidth- or
memory-latency-bound — see the per-stage timing breakdown in
`step0_pipeline/profile.py` output before deciding what to port next.

## Wall

Mode-B end-to-end on the YS cohort (**40 subjects × 6 sessions ×
fsaverage6 × L=300, gMSHBM**). RTX 5090 Laptop, 24 GB. 2026-05-20.
Source: internal 40-subject × 6-session cohort run; raw timing log
archived internally.

End-to-end wall on the production GPU path: step 0 — **227.41 s** ·
step 1 — 223.85 s · step 2 — 373.94 s · step 3 — 129.35 s ·
**total 954.84 s (15.91 min)**. Step 0 is 24 % of the end-to-end
total — per-subject independent, the unified driver runs all 40 in
sequence (no per-subject parallelism wrapping the driver yet).

### Per-subject wall (3 representative subjects, warm cache except sub-001)

| sub        | CPU wall (s) | GPU wall (s) | GPU speedup |
|------------|-------------:|-------------:|------------:|
| sub-001 †  |   28.75      |    4.18      |    6.9×     |
| sub-002    |   25.27      |    4.43      |    5.7×     |
| sub-003    |   25.03      |    4.19      |    6.0×     |
| **median (warm)** | **25.15** | **4.31** | **5.8×**  |

† sub-001 carries the cold numba JIT load (~3 s extra under
``load_inputs``); sub-002 / sub-003 are warm and representative.

### Per-subgraph breakdown (sub-002, warm)

| Subgraph | CPU (s) | GPU (s) | Bottleneck |
|---|---:|---:|---|
| load_inputs            |  0.00 |  0.06 | mesh + smoothing-gather one-shot |
| A — RSFC gradients     |  6.29 |  2.33 | fc_similarity (cuBLAS sgemm + fused demean+norm) |
| B — gradient distance  |  1.69 |  0.93 | Gauss-Seidel pull BF on (N, N) (CuPy RawKernel) |
| C — diffusion embedding | 17.17 |  0.96 | scipy.eigsh → cupyx.scipy eigsh; dist consumed device-resident |
| D — upsample emb       |  0.06 |  0.07 | trivial |
| **WALL (sub-002)**     | **25.27** | **4.43** | — |

Subgraph C alone is **68 % of the CPU per-subject wall** (Lanczos partial
eigh on the N_down × N_down affinity); the GPU port routes it through
cuSPARSE eigsh and recovers ~17× on that leaf. Subgraph A's GPU win
(~2.7×) is from `fc_similarity` (curr_data + t_series device-resident
across iter_a/iter_b, sgemm via cuBLAS, fused demean+norm RawKernel
replacing the prior fp64-cast-and-reduce chain — see
[`## fc_similarity fused demean+norm`](#fc-similarity-fused-demean-norm)
below); the rest of A (watershed, local_minima, surface_smoothing's
per-iter SpMV) stays on the numba CPU path. Subgraph B's GPU port is
documented inline under [`## Subgraph B GPU port`](#subgraph-b-gpu-port).

### Subgraph B GPU port

Sub-B used to be the largest GPU-wall single leaf (1.72 s, 30 % of the
old GPU per-sub wall) because the CPU numba prange Dijkstra ran on
both backends — no GPU port. Replaced in this fork with a
[**batched pull-based Bellman-Ford**](../arealmshbm/graph_distance/_kernels_gpu.py)
on the full (N, N) fp32 distance matrix:

* **Layout invariant.** Store ``D[v, s]`` (destination outer, source
  inner). A warp of 32 threads with shared ``v`` and consecutive ``s``
  reads any neighbor row ``D[u, s..s+31]`` as one 128 B coalesced
  transaction.
* **Pre-fused edge weights.** ``(g_v + g_u) / 2`` is computed once
  into ``edge_weights[N, M] fp32`` with a ``+inf`` sentinel for
  absent slots — the iter kernel needs no per-slot validity branch.
* **Bank-conflict-free broadcast shared mem.** Per-block cache of
  the M=6 vertex_nbors + edge_weights for the block's one ``v``;
  all 256 threads read the same ``slot`` index in the hot loop
  → CUDA broadcasts from a single bank in one cycle.
* **Gauss-Seidel in-place updates.** Reads + writes the same buffer
  each iter. Race conditions are benign for SSSP (min-reduction is
  monotone, so stale reads are upper-bound candidates that never
  produce wrong answers). Drops iteration count from 131 (Jacobi
  double-buffer) to 114 — saves both compute and memory traffic.
* **Cross-leaf fusion.** Sub-B's GPU output stays device-resident;
  subgraph-C's ``compute_diffusion_map_gpu_repaired`` consumes the
  device ``cp.ndarray`` directly, skipping a 670 MB D2H per hemi
  (and the matching H2D inside diffmap that the host-numpy path
  would otherwise pay).

Per-leaf measurement at fsa6 (N=12962, M=6), both hemis:

| | CPU (numba prange) | GPU (pull-BF) | speedup |
|---|---:|---:|---:|
| sub-B (both hemis) | 1.70 s | **0.78 s** | **2.2×** |

The GPU kernel is bandwidth-bound at ~2.6 ms/iter (theoretical floor
~2 ms/iter for the (N, N) × 7 fp32 read pattern), so further wins
require either fp16 storage (precision risk on the ~100-hop diameter)
or algorithmic change (Δ-stepping, frontier tracking). The
device-resident hand-off to subgraph C is the bigger win at this
scale — it shaves another ~150 ms/hemi of round-trip that doesn't
show up as a sub-B leaf cost but as a subgraph-C reduction (1.02 s
→ 0.95 s).

The CPU/GPU sub-B output comparison on real upstream-A inputs lands
at ``max-abs-diff ≈ 1e-2`` — that delta is the well-known step-0
RNG / Lanczos algorithm-class drift in subgraph A's gradient
production, propagated through into sub-B's input ``grad_data``,
not a regression in the BF kernel. On synthetic inputs (same
grad_data fed to both backends) sub-B's CPU/GPU max-abs-diff is
**7.7e-7** — bit-equivalent to fp32 ULP. The final ``emb_up``
top-7 ``|cos|`` between Python CPU and Python GPU stays at **1.0000**
(well above the historical reference bar of 0.96).

### fc_similarity fused demean+norm

Sub-A's biggest non-GEMM cost was `_demean_norm_columns` /
`_demean_norm_rows` — a 5-pass implementation with fp64-buffer
materialization:

```python
mean64 = x_d.astype(cp.float64).mean(axis=0, keepdims=True)
x_d -= mean64.astype(cp.float32)
ss64 = (x_d.astype(cp.float64) ** 2).sum(axis=0)
mag = cp.sqrt(ss64).astype(cp.float32)
```

Each `astype(cp.float64)` materialises an fp64 array 2× the size of
the fp32 input; the `.mean` + `**2.sum` then each do another full
scan. Called 576× per subject across the iter_a × iter_b loop,
totalling ~570 ms per subject — about a quarter of sub-A's GPU wall
and 7× the cost of all sgemms combined.

Replaced in this fork with a [**fused RawKernel**](../arealmshbm/fc_similarity/_kernels_fused_gpu.py)
that does both passes through the fp32 buffer with fp64 accumulators
**in registers** (no global-memory fp64 staging):

* **Pass 1** — accumulate fp64 mean over the reduction axis.
* **Pass 2** — in-place demean + accumulate fp64 SS in the same pass;
  pass-2 reads hit L1 cache from pass-1.

Layout-specific kernels for the two callers — same algorithm, flipped
coalescing pattern:

* `axis0` ((T, K) row-major, reduce over T): block (32, 1); each
  thread owns one column j. Warp loads `x[t, j..j+31]` = 32
  consecutive fp32 → one coalesced 128 B transaction. No shared
  memory.
* `axis1` ((K, T) row-major, reduce over T): block (256, 1), one
  block per row k. Threads cooperate via an 8-byte-per-cell shared
  fp64 array with stride-1 access — no bank conflicts in the tree
  reduce.

Memory traffic per call (B = buffer bytes):

|                            | passes | DRAM traffic | speedup |
|----------------------------|-------:|-------------:|--------:|
| prior `_demean_norm_*`     | 5      | ~16 B        | 1×      |
| fused RawKernel            | 2      | ~3 B         | ~5× theoretical |

Measured on the FC_B-shaped buffer (T=240, K=7833) over 50 reps:

| | wall / call |
|---|---:|
| prior implementation | 0.478 ms |
| fused RawKernel      | **0.065 ms** (**7.37×**) |

Precision contract: the demeaned `x_d` is **bit-identical** to the
prior implementation per cell (same `x - float(mean64)` arithmetic);
`mag` differs by ≤1 ULP fp32 from a different reduction-tree shape.
The final emb_up top-7 |cos| Python-CPU vs Python-GPU stays at
**0.9999** (vs historical reference bar of 0.96).

End-to-end impact on sub-002 sub-A wall: **2.76 → 2.33 s (-15%)**.
Across the 40-subject cohort that's ~17 s saved on step 0 e2e
(roughly noise-band but consistent on multi-run averaging).

### Cohort scaling

End-to-end step 0 wall on GPU is **227.41 s for S=40** (~5.7 s/sub),
matching the warm per-subject median (4.6 s) plus ~1.0 s/sub of
``Step0Pipeline`` per-subject construct/teardown + ``save()`` disk
I/O + ``cohort.json`` hand-off. Extrapolating from the warm CPU
median: 40 × 25.0 ≈ **1002 s (~16.7 min) if step 0 ran CPU**.

### Reproduce

```bash
# end-to-end Mode B (all steps)
python -m arealmshbm.pipeline projects/<name>
```

Per-step wall timing is recorded in the driver's per-run JSON log
under `<project>/logs/`.

## RNG / orientation tolerances

Every hot-path kernel is `@njit` numba CPU (with one CuPy path in
subgraph C); the tech-stack exceptions delegate to BLAS sgemm (numpy
`@`) and Lanczos partial eigh (`scipy.sparse.linalg.eigsh` /
`cupyx.scipy.sparse.linalg.eigsh`). Three RNG / orientation
tolerances are inherent to the algorithm semantics:

* **L5 `subsampling`** seeds `np.random.seed(scan_idx)` and uses
  `np.random.permutation`. Any uniform permutation is acceptable;
  downstream consumers tolerate the draw.
* **L10 `watershed`** seeds catchments via a permutation of
  ``1..n_min`` and visits below-threshold vertices in a fresh
  permutation per threshold step. The boundary mask (label == 0) is
  statistically stable, which is what the downstream `edge_density`
  accumulator consumes.
* **L11 `icosphere`** orientation depends on the subdivision-
  traversal order; our pure-numpy builder produces the canonical
  V/F counts and a valid sphere triangulation, rotated by an
  implementation-defined transform.

sub-005 historically had a stuck-vertex session whose dynamic-range
collapse made the subgraph-C Lanczos eigensolver operate near a
dense-cluster eigenvalue. The fc_similarity NaN-clamp keeps the
gradient field in range so the cohort runs cleanly end-to-end.
