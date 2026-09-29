# Conventions

Shape conventions for new code in this repo. Each rule has a concrete
reference inside the existing tree — copy that shape, don't reinvent.

## 1. Module shape

`arealmshbm/` is laid out as one folder per algorithmic primitive plus
one orchestrator folder per step. New steps follow the same shape:

- **Leaf modules** — one folder per primitive, e.g. `m_step/`,
  `V_lambda/`, `vmf_clustering/`, `data_io/`:
  - `__init__.py` — small public API, usually one or two functions.
  - `<leaf>.py` — top-level entry that orchestrates the kernels.
  - `_kernels.py` — numba CPU kernels, where the leaf has any.
  - `_kernels_gpu.py` — CuPy / CUDA kernel sources for the leaf's GPU
    path (`<leaf>_gpu.py` or a session module drives them). Leaves
    with a CPU and a GPU path carry both (`vmf_clustering/`,
    `generate_profiles/`, `ini_params/`, `radius_mask/`,
    `step2_em_iter_master/`); `graph_distance/` and `watershed/` carry
    `_kernels_gpu.py` and keep their numba CPU kernels in a named
    module (`graph_distance.py`, `watershed_repaired.py`); many leaves
    keep their kernels inline or in a named module instead
    (`m_step/_xdot_kernel.py`, `data_io/_gifti_kernels*.py`).
  - `_cdln.py` / `_invad.py` / etc. — math helpers private to the
    leaf.
- **Pipeline orchestrator** — `<stepN>_pipeline/` per step, layout:
  `config.py` (dataclass with all knobs, validation in `__post_init__`)
  + `pipeline.py` (single-subject load → run → save lifecycle).
  Multi-subject batch runs go through the unified
  `arealmshbm/pipeline/` driver (mode dispatch). Step 3 also has a
  step-specific `variant.py` for the three Areal-MSHBM variants —
  that is unlikely to recur, don't add a `variant.py` to other steps
  without an analogous reason.
- **IO leaves are shared across steps** — `data_io/` holds readers
  for the gradient `.mat`/`.npy`, the bit-packed `.b2nd` profile, group
  prior, and spatial mask. New steps reuse these; format-specific
  per-step loaders live under `step<N>_io/` only when the new format
  doesn't fit the shared shape (see `step2_io/` for an example).

## 2. Backend dispatch

`Step3Config.backend` is `'cpu' | 'gpu'`, like every step;
`Step3Pipeline` routes it to `VmfClusteringSession`
(`vmf_clustering/vmf_clustering.py`) or the candidate-set session
`VmfClusteringSessionSparseCUDA` (`vmf_clustering/vmf_clustering_gpu.py`).
New steps adopt the same shape: a config-level `backend` field, side-by-
side `<leaf>.py` / `<leaf>_gpu.py` files only where GPU divergence is
real.

GPU steps must release the cupy memory pool between subjects in batch
loops; the `with Step3Pipeline(cfg) as pipe` pattern is the reference
(`Step3Pipeline.__exit__` → `close()` frees the default and pinned
pools, `arealmshbm/step3_pipeline/pipeline.py`). Skipping it shows up
as a non-decreasing `nvidia-smi` memory curve across subjects.

## 3. Numba AOT cache hygiene

Every CPU kernel is decorated `@njit(cache=True)`. Numba persists the
compiled object next to the source under `__pycache__/` (the standard
gitignored cache dir — there is no separate `.numba_cache`). The cache
is keyed on each kernel's signature, its bytecode and the host CPU
(target triple, CPU name and feature set); numba discards a kernel's
entries when the numba version or the mtime/size of the kernel's own
source file changes. A checkout moved to a different CPU, or a numba
upgrade, therefore recompiles by itself. The key does not cover
`@njit` helpers called from other files (`graph_distance/_heap.py`,
`em_stop_criterion/_cdln.py`, `m_step/_invad.py`): after editing one,
clear the cache, or its callers keep running the old compiled copy:

```bash
# from the repo root:
find arealmshbm -type d -name __pycache__ -prune -exec rm -rf {} +
```

## 4. Testing

Per-leaf pytest suites live next to the code under
`arealmshbm/<leaf>/tests/`. They cover unit-level invariants (NaN
clamping, mask-wraparound, kernel numerical match between CPU and GPU
backends, cohort.json merge semantics, etc.) — not end-to-end
parcellation quality, which is measured by the out-of-tree edge-ICC
scoring pipeline.
