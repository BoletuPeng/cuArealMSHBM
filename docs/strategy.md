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
  - `_kernels.py` — numba CPU kernels (present in every leaf).
  - `_kernels_gpu.py` — CuPy / CUDA kernels, **only** where the GPU
    path diverges from numba. When the same kernel runs on both via
    numba, keep just `_kernels.py` (most leaves are this case; only
    `vmf_clustering/` carries both).
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

`Step3Config.backend` is a single string knob (`'cpu' | 'gpu_elambda'
| 'gpu_full'`) routed to alternative session classes inside
`vmf_clustering/` (`vmf_clustering.py` / `vmf_clustering_gpu.py`). New
steps adopt the same shape: a config-level `backend` field, side-by-
side `<leaf>.py` / `<leaf>_gpu.py` files only where GPU divergence is
real.

GPU steps must release the cupy memory pool between subjects in batch
loops; the `with Step3Pipeline(cfg) as pipe` pattern is the reference.
The failure mode when this is skipped is a non-decreasing `nvidia-smi`
memory curve across the batch.

## 3. Numba AOT cache hygiene

Every CPU kernel is decorated `@njit(cache=True)`. Numba persists the
compiled object next to the source under `__pycache__/` (the standard
gitignored cache dir — there is no separate `.numba_cache`). The cache
is keyed on numba version + source hash, **not** on the host CPU's
feature set. If you migrate a checkout between machines with materially
different CPUs (e.g. clone from an AVX-512 box onto a CPU without it)
clear the cache once so kernels recompile for the new target:

```bash
# from the repo root, on the destination machine, once after first clone:
find arealmshbm -type d -name __pycache__ -prune -exec rm -rf {} +
```

Same applies if numba is upgraded across a major version. Day-to-day
edits inside a single checkout invalidate per-file automatically.

## 4. Testing

Per-leaf pytest suites live next to the code under
`arealmshbm/<leaf>/tests/`. They cover unit-level invariants (NaN
clamping, mask-wraparound, kernel numerical match between CPU and GPU
backends, cohort.json merge semantics, etc.) — not end-to-end
parcellation quality, which is measured by the out-of-tree edge-ICC
scoring pipeline.
