# cuArealMSHBM

<a id="english"></a>
**English** | [简体中文](#chinese)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21922832.svg)](https://doi.org/10.5281/zenodo.21922832)

**CPU/GPU-accelerated Areal-MSHBM** — individual-specific cortical
parcellation from resting-state fMRI, ported end-to-end to Python with
numba (CPU) and CuPy (GPU) backends. Validated at large scale (300+
subjects, 1,800+ fMRI sessions) against the original MATLAB
implementation. **v2.0.0** adds candidate-set sparse GPU backends for
steps 2 and 3 and a rewritten step-0 / step-1 GPU path: a single
subject now goes from raw BOLD to parcellation in **6.4 s** end to end
on one laptop GPU.

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
- **Candidate-set GPU backends** (step 2's `gpu`, step 3's
  `gpu_sparse`) — the EM runs on the candidate set implied by the group
  prior's support instead of the dense (N × L) posterior, so an entire
  cohort stays resident on the device for the whole run.

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
  1.45 GB of VRAM (vs ~11.6 GB unpacked), with popcount-based kernels
  consuming the packed form directly.
- **Kernel fusion and operator tuning** across the bottleneck steps:
  fused profile-generation kernels, fused E-step softmax chains with
  per-subject fp64 scratch (fp32 storage, fp64 where precision
  demands it), tuned reductions, and stream-pipelined BOLD decode that
  hides I/O behind GPU compute.

The result: on the heaviest step the speedup over the MATLAB
reference reaches **three orders of magnitude** on a single Blackwell
GPU (step-3 parcellation EM: 763.1 s → 0.6 s, ~1,300×; step-0 gradient
embedding: 233.6 s → 2.3 s, ~100×), and the full single-subject
pipeline runs **~170× faster** end-to-end.

## Status

**This is the second release (v2.0.0)**, containing the CPU and GPU
backends, with candidate-set GPU paths for steps 2 and 3. In large-scale validation — over **300 subjects
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
below was re-measured on the v2.0.0 tree, and each "total" is
end-to-end driver wall time on a project with no caches (interpreter
start and input validation included). The label-agreement rows, and Mode A's
host peak-memory row, are carried over from the v1.0.0 measurements.

### Mode A — individual parcellation under a pre-trained prior

Single subject × 6 sessions, fsaverage6 (74,947 cortical vertices),
K = 300 parcels, gMSHBM, β = 5, w = 50, c = 10:

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 0 — gradient embedding | 233.6 s | 25.0 s | 2.3 s |
| step 1 — profiles + init | 47.2 s | 13.2 s | 1.0 s |
| step 3 — parcellation EM | 763.1 s | 26.1 s | **0.6 s** |
| **total (end-to-end wall)** | **1073.0 s** | **65.9 s (16×)** | **6.4 s (168×)** |
| per-vertex agreement vs MATLAB labels¹ | — | 98.33 % | 98.05 % |
| peak memory (step 3, whole process) | ~11 GB RAM | — | **~4 GB (2.7× less)** |

GPU column: `backend_step0` / `backend_step1` = `gpu`,
`backend_step3` = `gpu_sparse`. The MATLAB path additionally required
a one-off 1,527 s GIFTI→NIfTI conversion pass; cuArealMSHBM ingests
`.func.gii` natively.

Against v1.0.0 on the same machine and config, GPU step 0–3 compute
drops **15.8 s → 3.9 s (4.1×)**: the step-0 Δ-stepping SSSP rewrite,
the fused whole-subject step-1 GPU leaf, and the step-3 `gpu_sparse`
backend. The 6.4 s wall adds ~1.0 s of BOLD-manifest validation and
~0.7 s of interpreter start — fixed costs that do not grow with the
cohort. A first run in a cold process pays another ~2 s of numba JIT
compilation (8.0 s wall); repeat runs land at 6.3–6.4 s.

¹ Cortical vertices, medial wall excluded. The two Python backends
agree with each other at 98.21 %. Bit-exact MATLAB↔Python equality is
not attainable in the first place: the residual disagreement is the
combined effect of several benign sources — BLAS reduction-order
differences between numerical libraries (every backend has its own
numerical noise band, and the GPU additionally shows small run-to-run
variation), different numerical routes for the Bessel-type functions
(ours is audited against mpmath in
`arealmshbm/spatial_priors/_cdln.py`), and a few small, deliberate
implementation choices where the port diverges from the reference.
The four-layer framework we use to judge whether differences of this
kind are functionally meaningful is described in
`docs/precision_impact_analysis.md`.

### Mode B — group-prior training on a local cohort

40 subjects × 6 sessions, fsaverage6, K = 300, gMSHBM:

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 2 — group-prior EM (production `max_iter_inter=10`; 48 EM iterations on both backends) | not feasible on a workstation² | 2,638.7 s | **16.5 s (160×)** |
| full pipeline steps 0–3, whole cohort (production run) | not feasible on a workstation² | — | **77.0 s (≈1.3 min)** |
| device memory, step 2, whole 40-subject cohort | **≈ 370–490 GB RAM** (estimated²) | — | **peak 6.7 GiB of 24 GB VRAM** |

The step-2 GPU column is the `gpu` backend — in v2.0.0 that name is the
candidate-set session; the dense CuPy port that held it in v1.0.0 has been
removed. Measured as a single cold run in a fresh process (a warm repeat is
14.9 s; inside a full Mode B driver run step 2 takes 15.1 s).

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
`uint8 (N, T, ⌈D/8⌉)` tensor (72 MB per subject, 1.45 GB for the
entire cohort — vs ~11.6 GB unpacked, and vs the reference's ~4.6 GB
of fp64 per subject *before* its extra copies), shared by reference
across all kernels. The posterior is stored fp32 instead of fp64
(3.93 GB for all 40 subjects; fp64 scratch only where the softmax
demands it), and kernel fusion means the N × L × T intermediates the
reference round-trips through RAM and disk are never materialized at
all. Measured peak device usage on the `gpu` backend: **6.7 GiB** — **the
whole 40-subject cohort fits in less memory than the reference needs for
a single subject**, with most of a 24 GB card left over. How far the
cohort can grow before the (S, N, ·) terms dominate has not been
measured.

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
tar -xzf avg_mesh-fsaverage6.tar.gz -C arealmshbm/data/precomputed/
```

`avg_mesh` (4.3 MB) is **required** — mesh geometry is read from these
bundles and never rebuilt at runtime.
`step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz` (82.6 MB) is a step-0
cache; skip it if you have a CBIG checkout (set `$CBIG_CODE_DIR` and the
first run builds it in ~3 s, then reuses it). Any location works if you
point `MSHBM_PRECOMPUTED_ROOT` at the directory holding `avg_mesh/`.

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
`Ind_parcellation_MSHBM_sub<S>_w<W>_MRF<C>_beta<B>.mat`.

### Operating modes

| mode | step path | group prior |
|---|---|---|
| `modeA_single` (1 subject) | 0 → 1 → 3 | pre-staged in the project |
| `modeA_batch` (N subjects) | 0 → 1 → 3 | pre-staged in the project |
| `modeB_train_prior` (cohort) | 0 → 1 → 2 → 3 | trained on your cohort by step 2 |

Mode A reads `<project>/priors/<variant>/beta<B>/Params_Final.mat`;
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
| `backend_step1` | `cpu`, `gpu` | bit-identical artifacts either way |
| `backend_step2` | `cpu`, `gpu` | `gpu` is the candidate-set session: it needs `seed_mesh: "fsaverage3"`, K ≤ 512 and ≤ 11,772 gradient components — anything larger runs on `cpu` |
| `backend_step3` | `cpu`, `gpu_elambda`, `gpu_full`, `gpu_sparse` | `gpu_sparse` needs `seed_mesh: "fsaverage3"` and covers gMSHBM / dMSHBM only |

Design notes and the measured backend-vs-backend agreement are in
[`docs/step2_sparse_design.md`](docs/step2_sparse_design.md) and
[`docs/step3_sparse_design.md`](docs/step3_sparse_design.md).

**Upgrading a v1.0.0 project config.** Two `step2` keys are now rejected
at parse time: `backend_step2: "gpu_sparse"` (write `"gpu"` — there is no
alias) and `enable_tf32` (delete it; it only ever scoped the dense port's
sgemms, and `step0.enable_tf32` is unaffected). A config that already said
`backend_step2: "gpu"` still parses, but now trains its prior on the
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

- Python 3.13; numba 0.63.1, numpy 2.2.6, scipy 1.16.0
- CuPy 13.6.0 (optional — GPU backends only); optional
  `nvidia-nvcomp-cu12` for GPU GIFTI decode
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

**CPU/GPU 加速的 Areal-MSHBM** —— 基于静息态 fMRI 的个体化皮层分区，以 numba（CPU）与 CuPy（GPU）双后端完成端到端的 Python 移植。已在大规模数据（300+ 名被试、1,800+ 个 fMRI session）上与原版 MATLAB 实现完成对照校验。**v2.0.0** 为 step 2 / step 3 新增了候选集稀疏 GPU 后端，并重写了 step 0 / step 1 的 GPU 路径：单被试从原始 BOLD 到分区结果，在一张笔记本 GPU 上端到端只需 **6.4 秒**。

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
- **候选集 GPU 后端**（step 2 的 `gpu`、step 3 的 `gpu_sparse`）—— EM 只在组先验支撑集所隐含的候选集上运行，而不是稠密的 (N × L) 后验，因此整个队列可以在整轮运行期间常驻显存。

工程亮点：

- **更快、更精确的贝塞尔类函数数值计算。** vMF 归一化常数与浓度参数更新均改用 `log I_ν` 的闭式解与渐近展开，而非参考实现的通用贝塞尔路径。经 mpmath 50 位精度审计：d=3 的 vMF 对数归一化常数精确到 0.31 × fp32 eps —— 在 κ = 1000 处比 MATLAB 原版**精确至多 19 倍**，且无分支、对 JIT/GPU 友好。
- **位打包的显存压缩。** 二值化 BOLD profile 在设备上以位打包的 `uint8 (N, T, ⌈D/8⌉)` 张量存放 —— 8 倍压缩，使 40 被试 × 6 session 队列的 BOLD 仅占 1.45 GB 显存（未打包约 11.6 GB），并由基于 popcount 的内核直接消费打包形式。
- **瓶颈步骤上的算子融合与算子调优**：融合的 profile 生成内核、带逐被试 fp64 暂存的融合 E-step softmax 链（fp32 存储，精度关键处fp64）、调优的归约，以及将 I/O 隐藏在 GPU 计算之后的流水线化 BOLD 解码。

结果：在最重的步骤上，相对 MATLAB 参考实现的加速在单张 Blackwell GPU 上达到**三个数量级**（step-3 分区 EM：763.1 s → 0.6 s，约 1,300 倍；step-0 梯度嵌入：233.6 s → 2.3 s，约 100 倍），单被试全流程端到端提速**约 170 倍**。

## 项目状态

**这是第二个发布版本（v2.0.0）**，包含 CPU 与 GPU 后端，step 2 与 step 3 另有候选集 GPU 路径。在迄今超过 **300 名被试、1,800 个 fMRI session** 的大规模校验中，它与原版实现的输出在数值所允许的极限水平上保持一致（逐顶点一致率约 98%；跨 BLAS/归约顺序边界的位级一致在数学上不可达），同时带来显著的速度与内存节约（见下方表格）。

项目仍在持续完善中。**所有预计算数据**（网格资产包、step-0 缓存、预训练 HCP 组先验、空间掩膜）**将由带 GUI 的发行版（开发中）的安装程序部署。** 本源码版本刻意不包含这些资产 —— 安装程序部署的目录布局见 [`arealmshbm/data/README.md`](arealmshbm/data/README.md)。

## 基准测试

所有数字均在同一台 Windows 11 笔记本上测得：RTX 5090 Laptop GPU（Blackwell，24 GB）、Python 3.13、numba 0.63.1、CuPy 13.6.0。MATLAB基线为 CBIG 参考流水线**并启用其编译 MEX 热路径**（`mtimesx`），在同一台机器上运行 —— 比典型集群节点更强的基线。下表中 cuArealMSHBM 的每一个**耗时**数字都在 v2.0.0 代码上重新测得，每一项"总计"均为无缓存项目上的端到端驱动器墙钟时间（含解释器启动与输入校验）；标签一致率各行、以及 Mode A 的主机内存峰值一行，沿用 v1.0.0 的测量结果。

### Mode A —— 基于预训练先验的个体化分区

单被试 × 6 session，fsaverage6（74,947 个皮层顶点），K = 300 分区，
gMSHBM，β = 5，w = 50，c = 10：

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 0 —— 梯度嵌入 | 233.6 s | 25.0 s | 2.3 s |
| step 1 —— profile + 初始化 | 47.2 s | 13.2 s | 1.0 s |
| step 3 —— 分区 EM | 763.1 s | 26.1 s | **0.6 s** |
| **总计（端到端墙钟）** | **1073.0 s** | **65.9 s（16×）** | **6.4 s（168×）** |
| 对 MATLAB 标签的逐顶点一致率¹ | — | 98.33 % | 98.05 % |
| 内存峰值（step 3，全进程） | ~11 GB RAM | — | **~4 GB（省 2.7×）** |

GPU 一列所用后端：`backend_step0` / `backend_step1` = `gpu`，`backend_step3` = `gpu_sparse`。MATLAB 路径还额外需要一次 1,527 s 的 GIFTI→NIfTI 一次性格式转换；cuArealMSHBM 原生读取 `.func.gii`。

与 v1.0.0 相比（同一台机器、同一配置），GPU 上 step 0–3 的计算耗时从 **15.8 s 降至 3.9 s（4.1×）**，来自 step-0 的 Δ-stepping SSSP 重写、step-1 融合的整被试 GPU 叶子算子，以及 step-3 的 `gpu_sparse` 后端。6.4 s 的墙钟时间还额外包含约 1.0 s 的 BOLD 清单校验与约 0.7 s 的解释器启动 —— 这些是不随队列规模增长的固定开销。冷进程首次运行还要多付约 2 s 的 numba JIT 编译（墙钟 8.0 s），重复运行稳定在 6.3–6.4 s。

¹ 皮层顶点，剔除内侧壁。两个 Python 后端彼此的一致率为 98.21%。MATLAB↔Python 的位级一致本就不可达：残余分歧是多个良性来源叠加的结果——不同数值库之间的 BLAS 归约顺序差异（每个后端各有自己的数值噪声带，GPU 还存在小幅的逐次运行波动）、贝塞尔类函数所采用的不同数值路径（我们的实现经 mpmath 审计，见 `arealmshbm/spatial_priors/_cdln.py`），以及移植中少数几处与参考实现有意为之的细小实现差异。我们用于判断此类差异是否具有功能意义的四层评估框架见 `docs/precision_impact_analysis.md`。

### Mode B —— 在本地队列上训练组先验

40 被试 × 6 session，fsaverage6，K = 300，gMSHBM：

| | MATLAB (CBIG, MEX) | cuArealMSHBM CPU | cuArealMSHBM GPU |
|---|---|---|---|
| step 2 —— 组先验 EM（生产配置 `max_iter_inter=10`；两个后端均为 48 次 EM 迭代） | 工作站上不可实施² | 2,638.7 s | **16.5 s（160×）** |
| 全流程 step 0–3，整个队列（生产运行） | 工作站上不可实施² | — | **77.0 s（≈1.3 分钟）** |
| step 2 在 40 被试全队列下的设备内存 | **≈ 370–490 GB RAM**（预估²） | — | **峰值 6.7 GiB / 24 GB 显存** |

step 2 的 GPU 一列取自 `gpu` 后端 —— 在 v2.0.0 中这个名字指的就是候选集 session，v1.0.0 中占用该名字的稠密 CuPy 移植已被移除。该数字为新进程中的单次冷启动运行（重复运行为 14.9 s；在完整 Mode B 驱动器运行中 step 2 为 15.1 s）。

**与 v1.0.0 的同机对比。** v1.0.0 公布的 step 2 数字是 794.0 s（CPU）/ 156.7 s（稠密 GPU），但那是在 `max_iter_inter=2` —— 短跑基准的迭代上限 —— 下测得的，并非生产默认值 10，见 [`docs/step2_flow_and_subgraphs.md`](docs/step2_flow_and_subgraphs.md) 的 § Production-cohort walls。在 v2.0.0 代码上重跑同样的形状，`gpu` 后端为 **11.1 s**：比 v1.0.0 的 GPU 数字快 **14×**，比其 CPU 数字快 **72×**。其实测显存峰值在 `max_iter_inter=2` 与 `=10` 下**同为 6.7 GiB** —— 不随外层迭代数增长，因为候选集 session 只分配一次并全程复用。v1.0.0 的稠密 GPU 后端在同一形状下实测为 **21 GiB**，在 `inter=10` 时已溢出到主机内存；v2.0.0 将其移除，除了显存原因，也因为它的 fp32 清零（flush-to-zero）E-step 是两者中精度较低的一个。

² 由参考实现源码（`CBIG_ArealMSHBM_gMSHBM_estimate_group_priors_{parent,child}.m`）推算，而非实测 —— 因为它在工作站上*无法*实测。参考实现的 step 2被架构为 **1 个 parent + S 个 child 集群作业**（其源码头注释原文：*"user should submit 1 parent job and num_sub child jobs"*），在**每个 M-step 内层迭代**都通过文件系统交换 `.mat` 文件。child 会阻塞在迭代中途等待 parent 的回复，因此被试无法串行处理：40 个 child 必须**同时**常驻。在本基准形状下（N = 81,924 顶点、D = 1,175 profile 维、T = 6 session、L = 300 分区、全 fp64），每个 child 持有 profile 主存储`data_series`（N × D × T，4.6 GB），外加代码以*不同内存布局*物化的多份全尺寸拷贝 —— 每 session 的 N × D 切片副本、`mtimesx` 产出的N × L × T κ 更新乘积、转置的 L × N × T log-vMF 堆栈、(N, L, T) 空间先验堆栈 —— 工作集达**每被试约 9–12 GB**；parent 另持有 (N, L, S) 后验（S = 40 时 7.9 GB）及同尺寸临时量。合计：41 个进程约 370–490 GB 的同时常驻内存，外加每迭代经共享存储的 `.mat` 流量 —— 计算集群是硬性前提，而非可选优化。

cuArealMSHBM 将这一切收拢到单个设备上：每份二值化 profile 在显存中**只存在一次**，以位打包的 `uint8 (N, T, ⌈D/8⌉)` 张量存放（每被试72 MB，整个队列 1.45 GB —— 对比未打包的约 11.6 GB，以及参考实现*尚未计入额外拷贝*的每被试约 4.6 GB fp64），并在所有内核间按引用共享。后验以 fp32 而非 fp64 存储（40 被试共 3.93 GB；仅在 softmax 需要处使用 fp64 暂存），算子融合则使参考实现在内存与磁盘间往返的 N × L × T中间量根本不被物化。`gpu` 后端上实测的设备峰值占用为 **6.7 GiB** —— **整个 40 被试队列所需的显存比参考实现单个被试所需的内存还少**，且 24 GB 显卡还剩下大半。队列还能扩到多大才会被 (S, N, ·) 项主导，尚未实测。

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
tar -xzf avg_mesh-fsaverage6.tar.gz -C arealmshbm/data/precomputed/
```

`avg_mesh`（4.3 MB）是**必需的** —— 网格几何全部从这些 bundle 读取，运行时不会重建。`step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz`（82.6 MB）是 step 0 的缓存；如果你手边有 CBIG checkout 可以不下（设好 `$CBIG_CODE_DIR`，首次运行会用约 3 秒自己建好并落盘复用）。放在其它位置也可以，把 `MSHBM_PRECOMPUTED_ROOT` 指向含 `avg_mesh/` 的那个目录即可。

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
`projects/<name>/ind_parcellation_<variant>/<N>_sess/beta<B>/`，文件名为`Ind_parcellation_MSHBM_sub<S>_w<W>_MRF<C>_beta<B>.mat`。

### 运行模式

| 模式 | 步骤路径 | 组先验 |
|---|---|---|
| `modeA_single`（1 被试） | 0 → 1 → 3 | 预置于项目内 |
| `modeA_batch`（N 被试） | 0 → 1 → 3 | 预置于项目内 |
| `modeB_train_prior`（队列） | 0 → 1 → 2 → 3 | 由 step 2 在你的队列上训练 |

Mode A 读取 `<project>/priors/<variant>/beta<B>/Params_Final.mat`；将先验放到该位置是项目创建者的职责。变体：gMSHBM 与 dMSHBM 已提供；cMSHBM 的 step 2 未接入（其 step 3 已接入）。参见[`docs/pipeline_modes.md`](docs/pipeline_modes.md) 与[`docs/pipeline_variants.md`](docs/pipeline_variants.md)。

### 如何选择后端

每个步骤在 `pipeline_config.json` 中独立选择后端（`backend_step0` … `backend_step3`）。`cpu` 始终可用；GPU 后端的前置条件如下表：

| 配置项 | 可选值 | 说明 |
|---|---|---|
| `backend_step0` | `cpu`、`gpu` | GPU 版 step 0 不保证逐次运行的位级可复现，实测波动范围见 [`docs/step0_flow_and_subgraphs.md`](docs/step0_flow_and_subgraphs.md) |
| `backend_step1` | `cpu`、`gpu` | 两者产物位级完全一致 |
| `backend_step2` | `cpu`、`gpu` | `gpu` 即候选集 session：要求 `seed_mesh: "fsaverage3"`、K ≤ 512、梯度分量数 ≤ 11,772；超出范围的配置走 `cpu` |
| `backend_step3` | `cpu`、`gpu_elambda`、`gpu_full`、`gpu_sparse` | `gpu_sparse` 要求 `seed_mesh: "fsaverage3"`，且仅覆盖 gMSHBM 与 dMSHBM |

设计说明与后端之间实测的一致性数据见 [`docs/step2_sparse_design.md`](docs/step2_sparse_design.md) 与 [`docs/step3_sparse_design.md`](docs/step3_sparse_design.md)。

**从 v1.0.0 升级项目配置。** `step2` 块中有两个键现在会在解析阶段被拒绝：`backend_step2: "gpu_sparse"`（请改写为 `"gpu"` —— 没有保留别名）与 `enable_tf32`（直接删除；它从来只作用于稠密移植的 sgemm，`step0.enable_tf32` 不受影响）。原本就写 `backend_step2: "gpu"` 的配置仍可解析，但其组先验现在由候选集 session 训练 —— 后者与 `cpu` 参考实现一致，而稠密移植并不一致。

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

- Python 3.13；numba 0.63.1、numpy 2.2.6、scipy 1.16.0
- CuPy 13.6.0（可选 —— 仅 GPU 后端需要）；可选 `nvidia-nvcomp-cu12` 用于 GPU GIFTI 解码
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
