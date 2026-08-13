# Pipeline variants — gMSHBM / cMSHBM / dMSHBM

The Python step-3 super-call covers all three Areal-MSHBM variants
through the single ``Step3Config.pipeline_type`` knob. The three
variants share ~95 % of the EM body; this file catalogues the deltas
and the structural impact on the Python implementation.

Step 2 (group-prior estimation) supports `gMSHBM` and `dMSHBM` only;
`cMSHBM` step-3 runs against a borrowed gMSHBM prior (see §4).

## 1. The three variants — one-liner

| Variant | One-liner |
|---|---|
| **dMSHBM** (distributed) | No spatial prior, no connectedness check. The "vanilla" areal-MSHBM. |
| **cMSHBM** (contiguous) | XYZ-vMF spatial prior on the sphere; forces tight, single-component parcels. |
| **gMSHBM** (gradient-based) | Gradient-embedding centroid prior + XYZ-on-distributed fallback. |

## 2. Variant deltas (canonical table)

Anything not listed here is identical across variants.

|  | **dMSHBM** | **cMSHBM** | **gMSHBM** |
|---|---|---|---|
| **Group prior file** | `priors/dMSHBM/Params_Final.mat` | `priors/cMSHBM/beta<B>/Params_Final.mat` | `priors/gMSHBM/beta<B>/Params_Final.mat` |
| **Beta knob exists** | ❌ no `beta` arg | ✅ raw `beta` (no ×1000) | ✅ `beta * 1000` |
| **Reads cohort.subjects[].gradient_*** | ❌ | ❌ | ✅ |
| **`spatial_connect_prior`** (gradient term) | ❌ | ❌ | ✅ once per EM iter |
| **`spatial_xyz_prior`** (sphere term) | ❌ | ✅ once per EM iter (always on) | ✅ inside comp_iter, only when `iter_em > 1` |
| **`check_connectedness`** | ❌ | ✅ uses `RemoveIsolatedSurfaceComponents` then components > **1** \| dist > 0 | ✅ raw labels, components > **3** \| dist > 15 |
| **`connect_th`** | n/a | 0 | 15 |
| **`comp_iter` loop wrap** | ❌ (single λ-loop pass) | ✅ (full wrap) | ✅ (full wrap; first EM iter skips) |
| **`E-step log_vmf` extra terms** | (none beyond data + theta + V) | `+ beta · spatial_xyz_vmf` | `+ beta · spatial_connect_vmf + spatial_xyz_vmf` |
| **`em_stop` cost extra term** | none | none | `+ Σ beta · s_lambda · spatial_connect_vmf` |
| **Final post-processing** | argmax | argmax → `RemoveIsolatedSurfaceComponents(5)` | argmax (no cleanup) |
| **Output dir** | `ind_parcellation_dMSHBM/<T>_sess/` | `ind_parcellation_cMSHBM/<T>_sess/beta<B>/` | `ind_parcellation_gMSHBM/<T>_sess/beta<B>/` |
| **Output filename** | `Ind_parcellation_MSHBM_sub<i>_w<w>_MRF<c>.mat` | `..._w<w>_MRF<c>_beta<B>.mat` | `..._w<w>_MRF<c>_beta<B>.mat` |

Subtle gMSHBM-only details that survive the table but are worth
highlighting:

- gMSHBM's `spatial_connect_vmf` has hemisphere blocking baked in
  (`vmf[:lh, rh_parcels] = -Inf` and vice versa); the L-axis is split as
  `[lh_parcels; rh_parcels]` halves. The Python `ConnectSession`
  encodes this. cMSHBM has no equivalent — `spatial_xyz_vmf` is full
  bilateral; cross-hemi cells of `s_muc` carry finite cosines but are
  gated to zero contribution by upstream `boundary_mask` pinning the
  softmax there.
- gMSHBM's `check_connectedness` clamps `max_components` to 3 when the
  parcellation is fully OK; cMSHBM clamps to 1.

## 3. Python architecture

### 3.1 Leaf nodes

| Leaf | Owner module | Used by |
|---|---|---|
| `remove_isolated_surface_components` | [`arealmshbm/postprocessing/remove_isolated.py`](../arealmshbm/postprocessing/remove_isolated.py) | cMSHBM only — twice (per comp_iter and at the end) |
| `compute_components_general` threshold | [`arealmshbm/check_connectedness/`](../arealmshbm/check_connectedness/) | gMSHBM (`>3`), cMSHBM (`>1`); the leaf returns the raw count, the calling glue applies the predicate |

`Cdln`, `vmf_probability`, the M-step κ/ν update, `intra_em_cost`,
`intra_subject_var`, `boundary_mask` build, `pipeline_setup`,
`initialize_params`, `initialize_concentration`, `load_group_prior`,
`load_spatial_mask`, `load_avg_mesh`, `derive_labels` are
variant-invariant.

### 3.2 Variant dispatch

A single immutable [`VariantSpec`](../arealmshbm/step3_pipeline/variant.py)
captures every variant-sensitive decision (which spatial priors fire,
which iteration first runs `check_connectedness`, what
`components_threshold` to use, whether the EM-stop cost includes the
β·scv term, what output paths look like, etc.). Three subgraphs read
from it:

1. **`Step3Config` / `Step3Pipeline.load_inputs`** — picks
   `group_prior_path` per variant, branches `out_dir` and filename
   templates, sets `beta_internal` (gMSHBM ×1000, cMSHBM ×1, dMSHBM 0),
   skips cohort.json's gradient fields when not gMSHBM (`fetch_data`
   accepts a `with_gradient: bool` flag).
2. **`VmfClusteringSession.run`** — reads the spec at construction and
   gates which sub-Sessions get built and which calls fire per iter.
   Both the CPU path and `gpu_full` mirror the same gating from the
   same spec; no per-variant Session subclass.
3. **`save_parcellation` / `derive_labels`** — applies the cMSHBM final
   `RemoveIsolatedSurfaceComponents(5)` cleanup; picks
   `..._beta<B>` vs `...` filename based on whether `beta` is set.

The "no spatial prior" cases (dMSHBM both, cMSHBM connect) are
implemented by feeding the existing fused E-step kernel zero `(N, L)`
buffers in the corresponding slots — no new "no-prior" kernel is
needed since `+ β·scv + sxv` with `β = 0` or zero buffers is already
algebraically correct.

## 4. Group-prior fallback (cMSHBM / dMSHBM)

Only gMSHBM priors are shipped — the local
`arealmshbm/data/group_priors/HCP_fsaverage6_40sub/<L>/gMSHBM/beta5/`
layout has only `gMSHBM/beta<B>` subdirs; cMSHBM and dMSHBM
directories don't exist. Step 2 trains gMSHBM / dMSHBM in-house
(cMSHBM is not wired for step 2).

When running cMSHBM step-3 in isolation against the shipped HCP
material, point at the gMSHBM `Params_Final.mat`. The four prior
fields (`mu`, `theta`, `epsil`, `sigma`) have identical meaning across
all three variants — they're vMF group parameters and a spatial-prior
probability map; the step-3 driver reads them positionally with no
variant-specific schema check. Feeding cMSHBM's EM body the gMSHBM
Params produces a valid algorithmic run — not a scientifically-
meaningful cMSHBM parcellation, but the EM converges.

The plumbing: `Step3Config.group_prior_path_override` accepts an
explicit path. Once a properly-trained cMSHBM/dMSHBM prior is available,
just point the override at it; no Python changes needed.
