# arealmshbm/data — precomputed assets

This directory ships **empty** in the source repository: subject-invariant
data is kept out of git so clones stay small. A missing asset is reported
as a hard `FileNotFoundError` telling you to (re)install — it is never
silently rebuilt at runtime.

## Where to get the assets

**Release assets** — published under the
[`assets-v1`](https://github.com/BoletuPeng/cuArealMSHBM/releases/tag/assets-v1) tag, versioned independently of the
code releases (the step-0 cache is keyed on `(mesh, smooth_sigma, K_hop)`
and carries its own format version, none of which move when the pipeline
changes). Verify downloads against the release's `SHA256SUMS.txt`.

| asset | size | needed by |
|---|---:|---|
| `avg_mesh-fsaverage6.tar.gz` | 4.3 MB | **everyone** |
| `step0_inputs-fsaverage6_sigma2.55_khop3.tar.gz` | 82.6 MB | anyone without a CBIG checkout |

```bash
curl -L -O https://github.com/BoletuPeng/cuArealMSHBM/releases/download/assets-v1/avg_mesh-fsaverage6.tar.gz
tar -xzf avg_mesh-fsaverage6.tar.gz -C arealmshbm/data/precomputed/
```

**From a CBIG checkout** — the group priors and spatial masks below are
byte-identical to CBIG's, so there is no second copy in this project.
Take them from
`stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/`.

## Layout once staged

```
arealmshbm/data/
  precomputed/                             # gitignored even when staged
    avg_mesh/<hemi>_<mesh>_<surface>.npz   # fsaverage-family mesh bundles
    step0_inputs/<mesh>_sigma<S>_khop<K>/  # step-0 gradient cache
  group_priors/                            # optional, from CBIG
    HCP_fsaverage6_40sub/<K>/<variant>/beta<B>/Params_Final.mat
    HCP_fs_LR_32k_40sub/<K>/<variant>/beta<B>/Params_Final.mat
  spatial_mask/                            # optional, from CBIG
    <K>/spatial_mask_<mesh>.mat
```

- `precomputed/avg_mesh/` is read by `load_avg_mesh` (override the root
  via `$MSHBM_PRECOMPUTED_ROOT`). There is **no** runtime rebuild from
  FreeSurfer sources — these bundles are the only source of mesh
  geometry.
- `precomputed/step0_inputs/` is a cache. `Step0Pipeline.load_inputs`
  rebuilds it on a miss and writes it back, but the rebuild
  (`python -m arealmshbm.precompute.step0_inputs_builder`) needs a CBIG
  checkout for the midthickness atlas and the raw FreeSurfer sphere
  surfaces. Download it instead if you do not have one.
- **Group priors** are model *inputs*, not required by the package: Mode A
  reads the prior only from `<project>/priors/<variant>/beta<B>/Params_Final.mat`
  and staging it there is the project creator's job; Mode B trains one
  from your own cohort. Copy one from CBIG's `lib/group_priors/`, or from
  another project's Mode B output.
- **Spatial masks** are likewise not required: step 1 generates the mask
  your project uses into `<project>/spatial_mask/spatial_mask_<mesh>.mat`.
  CBIG's `lib/spatial_mask/` copies are useful as a cross-check.
- `MSHBM_ATLAS_DIR` is a separate, always-required runtime input (the live
  `label/*.annot` reads in step 1) and is not part of any of the above.

## Terms

`avg_mesh` bundles are baked from FreeSurfer fsaverage6 surface geometry
and `cortex.label`; `step0_inputs` is derived from the same meshes plus
CBIG's midthickness atlas. These staged assets are distributed under the
terms of their upstream sources (e.g. the HCP data-use terms for the
HCP-derived group priors); this repository's MIT license covers code only.
