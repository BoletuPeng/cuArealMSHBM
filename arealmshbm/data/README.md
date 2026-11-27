# arealmshbm/data — precomputed assets

This directory ships **empty** in the source repository: subject-invariant
data is kept out of git so clones stay small. A missing asset — a mesh
bundle (`avg_mesh`) or the step-0 inputs bundle (`step0_inputs`) — is
reported as a hard `FileNotFoundError` telling you to (re)install; neither
is rebuilt at runtime.

## Where to get the assets

**Release assets** — published under the
[`assets-v2`](https://github.com/BoletuPeng/cuArealMSHBM/releases/tag/assets-v2) tag, versioned independently of the
code releases (the bundles carry their own format version and are only
re-cut when the layout changes). Verify downloads against the release's
`SHA256SUMS.txt`.

| asset | size | needed by |
|---|---:|---|
| `avg_mesh.tar.gz` | 4.5 MB | **everyone** |
| `step0_inputs.tar.gz` | 82.7 MB | **everyone** |

```bash
curl -L -O https://github.com/BoletuPeng/cuArealMSHBM/releases/download/assets-v2/avg_mesh.tar.gz
curl -L -O https://github.com/BoletuPeng/cuArealMSHBM/releases/download/assets-v2/step0_inputs.tar.gz
tar -xzf avg_mesh.tar.gz -C arealmshbm/data/precomputed/
tar -xzf step0_inputs.tar.gz -C arealmshbm/data/precomputed/
```

[`assets-v1`](https://github.com/BoletuPeng/cuArealMSHBM/releases/tag/assets-v1) holds the same data under the file
names v2.2.0 and earlier read (`avg_mesh/<hemi>_fsaverage6_<surface>.npz`,
`step0_inputs/fsaverage6_sigma2.55_khop3/`); use it with those releases.

**From a CBIG checkout** — the group priors and spatial masks below are
byte-identical to CBIG's, so there is no second copy in this project.
Take them from
`stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/`.

## Layout once staged

```
arealmshbm/data/
  precomputed/                             # gitignored even when staged
    avg_mesh/<hemi>_<surface>.npz          # fsaverage6 mesh bundles
    step0_inputs/                          # step-0 inputs bundle
      manifest.json
      *.npy
  group_priors/                            # optional, from CBIG
    HCP_40sub/<K>/<variant>/beta<B>/Params_Final.mat
```

- `precomputed/avg_mesh/` is read by `load_avg_mesh`: `<hemi>` is `lh` /
  `rh`, `<surface>` is `inflated` / `sphere`. These bundles are the only
  source of mesh geometry; there is **no** runtime rebuild from FreeSurfer
  sources.
- `precomputed/step0_inputs/` is read by `Step0Pipeline.load_inputs`
  under the same contract: the sphere and midthickness meshes, the medial
  mask, the smoothing and gradient mesh tables, the Gaussian smoothing
  gather and the cortex neighbour table with its 3-hop expansion. The
  2.55 mm smoothing sigma and the 3-hop neighbourhood are baked into it;
  neither is a config key. A bundle whose `manifest.json` carries another
  format version is refused.
- `$MSHBM_PRECOMPUTED_ROOT` relocates `precomputed/` as a whole: point it
  at a directory that holds `avg_mesh/` and `step0_inputs/`.
- **Group priors** are model *inputs*, not required by the package: Mode A
  reads the prior only from `<project>/priors/<variant>/beta<B>/Params_Final.mat`
  (no `beta<B>/` segment for dMSHBM) and staging it there is the project
  creator's job; Mode B trains one from your own cohort. Copy one from
  CBIG's `lib/group_priors/HCP_fsaverage6_40sub/`, or from another
  project's Mode B output. The optional `group_priors/HCP_40sub/` slot
  above is read by one test only
  (`generate_profiles/tests/test_seed_vertices.py`, which checks the seed
  count against the prior's profile dimension at
  `HCP_40sub/100/gMSHBM/beta5/` and skips that half without it).
- **Spatial masks** are likewise not required: step 1 generates the mask
  your project uses into `<project>/spatial_mask/spatial_mask.mat`.
  CBIG's `lib/spatial_mask/` copies are useful as a cross-check.
- `MSHBM_ATLAS_DIR` is a separate, always-required runtime input (the live
  `fsaverage6/label/*.annot` reads in step 1) and is not part of any of
  the above.

## Terms

`avg_mesh` bundles are baked from FreeSurfer fsaverage6 surface geometry
and `cortex.label`; `step0_inputs` is derived from the same meshes plus
CBIG's midthickness atlas. These staged assets are distributed under the
terms of their upstream sources (e.g. the HCP data-use terms for the
HCP-derived group priors); this repository's MIT license covers code only.
