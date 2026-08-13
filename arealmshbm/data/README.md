# arealmshbm/data — precomputed assets (staged by the installer)

This directory ships **empty** in the source repository. All
subject-invariant precomputed data is deliberately kept out of git and
will be deployed by the installer of the GUI distribution (in
development). A missing asset is reported as a hard
`FileNotFoundError` telling you to (re)install — it is never silently
rebuilt at runtime.

Expected layout once staged:

```
arealmshbm/data/
  group_priors/                  # pre-trained HCP group priors (Mode A source to copy from)
    HCP_fsaverage6_40sub/<K>/<variant>/beta<B>/Params_Final.mat
    HCP_fs_LR_32k_40sub/<K>/<variant>/beta<B>/Params_Final.mat
  spatial_mask/                  # per-K spatial radius masks
    <K>/spatial_mask_fsaverage6.mat
    <K>/spatial_mask_fs_LR_32k.mat
  precomputed/                   # gitignored even when staged
    avg_mesh/<hemi>_<mesh>_<surface>.npz   # fsaverage-family mesh bundles
    step0_inputs/                          # step-0 gradient cache (~158 MB)
```

- `precomputed/avg_mesh/` bundles are read by `load_avg_mesh`
  (override root via `$MSHBM_PRECOMPUTED_ROOT`). There is no runtime
  rebuild from FreeSurfer sources.
- `precomputed/step0_inputs/` can be rebuilt offline by developers via
  `python -m arealmshbm.precompute.step0_inputs_builder` (requires a
  CBIG checkout; runtime never touches CBIG).
- Group priors and spatial masks are model *inputs*; Mode A copies a
  prior into `<project>/priors/<variant>/beta<B>/Params_Final.mat`,
  Mode B trains one from your own cohort.
- These staged assets are distributed under their own terms (e.g. the
  HCP data-use terms for the HCP-derived group priors); the
  repository's MIT license covers code only.
