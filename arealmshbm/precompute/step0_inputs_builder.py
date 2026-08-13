"""arealmshbm/precompute/step0_inputs_builder.py

Build the subject-invariant step-0 inputs cache to disk.

This is the **original heavy precomputation** for ``Step0Inputs``,
moved out of the production path (where ``Step0Pipeline.load_inputs``
now reads the cached output instead). The leaves it runs:

  * raw FreeSurfer sphere read: ``read_geometry`` for vertices + faces,
    plus ``read_label`` on ``<mesh>/label/<hemi>.cortex.label`` to
    synthesize MARS_label, from
    ``<cbig>/data/templates/surface/<mesh>/{surf,label}/``. ~619 ms.
  * ``compute_topology`` — sphere vertex_nbors + vertex_faces. ~10 ms.
  * ``read_surface_mesh`` — midthickness .gii read. ~16 ms.
  * ``prepare_smoothing_mesh`` — Layer 1/2 geodesic CSR + vertex areas.
    ~241 ms.
  * ``prepare_gradient_mesh`` — 1-ring nbors + vertex normals + areas.
    ~19 ms.
  * ``prepare_smoothing_gather`` — per-source Dijkstra-weighted gather
    CSR (depends on mesh + cortex roi + smooth_sigma). ~187 ms.
  * ``neighbors_exclude_medial`` + ``find_K_neighbors`` (K=3). The
    K-hop BFS expansion is the single largest cost — **~1846 ms**,
    62 % of the cold-load total.

Everything here is determined entirely by (mesh, cbig_code_dir,
smooth_sigma, K_hop). Building once and caching to ``.npy`` files on
disk turns a ~3.0 s cold start into a ~30-100 ms ``.npy`` load on
every subsequent process.

Dev-only tool — needs a CBIG checkout for the midthickness atlas and
FreeSurfer source meshes. Runtime reads the precomputed .npy output,
not CBIG.

CLI usage::

    python -m arealmshbm.precompute.step0_inputs_builder \\
        --mesh fsaverage6 \\
        --smooth-sigma 2.55 \\
        --K-hop 3 \\
        --cbig-code-dir <path to a CBIG checkout>

Programmatic usage::

    from arealmshbm.precompute.step0_inputs_builder import build_step0_inputs
    inputs = build_step0_inputs(cfg)  # cfg: Step0Config

Cache directory layout is owned by
:mod:`arealmshbm.step0_pipeline.inputs_cache`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def build_step0_inputs(cfg) -> "Step0Inputs":
    """Run the full precomputation, returning a CPU-only ``Step0Inputs``.

    Equivalent to the original ``Step0Pipeline.load_inputs`` body —
    factored out so the cache builder CLI and the live-rebuild fallback
    share one source of truth. GPU mirrors are NOT built here (the
    production path wraps the cached output via ``_build_gpu_mirrors``
    when ``cfg.backend == 'gpu'``).
    """
    # Lazy imports so this module is importable without numba/scipy
    # warming up just to inspect signatures.
    from arealmshbm.surface_smoothing import (
        prepare_smoothing_mesh, prepare_smoothing_gather,
    )
    from arealmshbm.surface_gradient import prepare_gradient_mesh
    from arealmshbm.mesh_topology import compute_topology
    from arealmshbm.step0_neighbors import (
        find_K_neighbors, neighbors_exclude_medial,
    )
    from arealmshbm.surface_io import read_surface_mesh
    from arealmshbm.step0_pipeline.pipeline import Step0Inputs

    cbig_dir = cfg.cbig_code_dir or os.environ.get("CBIG_CODE_DIR")
    if not cbig_dir:
        raise RuntimeError(
            "step0_inputs_builder needs a CBIG checkout — pass "
            "Step0Config(cbig_code_dir=...) or set $CBIG_CODE_DIR."
        )
    cbig_dir = Path(cbig_dir)
    atlas_dir = (cbig_dir / "utilities" / "matlab" / "speedup_gradients" /
                  "utilities" / "fs6_surface_template")
    if not atlas_dir.exists():
        raise FileNotFoundError(f"midthickness atlas missing: {atlas_dir}")
    # The raw FreeSurfer sphere surfaces live inside CBIG at
    # data/templates/surface/<mesh>/surf/. NOTE: this only covers meshes
    # CBIG ships there — fsaverage{5,6}. fsaverage{3,4} live at
    # fake_freesurfer/subjects/<mesh>/ in CBIG. The builder only uses
    # cfg.mesh today (default fsaverage6), so this is sufficient; if a
    # future caller asks for a lower-res mesh, add a fallback.
    surf_root = cbig_dir / "data" / "templates" / "surface"

    # --- fsaverage6 sphere meshes (full resolution) ---
    # Dev-only raw read: the runtime loader (load_avg_mesh) is asset-only
    # and no longer parses raw FreeSurfer surfaces. This builder is the
    # build-side recipe, so it reads the sphere geometry + cortex.label
    # directly (synthesizing MARS_label) to bake the shipped step0 cache
    # from a CBIG checkout.
    import nibabel.freesurfer.io as fsio

    def _read_sphere(hemi: str):
        """Sphere geometry + cortex.label → MARS_label, the build-side
        equivalent of the retired ``load_avg_mesh(..., "sphere")`` read.
        Both files live under the CBIG flat layout
        ``<surf_root>/<mesh>/{surf,label}/``.
        """
        surf_path = surf_root / cfg.mesh / "surf" / f"{hemi}.sphere"
        if not surf_path.exists():
            raise FileNotFoundError(f"sphere surface missing: {surf_path}")
        v, f = fsio.read_geometry(str(surf_path))
        label_path = surf_root / cfg.mesh / "label" / f"{hemi}.cortex.label"
        if not label_path.exists():
            raise FileNotFoundError(f"cortex label missing: {label_path}")
        cortex = fsio.read_label(str(label_path))
        # MATLAB MARS_label convention: 1 = medial wall, 2 = cortex.
        mars = np.ones(v.shape[0], dtype=np.int64)
        mars[cortex.astype(np.int64)] = 2
        return (np.ascontiguousarray(v, dtype=np.float32),
                np.ascontiguousarray(f, dtype=np.int32),
                mars)

    lh_verts, lh_faces, lh_mars = _read_sphere("lh")
    rh_verts, rh_faces, rh_mars = _read_sphere("rh")
    n_lh, n_rh = lh_verts.shape[0], rh_verts.shape[0]
    n_full = n_lh + n_rh

    lh_vert_nbors, lh_vert_faces = compute_topology(lh_verts, lh_faces)
    rh_vert_nbors, rh_vert_faces = compute_topology(rh_verts, rh_faces)

    # --- midthickness meshes (for gradient + smoothing) ---
    lh_mid = read_surface_mesh(atlas_dir / "fsaverage6.L.midthickness.surf.gii")
    rh_mid = read_surface_mesh(atlas_dir / "fsaverage6.R.midthickness.surf.gii")

    # --- surface_smoothing per-hemi mesh prep ---
    lh_smooth_prep = prepare_smoothing_mesh(lh_mid["vertices"], lh_mid["faces"])
    rh_smooth_prep = prepare_smoothing_mesh(rh_mid["vertices"], rh_mid["faces"])

    # --- surface_gradient per-hemi mesh prep ---
    lh_grad_prep = prepare_gradient_mesh(lh_mid["vertices"], lh_mid["faces"])
    rh_grad_prep = prepare_gradient_mesh(rh_mid["vertices"], rh_mid["faces"])

    # --- medial mask (lh first, rh second) ---
    medial_mask = np.concatenate([
        (lh_mars.ravel() == 1),
        (rh_mars.ravel() == 1),
    ]).astype(bool)
    n_cortex = int((~medial_mask).sum())

    # --- surface_smoothing per-hemi gather (depends on sigma) ---
    cortex_bool = ~medial_mask
    lh_roi = cortex_bool[:n_lh]
    rh_roi = cortex_bool[n_lh:]
    lh_smooth_gather = prepare_smoothing_gather(
        lh_smooth_prep, lh_roi, cfg.smooth_sigma)
    rh_smooth_gather = prepare_smoothing_gather(
        rh_smooth_prep, rh_roi, cfg.smooth_sigma)

    # --- neighbors (port of CBIG neighbors_exclude_medial + find_K_neighbors) ---
    rh_nb_offset = rh_vert_nbors.copy()
    rh_mask = rh_nb_offset != 0
    rh_nb_offset[rh_mask] += n_lh
    all_nb = np.concatenate(
        [lh_vert_nbors, rh_nb_offset], axis=0).astype(np.float64)
    all_nb[all_nb == 0] = np.nan
    self_col = np.arange(1, n_full + 1, dtype=np.float64)[:, None]
    neighbors_raw = np.concatenate([self_col, all_nb], axis=1)

    neighbors_table = neighbors_exclude_medial(neighbors_raw, medial_mask)
    K_neighbors = find_K_neighbors(neighbors_table, cfg.K_hop)

    return Step0Inputs(
        medial_mask=medial_mask,
        n_lh=n_lh, n_rh=n_rh, n_full=n_full, n_cortex=n_cortex,
        lh_sphere_verts=lh_verts, lh_sphere_faces=lh_faces,
        lh_sphere_vertex_faces=lh_vert_faces,
        rh_sphere_verts=rh_verts, rh_sphere_faces=rh_faces,
        rh_sphere_vertex_faces=rh_vert_faces,
        lh_mid_verts=lh_mid["vertices"], lh_mid_faces=lh_mid["faces"],
        rh_mid_verts=rh_mid["vertices"], rh_mid_faces=rh_mid["faces"],
        lh_smooth_prep=lh_smooth_prep,
        rh_smooth_prep=rh_smooth_prep,
        lh_smooth_gather=lh_smooth_gather,
        rh_smooth_gather=rh_smooth_gather,
        lh_grad_prep=lh_grad_prep,
        rh_grad_prep=rh_grad_prep,
        neighbors_table=neighbors_table,
        K_neighbors=K_neighbors,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", default="fsaverage6")
    p.add_argument("--smooth-sigma", type=float, default=2.55)
    p.add_argument("--K-hop", type=int, default=3)
    p.add_argument("--cbig-code-dir", default=None)
    p.add_argument("--cache-root", type=Path, default=None,
                   help="Override cache root directory.")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if cache exists.")
    args = p.parse_args(argv)

    # Build a minimal cfg shim for the builder.
    from arealmshbm.step0_pipeline.config import Step0Config
    cfg = Step0Config(
        project_dir="/tmp/_builder_shim",
        sub_id="sub-builder",
        sess_list=("ses-01",),
        mesh=args.mesh,
        smooth_sigma=args.smooth_sigma,
        K_hop=args.K_hop,
        cbig_code_dir=args.cbig_code_dir,
    )

    from arealmshbm.step0_pipeline.inputs_cache import (
        cache_dir_for, save_inputs, cache_exists,
    )
    cache_dir = cache_dir_for(cfg, root=args.cache_root)
    if cache_exists(cache_dir) and not args.force:
        print(f"Cache already exists at {cache_dir}. Use --force to rebuild.",
              flush=True)
        return 0

    print(f"Building step0 inputs:", flush=True)
    print(f"  mesh         = {cfg.mesh}", flush=True)
    print(f"  smooth_sigma = {cfg.smooth_sigma}", flush=True)
    print(f"  K_hop        = {cfg.K_hop}", flush=True)
    print(f"  cache_dir    = {cache_dir}", flush=True)

    t0 = time.perf_counter()
    inputs = build_step0_inputs(cfg)
    build_t = time.perf_counter() - t0
    print(f"  build wall   = {build_t * 1000:.0f} ms", flush=True)

    t0 = time.perf_counter()
    manifest = save_inputs(inputs, cache_dir, cfg)
    save_t = time.perf_counter() - t0
    total_bytes = sum(
        (cache_dir / f).stat().st_size for f in cache_dir.iterdir()
        if (cache_dir / f).is_file())
    print(f"  save wall    = {save_t * 1000:.0f} ms", flush=True)
    print(f"  cache size   = {total_bytes / 1e6:.1f} MB across "
          f"{len(list(cache_dir.iterdir()))} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
