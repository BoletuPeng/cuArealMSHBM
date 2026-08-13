"""inputs_cache.py

On-disk cache for the subject-invariant portion of ``Step0Inputs`` —
the mesh + neighbor + smoothing-gather blob that ``load_inputs`` would
otherwise rebuild from scratch (~3.0 s cold start, dominated by the
~1.85 s ``find_K_neighbors`` BFS expansion).

Layout per cache directory::

    <cache_root>/<mesh>_sigma<S>_khop<K>/
        manifest.json       # version, cfg, file inventory
        medial_mask.npy
        lh_sphere_verts.npy
        ...
        lh_smooth_prep_0.npy ... lh_smooth_prep_6.npy   # tuple, 7 arrays
        lh_grad_prep_0.npy   ... lh_grad_prep_3.npy     # tuple, 4 arrays
        lh_smooth_gather_csr_data.npy                   # CSR triple
        lh_smooth_gather_csr_indices.npy
        lh_smooth_gather_csr_indptr.npy
        lh_smooth_gather_inv.npy                        # inv_weight_sum (N,)
        ...

The cache is keyed on ``(mesh, smooth_sigma, K_hop)`` — the only cfg
fields that determine the output. ``cbig_code_dir`` is intentionally
NOT part of the key: two CBIG checkouts of the same revision produce
identical mesh atlases. If you switch CBIG revisions, delete the
cache or use ``--force`` on the builder.

Read path uses raw ``.npy`` (no compression) so a cache load is just
the OS reading bytes off disk — ~30-100 ms total for the ~158 MB
blob on a warm-FS cold-process. Eager load by default; pass
``mmap=True`` to ``load_cached_inputs`` to defer page-in until first
access (saves no wall on the eager-access step0 hot path, but useful
for tools that only need a subset of fields).

Built by :mod:`arealmshbm.precompute.step0_inputs_builder`; loaded
transparently by ``Step0Pipeline.load_inputs`` on cache hit.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


CACHE_VERSION = "1"

# Plain ndarray fields on Step0Inputs.
_ARRAY_FIELDS = (
    "medial_mask",
    "lh_sphere_verts", "lh_sphere_faces", "lh_sphere_vertex_faces",
    "rh_sphere_verts", "rh_sphere_faces", "rh_sphere_vertex_faces",
    "lh_mid_verts", "lh_mid_faces",
    "rh_mid_verts", "rh_mid_faces",
    "neighbors_table", "K_neighbors",
)

# Tuple-of-ndarray fields (per-hemi prep bundles).
_TUPLE_FIELDS = (
    "lh_smooth_prep", "rh_smooth_prep",
    "lh_grad_prep",   "rh_grad_prep",
)

# CSR-matrix + sidecar-ndarray fields (per-hemi smoothing gather).
# Stored as 4 .npy files: csr_data, csr_indices, csr_indptr, inv.
_CSR_FIELDS = (
    "lh_smooth_gather", "rh_smooth_gather",
)


def cache_dir_for(cfg, root: Optional[Path] = None) -> Path:
    """Compute the canonical cache directory for ``Step0Config``.

    Default root is ``<repo>/arealmshbm/data/precomputed/step0_inputs/``.
    Override via ``root=`` for tests / experiments.
    """
    if root is None:
        root = (Path(__file__).parent.parent.parent / "arealmshbm" /
                "data" / "precomputed" / "step0_inputs")
    # Use a round-stable key for sigma (e.g. 2.55 → "2.55").
    sigma_str = f"{float(cfg.smooth_sigma):g}"
    khop = int(cfg.K_hop)
    key = f"{cfg.mesh}_sigma{sigma_str}_khop{khop}"
    return Path(root) / key


def cache_exists(cache_dir: Path) -> bool:
    """True iff the cache directory has a valid manifest."""
    manifest = Path(cache_dir) / "manifest.json"
    if not manifest.exists():
        return False
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            m = json.load(f)
        return m.get("version") == CACHE_VERSION
    except (json.JSONDecodeError, OSError):
        return False


def save_inputs(inputs, cache_dir: Path, cfg) -> dict:
    """Serialise a ``Step0Inputs`` to disk under ``cache_dir``.

    Returns the manifest dict that was written.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": CACHE_VERSION,
        "cfg": {
            "mesh": str(cfg.mesh),
            "smooth_sigma": float(cfg.smooth_sigma),
            "K_hop": int(cfg.K_hop),
            "cbig_code_dir": (str(cfg.cbig_code_dir)
                               if cfg.cbig_code_dir is not None else None),
        },
        "scalars": {
            "n_lh": int(inputs.n_lh),
            "n_rh": int(inputs.n_rh),
            "n_full": int(inputs.n_full),
            "n_cortex": int(inputs.n_cortex),
        },
        "arrays": {},
        "tuples": {},
        "csr": {},
    }

    # Plain arrays.
    for name in _ARRAY_FIELDS:
        arr = getattr(inputs, name)
        fn = f"{name}.npy"
        np.save(cache_dir / fn, arr, allow_pickle=False)
        manifest["arrays"][name] = fn

    # Tuples of arrays.
    for name in _TUPLE_FIELDS:
        t = getattr(inputs, name)
        fns = []
        for i, arr in enumerate(t):
            fn = f"{name}_{i}.npy"
            np.save(cache_dir / fn, arr, allow_pickle=False)
            fns.append(fn)
        manifest["tuples"][name] = fns

    # CSR + inv_weight_sum sidecars.
    for name in _CSR_FIELDS:
        pair = getattr(inputs, name)
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise TypeError(
                f"{name} expected (csr_matrix, ndarray) tuple, got {type(pair)}")
        csr, inv = pair
        d = {
            "data": f"{name}_csr_data.npy",
            "indices": f"{name}_csr_indices.npy",
            "indptr": f"{name}_csr_indptr.npy",
            "shape": [int(csr.shape[0]), int(csr.shape[1])],
            "inv_weight_sum": f"{name}_inv.npy",
        }
        np.save(cache_dir / d["data"], csr.data, allow_pickle=False)
        np.save(cache_dir / d["indices"], csr.indices, allow_pickle=False)
        np.save(cache_dir / d["indptr"], csr.indptr, allow_pickle=False)
        np.save(cache_dir / d["inv_weight_sum"], inv, allow_pickle=False)
        manifest["csr"][name] = d

    # Manifest last so a partial write isn't mistaken for a valid cache.
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_cached_inputs(cache_dir: Path, *, mmap: bool = False):
    """Load a ``Step0Inputs`` (CPU-only, no GPU mirrors) from disk.

    Parameters
    ----------
    cache_dir : Path
        The per-key directory produced by :func:`save_inputs`.
    mmap : bool
        If True, use ``mmap_mode='r'`` for the heavy arrays. Saves
        no wall on the step0 hot path (every field is eagerly
        consumed by the first sub-graph), but useful for diagnostic
        tools. Defaults to eager load.

    Returns
    -------
    inputs : Step0Inputs
        With ``*_gpu`` fields set to ``None``. The caller
        (``Step0Pipeline.load_inputs``) populates GPU mirrors via
        :func:`_build_gpu_mirrors`.
    """
    from .pipeline import Step0Inputs

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No step0 inputs cache at {cache_dir}. Build with:\n"
            f"  python -m arealmshbm.precompute.step0_inputs_builder "
            f"--mesh <mesh> --smooth-sigma <s> --K-hop <k>")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError(
            f"Cache version mismatch at {cache_dir}: "
            f"on-disk='{manifest.get('version')}' code='{CACHE_VERSION}'. "
            f"Rebuild with the builder script.")

    mmap_mode = "r" if mmap else None
    kwargs: dict = dict(manifest["scalars"])

    # Plain arrays.
    for name, fn in manifest["arrays"].items():
        kwargs[name] = np.load(cache_dir / fn, mmap_mode=mmap_mode,
                                allow_pickle=False)

    # Tuples.
    for name, fns in manifest["tuples"].items():
        kwargs[name] = tuple(
            np.load(cache_dir / fn, mmap_mode=mmap_mode, allow_pickle=False)
            for fn in fns)

    # CSR matrices.
    for name, d in manifest["csr"].items():
        data = np.load(cache_dir / d["data"], mmap_mode=mmap_mode,
                        allow_pickle=False)
        indices = np.load(cache_dir / d["indices"], mmap_mode=mmap_mode,
                           allow_pickle=False)
        indptr = np.load(cache_dir / d["indptr"], mmap_mode=mmap_mode,
                          allow_pickle=False)
        inv = np.load(cache_dir / d["inv_weight_sum"], mmap_mode=mmap_mode,
                       allow_pickle=False)
        # Build a CSR view over the loaded arrays. Note: if mmap=True the
        # data array is read-only — CSR ops that mutate (e.g.
        # sort_indices, sum_duplicates) will error. The cifti_smoothing
        # consumer is read-only so this is fine in production; pass
        # mmap=False if your tool needs to mutate.
        csr = csr_matrix((data, indices, indptr),
                          shape=tuple(d["shape"]), copy=False)
        kwargs[name] = (csr, inv)

    # GPU mirrors are not cached — the dataclass defaults them to None;
    # the caller populates them when backend == 'gpu'.
    return Step0Inputs(**kwargs)
