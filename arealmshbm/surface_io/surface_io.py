"""surface_io.py

Public entry: ``read_surface_mesh(path)``.

Dispatches by file extension:

* ``.surf.gii`` — GIFTI; uses ``nibabel.gifti``.
  - darray with intent ``NIFTI_INTENT_POINTSET`` (1008) is the (N,3) verts.
  - darray with intent ``NIFTI_INTENT_TRIANGLE`` (1009) is the (F,3) faces.
  - Faces in the GIFTI spec are 0-indexed; we keep them as such.

* ``.sphere`` (FreeSurfer binary, big-endian) — uses
  ``nibabel.freesurfer.io.read_geometry``. Faces are 0-indexed.

Returned arrays are always:
    vertices : (N, 3) float32
    faces    : (F, 3) int32 (0-indexed)

Note on face orientation: CBIG's ``read_surf.m`` (which the MATLAB
pipeline calls) and ``nibabel.freesurfer.io.read_geometry`` produce the
same triangles with *possibly cyclically rotated* vertex order in each
row (i.e. (a,b,c) vs (c,a,b)). This is a benign difference: same
triangle, same orientation. The set of faces is identical.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np


def read_surface_mesh(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read a surface mesh from GIFTI or FreeSurfer binary.

    Parameters
    ----------
    path : str | Path
        Path ending in ``.surf.gii`` (GIFTI) or one of the FreeSurfer
        binary suffixes recognized by ``nibabel.freesurfer.io``
        (``.sphere``, ``.pial``, ``.white``, ``.inflated``, ...).

    Returns
    -------
    mesh : dict
        - ``vertices`` : (N, 3) float32
        - ``faces``    : (F, 3) int32 (0-indexed)
    """
    p = Path(path)
    name = p.name.lower()

    # Only ``.surf.gii`` is a surface mesh; ``.func.gii`` / ``.shape.gii``
    # carry per-vertex data and must NOT be dispatched here (the GIFTI
    # reader below assumes POINTSET + TRIANGLE darrays).
    if name.endswith(".surf.gii"):
        return _read_gifti(p)
    if name.endswith(".gii"):
        raise ValueError(
            f"read_surface_mesh only accepts surface GIFTIs (.surf.gii); "
            f"got {p.name!r}. Use a dedicated data reader for "
            f".func.gii / .shape.gii / .label.gii etc."
        )

    # FreeSurfer binary surface — recognized by extension. The .sphere
    # files in step 0 have no further suffix, but we accept the common
    # FreeSurfer surface suffixes for robustness.
    return _read_freesurfer(p)


def _read_gifti(path: Path) -> Dict[str, np.ndarray]:
    import nibabel as nib

    img = nib.load(str(path))

    verts = None
    faces = None
    # Prefer the NIfTI intent codes (1008 = POINTSET, 1009 = TRIANGLE).
    # Fall back to positional darrays (0 = verts, 1 = faces) which is
    # the convention the GIFTI spec uses for surface files.
    for d in img.darrays:
        intent = int(getattr(d, "intent", 0) or 0)
        if intent == 1008 and verts is None:
            verts = np.asarray(d.data)
        elif intent == 1009 and faces is None:
            faces = np.asarray(d.data)

    if verts is None or faces is None:
        # Positional fallback.
        if len(img.darrays) < 2:
            raise ValueError(
                f"GIFTI file {path} has fewer than 2 darrays; "
                f"cannot extract verts+faces."
            )
        if verts is None:
            verts = np.asarray(img.darrays[0].data)
        if faces is None:
            faces = np.asarray(img.darrays[1].data)

    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"GIFTI verts have unexpected shape {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"GIFTI faces have unexpected shape {faces.shape}")

    return {"vertices": verts, "faces": faces}


def _read_freesurfer(path: Path) -> Dict[str, np.ndarray]:
    import nibabel.freesurfer.io as fsio

    verts, faces = fsio.read_geometry(str(path))
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"FreeSurfer verts have unexpected shape {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"FreeSurfer faces have unexpected shape {faces.shape}")

    return {"vertices": verts, "faces": faces}
