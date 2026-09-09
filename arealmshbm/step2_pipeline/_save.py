"""_save.py — ``Params_Final.mat`` writer for step 2 (every backend).

Lifted out of ``Step2Pipeline._save_params`` so both the dense/CPU and
the sparse GPU paths write through one function (design contract
``docs/step2_sparse_design.md`` §6). ``pipeline.py`` keeps a one-line
delegation.

Two things happen here and nowhere else:

* **Internal → external layout.** The ``Params`` dict carried through
  ``run_em`` holds ``mu`` as ``(L, D)``; MATLAB's convention (and every
  reader of this file) wants ``(D, L)``. The per-subject keys that would
  also need transposing (``s_lambda``/``s_psi``/``s_t_nu``) are stripped
  by the caller, so ``mu`` is the only transpose.
* **The ``theta`` container.** ``theta`` always lands as a dense fp64
  ``(N, L)`` block in a ``do_compression=True`` container — the on-disk
  contract this file has always had. CBIG's MATLAB
  ``CBIG_MSHBM_generate_individual_parcellation.m`` evaluates
  ``log(Params.theta)``, which MATLAB will not do on a sparse input, so
  a scipy-sparse ``theta`` (what the ``gpu`` backend's
  ``export_params`` returns) is densified here.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def save_params_final(Params: Dict[str, Any], path: str | Path) -> None:
    """Write ``Params`` as the MATLAB v5 struct ``Params`` at ``path``.

    Parameters
    ----------
    Params : the run's Params dict, already stripped of the per-subject
        keys (``s_lambda``, ``s_psi``, ``s_t_nu``).
    path : destination ``Params_Final.mat``; parent dirs are created.

    Field conversions (unchanged from the original ``_save_params``):
    float ndarrays → fp64; other ndarrays verbatim; ``int``/``float``/
    ``str`` verbatim; ``list`` → ``np.asarray``; anything else passed
    through to ``savemat`` untouched. ``mu`` is transposed ``(L, D) →
    (D, L)``; a scipy-sparse ``theta`` is densified.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {}
    for k, v in Params.items():
        if k == "theta" and v is not None and sp.issparse(v):
            # The writer owns this policy — callers hand over whatever
            # ``export_params`` gave them, and MATLAB CBIG cannot read a
            # sparse ``Params.theta``.
            v = v.toarray()
        if isinstance(v, np.ndarray):
            arr = v
            if k == "mu" and arr.ndim == 2:
                arr = np.ascontiguousarray(arr.T)
            if arr.dtype.kind == "f":
                out[k] = arr.astype(np.float64, copy=False)
            else:
                out[k] = arr
        elif isinstance(v, (int, float, str)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = np.asarray(v)
        else:
            out[k] = v

    # Compression stays on: a 199 MB payload is not something to put on
    # disk uncompressed, and this is the byte contract the legacy saver
    # had.
    sio.savemat(p, {"Params": out}, do_compression=True, format="5")
