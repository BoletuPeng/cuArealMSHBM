"""load_group_mtc.py — step-2 L6 leaf.

Read the step-1 ``group.mat`` written by
:func:`arealmshbm.ini_params.ini_params.generate_ini_params_xxx`
(see ``ini_params.py`` lines 202-213). Used by step-2 group-prior
estimation to seed the EM (``setting_params.dim = size(mtc, 1) - 1``;
the loader returns mtc verbatim, downstream code slices as needed).

MATLAB source: implicit — step 2's sequential driver loads this via
``load(fullfile(work_dir, 'group', 'group.mat'))``.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.io import loadmat


def load_group_mtc(group_mat_path: str | Path) -> Dict[str, Any]:
    """Read ``group.mat`` and return a plain dict.

    Parameters
    ----------
    group_mat_path : path to ``group.mat`` (typically
        ``<project_dir>/group/group.mat``).

    Returns
    -------
    dict with keys:
        ``mtc``        : (D+1, L) fp64 array — bytewise-identical to
                         ``scipy.io.loadmat(...)["mtc"]``. Downstream
                         consumers slice along axis 0 as ``D = mtc.shape[0] - 1``.
        ``epsil``      : scalar float — vMF concentration init.
        ``lh_labels``  : (N_lh,) int — flattened to 1-D (file stores
                         column vector).
        ``rh_labels``  : (N_rh,) int — same.

    Sanity
    ------
    Asserts ``mtc`` is 2-D and that ``L = mtc.shape[1]`` columns and
    ``D+1 = mtc.shape[0]`` rows are sane positive integers.
    """
    p = Path(group_mat_path)
    if not p.exists():
        raise FileNotFoundError(f"group.mat not found: {p}")

    m = loadmat(str(p))

    if "mtc" not in m:
        raise KeyError(f"group.mat missing 'mtc' field: keys={list(m.keys())}")
    if "epsil" not in m:
        raise KeyError(f"group.mat missing 'epsil' field: keys={list(m.keys())}")
    if "lh_labels" not in m or "rh_labels" not in m:
        raise KeyError(
            f"group.mat missing 'lh_labels' / 'rh_labels': keys={list(m.keys())}"
        )

    mtc = np.asarray(m["mtc"])
    if mtc.ndim != 2:
        raise ValueError(f"group.mat 'mtc' must be 2-D; got shape {mtc.shape}")
    Dp1, L = mtc.shape
    if Dp1 < 2 or L < 1:
        raise ValueError(
            f"group.mat 'mtc' shape {mtc.shape} is degenerate "
            f"(expected D+1 >= 2 rows and L >= 1 cols)"
        )

    epsil_arr = np.asarray(m["epsil"])
    # MATLAB writes epsil as (1, 1) — collapse to a Python float.
    epsil = float(epsil_arr.ravel()[0])

    lh_labels = np.asarray(m["lh_labels"]).ravel().astype(np.int64, copy=False)
    rh_labels = np.asarray(m["rh_labels"]).ravel().astype(np.int64, copy=False)

    return {
        "mtc": mtc,
        "epsil": epsil,
        "lh_labels": lh_labels,
        "rh_labels": rh_labels,
    }
