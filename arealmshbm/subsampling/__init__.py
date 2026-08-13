"""subsampling

Random-index subsampling for FC speed-up.

Public API:
    set_downsample_params(num_vertices, sub_verts, sub_FC, scan_idx)
        -> (randinds_verts, randinds_FC), both 0-indexed int64.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""

from .subsampling import set_downsample_params

__all__ = ["set_downsample_params"]
