"""step1_pipeline

Step-1 has no top-level pipeline class — the four subgraphs
(generate_profiles, avg_profiles, ini_params, radius_mask) live at
:mod:`arealmshbm.pipeline.step1_runners` and are called directly by
the unified pipeline driver (:mod:`arealmshbm.pipeline`). For
end-to-end runs use the driver, not this folder.

What's left in this folder:

    profile.py — per-stage wall-time report (kept for ad-hoc profile
                 runs against an existing project_dir).

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
