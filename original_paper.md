# Areal-MSHBM — original publication

This project implements the Areal multi-session hierarchical Bayesian
model (Areal-MSHBM). The algorithm is described in:

> Kong R, Yang Q, Gordon E, Xue A, Yan X, Orban C, Zuo X-N, Spreng N,
> Ge T, Holmes A, Eickhoff S, Yeo BTT.
> **Individual-Specific Areal-Level Parcellations Improve Functional
> Connectivity Prediction of Behavior.**
> *Cerebral Cortex*, 2021;31(10):4477-4500.
> doi: [10.1093/cercor/bhab101](https://doi.org/10.1093/cercor/bhab101)

Reference implementation by the original authors (MATLAB), in the CBIG
repository: <https://github.com/ThomasYeoLab/CBIG> under
`stable_projects/brain_parcellation/Kong2022_ArealMSHBM`.

All algorithmic details (generative model, EM updates, spatial priors,
the gMSHBM / dMSHBM / cMSHBM variants) follow the paper and the
reference implementation; see `docs/` for how each pipeline step maps
onto them.
