"""cohort_writer.py — cohort.json writer for the unified driver.

Builds + merges ``<project_dir>/cohort.json`` from the artifacts a
step-1 run produced. The unified pipeline driver calls this at the
end of its step-1 phase.

The cohort.json schema + the field-level merge semantics
(roster-keyed update, artifacts preserved across runs) are owned by
:mod:`arealmshbm.data_io.cohort` — this module is just the
glue between a step-1 run's outputs and that writer.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def write_cohort_manifest(
    *,
    project_dir: Path,
    subjects: Sequence[str],
    sessions: Sequence[str],
    targ_mesh: str,
    seed_mesh: str,
    n_grad_components: int,
    profile_b2nd_paths: Sequence[Tuple[str, Path]],
    group_mat_path: Optional[Path],
    spatial_mask_path: Optional[Path],
    avg_profile_paths: Optional[Tuple[Path, Path]],
) -> Path:
    """Field-level merge cohort.json from a step-1 run's outputs.

    Per-subject ``gradient_lh`` / ``gradient_rh`` are probed from
    ``project_dir/gradients/sub<S>/{lh,rh}_emb_<N>_distance_matrix.{npy,mat}``
    and recorded as .npy if present, else .mat.

    ``profile_b2nd_paths`` is a list of ``(sub_id, path)`` tuples
    produced by :func:`pipeline.step1_runners.run_generate_profiles`.
    """
    from arealmshbm.data_io.cohort import (
        CohortSubject, cohort_json_path, compute_run_id, relpath_or_abs,
        write_cohort_partial,
    )

    subject_ids = [int(s) for s in subjects]
    session_ids = [int(s) for s in sessions]
    if not subject_ids:
        raise ValueError("write_cohort_manifest: subject list is empty.")
    if not session_ids:
        raise ValueError("write_cohort_manifest: session list is empty.")

    nc = int(n_grad_components)
    lh_grad_paths: List[Optional[str]] = []
    rh_grad_paths: List[Optional[str]] = []
    for sub_id in subject_ids:
        sub_dir = (project_dir / "gradients" / f"sub{sub_id}")
        for hemi, dst in (("lh", lh_grad_paths),
                           ("rh", rh_grad_paths)):
            npy_p = sub_dir / f"{hemi}_emb_{nc}_distance_matrix.npy"
            mat_p = sub_dir / f"{hemi}_emb_{nc}_distance_matrix.mat"
            chosen = npy_p if npy_p.exists() else mat_p
            dst.append(relpath_or_abs(project_dir, chosen))

    profile_b2nd_by_id: Dict[str, str] = {
        str(key): relpath_or_abs(project_dir, path)
        for key, path in profile_b2nd_paths
    }

    subjects_cohort = [
        CohortSubject(
            id=str(sub_id),
            sessions=[str(s) for s in session_ids],
            profile_b2nd=profile_b2nd_by_id.get(str(sub_id)),
            gradient_lh=lh_grad_paths[i],
            gradient_rh=rh_grad_paths[i],
        )
        for i, sub_id in enumerate(subject_ids)
    ]
    run_id = compute_run_id(
        subjects=[str(s) for s in subject_ids],
        sessions=[str(s) for s in session_ids],
        targ_mesh=targ_mesh,
        seed_mesh=seed_mesh,
        n_grad_components=nc,
    )

    group_mat = (relpath_or_abs(project_dir, group_mat_path)
                 if group_mat_path is not None else None)
    spatial_mask = (relpath_or_abs(project_dir, spatial_mask_path)
                    if spatial_mask_path is not None else None)
    if avg_profile_paths is not None:
        lh_avg, rh_avg = avg_profile_paths
        avg_lh = relpath_or_abs(project_dir, lh_avg)
        avg_rh = relpath_or_abs(project_dir, rh_avg)
    else:
        avg_lh, avg_rh = None, None

    write_cohort_partial(
        project_dir,
        subjects=subjects_cohort,
        mesh={"targ": targ_mesh, "seed": seed_mesh},
        n_grad_components=nc,
        run_id=run_id,
        group_mat=group_mat,
        spatial_mask_mat=spatial_mask,
        avg_profile_lh_npy=avg_lh,
        avg_profile_rh_npy=avg_rh,
    )
    return cohort_json_path(project_dir)
