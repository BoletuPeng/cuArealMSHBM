"""test_removed_symbols.py — pin the superseded-symbol contract.

Guard against accidental re-introduction of a read surface that a
later, bit-equal implementation replaced: the legacy non-bitpacked
profile reader (2026-06 bitpacked-only consolidation) and the
per-file nvCOMP ``Codec`` GIFTI reader plus its step-0 GPU prefetcher
branch (superseded by the batched whole-subject
``gifti_bold_gpu.read_subject_bold_gpu``), the pull Bellman-Ford
geodesic solver and the per-session GPU profile leaf (each folded
into the bit-identical path that replaced it, 2026-09), and the dense
step-2 CuPy port, retired for the P-layout session that now answers
to ``backend='gpu'`` (2026-09). If any name
reappears as a public export of its old module, the import succeeds
and the corresponding assertion fires — flagging the regression in CI
before a downstream consumer can latch onto it. Whole modules,
constructor kwargs and function kwargs that went the same way are
pinned further down.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import importlib
import importlib.util

import pytest


# Module path → list of attribute names that must NOT be present.
# Each entry maps a live module to the set of names it exported before
# the cleanup. Module imports must still succeed (the module itself
# wasn't removed); only the named attrs are gone.
_REMOVED = {
    "arealmshbm.data_io.profile_io": (
        "read_subject_profile_tnd",
        "open_subject_profile_tnd",
        "SubjectProfileFp32View",
    ),
    # ``_kernels_gpu`` now holds the P-layout kernels; neither the dense
    # port's names nor the legacy names it had already shed may return.
    "arealmshbm.step2_em_iter_master._kernels_gpu": (
        "_widen_normalize_bold_uint8_to_f32_cupy",
        "_normalize_bold_u8_to_f32_kernel",
        "pack_binary_NTD",
        "em_iter_master_kernel_streaming_cupy",
        "_widen_normalize_bold_bitpacked_to_f32_cupy",
        "_sigma_psi_SLD_compute_cupy",
        "_compute_X_dot_sl_s_NTD_cupy",
        "_mstep_inner_loop_step2_cupy",
        "_spatial_connect_per_subject_cupy",
        "_fused_estep_per_subject_NTD_cupy",
        "_phase_e1_normalize_per_subject_cupy",
        "_phase_e2_theta_only_cupy",
        "warmup_step2_gpu_sparse",
    ),
    "arealmshbm.step2_em_iter_master": (
        "Step2EmIterSessionCUDA",
        "em_iter_master_kernel_streaming_cupy",
        "warmup_step2_gpu_sparse",
    ),
    "arealmshbm.step2_em_iter_master.session_gpu": (
        "Step2EmIterSessionCUDA",
    ),
    # The step-2 pipeline no longer keeps a set of GPU backend values,
    # and the config parser no longer keeps a step-2-only backend tuple.
    "arealmshbm.step2_pipeline.pipeline": ("_GPU_BACKENDS",),
    "arealmshbm.pipeline.config": ("_VALID_BACKENDS_STEP2",),
    # Second-round cleanup: fetch_data → packed-only + Session-side
    # unpack+normalize. The numba normalize kernel moved out of
    # fetch_data into arealmshbm.data_io.bitpacked_norm.
    "arealmshbm.data_io.fetch_data": (
        "_read_b2nd_series_normalize_bitpacked",
        "_normalize_session_bitpacked_numba",
    ),
    # The legacy fp32 in-place CUDA normalize (and its wrapper) — only
    # caller was vmf_clustering_gpu's fp32 BOLD input branch, both gone.
    "arealmshbm.vmf_clustering.vmf_clustering_gpu": (
        "_normalize_bold_kernel",
        "_normalize_bold_NTD_inplace",
    ),
    # The host fp32-input demean+L2-norm helpers — production code
    # path now goes through _widen_normalize_bitpacked_to_f32_NTD_kernel
    # directly; the only remaining caller was a test-side reference
    # impl, now inlined into the test file.
    "arealmshbm.step2_io.load_subject_profiles": (
        "_normalize_session",
        "_normalize_session_inplace_kernel",
    ),
    # Per-file nvCOMP ``Codec`` GIFTI reader — superseded by the
    # batched whole-subject ``gifti_bold_gpu.read_subject_bold_gpu``,
    # which is bit-equal to the CPU reader (pinned by
    # ``test_gifti_readers.py``). Its only production caller was the
    # step-0 prefetcher's ``backend='gpu'`` branch, also removed.
    "arealmshbm.data_io.gifti_io": (
        "read_surface_gifti_gpu",
        "_get_gpu_codec",
        "_GPU_CODEC_CACHE",
        "_GPU_CODEC_LOCK",
    ),
    # Device-side ``concat_hemis_drop_medial`` mirror — its module
    # ``arealmshbm.bold_io.bold_io_gpu`` went with the GPU prefetcher
    # branch, so the name must not reappear on the package either.
    "arealmshbm.bold_io": (
        "concat_hemis_drop_medial_gpu",
        "bold_io_gpu",
    ),
    "arealmshbm.pipeline._step0_bold_prefetcher": (
        "_load_session_bold_gpu",
    ),
    # The legacy ``inv_norm = +inf`` zscore kernel of the retired
    # per-session GPU leaf (``profiles_gpu.py``, deleted outright); the
    # fused subject leaf uses the zero-variance-safe kernel. Its source
    # now lives only in ``test_subject_profiles_gpu.py``'s oracle.
    "arealmshbm.generate_profiles._kernels_gpu": (
        "zscore_unit_norm_columns_cupy",
        "_zscore_kernel",
    ),
    # The batched pull Bellman-Ford geodesic solver and the dispatcher
    # that could still route to it. ``graph_distance/_kernels_gpu.py``
    # now holds the Δ-stepping kernels it was bit-identical to; BF
    # survives only as the oracle in
    # ``graph_distance/tests/test_gpu_delta_correctness.py``.
    "arealmshbm.graph_distance.graph_distance_gpu": (
        "_gpu_device_bf",
        "_delta_eligible",
    ),
    "arealmshbm.graph_distance._kernels_gpu": (
        "get_bf_iter_kernel",
        "_build_bf_iter_kernel",
        "precompute_edge_weights_cupy",
        "init_distance_cupy",
        "MAX_ITERS",
        "_BF_ITER_KERNEL_SRC",
    ),
}

# Modules deleted outright with the path they implemented.
_REMOVED_MODULES = (
    # The per-session GPU profile leaf, folded into the fused
    # whole-subject leaf (``profiles_subject_gpu``) it was bit-identical
    # to; its algorithm is the oracle in ``test_subject_profiles_gpu.py``.
    "arealmshbm.generate_profiles.profiles_gpu",
    # The device-side ``concat_hemis_drop_medial`` mirror behind the
    # step-0 prefetcher's GPU branch.
    "arealmshbm.bold_io.bold_io_gpu",
    # The P-layout step-2 backend's modules under the names they had
    # while the dense port held ``_kernels_gpu`` / ``session_gpu``.
    "arealmshbm.step2_em_iter_master._kernels_gpu_sparse",
    "arealmshbm.step2_em_iter_master.session_gpu_sparse",
)


@pytest.mark.parametrize(
    "modname,attr",
    [(m, a) for m, attrs in _REMOVED.items() for a in attrs],
)
def test_removed_symbol_is_absent(modname: str, attr: str) -> None:
    """The named attribute must NOT exist on the live module."""
    mod = importlib.import_module(modname)
    assert not hasattr(mod, attr), (
        f"{modname}.{attr} re-appeared — it was removed in favour of a "
        f"bit-equal replacement. If this symbol is genuinely needed "
        f"again, update _REMOVED in this test."
    )


@pytest.mark.parametrize("modname", _REMOVED_MODULES)
def test_removed_module_is_absent(modname: str) -> None:
    """The module must not be importable at all any more."""
    assert importlib.util.find_spec(modname) is None, (
        f"{modname} re-appeared — its path was folded into the bit-equal "
        f"replacement. If it is genuinely needed again, update "
        f"_REMOVED_MODULES in this test."
    )


def test_subject_profile_loader_has_no_load_raw_into() -> None:
    """``load_raw_into`` was removed from SubjectProfileLoader (the only
    consumer was the deleted GPU 'eager' unpacked-uint8 cache mode)."""
    from arealmshbm.step2_io.subject_loaders import SubjectProfileLoader
    assert not hasattr(SubjectProfileLoader, "load_raw_into")


def test_subject_profile_loader_has_no_load_packed_into() -> None:
    """``load_packed_into`` fed the dense step-2 CuPy port's device cache;
    the P-layout backend reads packed bytes through its own loader."""
    from arealmshbm.step2_io.subject_loaders import SubjectProfileLoader
    assert not hasattr(SubjectProfileLoader, "load_packed_into")


def test_cpu_step2_session_has_no_sync_to_host() -> None:
    """``sync_to_host`` existed for API parity with the dense CuPy port;
    the CPU Session aliases ``Params`` to its buffers and never needed it."""
    from arealmshbm.step2_em_iter_master import Step2EmIterSession
    assert not hasattr(Step2EmIterSession, "sync_to_host")


def test_step2_has_no_tf32_knob() -> None:
    """Step 2's TF32 toggle wrapped the dense CuPy port's sgemms; the
    P-layout backend makes one cuBLAS call per run."""
    from arealmshbm.pipeline.config import Step2Knobs
    from arealmshbm.pipeline.driver import Pipeline
    assert "enable_tf32" not in Step2Knobs.__dataclass_fields__
    assert not hasattr(Pipeline, "_run_step2_train_prior_maybe_tf32")


# ─────────────────────────────────────────────────────────────────────
# kwarg absence — same idea as symbol absence, but for constructor
# kwargs whose alternative branch was deleted.
# ─────────────────────────────────────────────────────────────────────
def _import_class(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


@pytest.mark.parametrize(
    "module_path,class_name,kwarg",
    [
        # Both Session classes lost ``bold_pre_normalized`` when the
        # caller-side dispatch became single-path (packed only).
        ("arealmshbm.vmf_clustering.vmf_clustering",
         "VmfClusteringSession", "bold_pre_normalized"),
        ("arealmshbm.vmf_clustering.vmf_clustering_gpu",
         "VmfClusteringSessionCUDA", "bold_pre_normalized"),
        # The step-0 prefetcher lost ``backend`` when its nvCOMP GPU
        # decode branch was dropped — decode is CPU isal, full stop.
        ("arealmshbm.pipeline._step0_bold_prefetcher",
         "Step0BoldPrefetcher", "backend"),
        # The step-1 stage pipeline lost ``backend`` with its packed/GPU
        # stages: it is the CPU backend's, and ``backend='gpu'`` is the
        # fused subject leaf.
        ("arealmshbm.pipeline._step1_stage_pipeline",
         "Step1GenerateProfilesStagePipeline", "backend"),
    ],
)
def test_removed_constructor_kwarg(module_path: str, class_name: str,
                                    kwarg: str) -> None:
    """The named kwarg must NOT be in the class ``__init__`` signature."""
    import inspect
    cls = _import_class(module_path, class_name)
    sig = inspect.signature(cls.__init__)
    assert kwarg not in sig.parameters, (
        f"{module_path}.{class_name}.__init__ has kwarg {kwarg!r} — it "
        f"was removed with the branch it selected. If genuinely needed "
        f"again, update the parametrize list in this test."
    )


@pytest.mark.parametrize(
    "module_path,func_name,kwarg",
    [
        # ``fetch_data`` lost ``bold_format`` + ``normalize`` when the
        # output became packed-only with consumer-side unpack+normalize.
        ("arealmshbm.data_io.fetch_data", "fetch_data", "bold_format"),
        ("arealmshbm.data_io.fetch_data", "fetch_data", "normalize"),
        # The CPU profile leaf lost ``backend`` when the per-session GPU
        # leaf it selected was folded into the fused subject leaf.
        ("arealmshbm.generate_profiles.profiles",
         "compute_profile_arrays", "backend"),
        # The Δ solver lost its ``algorithm`` knob and the Bellman-Ford
        # iteration cap with the BF path it was bit-identical to.
        ("arealmshbm.graph_distance.graph_distance_gpu",
         "gradient_geodesic_distance_gpu_device", "algorithm"),
        ("arealmshbm.graph_distance.graph_distance_gpu",
         "gradient_geodesic_distance_gpu_device", "max_iters"),
        # The one-shot .b2nd writer lost its pre-packed input mode with
        # its last caller (the stage pipeline's GPU branch); packed
        # slabs stream through ``SubjectProfileStreamWriter``.
        ("arealmshbm.data_io.profile_io",
         "write_subject_profile_tnd", "D_unpacked"),
    ],
)
def test_removed_function_kwarg(module_path: str, func_name: str,
                                kwarg: str) -> None:
    """The named kwarg must NOT be in the function signature."""
    import inspect
    fn = getattr(importlib.import_module(module_path), func_name)
    sig = inspect.signature(fn)
    assert kwarg not in sig.parameters, (
        f"{module_path}.{func_name} has kwarg {kwarg!r} — it was removed "
        f"with the branch it selected. If genuinely needed again, update "
        f"the parametrize list in this test."
    )
