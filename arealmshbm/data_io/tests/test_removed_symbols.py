"""test_removed_symbols.py — pin the 2026-06 cleanup contract.

Guard against accidental re-introduction of the legacy non-bitpacked
profile read surface. Every symbol below was removed when the BOLD
profile reader was consolidated to bitpacked-only (CPU + GPU, eager
+ stream). If any name reappears as a public export of its old
module, the import succeeds and the corresponding assertion fires —
flagging the regression in CI before a downstream consumer can latch
onto it.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import importlib

import pytest


# Module path → list of attribute names that must NOT be present.
# Each entry maps a live module to the set of names that lived there
# before the 2026-06 cleanup. Module imports must still succeed (the
# module itself wasn't removed); only the named attrs are gone.
_REMOVED = {
    "arealmshbm.data_io.profile_io": (
        "read_subject_profile_tnd",
        "open_subject_profile_tnd",
        "SubjectProfileFp32View",
    ),
    "arealmshbm.step2_em_iter_master._kernels_gpu": (
        "_widen_normalize_bold_uint8_to_f32_cupy",
        "_normalize_bold_u8_to_f32_kernel",
        "pack_binary_NTD",
    ),
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
}


@pytest.mark.parametrize(
    "modname,attr",
    [(m, a) for m, attrs in _REMOVED.items() for a in attrs],
)
def test_removed_symbol_is_absent(modname: str, attr: str) -> None:
    """The named attribute must NOT exist on the live module."""
    mod = importlib.import_module(modname)
    assert not hasattr(mod, attr), (
        f"{modname}.{attr} re-appeared — it was removed in the bitpacked-"
        f"only consolidation. If this symbol is genuinely needed again, "
        f"update _REMOVED in this test."
    )


def test_subject_profile_loader_has_no_load_raw_into() -> None:
    """``load_raw_into`` was removed from SubjectProfileLoader (the only
    consumer was the deleted GPU 'eager' unpacked-uint8 cache mode)."""
    from arealmshbm.step2_io.subject_loaders import SubjectProfileLoader
    assert not hasattr(SubjectProfileLoader, "load_raw_into")


# ─────────────────────────────────────────────────────────────────────
# kwarg absence — same idea as symbol absence, but for constructor kwargs
# that were deleted when fetch_data + Session collapsed to packed-only.
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
        f"was removed in the bitpacked-only consolidation. If genuinely "
        f"needed again, update the parametrize list in this test."
    )


@pytest.mark.parametrize(
    "kwarg",
    ["bold_format", "normalize"],
)
def test_fetch_data_no_removed_kwarg(kwarg: str) -> None:
    """``fetch_data`` lost ``bold_format`` + ``normalize`` when the
    output became packed-only with consumer-side unpack+normalize."""
    import inspect
    from arealmshbm.data_io.fetch_data import fetch_data
    sig = inspect.signature(fetch_data)
    assert kwarg not in sig.parameters, (
        f"fetch_data has kwarg {kwarg!r} — it was removed in the "
        f"bitpacked-only consolidation."
    )
