"""test_atlas_resolver.py — `_atlas_dir()` fail-fast + avg_mesh asset contract.

``MSHBM_ATLAS_DIR`` is the runtime atlas resolver, used ONLY for the live
``label/*.annot`` reads in step 1 (aparc + Schaefer). Mesh geometry no
longer touches it — it comes wholly from the shipped avg_mesh ``.npz``
bundles, which ``load_avg_mesh`` reads as assets with no raw-FreeSurfer
fallback.

Coverage:
  * unset / empty env ⇒ RuntimeError (guards against a silent default)
  * set env  ⇒ Path(env) returned as-is
  * error message describes the annot layout the resolver now serves
  * a missing avg_mesh bundle is a hard FileNotFoundError naming the
    asset — NOT a recoverable cache miss

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path

import pytest

from arealmshbm.data_io.load_avg_mesh import (
    _atlas_dir, _load_avg_mesh_cached, load_avg_mesh,
)


def test_atlas_dir_unset_raises(monkeypatch):
    """Missing MSHBM_ATLAS_DIR must raise RuntimeError, not fall back."""
    monkeypatch.delenv("MSHBM_ATLAS_DIR", raising=False)
    with pytest.raises(RuntimeError, match="MSHBM_ATLAS_DIR is unset or empty"):
        _atlas_dir()


def test_atlas_dir_empty_raises(monkeypatch):
    """Shell-set ``MSHBM_ATLAS_DIR=""`` must raise the same error as unset
    — the empty string is the most common accidental "default" we defend
    against, and the message must spell out that both states are bad.
    """
    monkeypatch.setenv("MSHBM_ATLAS_DIR", "")
    with pytest.raises(RuntimeError, match="MSHBM_ATLAS_DIR is unset or empty"):
        _atlas_dir()


def test_atlas_dir_set_returns_path(monkeypatch, tmp_path):
    """When set, ``_atlas_dir()`` returns ``Path(env)`` as-is. Directory
    existence is the downstream caller's responsibility — keeps the
    resolver pure.
    """
    monkeypatch.setenv("MSHBM_ATLAS_DIR", str(tmp_path))
    result = _atlas_dir()
    assert isinstance(result, Path)
    assert result == tmp_path


def test_atlas_dir_error_describes_annot_layout(monkeypatch):
    """The RuntimeError must describe what the atlas dir is now FOR — the
    live ``<targ_mesh>/label/*.annot`` reads — so a library-only consumer
    can act on it. It must NOT claim surf/ is read (mesh geometry comes
    from the shipped bundles now); pin that the message stays annot-scoped.
    """
    monkeypatch.delenv("MSHBM_ATLAS_DIR", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        _atlas_dir()
    msg = str(excinfo.value)
    assert "label" in msg, "error must mention the label/ subdir"
    assert "annot" in msg, "error must name the annot files it serves"


def test_load_avg_mesh_missing_bundle_is_hard_error(monkeypatch, tmp_path):
    """A missing avg_mesh bundle is a broken install, not a cache miss:
    ``load_avg_mesh`` must raise FileNotFoundError naming the asset. There
    is NO raw-FreeSurfer rebuild path, so MSHBM_ATLAS_DIR being unset is
    irrelevant here — the failure is purely the absent ``.npz``.
    """
    monkeypatch.delenv("MSHBM_ATLAS_DIR", raising=False)
    # Point the bundle root at an empty tmp dir so the asset is absent.
    monkeypatch.setenv("MSHBM_PRECOMPUTED_ROOT", str(tmp_path / "precomputed"))
    # Clear the in-process lru_cache so a prior call can't pre-empt the
    # miss. Clear again in finally so the miss doesn't leak into later
    # tests (lru_cache interactions have bitten this codebase before).
    _load_avg_mesh_cached.cache_clear()
    try:
        with pytest.raises(FileNotFoundError) as excinfo:
            load_avg_mesh("lh", "fsaverage3", "inflated")
        msg = str(excinfo.value)
        assert "fsaverage3" in msg, "must name the missing bundle"
        assert "asset" in msg, "must frame it as a shipped asset, not a cache"
    finally:
        _load_avg_mesh_cached.cache_clear()


def test_load_avg_mesh_validates_hemi_and_surface(monkeypatch, tmp_path):
    """Bad ``hemi`` / ``surface`` are rejected before any bundle lookup,
    so the error is deterministic regardless of cache state.
    """
    monkeypatch.setenv("MSHBM_PRECOMPUTED_ROOT", str(tmp_path / "precomputed"))
    with pytest.raises(ValueError, match="hemi"):
        load_avg_mesh("LH", "fsaverage6", "inflated")
    with pytest.raises(ValueError, match="surface"):
        load_avg_mesh("lh", "fsaverage6", "midthickness")
