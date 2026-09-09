"""test_lazy_package_exports.py — the PEP-562 package ``__getattr__``s.

``arealmshbm``, ``arealmshbm.data_io`` and ``arealmshbm.step0_pipeline``
resolve their public names lazily. That is what keeps a step-0-only
process out of the step-2/3 import graph — but a lazy re-export is easy
to break silently, so this pins the surface: every name still resolves,
submodule access still works, and an unknown name is still an
``AttributeError``.

``step3_pipeline`` is eager; the ``step3_pipeline`` -> ``vmf_clustering``
-> ``.variant`` cycle is broken inside ``step3_pipeline/pipeline.py``,
which imports ``vmf_clustering`` function-locally. The first test below
pins that it stays broken.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import pytest


def test_the_step3_import_cycle_stays_broken():
    import arealmshbm.vmf_clustering  # noqa: F401
    from arealmshbm.step3_pipeline import Step3Pipeline  # noqa: F401


def test_lazy_names_resolve_to_the_real_objects():
    import arealmshbm

    assert (arealmshbm.Step0Pipeline
            is arealmshbm.step0_pipeline.pipeline.Step0Pipeline)
    assert (arealmshbm.Step3Config
            is arealmshbm.step3_pipeline.config.Step3Config)


def test_submodule_attribute_access_still_works():
    import arealmshbm

    assert arealmshbm.data_io.__name__ == "arealmshbm.data_io"
    assert (arealmshbm.data_io.read_surface_gifti
            is arealmshbm.data_io.gifti_io.read_surface_gifti)


def test_all_is_the_twelve_step_entry_points():
    import arealmshbm

    assert set(arealmshbm.__all__) == {
        f"Step{n}{k}" for n in (0, 2, 3)
        for k in ("Config", "Pipeline", "Inputs", "Result")}
    for name in arealmshbm.__all__:
        assert getattr(arealmshbm, name) is not None


def test_unknown_names_still_raise():
    import arealmshbm
    import arealmshbm.data_io
    import arealmshbm.step0_pipeline

    for mod in (arealmshbm, arealmshbm.data_io, arealmshbm.step0_pipeline):
        with pytest.raises(AttributeError):
            getattr(mod, "Step9Pipeline")
