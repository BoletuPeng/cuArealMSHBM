"""step0_pipeline

End-to-end Python step-0 super-call (Mode A, fsaverage6).

Public API:
    Step0Config         — pipeline-config dataclass (every knob).
    Step0Pipeline       — single-subject load → run → save lifecycle.
    Step0Inputs         — bundled subject-independent loaded inputs.
    Step0Result         — bundled outputs of one ``run()``.

For multi-subject batch runs, use the unified pipeline driver at
``arealmshbm.pipeline.Pipeline`` with ``mode='modeA_batch'``.

Production path (gpu backend; cpu also available)::

    from arealmshbm.step0_pipeline import Step0Config, Step0Pipeline

    cfg = Step0Config(
        project_dir="projects/demo/sub-001",
        sub_id="sub-001",
        sess_list=("ses-01", "ses-02", "ses-03",
                   "ses-04", "ses-05", "ses-06"),
        backend="gpu",
    )
    with Step0Pipeline(cfg) as pipe:
        result = pipe.run_and_save()

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""


# Lazy public re-exports (PEP 562): ``.pipeline`` pulls the whole step-0
# leaf set (numba, scipy.sparse, ~0.74 s cold) and ``.config`` imports
# cupy, so deferring both keeps importing this package cheap for a
# caller that only wants a submodule. Other submodules resolve via
# ``__getattr__``'s find_spec.

_LAZY_EXPORTS = {
    "Step0Config": "config",
    "Step0Pipeline": "pipeline",
    "Step0Inputs": "pipeline",
    "Step0Result": "pipeline",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    """Resolve a lazy step-0 re-export, or a submodule, on first access."""
    from importlib import import_module
    sub = _LAZY_EXPORTS.get(name)
    if sub is not None:
        value = getattr(import_module(f"{__name__}.{sub}"), name)
        globals()[name] = value
        return value
    if not name.startswith("_"):
        from importlib.util import find_spec
        try:
            found = find_spec(f"{__name__}.{name}") is not None
        except (ImportError, AttributeError, ValueError):
            found = False
        if found:
            return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals()) + __all__))
