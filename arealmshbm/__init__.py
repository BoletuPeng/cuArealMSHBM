"""arealmshbm

Python implementation of the Kong-2022 Areal multi-session
hierarchical Bayesian model (Areal-MSHBM) for individual-specific
cortical parcellation. Both Mode A (HCP-prior personalisation) and
Mode B (self-trained prior) are supported end-to-end.

Top-level entry: the project-driven driver under
:mod:`arealmshbm.pipeline`. The per-step super-calls re-exported below
are kept for targeted regression / profile runs:

    Step0Config / Step0Pipeline — RSFC gradient + diffusion embedding.
    Step2Config / Step2Pipeline — group prior estimation (Mode B).
    Step3Config / Step3Pipeline — individual parcellation EM.

Step 1 has no top-level re-export because it has no ``StepNPipeline``
class — the four subgraphs (generate_profiles, avg_profiles, ini_params,
radius_mask) are called directly from
:mod:`arealmshbm.pipeline.step1_runners`.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""


# Lazy top-level re-exports (PEP 562): eager imports of the three step
# packages cost a step-0-only process ~0.88 s of import graph it never
# uses. The ``find_spec`` fallback in ``__getattr__`` preserves the
# submodule spelling (``arealmshbm.data_io``) the eager imports gave.

_LAZY_EXPORTS = {
    "Step0Config": "step0_pipeline",
    "Step0Pipeline": "step0_pipeline",
    "Step0Inputs": "step0_pipeline",
    "Step0Result": "step0_pipeline",
    "Step2Config": "step2_pipeline",
    "Step2Pipeline": "step2_pipeline",
    "Step2Inputs": "step2_pipeline",
    "Step2Result": "step2_pipeline",
    "Step3Config": "step3_pipeline",
    "Step3Pipeline": "step3_pipeline",
    "Step3Inputs": "step3_pipeline",
    "Step3Result": "step3_pipeline",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    """Resolve a lazy step re-export, or a submodule, on first access."""
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
