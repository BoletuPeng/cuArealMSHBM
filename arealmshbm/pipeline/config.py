"""config.py — reader + validator for ``pipeline_config.json`` (schema v2).

Schema v2 (current). The driver consumes one project-level JSON that is
the **sole authoritative source** for every algorithm-relevant
hyperparameter. There is no Knobs-level default fallback, no
inference-from-environment, no "missing field → reasonable guess." The
project creator (studio app, or hand-written for a manual run) owns the
contract: ship a complete config or the parser raises.

Top-level fields are campaign-spanning (mode, variant, cohort weights,
backends). Per-step internal knobs live under nested objects
``step0`` / ``step1`` / ``step2`` / ``step3`` — each is a small frozen
dataclass mirroring the ``lib/hyperparameters/step{N}.json`` shape.

Mode-driven block requirement
-----------------------------
Each mode requires a specific subset of step blocks (matching
``lib/hyperparameters/modes.json``'s ``includes`` field):

  * ``modeA_single`` / ``modeA_batch`` — step0 + step1 + step3
  * ``modeB_train_prior``              — step0 + step1 + step2 + step3

Including a block the mode doesn't run (e.g. a ``step2`` block in a
Mode A config) raises as an unknown key. Omitting a required block
raises with a "missing required block" message that names both the
block and the mode.

Mode A reads the group prior from ``<project>/priors/<variant>/
beta<beta_scalar>/Params_Final.mat``. Mode B writes it there via step 2.

The matching authoritative parameter catalog lives at
``lib/hyperparameters/`` — every knob below has a corresponding
entry there with type/range/default/description metadata. **The catalog
``default`` field is a recommended starting value for the project
creator's UI, NOT an implicit pipeline fallback.**

Strict input validation
-----------------------
The parser enforces four safety invariants:

  1. **Every field required** — the config must explicitly set every
     algorithm knob. Missing top-level fields and missing fields inside
     each step block both raise with ``pipeline_config.json: missing
     required key '<block>.<key>'`` (or the bare key for top-level).

  2. **Unknown-key rejection** — a typo like ``"smooth_simga"`` raises
     instead of being silently dropped. Applied at both the top level
     (mode-aware: Mode A configs can't include a step2 block) and
     inside every step block.

  3. **Strict bool/int/float type guards** — JSON ``true`` does NOT
     coerce to ``1`` in a numeric field; a stringified ``"false"``
     does NOT coerce to ``True`` in a bool field. ``bool`` being a
     subclass of ``int`` in Python makes these footguns easy; the
     ``_need_in`` helper rejects them explicitly with a per-field error
     prefixed by the block name.

  4. **Strict enum type matching** — ``num_clusters: 100.0`` is rejected
     even though ``100.0 == 100``; the allowed-tuple's element type is
     the contract.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional


PipelineMode = Literal["modeA_single", "modeA_batch", "modeB_train_prior"]
PipelineVariant = Literal["gMSHBM", "cMSHBM", "dMSHBM"]
_VALID_MODES = ("modeA_single", "modeA_batch", "modeB_train_prior")
_VALID_VARIANTS = ("gMSHBM", "cMSHBM", "dMSHBM")
_VALID_BACKENDS_STEP012 = ("cpu", "gpu")
_VALID_BACKENDS_STEP3 = ("cpu", "gpu_elambda", "gpu_full")
_VALID_NUM_CLUSTERS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
_VALID_EMB_FORMATS = ("npy", "mat", "both")
_VALID_DTYPES = ("float32", "float64")
_VALID_BOLD_CACHE_MODES = ("auto", "eager_bitpacked", "stream")
# Enum-of-one until the ``split_flag != "0"`` path lands in
# ``generate_profiles/profiles.py`` (currently raises NotImplementedError).
_VALID_SPLIT_FLAGS = ("0",)


# ─────────────────────────────────────────────────────────────────────
# Per-step knob bundles
#
# None of the Knobs classes carry default values: under schema v2 the
# project creator is responsible for supplying every algorithm knob in
# pipeline_config.json. The parser fills these dataclasses from explicit
# user input; constructing ``Step{N}Knobs()`` with no args raises
# TypeError, which is the correct behavior — these are typed containers,
# not implicit-default providers.
#
# Recommended starting values for the project creator's UI live in the
# matching ``lib/hyperparameters/step{N}.json`` ``default`` field.
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Step0Knobs:
    """Step 0 internal knobs (mirror Step0Config algorithm fields).

    ``enable_tf32`` is the step-scoped cuBLAS TF32 toggle (was
    ``enable_tf32_step0`` at the top level until 2026-06; relocated
    into this block so Mode A configs — which still include step0 —
    naturally carry it, and so the catalog's ``step0.json`` owns the
    field end-to-end rather than referring to a top-level field).
    Only ``fc_similarity`` dispatches through cuBLAS sgemm inside step0
    — other GPU leaves are unaffected. Has effect only when
    ``backend_step0 == 'gpu'``.
    """
    sub_FC: int
    sub_verts: int
    block_a: int
    block_b: int
    smooth_sigma: float
    K_hop: int
    watershed_steps: int
    watershed_frac: float
    downsample: float
    emb_output_format: Literal["npy", "mat", "both"]
    save_geodesic_distance: bool
    save_edge_density: bool
    enable_tf32: bool
    # NB: Literal-typed fields are runtime-checked by ``_enum_in`` at
    # parse time (not by dataclass machinery). Constructing
    # ``Step0Knobs(emb_output_format="invalid", ...)`` directly bypasses
    # the check — only the parser is the gate. mypy/pyright catches
    # static-typo cases.


@dataclass(frozen=True)
class Step1Knobs:
    """Step 1 leaf-default knobs (mirror step1_runners.* kwargs).

    Inclusion criterion: precision / iteration / radius knobs the user
    is plausibly tuning between runs. Kernel-internal precision dtypes
    (``run_generate_profiles.profile_dtype_reduce`` and
    ``run_radius_mask.dtype``, both pinned at fp32 by validated CBIG
    convention) are intentionally NOT exposed — the runners' positional
    defaults stay authoritative for them.
    """
    threshold: float
    split_flag: Literal["0"]
    profile_dtype: Literal["float32", "float64"]
    reduction_dtype: Literal["float32", "float64"]
    radius_mask_radius_mm: float


@dataclass(frozen=True)
class Step2Knobs:
    """Step 2 internal knobs (mirror Step2Config algorithm fields).

    ``enable_tf32`` is the step-scoped cuBLAS TF32 toggle (relocated
    from top-level ``enable_tf32_step2`` to this block in 2026-06).
    Because the entire block is forbidden in Mode A configs, the
    "Mode A user writes enable_tf32_step2: true silently does nothing"
    confusion is gone by construction. Has effect only when
    ``backend_step2 == 'gpu'``.
    """
    max_iter_inter: int
    max_iter_intra_em: int
    max_iter_em: int
    max_iter_m: int
    max_iter_intra_var: int
    epsilon: float
    intra_em_convergence_eps: float
    inter_convergence_eps: float
    em_convergence_eps: float
    bold_cache_mode: Literal["auto", "eager_bitpacked", "stream"]
    gpu_cache_safety_margin_gb: float
    verbose: bool
    enable_tf32: bool


@dataclass(frozen=True)
class Step3Knobs:
    """Step 3 internal knobs (mirror Step3Config algorithm fields).

    ``connect_th`` is the only nullable field. The recommended practice
    for the project creator is to write the explicit canonical float
    for the variant — 15.0 for gMSHBM, 0.0 for cMSHBM, ``null`` for
    dMSHBM (which doesn't read it). Step3Config's ``__post_init__``
    resolves ``None`` to the variant default; the unified-driver path
    reaches that branch ONLY when the project creator wrote ``null``
    explicitly (the in-tree sample projects ship explicit floats for
    gMSHBM / cMSHBM and reserve ``null`` for dMSHBM, so the resolution
    branch is exercised only in dMSHBM runs and per-step API tests).
    """
    connect_th: Optional[float]
    epsilon: float
    max_iter_intra_em: int
    max_iter_em: int
    max_iter_lambda: int
    max_iter_m: int
    max_iter_comp: int
    cMSHBM_isolated_component_min_size: int


# ─────────────────────────────────────────────────────────────────────
# Top-level PipelineConfig
#
# Every field is required — no Knobs default_factory, no enable_tf32
# default. step2 is Optional[Step2Knobs] only because Mode A doesn't
# run step2; Mode A configs MUST NOT include a step2 block, and the
# parser sets this to ``None`` in that case. Mode B configs MUST
# include step2; the parser sets a non-None Step2Knobs there.
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PipelineConfig:
    schema_version: str
    mode: PipelineMode
    variant: PipelineVariant
    num_clusters: int
    beta_scalar: int
    w: int
    c: int
    n_grad_components: int
    backend_step0: str
    backend_step1: str
    backend_step2: str
    backend_step3: str
    # Per-step internal knob bundles. Required by mode:
    #   Mode A — step0 + step1 + step3 (step2 is None)
    #   Mode B — step0 + step1 + step2 + step3 (all non-None)
    # cuBLAS TF32 toggles live INSIDE the relevant step block
    # (``step0.enable_tf32`` and ``step2.enable_tf32``) — they used to
    # be top-level until 2026-06, but the relocation lets Mode A
    # configs naturally not carry ``enable_tf32_step2`` and lines the
    # field up with the catalog file that owns its docs (step{N}.json).
    step0: Step0Knobs
    step1: Step1Knobs
    step2: Optional[Step2Knobs]
    step3: Step3Knobs

    @property
    def is_mode_a(self) -> bool:
        return self.mode in ("modeA_single", "modeA_batch")


# Frozen key-sets for unknown-key rejection. Per-block sets are built
# from each Knobs dataclass's field list so adding a knob automatically
# expands the allow-set.
_STEP0_KEYS = frozenset(Step0Knobs.__dataclass_fields__.keys())
_STEP1_KEYS = frozenset(Step1Knobs.__dataclass_fields__.keys())
_STEP2_KEYS = frozenset(Step2Knobs.__dataclass_fields__.keys())
_STEP3_KEYS = frozenset(Step3Knobs.__dataclass_fields__.keys())

# Campaign-spanning top-level keys, present in every mode. step{N}
# block names are added per mode via _MODE_REQUIRED_BLOCKS. The
# enable_tf32 toggles live INSIDE step0 / step2 blocks (relocated
# 2026-06), not at top level.
_TOP_LEVEL_BASE_KEYS = frozenset({
    "schema_version", "mode", "variant", "num_clusters", "beta_scalar",
    "w", "c", "n_grad_components",
    "backend_step0", "backend_step1", "backend_step2", "backend_step3",
})

# Mode → required step blocks. Mirrors lib/hyperparameters/modes.json's
# per-mode ``includes`` field (minus ``project_basics`` which is the
# top-level itself). Including a block the mode doesn't run is treated
# as an unknown top-level key; omitting a required block raises a
# "missing required block" error that names the mode.
_MODE_REQUIRED_BLOCKS: Dict[str, frozenset] = {
    "modeA_single":      frozenset({"step0", "step1", "step3"}),
    "modeA_batch":       frozenset({"step0", "step1", "step3"}),
    "modeB_train_prior": frozenset({"step0", "step1", "step2", "step3"}),
}


# ─────────────────────────────────────────────────────────────────────
# Parser helpers
#
# Schema v2 requires every key — there is no default-fallback path.
# ``_need_in`` / ``_enum_in`` are the full-bodied helpers (nested or
# top-level via ``block=""``); ``_need`` / ``_enum`` are thin top-level
# aliases for callsite readability. Adding a new strict-type rule lives
# in exactly one place (``_need_in`` / ``_enum_in``).
# ─────────────────────────────────────────────────────────────────────
def _need_in(
    d: Dict[str, Any],
    block: str,
    key: str,
    type_: type,
    *,
    positive: bool = False,
) -> Any:
    """Strict typed reader. Raises on missing key OR wrong type.

    Correctness invariants:
      * Missing key → raises with ``missing required key '<block>.<key>'``
        (or bare ``'<key>'`` when block is empty).
      * bool is NOT silently accepted as int/float (bool is an ``int``
        subclass in Python; without this guard ``true`` would parse as 1).
      * int IS auto-widened to float so JSON ``5`` satisfies a float field.
      * type mismatches raise with a ``pipeline_config.json: <block>.<key>
        ...`` prefix so the error names exactly which field is wrong.

    Pass ``block=""`` for top-level fields; the error message then reads
    ``pipeline_config.json: <key> ...`` matching the top-level style.
    """
    prefix = f"{block}.{key}" if block else key
    if key not in d:
        raise ValueError(
            f"pipeline_config.json: missing required key {prefix!r}"
        )
    v = d[key]
    # Single bool-handling block: if target is bool, demand bool;
    # otherwise (int/float/str/...) reject any bool to prevent the
    # silent True→1 / False→0 coercion via the int subclass relationship.
    if type_ is bool:
        if not isinstance(v, bool):
            raise ValueError(
                f"pipeline_config.json: {prefix} must be bool "
                f"(got {type(v).__name__})"
            )
    elif isinstance(v, bool):
        name = "number" if type_ is float else type_.__name__
        raise ValueError(
            f"pipeline_config.json: {prefix} must be {name} (got bool)"
        )
    if type_ is float and isinstance(v, int):
        v = float(v)
    if not isinstance(v, type_):
        raise ValueError(
            f"pipeline_config.json: {prefix} must be {type_.__name__} "
            f"(got {type(v).__name__})"
        )
    if positive and v <= 0:
        raise ValueError(
            f"pipeline_config.json: {prefix} must be positive (got {v!r})"
        )
    return v


def _enum_in(
    d: Dict[str, Any],
    block: str,
    key: str,
    allowed: tuple,
) -> Any:
    """Strict-typed enum reader. Raises on missing key OR wrong type/value.

    Pythonic loose numeric equality (``100 == 100.0``) would otherwise
    let a float-typed value satisfy an int enum and propagate into
    downstream layout / schaefer-resolution code that expects ``int``.
    Strict type matching against ``allowed[0]``'s type prevents that;
    bool is rejected as int even though ``isinstance(True, int)`` is True.

    Pass ``block=""`` for top-level fields; matches ``_need_in``.
    """
    prefix = f"{block}.{key}" if block else key
    if key not in d:
        raise ValueError(
            f"pipeline_config.json: missing required key {prefix!r}"
        )
    v = d[key]
    if allowed:
        expected_type = type(allowed[0])
        if expected_type is int and isinstance(v, bool):
            raise ValueError(
                f"pipeline_config.json: {prefix} must be int "
                f"(got bool); allowed = {allowed}"
            )
        if not isinstance(v, expected_type):
            raise ValueError(
                f"pipeline_config.json: {prefix} must be "
                f"{expected_type.__name__} (got {type(v).__name__}); "
                f"allowed = {allowed}"
            )
    if v not in allowed:
        raise ValueError(
            f"pipeline_config.json: {prefix} must be one of {allowed} "
            f"(got {v!r})"
        )
    return v


def _need(d: Dict[str, Any], key: str, type_: type, *, positive: bool = False) -> Any:
    """Top-level required reader — thin alias for ``_need_in(d, "", ...)``."""
    return _need_in(d, "", key, type_, positive=positive)


def _enum(d: Dict[str, Any], key: str, allowed: tuple) -> Any:
    """Top-level enum reader — thin alias for ``_enum_in(d, "", ...)``."""
    return _enum_in(d, "", key, allowed)


def _check_unknown_keys(
    raw: Dict[str, Any], allowed: frozenset, label: str
) -> None:
    """Reject keys outside ``allowed``. ``label`` names the scope in errors."""
    extra = set(raw.keys()) - allowed
    if extra:
        raise ValueError(
            f"pipeline_config.json: unknown key(s) in {label}: "
            f"{sorted(extra)}. Allowed: {sorted(allowed)}."
        )


# ─────────────────────────────────────────────────────────────────────
# Per-step parsers
#
# Each parser expects a fully-populated dict (every field present) and
# raises on missing or extraneous keys. The mode-level presence check
# (which blocks are required) happens in ``read_pipeline_config`` before
# these are called — by the time a parser runs, the block has been
# proven required-and-present.
# ─────────────────────────────────────────────────────────────────────
def _parse_step0(raw: Dict[str, Any]) -> Step0Knobs:
    if not isinstance(raw, dict):
        raise ValueError(
            f"pipeline_config.json: step0 must be an object "
            f"(got {type(raw).__name__})"
        )
    _check_unknown_keys(raw, _STEP0_KEYS, "step0")
    return Step0Knobs(
        sub_FC=_need_in(raw, "step0", "sub_FC", int),
        sub_verts=_need_in(raw, "step0", "sub_verts", int),
        block_a=_need_in(raw, "step0", "block_a", int),
        block_b=_need_in(raw, "step0", "block_b", int),
        smooth_sigma=_need_in(raw, "step0", "smooth_sigma", float),
        K_hop=_need_in(raw, "step0", "K_hop", int),
        watershed_steps=_need_in(raw, "step0", "watershed_steps", int),
        watershed_frac=_need_in(raw, "step0", "watershed_frac", float),
        downsample=_need_in(raw, "step0", "downsample", float),
        emb_output_format=_enum_in(raw, "step0", "emb_output_format",
                                   _VALID_EMB_FORMATS),
        save_geodesic_distance=_need_in(raw, "step0", "save_geodesic_distance",
                                        bool),
        save_edge_density=_need_in(raw, "step0", "save_edge_density", bool),
        enable_tf32=_need_in(raw, "step0", "enable_tf32", bool),
    )


def _parse_step1(raw: Dict[str, Any]) -> Step1Knobs:
    if not isinstance(raw, dict):
        raise ValueError(
            f"pipeline_config.json: step1 must be an object "
            f"(got {type(raw).__name__})"
        )
    _check_unknown_keys(raw, _STEP1_KEYS, "step1")
    return Step1Knobs(
        threshold=_need_in(raw, "step1", "threshold", float),
        split_flag=_enum_in(raw, "step1", "split_flag", _VALID_SPLIT_FLAGS),
        profile_dtype=_enum_in(raw, "step1", "profile_dtype", _VALID_DTYPES),
        reduction_dtype=_enum_in(raw, "step1", "reduction_dtype", _VALID_DTYPES),
        radius_mask_radius_mm=_need_in(raw, "step1", "radius_mask_radius_mm",
                                       float),
    )


def _parse_step2(raw: Dict[str, Any]) -> Step2Knobs:
    if not isinstance(raw, dict):
        raise ValueError(
            f"pipeline_config.json: step2 must be an object "
            f"(got {type(raw).__name__})"
        )
    _check_unknown_keys(raw, _STEP2_KEYS, "step2")
    return Step2Knobs(
        max_iter_inter=_need_in(raw, "step2", "max_iter_inter", int),
        max_iter_intra_em=_need_in(raw, "step2", "max_iter_intra_em", int),
        max_iter_em=_need_in(raw, "step2", "max_iter_em", int),
        max_iter_m=_need_in(raw, "step2", "max_iter_m", int),
        max_iter_intra_var=_need_in(raw, "step2", "max_iter_intra_var", int),
        epsilon=_need_in(raw, "step2", "epsilon", float),
        intra_em_convergence_eps=_need_in(
            raw, "step2", "intra_em_convergence_eps", float),
        inter_convergence_eps=_need_in(
            raw, "step2", "inter_convergence_eps", float),
        em_convergence_eps=_need_in(
            raw, "step2", "em_convergence_eps", float),
        bold_cache_mode=_enum_in(raw, "step2", "bold_cache_mode",
                                 _VALID_BOLD_CACHE_MODES),
        gpu_cache_safety_margin_gb=_need_in(
            raw, "step2", "gpu_cache_safety_margin_gb", float),
        verbose=_need_in(raw, "step2", "verbose", bool),
        enable_tf32=_need_in(raw, "step2", "enable_tf32", bool),
    )


def _parse_step3(raw: Dict[str, Any]) -> Step3Knobs:
    if not isinstance(raw, dict):
        raise ValueError(
            f"pipeline_config.json: step3 must be an object "
            f"(got {type(raw).__name__})"
        )
    _check_unknown_keys(raw, _STEP3_KEYS, "step3")
    # connect_th has a (number | null) contract — the key is required
    # but its value can be null (dMSHBM doesn't read it; gMSHBM/cMSHBM
    # expect a number). Handle the polymorphism by hand rather than via
    # _need_in (which assumes a single type_).
    if "connect_th" not in raw:
        raise ValueError(
            "pipeline_config.json: missing required key 'step3.connect_th'"
        )
    ct = raw["connect_th"]
    if ct is None:
        pass
    elif isinstance(ct, bool):
        raise ValueError(
            "pipeline_config.json: step3.connect_th must be number or null "
            "(got bool)"
        )
    elif isinstance(ct, int):
        ct = float(ct)
    elif not isinstance(ct, float):
        raise ValueError(
            f"pipeline_config.json: step3.connect_th must be number or null "
            f"(got {type(ct).__name__})"
        )
    return Step3Knobs(
        connect_th=ct,
        epsilon=_need_in(raw, "step3", "epsilon", float),
        max_iter_intra_em=_need_in(raw, "step3", "max_iter_intra_em", int),
        max_iter_em=_need_in(raw, "step3", "max_iter_em", int),
        max_iter_lambda=_need_in(raw, "step3", "max_iter_lambda", int),
        max_iter_m=_need_in(raw, "step3", "max_iter_m", int),
        max_iter_comp=_need_in(raw, "step3", "max_iter_comp", int),
        cMSHBM_isolated_component_min_size=_need_in(
            raw, "step3", "cMSHBM_isolated_component_min_size", int),
    )


# ─────────────────────────────────────────────────────────────────────
# Per-step range / positivity validators
#
# Defense in depth alongside Step{N}Config.__post_init__: these
# parser-side validators fire at PARSE TIME so the project creator
# sees range errors immediately when reading the config file, rather
# than mid-pipeline when the driver builds a Step{N}Config and
# __post_init__ rejects.
#
# Reviewer round 2 flagged the "edit two places to add a range check"
# duplication as drift risk and suggested a shared helper. We
# considered it but kept the small (~10 lines/step) duplication
# because the cleanest shared-helper factoring required
# pipeline/config.py to import range-check functions from
# step{N}_pipeline/, coupling the package layers in a way that hurts
# more than the duplication saves. The two call sites of each check
# are adjacent and the maintenance task is "change two nearby
# functions when adding a range check" — low cognitive load.
#
# Note the threshold range check below is HARD — it's documented as
# an exception to the advisory ``range_policy`` in
# lib/hyperparameters/step1.json (top-fraction outside [0,1] is
# meaningless for FC binarization, not a user-tuning question).
# ─────────────────────────────────────────────────────────────────────
def _validate_step0(k: Step0Knobs) -> None:
    # Enum check on emb_output_format already fired at parse time via
    # ``_enum_in`` — this validator only handles range / positivity.
    for name in ("sub_FC", "sub_verts", "block_a", "block_b",
                 "K_hop", "watershed_steps"):
        if getattr(k, name) <= 0:
            raise ValueError(f"pipeline_config.json: step0.{name} must be positive")
    for name in ("smooth_sigma", "watershed_frac", "downsample"):
        if getattr(k, name) <= 0.0:
            raise ValueError(f"pipeline_config.json: step0.{name} must be positive")


def _validate_step1(k: Step1Knobs) -> None:
    # Enum checks (split_flag / profile_dtype / reduction_dtype) already
    # fired at parse time via ``_enum_in``.
    if not (0.0 <= k.threshold <= 1.0):
        raise ValueError(
            f"pipeline_config.json: step1.threshold must be in [0, 1] (got {k.threshold})"
        )
    if k.radius_mask_radius_mm <= 0.0:
        raise ValueError(
            f"pipeline_config.json: step1.radius_mask_radius_mm must be positive"
        )


def _validate_step2(k: Step2Knobs) -> None:
    # Enum check on bold_cache_mode already fired at parse time via
    # ``_enum_in`` — this validator only handles range / positivity.
    for name in ("max_iter_inter", "max_iter_intra_em", "max_iter_em",
                 "max_iter_m", "max_iter_intra_var"):
        if getattr(k, name) <= 0:
            raise ValueError(f"pipeline_config.json: step2.{name} must be positive")
    for name in ("epsilon", "intra_em_convergence_eps", "inter_convergence_eps",
                 "em_convergence_eps"):
        if getattr(k, name) <= 0.0:
            raise ValueError(f"pipeline_config.json: step2.{name} must be positive")
    if k.gpu_cache_safety_margin_gb < 0.0:
        raise ValueError(
            f"pipeline_config.json: step2.gpu_cache_safety_margin_gb must be >= 0"
        )


def _validate_step3(k: Step3Knobs) -> None:
    if k.connect_th is not None and k.connect_th < 0.0:
        raise ValueError(
            f"pipeline_config.json: step3.connect_th must be >= 0 (got {k.connect_th})"
        )
    if k.epsilon <= 0.0:
        raise ValueError("pipeline_config.json: step3.epsilon must be positive")
    for name in ("max_iter_intra_em", "max_iter_em", "max_iter_lambda",
                 "max_iter_m", "max_iter_comp"):
        if getattr(k, name) <= 0:
            raise ValueError(f"pipeline_config.json: step3.{name} must be positive")
    if k.cMSHBM_isolated_component_min_size <= 0:
        raise ValueError(
            f"pipeline_config.json: step3.cMSHBM_isolated_component_min_size "
            f"must be positive"
        )


def read_pipeline_config(path: Path | str) -> PipelineConfig:
    """Parse + validate ``pipeline_config.json`` (schema v2).

    Every algorithm-relevant knob is required. Every step block required
    by the mode is required; including an unused block raises. There is
    NO default-fallback path — the project creator must produce a
    complete config or this function raises with a message that names
    the missing field exactly.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pipeline_config.json not found at {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"pipeline_config.json: top-level must be an object "
            f"(got {type(raw).__name__})"
        )

    sv = raw.get("schema_version")
    if sv == "1":
        raise ValueError(
            "pipeline_config.json: schema_version '1' is no longer supported. "
            "Migrate to schema_version '2' — every per-step internal knob "
            "must now be nested under step0 / step1 / step2 / step3 objects "
            "AND explicitly set (v2 has no default fallback). See "
            "projects/sample_modeB/pipeline_config.json (Mode B), "
            "projects/sample_modeA_single/pipeline_config.json (Mode A, K=1), "
            "or projects/sample_modeA_batch/pipeline_config.json (Mode A, K>=1) "
            "for complete v2 examples; lib/hyperparameters/step{N}.json "
            "lists the recommended starting value of each field. "
            "v2 also (i) hard-restricts num_clusters to the Schaefer-2018 "
            "atlas resolutions {100, 200, ..., 1000} — v1 silently let step1 "
            "FileNotFoundError on any other N — (ii) rejects unknown keys "
            "at parse time (including underscore-prefixed comment fields "
            "like _comment / _notes / _todo; drop those before migrating), "
            "(iii) is mode-aware about which step blocks must be present "
            "(Mode A configs MUST NOT contain a step2 block; Mode B configs "
            "MUST contain all four), and (iv) moves the cuBLAS TF32 toggles "
            "from top-level ``enable_tf32_step{0,2}`` into the matching step "
            "block as ``step0.enable_tf32`` / ``step2.enable_tf32``."
        )
    if sv != "2":
        raise ValueError(
            f"pipeline_config.json: schema_version must be '2' (got {sv!r})"
        )

    # Mode must be resolved first because it decides which step blocks
    # are part of the allowed top-level key set.
    mode = _enum(raw, "mode", _VALID_MODES)
    required_blocks = _MODE_REQUIRED_BLOCKS[mode]
    allowed_top_level = _TOP_LEVEL_BASE_KEYS | required_blocks
    # Targeted hint for the most likely Mode-A confusion: a user with a
    # Mode B config they're adapting to Mode A may forget to drop the
    # step2 block. The generic unknown-key error names step2 but doesn't
    # explain *why* it's unknown — this targeted check beats the user
    # to the punch with a Mode-A-specific message.
    if (mode in ("modeA_single", "modeA_batch")) and ("step2" in raw):
        raise ValueError(
            f"pipeline_config.json: mode={mode!r} does not run step2; "
            f"remove the step2 block. (Mode A reads a pre-staged group "
            f"prior from project-local "
            f"priors/<variant>/beta<X>/Params_Final.mat. See "
            f"projects/sample_modeA_single/pipeline_config.json or "
            f"projects/sample_modeA_batch/pipeline_config.json for "
            f"Mode A templates.)"
        )
    # Unknown top-level keys (typos like ``step_0``, stale fields from
    # an older campaign config). Reject — the project creator is
    # responsible for a complete and shape-correct config.
    _check_unknown_keys(raw, allowed_top_level, "top level")
    # Missing required step blocks for this mode.
    missing_blocks = [b for b in sorted(required_blocks) if b not in raw]
    if missing_blocks:
        raise ValueError(
            f"pipeline_config.json: mode={mode!r} requires step block(s) "
            f"{missing_blocks} which are missing. Add a complete "
            f"{missing_blocks[0]} object (see lib/hyperparameters/"
            f"{missing_blocks[0]}.json for fields)."
        )

    variant = _enum(raw, "variant", _VALID_VARIANTS)
    num_clusters = _enum(raw, "num_clusters", _VALID_NUM_CLUSTERS)
    beta_scalar = _need(raw, "beta_scalar", int, positive=True)
    w = _need(raw, "w", int)
    c = _need(raw, "c", int)
    n_grad_components = _need(raw, "n_grad_components", int, positive=True)
    if w < 0:
        raise ValueError(f"pipeline_config.json: w must be >= 0 (got {w})")
    if c < 0:
        raise ValueError(f"pipeline_config.json: c must be >= 0 (got {c})")
    backend_step0 = _enum(raw, "backend_step0", _VALID_BACKENDS_STEP012)
    backend_step1 = _enum(raw, "backend_step1", _VALID_BACKENDS_STEP012)
    backend_step2 = _enum(raw, "backend_step2", _VALID_BACKENDS_STEP012)
    backend_step3 = _enum(raw, "backend_step3", _VALID_BACKENDS_STEP3)

    if variant == "cMSHBM" and mode == "modeB_train_prior":
        raise NotImplementedError(
            "pipeline_config.json: cMSHBM is not wired for step2 (Mode B). "
            "Use gMSHBM or dMSHBM for Mode B."
        )

    # Step blocks: parse only those the mode requires. step2 is the
    # only one Mode A omits — that branch sets step2_knobs to None on
    # the PipelineConfig.
    step0 = _parse_step0(raw["step0"])
    step1 = _parse_step1(raw["step1"])
    step2 = _parse_step2(raw["step2"]) if "step2" in required_blocks else None
    step3 = _parse_step3(raw["step3"])
    _validate_step0(step0)
    _validate_step1(step1)
    if step2 is not None:
        _validate_step2(step2)
    _validate_step3(step3)

    return PipelineConfig(
        schema_version=sv,
        mode=mode,  # type: ignore[arg-type]
        variant=variant,  # type: ignore[arg-type]
        num_clusters=num_clusters,
        beta_scalar=beta_scalar,
        w=w, c=c,
        n_grad_components=n_grad_components,
        backend_step0=backend_step0,
        backend_step1=backend_step1,
        backend_step2=backend_step2,
        backend_step3=backend_step3,
        step0=step0, step1=step1, step2=step2, step3=step3,
    )
