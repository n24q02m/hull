"""TaskProfile registry: map an application's task -> preferred sources + constraints.

Source order: specialized per-task boards FIRST, ``LM_GENERAL_SOURCES``
SECOND (for ``lm`` profiles), OpenRouter as the catalog + price + ZDR layer.
Every source name declared here resolves in ``SOURCE_REGISTRY``.

``embedding``, ``rerank``, and ``decision`` intentionally have empty
``aggregate_sources``: they lock directly to their specialized 11/2026 boards
with OpenRouter catalog + ``require_zdr`` filtering. ``or_usage`` is disabled
from all default ``TASKS`` (never used for quality or tie-breaking).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 11/2026 general LLM leaderboard tier (Tier 2 fallback ONLY for lm profiles).
LM_GENERAL_SOURCES: tuple[str, ...] = (
    "artificial_analysis",
    "epoch_eci",
    "arena",
    "livebench_v2",
    "scale_seal",
    "benchlm",
)


@dataclass(frozen=True)
class Constraints:
    """Hard prefilter before pareto."""

    min_context: int = 0  # minimum context_length
    required_input_modality: str | None = None  # e.g. "image" for vision/manga
    require_tools: bool = False  # OR supported_parameters contains "tools"
    require_structured_outputs: bool = False  # OR supported_parameters contains "structured_outputs"
    require_zdr: bool = False  # needs an endpoint tagged zdr (fail-closed: unknown = drop)
    min_uptime_1d: float | None = None  # SLO uptime; None = do not filter
    max_cost_1m: float | None = None  # USD / 1M blended tokens


@dataclass(frozen=True)
class TaskProfile:
    """A task's profile: preferred sources + constraints + measured token mix."""

    name: str
    specialized_sources: tuple[str, ...] = ()  # Tier 1: tried first
    aggregate_sources: tuple[str, ...] = ()  # Tier 2 fallback ONLY (lm_general for lm profiles)
    usage_sources: tuple[str, ...] = ()  # Disabled by default; optional legacy tie-break override only
    constraints: Constraints = field(default_factory=Constraints)
    quality_weight: float = 0.7  # blend weight when two_tier=False and specialized covers >= threshold
    specialized_coverage_threshold: float = 0.8  # minimum specialized coverage fraction for Tier 1
    two_tier: bool = False  # True: Tier 1 specialized when coverage >= threshold -> Tier 2 aggregate_sources
    # Token mix (input, output, cache_read) measured on the app.
    token_mix: tuple[float, float, float] = (0.2, 0.1, 0.7)
    expected_prompt_tokens: int = 0  # to apply OR pricing.overrides by min_prompt_tokens


# --- Registry ----------------------------------------------------------------

TASKS: dict[str, TaskProfile] = {
    "translation": TaskProfile(
        name="translation",
        specialized_sources=("wmt25_gmtr", "wmt24pp"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(min_context=8_192),
        token_mix=(0.7, 0.3, 0.0),
    ),
    "story-gen": TaskProfile(
        name="story-gen",
        specialized_sources=("eqbench_creative_v3", "eqbench_longform", "eqbench4"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(min_context=32_768),
        token_mix=(0.4, 0.6, 0.0),
    ),
    "embedding": TaskProfile(
        name="embedding",
        specialized_sources=("rteb_holdout", "mteb_v2", "vidore_v3"),
        aggregate_sources=(),
        constraints=Constraints(require_zdr=True),
        token_mix=(1.0, 0.0, 0.0),
    ),
    "rerank": TaskProfile(
        name="rerank",
        specialized_sources=("mteb_v2_crossencoder", "air_bench", "agentset_rerank"),
        aggregate_sources=(),
        constraints=Constraints(require_zdr=True),
        token_mix=(1.0, 0.0, 0.0),
    ),
    "manga-text": TaskProfile(
        name="manga-text",
        specialized_sources=("openvlm", "ocrbench", "manga109_v2026", "mangavqa"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(min_context=16_384, required_input_modality="image"),
        token_mix=(0.6, 0.4, 0.0),
    ),
    "healthcare-advice": TaskProfile(
        name="healthcare-advice",
        specialized_sources=("aa_healthcare_index", "healthbench", "vals_medscribe", "vectara_hhem"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(min_context=32_768, min_uptime_1d=99.0, require_zdr=True),
        token_mix=(0.2, 0.1, 0.7),
    ),
    "aqi-advice": TaskProfile(
        name="aqi-advice",
        specialized_sources=("aa_healthcare_index", "healthbench", "vals_medscribe", "vectara_hhem"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(min_context=16_384, require_zdr=True),
        token_mix=(0.2, 0.1, 0.7),
    ),
    "classification": TaskProfile(
        name="classification",
        specialized_sources=("aiora_triage_eval", "vals_legal_bench"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(require_structured_outputs=True, require_zdr=True),
        token_mix=(0.8, 0.2, 0.0),
    ),
    "agentic": TaskProfile(
        name="agentic",
        specialized_sources=("arena_agent", "livebench_v2_agentic", "scale_agentic"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(min_context=32_768, require_tools=True, min_uptime_1d=99.0),
        token_mix=(0.5, 0.4, 0.1),
    ),
    "vision": TaskProfile(
        name="vision",
        specialized_sources=("openvlm", "ocrbench", "manga109_v2026", "mangavqa"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(required_input_modality="image"),
    ),
    "permissive": TaskProfile(
        name="permissive",
        specialized_sources=("ugi",),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        constraints=Constraints(min_context=16_384),
        quality_weight=0.8,
        token_mix=(0.5, 0.5, 0.0),
    ),
    "decision": TaskProfile(
        name="decision",
        specialized_sources=("aa_decision_index_v2_1", "jev_decision_index", "s1mb", "system_one_bench"),
        aggregate_sources=(),
        constraints=Constraints(require_zdr=True),
        token_mix=(0.9, 0.1, 0.0),
    ),
}


TASK_ALIASES: dict[str, str] = {"embed": "embedding"}


def get_task(task: TaskProfile | str) -> TaskProfile:
    """Resolve a TaskProfile from the registry by name; KeyError if unregistered."""
    if isinstance(task, TaskProfile):
        return task
    return TASKS[TASK_ALIASES.get(task, task)]
