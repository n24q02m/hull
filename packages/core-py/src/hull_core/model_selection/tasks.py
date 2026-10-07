"""TaskProfile registry: map an application's task -> preferred sources + constraints.

Source order (per the 22/09 directive): specialized per-task boards FIRST,
aggregate boards SECOND, OpenRouter as the constraint/tiebreak layer. Every
source name declared here resolves in ``SOURCE_REGISTRY`` — live fetchers,
InHouseAnchorSource sets, or an UnimplementedSource recording a probed gap
(medarena). Consumers may still override individual sources via
``candidates(..., sources={...})`` plugin maps.

Ported from web_core.model_selection.tasks v2.10.6. The ``embedding`` and
``rerank`` profiles intentionally have empty ``aggregate_sources``: embedders
and rerankers mostly do not route through OpenRouter, so the backbone is
expected to be thin/empty there (see ``hull_core.model_selection`` docs for the
auto-promotion policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    specialized_sources: tuple[str, ...] = ()  # tried first, higher weight
    aggregate_sources: tuple[str, ...] = ()  # merge/fallback layer
    constraints: Constraints = field(default_factory=Constraints)
    quality_weight: float = 0.7  # w_spec when specialized covers >=80% of candidates; else 0.5
    # Token mix (input, output, cache_read) measured on the app — do NOT hardcode
    # a generic one: KP ingestion is input-heavy, Aiora chat is ~ AA 7:2:1
    # (cache:input:output).
    token_mix: tuple[float, float, float] = (0.2, 0.1, 0.7)
    expected_prompt_tokens: int = 0  # to apply OR pricing.overrides by min_prompt_tokens


# --- Registry ----------------------------------------------------------------

TASKS: dict[str, TaskProfile] = {
    "translation": TaskProfile(
        name="translation",
        specialized_sources=("wmt24pp", "flores_speakleash"),
        aggregate_sources=("benchlm", "arena", "artificial_analysis", "vals_index"),
        constraints=Constraints(min_context=8_192),
        token_mix=(0.7, 0.3, 0.0),  # ingestion input-heavy
    ),
    "story-gen": TaskProfile(
        name="story-gen",
        specialized_sources=("eqbench_creative_v3", "eqbench_longform", "eqbench4"),
        aggregate_sources=("benchlm", "arena", "artificial_analysis", "vals_index"),
        constraints=Constraints(min_context=32_768),
        token_mix=(0.4, 0.6, 0.0),
    ),
    "embedding": TaskProfile(
        name="embedding",
        specialized_sources=("mteb_classification", "mteb_retrieval", "mteb_sts"),
        # Embedders do not route through OpenRouter -> the OR backbone returns
        # thin/empty for now; the profile stays so consumers can supply their
        # own sources + join keys via a ``sources`` override.
        aggregate_sources=(),
        constraints=Constraints(),
    ),
    "rerank": TaskProfile(
        name="rerank",
        specialized_sources=("mteb_reranking",),
        aggregate_sources=(),
        constraints=Constraints(),
    ),
    "manga-text": TaskProfile(
        name="manga-text",
        specialized_sources=("manga109_v2026", "mangavqa"),
        aggregate_sources=("benchlm", "arena", "ocrbench"),
        constraints=Constraints(min_context=16_384, required_input_modality="image"),
        token_mix=(0.6, 0.4, 0.0),
    ),
    "healthcare-advice": TaskProfile(
        name="healthcare-advice",
        specialized_sources=("aa_healthcare_index", "healthbench"),
        aggregate_sources=(
            "benchlm",
            "artificial_analysis",
            "medhelm",
            "medarena",  # UnimplementedSource — recorded unimplemented_access in snapshots
            "vals_medscribe",
            "open_medical_llm",
            "bridge",
        ),
        constraints=Constraints(min_context=32_768, min_uptime_1d=99.0),
        token_mix=(0.2, 0.1, 0.7),  # chat ~ AA 7:2:1 cache:input:output
    ),
    "aqi-advice": TaskProfile(
        name="aqi-advice",
        # GAP: no dedicated AQI board -> proxied by healthcare; backlog =
        # in-house eval on synthetic AQI readings.
        specialized_sources=("aa_healthcare_index", "healthbench"),
        aggregate_sources=("benchlm", "arena", "artificial_analysis", "vals_index"),
        constraints=Constraints(min_context=16_384),
    ),
    "classification": TaskProfile(
        name="classification",
        specialized_sources=("mteb_classification", "aiora_triage_eval"),
        aggregate_sources=("benchlm", "artificial_analysis", "arena", "vals_index"),
        constraints=Constraints(require_structured_outputs=True),
        token_mix=(0.8, 0.2, 0.0),
    ),
    "agentic": TaskProfile(
        name="agentic",
        specialized_sources=("tau2_bench_or", "bfcl", "vals_cua_bench"),
        aggregate_sources=("benchlm", "artificial_analysis", "gaia", "vals_index"),
        constraints=Constraints(min_context=32_768, require_tools=True, min_uptime_1d=99.0),
        token_mix=(0.5, 0.4, 0.1),
    ),
    "vision": TaskProfile(
        name="vision",
        specialized_sources=("ocrbench",),
        aggregate_sources=("benchlm", "arena", "artificial_analysis"),
        constraints=Constraints(required_input_modality="image"),
    ),
    # Spec 2026-09-28 (D-KP6, phase H): permissive/NSFW selection rides the UGI
    # leaderboard as its specialized board; ``aa_agentic_index`` (embedded in
    # OR backbone rows) is the aggregate proxy. Refusal chains and the
    # rating->permissive downgrade are consumer policy (KlPrism), not this
    # module's concern.
    "permissive": TaskProfile(
        name="permissive",
        specialized_sources=("ugi",),
        aggregate_sources=("aa_agentic_index",),
        constraints=Constraints(min_context=16_384),
        quality_weight=0.8,  # UGI is the signal that matters for this profile
        token_mix=(0.5, 0.5, 0.0),  # generation-shaped, like story-gen
    ),
}


# ``get_task`` also resolves hull's shorter task names (``embed`` ->
# ``embedding``); ported profile names stay canonical.
TASK_ALIASES: dict[str, str] = {"embed": "embedding"}


def get_task(task: TaskProfile | str) -> TaskProfile:
    """Resolve a TaskProfile from the registry by name; KeyError if unregistered."""
    if isinstance(task, TaskProfile):
        return task
    return TASKS[TASK_ALIASES.get(task, task)]
