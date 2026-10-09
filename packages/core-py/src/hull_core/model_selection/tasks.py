"""TaskProfile registry: map an application's task -> preferred sources + constraints.

Source order (per the 22/09 directive): specialized per-task boards FIRST,
aggregate boards SECOND, OpenRouter as the constraint/tiebreak layer. Every
source name declared here resolves in ``SOURCE_REGISTRY`` — live fetchers,
InHouseAnchorSource sets, or an UnimplementedSource recording a probed gap
(medarena). Consumers may still override individual sources via
``candidates(..., sources={...})`` plugin maps.

Ported from web_core.model_selection.tasks v2.10.6. The ``embedding`` and
``rerank`` profiles intentionally have empty ``aggregate_sources``: the
specialized boards (MTEB + boards-v2 closed-inclusive AgentSet/hindsight) are
the quality axis, while the OpenRouter backbone (now including its
embeddings/rerank catalog segments) is the candidate + price axis — embed/
rerank picks must be OR-servable (2026-10-08 directive). OR usage
(``usage_sources``) is a quadrant tie-break only, never a quality axis.
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
    # OR real-world usage feeds (boards-v2 2026-10-09): folded into
    # ``ModelCandidate.or_task_spend_share`` — quadrant tie-break ONLY, never
    # a board/quality axis; absence neutral. Defaults to ``or_usage`` for
    # every profile (chat, embed and rerank alike).
    usage_sources: tuple[str, ...] = ("or_usage",)
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
        # boards-v2 (2026-10-09): wmt24pp stays but is labeled self-reported
        # (llm-stats mirror, 0/24 verified); wmt25_gmtr is the annual
        # closed-inclusive anchor. flores_speakleash removed (stale 9 months).
        specialized_sources=("wmt24pp", "wmt25_gmtr"),
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
        # boards-v2 (2026-10-09): the MTEB boards (open-biased) are paired
        # with AgentSet Elo (closed-inclusive, stale ~2026-03) so a pick needs
        # >=2 independent boards; hindsight embeddings rides along as an
        # optional third signal (7 models only). Embedding picks MUST be
        # OpenRouter-servable (2026-10-08 directive): OR's embeddings segment
        # lists + prices, the boards qualify.
        specialized_sources=(
            "mteb_classification",
            "mteb_retrieval",
            "mteb_sts",
            "agentset_elo",
            "hindsight_embeddings",
        ),
        aggregate_sources=(),
        constraints=Constraints(),
        token_mix=(1.0, 0.0, 0.0),
    ),
    "rerank": TaskProfile(
        name="rerank",
        # boards-v2 (2026-10-09): AgentSet rerankers (closed-inclusive, stale
        # ~2026-02) + hindsight MRR (directional, n=165) join MTEB reranking,
        # so a strong rerank pick (e.g. cohere-rerank-4-pro) is now reachable;
        # models measured only by weak boards still trail as weak_evidence.
        # Same 2026-10-08 directive: rerank picks must be OR-servable. OR
        # lists a rerank segment but its catalog pricing fields are $0
        # placeholders (OR bills per request, not per token) — those rows
        # count as unpriced, never as free.
        specialized_sources=("mteb_reranking", "agentset_rerank", "hindsight_reranker"),
        aggregate_sources=(),
        constraints=Constraints(),
        token_mix=(1.0, 0.0, 0.0),
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
        # boards-v2 (2026-10-09): mteb_classification removed — it only joins
        # embedding models (8 rows, 0 chat candidates) and classification has
        # no living specialized board; vals_legal_bench is an explicitly-
        # labeled PROXY. aiora_triage_eval remains the in-house anchor until
        # its owner (mnemo) publishes it.
        specialized_sources=("aiora_triage_eval", "vals_legal_bench"),
        aggregate_sources=("benchlm", "artificial_analysis", "arena", "vals_index"),
        constraints=Constraints(require_structured_outputs=True),
        token_mix=(0.8, 0.2, 0.0),
    ),
    "agentic": TaskProfile(
        name="agentic",
        # boards-v2 (2026-10-09): arena_agent joins the specialized tier —
        # LMArena's agent arena (published 2026-10-02, weekly, closed-
        # inclusive) is the strongest fresh agentic evidence after tau2.
        specialized_sources=("tau2_bench_or", "bfcl", "vals_cua_bench", "arena_agent"),
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
    "decision": TaskProfile(
        name="decision",
        # Profile `decision` (2026-10-09): cheap gate/judge ("System One")
        # models that return a typed choice/score/yes-no. Pool = (a) OR's
        # structured-decisions catalog segment (gated to this profile in
        # candidates()) union (b) OR-servable models measured by a decision
        # board — evidence-driven, never hand-picked. ``jevbench`` is the
        # independent second decision board: a native-decision model with
        # rows on BOTH jevals and jevbench reaches strong evidence (>=2
        # boards; e.g. Mercury Decide, Jev 1.13). ``judgemark_v4`` is a
        # creative-writing JUDGING domain proxy: it may only CORROBORATE a
        # model also measured elsewhere; the >=2-independent-boards strong
        # rule already keeps a judgemark-only model weak_evidence, unable to
        # take rank 0 single-handedly. Honest state: models measured by NO
        # decision board stay weak_evidence until measured (watchlist:
        # JudgeArena, CodeJudgeBench). Weight law unchanged: specialized
        # quality_weight 0.7; OR usage stays a tie-break only.
        specialized_sources=("jevals", "jevbench", "judgemark_v4"),
        aggregate_sources=(),
        usage_sources=("or_usage",),
        constraints=Constraints(),
        token_mix=(0.9, 0.1, 0.0),  # short prompt, tiny typed output
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
