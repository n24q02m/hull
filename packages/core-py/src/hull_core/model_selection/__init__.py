"""hull_core.model_selection — leaderboard-driven model picker, publicly reusable.

Usage::

    from hull_core.model_selection import candidates, pick, TASKS

    cands = candidates("healthcare-advice")   # rank-aggregated: board points + price, strong first
    best = pick("healthcare-advice", strategy="knee")

    # runtime registry + refresh -> eval-on-change -> promote (hull addition)
    from hull_core.model_selection import ModelRegistry, refresh
    reg = ModelRegistry.load("model_rankings.json")
    report = refresh("translation", reg, cands, eval_fn=my_eval_fn)

Pipeline: fetch the OR backbone (text + embeddings + rerank catalog segments)
-> fetch each source (fail-open) -> join by normalized name -> version guard
-> min-max normalize per board -> blended cost -> constraint prefilter ->
rank aggregation (2026-10-07 directive): each model accumulates its
per-board standing over every board that measured it (0-100 points),
``board_rank`` by total points, ``price_rank`` cheapest-first among
cost-known models; the order optimizes both dimensions — smallest
``board_rank + price_rank``, then smallest ``rank_distance =
|board_rank - price_rank|``, then ``board_rank``. Models with weak evidence
trail behind strong ones. The module does not run evals — consumers eval and
promote from the ranked candidate list.

Selection constraints (2026-09-25 + 2026-10-07/08 directives)
-------------------------------------------------------------

- Every candidate is joined into the OR ``/api/v1/models`` catalog
  (including its embeddings/rerank output-modality segments); non-catalog
  rows are dropped at join time and can never reach rank-0.
- Embed/rerank picks MUST be OR-servable (2026-10-08 directive): the MTEB
  boards are the quality axis, OR's embeddings/rerank segments are the
  candidate + price axis.
- Free-tier models are never selected: OR's ':free' promo variants are
  excluded outright. A $0 catalog price WITHOUT the suffix (the rerank
  segment — OR bills per request) is "no price signal", not free: those
  models stay listed but unpriced.
- Unpriced models cannot take a price rank; they stay in the list but trail
  with ``pareto_rank=None``. Models measured by fewer than 2 boards carry
  ``weak_evidence=True`` and trail after strong ones — consumers should gate
  on ``rank == 0 and not weak_evidence``.

Ported from web_core.model_selection v2.10.6 (itself ported from
knowledge_core.model_selection); hull_core conventions: stdlib logging, no structlog, and the
``mteb_*`` fetchers implemented on the public ``mteb/results`` dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from hull_core.model_selection.cache import FileCache
from hull_core.model_selection.normalize import (
    ModelCandidate,
    blend_quality,
    blended_cost_1m,
    join_sources,
    minmax_normalize,
    passes_constraints,
    version_guard,
)
from hull_core.model_selection.pareto import pick as _pick
from hull_core.model_selection.registry import (
    ModelRegistry,
    PromotionDecision,
    RefreshReport,
    challengers,
    decide_promotion,
    eval_needed,
    incumbent_or_slug,
    refresh,
)
from hull_core.model_selection.sources import (
    SOURCE_REGISTRY,
    Source,
    SourceRecord,
    SourceStatus,
    UnimplementedSource,
    fetch_endpoint_stats,
)
from hull_core.model_selection.tasks import TASKS, Constraints, TaskProfile, get_task

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCE_REGISTRY",
    "TASKS",
    "Constraints",
    "FileCache",
    "ModelCandidate",
    "Source",
    "SourceRecord",
    "SourceStatus",
    "TaskProfile",
    "candidates",
    "enrich_uptime",
    "get_task",
    "pick",
]


# Runtime registry + refresh/eval/promote (hull addition over the web-core port)
# is imported at the top of this module and re-exported via __all__.
__all__ += [
    "ModelRegistry",
    "PromotionDecision",
    "RefreshReport",
    "challengers",
    "decide_promotion",
    "eval_needed",
    "incumbent_or_slug",
    "refresh",
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fetched_iso(epoch: float | None) -> str | None:
    return _now_iso() if epoch is None else datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


def _record_status(
    status_out: dict[str, dict[str, Any]] | None,
    name: str,
    status: str,
    *,
    row_count: int = 0,
    reason: str | None = None,
    fetched_at: str | None = None,
) -> None:
    if status_out is not None:
        status_out[name] = SourceStatus(status, fetched_at=fetched_at, row_count=row_count, reason=reason).to_dict()


def _fetch_source(
    source: Source,
    cache: FileCache | None,
    refresh: bool,
    status_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, SourceRecord]:
    """Fetch one source through the cache; empty/failed fetch -> try stale snapshot.

    When ``status_out`` is given, records per-source health (ok / stale /
    missing / unimplemented_access) under the source's name.
    """
    if isinstance(source, UnimplementedSource):
        _record_status(status_out, source.name, "unimplemented_access", reason=source.reason)
        return {}
    if cache is not None and not refresh:
        cached = cache.get(source.name, source.ttl_seconds)
        if cached is not None:
            _record_status(
                status_out,
                source.name,
                "ok",
                row_count=len(cached),
                fetched_at=_fetched_iso(cache.fetched_at(source.name)),
            )
            return {k: SourceRecord.from_dict(v) for k, v in cached.items()}
    try:
        records = source.fetch()
    except Exception as exc:  # plugin sources that do not inherit _BaseSource
        logger.warning("model_selection source raised: source=%s error=%s", source.name, exc)
        records = {}
    if records:
        if cache is not None:
            cache.set(source.name, {k: r.to_dict() for k, r in records.items()})
        _record_status(status_out, source.name, "ok", row_count=len(records), fetched_at=_now_iso())
        return records
    reason = getattr(source, "missing_reason", None) or "empty_fetch"
    if cache is not None:
        stale = cache.get(source.name, float("inf"), allow_stale=True)
        if stale:
            logger.info("model_selection using stale cache: source=%s", source.name)
            _record_status(
                status_out,
                source.name,
                "stale",
                row_count=len(stale),
                reason=f"{reason}; served from stale cache",
                fetched_at=_fetched_iso(cache.fetched_at(source.name)),
            )
            return {k: SourceRecord.from_dict(v) for k, v in stale.items()}
    _record_status(status_out, source.name, "missing", reason=reason)
    return records


def candidates(
    task: TaskProfile | str,
    *,
    refresh: bool = False,
    cache: FileCache | None = None,
    sources: Mapping[str, Source] | None = None,
    status_out: dict[str, dict[str, Any]] | None = None,
) -> list[ModelCandidate]:
    """Ranked candidate list for a task, rank-aggregated (board + price), weak after.

    ``sources`` overrides the registry (tests/plugins); defaults to
    ``SOURCE_REGISTRY``. Names in the TaskProfile resolve against the map;
    a name missing from the override map is skipped fail-open (registry
    defaults resolve everything).

    ``status_out`` (optional) collects per-source health as
    ``{source_name: {status, fetched_at, row_count, reason}}`` — the same dict
    can be reused across ``candidates()`` calls (the publisher does exactly
    that for its ``source_status`` snapshot block).

    Returns ``[]`` when the OpenRouter backbone is missing or empty
    (``or_backbone_empty`` is logged): without catalog rows there is nothing
    the module is allowed to auto-promote. Free-tier (':free') models are
    never selected; $0 catalog prices without the suffix (OR rerank
    segments) count as unpriced, not free.
    """
    profile = get_task(task)
    registry = sources if sources is not None else SOURCE_REGISTRY
    use_cache = cache if sources is None else None  # fixtures/plugins do not write cache

    or_source = registry.get("openrouter_models")
    if or_source is None:
        logger.warning("model_selection: missing openrouter_models backbone")
        _record_status(status_out, "openrouter_models", "missing", reason="backbone_source_absent")
        return []
    or_records = _fetch_source(or_source, use_cache, refresh, status_out)
    if not or_records:
        logger.warning(
            "model_selection or_backbone_empty: task=%s — no OpenRouter-listed models to "
            "auto-promote; pass a sources override with joinable OR-catalog entries to proceed",
            profile.name,
        )
        return []

    source_records: dict[str, dict[str, SourceRecord]] = {}
    for name in (*profile.specialized_sources, *profile.aggregate_sources):
        source = registry.get(name)
        if source is None:
            continue  # source missing from an override map — fail-open
        source_records[name] = _fetch_source(source, use_cache, refresh, status_out)

    cands = join_sources(or_records, source_records, task=profile, status_out=status_out)
    version_guard(cands)
    minmax_normalize(cands)
    blend_quality(cands, profile)
    unmeasured = len(cands) - sum(1 for c in cands if c.scores)
    if unmeasured:
        logger.info("model_selection: dropped %d catalog rows with no board score", unmeasured)
    cands = [c for c in cands if c.scores]
    for cand in cands:
        cand.cost_1m_blended = blended_cost_1m(cand, profile)
    # Free-tier exclusion (2026-10-07 directive): OR's ':free' promo variants
    # are never selected. A $0 catalog price WITHOUT the suffix (e.g. the
    # rerank segment — OR bills per request, not per token) is "no price
    # signal", not free: cost drops to None so the model stays listed but
    # cannot take a price rank (2026-10-08 directive).
    free = [c for c in cands if c.cost_1m_blended == 0 and str(c.or_slug or "").endswith(":free")]
    unpriced_zero = 0
    for cand in cands:
        if cand.cost_1m_blended == 0 and cand not in free:
            cand.cost_1m_blended = None
            unpriced_zero += 1
    if free:
        logger.info("model_selection: excluded %d free-tier models", len(free))
    if unpriced_zero:
        logger.info("model_selection: %d $0-catalog rows treated as unpriced", unpriced_zero)
    cands = [c for c in cands if c not in free]
    cands = [c for c in cands if passes_constraints(c, profile.constraints)]

    # Rank-aggregation picker (2026-10-07 directive): each model accumulates
    # its standing on every board that measured it (scores are per-board
    # normalized 0-100, name-normalized join) -> board_rank by total points;
    # price_rank by blended cost. The pick optimizes BOTH dimensions: order
    # by (board_rank + price_rank), then |board_rank - price_rank|, then
    # board_rank. Models measured by <2 boards are not joined-ranking
    # evidence (weak_evidence stays the honest label) and unpriced models
    # cannot take a price rank — both trail with pareto_rank=None.
    def _strong(c: ModelCandidate) -> bool:
        return len(c.scores) >= 2 and c.cost_1m_blended is not None

    strong = [c for c in cands if _strong(c)]
    weak = [c for c in cands if not _strong(c)]

    def _cost(c: ModelCandidate) -> float:
        assert c.cost_1m_blended is not None  # filtered into ``strong`` above
        return c.cost_1m_blended

    board_order = sorted(strong, key=lambda c: (-sum(c.scores.values()), c.or_slug or ""))
    price_order = sorted(strong, key=lambda c: (_cost(c), -c.quality))
    board_rank = {id(c): i + 1 for i, c in enumerate(board_order)}
    price_rank = {id(c): i + 1 for i, c in enumerate(price_order)}
    keyed = []
    for cand in strong:
        cand.rank_distance = abs(board_rank[id(cand)] - price_rank[id(cand)])
        keyed.append(
            (
                board_rank[id(cand)] + price_rank[id(cand)],
                cand.rank_distance,
                board_rank[id(cand)],
                cand,
            )
        )
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    strong = [t[3] for t in keyed]
    weak.sort(key=lambda c: -c.quality)
    for i, cand in enumerate(strong):
        cand.pareto_rank = i
    cands = strong + weak
    return cands


def pick(
    task: TaskProfile | str,
    *,
    strategy: Literal["knee", "quadrant", "cheapest"] = "knee",
    refresh: bool = False,
    cache: FileCache | None = None,
    sources: Mapping[str, Source] | None = None,
) -> ModelCandidate | None:
    """Pick one model per strategy; ``None`` when no candidate survives."""
    return _pick(candidates(task, refresh=refresh, cache=cache, sources=sources), strategy)


def enrich_uptime(cands: list[ModelCandidate], *, limit: int = 20) -> list[ModelCandidate]:
    """Fetch uptime_1d + zdr_available from OR /endpoints for the top-N candidates.

    Call AFTER shortlisting (1 request/model). Constraint ``min_uptime_1d``
    only filters when the field has data; ``require_zdr`` is fail-closed on
    this data.
    """
    for cand in cands[:limit]:
        stats = fetch_endpoint_stats(cand.or_slug)
        if not stats:
            continue
        cand.uptime_1d = stats.get("uptime_1d")
        cand.zdr_available = stats.get("zdr_available")
    return cands
