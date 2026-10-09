"""Join sources into the OpenRouter backbone + normalize/blend scores + prefilter.

Backbone = OR ``/api/v1/models`` (text + embeddings + rerank segments): a
model that is not in the OR catalog never becomes a candidate (2026-10-08
directive: embed/rerank picks must be OR-servable too). Join key priority:
canonical_slug -> id -> hugging_face_id -> slug of display name.

Cost semantics: ``ModelCandidate.cost_1m_blended`` is ``None`` when no OR
pricing can be parsed (unlisted / free / unknown). ``None``-cost candidates
are excluded from the pareto frontier — without a cost signal a model cannot
be a best-value pick (it also cannot be beaten on price, which would make the
frontier meaningless).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from hull_core.model_selection.board_aliases import BOARD_ALIASES, UNMATCHED_REPORT_LIMIT
from hull_core.model_selection.sources import SourceRecord, slugify
from hull_core.model_selection.tasks import Constraints, TaskProfile

logger = logging.getLogger(__name__)

_COST_UNKNOWN = math.inf


@dataclass
class ModelCandidate:
    """An OR-routable model after join + normalize + pareto."""

    or_slug: str  # OR catalog id — every candidate is OR-servable (2026-10-08 directive)
    name: str = ""
    hf_id: str | None = None
    litellm_id: str | None = None
    canonical_slug: str | None = None
    scores: dict[str, float] = field(default_factory=dict)  # per-source, normalized 0-100
    score_cis: dict[str, float] = field(default_factory=dict)
    board_versions: dict[str, str] = field(default_factory=dict)
    quality: float = 0.0  # blended 0-100
    quality_ci: float | None = None
    cost_1m_blended: float | None = None  # USD / 1M tokens per task token_mix; None = unknown cost
    context: int = 0
    modalities: tuple[str, ...] = ()
    supported_parameters: tuple[str, ...] = ()
    uptime_1d: float | None = None
    zdr_available: bool | None = None
    or_task_spend_share: float | None = None
    pareto_rank: int | None = None  # 0..k-1 on the frontier; None = dominated or unknown cost
    rank_distance: int | None = None  # |board_rank - price_rank|; None when no price rank
    evidence: tuple[str, ...] = ()  # boards that contributed a score
    weak_evidence: bool = False  # fewer than two contributing boards (honest label)
    raw: dict[str, Any] = field(default_factory=dict)


def _alias_index(or_records: dict[str, SourceRecord]) -> dict[str, str]:
    """Alias -> or_slug index: id, canonical_slug, hugging_face_id, slug(name)."""
    index: dict[str, str] = {}
    tilde: list[tuple[str, str]] = []
    for slug, rec in or_records.items():
        row = rec.raw
        keys = {slug, slugify(slug)}
        canonical = row.get("canonical_slug")
        if canonical:
            keys.add(canonical)
            keys.add(slugify(canonical))
        hf = row.get("hugging_face_id")
        if hf:
            keys.add(hf)
            keys.add(slugify(hf))
        name = rec.name or row.get("name")
        if name:
            keys.add(slugify(name))
        for k in keys:
            index.setdefault(k, slug)
        # Tilde-prefix alias routes (profile decision, 2026-10-09): OR lists
        # redirect ids like "~typesafe/jev-latest" whose ``alias_target.slug``
        # names the concrete route they resolve to. A second pass maps the
        # alias id (and its target slug) onto the target catalog entry — only
        # when the target really is in the catalog, never guessed.
        if str(slug).startswith("~") and isinstance(row.get("alias_target"), dict):
            target = str(row["alias_target"].get("slug") or "")
            if target:
                tilde.append((slug, target))
    for tilde_slug, target in tilde:
        if target in or_records:
            # a tilde id is a pure redirect — it must resolve to the concrete
            # target route, never to itself
            index[tilde_slug] = target
            index[slugify(tilde_slug)] = target
            index.setdefault(target, target)
            index.setdefault(slugify(target), target)
    return index


def _part_index(or_records: dict[str, SourceRecord]) -> dict[str, list[str]]:
    """Model-part -> or_slugs index: last ``/`` segment of each slug, slugified.

    Boards frequently publish vendor-less keys (``claude-opus-5-5`` vs the
    catalog's ``anthropic/claude-opus-5.5``). The part maps back only when it
    is unique across the catalog — ambiguous parts stay unmatched.
    """
    index: dict[str, list[str]] = {}
    for slug in or_records:
        part = slugify(slug.rsplit("/", 1)[-1])
        slugs = index.setdefault(part, [])
        if slug not in slugs:
            slugs.append(slug)
    return index


# --- Escalating name normalization (2026-10-08 join fix) ----------------------
#
# Board-side join audit (live OR catalog + 13 boards, 2026-10-08) split the
# unmatched rows into (a) legitimately non-OR models — UGI community
# finetunes, open_medical_llm biomedical finetunes, GAIA agent scaffolds,
# retired claude-3.x/gemini-1.x rows — and (b) OR-listed models lost to
# spelling: dated snapshots (``claude-opus-4-5-20251101``), route markers
# (``-preview``/``-thinking``/``-latest``), vendor-prefixed keys
# (``alibaba_qwen3.8-27b``, ``xai-grok-4.3``), version-dot placement
# (``qwen2-5-7b-instruct`` vs ``qwen-2.5-7b-instruct``) and token order
# (``claude-4-1-opus`` vs ``claude-opus-4.1``). The ladder below rescues
# class (b) only, always uniqueness-gated against the catalog: an ambiguous
# variant stays unmatched instead of guessing.

_ROUTE_MARKERS = frozenset({"non-thinking", "thinking", "preview", "exp", "beta", "latest", "instruct", "it", "chat"})
_QUANT_TAILS = frozenset({"gguf", "f16", "fp8", "awq", "gptq", "exl2"})
_DATE_TAIL = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")
_SHORT_DATE_TAIL = re.compile(r"-\d{4}$")
_VERSION_ZERO_TAIL = re.compile(r"-v(\d+)-0$")
# Vendor org tokens boards prepend with ``-`` or ``_`` where OR uses ``/``.
_VENDOR_PREFIXES: tuple[str, ...] = (
    "meta-llama",
    "deepseek-ai",
    "mistralai",
    "moonshotai",
    "anthropic",
    "microsoft",
    "alibaba",
    "cohere",
    "deepseek",
    "google",
    "minimax",
    "mistral",
    "moonshot",
    "nvidia",
    "openai",
    "perplexity",
    "amazon",
    "qwen",
    "x-ai",
    "xai",
    "z-ai",
    "zai",
)
# Tokens a catalog slug may have beyond a board key and still be the same
# model route: dates, preview classes, and the instruction-tuned suffixes
# (``-it``/``-instruct``) — boards publish bare names (``gemma-3-12b``,
# ``qwen2-vl-72b``) while OR lists only the served instruct route. Model
# variants beyond that (``-thinking``, size suffixes) stay distinct.
_PREFIX_MARKERS = re.compile(r"^(?:\d{8}|\d{4}-\d{2}-\d{2}|\d{4}|preview|exp|beta|latest|it|instruct)$")
_MAX_STRIPS = 4


def _condensed(key: str) -> str:
    """Alnum-only form so ``qwen2-5`` == ``qwen-2.5`` == ``qwen25``."""
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _strip_one_tail(key: str) -> str | None:
    """One tail-token normalization round: version-zero collapse, date, then
    route-marker/quant token. ``None`` when nothing applies."""
    if _VERSION_ZERO_TAIL.search(key):
        return _VERSION_ZERO_TAIL.sub(r"-v\1", key)
    if _DATE_TAIL.search(key):
        return _DATE_TAIL.sub("", key)
    if _SHORT_DATE_TAIL.search(key):
        return _SHORT_DATE_TAIL.sub("", key)
    head, sep, last = key.rpartition("-")
    if sep and (last in _ROUTE_MARKERS or last in _QUANT_TAILS):
        return head
    return None


def _strip_vendor(key: str) -> str | None:
    """Drop one leading vendor-org token (longest match first)."""
    for vendor in _VENDOR_PREFIXES:
        if key.startswith(vendor + "-"):
            return key[len(vendor) + 1 :]
    return None


def _variant_lattice(key: str) -> list[tuple[str, int]]:
    """``(variant, strip_count)`` pairs reachable by tail strips and vendor
    strips, fewest-strips first (original excluded). Bounded depth keeps the
    lattice from drifting into unrelated model names."""
    out: dict[str, int] = {}
    frontier: list[tuple[int, str]] = [(0, key)]
    while frontier:
        cost, cur = frontier.pop()
        for step in (_strip_one_tail(cur), _strip_vendor(cur)):
            if not step:
                continue
            nxt = cost + 1
            if nxt > _MAX_STRIPS:
                continue
            if step not in out or out[step] > nxt:
                out[step] = nxt
                frontier.append((nxt, step))
    return sorted(out.items(), key=lambda t: (t[1], t[0]))


class _JoinIndexes:
    """OR-side indexes for the escalation ladder, precomputed once per join.

    Route variants (``:batch``, ``:free`` suffixes) are excluded here: they
    share the base model's quality and the exact-match alias index already
    reaches them directly, while their twin parts would only add ambiguity.
    """

    def __init__(self, or_records: dict[str, SourceRecord]) -> None:
        self.alias: dict[str, str] = _alias_index(or_records)
        self.part: dict[str, list[str]] = _part_index(or_records)
        self.cond_part: dict[str, list[str]] = {}
        self.cond_full: dict[str, list[str]] = {}
        self.sorted_part: dict[str, list[str]] = {}
        self.prefix_part: dict[str, list[str]] = {}
        for slug in or_records:
            base = slug.split(":", 1)[0]
            part = slugify(base.rsplit("/", 1)[-1])
            tokens = part.split("-")
            self.cond_part.setdefault(_condensed(part), []).append(slug)
            self.cond_full.setdefault(_condensed(base), []).append(slug)
            if len(tokens) >= 3:
                self.sorted_part.setdefault(" ".join(sorted(tokens)), []).append(slug)
            for i in range(1, len(tokens)):
                if _PREFIX_MARKERS.match(tokens[-i]):
                    hits = self.prefix_part.setdefault("-".join(tokens[:-i]), [])
                    if slug not in hits:
                        hits.append(slug)

    @staticmethod
    def _unique(hits: list[str] | None) -> str | None:
        return hits[0] if hits and len(hits) == 1 else None

    def _by_part(self, key: str) -> str | None:
        """Exact part/alias hit for a (possibly transformed) board variant."""
        for cand in (key.rsplit("/", 1)[-1], key):
            hit = self._unique(self.part.get(cand))
            if hit:
                return hit
        return self.alias.get(key)

    def rescue(self, key: str, name: str) -> str | None:
        """Escalating uniqueness-gated match for one board row: tail/date
        strips -> vendor strip -> condensed -> sorted tokens -> prefix+route
        marker. Fewest transformations wins; an ambiguous level is skipped
        (deeper levels may still disambiguate) and full ambiguity -> None."""
        base = slugify(key)
        lattices = [[(base, 0), *_variant_lattice(base)]]
        display = slugify(name) if name else ""
        if display and display != base:
            lattices.append([(display, 0), *_variant_lattice(display)])
        for lattice in lattices:
            for variant, _cost in lattice:
                hit = self._by_part(variant) or self._unique(self.cond_part.get(_condensed(variant.rsplit("/", 1)[-1])))
                if hit:
                    return hit
                tokens = variant.split("-")
                if len(tokens) >= 3:
                    hit = self._unique(self.sorted_part.get(" ".join(sorted(tokens))))
                    if hit:
                        return hit
                hit = self._unique(self.cond_full.get(_condensed(variant)))
                if hit:
                    return hit
                hit = self._unique(self.prefix_part.get(variant))
                if hit:
                    return hit
        return None


def _match_key(
    rec: SourceRecord,
    alias_index: dict[str, str],
    part_index: dict[str, list[str]] | None = None,
    join_indexes: "_JoinIndexes | None" = None,
) -> str | None:
    for cand in (rec.key, slugify(rec.key), slugify(rec.name) if rec.name else None):
        if cand and cand in alias_index:
            return alias_index[cand]
    if part_index is not None:
        for cand in (slugify(rec.key), slugify(rec.name) if rec.name else None):
            hits = part_index.get(cand) if cand else None
            if hits and len(hits) == 1:
                return hits[0]
    if join_indexes is not None:
        return join_indexes.rescue(rec.key, rec.name)
    return None


def join_sources(
    or_records: dict[str, SourceRecord],
    source_records: dict[str, dict[str, SourceRecord]],
    *,
    task: TaskProfile | None = None,
    status_out: dict[str, dict[str, Any]] | None = None,
) -> list[ModelCandidate]:
    """Join every source into the OR backbone. Unmatched records are dropped.

    Scores come ONLY from fetched board records (all boards join equally —
    specialized and aggregate alike). The OpenRouter catalog is the pricing/
    routing backbone, never a source of quality; models with no board score
    are not measurable and are filtered downstream (boards-first
    qualification). Picks must be OR-servable (2026-10-08 directive), so
    board rows that match no OR catalog entry create no candidate.

    boards-v2 (2026-10-09): sources declared in ``board_aliases.BOARD_ALIASES``
    resolve their rows through the versioned explicit mapping table FIRST
    (uniqueness-gated ladder still applies to unmapped rows); rows matching
    nothing are REPORTED via ``status_out[source]["unmatched_names"]``, never
    guessed.
    """
    alias_index = _alias_index(or_records)
    part_index = _part_index(or_records)
    join_idx = _JoinIndexes(or_records)
    candidates: dict[str, ModelCandidate] = {}
    for slug, rec in or_records.items():
        row = rec.raw
        modality = str(row.get("architecture", {}).get("modality") or "")
        inputs = tuple(m.strip() for m in modality.split("->")[0].split("+") if m.strip())
        candidates[slug] = ModelCandidate(
            or_slug=slug,
            name=rec.name,
            hf_id=row.get("hugging_face_id"),
            litellm_id=f"openrouter/{slug}",
            canonical_slug=row.get("canonical_slug"),
            context=int(row.get("context_length") or 0),
            modalities=inputs,
            supported_parameters=tuple(row.get("supported_parameters") or ()),
            raw=row,
        )

    for source_name, records in source_records.items():
        if source_name == "openrouter_models":
            continue
        # boards-v2 (2026-10-09): explicit versioned name->or_slug mappings
        # win over the ladder; a mapped slug missing from the catalog is
        # UNMATCHED (reported — the table is authoritative, never re-guessed).
        alias_map = BOARD_ALIASES.get(source_name, {})
        matched = 0
        unmatched: list[str] = []
        for rec in records.values():
            mapped = alias_map.get(rec.key) or (alias_map.get(slugify(rec.name)) if rec.name else None)
            if mapped is not None:
                slug = mapped if mapped in or_records else None
            else:
                slug = _match_key(rec, alias_index, part_index, join_idx)
            if slug is None:
                if source_name in BOARD_ALIASES:  # declared boards report, empty map included
                    unmatched.append(rec.name or rec.key)
                continue
            matched += 1
            cand = candidates[slug]
            if rec.score is not None:
                cand.scores[source_name] = rec.score
            if rec.score_ci is not None:
                cand.score_cis[source_name] = rec.score_ci
            if rec.board_version:
                cand.board_versions[source_name] = rec.board_version
        if status_out is not None:
            # Self-monitoring: ``row_count=214 matched_count=0`` exposes a broken
            # join instead of silently falling back to catalog-embedded scores.
            entry = status_out.setdefault(source_name, {})
            entry["matched_count"] = matched
            # Zero-join starvation (2026-10-08): a board that fetched rows but
            # joined NONE of them into the OR backbone reports ``degraded`` —
            # ``ok`` would hide that the board contributes nothing to
            # OR-routable selection.
            if task is not None and matched == 0 and records:
                entry["status"] = "degraded"
                entry["reason"] = "zero_join"
            # boards-v2 unmapped-name report (2026-10-09): every alias-declared
            # board (including an intentionally empty map like arena_agent)
            # publishes the rows neither the table nor the ladder resolved, so
            # a renamed board or a delisted OR slug is visible in the snapshot
            # instead of silently dropped.
            if source_name in BOARD_ALIASES and unmatched:
                entry["unmatched_names"] = sorted(set(unmatched))[:UNMATCHED_REPORT_LIMIT]
                if len(unmatched) > UNMATCHED_REPORT_LIMIT:
                    entry["unmatched_names_total"] = len(unmatched)

    return list(candidates.values())


def version_guard(candidates: list[ModelCandidate]) -> None:
    """Drop scores from stale board versions when one source has many majors.

    AA re-baselines its index between major versions (e.g. v3 -> v4):
    cross-version comparison is wrong. Keep only the newest major version per
    source, drop the rest.
    """
    newest: dict[str, int] = {}
    for cand in candidates:
        for source, ver in cand.board_versions.items():
            m = re.match(r"v?(\d+)", ver)
            if m:
                newest[source] = max(newest.get(source, 0), int(m.group(1)))
    for cand in candidates:
        for source, ver in list(cand.board_versions.items()):
            m = re.match(r"v?(\d+)", ver)
            if m and int(m.group(1)) < newest[source]:
                logger.warning(
                    "model_selection version guard drop: source=%s model=%s version=%s newest=v%d",
                    source,
                    cand.or_slug,
                    ver,
                    newest[source],
                )
                cand.scores.pop(source, None)
                cand.score_cis.pop(source, None)


def minmax_normalize(candidates: list[ModelCandidate]) -> None:
    """Min-max normalize each board to 0-100 IN-PLACE before blending.

    A board with a single value (or all values equal) gives 100.0 to every
    scored model: nothing to differentiate on, let cost decide.
    """
    sources = {s for c in candidates for s in c.scores}
    for source in sources:
        vals = [c.scores[source] for c in candidates if source in c.scores]
        lo, hi = min(vals), max(vals)
        for cand in candidates:
            if source not in cand.scores:
                continue
            cand.scores[source] = 100.0 if hi == lo else (cand.scores[source] - lo) / (hi - lo) * 100.0


def blend_quality(candidates: list[ModelCandidate], task: TaskProfile) -> None:
    """q = w_spec * q_specialized + (1 - w_spec) * q_aggregate.

    w_spec = task.quality_weight when specialized covers >=80% of scored
    candidates, else 0.5. A model with only one layer of scores uses that
    layer (not a 0).

    Evidence semantics (honest labels): fewer than TWO contributing boards is
    weak evidence — a single-board candidate can be a stale or idiosyncratic
    ranking. Consumers gate strong selection on ``weak_evidence == False``
    AND ``pareto_rank == 0``.
    """
    spec = set(task.specialized_sources)
    agg = set(task.aggregate_sources)
    scored = [c for c in candidates if c.scores]
    covered = sum(1 for c in scored if spec & set(c.scores))
    w_spec = task.quality_weight if scored and covered / len(scored) >= 0.8 else min(task.quality_weight, 0.5)
    for cand in candidates:
        spec_vals = [v for s, v in cand.scores.items() if s in spec]
        agg_vals = [v for s, v in cand.scores.items() if s in agg]
        if spec_vals and agg_vals:
            cand.quality = w_spec * (sum(spec_vals) / len(spec_vals)) + (1 - w_spec) * (sum(agg_vals) / len(agg_vals))
        elif spec_vals:
            cand.quality = sum(spec_vals) / len(spec_vals)
        elif agg_vals:
            cand.quality = sum(agg_vals) / len(agg_vals)
        else:
            cand.quality = 0.0
        cand.evidence = tuple(sorted(cand.scores))
        # weak = fewer than two contributing boards (zero-board candidates
        # included); multi-board data is what makes a strong label achievable.
        cand.weak_evidence = len(cand.evidence) < 2
        cis = [cand.score_cis[s] for s in cand.scores if s in cand.score_cis]
        cand.quality_ci = max(cis) if cis else None


def _tiered_price(pricing: dict[str, Any], expected_prompt_tokens: int) -> dict[str, Any]:
    """Apply OR ``pricing.overrides`` when the expected prompt exceeds ``min_prompt_tokens``."""
    if not expected_prompt_tokens:
        return pricing
    best: dict[str, Any] | None = None
    best_min = -1
    for ov in pricing.get("overrides") or []:
        if not isinstance(ov, dict):
            continue
        sub = ov.get("pricing") if isinstance(ov.get("pricing"), dict) else ov
        try:
            min_tokens = int(ov.get("min_prompt_tokens") or 0)
        except (TypeError, ValueError):
            continue
        if min_tokens > best_min and expected_prompt_tokens >= min_tokens:
            best, best_min = sub, min_tokens
    return best or pricing


def blended_cost_1m(cand: ModelCandidate, task: TaskProfile) -> float | None:
    """c = in_share*p_in + out_share*p_out + cache_share*p_cache_read (USD/1M).

    OR prices are USD/token -> x1e6. Unparseable price (unlisted / free /
    unknown) -> ``None``: unknown-cost candidates are excluded from the pareto
    frontier instead of silently ranking as free or infinitely expensive.
    """
    pricing = cand.raw.get("pricing")
    if not isinstance(pricing, dict):
        return None
    pricing = _tiered_price(pricing, task.expected_prompt_tokens)

    def _price(key: str) -> float | None:
        try:
            return float(pricing[key])
        except (KeyError, TypeError, ValueError):
            return None

    p_in = _price("prompt")
    p_out = _price("completion")
    p_cache = _price("input_cache_read")
    if p_in is None or p_out is None:
        return None
    if p_cache is None:
        p_cache = p_in
    in_share, out_share, cache_share = task.token_mix
    blended = (in_share * p_in + out_share * p_out + cache_share * p_cache) * 1e6
    # OR uses ``-1`` as a sentinel price on internal routes (e.g.
    # ``openrouter/auto``): a negative blended cost is a missing signal, not a
    # bargain — treat it as unknown so it cannot dominate the frontier.
    if blended < 0:
        return None
    return blended


def passes_constraints(cand: ModelCandidate, constraints: Constraints) -> bool:
    """Hard prefilter before pareto. ZDR fail-closed (unknown = drop); uptime
    only filters when data exists (before ``enrich_uptime`` nothing drops)."""
    if cand.context < constraints.min_context:
        return False
    if constraints.required_input_modality and constraints.required_input_modality not in cand.modalities:
        return False
    params = set(cand.supported_parameters)
    if constraints.require_tools and "tools" not in params:
        return False
    if constraints.require_structured_outputs and "structured_outputs" not in params:
        return False
    if constraints.require_zdr and cand.zdr_available is not True:
        return False
    if (
        constraints.min_uptime_1d is not None
        and cand.uptime_1d is not None
        and cand.uptime_1d < constraints.min_uptime_1d
    ):
        return False
    if constraints.max_cost_1m is None:
        return True
    return cand.cost_1m_blended is not None and cand.cost_1m_blended <= constraints.max_cost_1m


def effective_cost(cand: ModelCandidate) -> float:
    """Sort/compare helper: ``None`` (unknown) cost behaves as +inf."""
    return cand.cost_1m_blended if cand.cost_1m_blended is not None else _COST_UNKNOWN
