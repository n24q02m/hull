"""Runtime model registry + the refresh -> eval-on-change -> promote flow.

New in the hull port (spec 2026-09-28, D-KP6 / phase H): ``web_core``'s
upstream module stops at a ranked candidate list — the consumer had to own
the registry file, the change detector and the promote gate. This module
moves that contract into hull so KP and Aiora share it.

Registry format (JSON on disk, dict-shaped for forward compatibility)::

    {"snapshot_meta": {...}, "models": [ {<entry fields>}, ... ]}

The schema matches what KlPrism's ``model_rankings.json`` already consumes:
``litellm_id``, ``provider``, ``openrouter_slug``, ``input_modalities``,
``supports_structured_output``, ``prompt_usd_per_1m``,
``completion_usd_per_1m``, ``nsfw_capable``, ``ugi``, ``active``, ``tiers``,
``chain_position``. The incumbent = the active entry with the lowest
``chain_position`` (0). ``promote()`` reorders the chain: challenger -> 0,
previous actives -> 1, except ``nsfw_capable`` entries which hold position 2
(the NSFW chain head is a human decision — same policy as KP's refresh:
UGI challengers are *recommended*, never auto-promoted, because
``nsfw_capable=True`` is set by a human on the registry entry, and promote()
never flips it).

Refresh pipeline::

    cands   = candidates(task)                       # scan (free)
    reg     = ModelRegistry.load(path)               # runtime registry
    chall   = challengers(reg, cands)                # frontier minus incumbent
    grid    = [c for c in chall if eval_needed(...)] # eval-on-change
    rows    = eval_fn([incumbent, *grid])            # paid, consumer harness
    decision= decide_promotion(...)                  # strict dominance
    if decision.promote: reg.promote(...) ; reg.save()

``eval_fn`` is consumer-injected (KP's shortlist eval speaks litellm ids and
returns COMET/QA-judge scores; Aiora plugs its own harness). Spend caps live
inside ``eval_fn`` / the caller's scheduler — this module decides *whether*
to eval, not how much it may cost.

Eval-on-change state (which challenger was last eval'd at which
quality/cost) persists in a sidecar JSON next to the registry
(``<registry>.refresh-state.json``) — the role KP's sticky issue played,
without the GitHub dependency.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hull_core.model_selection.normalize import ModelCandidate

logger = logging.getLogger(__name__)

EvalRow = dict[str, Any]
"""One eval result: ``{"litellm_id", "score", "secondary", "n_scored", "n_items"}``.

``score`` is the primary metric (KP: COMET), ``secondary`` a guard metric
(KP: QA judge). Missing -> ``None`` = unknown, never 0.
"""

EvalFn = Callable[[list[str]], list[EvalRow]]
"""Consumer eval harness: litellm_ids in, one EvalRow per id (same order)."""


# --- Registry ------------------------------------------------------------------


class ModelRegistry:
    """JSON runtime registry: load/save + incumbent + promote.

    Entries stay plain dicts so unknown consumer fields survive a
    load->promote->save round-trip untouched.
    """

    def __init__(self, path: Path | str, data: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        if data is None:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        self.data = data
        self.models: list[dict[str, Any]] = data.setdefault("models", [])

    @classmethod
    def load(cls, path: Path | str) -> ModelRegistry:
        return cls(path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def active(self) -> list[dict[str, Any]]:
        """Active entries sorted by chain_position (0 = incumbent first)."""
        return sorted(
            (m for m in self.models if m.get("active")),
            key=lambda m: m.get("chain_position", 99),
        )

    def incumbent(self) -> dict[str, Any]:
        """Runtime primary = active entry with chain_position 0."""
        act = self.active()
        if not act:
            raise ValueError(f"{self.path}: registry has no active model")
        return act[0]

    def promote(
        self,
        or_slug: str,
        *,
        candidate: ModelCandidate | None = None,
        tiers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Promote an OR-listed challenger to runtime primary.

        Challenger -> active, chain_position 0, on ``tiers``. Other actives
        demote to position 1, except ``nsfw_capable`` entries which keep the
        fallback lead at position 2 (ported from KP apply_promotion). New
        entries are created nsfw_capable=False — auto-promote can never mint
        an NSFW-capable model.
        """
        litellm_id = f"openrouter/{or_slug}"
        entry = next((m for m in self.models if m.get("litellm_id") == litellm_id), None)
        if entry is None:
            entry = {
                "litellm_id": litellm_id,
                "display_name": (candidate.name if candidate and candidate.name else or_slug.split("/", 1)[-1]),
                "provider": "openrouter",
                "openrouter_slug": or_slug,
                "input_modalities": list(candidate.modalities) if candidate and candidate.modalities else ["text"],
                "supports_structured_output": (
                    "structured_outputs" in candidate.supported_parameters if candidate else True
                ),
                "nsfw_capable": False,
                "category_rank": {},
                "ugi": None,
            }
            self.models.append(entry)
        if candidate is not None:
            raw_pricing: Any = candidate.raw.get("pricing")
            pricing: dict[str, Any] = raw_pricing if isinstance(raw_pricing, dict) else {}
            for field_name, key in (("prompt_usd_per_1m", "prompt"), ("completion_usd_per_1m", "completion")):
                try:
                    price = float(pricing[key]) * 1e6
                except (KeyError, TypeError, ValueError):
                    continue
                if price >= 0:  # -1 sentinel = unknown, not free
                    entry[field_name] = price
        entry.update({"active": True, "chain_position": 0})
        if tiers is not None:
            entry["tiers"] = list(tiers)
        elif "tiers" not in entry:
            entry["tiers"] = ["default"]
        for m in self.models:
            if m is entry or not m.get("active"):
                continue
            m["chain_position"] = 2 if m.get("nsfw_capable") else 1
        return entry


def incumbent_or_slug(litellm_id: str) -> str | None:
    """Map a registry litellm_id to its OpenRouter catalog slug, if routable.

    ``vertex_express``/``vertex_ai``/``gemini`` pass through to ``google/*``;
    ``xai/*`` maps to ``x-ai/*``; ``openrouter/*`` unwraps bare.
    """
    provider, _, model = litellm_id.partition("/")
    if not model:
        return None
    if provider == "openrouter":
        return model
    if provider == "xai":
        return f"x-ai/{model}"
    if provider in ("vertex_express", "vertex_ai", "gemini"):
        return f"google/{model}"
    return None


def registry_entry_cost_1m(entry: dict[str, Any]) -> float | None:
    """Blended proxy cost from stored per-1m prices; None when unknown."""
    try:
        return (float(entry["prompt_usd_per_1m"]) + float(entry["completion_usd_per_1m"])) / 2
    except (KeyError, TypeError, ValueError):
        return None


# --- Scan -> challengers --------------------------------------------------------


def challengers(
    registry: ModelRegistry,
    cands: list[ModelCandidate],
    *,
    frontier_only: bool = True,
) -> list[ModelCandidate]:
    """Frontier members not dominated by the incumbent, best quality first.

    The incumbent counts as a comparison point even when absent from today's
    OR catalog (retired slug, non-OR route): its registry entry still carries
    pricing, and its board score stays unknown (``quality=0`` would fake
    dominance, so a catalog-absent incumbent can only lose on price, never
    "win" on quality).
    """
    inc = registry.incumbent()
    inc_or = incumbent_or_slug(inc.get("litellm_id", ""))
    inc_cand = next((c for c in cands if c.or_slug == inc_or), None)
    inc_cost = inc_cand.cost_1m_blended if inc_cand is not None else registry_entry_cost_1m(inc)
    inc_quality = inc_cand.quality if inc_cand is not None else None

    pool = cands
    if frontier_only:
        front = [c for c in cands if c.pareto_rank is not None]
        if front:
            pool = front
        # empty frontier (all unknown cost) -> nobody is promotable anyway
        else:
            return []
    out = []
    for c in pool:
        if inc_or is not None and c.or_slug == inc_or:
            continue
        if inc_quality is not None and inc_cost is not None:
            if inc_quality >= c.quality and inc_cost <= (c.cost_1m_blended or math.inf):
                continue  # strictly dominated by the incumbent
        out.append(c)
    out.sort(key=lambda c: -c.quality)
    return out


# --- Eval-on-change ------------------------------------------------------------


def _state_path_for(registry_path: Path) -> Path:
    # model_rankings.json -> model_rankings.refresh-state.json
    return registry_path.with_suffix(".refresh-state.json")


def load_eval_state(path: Path | str | None) -> dict[str, dict[str, Any]]:
    """``{or_slug: {"quality": q, "cost_1m": c}}`` from the refresh sidecar."""
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    raw = data.get("evaluated") if isinstance(data, dict) else None
    return raw if isinstance(raw, dict) else {}


def save_eval_state(path: Path | str, evaluated: dict[str, dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"updated_at": time.time(), "evaluated": evaluated}, indent=1) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def eval_needed(
    cand: ModelCandidate,
    evaluated: dict[str, dict[str, Any]],
    *,
    quality_tolerance: float = 1.0,
    cost_tolerance: float = 0.01,
) -> bool:
    """A challenger earns the paid eval when never evaluated, or when its
    (quality, price) moved since the last eval. Mirrors KP's drift check:
    |Δquality| > 1 point or |Δcost| > 1% re-triggers the eval."""
    prev = evaluated.get(cand.or_slug)
    if prev is None:
        return True
    if abs((prev.get("quality") or 0.0) - cand.quality) > quality_tolerance:
        return True
    prev_cost = prev.get("cost_1m")
    cost = cand.cost_1m_blended
    if prev_cost is None or cost is None:
        return prev_cost != cost
    return abs(prev_cost - cost) / max(cost, 1e-9) > cost_tolerance


# --- Promote gate ---------------------------------------------------------------

Gate = Literal["promote", "no_dominance", "eval_incomplete"]


@dataclass
class PromotionDecision:
    """Outcome of the strict-dominance gate."""

    gate: Gate
    winner_slug: str | None = None
    winner: ModelCandidate | None = None
    reason: str = ""
    per_challenger: list[str] = field(default_factory=list)

    @property
    def promote(self) -> bool:
        return self.gate == "promote"


def decide_promotion(
    grid: list[ModelCandidate],
    eval_rows: list[EvalRow],
    *,
    incumbent_row: EvalRow,
    incumbent_cost_1m: float | None,
    max_challengers: int | None = None,
) -> PromotionDecision:
    """Strict-dominance gate (ported from KP model_refresh):

    promote requires ALL of — every eval item scored, primary score >=
    incumbent's, secondary score >= incumbent's when both known, and blended
    cost <= incumbent's. Among qualifying challengers the highest primary
    score wins. ``eval_rows`` covers ``grid`` in order; a row count shorter
    than ``grid`` marks the tail ``eval_incomplete``.
    """
    decision = PromotionDecision(gate="no_dominance")
    rows = {r.get("litellm_id"): r for r in eval_rows}
    best: tuple[float, ModelCandidate] | None = None
    incompletes = 0
    for cand in grid[:max_challengers] if max_challengers else grid:
        row = rows.get(f"openrouter/{cand.or_slug}") or rows.get(cand.or_slug)
        if row is None:
            decision.per_challenger.append(f"{cand.or_slug}: no eval row")
            incompletes += 1
            continue
        n_items = row.get("n_items") or 0
        n_scored = row.get("n_scored") or 0
        if n_items <= 0 or n_scored != n_items:
            decision.per_challenger.append(f"{cand.or_slug}: eval items failed ({n_scored}/{n_items})")
            incompletes += 1
            continue
        score, inc_score = row.get("score"), incumbent_row.get("score")
        if score is None or inc_score is None or score < inc_score:
            decision.per_challenger.append(
                f"{cand.or_slug}: score {score} < incumbent {inc_score}"
                if score is not None
                else f"{cand.or_slug}: no score"
            )
            continue
        sec, inc_sec = row.get("secondary"), incumbent_row.get("secondary")
        if sec is not None and inc_sec is not None and sec < inc_sec:
            decision.per_challenger.append(f"{cand.or_slug}: secondary {sec} < incumbent {inc_sec}")
            continue
        cost = cand.cost_1m_blended
        if cost is None or incumbent_cost_1m is None or cost > incumbent_cost_1m:
            decision.per_challenger.append(f"{cand.or_slug}: cost {cost} does not beat incumbent {incumbent_cost_1m}")
            continue
        if best is None or score > best[0]:
            best = (score, cand)
        else:
            decision.per_challenger.append(f"{cand.or_slug}: dominant but not best")
    if best is not None:
        decision.gate = "promote"
        decision.winner = best[1]
        decision.winner_slug = best[1].or_slug
        decision.reason = f"{best[1].or_slug} strictly dominates incumbent (score {best[0]})"
    elif incompletes:
        decision.gate = "eval_incomplete"
        decision.reason = "no dominance proven; some challengers unevaluated"
    else:
        decision.reason = "no strict dominance — recommend only"
    return decision


# --- Orchestration --------------------------------------------------------------


@dataclass
class RefreshReport:
    """Outcome of one refresh -> eval -> promote run."""

    task: str
    gate: str  # dry_run | no_challengers | eval_skipped_no_change | <PromotionDecision.gate>
    candidates: int
    challengers: list[str] = field(default_factory=list)
    evaluated: list[str] = field(default_factory=list)
    promoted: str | None = None
    reason: str = ""
    notes: list[str] = field(default_factory=list)


def refresh(
    task: str,
    registry: ModelRegistry,
    cands: list[ModelCandidate],
    *,
    eval_fn: EvalFn | None = None,
    state_path: Path | str | None = None,
    max_challengers: int = 3,
    frontier_only: bool = True,
    dry_run: bool = False,
) -> RefreshReport:
    """refresh -> eval-on-change -> promote, one task.

    ``cands`` comes from ``model_selection.candidates(task, ...)`` (already
    joined/normalized/pareto-ranked). ``eval_fn`` gets litellm ids for
    ``[incumbent, *new_challengers]`` and returns EvalRows in the same order;
    without it nothing is promoted (dry-run semantics). The incumbent is
    always re-evaluated so dominance compares equal-harness numbers.
    """
    report = RefreshReport(task=task, gate="no_challengers", candidates=len(cands))
    state_path = state_path or _state_path_for(registry.path)
    evaluated = load_eval_state(state_path)

    chall = challengers(registry, cands, frontier_only=frontier_only)
    report.challengers = [c.or_slug for c in chall]
    if not chall:
        report.reason = "no challenger on the frontier"
        return report
    if dry_run or eval_fn is None:
        report.gate = "dry_run"
        report.reason = "scan only — eval/promote skipped"
        return report

    # Eval-on-change first, cap second (KP order): a stale challenger must not
    # consume a max_challengers slot a new challenger needs.
    grid = [c for c in chall if eval_needed(c, evaluated)][:max_challengers]
    if not grid:
        report.gate = "eval_skipped_no_change"
        report.reason = "no NEW challengers — eval skipped (eval-on-change)"
        return report

    inc = registry.incumbent()
    grid_ids = [f"openrouter/{c.or_slug}" for c in grid]
    rows = eval_fn([inc["litellm_id"], *grid_ids])
    report.evaluated = [c.or_slug for c in grid]
    for c in grid:
        evaluated[c.or_slug] = {"quality": c.quality, "cost_1m": c.cost_1m_blended}
    save_eval_state(state_path, evaluated)

    inc_row = rows[0] if rows else {}
    inc_cand = next((c for c in cands if c.or_slug == incumbent_or_slug(inc.get("litellm_id", ""))), None)
    inc_cost = inc_cand.cost_1m_blended if inc_cand is not None else registry_entry_cost_1m(inc)

    decision = decide_promotion(grid, rows[1:], incumbent_row=inc_row, incumbent_cost_1m=inc_cost)
    report.gate = decision.gate
    report.reason = decision.reason
    report.notes = decision.per_challenger
    if decision.promote and decision.winner is not None:
        entry = registry.promote(decision.winner_slug or decision.winner.or_slug, candidate=decision.winner)
        registry.save()
        report.promoted = entry["litellm_id"]
        logger.info("model_selection promoted %s -> runtime primary (task=%s)", report.promoted, task)
    return report
