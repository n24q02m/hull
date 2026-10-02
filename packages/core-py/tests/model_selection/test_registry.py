"""Runtime registry + refresh->eval->promote tests — fixtures only, no network.

These behaviors are NEW in the hull port (web_core has no registry), so this
suite is the regression harness for the 2026-09-28 spec: refresh -> eval-on-
change -> auto-promote into the runtime registry under a strict-dominance
gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hull_core.model_selection import (
    ModelRegistry,
    challengers,
    decide_promotion,
    eval_needed,
    incumbent_or_slug,
    refresh,
)
from hull_core.model_selection.normalize import ModelCandidate


def _registry(tmp_path: Path, models: list[dict] | None = None) -> ModelRegistry:
    """Write a model_rankings.json-shaped registry with one incumbent."""
    data = {
        "snapshot_meta": {"generated_at": "test"},
        "models": models
        or [
            {
                "litellm_id": "openrouter/m-incumbent",
                "display_name": "Incumbent",
                "provider": "openrouter",
                "openrouter_slug": "m-incumbent",
                "prompt_usd_per_1m": 1.0,
                "completion_usd_per_1m": 2.0,
                "nsfw_capable": False,
                "tiers": ["echo"],
                "chain_position": 0,
                "active": True,
            },
            {
                "litellm_id": "openrouter/m-nsfw",
                "provider": "openrouter",
                "openrouter_slug": "m-nsfw",
                "nsfw_capable": True,
                "chain_position": 2,
                "active": True,
            },
        ],
    }
    path = tmp_path / "model_rankings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return ModelRegistry.load(path)


def _cand(slug: str, q: float, cost: float | None, rank: int | None = 0) -> ModelCandidate:
    c = ModelCandidate(or_slug=slug, quality=q, cost_1m_blended=cost, pareto_rank=rank)
    c.raw = {"pricing": {"prompt": str(cost / 1e6) if cost is not None else "0", "completion": "0.000002"}}
    return c


# --- promote gate --------------------------------------------------------------


def test_decide_promotion_strict_dominance_promotes():
    """Full score + secondary >= incumbent + cheaper -> promote, pick best score."""
    grid = [_cand("m/worse", 90, 0.5), _cand("m/better", 95, 0.5)]
    rows = [
        {"litellm_id": "openrouter/m/worse", "score": 80.0, "secondary": 90.0, "n_scored": 5, "n_items": 5},
        {"litellm_id": "openrouter/m/better", "score": 85.0, "secondary": 91.0, "n_scored": 5, "n_items": 5},
    ]
    inc = {"litellm_id": "openrouter/m-incumbent", "score": 82.0, "secondary": 90.0, "n_scored": 5, "n_items": 5}
    d = decide_promotion(grid, rows, incumbent_row=inc, incumbent_cost_1m=1.0)
    assert d.promote and d.winner_slug == "m/better"


def test_decide_promotion_pricier_challenger_blocked():
    """Challenger eval-better but costlier -> no promote (price is part of dominance)."""
    grid = [_cand("m/pricy", 95, 5.0)]
    rows = [{"litellm_id": "openrouter/m/pricy", "score": 99.0, "n_scored": 5, "n_items": 5}]
    inc = {"litellm_id": "x", "score": 80.0, "secondary": None, "n_scored": 5, "n_items": 5}
    d = decide_promotion(grid, rows, incumbent_row=inc, incumbent_cost_1m=1.0)
    assert not d.promote and d.gate == "no_dominance"


def test_decide_promotion_incomplete_eval_blocks():
    """An eval row with unscored items can never promote (fail-closed)."""
    grid = [_cand("m/half", 95, 0.5)]
    rows = [{"litellm_id": "openrouter/m/half", "score": 99.0, "n_scored": 2, "n_items": 5}]
    inc = {"litellm_id": "x", "score": 80.0, "secondary": None, "n_scored": 5, "n_items": 5}
    d = decide_promotion(grid, rows, incumbent_row=inc, incumbent_cost_1m=1.0)
    assert d.gate == "eval_incomplete" and not d.promote


def test_decide_promotion_lower_score_never_promotes_even_cheaper():
    grid = [_cand("m/cheap-bad", 40, 0.1)]
    rows = [{"litellm_id": "openrouter/m/cheap-bad", "score": 50.0, "n_scored": 5, "n_items": 5}]
    inc = {"litellm_id": "x", "score": 80.0, "secondary": None, "n_scored": 5, "n_items": 5}
    d = decide_promotion(grid, rows, incumbent_row=inc, incumbent_cost_1m=1.0)
    assert not d.promote


# --- registry ------------------------------------------------------------------


def test_promote_reorders_chain_and_keeps_nsfw_lead(tmp_path):
    """Challenger -> 0; prior incumbent -> 1; nsfw_capable entry holds 2."""
    reg = _registry(tmp_path)
    cand = _cand("m/challenger", 95, 0.4)
    entry = reg.promote("m/challenger", candidate=cand, tiers=["echo", "aura"])
    reg.save()

    assert entry["litellm_id"] == "openrouter/m/challenger"
    assert entry["chain_position"] == 0 and entry["nsfw_capable"] is False
    by_id = {m["litellm_id"]: m for m in reg.models}
    assert by_id["openrouter/m-incumbent"]["chain_position"] == 1
    assert by_id["openrouter/m-nsfw"]["chain_position"] == 2  # NSFW lead protected
    # persisted
    reloaded = ModelRegistry.load(reg.path)
    assert reloaded.incumbent()["litellm_id"] == "openrouter/m/challenger"


def test_promote_existing_entry_and_incumbent(tmp_path):
    reg = _registry(tmp_path)
    assert reg.incumbent()["litellm_id"] == "openrouter/m-incumbent"
    entry = reg.promote("m-incumbent")  # self-promotion = re-pin, entry exists
    assert entry["litellm_id"] == "openrouter/m-incumbent"


def test_incumbent_or_slug_mapping():
    assert incumbent_or_slug("openrouter/z-ai/glm-5.3-flash") == "z-ai/glm-5.3-flash"
    assert incumbent_or_slug("xai/grok-4") == "x-ai/grok-4"
    assert incumbent_or_slug("vertex_express/gemini-3") == "google/gemini-3"
    assert incumbent_or_slug("selfhost/foo") is None


def test_challengers_skip_incumbent_and_dominated(tmp_path):
    reg = _registry(tmp_path)
    cands = [
        _cand("m-incumbent", 80, 1.0, rank=0),  # the incumbent itself
        _cand("m/dominated", 70, 1.5, rank=1),  # worse AND pricier
        _cand("m/better", 95, 0.8, rank=2),  # better AND cheaper
    ]
    out = challengers(reg, cands)
    assert [c.or_slug for c in out] == ["m/better"]


def test_challengers_incumbent_absent_from_catalog(tmp_path):
    """Incumbent missing from today's OR catalog: quality unknown, cost from registry."""
    reg = _registry(tmp_path)
    cands = [_cand("m/new", 90, 0.5, rank=0)]
    out = challengers(reg, cands)
    assert [c.or_slug for c in out] == ["m/new"]


def test_challengers_empty_frontier_promotes_nothing(tmp_path):
    reg = _registry(tmp_path)
    cands = [_cand("m/unknown-cost", 95, None, rank=None)]
    assert challengers(reg, cands) == []


# --- eval-on-change ------------------------------------------------------------


def test_eval_needed_new_or_moved_challenger():
    c = _cand("m/a", 90, 1.0)
    assert eval_needed(c, {})
    assert not eval_needed(c, {"m/a": {"quality": 90.0, "cost_1m": 1.0}})
    assert eval_needed(c, {"m/a": {"quality": 87.0, "cost_1m": 1.0}})  # quality moved >1
    assert eval_needed(c, {"m/a": {"quality": 90.0, "cost_1m": 1.05}})  # cost moved >1%


# --- refresh end-to-end ---------------------------------------------------------


def test_refresh_dry_run_never_evals_or_promotes(tmp_path):
    reg = _registry(tmp_path)
    cands = [_cand("m/challenger", 95, 0.4, rank=0)]
    calls: list[list[str]] = []
    report = refresh("translation", reg, cands, eval_fn=lambda ids: calls.append(ids) or [], dry_run=True)
    assert report.gate == "dry_run" and calls == []
    assert ModelRegistry.load(reg.path).incumbent()["litellm_id"] == "openrouter/m-incumbent"


def test_refresh_eval_on_change_skips_seen_challengers(tmp_path):
    """Second refresh with unchanged challenger signals does not re-eval."""
    reg = _registry(tmp_path)
    cands = [_cand("m/challenger", 95, 0.4, rank=0)]

    def eval_fn(ids):
        return [
            {"litellm_id": ids[0], "score": 80.0, "n_scored": 5, "n_items": 5},
            *[{"litellm_id": i, "score": 85.0, "n_scored": 5, "n_items": 5} for i in ids[1:]],
        ]

    r1 = refresh("translation", reg, cands, eval_fn=eval_fn)
    assert r1.gate == "promote" and r1.promoted == "openrouter/m/challenger"
    # Challenger is now the incumbent — second run sees no challengers
    r2 = refresh("translation", reg, cands, eval_fn=eval_fn)
    assert r2.gate == "no_challengers"


def test_refresh_promotes_and_writes_state(tmp_path):
    """refresh persists eval state sidecar + promotes in one pass."""
    reg = _registry(tmp_path)
    cand = _cand("m/challenger", 95, 0.4, rank=0)
    seen_ids: list[str] = []

    def eval_fn(ids):
        seen_ids.extend(ids)
        return [
            {"litellm_id": ids[0], "score": 70.0, "n_scored": 5, "n_items": 5},
            {"litellm_id": ids[1], "score": 90.0, "n_scored": 5, "n_items": 5},
        ]

    report = refresh("translation", reg, [cand], eval_fn=eval_fn)
    assert report.promoted == "openrouter/m/challenger"
    assert seen_ids == ["openrouter/m-incumbent", "openrouter/m/challenger"]
    state = json.loads((tmp_path / "model_rankings.refresh-state.json").read_text(encoding="utf-8"))
    assert "m/challenger" in state["evaluated"]


def test_load_eval_state_missing_or_corrupt(tmp_path):
    from hull_core.model_selection.registry import load_eval_state

    assert load_eval_state(None) == {}
    assert load_eval_state(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_eval_state(bad) == {}
    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text('{"evaluated": "nope"}', encoding="utf-8")
    assert load_eval_state(wrong_shape) == {}


def test_eval_needed_cost_none_edges():
    """Cost appears/disappears between runs -> re-eval."""
    c = _cand("m/a", 90, 1.0)
    c.cost_1m_blended = None
    assert not eval_needed(c, {"m/a": {"quality": 90.0, "cost_1m": None}})  # both unknown: stable
    assert eval_needed(c, {"m/a": {"quality": 90.0, "cost_1m": 1.0}})  # cost vanished


def test_decide_promotion_missing_row_and_secondary_fail():
    """A challenger with no eval row can't promote; secondary regression blocks."""
    grid = [_cand("m/no-row", 90, 0.5), _cand("m/sec-fail", 95, 0.5)]
    rows = [
        {"litellm_id": "openrouter/m/sec-fail", "score": 99.0, "secondary": 40.0, "n_scored": 5, "n_items": 5},
    ]
    inc = {"litellm_id": "x", "score": 80.0, "secondary": 90.0, "n_scored": 5, "n_items": 5}
    d = decide_promotion(grid, rows, incumbent_row=inc, incumbent_cost_1m=1.0)
    assert d.gate == "eval_incomplete" and not d.promote
    assert any("no eval row" in n for n in d.per_challenger)
    assert any("secondary" in n for n in d.per_challenger)


def test_decide_promotion_two_dominant_picks_higher_score():
    grid = [_cand("m/dom1", 90, 0.5), _cand("m/dom2", 95, 0.5)]
    rows = [
        {"litellm_id": "openrouter/m/dom1", "score": 85.0, "n_scored": 5, "n_items": 5},
        {"litellm_id": "openrouter/m/dom2", "score": 90.0, "n_scored": 5, "n_items": 5},
    ]
    inc = {"litellm_id": "x", "score": 80.0, "secondary": None, "n_scored": 5, "n_items": 5}
    d = decide_promotion(grid, rows, incumbent_row=inc, incumbent_cost_1m=1.0)
    assert d.promote and d.winner_slug == "m/dom2"


def test_incumbent_or_slug_bare_and_registry_cost(tmp_path):
    from hull_core.model_selection.registry import registry_entry_cost_1m

    assert incumbent_or_slug("noslash") is None
    assert registry_entry_cost_1m({}) is None
    assert registry_entry_cost_1m({"prompt_usd_per_1m": "bad", "completion_usd_per_1m": 1.0}) is None


def test_challengers_incumbent_dominates_all(tmp_path):
    """Incumbent better AND cheaper than every challenger -> empty list."""
    reg = _registry(tmp_path)
    cands = [
        _cand("m-incumbent", 95, 0.5, rank=0),
        _cand("m/worse", 80, 1.0, rank=1),
    ]
    assert challengers(reg, cands) == []


def test_refresh_promotes_existing_registry_entry(tmp_path):
    """Challenger already in the registry (inactive) -> entry reused, prices updated."""
    reg = _registry(
        tmp_path,
        models=[
            {
                "litellm_id": "openrouter/m-incumbent",
                "provider": "openrouter",
                "openrouter_slug": "m-incumbent",
                "prompt_usd_per_1m": 1.0,
                "completion_usd_per_1m": 2.0,
                "chain_position": 0,
                "active": True,
            },
            {
                "litellm_id": "openrouter/m/challenger",
                "provider": "openrouter",
                "openrouter_slug": "m/challenger",
                "active": False,
            },
        ],
    )
    cand = _cand("m/challenger", 95, 0.4, rank=0)

    def eval_fn(ids):
        return [
            {"litellm_id": ids[0], "score": 70.0, "n_scored": 5, "n_items": 5},
            {"litellm_id": ids[1], "score": 90.0, "n_scored": 5, "n_items": 5},
        ]

    report = refresh("translation", reg, [cand], eval_fn=eval_fn)
    assert report.promoted == "openrouter/m/challenger"
    entry = next(m for m in reg.models if m["litellm_id"] == "openrouter/m/challenger")
    assert entry["active"] is True and entry["chain_position"] == 0
    # pricing fields filled from the candidate's OR row
    assert entry["prompt_usd_per_1m"] == pytest.approx(0.4)


def test_refresh_no_eval_fn_is_dry_run(tmp_path):
    reg = _registry(tmp_path)
    report = refresh("translation", reg, [_cand("m/c", 95, 0.4, rank=0)])
    assert report.gate == "dry_run"
    assert ModelRegistry.load(reg.path).incumbent()["litellm_id"] == "openrouter/m-incumbent"


def test_refresh_non_dominant_eval_does_not_promote(tmp_path):
    reg = _registry(tmp_path)
    cand = _cand("m/challenger", 95, 0.4, rank=0)

    def eval_fn(ids):
        return [
            {"litellm_id": ids[0], "score": 95.0, "n_scored": 5, "n_items": 5},
            {"litellm_id": ids[1], "score": 60.0, "n_scored": 5, "n_items": 5},  # worse than incumbent
        ]

    report = refresh("translation", reg, [cand], eval_fn=eval_fn)
    assert report.gate == "no_dominance" and report.promoted is None
    assert ModelRegistry.load(reg.path).incumbent()["litellm_id"] == "openrouter/m-incumbent"


def test_registry_rejects_empty(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"models": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no active model"):
        ModelRegistry.load(path).incumbent()
