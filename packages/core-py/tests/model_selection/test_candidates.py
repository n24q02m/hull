"""candidates()/pick() end-to-end + join/blend/cost tests — fixture sources, no network.

Ported from knowledge_core tests, plus the OpenRouter auto-promotion
constraint (2026-09-25 directive): empty backbone -> [] + ``or_backbone_empty``
log; models absent from the OR catalog can never surface; unknown-cost
candidates are excluded from rank-0.
"""

from __future__ import annotations

import logging

import pytest

from hull_core.model_selection import candidates, pick
from hull_core.model_selection.normalize import (
    ModelCandidate,
    blend_quality,
    blended_cost_1m,
    join_sources,
    minmax_normalize,
    passes_constraints,
)
from hull_core.model_selection.sources import SourceRecord, slugify
from hull_core.model_selection.tasks import Constraints, TaskProfile


def _or_row(
    slug: str,
    *,
    name: str | None = None,
    canonical: str | None = None,
    hf: str | None = None,
    context: int = 128_000,
    modality: str = "text->text",
    prompt: str = "0.000001",
    completion: str = "0.000002",
    params: list[str] | None = None,
) -> dict:
    return {
        "id": slug,
        "name": name or slug,
        "canonical_slug": canonical or slug,
        "hugging_face_id": hf,
        "context_length": context,
        "architecture": {"modality": modality},
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": params or ["tools"],
    }


def _or_source(rows: list[dict]) -> dict[str, SourceRecord]:
    return {r["id"]: SourceRecord(key=r["id"], name=r["name"], raw=r) for r in rows}


class _FakeSource:
    """Source stub for tests — no network."""

    def __init__(self, name: str, records: dict[str, SourceRecord] | Exception) -> None:
        self.name = name
        self.ttl_seconds = 3600
        self._records = records

    def fetch(self) -> dict[str, SourceRecord]:
        if isinstance(self._records, Exception):
            raise self._records
        return self._records


class _FailOpenSource(_FakeSource):
    """Wraps fetch in try/except — mimics the real fail-open contract."""

    def fetch(self) -> dict[str, SourceRecord]:
        try:
            if isinstance(self._records, Exception):
                raise self._records
            return self._records
        except Exception:
            return {}


# --- constraint prefilter ----------------------------------------------------


def test_constraint_prefilter():
    cons = Constraints(min_context=32_000, required_input_modality="image", require_tools=True)
    ok = ModelCandidate(
        or_slug="ok",
        context=64_000,
        modalities=("text", "image"),
        supported_parameters=("tools",),
    )
    no_ctx = ModelCandidate(
        or_slug="no-ctx", context=8_000, modalities=("text", "image"), supported_parameters=("tools",)
    )
    no_img = ModelCandidate(or_slug="no-img", context=64_000, modalities=("text",), supported_parameters=("tools",))
    no_tools = ModelCandidate(or_slug="no-tools", context=64_000, modalities=("text", "image"), supported_parameters=())
    assert passes_constraints(ok, cons)
    assert not passes_constraints(no_ctx, cons)
    assert not passes_constraints(no_img, cons)
    assert not passes_constraints(no_tools, cons)


def test_zdr_fail_closed():
    cons = Constraints(require_zdr=True)
    assert not passes_constraints(ModelCandidate(or_slug="unknown", zdr_available=None), cons)
    assert not passes_constraints(ModelCandidate(or_slug="no", zdr_available=False), cons)
    assert passes_constraints(ModelCandidate(or_slug="yes", zdr_available=True), cons)


def test_max_cost_prefilter():
    cons = Constraints(max_cost_1m=5.0)
    assert passes_constraints(ModelCandidate(or_slug="cheap", cost_1m_blended=3.0), cons)
    assert not passes_constraints(ModelCandidate(or_slug="pricey", cost_1m_blended=10.0), cons)
    # unknown cost cannot satisfy a cost ceiling
    assert not passes_constraints(ModelCandidate(or_slug="unknown", cost_1m_blended=None), cons)


# --- alias join --------------------------------------------------------------


def test_alias_join_canonical_and_hf():
    """Boards using canonical_slug or hugging_face_id both join into the OR id."""
    or_records = _or_source(
        [
            _or_row("x-ai/grok-4.7", canonical="x-ai/grok-4.7-20260916", hf="xai-org/grok-4.7"),
            _or_row("openai/gpt-6", name="GPT-6 Astra"),
        ]
    )
    board = {
        "grok": SourceRecord(key="x-ai/grok-4.7-20260916", name="Grok 4.7", score=88.0),
        "gpt": SourceRecord(key="xai-org/grok-4.7", name="dup hf", score=99.0),  # hf joins same model
        "gpt6": SourceRecord(key="gpt-6-astra", name="GPT-6 Astra", score=91.0),  # slug(name) join
        "ghost": SourceRecord(key="nonexistent/model", score=50.0),  # cannot join
    }
    cands = {c.or_slug: c for c in join_sources(or_records, {"board": board})}
    assert cands["x-ai/grok-4.7"].scores["board"] == 99.0  # hf record overwrites canonical (same model)
    assert cands["openai/gpt-6"].scores["board"] == 91.0
    assert len(cands) == 2  # ghost creates no new candidate


def test_boards_first_qualification_drops_unmeasured():
    """Catalog rows with NO board score are not candidates — free is not evidence.

    Boards-first directive: quality evidence comes from fetched boards only;
    the OpenRouter catalog prices/routes the board-evidenced set, it never
    qualifies a model on its own.
    """
    or_rows = [
        _or_row("m/evidenced", context=64_000, prompt="0.000005", completion="0.00001"),
        _or_row("m/free-unmeasured", context=64_000, prompt="0", completion="0"),
    ]
    spec = {"m/evidenced": SourceRecord(key="m/evidenced", score=90.0)}
    sources = {
        "openrouter_models": _FakeSource("openrouter_models", _or_source(or_rows)),
        "spec_board": _FakeSource("spec_board", spec),
        "agg_board": _FakeSource("agg_board", {}),
    }
    cands = candidates(_TASK, sources=sources)
    slugs = [c.or_slug for c in cands]
    assert "m/free-unmeasured" not in slugs  # zero board score -> not a candidate
    assert slugs == ["m/evidenced"]


def test_boards_backbone_ranks_mteb_without_or():
    """Embed/rerank profiles: the boards are the backbone — OR absence is not fatal."""
    task = TaskProfile(
        name="rerank-boards",
        specialized_sources=("mteb_reranking",),
        aggregate_sources=(),
        backbone="boards",
    )
    sources = {
        # no openrouter_models source at all
        "mteb_reranking": _FakeSource(
            "mteb_reranking",
            {
                "BAAI/bge-reranker-v2.5": SourceRecord(key="BAAI/bge-reranker-v2.5", name="bge-reranker", score=0.61),
                "jina/reranker-v4": SourceRecord(key="jina/reranker-v4", name="jina-reranker", score=0.68),
            },
        ),
    }
    cands = candidates(task, sources=sources)
    assert [c.or_slug for c in cands] == [None, None]
    assert [c.litellm_id for c in cands] == ["jina/reranker-v4", "BAAI/bge-reranker-v2.5"]  # quality desc
    assert [c.pareto_rank for c in cands] == [0, 1]
    assert cands[0].quality == pytest.approx(100.0)  # minmax-normalized 0-100


def test_or_backbone_excludes_free_tier_models():
    """$0 promo slots are never selected — paid board-evidenced models only."""
    or_rows = [
        _or_row("vendor-a/free-promo", prompt="0", completion="0"),  # free -> excluded
        _or_row("vendor-a/paid", prompt="0.000005", completion="0.00001"),
    ]
    spec = {
        "vendor-a/free-promo": SourceRecord(key="vendor-a/free-promo", score=99.0),
        "vendor-a/paid": SourceRecord(key="vendor-a/paid", score=70.0),
    }
    sources = {
        "openrouter_models": _FakeSource("openrouter_models", _or_source(or_rows)),
        "spec_board": _FakeSource("spec_board", spec),
        "agg_board": _FakeSource("agg_board", {}),
    }
    cands = candidates(_TASK, sources=sources)
    assert [c.or_slug for c in cands] == ["vendor-a/paid"]  # free model dropped despite q=99


def test_leaderboard_rank_first_cheapest_wins_top_tier():
    """Within the top quality tier price decides; below the tier quality leads.

    Board scores are relative (normalized per source), so the fixture keeps the
    two rivals' relative standing within the 1.0 tier tolerance after blending:
    spec favors pro, agg favors value, blended gap ~0.67 < 1.0."""
    or_rows = [
        _or_row("vendor-a/pro", prompt="0.00001", completion="0.00002"),  # 10x pricier
        _or_row("vendor-a/value", prompt="0.000001", completion="0.000002"),
        _or_row("vendor-a/anchor", prompt="0.000001", completion="0.000002"),
    ]
    spec = {
        "vendor-a/pro": SourceRecord(key="vendor-a/pro", score=60.0),
        "vendor-a/value": SourceRecord(key="vendor-a/value", score=59.0),
        "vendor-a/anchor": SourceRecord(key="vendor-a/anchor", score=0.0),
    }
    agg = {
        "vendor-a/pro": SourceRecord(key="vendor-a/pro", score=59.0),
        "vendor-a/value": SourceRecord(key="vendor-a/value", score=60.0),
        "vendor-a/anchor": SourceRecord(key="vendor-a/anchor", score=0.0),
    }
    sources = {
        "openrouter_models": _FakeSource("openrouter_models", _or_source(or_rows)),
        "spec_board": _FakeSource("spec_board", spec),
        "agg_board": _FakeSource("agg_board", agg),
    }
    cands = candidates(_TASK, sources=sources)
    # pro and value blend into the same top tier (within 1.0) -> cheaper value leads;
    # anchor has bottom-tier quality -> ranked after despite being cheap.
    assert cands[0].or_slug == "vendor-a/value"
    assert cands[0].pareto_rank == 0
    assert cands[1].or_slug == "vendor-a/pro"
    assert cands[2].or_slug == "vendor-a/anchor"


def test_single_board_models_stay_unranked():
    """One board is not a joined ranking: single-board models keep rank None."""
    or_rows = [_or_row("m/a", context=64_000)]
    spec = {"m/a": SourceRecord(key="m/a", score=90.0)}
    sources = {
        "openrouter_models": _FakeSource("openrouter_models", _or_source(or_rows)),
        "spec_board": _FakeSource("spec_board", spec),
        "agg_board": _FakeSource("agg_board", {}),
    }
    cands = candidates(_TASK, sources=sources)
    assert [c.or_slug for c in cands] == ["m/a"]
    assert cands[0].pareto_rank is None
    assert cands[0].weak_evidence is True


def test_join_sources_records_matched_count():
    """source_status gains matched_count: row_count=214 matched_count=0 exposes a broken join."""
    or_records = _or_source([_or_row("m/a"), _or_row("m/b")])
    board = {
        "a": SourceRecord(key="m/a", score=80.0),
        "ghost1": SourceRecord(key="x/ghost1", score=10.0),
        "ghost2": SourceRecord(key="x/ghost2", score=20.0),
    }
    status: dict = {"board": {"status": "ok", "row_count": 3}}
    join_sources(or_records, {"board": board}, status_out=status)
    assert status["board"]["matched_count"] == 1
    # unknown source (not fetched through candidates()) still gets an entry
    status2: dict = {}
    join_sources(or_records, {"empty_board": {}}, status_out=status2)
    assert status2["empty_board"]["matched_count"] == 0


def test_alias_join_vendorless_part_unique():
    """Boards publishing vendor-less keys join via the unique model-part fallback."""
    or_records = _or_source(
        [
            _or_row("anthropic/claude-opus-5.5", name="Claude Opus 5.5"),
            _or_row("google/gemini-4-argon", name="Gemini 4 Argon"),
            _or_row("vendor-a/mini", name="Mini A"),
            _or_row("vendor-b/mini", name="Mini B"),  # ambiguous part must not join
        ]
    )
    board = {
        "claude": SourceRecord(key="claude-opus-5-5", name="claude-opus-5-5", score=91.0),
        "gemini": SourceRecord(key="google_gemini-4-argon", name="google_gemini-4-argon", score=88.0),
        "mini": SourceRecord(key="mini", name="mini", score=50.0),  # ambiguous -> dropped
        "ghost": SourceRecord(key="totally-unknown-model", name="totally-unknown-model", score=1.0),
    }
    cands = {c.or_slug: c for c in join_sources(or_records, {"board": board})}
    assert cands["anthropic/claude-opus-5.5"].scores["board"] == 91.0
    assert cands["google/gemini-4-argon"].scores["board"] == 88.0
    assert "board" not in cands["vendor-a/mini"].scores
    assert "board" not in cands["vendor-b/mini"].scores
    assert len(cands) == 4  # board records create no new candidates


def test_slugify():
    assert slugify("GPT-6 Astra") == "gpt-6-astra"
    assert slugify("x-ai/grok-4.7-20260916") == "x-ai-grok-4-7-20260916"


# --- blend + normalize -------------------------------------------------------


def test_minmax_normalize_per_source():
    cands = [
        ModelCandidate(or_slug="a", scores={"b1": 10.0, "b2": 200.0}),
        ModelCandidate(or_slug="b", scores={"b1": 20.0, "b2": 400.0}),
    ]
    minmax_normalize(cands)
    assert cands[0].scores["b1"] == 0.0 and cands[1].scores["b1"] == 100.0
    assert cands[0].scores["b2"] == 0.0 and cands[1].scores["b2"] == 100.0


def test_blend_quality_weights():
    task = TaskProfile(
        name="t",
        specialized_sources=("spec",),
        aggregate_sources=("agg",),
        quality_weight=0.7,
    )
    cands = [
        ModelCandidate(or_slug="a", scores={"spec": 100.0, "agg": 0.0}),
        ModelCandidate(or_slug="b", scores={"spec": 0.0, "agg": 100.0}),
    ]
    blend_quality(cands, task)
    # spec covers 100% of scored candidates -> w_spec = 0.7
    assert cands[0].quality == pytest.approx(70.0)
    assert cands[1].quality == pytest.approx(30.0)


def test_blend_quality_agg_only_and_low_coverage():
    task = TaskProfile(
        name="t",
        specialized_sources=("spec",),
        aggregate_sources=("agg",),
        quality_weight=0.9,
    )
    cands = [
        ModelCandidate(or_slug="a", scores={"spec": 100.0, "agg": 0.0}),
        ModelCandidate(or_slug="b", scores={"agg": 100.0}),
        ModelCandidate(or_slug="c", scores={"agg": 50.0}),
        ModelCandidate(or_slug="d", scores={}),
    ]
    blend_quality(cands, task)
    # only 1/3 scored candidates have spec -> w_spec = min(0.9, 0.5)
    assert cands[0].quality == pytest.approx(50.0)
    # agg-only model uses its only layer; empty model is weak evidence
    assert cands[1].quality == 100.0
    assert cands[3].quality == 0.0 and cands[3].weak_evidence


# --- blended cost --------------------------------------------------------------


def test_blended_cost_tiered_override():
    task = TaskProfile(name="t", token_mix=(1.0, 0.0, 0.0), expected_prompt_tokens=100_000)
    cand = ModelCandidate(
        or_slug="m/a",
        raw={
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
                "overrides": [
                    "junk",
                    {"min_prompt_tokens": "bad"},
                    {"min_prompt_tokens": 50_000, "pricing": {"prompt": "0.000005", "completion": "0.000006"}},
                    {"min_prompt_tokens": 500_000, "pricing": {"prompt": "0.000009", "completion": "0.000009"}},
                ],
            }
        },
    )
    # 50k override applies (100k >= 50k), 500k does not
    assert blended_cost_1m(cand, task) == pytest.approx(5.0)


def test_blended_cost_no_pricing_is_none():
    task = TaskProfile(name="t")
    assert blended_cost_1m(ModelCandidate(or_slug="x", raw={}), task) is None
    assert blended_cost_1m(ModelCandidate(or_slug="x", raw={"pricing": "nope"}), task) is None


def test_blended_cost_missing_completion_is_none():
    task = TaskProfile(name="t")
    cand = ModelCandidate(or_slug="x", raw={"pricing": {"prompt": "0.000001"}})
    assert blended_cost_1m(cand, task) is None


def test_blended_cost_cache_read_defaults_to_prompt():
    task = TaskProfile(name="t", token_mix=(0.5, 0.0, 0.5))
    cand = ModelCandidate(or_slug="x", raw={"pricing": {"prompt": "0.000001", "completion": "0.000002"}})
    # p_cache absent -> use p_in
    assert blended_cost_1m(cand, task) == pytest.approx(1.0)


def test_blended_cost_negative_sentinel_pricing_is_none():
    """OR ``-1`` sentinel prices (openrouter/auto etc.) are not a bargain."""
    task = TaskProfile(name="t")
    cand = ModelCandidate(or_slug="openrouter/auto", raw={"pricing": {"prompt": "-1", "completion": "-1"}})
    assert blended_cost_1m(cand, task) is None


def test_blended_cost_zero_is_free_not_none():
    """Genuinely free models keep cost 0.0 (a legitimate frontier point)."""
    task = TaskProfile(name="t")
    cand = ModelCandidate(or_slug="m/free", raw={"pricing": {"prompt": "0", "completion": "0"}})
    assert blended_cost_1m(cand, task) == 0.0


# --- candidates() end-to-end (fixture sources, no network) -------------------

_TASK = TaskProfile(
    name="test-task",
    specialized_sources=("spec_board",),
    aggregate_sources=("agg_board",),
    constraints=Constraints(min_context=16_000),
    token_mix=(0.5, 0.5, 0.0),
)


def _fixture_sources() -> dict[str, _FakeSource]:
    or_rows = [
        _or_row("m/knee", context=64_000, prompt="0.0000005", completion="0.000001"),
        _or_row("m/best", context=64_000, prompt="0.000005", completion="0.00001"),
        _or_row("m/cheap-bad", context=64_000, prompt="0.0000001", completion="0.0000002"),
        _or_row("m/dominated", context=64_000, prompt="0.00001", completion="0.00002"),
        _or_row("m/no-ctx", context=4_000),  # dropped by prefilter
    ]
    spec = {
        "m/knee": SourceRecord(key="m/knee", score=95.0),
        "m/best": SourceRecord(key="m/best", score=100.0),
        "m/cheap-bad": SourceRecord(key="m/cheap-bad", score=50.0),
        "m/dominated": SourceRecord(key="m/dominated", score=40.0),
        "m/no-ctx": SourceRecord(key="m/no-ctx", score=99.0),
    }
    agg = {
        "m/knee": SourceRecord(key="m/knee", score=90.0),
        "m/best": SourceRecord(key="m/best", score=98.0),
        "m/cheap-bad": SourceRecord(key="m/cheap-bad", score=40.0),
        "m/dominated": SourceRecord(key="m/dominated", score=30.0),
    }
    return {
        "openrouter_models": _FakeSource("openrouter_models", _or_source(or_rows)),
        "spec_board": _FakeSource("spec_board", spec),
        "agg_board": _FakeSource("agg_board", agg),
    }


def test_candidates_end_to_end():
    cands = candidates(_TASK, sources=_fixture_sources())
    slugs = [c.or_slug for c in cands]
    assert "m/no-ctx" not in slugs  # prefilter
    # leaderboard-rank-first: highest blended board quality leads the list
    assert cands[0].or_slug == "m/best"
    ranks = {c.or_slug: c.pareto_rank for c in cands}
    assert set(ranks) == {"m/best", "m/knee", "m/cheap-bad", "m/dominated"}
    assert all(r is not None for r in ranks.values())  # every 2-board model is ranked
    assert sorted(r for r in ranks.values() if r is not None) == [0, 1, 2, 3]


def test_pick_knee_end_to_end():
    best = pick(_TASK, strategy="knee", sources=_fixture_sources())
    assert best is not None
    assert best.or_slug == "m/knee"


def test_pick_cheapest():
    best = pick(_TASK, strategy="cheapest", sources=_fixture_sources())
    assert best is not None and best.or_slug == "m/cheap-bad"


def test_fail_open_empty_and_broken_sources():
    """Empty source + raising source -> no crash; no measurable model -> honest []."""
    sources = _fixture_sources()
    sources["spec_board"] = _FailOpenSource("spec_board", ConnectionError("board down"))
    sources["agg_board"] = _FailOpenSource("agg_board", {})
    cands = candidates(_TASK, sources=sources)
    # No crash; but with zero board scores nothing is measurable, so the
    # boards-first contract returns no candidates (free catalog rows are not
    # evidence) instead of a quality=0.0 catalog dump.
    assert cands == []
    assert pick(_TASK, sources=sources) is None


def test_candidates_no_backbone_returns_empty():
    sources = _fixture_sources()
    sources["openrouter_models"] = _FailOpenSource("openrouter_models", {})
    assert candidates(_TASK, sources=sources) == []
    assert pick(_TASK, sources=sources) is None


def test_candidates_missing_backbone_source_returns_empty(caplog):
    task = TaskProfile(name="t")
    with caplog.at_level(logging.WARNING, logger="hull_core.model_selection"):
        assert candidates(task, sources={"spec_board": _FailOpenSource("spec_board", {})}) == []
    assert "missing openrouter_models" in caplog.text


def test_candidates_unknown_source_name_skipped():
    task = TaskProfile(name="t", specialized_sources=("ghost_board",))
    sources = {
        "openrouter_models": _FakeSource(
            "openrouter_models",
            {"m/a": SourceRecord(key="m/a", name="m/a", raw=_or_row("m/a"))},
        )
    }
    # Ghost source is skipped fail-open; with no real board scores the only
    # catalog row is unmeasured and must NOT surface as a candidate.
    assert candidates(task, sources=sources) == []


def test_candidates_or_records_empty_returns_empty():
    task = TaskProfile(name="t")
    sources = {"openrouter_models": _FakeSource("openrouter_models", {})}
    assert candidates(task, sources=sources) == []


# --- OpenRouter auto-promotion constraint (2026-09-25 directive) --------------


def test_empty_backbone_logs_or_backbone_empty_and_promotes_nothing(caplog):
    """Embed/rerank tasks with an empty OR backbone: [] + or_backbone_empty, never a non-OR pick."""
    task = TaskProfile(
        name="embed-ish",
        specialized_sources=("mteb_reranking",),
        aggregate_sources=(),
    )
    board = {
        "baai/bge-reranker-v2.5": SourceRecord(key="baai/bge-reranker-v2.5", name="bge-reranker", score=60.0),
    }
    sources = {
        "openrouter_models": _FakeSource("openrouter_models", {}),  # OR lists nothing joinable
        "mteb_reranking": _FakeSource("mteb_reranking", board),
    }
    with caplog.at_level(logging.WARNING, logger="hull_core.model_selection"):
        assert candidates(task, sources=sources) == []
        assert pick(task, sources=sources) is None
    assert "or_backbone_empty" in caplog.text


def test_sources_override_with_or_listed_entries_promotes():
    """An override whose OR entry lists the model (joinable) yields candidates."""
    task = TaskProfile(
        name="embed-ish",
        specialized_sources=("mteb_reranking",),
        aggregate_sources=(),
        constraints=Constraints(),
    )
    or_rows = [
        # a reranker that IS listed on the OR catalog, with pricing
        _or_row("baai/bge-reranker", hf="baai/bge-reranker", prompt="0.00002", completion="0.00002"),
    ]
    board = {
        "baai/bge-reranker": SourceRecord(key="baai/bge-reranker", name="bge-reranker", score=60.0),
        "selfhost/mine": SourceRecord(key="selfhost/mine", name="mine", score=99.0),  # not OR-listed
    }
    sources = {
        "openrouter_models": _FakeSource("openrouter_models", _or_source(or_rows)),
        "mteb_reranking": _FakeSource("mteb_reranking", board),
    }
    cands = candidates(task, sources=sources)
    assert [c.or_slug for c in cands] == ["baai/bge-reranker"]  # non-OR model dropped at join
    best = pick(task, sources=sources)
    assert best is not None and best.or_slug == "baai/bge-reranker"


def test_unpriced_or_model_gets_no_rank_and_cost_none():
    """An OR-listed model without parseable pricing: cost=None, never ranked;
    and a single-board model is unranked too (one board is not a joined ranking)."""
    task = TaskProfile(
        name="embed-ish",
        specialized_sources=("mteb_reranking",),
        aggregate_sources=("benchlm",),
    )
    unpriced = _or_row("baai/bge-reranker", hf="baai/bge-reranker")
    del unpriced["pricing"]  # listed but no pricing payload
    priced = _or_row("baai/bge-reranker-lite", hf="baai/bge-reranker-lite", prompt="0.00001", completion="0.00001")
    board = {
        "baai/bge-reranker": SourceRecord(key="baai/bge-reranker", score=90.0),
        "baai/bge-reranker-lite": SourceRecord(key="baai/bge-reranker-lite", score=50.0),
    }
    agg_board = {
        "baai/bge-reranker": SourceRecord(key="baai/bge-reranker", score=88.0),
        "baai/bge-reranker-lite": SourceRecord(key="baai/bge-reranker-lite", score=52.0),
    }
    sources = {
        "openrouter_models": _FakeSource("openrouter_models", _or_source([unpriced, priced])),
        "mteb_reranking": _FakeSource("mteb_reranking", board),
        "benchlm": _FakeSource("benchlm", agg_board),
    }
    cands = {c.or_slug: c for c in candidates(task, sources=sources)}
    assert cands["baai/bge-reranker"].cost_1m_blended is None
    assert cands["baai/bge-reranker"].pareto_rank is None  # no cost -> no rank, even with 2 boards
    assert cands["baai/bge-reranker-lite"].pareto_rank == 0  # only priced model holds rank-0
    best = pick(task, sources=sources)
    assert best is not None and best.or_slug == "baai/bge-reranker-lite"
