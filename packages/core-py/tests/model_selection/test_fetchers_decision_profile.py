"""decision-profile fetcher/gate tests — fixtures captured live 2026-10-09, no network.

Covers the profile `decision` wave: OR decisions catalog segment + modality
gate, jevals board (release resolution, min-max mean scoring, CC-BY-4.0
attribution, 90-day static-anchor staleness guard), eqbench Judgemark V4, the
versioned jevals alias entry, and the catalog tilde alias_target rule.
Live-fixture provenance:
- jevals_home.html — SSR home trimmed around the embedded `release_id`
  payload token (the exact string the fetcher regex resolves), verbatim
- jevals_board.json — verbatim release 2026-09-18 board.json (24 rows)
- judgemark_v4.js — verbatim (leaderboardDataJudgemarkV4 CSV template)
- or_decisions.json — verbatim ?output_modalities=decisions response (16 rows)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import hull_core.model_selection as ms
from hull_core.model_selection import candidates
from hull_core.model_selection.board_aliases import BOARD_ALIASES
from hull_core.model_selection.normalize import _alias_index, join_sources
from hull_core.model_selection.sources import (
    SOURCE_REGISTRY,
    EqBenchCsvSource,
    JevDecisionIndexSource,
    JevbenchSource,
    JevalsSource,
    OpenRouterModelsSource,
    SourceRecord,
    _parse_jevals_board,
    slugify,
)
from hull_core.model_selection.tasks import TASKS, TaskProfile

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _json_fixture(name: str) -> Any:
    return json.loads(_fixture(name).decode("utf-8"))


def _resp(payload=None, *, content: bytes | None = None):
    r = MagicMock()
    r.json.return_value = payload
    r.content = content if content is not None else b""
    r.raise_for_status.return_value = None
    return r


class _StaticSource:
    """Typed Source-protocol stub: name/ttl class attrs, callable fetch."""

    name: str = "stub"
    ttl_seconds: int = 86400

    def __init__(self, name: str, records: dict[str, SourceRecord]) -> None:
        self.name = name
        self._records = records

    def fetch(self) -> dict[str, SourceRecord]:
        return self._records


# --- OR decisions segment + modality gate --------------------------------------


def test_backbone_merges_decisions_segment_with_tags(monkeypatch):
    """All four catalog segments are fetched; a slug present in two segments
    carries both tags, decisions rows are tagged for the modality gate."""
    text_row = {"id": "a/chat", "name": "A Chat", "architecture": {"modality": "text->text"}}
    decisions_row = {
        "id": "d/only",
        "name": "D Only",
        "architecture": {"modality": "text->decisions", "output_modalities": ["decisions"]},
    }
    dual_row = {
        "id": "a/dual",
        "name": "A Dual",
        "architecture": {"modality": "text->decisions", "output_modalities": ["decisions"]},
    }

    def fake_get_json(url, **_kw):
        if "output_modalities=decisions" in url:
            return {"data": [decisions_row, dual_row]}
        if "output_modalities=" in url:
            return {"data": []}
        return {"data": [text_row, dual_row]}

    monkeypatch.setattr("hull_core.model_selection.sources._get_json", fake_get_json)
    records = OpenRouterModelsSource().fetch()
    assert set(records) == {"a/chat", "d/only", "a/dual"}
    assert records["a/chat"].raw["_or_segments"] == ["text"]
    assert records["d/only"].raw["_or_segments"] == ["decisions"]
    assert records["a/dual"].raw["_or_segments"] == ["text", "decisions"]


def _or_rec(slug: str, *, segments: list[str] | None = None) -> SourceRecord:
    row = {"id": slug, "name": slug, "architecture": {"modality": "text->text"}}
    if segments is not None:
        row["_or_segments"] = segments
    return SourceRecord(key=slug, name=slug, raw=row)


def test_modality_gate_keeps_decisions_rows_out_of_chat_pools():
    """Decisions-only catalog rows enter the `decision` profile pool alone —
    never chat pools, even when a board score would join them."""
    or_rows = {
        "a/chat": _or_rec("a/chat", segments=["text"]),
        "d/only": _or_rec("d/only", segments=["decisions"]),
    }
    board = {
        "a-chat": SourceRecord(key="a-chat", score=80.0),
        "d-only": SourceRecord(key="d-only", score=80.0),
    }
    sources = {
        "openrouter_models": _StaticSource("openrouter_models", or_rows),
        "board_x": _StaticSource("board_x", board),
    }
    chat_profile = TaskProfile(name="chat", specialized_sources=("board_x",))
    decision_profile = TaskProfile(name="decision", specialized_sources=("board_x",))
    chat = candidates(chat_profile, sources=sources)
    decision = candidates(decision_profile, sources=sources)
    assert [c.or_slug for c in chat] == ["a/chat"]
    assert {c.or_slug for c in decision} == {"a/chat", "d/only"}


def test_decisions_rows_live_in_backbone_segment():
    """Fixture sanity: the live decisions segment lists the jev family, the
    tilde alias route and mercury-decide (16 rows)."""
    ids = {row["id"] for row in _json_fixture("or_decisions.json")["data"]}  # type: ignore[index]
    assert len(ids) == 16
    assert {"typesafe/jev-1.13", "~typesafe/jev-latest", "inception/mercury-decide"} <= ids


# --- jevals board ----------------------------------------------------------------


def test_jevals_parse_board(monkeypatch):
    data = _json_fixture("jevals_board.json")

    def fake_get_json(url, **_kw):
        assert "/data/releases/2026-09-18/board.json" in url
        return data

    monkeypatch.setattr("hull_core.model_selection.sources._get_json", fake_get_json)
    records = _parse_jevals_board(data)  # type: ignore[arg-type]
    # 7 models on the board; the label-prior baseline (model_id null) excluded
    assert len(records) == 7
    assert "typesafe-ai-jev" in records
    assert "google-gemini-3-8-flash" in records
    assert all(r.score is not None and 0.0 <= r.score <= 100.0 for r in records.values())
    assert all(r.board_version == "2026-09-18" for r in records.values())


def test_jevals_source_resolution_attribution_staleness(monkeypatch):
    src = JevalsSource()

    def fake_get_json(url, **_kw):
        if url == "https://jevals.com/data/releases/2026-09-18/board.json":
            return _json_fixture("jevals_board.json")
        raise AssertionError(f"unexpected fetch {url}")

    monkeypatch.setattr(
        "hull_core.model_selection.sources._get_bytes",
        lambda url, **_kw: _fixture("jevals_home.html"),
    )
    monkeypatch.setattr("hull_core.model_selection.sources._get_json", fake_get_json)
    records = src.fetch()
    assert len(records) == 7
    assert src.board_updated_at == "2026-09-18"
    assert src.attribution and "CC-BY-4.0" in src.attribution and "jevals.com" in src.attribution
    # fresh release (17 days old at fixture date) — not a static anchor
    assert src.degraded_reason is None
    status: dict = {}
    assert ms._fetch_source(src, None, False, status)
    assert status["jevals"]["status"] == "ok"


def test_jevals_staleness_guard_reports_static_anchor(monkeypatch):
    """A release unchanged for >90 days still serves rows but reports
    ``degraded`` / ``static_anchor`` — a frozen anchor, not live evidence."""
    data = dict(_json_fixture("jevals_board.json"))  # type: ignore[arg-type]
    data["as_of"] = "2026-01-01"

    def fake_get_json(url, **_kw):
        if url == "https://jevals.com/data/releases/2026-09-18/board.json":
            return data
        raise AssertionError(f"unexpected fetch {url}")

    monkeypatch.setattr(
        "hull_core.model_selection.sources._get_bytes",
        lambda url, **_kw: _fixture("jevals_home.html"),
    )
    monkeypatch.setattr("hull_core.model_selection.sources._get_json", fake_get_json)
    src = JevalsSource()
    records = src.fetch()
    assert len(records) == 7  # rows still served
    assert src.degraded_reason and src.degraded_reason.startswith("static_anchor")
    status: dict = {}
    ms._fetch_source(src, None, False, status)
    assert status["jevals"]["status"] == "degraded"
    assert "static_anchor" in status["jevals"]["reason"]


def test_jevals_unresolvable_release_fails_open(monkeypatch):
    monkeypatch.setattr(
        "hull_core.model_selection.sources._get_bytes",
        lambda url, **_kw: b"<html><title>Jevals</title></html>",
    )
    src = JevalsSource()
    assert src.fetch() == {}
    assert src.missing_reason == "jevals_release_unresolved"


# --- judgemark v4 ----------------------------------------------------------------


def test_judgemark_v4_parse(monkeypatch):
    src = EqBenchCsvSource(
        "judgemark_v4",
        "file://unused",
        "leaderboardDataJudgemarkV4",
        "score",
        name_cols=("model",),
    )
    js = _fixture("judgemark_v4.js").decode("utf-8")
    monkeypatch.setattr("hull_core.model_selection.sources._get_bytes", lambda url, **_kw: js.encode("utf-8"))
    records = src.fetch()
    assert len(records) > 5
    top = max(records.values(), key=lambda r: r.score or 0.0)
    assert top.key == "claude-opus-4-6"
    assert top.score is not None and 0.0 < top.score < 1.0  # raw 0-1 score, NOT rescaled — corroborate only
    assert "jevals" not in records


# --- jevbench board ----------------------------------------------------------------


def test_jevbench_parse(monkeypatch):
    """Fixture is row-trimmed from the live 140-system response (2026-10-09):
    the 2 alias-mapped api rows + 1 api row + 2 self-hosted rows, verbatim."""
    src = JevbenchSource()
    monkeypatch.setattr(
        "hull_core.model_selection.sources._get_json", lambda url, **_kw: _json_fixture("jevbench_v161.json")
    )
    records = src.fetch()
    assert len(records) == 5
    jev = records["jev-1-13-0-typesafe-ai"]
    assert jev.score is not None and 0.0 < jev.score < 100.0
    assert jev.board_version == "v1.6.1"
    assert src.board_updated_at == "v1.6.0"
    assert src.attribution and "JevBench" in src.attribution and "v1.6.1" in src.attribution


def test_jevbench_joins_api_routes_only_and_reports_selfhosted():
    """Only api_flag==true hosted routes are mapped (versioned alias table);
    self-hosted open-weight systems report unmatched — never mapped, even
    when a display name hints at an OR route."""
    data = _json_fixture("jevbench_v161.json")
    records = {
        slugify(row["display"]): SourceRecord(
            key=slugify(row["display"]),
            name=row["display"],
            score=row["jevbench_score"],
            raw=dict(row),
        )
        for row in data["systems"]
    }
    or_rows = {
        "typesafe/jev-1.13": _or_rec("typesafe/jev-1.13"),
        "inception/mercury-decide": _or_rec("inception/mercury-decide"),
    }
    status: dict = {}
    cands = join_sources(or_rows, {"jevbench": records}, status_out=status)
    by_slug = {c.or_slug: c for c in cands}
    assert by_slug["typesafe/jev-1.13"].scores["jevbench"] == pytest.approx(records["jev-1-13-0-typesafe-ai"].score)
    assert by_slug["inception/mercury-decide"].scores["jevbench"] == pytest.approx(
        records[
            "mercury-decide-inception-system-one-decisions-api-served-free-on-openrouter-as-inception-mercury-decide-free"
        ].score
    )
    unmatched = status["jevbench"]["unmatched_names"]
    # sage (api, no OR route) + the two self-hosted systems all report
    assert any("Sage 1.3.0" in n for n in unmatched)
    assert any("Jev-Omni" in n for n in unmatched)
    assert any("Plumb-4B" in n for n in unmatched)


# --- jev decision index board -------------------------------------------------------


def test_jev_decision_index_parse(monkeypatch):
    """Fixture is row-trimmed from the live 114-model response (2026-10-09):
    the 4 OR-servable repros + 2 open-only repros, heavy per-benchmark result
    arrays dropped (parser reads engine/name/scores only)."""
    src = JevDecisionIndexSource()
    monkeypatch.setattr(
        "hull_core.model_selection.sources._get_json",
        lambda url, **_kw: _json_fixture("jev_decision_index.json"),
    )
    records = src.fetch()
    assert len(records) == 6
    clef = records["clef"]
    assert clef.score is not None and 0.0 < clef.score < 100.0
    assert clef.name == "Cloudflare clef"
    assert clef.board_version == "2026-10-07"
    assert src.board_updated_at == "2026-10-07T18:25:58Z"
    assert src.attribution and "jev-decision-index" in src.attribution
    # honest attribution: no license asserted (none declared in the space)
    assert "no license declared" in src.attribution


def test_jev_decision_index_joins_survivors_reports_open_repros():
    """Only the OR-servable open repros map (versioned alias table); the
    open-only repros report unmatched — a thin board post-filter BY DESIGN."""
    data = _json_fixture("jev_decision_index.json")
    records = {
        slugify(row["engine"]): SourceRecord(
            key=slugify(row["engine"]),
            name=row["name"],
            score=row["scores"]["balanced_skill"],
            raw=dict(row),
        )
        for row in data["models"]
    }
    or_rows = {
        "cloudflare/clef": _or_rec("cloudflare/clef"),
        "cloudflare/clef-flash": _or_rec("cloudflare/clef-flash"),
        "jaredpalmer/kev-4b": _or_rec("jaredpalmer/kev-4b"),
        "togethercomputer/tev1-4b-experimental": _or_rec("togethercomputer/tev1-4b-experimental"),
    }
    status: dict = {}
    cands = join_sources(or_rows, {"jev_decision_index": records}, status_out=status)
    by_slug = {c.or_slug: c for c in cands}
    assert by_slug["cloudflare/clef"].scores["jev_decision_index"] == pytest.approx(records["clef"].score)
    assert by_slug["togethercomputer/tev1-4b-experimental"].scores["jev_decision_index"] == pytest.approx(
        records["tev1-4b"].score
    )
    unmatched = status["jev_decision_index"]["unmatched_names"]
    # the 2 open-only repros report; nothing guessed onto OR routes
    assert len(unmatched) == 2


# --- aliases: jevals explicit entry + tilde alias_target rule ---------------------


def test_jevals_alias_table_explicit_wins_and_reports_unmatched():
    or_rows = {
        "typesafe/jev-1.13": _or_rec("typesafe/jev-1.13"),
        "google/gemini-3.8-flash": _or_rec("google/gemini-3.8-flash"),
    }
    records = {
        # explicit table entry wins (ladder cannot bridge typesafe-ai/jev)
        "typesafe-ai-jev": SourceRecord(key="typesafe-ai-jev", name="Jev", score=67.78),
        # vendor/org row joins directly through the ladder (exact slugified id)
        "google-gemini-3-8-flash": SourceRecord(key="google-gemini-3-8-flash", name="Gemini 3.8 Flash", score=90.0),
        # not OR-servable: reported, never guessed
        "not-on-or": SourceRecord(key="not-on-or", name="Not On Or", score=10.0),
    }
    status: dict = {}
    cands = join_sources(or_rows, {"jevals": records}, status_out=status)
    by_slug = {c.or_slug: c for c in cands}
    assert by_slug["typesafe/jev-1.13"].scores["jevals"] == pytest.approx(67.78)
    assert by_slug["google/gemini-3.8-flash"].scores["jevals"] == pytest.approx(90.0)
    assert status["jevals"]["unmatched_names"] == ["Not On Or"]
    # the table entry is the ONLY explicit jevals mapping — direct joins carry
    # the vendor rows, per the ticket's mapping table
    assert BOARD_ALIASES["jevals"] == {"typesafe-ai-jev": "typesafe/jev-1.13"}


def test_mercury_rows_stay_distinct():
    """jevals 'Mercury 2.5' joins only the exact inception/mercury-2.5 id;
    the different inception/mercury-decide route is never mapped to it."""
    or_rows = {
        "inception/mercury-2.5": _or_rec("inception/mercury-2.5"),
        "inception/mercury-decide": _or_rec("inception/mercury-decide"),
    }
    mercury_row = "inception-mercury-2-5"
    records = {
        mercury_row: SourceRecord(key=mercury_row, name="Mercury 2.5", score=55.0),
    }
    status: dict = {}
    cands = join_sources(or_rows, {"jevals": records}, status_out=status)
    by_slug = {c.or_slug: c for c in cands}
    assert by_slug["inception/mercury-2.5"].scores.get("jevals") == pytest.approx(55.0)
    assert "jevals" not in by_slug["inception/mercury-decide"].scores
    assert "unmatched_names" not in status["jevals"]


def test_tilde_alias_target_resolves_to_canonical():
    """The catalog's ~vendor/latest redirect row (alias_target.slug) resolves
    to the concrete route — never to itself."""
    tilde_row = {
        "id": "~typesafe/jev-latest",
        "name": "TypeSafe: Jev Latest",
        "canonical_slug": "~typesafe/jev-latest",
        "alias_target": {"name": "TypeSafe: Jev 1.13", "slug": "typesafe/jev-1.13"},
        "architecture": {"modality": "text->decisions"},
    }
    target_row = {
        "id": "typesafe/jev-1.13",
        "name": "TypeSafe: Jev 1.13",
        "canonical_slug": "typesafe/jev-1.13-20260917",
        "architecture": {"modality": "text->decisions"},
    }
    or_rows = {
        "~typesafe/jev-latest": SourceRecord(key="~typesafe/jev-latest", name="TypeSafe: Jev Latest", raw=tilde_row),
        "typesafe/jev-1.13": SourceRecord(key="typesafe/jev-1.13", name="TypeSafe: Jev 1.13", raw=target_row),
    }
    idx = _alias_index(or_rows)
    assert idx["~typesafe/jev-latest"] == "typesafe/jev-1.13"
    assert idx["typesafe-jev-latest"] == "typesafe/jev-1.13"
    assert idx["typesafe-jev-1-13"] == "typesafe/jev-1.13"
    # a board row keyed on the tilde route joins the canonical target
    records = {"typesafe-jev-latest": SourceRecord(key="typesafe-jev-latest", name="Jev Latest", score=42.0)}
    cands = join_sources(or_rows, {"board_y": records})
    by_slug = {c.or_slug: c for c in cands}
    assert by_slug["typesafe/jev-1.13"].scores["board_y"] == pytest.approx(42.0)
    assert "~typesafe/jev-latest" not in by_slug or "board_y" not in by_slug["~typesafe/jev-latest"].scores


# --- profile registration regression ----------------------------------------------


def test_decision_profile_registered_and_sources_resolve():
    profile = TASKS["decision"]
    assert profile.specialized_sources == ("jevals", "jevbench", "jev_decision_index", "judgemark_v4")
    assert profile.usage_sources == ("or_usage",)
    for name in profile.specialized_sources:
        assert name in SOURCE_REGISTRY, name
    # corroborate-only law is encoded in the profile comment/shape: judgemark
    # is one of >=2 boards required for a strong (rank-capable) candidate — a
    # judgemark-only model stays weak_evidence (enforced by candidates()).
    assert "judgemark_v4" in profile.specialized_sources
    assert "jevals" in profile.specialized_sources


def test_existing_profiles_unchanged_by_decision_wave():
    """Regression: the decision wave must not reshape any existing profile."""
    assert TASKS["translation"].specialized_sources == ("wmt24pp", "wmt25_gmtr")
    assert TASKS["embedding"].specialized_sources == (
        "mteb_classification",
        "mteb_retrieval",
        "mteb_sts",
        "agentset_elo",
        "hindsight_embeddings",
    )
    assert TASKS["rerank"].specialized_sources == ("mteb_reranking", "agentset_rerank", "hindsight_reranker")
    assert "decision" not in TASKS["translation"].specialized_sources
