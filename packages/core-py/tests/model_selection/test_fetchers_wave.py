"""Wave-6 fetcher tests — fixture files captured live 2026-10-07, no network.

Every TaskProfile-declared source name resolves in SOURCE_REGISTRY; each new
fetcher parses a trimmed real response; InHouseAnchor/AA/Unimplemented record
honest missing reasons; source_status plumbing through candidates() and the
publisher's degraded-source bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hull_core.model_selection as ms
from hull_core.model_selection import TaskProfile, candidates
from hull_core.model_selection.normalize import ModelCandidate, blend_quality
from hull_core.model_selection.sources import (
    SOURCE_REGISTRY,
    AaCapabilitySource,
    BFCLSource,
    BenchLMSource,
    BridgeSource,
    EqBench4Source,
    EqBenchCsvSource,
    GaiaSource,
    InHouseAnchorSource,
    LLMStatsSource,
    MangaBenchSource,
    SourceRecord,
    Tau2BenchORSource,
    UnimplementedSource,
    ValsSource,
    _parse_bfcl_csv,
    _parse_bridge_blob,
    _parse_benchlm_md,
    _parse_llmstats_table,
    _parse_manga_bench_table,
    _parse_or_bench_table,
    _parse_vals_table,
)
from hull_core.model_selection.tasks import TASKS

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _resp(payload=None, *, content: bytes | None = None) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    r.content = content if content is not None else b""
    r.raise_for_status.return_value = None
    return r


# --- Registry completeness (acceptance) ----------------------------------------


def test_every_profile_source_resolves_in_registry():
    """Zero silent skips: each TaskProfile source name has a registry entry."""
    missing = []
    for task_name, profile in TASKS.items():
        for src in (*profile.specialized_sources, *profile.aggregate_sources, *profile.usage_sources):
            if src not in SOURCE_REGISTRY:
                missing.append(f"{task_name}:{src}")
    assert not missing, missing


# --- vals.ai -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "source_name", "board"),
    [
        ("vals_index.html", "vals_index", "vals_index"),
        ("vals_medscribe.html", "vals_medscribe", "medscribe"),
        ("vals_cua_bench.html", "vals_cua_bench", "cua_bench"),
    ],
)
def test_vals_fixture_parse(fixture, source_name, board):
    recs = _parse_vals_table(_fixture(fixture).decode("utf-8"))
    assert len(recs) >= 2
    first = next(iter(recs.values()))
    assert first.name and isinstance(first.score, float) and 0 < first.score <= 100
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture(fixture)),
    ):
        got = ValsSource(source_name, board).fetch()
    assert set(got) == set(recs)


# --- benchlm -------------------------------------------------------------------


def test_benchlm_md_fixture_parse():
    recs = _parse_benchlm_md(_fixture("benchlm_index.md").decode("utf-8"))
    assert len(recs) >= 4
    assert recs["claude-opus-5-5"].score == 86.39
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture("benchlm_index.md")),
    ):
        assert set(BenchLMSource().fetch()) == set(recs)


# --- tau2 via OpenRouter benchmark pages ----------------------------------------


def test_tau2_or_fixture_parse_and_domain_mean():
    html = _fixture("or_tau2_airline.html").decode("utf-8")
    per_page = _parse_or_bench_table(html)
    assert len(per_page) >= 2
    slug, (acc, ci) = next(iter(per_page.items()))
    assert 0 < acc <= 100
    src = Tau2BenchORSource(benches=("tau2-bench-airline", "tau2-bench-retail"))
    # second domain adds a row for the same model -> score = mean of the two
    other = {k: (v[0] + 5.0, v[1]) for k, v in per_page.items()}
    with (
        patch("hull_core.model_selection.sources._get_bytes", side_effect=[html.encode(), html.encode()]),
        patch(
            "hull_core.model_selection.sources._parse_or_bench_table",
            side_effect=[per_page, other],
        ),
    ):
        recs = src.fetch()
    merged = recs[slug]
    assert merged.score == pytest.approx((acc + acc + 5.0) / 2)
    assert set(merged.raw["domains"]) == {"tau2-bench-airline", "tau2-bench-retail"}


def test_tau2_or_all_domains_dead_fail_open():
    src = Tau2BenchORSource(benches=("tau2-bench-airline",))
    with patch(
        "hull_core.model_selection.sources._get_bytes",
        side_effect=ConnectionError("down"),
    ):
        assert src.fetch() == {}


# --- llm-stats boards -----------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "name", "board"),
    [("llmstats_healthbench.html", "healthbench", "healthbench"), ("llmstats_ocrbench.html", "ocrbench", "ocrbench")],
)
def test_llmstats_fixture_parse(fixture, name, board):
    recs = _parse_llmstats_table(_fixture(fixture).decode("utf-8"))
    assert len(recs) >= 2
    first = next(iter(recs.values()))
    assert first.name and first.score is not None and 0 < first.score <= 1  # 0-1 scale
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture(fixture)),
    ):
        assert set(LLMStatsSource(name, board).fetch()) == set(recs)


# --- BFCL -----------------------------------------------------------------------


def test_bfcl_fixture_parse():
    text = _fixture("bfcl_overall.csv").decode("utf-8")
    recs = _parse_bfcl_csv(text)
    assert len(recs) >= 2
    top = recs["claude-opus-4-5-20251101"]
    assert top.score == 77.47
    assert top.raw["mode"] == "FC"
    with patch("httpx.Client.get", return_value=_resp(content=text.encode())):
        assert set(BFCLSource().fetch()) == set(recs)


# --- EQ-Bench family ------------------------------------------------------------


def test_eqbench_creative_v3_fixture_parse():
    src = EqBenchCsvSource(
        "eqbench_creative_v3",
        "https://eqbench.com/creative_writing.js",
        "leaderboardDataCreativeWritingV3",
        "elo_score",
    )
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture("eqbench_creative_v3.js")),
    ):
        recs = src.fetch()
    assert "gpt-6-astra" in recs  # leading '*' legacy marker dropped by slugify
    assert recs["gpt-6-astra"].score == 2173.3


def test_eqbench_longform_fixture_parse():
    src = EqBenchCsvSource(
        "eqbench_longform",
        "https://eqbench.com/creative_writing_longform.js",
        "leaderboardDataLongformV3",
        "overall_score_100",
    )
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture("eqbench_longform.js")),
    ):
        recs = src.fetch()
    assert "claude-sonnet-4-6" in recs
    assert recs["claude-sonnet-4-6"].score == 79.9


def test_eqbench4_fixture_parse():
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture("eqbench4_data.js")),
    ):
        recs = EqBench4Source().fetch()
    assert "claude-opus-5" in recs
    rec = recs["claude-opus-5"]
    assert rec.score == 1385.0
    assert rec.score_ci == pytest.approx((1412.0 - 1362.4) / 2)
    assert rec.board_version == "2026-07-26"


# --- BRIDGE ---------------------------------------------------------------------


def test_bridge_fixture_parse():
    data = json.loads(_fixture("bridge_zeroshot.json"))
    scores = _parse_bridge_blob(data)
    assert len(scores) >= 3
    assert all(0 <= v <= 100 for v in scores.values())
    src = BridgeSource(modes=("Zero-Shot",))
    with patch("hull_core.model_selection.sources._get_json", return_value=data):
        recs = src.fetch()
    assert set(recs) == set(scores)


def test_bridge_missing_mode_degrades_not_dies():
    data = json.loads(_fixture("bridge_zeroshot.json"))
    src = BridgeSource(modes=("Zero-Shot", "CoT"))
    with patch("hull_core.model_selection.sources._get_json", side_effect=[data, RuntimeError("404")]):
        recs = src.fetch()
    assert recs  # one mode fetched, the other skipped


# --- GAIA ------------------------------------------------------------------------


def _gaia_parquet_blob():
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    import io

    table = pa.table(
        {
            "model": ["AlphaAgent v0.1", "AutoGPT4", "AlphaAgent v0.1"],
            "score": [0.023, 0.049, 0.051],
            "date": ["2024-10-24", "2023-11-03", "2024-11-01"],
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def test_gaia_best_score_per_model():
    with patch("httpx.Client.get", return_value=_resp(content=_gaia_parquet_blob())):
        recs = GaiaSource().fetch()
    assert recs["alphaagent-v0-1"].score == 0.051  # best run wins
    assert recs["autogpt4"].score == 0.049


# --- AA capability -------------------------------------------------------------------


def test_aa_capability_no_key_records_missing(monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    src = AaCapabilitySource("aa_healthcare_index", "healthcare_and_medical")
    assert src.fetch() == {}
    assert src.missing_reason == "aa_api_key_required"


def test_aa_capability_parses_docs_field_names(monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    payload = {
        "data": [
            {
                "slug": "GPT-6",
                "name": "GPT-6",
                "evaluations": {
                    "artificial_analysis_healthcare_and_medical_index": 51.2,
                    "artificial_analysis_healthcare_and_medical_index_version": "v2.1",
                },
            },
            {"slug": "Null-Cap", "evaluations": {"artificial_analysis_healthcare_and_medical_index": None}},
        ]
    }
    with patch("httpx.Client.get", return_value=_resp(payload)) as get:
        recs = AaCapabilitySource("aa_healthcare_index", "healthcare_and_medical", api_key="k").fetch()
    assert get.call_args.kwargs["headers"]["x-api-key"] == "k"
    assert recs["GPT-6"].score == 51.2
    assert recs["GPT-6"].board_version == "v2.1"
    assert "Null-Cap" not in recs  # null capability = missing, never 0


# --- In-house anchors ----------------------------------------------------------------


def test_anchor_present_file_parses(tmp_path):
    anchor = tmp_path / "aiora_triage_eval.json"
    anchor.write_text(
        json.dumps(
            {
                "updated_at": "2026-10-01",
                "records": [
                    {"key": "google/gemini-4-argon", "name": "Gemini 4 Argon", "score": 72.5, "score_ci": 1.1},
                    {"key": "bad", "score": None},  # dropped: no score
                ],
            }
        ),
        encoding="utf-8",
    )
    recs = InHouseAnchorSource("aiora_triage_eval", root=tmp_path).fetch()
    assert set(recs) == {"google/gemini-4-argon"}
    assert recs["google/gemini-4-argon"].score == 72.5


def test_anchor_absent_file_records_not_published(tmp_path):
    src = InHouseAnchorSource("aiora_triage_eval", root=tmp_path)
    assert src.fetch() == {}
    assert src.missing_reason == "in_house_anchor_not_published"


def test_anchor_invalid_schema_records_reason(tmp_path):
    (tmp_path / "aiora_triage_eval.json").write_text("{not json", encoding="utf-8")
    src = InHouseAnchorSource("aiora_triage_eval", root=tmp_path)
    assert src.fetch() == {}
    assert src.missing_reason and src.missing_reason.startswith("anchor_")


def test_former_anchor_boards_are_now_public_fetchers():
    """wmt24pp / manga109_v2026 / mangavqa graduated from unpublished anchors
    to real public-board fetchers (2026-10-08); only aiora_triage_eval stays
    an in-house anchor."""
    assert isinstance(SOURCE_REGISTRY["aiora_triage_eval"], InHouseAnchorSource)
    assert isinstance(SOURCE_REGISTRY["wmt24pp"], LLMStatsSource)
    assert SOURCE_REGISTRY["wmt24pp"].board == "wmt24++"
    for name in ("manga109_v2026", "mangavqa"):
        assert isinstance(SOURCE_REGISTRY[name], MangaBenchSource)


# --- WMT24++ (llm-stats mirror) ------------------------------------------------------


def test_wmt24pp_fixture_parse_and_fetch():
    recs = _parse_llmstats_table(_fixture("llmstats_wmt24pp.html").decode("utf-8"))
    assert len(recs) >= 20
    first = next(iter(recs.values()))
    assert first.name and first.score is not None and 0 < first.score <= 1
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture("llmstats_wmt24pp.html")),
    ):
        fetched = LLMStatsSource("wmt24pp", "wmt24++").fetch()
    assert set(fetched) == set(recs)
    assert fetched["qwen3.7-max"].score == pytest.approx(0.858)


# --- Manga109-derived boards (project site) ------------------------------------------


def test_manga_bench_ocr_column():
    recs = _parse_manga_bench_table(_fixture("manga_bench.html").decode("utf-8"), score_col=1)
    # section rows (``Proprietary Models``) and the count table must not leak in
    assert "proprietary-models" not in recs and "qa-pairs" not in recs
    assert recs["mangalmm"].score == pytest.approx(71.5)
    assert recs["gpt-4o"].score == 0.0  # honest zeros stay (published data)
    assert len(recs) == 11


def test_manga_bench_vqa_column_and_fetch():
    recs = _parse_manga_bench_table(_fixture("manga_bench.html").decode("utf-8"), score_col=2)
    assert recs["gemini-2-5-flash"].score == pytest.approx(7.26)
    assert recs["mangalmm"].score == pytest.approx(6.68)  # "(Ours)" stripped from the name
    with patch(
        "httpx.Client.get",
        return_value=_resp(content=_fixture("manga_bench.html")),
    ):
        fetched = MangaBenchSource("mangavqa", score_col=2).fetch()
    assert set(fetched) == set(recs)
    assert all(r.board_version == "2026" for r in fetched.values())


def test_manga_bench_table_missing_records_reason():
    html = b"<html><table><tr><th>Count Type</th><th>Total</th></tr><tr><td>QA pairs</td><td>40,363</td></tr></table></html>"
    assert _parse_manga_bench_table(html.decode(), score_col=1) == {}
    with patch("httpx.Client.get", return_value=_resp(content=html)):
        src = MangaBenchSource("mangavqa", score_col=2)
        assert src.fetch() == {}
        assert src.missing_reason == "manga_bench_table_empty"


# --- Unimplemented + status plumbing -------------------------------------------------


def test_unimplemented_source_reports_reason():
    src = UnimplementedSource("medarena", "requires sign-in")
    assert src.fetch() == {}
    status: dict = {}
    ms._fetch_source(src, None, False, status)
    assert status["medarena"]["status"] == "unimplemented_access"
    assert status["medarena"]["reason"] == "requires sign-in"


class _StubSource:
    def __init__(self, name: str, records, missing_reason: str | None = None):
        self.name = name
        self.ttl_seconds = 3600
        self._records = records
        self.missing_reason = missing_reason

    def fetch(self):
        return self._records


def test_fetch_source_statuses_ok_missing_stale(tmp_path):
    ok = _StubSource("ok_src", {"m/a": SourceRecord(key="m/a", score=1.0)})
    missing = _StubSource("missing_src", {}, missing_reason="aa_api_key_required")
    status: dict = {}
    assert ms._fetch_source(ok, None, False, status)["m/a"].score == 1.0
    assert status["ok_src"]["status"] == "ok"
    assert status["ok_src"]["row_count"] == 1
    assert ms._fetch_source(missing, None, False, status) == {}
    assert status["missing_src"]["status"] == "missing"
    assert status["missing_src"]["reason"] == "aa_api_key_required"

    # stale fallback: cached snapshot served after an empty fetch
    import time as _time

    cache = ms.FileCache(tmp_path)
    blob = {"fetched_at": _time.time() - 999_999, "payload": {"m/old": {"key": "m/old", "score": 9.0}}}
    (tmp_path / "stale_src.json").write_text(json.dumps(blob), encoding="utf-8")
    stale = _StubSource("stale_src", {}, missing_reason="empty_fetch")
    recs = ms._fetch_source(stale, cache, False, status)
    assert recs["m/old"].score == 9.0
    assert status["stale_src"]["status"] == "stale"
    assert "stale cache" in status["stale_src"]["reason"]


def test_candidates_status_out_records_backbone_and_sources():
    task = TASKS["rerank"]  # mteb_reranking specialized
    sources = {
        "openrouter_models": _StubSource("openrouter_models", {}),
    }
    status: dict = {}
    assert candidates(task, sources=sources, status_out=status) == []
    assert status["openrouter_models"]["status"] == "missing"


def test_candidates_status_out_ok_rows():
    task = TASKS["rerank"]
    or_row = {
        "id": "m/a",
        "name": "m/a",
        "context_length": 64_000,
        "architecture": {"modality": "text->text"},
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": [],
        "_or_zdr_available": True,
    }
    sources = {
        "openrouter_models": _StubSource("openrouter_models", {"m/a": SourceRecord(key="m/a", raw=or_row)}),
        "mteb_v2_crossencoder": _StubSource("mteb_v2_crossencoder", {"m/a": SourceRecord(key="m/a", score=0.5)}),
    }
    status: dict = {}
    cands = candidates(task, sources=sources, status_out=status)
    assert status["openrouter_models"]["status"] == "ok"
    assert status["mteb_v2_crossencoder"]["status"] == "ok"
    assert len(cands) == 1


@pytest.mark.parametrize(
    ("profile", "board", "evidence"),
    [
        (
            "permissive",
            "aa_agentic_index",
            ("no longer publishes an Agentic Index", "artificial_analysis_agentic_index", "2026-11-20"),
        ),
        (
            "healthcare-advice",
            "open_medical_llm",
            ("openlifescienceai/results", "2025-01-29", "openai/gpt-4-0314", "openai/gpt-3.5-turbo-1106", "2026-11-20"),
        ),
    ],
)
def test_board_without_servable_rows_is_recorded_gap_without_fetch(profile, board, evidence):
    """aa_agentic_index: the AA Data API reference marks
    artificial_analysis_agentic_index deprecated and always null, so every
    fetch came back empty (empty_fetch). open_medical_llm: the frozen board
    holds no model OpenRouter serves today, so every fetch joined nothing
    (zero_join). The snapshot must record each gap as unimplemented_access with
    the probed upstream evidence, never call the board, and never invent rows."""
    or_row = {
        "id": "a/one",
        "name": "a/one",
        "context_length": 64_000,
        "architecture": {"modality": "text->text"},
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": [],
    }
    sources = {
        "openrouter_models": _StubSource("openrouter_models", {"a/one": SourceRecord(key="a/one", raw=or_row)}),
        board: SOURCE_REGISTRY[board],  # the shipped registry entry, unmodified
    }
    gap_profile = TaskProfile(
        name=f"{profile}-{board}",
        specialized_sources=(board,),
        aggregate_sources=(),
    )
    status: dict = {}
    with patch(
        "httpx.Client.get", side_effect=AssertionError("board without servable rows must not be fetched")
    ) as get:
        cands = candidates(gap_profile, sources=sources, status_out=status)
    entry = status[board]
    assert entry["status"] == "unimplemented_access"
    assert entry["row_count"] == 0
    for fragment in evidence:
        assert fragment in entry["reason"]
    get.assert_not_called()
    assert cands == []  # no board rows -> no scored candidate, nothing faked


# --- Weak evidence semantics ----------------------------------------------------------


def test_blend_quality_single_board_is_weak():
    task = TASKS["translation"]
    one = ModelCandidate(or_slug="a", scores={"artificial_analysis": 60.0})
    two = ModelCandidate(or_slug="b", scores={"arena": 60.0, "artificial_analysis": 60.0})
    zero = ModelCandidate(or_slug="c", scores={})
    blend_quality([one, two, zero], task)
    assert one.weak_evidence is True  # single AA board -> weak (the snapshot defect)
    assert two.weak_evidence is False  # two boards -> strong achievable
    assert zero.weak_evidence is True


# --- Publisher degraded bookkeeping ----------------------------------------------------


def _load_publisher():
    import importlib.util

    script = Path(__file__).parents[4] / "scripts" / "publish_candidates.py"
    spec = importlib.util.spec_from_file_location("publish_candidates", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fill_unscanned_profiles_records_empty_backbone_sources():
    publisher = _load_publisher()
    tasks = {"rerank": {"or_backbone_empty": True, "candidates": []}}
    status: dict = {}
    publisher._fill_unscanned_profiles(tasks, status, "2026-10-07T00:00:00+00:00")
    assert status["mteb_v2_crossencoder"]["status"] == "missing"
    assert status["mteb_v2_crossencoder"]["reason"] == "or_backbone_empty_profile_not_scanned"
