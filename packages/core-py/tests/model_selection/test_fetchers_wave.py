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
from hull_core.model_selection import candidates
from hull_core.model_selection.normalize import ModelCandidate, blend_quality
from hull_core.model_selection.sources import (
    SOURCE_REGISTRY,
    AaAgenticIndexSource,
    AaCapabilitySource,
    BFCLSource,
    BenchLMSource,
    BridgeSource,
    EqBench4Source,
    EqBenchCsvSource,
    FloresSpeakleashSource,
    GaiaSource,
    InHouseAnchorSource,
    LLMStatsSource,
    OpenMedicalLLMSource,
    SourceRecord,
    Tau2BenchORSource,
    UnimplementedSource,
    ValsSource,
    _parse_bfcl_csv,
    _parse_bridge_blob,
    _parse_benchlm_md,
    _parse_flores_csv,
    _parse_llmstats_table,
    _parse_medhelm_group,
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
        for src in (*profile.specialized_sources, *profile.aggregate_sources):
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
        "hull_core.model_selection.sources.httpx.get",
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
        "hull_core.model_selection.sources.httpx.get",
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
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=_fixture(fixture)),
    ):
        assert set(LLMStatsSource(name, board).fetch()) == set(recs)


# --- MedHELM --------------------------------------------------------------------


def test_medhelm_group_fixture_parse():
    recs = _parse_medhelm_group(_fixture("medhelm_groups.json"))
    assert len(recs) >= 2
    first = next(iter(recs.values()))
    assert isinstance(first.score, float)
    # the fixture carries a 1-5 "jury score" column: it must be excluded from
    # the composite (values > 1.0) but stay visible in raw task values
    jury = [v for v in first.raw["tasks"].values() if v > 1.0]
    assert jury, "fixture lost its mixed-scale column"
    assert first.score <= 1.0
    assert first.raw["scored_tasks"] < len(first.raw["tasks"])


def test_medhelm_pick_latest_release():
    from hull_core.model_selection.sources import _latest_medhelm_release

    listing = {
        "prefixes": [
            {"name": "medhelm/benchmark_output/releases/v3.0.0/"},
            {"name": "medhelm/benchmark_output/releases/v4.0.0/"},
            {"name": "medhelm/benchmark_output/releases/v10.0.1/"},
        ]
    }
    with patch("hull_core.model_selection.sources._get_json", return_value=listing):
        assert _latest_medhelm_release() == "v10.0.1"
    with patch("hull_core.model_selection.sources._get_json", return_value={}):
        with pytest.raises(RuntimeError, match="no medhelm release"):
            _latest_medhelm_release()


# --- BFCL -----------------------------------------------------------------------


def test_bfcl_fixture_parse():
    text = _fixture("bfcl_overall.csv").decode("utf-8")
    recs = _parse_bfcl_csv(text)
    assert len(recs) >= 2
    top = recs["claude-opus-4-5-20251101"]
    assert top.score == 77.47
    assert top.raw["mode"] == "FC"
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_resp(content=text.encode())):
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
        "hull_core.model_selection.sources.httpx.get",
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
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=_fixture("eqbench_longform.js")),
    ):
        recs = src.fetch()
    assert "claude-sonnet-4-6" in recs
    assert recs["claude-sonnet-4-6"].score == 79.9


def test_eqbench4_fixture_parse():
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=_fixture("eqbench4_data.js")),
    ):
        recs = EqBench4Source().fetch()
    assert "claude-opus-5" in recs
    rec = recs["claude-opus-5"]
    assert rec.score == 1385.0
    assert rec.score_ci == pytest.approx((1412.0 - 1362.4) / 2)
    assert rec.board_version == "2026-07-26"


# --- FLORES (speakleash) ---------------------------------------------------------


def test_flores_fixture_parse():
    text = _fixture("flores_results.csv").decode("utf-8")
    recs = _parse_flores_csv(text)
    assert recs, "expected chrf means"
    first = next(iter(recs.values()))
    assert first.raw["n_tasks"] >= 1
    assert first.score is not None and 0 < first.score <= 100  # chrF scale
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_resp(content=text.encode())):
        assert set(FloresSpeakleashSource().fetch()) == set(recs)


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
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_resp(content=_gaia_parquet_blob())):
        recs = GaiaSource().fetch()
    assert recs["alphaagent-v0-1"].score == 0.051  # best run wins
    assert recs["autogpt4"].score == 0.049


# --- Open Medical LLM -------------------------------------------------------------


def test_open_medical_tree_and_results():
    tree = json.loads(_fixture("open_medical_tree.json"))
    result = json.loads(_fixture("open_medical_result.json"))
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = _resp(result)
    with (
        patch("hull_core.model_selection.sources._get_json", return_value=tree),
        patch("hull_core.model_selection.sources.httpx.Client", return_value=mock_client),
    ):
        recs = OpenMedicalLLMSource().fetch()
    assert recs["01-ai-yi-1-5-9b-32k"].score == pytest.approx((0.716 + 0.7095588235294118 + 0.78) / 3)
    assert mock_client.get.call_count == 3


def test_open_medical_empty_tree_records_missing():
    src = OpenMedicalLLMSource()
    with patch("hull_core.model_selection.sources._get_json", return_value=[]):
        assert src.fetch() == {}
    assert src.missing_reason == "open_medical_results_empty"


# --- AA capability + agentic -------------------------------------------------------


def test_aa_capability_no_key_records_missing(monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    src = AaCapabilitySource("aa_healthcare_index", "healthcare_and_medical")
    assert src.fetch() == {}
    assert src.missing_reason == "aa_api_key_absent"


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
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_resp(payload)) as get:
        recs = AaCapabilitySource("aa_healthcare_index", "healthcare_and_medical", api_key="k").fetch()
    assert get.call_args.kwargs["headers"]["x-api-key"] == "k"
    assert recs["GPT-6"].score == 51.2
    assert recs["GPT-6"].board_version == "v2.1"
    assert "Null-Cap" not in recs  # null capability = missing, never 0


def test_aa_agentic_without_key_never_reads_or_embedded(monkeypatch):
    """No AA_API_KEY -> honest missing gap; the OR-embedded copy is NOT fetched as evidence."""
    monkeypatch.delenv("AA_API_KEY", raising=False)
    with patch("hull_core.model_selection.sources.httpx.get") as get:
        recs = AaAgenticIndexSource().fetch()
    assert recs == {}
    get.assert_not_called()


# --- In-house anchors ---------------------------------------------------------------


def test_anchor_present_file_parses(tmp_path):
    anchor = tmp_path / "wmt24pp.json"
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
    recs = InHouseAnchorSource("wmt24pp", root=tmp_path).fetch()
    assert set(recs) == {"google/gemini-4-argon"}
    assert recs["google/gemini-4-argon"].score == 72.5


def test_anchor_absent_file_records_not_published(tmp_path):
    src = InHouseAnchorSource("mangavqa", root=tmp_path)
    assert src.fetch() == {}
    assert src.missing_reason == "in_house_anchor_not_published"


def test_anchor_invalid_schema_records_reason(tmp_path):
    (tmp_path / "manga109_v2026.json").write_text("{not json", encoding="utf-8")
    src = InHouseAnchorSource("manga109_v2026", root=tmp_path)
    assert src.fetch() == {}
    assert src.missing_reason and src.missing_reason.startswith("anchor_")


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
    missing = _StubSource("missing_src", {}, missing_reason="aa_api_key_absent")
    status: dict = {}
    assert ms._fetch_source(ok, None, False, status)["m/a"].score == 1.0
    assert status["ok_src"]["status"] == "ok"
    assert status["ok_src"]["row_count"] == 1
    assert ms._fetch_source(missing, None, False, status) == {}
    assert status["missing_src"]["status"] == "missing"
    assert status["missing_src"]["reason"] == "aa_api_key_absent"

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
    }
    sources = {
        "openrouter_models": _StubSource("openrouter_models", {"m/a": SourceRecord(key="m/a", raw=or_row)}),
        "mteb_reranking": _StubSource("mteb_reranking", {"m/a": SourceRecord(key="m/a", score=0.5)}),
    }
    status: dict = {}
    cands = candidates(task, sources=sources, status_out=status)
    assert status["openrouter_models"]["status"] == "ok"
    assert status["mteb_reranking"]["status"] == "ok"
    assert len(cands) == 1


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
    assert status["mteb_reranking"]["status"] == "missing"
    assert status["mteb_reranking"]["reason"] == "or_backbone_empty_profile_not_scanned"
