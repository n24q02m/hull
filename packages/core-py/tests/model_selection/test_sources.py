"""model_selection fetchers + cache tests — mock httpx, no real network.

Ported from knowledge_core tests: OpenRouterModelsSource, ArtificialAnalysis
(key absent/present), UGI CSV parse, Arena parquet, fetch_endpoint_stats,
FileCache TTL/stale/corrupt, _fetch_source cache paths, enrich_uptime,
version guard, join edge paths.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

import hull_core.model_selection as ms
from hull_core.model_selection import candidates, enrich_uptime
from hull_core.model_selection.cache import FileCache
from hull_core.model_selection.normalize import (
    ModelCandidate,
    blend_quality,
    blended_cost_1m,
    join_sources,
    minmax_normalize,
    version_guard,
)
from hull_core.model_selection.sources import (
    ArenaSource,
    ArtificialAnalysisSource,
    OpenRouterModelsSource,
    SourceRecord,
    UGISource,
    _parse_csv_records,
    fetch_endpoint_stats,
)
from hull_core.model_selection.tasks import TaskProfile


def _resp(payload=None, *, content: bytes | None = None) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    r.content = content if content is not None else b""
    r.raise_for_status.return_value = None
    return r


# --- OpenRouterModelsSource ---------------------------------------------------


def test_openrouter_models_source_parses_rows():
    payload = {
        "data": [
            {"id": "m/a", "name": "Model A", "pricing": {"prompt": "1"}},
            {"id": "m/b"},
            {"name": "no id"},  # skip
        ]
    }
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_resp(payload)):
        recs = OpenRouterModelsSource().fetch()
    assert set(recs) == {"m/a", "m/b"}
    assert recs["m/a"].raw["pricing"]["prompt"] == "1"


def test_openrouter_models_source_merges_modality_segments():
    """Base catalog + embeddings + rerank segments merge by id (2026-10-08
    directive): embed/rerank candidates must appear without losing text rows."""

    def _by_url(url: str, **_kw):
        if "output_modalities=embeddings" in url:
            return _resp({"data": [{"id": "thenlper/gte-large", "name": "gte-large"}]})
        if "output_modalities=rerank" in url:
            return _resp({"data": [{"id": "cohere/rerank-4.5", "name": "rerank"}]})
        return _resp({"data": [{"id": "m/a", "name": "Model A"}]})

    with patch("hull_core.model_selection.sources.httpx.get", side_effect=_by_url):
        recs = OpenRouterModelsSource().fetch()
    assert set(recs) == {"m/a", "thenlper/gte-large", "cohere/rerank-4.5"}


def test_openrouter_models_source_segment_failure_degrades_not_dies():
    """A failing category segment drops only that segment; text rows survive."""

    def _by_url(url: str, **_kw):
        if "output_modalities=embeddings" in url:
            raise ConnectionError("embed segment down")
        if "output_modalities=rerank" in url:
            return _resp({"data": [{"id": "cohere/rerank-4.5"}]})
        return _resp({"data": [{"id": "m/a"}]})

    with patch("hull_core.model_selection.sources.httpx.get", side_effect=_by_url):
        recs = OpenRouterModelsSource().fetch()
    assert set(recs) == {"m/a", "cohere/rerank-4.5"}


def test_openrouter_models_source_list_payload():
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp([{"id": "m/x"}]),
    ):
        assert set(OpenRouterModelsSource().fetch()) == {"m/x"}


def test_source_fail_open_on_http_error():
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        side_effect=ConnectionError("down"),
    ):
        assert OpenRouterModelsSource().fetch() == {}


# --- ArtificialAnalysisSource -------------------------------------------------


def test_aa_source_no_key_returns_empty(monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    assert ArtificialAnalysisSource().fetch() == {}


def test_aa_source_parses_evaluations(monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    payload = {
        "data": [
            {
                "slug": "GPT-6",
                "name": "GPT-6",
                "evaluations": {
                    "artificial_analysis_intelligence_index": 52.3,
                    "artificial_analysis_index_version": "v4",
                },
            },
            {"name": "Flat Row", "intelligence_index": 40.0, "index_version": "v4"},
            {"slug": ""},  # no slug -> skip
            "not-a-dict",  # skip
        ]
    }
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_resp(payload)) as get:
        recs = ArtificialAnalysisSource(api_key="k").fetch()
    assert get.call_args.kwargs["headers"]["x-api-key"] == "k"
    assert recs["GPT-6"].score == 52.3  # dict key = raw slug, record.key = slugified
    assert recs["GPT-6"].board_version == "v4"
    assert recs["flat-row"].score == 40.0


# --- CSV sources --------------------------------------------------------------


def test_ugi_csv_parse():
    csv_text = "model,ugi\nModel X,88.0\n"
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=csv_text.encode()),
    ):
        recs = UGISource().fetch()
    assert recs["model-x"].score == 88.0


def test_ugi_space_mirror_schema_parse():
    """2026-10-07 mirror schema: BOM + 'author/model_name' + 'UGI <trophy>'."""
    csv_text = "\ufeffauthor/model_name,UGI 🏆,W/10 👍\nopenai/gpt-6,72.4,88.0\n"
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=csv_text.encode("utf-8")),
    ):
        recs = UGISource().fetch()
    assert recs["openai-gpt-6"].score == 72.4


def test_parse_csv_unrecognized_columns_returns_empty():
    assert _parse_csv_records("foo,bar\n1,2\n", name_cols=("model",), score_cols=("score",)) == {}


def test_parse_csv_empty_text():
    assert _parse_csv_records("", name_cols=("model",), score_cols=("score",)) == {}


def test_first_col_substring_fallback():
    csv_text = "Model Name,UGI Score\nX,10\n"
    recs = _parse_csv_records(csv_text, name_cols=("model",), score_cols=("ugi_score", "score"))
    assert recs["x"].score == 10.0


# --- ArenaSource --------------------------------------------------------------


def _arena_parquet_rows() -> list[dict]:
    """Rows in the lmarena-ai/leaderboard-dataset schema (category-tagged)."""
    return [
        {
            "model_name": "GPT-6",
            "organization": "openai",
            "rating": 1400.0,
            "rating_lower": 1388.0,
            "rating_upper": 1412.0,
            "category": "overall",
            "leaderboard_publish_date": "2026-10-02",
        },
        {
            "model_name": "NoScore",
            "rating": None,
            "category": "overall",
        },
        {
            "model_name": "",
            "rating": 1300.0,
            "category": "overall",
        },
        {
            "model_name": "GPT-6",  # other category view -> skipped
            "rating": 1234.0,
            "category": "creative_writing",
        },
    ]


def test_arena_parquet_parse_with_pandas():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_arena_parquet_rows())
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=buf.getvalue()),
    ):
        recs = ArenaSource().fetch()
    assert recs["gpt-6"].score == 1400.0
    assert recs["gpt-6"].score_ci == pytest.approx(12.0)  # (upper - lower) / 2
    assert "noscore" not in recs


def test_arena_parquet_parse_with_pyarrow():
    pa = pytest.importorskip("pyarrow")
    df = pa.table(
        {
            "model_name": ["GPT-6", "NoScore"],
            "rating": [1400.0, None],
            "rating_lower": [1388.0, None],
            "rating_upper": [1412.0, None],
            "category": ["overall", "overall"],
        }
    )
    buf = io.BytesIO()
    import pyarrow.parquet as pq

    pq.write_table(df, buf)
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=buf.getvalue()),
    ):
        recs = ArenaSource().fetch()
    assert recs["gpt-6"].score == 1400.0
    assert "noscore" not in recs


def test_arena_parquet_no_parser_raises_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    from hull_core.model_selection.sources import _read_parquet_rows

    with pytest.raises(RuntimeError, match="pandas or pyarrow"):
        _read_parquet_rows(b"not parquet")


def test_arena_parquet_bad_blob_fail_open():
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        return_value=_resp(content=b"not parquet"),
    ):
        assert ArenaSource().fetch() == {}


# --- fetch_endpoint_stats -----------------------------------------------------


def test_fetch_endpoint_stats_aggregates():
    payload = {
        "data": [
            {"uptime_last_1d": 99.0, "latency_last_30m": 300, "tag": "openai"},
            {"uptime_last_1d": 99.9, "latency_last_30m": 250, "tag": "zdr"},
            "junk-row",
            {"uptime_last_1d": "bad", "latency_last_30m": None},
        ]
    }
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_resp(payload)):
        stats = fetch_endpoint_stats("m/a")
    assert stats["uptime_1d"] == 99.9
    assert stats["zdr_available"] is True
    assert stats["latency_30m"] == 250


def test_fetch_endpoint_stats_fail_open():
    with patch(
        "hull_core.model_selection.sources.httpx.get",
        side_effect=TimeoutError("slow"),
    ):
        assert fetch_endpoint_stats("m/a") == {}


# --- FileCache ----------------------------------------------------------------


def test_file_cache_roundtrip_and_ttl(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("src/a b", {"k": {"key": "k"}})
    # source name is sanitized into a safe filename
    assert (tmp_path / "src_a_b.json").exists()
    assert cache.get("src/a b", ttl_seconds=3600) == {"k": {"key": "k"}}
    assert cache.get("src/a b", ttl_seconds=0) is None  # expired
    assert cache.get("src/a b", ttl_seconds=0, allow_stale=True) == {"k": {"key": "k"}}


def test_file_cache_miss_and_corrupt(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.get("absent", 3600) is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert cache.get("bad", 3600) is None
    (tmp_path / "noschema.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert cache.get("noschema", 3600) is None


def test_file_cache_write_failure_logged(tmp_path, caplog):
    cache = FileCache(tmp_path / "sub")
    with (
        caplog.at_level(logging.WARNING, logger="hull_core.model_selection.cache"),
        patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
    ):
        cache.set("s", {"a": 1})  # must not raise
    assert "cache write failed" in caplog.text


# --- _fetch_source cache paths -------------------------------------------------


class _StubSource:
    def __init__(self, name: str, records):
        self.name = name
        self.ttl_seconds = 3600
        self._records = records

    def fetch(self):
        if isinstance(self._records, Exception):
            raise self._records
        return self._records


def test_fetch_source_cache_hit_skips_fetch(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("s", {"m/a": {"key": "m/a", "score": 5.0}})
    src = _StubSource("s", ConnectionError("should not be called"))
    recs = ms._fetch_source(src, cache, refresh=False)
    assert recs["m/a"].score == 5.0


def test_fetch_source_refresh_bypasses_cache_and_writes(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("s", {"old": {"key": "old"}})
    src = _StubSource("s", {"new": SourceRecord(key="new", score=1.0)})
    recs = ms._fetch_source(src, cache, refresh=True)
    assert set(recs) == {"new"}
    cached = cache.get("s", 3600)
    assert cached is not None and cached["new"]["key"] == "new"


def test_fetch_source_empty_falls_back_to_stale(tmp_path):
    cache = FileCache(tmp_path)
    stale = {"m/old": {"key": "m/old", "score": 9.0}}
    blob = {"fetched_at": time.time() - 999_999, "payload": stale}
    (tmp_path / "s.json").write_text(json.dumps(blob), encoding="utf-8")
    src = _StubSource("s", {})  # empty fetch
    recs = ms._fetch_source(src, cache, refresh=False)
    assert recs["m/old"].score == 9.0


def test_fetch_source_plugin_raise_returns_empty(tmp_path):
    src = _StubSource("s", RuntimeError("plugin boom"))
    assert ms._fetch_source(src, FileCache(tmp_path), refresh=False) == {}


# --- candidates() registry edge paths ------------------------------------------


def _or_row(slug: str) -> dict:
    return {
        "id": slug,
        "name": slug,
        "context_length": 64_000,
        "architecture": {"modality": "text->text"},
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": ["tools"],
    }


def test_candidates_missing_backbone_source_returns_empty():
    task = TaskProfile(name="t")
    assert candidates(task, sources={"spec_board": _StubSource("spec_board", {})}) == []


# --- enrich_uptime --------------------------------------------------------------


def test_enrich_uptime_sets_fields_and_respects_limit():
    cands = [ModelCandidate(or_slug=f"m/{i}") for i in range(3)]
    with patch(
        "hull_core.model_selection.fetch_endpoint_stats",
        return_value={"uptime_1d": 99.5, "zdr_available": True},
    ) as stats:
        out = enrich_uptime(cands, limit=2)
    assert stats.call_count == 2
    assert out[0].uptime_1d == 99.5 and out[0].zdr_available is True
    assert out[2].uptime_1d is None


def test_enrich_uptime_empty_stats_skipped():
    cand = ModelCandidate(or_slug="m/a")
    with patch("hull_core.model_selection.fetch_endpoint_stats", return_value={}):
        enrich_uptime([cand])
    assert cand.uptime_1d is None


# --- normalize edge paths -------------------------------------------------------


def test_join_sources_skips_openrouter_key_and_embedded_aa():
    or_records = {
        "m/a": SourceRecord(
            key="m/a",
            name="m/a",
            raw={
                **_or_row("m/a"),
                "benchmarks": {
                    "artificial_analysis": {
                        "intelligence_index": 50.0,
                        "coding_index": 60.0,
                        "agentic_index": "bad",  # not a number -> skip
                    }
                },
            },
        )
    }
    source_records = {
        "openrouter_models": {"m/a": SourceRecord(key="m/a", score=999.0)},  # skipped
        "board": {
            "m/a": SourceRecord(key="m/a", score=80.0, score_ci=2.0, board_version="v4"),
        },
    }
    cands = join_sources(or_records, source_records)
    cand = cands[0]
    assert cand.scores["board"] == 80.0
    assert cand.score_cis["board"] == 2.0
    assert cand.board_versions["board"] == "v4"
    # catalog-embedded AA indexes are NEVER board evidence: quality must come
    # from fetched boards only (boards-first qualification).
    assert "artificial_analysis" not in cand.scores
    assert "aa_coding_index" not in cand.scores
    assert "aa_agentic_index" not in cand.scores
    assert cand.scores.get("openrouter_models") != 999.0


def test_embedded_aa_respects_wanted_and_existing():
    or_records = {
        "m/a": SourceRecord(
            key="m/a",
            name="m/a",
            raw={**_or_row("m/a"), "benchmarks": {"artificial_analysis": {"intelligence_index": 50.0}}},
        )
    }
    task = TaskProfile(name="t", aggregate_sources=("artificial_analysis",))
    # a board already has a score for this source -> embedded does not overwrite
    cands = join_sources(
        or_records,
        {"artificial_analysis": {"m/a": SourceRecord(key="m/a", score=70.0)}},
        task=task,
    )
    assert cands[0].scores["artificial_analysis"] == 70.0
    # task does not want aa_coding_index -> embedded not applied
    assert "aa_coding_index" not in cands[0].scores


def test_version_guard_drops_old_major():
    cands = [
        ModelCandidate(
            or_slug="a",
            scores={"aa": 10.0},
            score_cis={"aa": 1.0},
            board_versions={"aa": "v3"},
        ),
        ModelCandidate(
            or_slug="b",
            scores={"aa": 90.0},
            board_versions={"aa": "v4", "other": "noversion"},
        ),
    ]
    version_guard(cands)
    assert "aa" not in cands[0].scores
    assert "aa" not in cands[0].score_cis
    assert cands[1].scores["aa"] == 90.0


def test_minmax_single_value_gives_100():
    cands = [
        ModelCandidate(or_slug="a", scores={"b": 5.0}),
        ModelCandidate(or_slug="b", scores={"b": 5.0}),
        ModelCandidate(or_slug="c", scores={}),
    ]
    minmax_normalize(cands)
    assert cands[0].scores["b"] == 100.0 and cands[1].scores["b"] == 100.0
    assert cands[2].scores == {}


def test_blend_quality_ci_propagates():
    task = TaskProfile(name="t", specialized_sources=("spec",), aggregate_sources=("agg",))
    cands = [
        ModelCandidate(or_slug="a", scores={"agg": 80.0}, score_cis={"agg": 3.0}),
        ModelCandidate(or_slug="b", scores={}),
    ]
    blend_quality(cands, task)
    assert cands[0].quality == 80.0
    assert cands[0].quality_ci == 3.0
    assert cands[1].quality == 0.0 and cands[1].weak_evidence


def test_blended_cost_missing_inputs_are_none():
    task = TaskProfile(name="t")
    assert blended_cost_1m(ModelCandidate(or_slug="x", raw={}), task) is None
    cand = ModelCandidate(or_slug="x", raw={"pricing": {"prompt": "0.000001"}})
    assert blended_cost_1m(cand, task) is None
