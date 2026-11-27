"""Wave 2 (11/2026) leaderboard parser/fetcher, 2-tier selection, ZDR, and per-profile candidate tests."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hull_core.model_selection import candidates
from hull_core.model_selection.sources import (
    SOURCE_REGISTRY,
    AaDecisionIndexV21Source,
    AirBenchSource,
    ArenaSource,
    ArtificialAnalysisSource,
    EpochECISource,
    LiveBenchV2AgenticSource,
    LiveBenchV2Source,
    MtebV2CrossEncoderSource,
    MtebV2Source,
    OpenRouterModelsSource,
    OpenVLMSource,
    RtebHoldoutSource,
    S1MBSource,
    ScaleAgenticSource,
    ScaleSealSource,
    Source,
    SourceRecord,
    SystemOneBenchSource,
    VectaraHHEMSource,
    ViDoReV3Source,
    _parse_aa_decision_index_v2_1,
    _parse_air_bench_json,
    _parse_epoch_eci_csv,
    _parse_livebench_v2_json,
    _parse_mteb_v2_crossencoder_json,
    _parse_mteb_v2_json,
    _parse_openvlm_json,
    _parse_rteb_holdout_json,
    _parse_s1mb_json,
    _parse_scale_agentic_json,
    _parse_scale_seal_json,
    _parse_system_one_bench_json,
    _parse_vectara_hhem,
    _parse_vidore_v3_json,
)
from hull_core.model_selection.tasks import LM_GENERAL_SOURCES, TASKS, Constraints, TaskProfile


class _StaticSource:
    ttl_seconds = 3600

    def __init__(self, name: str, records: dict[str, SourceRecord]) -> None:
        self.name = name
        self._records = records

    def fetch(self) -> dict[str, SourceRecord]:
        return self._records


def _or_row(
    slug: str,
    *,
    modality: str = "text->text",
    segments: list[str] | None = None,
    zdr: bool | None = True,
    context: int = 128_000,
    params: list[str] | None = None,
) -> SourceRecord:
    raw = {
        "id": slug,
        "name": slug,
        "canonical_slug": slug,
        "context_length": context,
        "architecture": {"modality": modality},
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": params or ["tools", "structured_outputs"],
        "_or_segments": segments or ["text"],
    }
    if zdr is not None:
        raw["_or_zdr_available"] = zdr
    return SourceRecord(key=slug, name=slug, raw=raw)


# --- 11/2026 parser fixture unit tests --------------------------------------


def test_parse_decision_boards_11_2026():
    aa_payload = {
        "data": [
            {
                "slug": "typesafe-jev-1-13",
                "name": "Jev 1.13",
                "evaluations": {"artificial_analysis_decision_index_v2_1": 81.4},
            }
        ]
    }
    aa_recs = _parse_aa_decision_index_v2_1(aa_payload)
    assert aa_recs["typesafe-jev-1-13"].score == pytest.approx(81.4)
    assert aa_recs["typesafe-jev-1-13"].board_version == "v2.1"

    s1mb = _parse_s1mb_json({"models": [{"model": "cloudflare/clef", "s1mb_score": 77.2}]})
    assert s1mb["cloudflare-clef"].score == pytest.approx(77.2)

    s1b = _parse_system_one_bench_json([{"name": "inception/mercury-decide", "system_one_score": 79.5}])
    assert s1b["inception-mercury-decide"].score == pytest.approx(79.5)


def test_parse_embed_and_rerank_boards_11_2026():
    rteb = _parse_rteb_holdout_json({"results": [{"model_name": "qwen/qwen3-embedding-8b", "ndcg_at_10": 68.4}]})
    assert rteb["qwen-qwen3-embedding-8b"].score == pytest.approx(68.4)

    mteb2 = _parse_mteb_v2_json({"data": [{"model": "google/gemini-embedding-2", "mteb_v2_mean": 71.9}]})
    assert mteb2["google-gemini-embedding-2"].score == pytest.approx(71.9)
    assert mteb2["google-gemini-embedding-2"].board_version == "v2"

    vidore = _parse_vidore_v3_json({"models": [{"model": "voyageai/voyage-4", "ndcg_at_5": 84.1}]})
    assert vidore["voyageai-voyage-4"].score == pytest.approx(84.1)
    assert vidore["voyageai-voyage-4"].board_version == "v3"

    ce = _parse_mteb_v2_crossencoder_json({"data": [{"model": "cohere/rerank-4-pro", "map": 66.3}]})
    assert ce["cohere-rerank-4-pro"].score == pytest.approx(66.3)

    air = _parse_air_bench_json({"data": [{"model": "cohere/rerank-4-fast", "air_bench": 64.0}]})
    assert air["cohere-rerank-4-fast"].score == pytest.approx(64.0)


def test_parse_lm_general_and_specialized_11_2026():
    eci = _parse_epoch_eci_csv("model,eci,ci\nz-ai/glm-5,142.5,1.8\nanthropic/claude-opus-4.6,149.0,1.2\n")
    assert eci["z-ai-glm-5"].score == pytest.approx(142.5)
    assert eci["z-ai-glm-5"].score_ci == pytest.approx(1.8)

    lb_payload = {
        "release_date": "2026-11-15",
        "data": [{"model": "z-ai/glm-5", "global_average": 74.2, "agentic_average": 79.1}],
    }
    lb = _parse_livebench_v2_json(lb_payload, agentic=False)
    lb_ag = _parse_livebench_v2_json(lb_payload, agentic=True)
    assert lb["z-ai-glm-5"].score == pytest.approx(74.2)
    assert lb_ag["z-ai-glm-5"].score == pytest.approx(79.1)

    seal = _parse_scale_seal_json({"leaderboard": [{"model": "openai/gpt-5.4", "elo": 1310.0}]})
    assert seal["openai-gpt-5-4"].score == pytest.approx(1310.0)

    seal_ag = _parse_scale_agentic_json({"entries": [{"model": "anthropic/claude-sonnet-4.6", "pass_rate": 63.5}]})
    assert seal_ag["anthropic-claude-sonnet-4-6"].score == pytest.approx(63.5)

    vlm = _parse_openvlm_json({"openai/gpt-5.4": {"average": 78.9}, "google/gemini-3.5-pro": 77.1})
    assert vlm["openai-gpt-5-4"].score == pytest.approx(78.9)
    assert vlm["google-gemini-3-5-pro"].score == pytest.approx(77.1)

    hhem_csv = _parse_vectara_hhem("model,hallucination_rate\nz-ai/glm-5,0.032\n")
    assert hhem_csv["z-ai-glm-5"].score == pytest.approx(96.8)
    hhem_json = _parse_vectara_hhem(json.dumps([{"model": "openai/gpt-5.4", "factual_consistency": 97.5}]))
    assert hhem_json["openai-gpt-5-4"].score == pytest.approx(97.5)


def test_source_fetchers_11_2026_wire_endpoints(monkeypatch):
    # AaDecisionIndexV21Source requires AA_API_KEY
    monkeypatch.delenv("AA_API_KEY", raising=False)
    aa = AaDecisionIndexV21Source()
    assert aa.fetch() == {}
    assert aa.missing_reason == "aa_api_key_required"

    with patch(
        "hull_core.model_selection.sources._get_json",
        return_value={"data": [{"slug": "m/a", "decision_index_v2_1": 80.0}]},
    ):
        recs = AaDecisionIndexV21Source(api_key="aa-test").fetch()
        assert recs["m-a"].score == pytest.approx(80.0)

    with patch("hull_core.model_selection.sources._get_bytes", return_value=b"model,eci\nm/a,140\n"):
        assert EpochECISource().fetch()["m-a"].score == pytest.approx(140.0)
    with patch("hull_core.model_selection.sources._get_bytes", return_value=b"model,factual_consistency\nm/a,95\n"):
        assert VectaraHHEMSource().fetch()["m-a"].score == pytest.approx(95.0)

    json_sources = [
        (S1MBSource(), {"data": [{"model": "m/a", "score": 70.0}]}),
        (SystemOneBenchSource(), {"data": [{"model": "m/a", "score": 71.0}]}),
        (RtebHoldoutSource(), {"data": [{"model": "m/a", "score": 72.0}]}),
        (MtebV2Source(), {"data": [{"model": "m/a", "score": 73.0}]}),
        (ViDoReV3Source(), {"data": [{"model": "m/a", "score": 74.0}]}),
        (MtebV2CrossEncoderSource(), {"data": [{"model": "m/a", "score": 75.0}]}),
        (AirBenchSource(), {"data": [{"model": "m/a", "score": 76.0}]}),
        (LiveBenchV2Source(), {"data": [{"model": "m/a", "global_average": 77.0}]}),
        (LiveBenchV2AgenticSource(), {"data": [{"model": "m/a", "agentic_average": 78.0}]}),
        (ScaleSealSource(), {"data": [{"model": "m/a", "score": 79.0}]}),
        (ScaleAgenticSource(), {"data": [{"model": "m/a", "pass_rate": 80.0}]}),
        (OpenVLMSource(), {"data": [{"model": "m/a", "average": 81.0}]}),
    ]
    for src, payload in json_sources:
        with patch("hull_core.model_selection.sources._get_json", return_value=payload):
            out = src.fetch()
        assert "m-a" in out and out["m-a"].score is not None, src.name


# --- OpenRouter ZDR + Arena Style Control + AA v5.0.2 ------------------------


def test_openrouter_models_source_enriches_zdr_fail_open():
    def fake_get_json(url, **_kw):
        if url.endswith("/endpoints/zdr"):
            return {"data": [{"model_id": "a/zdr-ok:free"}, {"slug": "a/zdr-only"}]}
        if "output_modalities=" in url:
            return {"data": []}
        return {
            "data": [
                {"id": "a/zdr-ok", "name": "ZDR OK", "architecture": {"modality": "text->text"}},
                {"id": "a/no-zdr", "name": "No ZDR", "architecture": {"modality": "text->text"}},
            ]
        }

    with patch("hull_core.model_selection.sources._get_json", side_effect=fake_get_json):
        recs = OpenRouterModelsSource().fetch()
    assert recs["a/zdr-ok"].raw["_or_zdr_available"] is True
    assert recs["a/no-zdr"].raw["_or_zdr_available"] is False


def test_arena_prefers_style_control_and_aa_prefers_v502():
    rows = [
        {"category": "overall", "model_name": "Model A", "rating": 1350.0, "leaderboard_publish_date": "2026-11-20"},
        {
            "category": "overall_style_control",
            "model_name": "Model A",
            "rating": 1320.0,
            "leaderboard_publish_date": "2026-11-20",
        },
    ]
    with (
        patch("hull_core.model_selection.sources._get_bytes", return_value=b"pq"),
        patch("hull_core.model_selection.sources._read_parquet_rows", return_value=rows),
    ):
        arena = ArenaSource().fetch()
    assert arena["model-a"].score == pytest.approx(1320.0)

    aa_data = {
        "data": [
            {
                "slug": "m/a",
                "name": "M A",
                "evaluations": {
                    "artificial_analysis_intelligence_index_v5_0_2": 91.2,
                    "artificial_analysis_intelligence_index": 80.0,
                },
            }
        ]
    }
    with patch("hull_core.model_selection.sources._get_json", return_value=aa_data):
        aa = ArtificialAnalysisSource(api_key="test-key").fetch()
    assert aa["m/a"].score == pytest.approx(91.2)
    assert aa["m/a"].board_version == "v5.0.2"


# --- 2-tier lm selection + candidates() per profile --------------------------


def test_two_tier_lm_selection_uses_specialized_when_coverage_passes_else_lm_general():
    or_rows = {
        "a/one": _or_row("a/one"),
        "a/two": _or_row("a/two"),
        "a/three": _or_row("a/three"),
    }
    task = TaskProfile(
        name="custom-lm",
        specialized_sources=("spec_a", "spec_b"),
        aggregate_sources=LM_GENERAL_SOURCES,
        two_tier=True,
        specialized_coverage_threshold=0.6,
        constraints=Constraints(min_context=16_000),
    )

    # Case 1: specialized coverage = 2/3 >= 0.6 -> Tier 1 specialized boards win, lm_general ignored for a/one & a/two
    sources_high_cov = {
        "openrouter_models": _StaticSource("openrouter_models", or_rows),
        "spec_a": _StaticSource(
            "spec_a",
            {"a/one": SourceRecord(key="a/one", score=90.0), "a/two": SourceRecord(key="a/two", score=70.0)},
        ),
        "spec_b": _StaticSource(
            "spec_b",
            {"a/one": SourceRecord(key="a/one", score=88.0), "a/two": SourceRecord(key="a/two", score=72.0)},
        ),
        "artificial_analysis": _StaticSource(
            "artificial_analysis",
            {
                "a/one": SourceRecord(key="a/one", score=10.0),
                "a/two": SourceRecord(key="a/two", score=95.0),
                "a/three": SourceRecord(key="a/three", score=80.0),
            },
        ),
    }
    by_slug_t1 = {c.or_slug: c for c in candidates(task, sources=sources_high_cov)}
    assert by_slug_t1["a/one"].quality == pytest.approx(100.0)
    assert by_slug_t1["a/one"].openrouter_slug == "a/one"

    # Case 2: specialized coverage = 1/3 < 0.6 -> falls back cleanly to Tier 2 lm_general
    sources_low_cov = {
        "openrouter_models": _StaticSource("openrouter_models", or_rows),
        "spec_a": _StaticSource("spec_a", {"a/one": SourceRecord(key="a/one", score=99.0)}),
        "artificial_analysis": _StaticSource(
            "artificial_analysis",
            {"a/two": SourceRecord(key="a/two", score=95.0), "a/three": SourceRecord(key="a/three", score=50.0)},
        ),
    }
    by_slug_t2 = {c.or_slug: c for c in candidates(task, sources=sources_low_cov)}
    assert "a/one" not in by_slug_t2 or by_slug_t2["a/one"].quality == pytest.approx(0.0)
    assert by_slug_t2["a/two"].quality == pytest.approx(100.0)


@pytest.mark.parametrize("profile_name", sorted(TASKS))
def test_candidates_per_profile_runs_and_enforces_zdr(profile_name):
    profile = TASKS[profile_name]
    mod = "text+image->text" if profile.constraints.required_input_modality == "image" else "text->text"
    seg = (
        "embeddings"
        if profile_name == "embedding"
        else ("rerank" if profile_name == "rerank" else ("decisions" if profile_name == "decision" else "text"))
    )
    or_rows = {
        "org/good-zdr": _or_row("org/good-zdr", modality=mod, segments=[seg], zdr=True, context=200_000),
        "org/no-zdr": _or_row("org/no-zdr", modality=mod, segments=[seg], zdr=False, context=200_000),
    }
    sources: dict[str, Source] = {"openrouter_models": _StaticSource("openrouter_models", or_rows)}
    for idx, src_name in enumerate((*profile.specialized_sources, *profile.aggregate_sources)):
        assert src_name in SOURCE_REGISTRY
        sources[src_name] = _StaticSource(
            src_name,
            {
                "org/good-zdr": SourceRecord(key="org/good-zdr", score=80.0 + idx),
                "org/no-zdr": SourceRecord(key="org/no-zdr", score=95.0 + idx),
            },
        )
    out = candidates(profile, sources=sources)
    slugs = [c.or_slug for c in out]
    assert "org/good-zdr" in slugs
    if profile.constraints.require_zdr:
        assert "org/no-zdr" not in slugs
    else:
        assert "org/no-zdr" in slugs
