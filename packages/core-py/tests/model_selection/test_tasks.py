"""TaskProfile registry tests — 11/2026 leaderboards, 2-tier lm fallback, ZDR strictness."""

from __future__ import annotations

import pytest

from hull_core.model_selection.sources import SOURCE_REGISTRY
from hull_core.model_selection.tasks import LM_GENERAL_SOURCES, TASKS, TASK_ALIASES, get_task


def test_required_profiles_present():
    """Core task profiles + aliases resolve cleanly."""
    for name in (
        "translation",
        "story-gen",
        "permissive",
        "vision",
        "embedding",
        "rerank",
        "decision",
        "healthcare-advice",
        "aqi-advice",
        "classification",
        "agentic",
        "manga-text",
    ):
        assert name in TASKS, name


def test_all_declared_sources_exist_in_registry():
    """Every specialized/aggregate/usage source declared in TASKS resolves in SOURCE_REGISTRY."""
    for name, profile in TASKS.items():
        for src in (*profile.specialized_sources, *profile.aggregate_sources, *profile.usage_sources):
            assert src in SOURCE_REGISTRY, f"{name}: {src} missing from SOURCE_REGISTRY"


def test_discontinued_sources_removed_from_registry_and_tasks():
    """Discontinued boards (medhelm, hindsight_*, or_usage in defaults) are absent."""
    for removed in ("medhelm", "hindsight_embeddings", "hindsight_reranker"):
        assert removed not in SOURCE_REGISTRY
    for name, profile in TASKS.items():
        assert profile.usage_sources == (), f"{name} unexpectedly enables usage_sources"
        all_sources = (*profile.specialized_sources, *profile.aggregate_sources)
        for removed in ("or_usage", "medhelm", "hindsight_embeddings", "hindsight_reranker"):
            assert removed not in all_sources, f"{name} still references {removed}"


def test_decision_profile_uses_11_2026_decision_boards_only():
    p = get_task("decision")
    assert set(p.specialized_sources) == {
        "aa_decision_index_v2_1",
        "jev_decision_index",
        "s1mb",
        "system_one_bench",
    }
    assert p.aggregate_sources == ()
    assert p.constraints.require_zdr is True
    for proxy in ("tau2_bench_or", "bfcl", "arena_agent"):
        assert proxy not in p.specialized_sources
        assert proxy not in p.aggregate_sources


def test_embedding_and_rerank_boards_and_zdr():
    emb = get_task("embedding")
    assert set(emb.specialized_sources) == {"rteb_holdout", "mteb_v2", "vidore_v3"}
    assert emb.aggregate_sources == ()
    assert emb.constraints.require_zdr is True

    rerank = get_task("rerank")
    assert set(rerank.specialized_sources) == {"mteb_v2_crossencoder", "air_bench", "agentset_rerank"}
    assert rerank.aggregate_sources == ()
    assert rerank.constraints.require_zdr is True


def test_lm_profiles_use_two_tier_lm_general_fallback():
    expected_general = (
        "artificial_analysis",
        "epoch_eci",
        "arena",
        "livebench_v2",
        "scale_seal",
        "benchlm",
    )
    assert LM_GENERAL_SOURCES == expected_general
    for name in (
        "translation",
        "story-gen",
        "manga-text",
        "healthcare-advice",
        "aqi-advice",
        "classification",
        "agentic",
        "vision",
        "permissive",
    ):
        p = get_task(name)
        assert p.two_tier is True, name
        assert p.aggregate_sources == LM_GENERAL_SOURCES, name


def test_specialized_lm_profiles_wire_11_2026_boards():
    assert set(get_task("translation").specialized_sources) == {"wmt25_gmtr", "wmt24pp"}
    assert set(get_task("story-gen").specialized_sources) == {
        "eqbench_creative_v3",
        "eqbench_longform",
        "eqbench4",
    }
    assert set(get_task("healthcare-advice").specialized_sources) == {
        "aa_healthcare_index",
        "healthbench",
        "vals_medscribe",
        "vectara_hhem",
    }
    assert set(get_task("vision").specialized_sources) == {
        "openvlm",
        "ocrbench",
        "manga109_v2026",
        "mangavqa",
    }
    assert set(get_task("agentic").specialized_sources) == {
        "arena_agent",
        "livebench_v2_agentic",
        "scale_agentic",
    }
    assert "ugi" in get_task("permissive").specialized_sources


def test_zdr_required_for_sensitive_profiles():
    for name in ("embedding", "rerank", "healthcare-advice", "aqi-advice", "classification", "decision"):
        assert get_task(name).constraints.require_zdr is True, name


def test_embed_alias_and_unknown_task():
    assert get_task("embed") is TASKS["embedding"]
    assert TASK_ALIASES["embed"] == "embedding"
    assert get_task("vision").constraints.required_input_modality == "image"
    profile = get_task("translation")
    assert get_task(profile) is profile
    with pytest.raises(KeyError):
        get_task("nonexistent-task")
