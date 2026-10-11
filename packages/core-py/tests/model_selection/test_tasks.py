"""TaskProfile registry tests — ported profiles + hull additions (permissive, embed alias)."""

from __future__ import annotations

import pytest

from hull_core.model_selection.tasks import TASKS, TASK_ALIASES, get_task


def test_required_profiles_present():
    """Spec 2026-09-28 phase H requires these task profiles to exist."""
    for name in ("translation", "story-gen", "permissive", "vision", "embedding", "rerank"):
        assert name in TASKS, name


def test_permissive_profile_wires_ugi():
    """The permissive profile must carry UGI as its specialized source."""
    p = get_task("permissive")
    assert "ugi" in p.specialized_sources
    assert p.constraints.min_context > 0


def test_embed_alias_resolves_to_embedding():
    assert get_task("embed") is TASKS["embedding"]
    assert TASK_ALIASES["embed"] == "embedding"


def test_vision_requires_image_input():
    assert get_task("vision").constraints.required_input_modality == "image"


def test_get_task_passthrough_and_unknown():
    profile = get_task("translation")
    assert get_task(profile) is profile
    with pytest.raises(KeyError):
        get_task("nonexistent-task")


# --- boards-v2 (2026-10-09) ---------------------------------------------------


def test_every_profile_defaults_to_or_usage_tiebreak():
    """OR usage is wired as the quadrant tie-break feed for every profile."""
    for name, profile in TASKS.items():
        assert profile.usage_sources == ("or_usage",), name


def test_embedding_wires_closed_inclusive_boards():
    p = get_task("embedding")
    assert "agentset_elo" in p.specialized_sources  # closed-inclusive (MTEB is open-biased)
    assert "hindsight_embeddings" in p.specialized_sources  # optional third signal
    assert set(p.specialized_sources) >= {"mteb_classification", "mteb_retrieval", "mteb_sts"}


def test_rerank_wires_agentset_and_hindsight():
    p = get_task("rerank")
    assert set(p.specialized_sources) == {"mteb_reranking", "agentset_rerank", "hindsight_reranker"}


def test_classification_drops_mteb_classification_uses_proxy():
    p = get_task("classification")
    assert "mteb_classification" not in p.specialized_sources  # only joined embedding models
    assert "vals_legal_bench" in p.specialized_sources  # explicit proxy board


def test_translation_swaps_flores_for_wmt25():
    p = get_task("translation")
    assert "flores_speakleash" not in p.specialized_sources  # stale 9 months, removed
    assert "wmt25_gmtr" in p.specialized_sources  # annual closed-inclusive anchor
    assert "wmt24pp" in p.specialized_sources  # kept, labeled self-reported


def test_agentic_wires_arena_agent():
    p = get_task("agentic")
    assert "arena_agent" in p.specialized_sources
