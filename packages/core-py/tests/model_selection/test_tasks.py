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
