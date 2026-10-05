"""Tests for hull_core.llm.dspy_lm (build_lm)."""

from __future__ import annotations

from unittest.mock import patch

from hull_core.llm.dspy_lm import build_lm


def test_build_lm_openrouter_uses_provider_params(monkeypatch):
    monkeypatch.setenv("KLPRISM_OPENROUTER_API_KEY", "or-key")
    with patch("hull_core.llm.dspy_lm.dspy.LM") as mock_lm:
        build_lm("openrouter/z-ai/glm-5", env_prefix="KLPRISM_", temperature=0.7, top_p=0.9)
    args, kwargs = mock_lm.call_args
    assert args[0] == "openrouter/z-ai/glm-5"
    assert kwargs["api_key"] == "or-key"
    assert kwargs["temperature"] == 0.7


def test_build_lm_bare_model_defaults_to_openrouter(monkeypatch):
    monkeypatch.setenv("HULL_OPENROUTER_API_KEY", "k")
    with patch("hull_core.llm.dspy_lm.dspy.LM") as mock_lm:
        build_lm("glm-5-flash", temperature=0.7, top_p=0.9)
    assert mock_lm.call_args.args[0] == "openrouter/glm-5-flash"
    assert mock_lm.call_args.kwargs["api_key"] == "k"


def test_build_lm_prefixed_id_passes_verbatim():
    """A provider-prefixed id goes to dspy.LM unchanged (litellm generic)."""
    with patch("hull_core.llm.dspy_lm.dspy.LM") as mock_lm:
        build_lm("xai/grok-4.3", temperature=0.3, top_p=0.9, api_key="xkey")
    args, kwargs = mock_lm.call_args
    assert args[0] == "xai/grok-4.3"
    assert kwargs["api_key"] == "xkey"


def test_build_lm_vertex_ai_passes_explicit_kwargs():
    """vertex_project/vertex_location flow through as ordinary LM kwargs."""
    with patch("hull_core.llm.dspy_lm.dspy.LM") as mock_lm:
        build_lm(
            "vertex_ai/gemini-3.1-pro-preview",
            temperature=0.3,
            top_p=0.9,
            vertex_project="proj",
            vertex_location="us-central1",
        )
    args, kwargs = mock_lm.call_args
    assert args[0] == "vertex_ai/gemini-3.1-pro-preview"
    assert kwargs["vertex_project"] == "proj"
    assert kwargs["vertex_location"] == "us-central1"
