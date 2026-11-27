"""dspy >= 3.5 native lm15 routing tests (no litellm)."""

from __future__ import annotations

import sys

import dspy

from hull_core.llm import OPENROUTER_API_BASE
from hull_core.llm.dspy_lm import _routed_model, build_lm


def test_routed_model_preserves_native_and_defaults_to_openrouter():
    assert _routed_model("z-ai/glm-5") == ("openrouter", "openrouter/z-ai/glm-5")
    assert _routed_model("openrouter/anthropic/claude-sonnet-4") == (
        "openrouter",
        "openrouter/anthropic/claude-sonnet-4",
    )
    assert _routed_model("openrouter:z-ai/glm-5") == ("openrouter", "openrouter:z-ai/glm-5")
    assert _routed_model("typesafe/jev-1.13") == ("typesafe", "typesafe/jev-1.13")
    assert _routed_model("gpt-4o-mini") == ("openrouter", "openrouter/gpt-4o-mini")


def test_build_lm_uses_native_lm15_without_litellm(monkeypatch):
    monkeypatch.setenv("HULL_OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("HULL_LLM_API_BASE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    lm = build_lm(
        "z-ai/glm-5",
        temperature=0.2,
        top_p=0.95,
        max_tokens=256,
    )
    assert isinstance(lm, dspy.LM)
    assert lm.model == "openrouter/z-ai/glm-5"
    assert lm.kwargs["api_key"] == "sk-or-test"
    assert lm.kwargs["api_base"] == OPENROUTER_API_BASE
    assert lm.kwargs["extra_body"] == {"provider": {"data_collection": "deny"}}
    assert lm.engine == "lm15"
    assert "litellm" not in sys.modules
