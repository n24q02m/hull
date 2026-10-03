"""Tests for hull_core.llm.dspy_lm (VertexExpressLM + build_lm)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from litellm import Choices, Message, ModelResponse, Usage

from hull_core.llm.dspy_lm import VertexExpressLM, build_lm


def _fake_response(content: str = "ok") -> ModelResponse:
    return ModelResponse(
        choices=[Choices(finish_reason="stop", index=0, message=Message(content=content, role="assistant"))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="gemini-3.5-flash",
    )


def test_vertex_express_lm_forward_calls_adapter():
    lm = VertexExpressLM("vertex_express/gemini-3.5-flash", api_key="K", temperature=0.5, top_p=0.9)
    fake = _fake_response("ok")
    with patch("hull_core.llm.dspy_lm.completion_express", return_value=fake) as mock_adapter:
        out = lm("translate hi")
    # DSPy parses ModelResponse -> ["ok"].
    assert out == ["ok"]
    kw = mock_adapter.call_args.kwargs
    assert kw["model"] == "vertex_express/gemini-3.5-flash"
    assert kw["api_key"] == "K"
    assert kw["temperature"] == 0.5
    assert kw["top_p"] == 0.9
    # DSPy-internal knobs must be stripped before hitting the adapter.
    assert "cache" not in kw
    assert "rollout_id" not in kw


async def test_vertex_express_lm_aforward_calls_adapter():
    """The async path (aforward, via acall) routes through acompletion_express."""
    lm = VertexExpressLM(
        "vertex_express/gemini-3.5-flash",
        api_key="K",
        temperature=0.3,
        top_p=0.9,
        max_tokens=4096,
    )
    fake = _fake_response("da")
    with patch(
        "hull_core.llm.dspy_lm.acompletion_express",
        new=AsyncMock(return_value=fake),
    ) as mock_adapter:
        out = await lm.acall("translate hi")
    assert out == ["da"]
    kw = mock_adapter.call_args.kwargs
    assert kw["model"] == "vertex_express/gemini-3.5-flash"
    assert kw["api_key"] == "K"
    assert kw["max_tokens"] == 4096


def test_vertex_express_lm_output_ceiling_sized_per_call():
    """With output_ceiling set, an unset max_tokens is sized from the messages."""
    ceiling = lambda msgs: 12345  # noqa: E731
    lm = VertexExpressLM(
        "vertex_express/m",
        api_key="K",
        output_ceiling=ceiling,
        temperature=0.7,
        top_p=0.9,
    )
    fake = _fake_response()
    with patch("hull_core.llm.dspy_lm.completion_express", return_value=fake) as m:
        lm.forward(messages=[{"role": "user", "content": "x" * 100}])
    assert m.call_args.kwargs["max_tokens"] == 12345


def test_vertex_express_lm_explicit_max_tokens_wins_over_ceiling():
    lm = VertexExpressLM(
        "vertex_express/m",
        api_key="K",
        output_ceiling=lambda _msgs: 999,
        temperature=0.7,
        top_p=0.9,
    )
    fake = _fake_response()
    with patch("hull_core.llm.dspy_lm.completion_express", return_value=fake) as m:
        lm.forward(messages=[{"role": "user", "content": "hi"}], max_tokens=777)
    assert m.call_args.kwargs["max_tokens"] == 777


def test_vertex_express_lm_no_ceiling_leaves_max_tokens_unset():
    lm = VertexExpressLM("vertex_express/m", api_key="K", temperature=0.7, top_p=0.9)
    fake = _fake_response()
    with patch("hull_core.llm.dspy_lm.completion_express", return_value=fake) as m:
        lm.forward(messages=[{"role": "user", "content": "hi"}])
    assert m.call_args.kwargs["max_tokens"] is None


def test_vertex_express_lm_env_prefix_forwarded():
    lm = VertexExpressLM("vertex_express/m", api_key="K", env_prefix="KLPRISM_", temperature=0.5, top_p=0.9)
    fake = _fake_response()
    with patch("hull_core.llm.dspy_lm.completion_express", return_value=fake) as m:
        lm.forward(messages=[{"role": "user", "content": "hi"}])
    assert m.call_args.kwargs["env_prefix"] == "KLPRISM_"


# ---------------------------------------------------------------------------
# build_lm
# ---------------------------------------------------------------------------


def test_build_lm_vertex_express_returns_express_lm():
    lm = build_lm(
        "vertex_express/gemini-3.5-flash",
        env_prefix="KLPRISM_",
        temperature=0.7,
        top_p=0.9,
        thinking_budget=1024,
    )
    assert isinstance(lm, VertexExpressLM)
    assert lm.kwargs["thinking_budget"] == 1024


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


def test_build_lm_xai_routes_openai_compatible(monkeypatch):
    """xai -> openai/<model> on api.x.ai (litellm reads OPENAI_API_KEY shape)."""
    monkeypatch.setenv("KLPRISM_XAI_API_KEY", "xkey")
    monkeypatch.delenv("KLPRISM_CF_AI_GATEWAY_URL", raising=False)
    monkeypatch.delenv("KLPRISM_CF_AIG_RUN_TOKEN", raising=False)
    with patch("hull_core.llm.dspy_lm.dspy.LM") as mock_lm:
        build_lm("xai/grok-4.3", env_prefix="KLPRISM_", temperature=0.3, top_p=0.9)
    args, kwargs = mock_lm.call_args
    assert args[0] == "openai/grok-4.3"
    assert kwargs["api_base"] == "https://api.x.ai/v1"
    assert kwargs["api_key"] == "xkey"


def test_build_lm_xai_gateway_flip(monkeypatch):
    monkeypatch.setenv("KLPRISM_XAI_API_KEY", "xkey")
    monkeypatch.setenv("KLPRISM_CF_AI_GATEWAY_URL", "https://gw.example")
    monkeypatch.setenv("KLPRISM_CF_AIG_RUN_TOKEN", "tok")
    with patch("hull_core.llm.dspy_lm.dspy.LM") as mock_lm:
        build_lm("xai/grok-4.3", env_prefix="KLPRISM_", temperature=0.3, top_p=0.9)
    kwargs = mock_lm.call_args.kwargs
    assert kwargs["api_base"] == "https://gw.example/grok/v1"
    assert kwargs["extra_headers"] == {"cf-aig-authorization": "Bearer tok"}


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
