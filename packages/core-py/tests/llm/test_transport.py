"""Tests for the unified LLM transport (dispatch + routing helpers)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hull_core.llm import transport


def _completion_response(content: str) -> MagicMock:
    """Build a litellm-shaped completion response object."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


# ---------------------------------------------------------------------------
# provider_params
# ---------------------------------------------------------------------------


def test_provider_params_openrouter_reads_prefixed_env(monkeypatch):
    monkeypatch.setenv("KLPRISM_OPENROUTER_API_KEY", "or-key")
    params = transport.provider_params("openrouter", env_prefix="KLPRISM_")
    assert params == {"api_key": "or-key"}


def test_provider_params_openrouter_base_override(monkeypatch):
    monkeypatch.setenv("HULL_OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("HULL_LLM_API_BASE_OPENROUTER", "https://gw.example/openrouter")
    params = transport.provider_params("openrouter")
    assert params["api_base"] == "https://gw.example/openrouter"


def test_provider_params_empty_env_yields_nothing(monkeypatch):
    for var in ("HULL_OPENROUTER_API_KEY", "HULL_LLM_API_BASE_OPENROUTER"):
        monkeypatch.delenv(var, raising=False)
    assert transport.provider_params("openrouter") == {}


def test_provider_params_xai_direct_when_gateway_incomplete(monkeypatch):
    monkeypatch.setenv("KLPRISM_XAI_API_KEY", "xkey")
    monkeypatch.setenv("KLPRISM_CF_AI_GATEWAY_URL", "https://gw.example")
    monkeypatch.delenv("KLPRISM_CF_AIG_RUN_TOKEN", raising=False)
    params = transport.provider_params("xai", env_prefix="KLPRISM_")
    # Only the api_key: gateway routing needs BOTH env vars.
    assert params == {"api_key": "xkey"}


def test_provider_params_xai_gateway_flip(monkeypatch):
    monkeypatch.setenv("KLPRISM_XAI_API_KEY", "xkey")
    monkeypatch.setenv("KLPRISM_CF_AI_GATEWAY_URL", "https://gw.example/")
    monkeypatch.setenv("KLPRISM_CF_AIG_RUN_TOKEN", "tok")
    params = transport.provider_params("xai", env_prefix="KLPRISM_")
    assert params["api_base"] == "https://gw.example/grok/v1"
    assert params["extra_headers"] == {"cf-aig-authorization": "Bearer tok"}


def test_provider_params_unknown_provider_empty():
    assert transport.provider_params("vertex_ai") == {}
    assert transport.provider_params("ollama") == {}


def test_provider_params_empty_prefix_reads_canonical(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "canonical")
    assert transport.provider_params("openrouter", env_prefix="") == {"api_key": "canonical"}


# ---------------------------------------------------------------------------
# cohere_routing
# ---------------------------------------------------------------------------


def test_cohere_routing_direct_without_gateway(monkeypatch):
    monkeypatch.setenv("KLPRISM_COHERE_API_KEY", "ck")
    monkeypatch.delenv("KLPRISM_CF_AI_GATEWAY_URL", raising=False)
    monkeypatch.delenv("KLPRISM_CF_AIG_RUN_TOKEN", raising=False)
    params = transport.cohere_routing("embed", env_prefix="KLPRISM_")
    assert params["api_key"] == "ck"
    assert params["api_base"] == "https://api.cohere.com/v2/embed"
    assert "extra_headers" not in params


def test_cohere_routing_rerank_path():
    params = transport.cohere_routing("rerank")
    assert params["api_base"] == "https://api.cohere.com/v1/rerank"


def test_cohere_routing_gateway_flip(monkeypatch):
    monkeypatch.setenv("KLPRISM_COHERE_API_KEY", "ck")
    monkeypatch.setenv("KLPRISM_CF_AI_GATEWAY_URL", "https://gw.example")
    monkeypatch.setenv("KLPRISM_CF_AIG_RUN_TOKEN", "tok")
    params = transport.cohere_routing("embed", env_prefix="KLPRISM_")
    assert params["api_base"] == "https://gw.example/cohere/v2/embed"
    assert params["headers"] == {"cf-aig-authorization": "Bearer tok"}
    assert params["extra_headers"] == params["headers"]


def test_cohere_routing_rejects_unknown_op():
    with pytest.raises(KeyError):
        transport.cohere_routing("chat")


# ---------------------------------------------------------------------------
# acompletion / completion
# ---------------------------------------------------------------------------


async def test_acompletion_litellm_passthrough():
    resp = _completion_response("hello")
    with patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)) as m:
        out = await transport.acompletion(
            model="vertex_ai/gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
        )
    assert out is resp
    assert m.call_args.kwargs["model"] == "vertex_ai/gemini-3.1-pro-preview"
    assert m.call_args.kwargs["temperature"] == 0.3


async def test_acompletion_injects_env_provider_params(monkeypatch):
    monkeypatch.setenv("KLPRISM_OPENROUTER_API_KEY", "or-key")
    resp = _completion_response("x")
    with patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)) as m:
        await transport.acompletion(model="openrouter/z-ai/glm", messages=[], env_prefix="KLPRISM_")
    assert m.call_args.kwargs["api_key"] == "or-key"


async def test_acompletion_explicit_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("HULL_OPENROUTER_API_KEY", "env-key")
    resp = _completion_response("x")
    with patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)) as m:
        await transport.acompletion(model="openrouter/m", messages=[], api_key="explicit")
    assert m.call_args.kwargs["api_key"] == "explicit"


async def test_acompletion_vertex_express_routes_to_adapter():
    resp = _completion_response("hola")
    with (
        patch("hull_core.llm.transport.acompletion_express", new=AsyncMock(return_value=resp)) as m,
        patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock()) as litellm_mock,
    ):
        out = await transport.acompletion(
            model="vertex_express/gemini-3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            api_key="K",
            env_prefix="KLPRISM_",
            response_format={"type": "json_object"},
        )
    assert out is resp
    litellm_mock.assert_not_called()
    kw = m.call_args.kwargs
    assert kw["model"] == "vertex_express/gemini-3.5-flash"
    assert kw["api_key"] == "K"
    assert kw["env_prefix"] == "KLPRISM_"
    assert kw["response_format"] == {"type": "json_object"}


def test_completion_sync_litellm():
    resp = _completion_response("ok")
    with patch("hull_core.llm.transport.litellm.completion", return_value=resp) as m:
        out = transport.completion(model="openai/gpt", messages=[], temperature=0.1)
    assert out is resp
    assert m.call_args.kwargs["temperature"] == 0.1


def test_completion_vertex_express_routes_to_adapter():
    resp = _completion_response("ok")
    with (
        patch("hull_core.llm.transport.completion_express", return_value=resp) as m,
        patch("hull_core.llm.transport.litellm.completion") as litellm_mock,
    ):
        out = transport.completion(model="vertex_express/m", messages=[{"role": "user", "content": "hi"}])
    assert out is resp
    litellm_mock.assert_not_called()
    assert m.call_args.kwargs["model"] == "vertex_express/m"


# ---------------------------------------------------------------------------
# *_text helpers
# ---------------------------------------------------------------------------


async def test_acompletion_text_returns_content():
    resp = _completion_response("the answer")
    with patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)):
        out = await transport.acompletion_text(model="openai/gpt", messages=[])
    assert out == "the answer"


async def test_acompletion_text_none_content_empty():
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=None))]
    with patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)):
        out = await transport.acompletion_text(model="openai/gpt", messages=[])
    assert out == ""


def test_completion_text_returns_content():
    resp = _completion_response("done")
    with patch("hull_core.llm.transport.litellm.completion", return_value=resp):
        assert transport.completion_text(model="openai/gpt", messages=[]) == "done"


# ---------------------------------------------------------------------------
# api_base vetting (SSRF)
# ---------------------------------------------------------------------------


async def test_acompletion_vets_custom_api_base():
    resp = _completion_response("ok")
    with (
        patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)),
        patch("hull_core.llm.transport.vet_api_base", side_effect=ValueError("blocked")) as vet,
    ):
        with pytest.raises(ValueError, match="blocked"):
            await transport.acompletion(model="openai/gpt", messages=[], api_base="http://10.0.0.1:11434/v1")
    vet.assert_called_once_with("http://10.0.0.1:11434/v1")


async def test_acompletion_skips_vet_without_api_base():
    resp = _completion_response("ok")
    with (
        patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)),
        patch("hull_core.llm.transport.vet_api_base") as vet,
    ):
        await transport.acompletion(model="openai/gpt", messages=[])
    vet.assert_not_called()


async def test_acompletion_vets_env_derived_api_base(monkeypatch):
    """An api_base from env provider_params is vetted the same as an explicit one."""
    monkeypatch.setenv("HULL_OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("HULL_LLM_API_BASE_OPENROUTER", "https://gw.example/or")
    resp = _completion_response("ok")
    with (
        patch("hull_core.llm.transport.litellm.acompletion", new=AsyncMock(return_value=resp)) as m,
        patch("hull_core.llm.transport.vet_api_base", side_effect=lambda u: u) as vet,
    ):
        await transport.acompletion(model="openrouter/m", messages=[])
    vet.assert_called_once_with("https://gw.example/or")
    assert m.call_args.kwargs["api_base"] == "https://gw.example/or"


# ---------------------------------------------------------------------------
# aembedding
# ---------------------------------------------------------------------------


async def test_aembedding_forwards():
    resp = MagicMock()
    with patch("hull_core.llm.transport.litellm.aembedding", new=AsyncMock(return_value=resp)) as m:
        out = await transport.aembedding(model="cohere/embed-v4.0", input=["a", "b"], dimensions=1024)
    assert out is resp
    kw = m.call_args.kwargs
    assert kw["model"] == "cohere/embed-v4.0"
    assert kw["input"] == ["a", "b"]
    assert kw["dimensions"] == 1024


# ---------------------------------------------------------------------------
# arerank
# ---------------------------------------------------------------------------


def _rerank_response(pairs: list[tuple[int, float]]) -> MagicMock:
    resp = MagicMock()
    resp.results = [{"index": i, "relevance_score": s} for i, s in pairs]
    return resp


async def test_arerank_normalizes_and_sorts():
    resp = _rerank_response([(0, 0.1), (1, 0.9), (2, 0.5)])
    with patch("hull_core.llm.transport.litellm.arerank", new=AsyncMock(return_value=resp)) as m:
        out = await transport.arerank(
            model="cohere/rerank-v3.5",
            query="q",
            documents=["a", "b", "c"],
            top_n=3,
        )
    assert out == [(1, 0.9), (2, 0.5), (0, 0.1)]
    assert m.call_args.kwargs["model"] == "cohere/rerank-v3.5"
    assert m.call_args.kwargs["top_n"] == 3


async def test_arerank_empty_documents_short_circuits():
    with patch("hull_core.llm.transport.litellm.arerank", new=AsyncMock()) as m:
        out = await transport.arerank(model="cohere/rerank-v3.5", query="q", documents=[])
    assert out == []
    m.assert_not_called()


async def test_arerank_attr_shaped_results():
    """litellm may return objects rather than dicts on .results."""
    resp = MagicMock()
    r0 = MagicMock()
    r0.index = 2
    r0.relevance_score = 0.7
    resp.results = [r0]
    with patch("hull_core.llm.transport.litellm.arerank", new=AsyncMock(return_value=resp)):
        out = await transport.arerank(model="jina_ai/m0", query="q", documents=["a"])
    assert out == [(2, 0.7)]


async def test_arerank_explicit_key_and_base_forwarded():
    resp = _rerank_response([(0, 1.0)])
    with (
        patch("hull_core.llm.transport.litellm.arerank", new=AsyncMock(return_value=resp)) as m,
        patch("hull_core.llm.transport.vet_api_base", side_effect=lambda u: u),
    ):
        await transport.arerank(
            model="cohere/rerank-v3.5",
            query="q",
            documents=["a"],
            api_key="k",
            api_base="https://gw.example/cohere/v1/rerank",
            headers={"cf-aig-authorization": "Bearer t"},
        )
    kw = m.call_args.kwargs
    assert kw["api_key"] == "k"
    assert kw["api_base"] == "https://gw.example/cohere/v1/rerank"
    assert kw["headers"] == {"cf-aig-authorization": "Bearer t"}
