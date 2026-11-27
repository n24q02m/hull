"""OpenRouter httpx/SSRF LLM transport tests — ZDR, CF AI Gateway, cache_control, circuit breaker."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import httpx
import pytest

from hull_core.http import SSRFBlockedError
from hull_core.llm import (
    OPENROUTER_API_BASE,
    AsyncCompletionStream,
    CompletionResponse,
    EmbeddingResponse,
    GatewayCircuitBreaker,
    SystemOneResponse,
    TransportHTTPError,
    UsageMetrics,
    ZDRUnavailableError,
    acompletion,
    acompletion_text,
    aembedding,
    arerank,
    asystemone,
    cf_aig_headers,
    completion,
    completion_text,
    extract_usage_metrics,
    provider_params,
    reset_gateway_circuit_breaker,
    systemone,
)
from hull_core.llm._extra import require_litellm_extra
from hull_core.llm.transport import (
    SyncCompletionStream,
    _check_zdr_error_response,
    _endpoint_url,
    _normalize_openrouter_model,
    _provider_of,
    _validate_and_apply_zdr,
)


def _json_response(payload: dict, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _sse_response(events: list[dict]) -> httpx.Response:
    lines = [f"data: {json.dumps(ev)}\n\n" for ev in events]
    lines.append("data: [DONE]\n\n")
    return httpx.Response(200, content="".join(lines).encode("utf-8"), headers={"content-type": "text/event-stream"})


def test_no_litellm_imported_at_transport_import(monkeypatch):
    """hull_core.llm transport runs purely on httpx + OpenRouter API without importing litellm."""
    assert "litellm" not in sys.modules
    monkeypatch.setitem(sys.modules, "litellm", None)
    with pytest.raises(ImportError, match="litellm"):
        require_litellm_extra()


def test_cf_aig_headers_safe_defaults_and_edge_cache():
    default = cf_aig_headers()
    assert default["cf-aig-collect-log"] == "false"
    assert default["cf-aig-skip-cache"] == "true"
    assert "cf-aig-cache-ttl" not in default

    cached = cf_aig_headers(edge_cache=True, cache_ttl=120, custom_cache_key="k1")
    assert cached["cf-aig-collect-log"] == "false"
    assert cached["cf-aig-skip-cache"] == "false"
    assert cached["cf-aig-cache-ttl"] == "120"
    assert cached["cf-aig-custom-cache-key"] == "k1"


def test_provider_params_resolves_openrouter_key_and_base(monkeypatch):
    monkeypatch.setenv("HULL_OPENROUTER_API_KEY", "or-key-123")
    monkeypatch.delenv("HULL_OPENROUTER_API_BASE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.delenv("HULL_AI_GATEWAY_URL", raising=False)
    monkeypatch.delenv("AI_GATEWAY_URL", raising=False)
    assert provider_params("openrouter", env_prefix="HULL_") == {"api_key": "or-key-123"}

    monkeypatch.setenv("HULL_AI_GATEWAY_URL", "https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter/")
    params = provider_params("openrouter", env_prefix="HULL_")
    assert params == {
        "api_key": "or-key-123",
        "api_base": "https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter",
    }


def test_zdr_fail_closed_when_unavailable():
    with pytest.raises(ZDRUnavailableError):
        completion(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            zdr=True,
            zdr_available=False,
        )


def test_completion_sends_zdr_cf_headers_and_preserves_cache_control():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(
            {
                "id": "gen-1",
                "model": "anthropic/claude-sonnet-4",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.0015,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            }
        )

    transport = httpx.MockTransport(handler)
    cache_ctrl = {"type": "ephemeral", "ttl": "1h"}
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "System prompt", "cache_control": cache_ctrl}],
        },
        {"role": "user", "content": "Hello"},
    ]
    gateway_url = "https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter"
    resp = completion(
        model="openrouter/anthropic/claude-sonnet-4",
        messages=messages,
        api_key="sk-test",
        api_base=gateway_url,
        transport=transport,
    )
    assert isinstance(resp, CompletionResponse)
    assert resp.choices[0].message.content == "ok"
    assert isinstance(resp.usage_metrics, UsageMetrics)
    assert resp.usage_metrics.usd_cost == pytest.approx(0.0015)
    assert resp.usage_metrics.cached_tokens == 80
    assert resp.usage_metrics.zdr_enforced is True

    req = seen[0]
    assert str(req.url) == f"{gateway_url}/chat/completions"
    assert req.headers["cf-aig-collect-log"] == "false"
    assert req.headers["cf-aig-skip-cache"] == "true"
    body = json.loads(req.content)
    assert body["model"] == "anthropic/claude-sonnet-4"
    assert body["provider"] == {"data_collection": "deny"}
    assert body["messages"][0]["content"][0]["cache_control"] == cache_ctrl


@pytest.mark.asyncio
async def test_acompletion_streaming_emits_deltas_and_real_usd_usage():
    events = [
        {"choices": [{"delta": {"content": "Hello "}}]},
        {
            "choices": [{"delta": {"content": "world"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": 0.00042,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
        },
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(events)

    stream = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        stream=True,
        transport=httpx.MockTransport(handler),
    )
    assert isinstance(stream, AsyncCompletionStream)
    chunks = [chunk["choices"][0]["delta"]["content"] async for chunk in stream]
    assert "".join(chunks) == "Hello world"
    assert stream.collected_text == "Hello world"
    assert stream.usage_metrics.usd_cost == pytest.approx(0.00042)
    assert stream.usage_metrics.cached_tokens == 6
    assert stream.usage_metrics.zdr_enforced is True


@pytest.mark.asyncio
async def test_gateway_circuit_breaker_falls_back_direct_to_openrouter():
    reset_gateway_circuit_breaker()
    breaker = GatewayCircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if request.url.host == "gateway.ai.cloudflare.com":
            return httpx.Response(502, json={"error": "bad gateway"})
        return _json_response(
            {
                "id": "gen-direct",
                "model": "z-ai/glm-5",
                "choices": [{"message": {"role": "assistant", "content": "direct-ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0001},
            }
        )

    transport = httpx.MockTransport(handler)
    gateway_url = "https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter"
    r1 = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "1"}],
        api_key="sk-test",
        api_base=gateway_url,
        circuit_breaker=breaker,
        transport=transport,
    )
    r2 = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "2"}],
        api_key="sk-test",
        api_base=gateway_url,
        circuit_breaker=breaker,
        transport=transport,
    )
    # 3rd call skips gateway completely because breaker is now OPEN.
    r3 = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "3"}],
        api_key="sk-test",
        api_base=gateway_url,
        circuit_breaker=breaker,
        transport=transport,
    )
    assert isinstance(r1, CompletionResponse)
    assert isinstance(r2, CompletionResponse)
    assert isinstance(r3, CompletionResponse)
    assert [
        r1.choices[0].message.content,
        r2.choices[0].message.content,
        r3.choices[0].message.content,
    ] == ["direct-ok", "direct-ok", "direct-ok"]
    assert urls == [
        f"{gateway_url}/chat/completions",
        f"{OPENROUTER_API_BASE}/chat/completions",
        f"{gateway_url}/chat/completions",
        f"{OPENROUTER_API_BASE}/chat/completions",
        f"{OPENROUTER_API_BASE}/chat/completions",
    ]


@pytest.mark.asyncio
async def test_aembedding_arerank_and_systemone_endpoints():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/embeddings"):
            return _json_response(
                {
                    "model": "qwen/qwen3-embedding-8b",
                    "data": [{"embedding": [0.1, 0.2], "index": 0}],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4, "cost": 0.00001},
                }
            )
        if path.endswith("/rerank"):
            return _json_response(
                {
                    "model": "cohere/rerank-4-pro",
                    "results": [{"index": 1, "relevance_score": 0.92}],
                    "usage": {"prompt_tokens": 12, "total_tokens": 12, "cost": 0.0002},
                }
            )
        if path.endswith("/v1/systemone"):
            body = json.loads(request.content)
            assert body["provider"] == {"data_collection": "deny"}
            return _json_response(
                {
                    "model": "typesafe/jev-1.13",
                    "answers": {"q1": {"answer": "yes", "probability": 0.91}},
                    "usage": {"prompt_tokens": 30, "completion_tokens": 1, "cost": 0.0003},
                }
            )
        raise AssertionError(f"unexpected path: {path}")

    transport = httpx.MockTransport(handler)
    emb = await aembedding(
        model="qwen/qwen3-embedding-8b",
        input=["hello"],
        api_key="sk-test",
        transport=transport,
    )
    assert isinstance(emb, EmbeddingResponse)
    assert [item.embedding for item in emb.data] == [[0.1, 0.2]]
    assert emb.usage_metrics.zdr_enforced is True

    rerank = await arerank(
        model="cohere/rerank-4-pro",
        query="q",
        documents=["d0", "d1"],
        api_key="sk-test",
        transport=transport,
    )
    assert rerank == [(1, pytest.approx(0.92))]

    s1 = await asystemone(
        model="typesafe/jev-1.13",
        state="patient has fever",
        questions=[{"id": "q1", "text": "urgent?", "type": "yes_no"}],
        api_key="sk-test",
        transport=transport,
    )
    assert isinstance(s1, SystemOneResponse)
    assert s1.answers["q1"]["answer"] == "yes"
    assert s1.usage_metrics.usd_cost == pytest.approx(0.0003)

    s1_sync = systemone(
        model="typesafe/jev-1.13",
        state="patient has fever",
        questions=[{"id": "q1", "text": "urgent?", "type": "yes_no"}],
        api_key="sk-test",
        transport=transport,
    )
    assert s1_sync.answers["q1"]["probability"] == pytest.approx(0.91)


def test_ssrf_blocks_private_api_base_and_http_errors_surface():
    with pytest.raises(SSRFBlockedError):
        completion(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            api_base="http://169.254.169.254/latest",
        )

    transport = httpx.MockTransport(lambda _r: httpx.Response(400, text="bad model"))
    with pytest.raises(TransportHTTPError, match="returned 400"):
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            transport=transport,
        )


@pytest.mark.asyncio
async def test_acompletion_text_and_usage_extraction():
    transport = httpx.MockTransport(
        lambda _r: _json_response(
            {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.002,
                    "cache_read_input_tokens": 4,
                },
            }
        )
    )
    text = await acompletion_text(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "say hi"}],
        api_key="sk-test",
        transport=transport,
    )
    assert text == "hello"
    metrics = extract_usage_metrics(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002},
        zdr_enforced=False,
    )
    assert metrics.usd_cost == pytest.approx(0.002)
    assert metrics.zdr_enforced is False

    # Also verify non-gateway SSRF-safe path respects custom public api_base without tripping breaker.
    with patch("hull_core.llm.transport.vet_api_base", return_value="https://api.example.com/v1"):
        res = await acompletion(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "ok"}],
            api_key="sk-test",
            api_base="https://api.example.com/v1",
            transport=transport,
        )
        assert isinstance(res, CompletionResponse)
        assert res.choices[0].message.content == "hello"


def test_breaker_lifecycle_usage_headers_provider_and_url_helpers(monkeypatch):
    with pytest.raises(ValueError, match="failure_threshold"):
        GatewayCircuitBreaker(failure_threshold=0)

    now = [100.0]
    breaker = GatewayCircuitBreaker(failure_threshold=1, recovery_timeout=5.0, clock=lambda: now[0])
    assert breaker.is_open() is False
    breaker.record_failure()
    assert breaker.is_open() is True
    now[0] = 106.0
    assert breaker.is_open() is False
    breaker.record_success()
    assert breaker.is_open() is False
    breaker.record_failure()
    breaker.reset()
    assert breaker.is_open() is False

    empty_metrics = extract_usage_metrics(None, zdr_enforced=False)
    assert empty_metrics == UsageMetrics(zdr_enforced=False)
    flat_metrics = extract_usage_metrics({"cached_tokens": 3, "total_cost": 0.01}, zdr_enforced=True)
    assert flat_metrics.cached_tokens == 3
    assert flat_metrics.usd_cost == pytest.approx(0.01)
    top_cost = extract_usage_metrics({"usage": {"prompt_tokens": 2}, "cost": 0.005})
    assert top_cost.usd_cost == pytest.approx(0.005)

    hdrs = cf_aig_headers(extra_headers={"x-custom": "1"})
    assert hdrs["x-custom"] == "1"

    assert provider_params("") == {}
    assert provider_params("unknown") == {}
    monkeypatch.setenv("HULL_TYPESAFE_API_KEY", "ts-key")
    monkeypatch.setenv("HULL_TYPESAFE_API_BASE", "https://ts.example.com/v1/")
    assert provider_params("typesafe") == {"api_key": "ts-key", "api_base": "https://ts.example.com/v1"}

    assert _provider_of("openrouter:z-ai/glm-5") == "openrouter"
    assert _provider_of("typesafe/jev-1.13") == "typesafe"
    assert _provider_of("typesafe:jev-1.13") == "typesafe"
    assert _normalize_openrouter_model("openrouter/z-ai/glm-5") == "z-ai/glm-5"
    assert _normalize_openrouter_model("openrouter:z-ai/glm-5") == "z-ai/glm-5"

    assert _endpoint_url("https://api.example.com/v1", "/v1/systemone") == "https://api.example.com/v1/systemone"
    assert _endpoint_url("https://api.example.com/v1", "/v1/embeddings") == "https://api.example.com/v1/embeddings"


def test_zdr_validation_and_error_responses():
    payload: dict[str, object] = {"model": "z-ai/glm-5"}
    assert _validate_and_apply_zdr(payload, zdr=False, zdr_available=False, zdr_endpoints=[]) is False
    assert _validate_and_apply_zdr(payload, zdr=True, zdr_available=True, zdr_endpoints=["ep-1"]) is True
    payload2: dict[str, object] = {"model": "z-ai/glm-5"}
    assert _validate_and_apply_zdr(payload2, zdr=True, zdr_available=True, zdr_endpoints=[{"tag": "ZDR"}]) is True

    with pytest.raises(ZDRUnavailableError, match="no ZDR endpoint listed"):
        _validate_and_apply_zdr({"model": "m"}, zdr=True, zdr_available=True, zdr_endpoints=[])

    with pytest.raises(ZDRUnavailableError, match="data_collection='deny'"):
        _validate_and_apply_zdr(
            {"model": "m", "provider": {"data_collection": "allow"}},
            zdr=True,
            zdr_available=True,
            zdr_endpoints=None,
        )

    with pytest.raises(ZDRUnavailableError, match="rejected ZDR"):
        _check_zdr_error_response(404, "No endpoints found for zero data retention", "m")


def test_sync_completion_streaming_post_errors_and_text_helpers():
    sse = (
        ": ping\n\n"
        'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "!"}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.0001}}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")

    def stream_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gateway.ai.cloudflare.com":
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    breaker = GatewayCircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    breaker.record_failure()
    stream = completion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        api_base="https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter",
        circuit_breaker=breaker,
        stream=True,
        transport=httpx.MockTransport(stream_handler),
    )
    assert isinstance(stream, SyncCompletionStream)
    chunks = list(stream)
    assert len(chunks) == 2
    assert stream.collected_text == "Hi!"
    assert stream.usage_metrics.usd_cost == pytest.approx(0.0001)

    with pytest.raises(TransportHTTPError, match="returned 500"):
        completion(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            stream=True,
            transport=httpx.MockTransport(lambda _r: httpx.Response(500, text="server error")),
        )

    with pytest.raises(ValueError, match="streaming is not supported"):
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            stream=True,
        )

    assert (
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            transport=httpx.MockTransport(lambda _r: _json_response({"choices": []})),
        )
        == ""
    )

    # Sync _post_json gateway HTTPError + 502 fallback paths + open breaker skip
    sync_breaker = GatewayCircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
    calls: list[str] = []

    def sync_gw_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "gateway.ai.cloudflare.com":
            if len(calls) == 1:
                raise httpx.ConnectError("gw connect err", request=request)
            return httpx.Response(502, text="gw down")
        return _json_response({"choices": [{"message": {"content": "sync-ok"}}]})

    gw_url = "https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter"
    tx = httpx.MockTransport(sync_gw_handler)
    assert (
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "1"}],
            api_key="k",
            api_base=gw_url,
            circuit_breaker=sync_breaker,
            transport=tx,
        )
        == "sync-ok"
    )
    assert (
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "2"}],
            api_key="k",
            api_base=gw_url,
            circuit_breaker=sync_breaker,
            transport=tx,
        )
        == "sync-ok"
    )
    assert (
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "3"}],
            api_key="k",
            api_base=gw_url,
            circuit_breaker=sync_breaker,
            transport=tx,
        )
        == "sync-ok"
    )

    # Gateway success resets failure counter
    sync_ok_gw = completion_text(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "4"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=GatewayCircuitBreaker(failure_threshold=2),
        transport=httpx.MockTransport(lambda _r: _json_response({"choices": [{"message": {"content": "gw-ok"}}]})),
    )
    assert sync_ok_gw == "gw-ok"

    def always_connect_fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(TransportHTTPError, match="transport error"):
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "5"}],
            api_key="k",
            transport=httpx.MockTransport(always_connect_fail),
        )

    with pytest.raises(TransportHTTPError, match="transport error: down"):
        completion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "6"}],
            api_key="k",
            api_base=gw_url,
            circuit_breaker=GatewayCircuitBreaker(failure_threshold=2),
            transport=httpx.MockTransport(always_connect_fail),
        )


@pytest.mark.asyncio
async def test_async_error_fallback_stream_and_endpoint_edge_cases():
    gw_url = "https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter"
    breaker = GatewayCircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

    def stream_gw_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gateway.ai.cloudflare.com":
            return httpx.Response(502, text="gw 502")
        return _sse_response([{"choices": [{"delta": {"content": "fallback-stream"}}]}])

    stream = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "s"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=breaker,
        stream=True,
        transport=httpx.MockTransport(stream_gw_handler),
    )
    assert isinstance(stream, AsyncCompletionStream)
    _ = [chunk async for chunk in stream]
    assert stream.collected_text == "fallback-stream"

    # Async stream gateway HTTPError -> fallback -> direct succeeds, then open breaker skips gateway
    def stream_err_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gateway.ai.cloudflare.com":
            raise httpx.ConnectError("gw reset", request=request)
        return _sse_response([{"choices": [{"delta": {"content": "ok2"}}]}])

    stream2 = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "s2"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=breaker,
        stream=True,
        transport=httpx.MockTransport(stream_err_handler),
    )
    assert isinstance(stream2, AsyncCompletionStream)
    _ = [chunk async for chunk in stream2]
    assert breaker.is_open() is True
    stream3 = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "s3"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=breaker,
        stream=True,
        transport=httpx.MockTransport(stream_err_handler),
    )
    assert isinstance(stream3, AsyncCompletionStream)
    _ = [chunk async for chunk in stream3]
    assert stream3.collected_text == "ok2"

    # Async stream error status & transport error
    err_stream = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "e"}],
        api_key="k",
        stream=True,
        transport=httpx.MockTransport(lambda _r: httpx.Response(500, text="fail")),
    )
    assert isinstance(err_stream, AsyncCompletionStream)
    with pytest.raises(TransportHTTPError, match="returned 500"):
        _ = [chunk async for chunk in err_stream]

    def direct_conn_fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    conn_err_stream = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "e2"}],
        api_key="k",
        stream=True,
        transport=httpx.MockTransport(direct_conn_fail),
    )
    assert isinstance(conn_err_stream, AsyncCompletionStream)
    with pytest.raises(TransportHTTPError, match="stream error"):
        _ = [chunk async for chunk in conn_err_stream]

    all_gw_fail_stream = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "e3"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=GatewayCircuitBreaker(failure_threshold=2),
        stream=True,
        transport=httpx.MockTransport(direct_conn_fail),
    )
    assert isinstance(all_gw_fail_stream, AsyncCompletionStream)
    with pytest.raises(TransportHTTPError, match="stream error: boom"):
        _ = [chunk async for chunk in all_gw_fail_stream]

    with pytest.raises(TransportHTTPError, match="transport error"):
        await acompletion(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "e4"}],
            api_key="k",
            transport=httpx.MockTransport(direct_conn_fail),
        )

    with pytest.raises(TransportHTTPError, match="transport error: boom"):
        await acompletion(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "e5"}],
            api_key="k",
            api_base=gw_url,
            circuit_breaker=GatewayCircuitBreaker(failure_threshold=2),
            transport=httpx.MockTransport(direct_conn_fail),
        )

    # _apost_json gateway ConnectError -> fallback + gateway success reset
    gw_then_direct_ok = await acompletion_text(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "ok"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=GatewayCircuitBreaker(failure_threshold=2),
        transport=httpx.MockTransport(
            lambda r: (
                (_ for _ in ()).throw(httpx.ConnectError("x", request=r))
                if r.url.host == "gateway.ai.cloudflare.com"
                else _json_response({"choices": [{"message": {"content": "recovered"}}]})
            )
        ),
    )
    assert gw_then_direct_ok == "recovered"

    gw_direct_ok = await acompletion_text(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "ok"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=GatewayCircuitBreaker(failure_threshold=2),
        transport=httpx.MockTransport(lambda _r: _json_response({"choices": [{"message": {"content": "gw-async"}}]})),
    )
    assert gw_direct_ok == "gw-async"

    with pytest.raises(ValueError, match="streaming is not supported"):
        await acompletion_text(
            model="z-ai/glm-5", messages=[{"role": "user", "content": "hi"}], api_key="k", stream=True
        )

    # Embeddings dimensions + empty rerank + top_n + flat systemone responses
    seen_payloads: list[dict[str, object]] = []

    def misc_handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        if request.url.path.endswith("/embeddings"):
            return _json_response({"data": [{"embedding": [0.5], "index": 0}], "usage": {}})
        if request.url.path.endswith("/rerank"):
            return _json_response({"results": [{"index": 0, "relevance_score": 0.8}]})
        return _json_response({"id": "s1", "model": "typesafe/jev-1.13", "q1": {"answer": "no"}, "usage": {}})

    misc_tx = httpx.MockTransport(misc_handler)
    emb = await aembedding(model="qwen/qwen3-embedding-8b", input="hi", dimensions=64, api_key="k", transport=misc_tx)
    assert emb.data[0].embedding == [0.5]
    assert seen_payloads[-1]["dimensions"] == 64

    assert await arerank(model="cohere/rerank-4-pro", query="q", documents=[], api_key="k", transport=misc_tx) == []
    ranked = await arerank(
        model="cohere/rerank-4-pro", query="q", documents=["d0"], top_n=1, api_key="k", transport=misc_tx
    )
    assert ranked == [(0, pytest.approx(0.8))]
    assert seen_payloads[-1]["top_n"] == 1

    s1_async = await asystemone(
        model="typesafe/jev-1.13", state="s", questions=[{"id": "q1"}], api_key="k", transport=misc_tx
    )
    assert s1_async.answers == {"q1": {"answer": "no"}}
    s1_sync = systemone(
        model="typesafe/jev-1.13",
        state="s",
        questions=[{"id": "q1"}],
        api_key="k",
        transport=httpx.MockTransport(misc_handler),
    )
    assert s1_sync.answers == {"q1": {"answer": "no"}}


@pytest.mark.asyncio
async def test_remaining_transport_branches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CF_AIG_TOKEN", raising=False)

    assert cf_aig_headers(edge_cache=True) == {
        "cf-aig-collect-log": "false",
        "cf-aig-skip-cache": "false",
    }

    out = completion_text(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "empty-key"}],
        api_key="",
        zdr_endpoints=[{"zdr": False}, {"zdr": True}],
        transport=httpx.MockTransport(lambda _r: _json_response({"choices": [{"message": {"content": "ok-zdr"}}]})),
    )
    assert out == "ok-zdr"

    with pytest.raises(TransportHTTPError, match="returned 500"):
        await acompletion_text(
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "err"}],
            api_key="k",
            zdr_only=False,
            transport=httpx.MockTransport(lambda _r: httpx.Response(500, text="server-err")),
        )

    gw_url = "https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter"
    breaker = GatewayCircuitBreaker(failure_threshold=2)
    gw_stream = await acompletion(
        model="z-ai/glm-5",
        messages=[{"role": "user", "content": "s"}],
        api_key="k",
        api_base=gw_url,
        circuit_breaker=breaker,
        stream=True,
        transport=httpx.MockTransport(
            lambda _r: _sse_response([{"choices": [{"delta": {"content": "gw-stream-ok"}}]}])
        ),
    )
    assert isinstance(gw_stream, AsyncCompletionStream)
    _ = [chunk async for chunk in gw_stream]
    assert gw_stream.collected_text == "gw-stream-ok"
