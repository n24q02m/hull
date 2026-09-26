"""OpenAI-spec wire shapes via MockTransport + SSRF policy enforcement."""

from __future__ import annotations

import json

import httpx
import pytest

from hull_core.config.models import ModelCell
from hull_core.http.ssrf import SSRFBlockedError
from hull_core.providers.openai_spec import OpenAICompatClient, ProviderError


@pytest.fixture(autouse=True)
def _fake_public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every hostname to a public IP so constructor vetting passes.

    The wire shapes under test use MockTransport, so no real network happens;
    only the constructor's DNS-pinning vet needs a resolvable host.
    """
    import socket

    def fake_getaddrinfo(host, port, *args, **kwargs):  # noqa: ANN001
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def _cell(**kw) -> ModelCell:
    defaults = {
        "task": "embed",
        "base_url": "https://llm.example.test/v1",
        "api_key": "sk-test",
        "model": "test-model",
    }
    defaults.update(kw)
    return ModelCell(**defaults)


def _mock_handler(routes: dict[str, dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = routes.get(request.url.path)
        if payload is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=payload)

    return handler


async def test_embeddings_wire_shape() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0.3, 0.4]}, {"index": 0, "embedding": [0.1, 0.2]}]},
        )

    client = OpenAICompatClient(_cell(), transport=httpx.MockTransport(handler))
    try:
        vectors = await client.embeddings(["hello", "world"])
    finally:
        await client.aclose()
    assert seen["path"] == "/v1/embeddings"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == {"model": "test-model", "input": ["hello", "world"]}
    # results re-ordered by index even when the provider shuffles them
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embeddings_dimensions_and_extra_passthrough() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1]}]},
        )

    client = OpenAICompatClient(_cell(), transport=httpx.MockTransport(handler))
    try:
        vectors = await client.embeddings(
            ["hello"], dimensions=1024, input_type="search_query"
        )
    finally:
        await client.aclose()
    # storage-width selection + provider-specific body fields travel verbatim
    assert seen["body"] == {
        "model": "test-model",
        "input": ["hello"],
        "dimensions": 1024,
        "input_type": "search_query",
    }
    assert vectors == [[0.1]]


async def test_embeddings_reserved_keys_rejected() -> None:
    client = OpenAICompatClient(
        _cell(), transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    )
    try:
        with pytest.raises(ValueError, match="cannot override"):
            await client.embeddings(["x"], model="evil-model")
        with pytest.raises(ValueError, match="cannot override"):
            await client.embeddings(["x"], input=["evil"])
    finally:
        await client.aclose()


async def test_chat_wire_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "pong"}}]})

    client = OpenAICompatClient(_cell(task="chat"), transport=httpx.MockTransport(handler))
    try:
        assert await client.chat([{"role": "user", "content": "ping"}], temperature=0) == "pong"
    finally:
        await client.aclose()


async def test_rerank_wire_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        body = json.loads(request.content)
        assert body["top_n"] == 1
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.2}]},
        )

    client = OpenAICompatClient(_cell(task="rerank"), transport=httpx.MockTransport(handler))
    try:
        results = await client.rerank("q", ["a", "b"], top_n=1)
    finally:
        await client.aclose()
    assert results[0]["relevance_score"] == 0.9


async def test_provider_error_carries_status() -> None:
    client = OpenAICompatClient(
        _cell(),
        transport=httpx.MockTransport(lambda request: httpx.Response(429, json={"error": "slow down"})),
    )
    try:
        with pytest.raises(ProviderError) as excinfo:
            await client.embeddings(["x"])
    finally:
        await client.aclose()
    assert excinfo.value.status == 429


async def test_no_auth_mode_allows_loopback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "local"}}]})

    client = OpenAICompatClient(
        _cell(task="chat", base_url="http://127.0.0.1:11434/v1"),
        auth_mode="no-auth",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await client.chat([{"role": "user", "content": "hi"}]) == "local"
    finally:
        await client.aclose()


async def test_multi_mode_blocks_loopback() -> None:
    with pytest.raises(SSRFBlockedError):
        OpenAICompatClient(
            _cell(task="chat", base_url="http://127.0.0.1:11434/v1"),
            auth_mode="multi",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        )


async def test_multi_mode_blocks_private_dns() -> None:
    import socket

    def fake_getaddrinfo(host, port, *args, **kwargs):  # noqa: ANN001
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port or 443))]

    import unittest.mock as mock
    with mock.patch.object(socket, "getaddrinfo", fake_getaddrinfo):
        with pytest.raises(SSRFBlockedError):
            OpenAICompatClient(
                _cell(base_url="https://internal.example.test/v1"),
                auth_mode="multi",
            )
