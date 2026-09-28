"""Provider calls: plain HTTP OpenAI-spec via httpx (no provider-routing layer).

One client per task cell. ``base_url`` is vetted through the SSRF guard with
mode-derived policy: shared/multi deployments block loopback + private ranges
unconditionally; single-user (no-auth) keeps loopback for self-hosted
Ollama/vLLM. Endpoints: ``/embeddings``, ``/chat/completions``, ``/rerank``
(Cohere-compatible shape served by OpenRouter and rerank providers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from hull_core.config.models import ModelCell
from hull_core.http.ssrf import (
    AsyncSSRFSafeTransport,
    SSRFBlockedError,
    validate_url_and_get_ip,
)


class ProviderError(RuntimeError):
    """Non-2xx provider response or transport failure."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"provider returned {status}: {detail[:500]}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class AuthPolicy:
    """SSRF policy flags derived from the configured auth mode."""

    allow_loopback: bool
    allow_private: bool

    @staticmethod
    def for_mode(auth_mode: str) -> "AuthPolicy":
        # no-auth = single-user local instance: loopback providers (Ollama on
        # this machine) are legitimate. token/multi = shared deployment: the
        # escape hatches are deliberately ignored, mirroring vet_api_base.
        if auth_mode == "no-auth":
            return AuthPolicy(allow_loopback=True, allow_private=False)
        return AuthPolicy(allow_loopback=False, allow_private=False)


class OpenAICompatClient:
    """Async OpenAI-spec client bound to one task's model cell."""

    def __init__(
        self,
        cell: ModelCell,
        *,
        auth_mode: str = "no-auth",
        timeout: float = 60.0,
        transport: httpx.AsyncTransport | None = None,
    ) -> None:
        if cell.base_url.startswith(("http://localhost", "http://127.0.0.1", "http://[::1]")):
            # Loopback literal: let the policy decide instead of the resolver.
            policy = AuthPolicy.for_mode(auth_mode)
            if not policy.allow_loopback:
                raise SSRFBlockedError(f"loopback provider base_url forbidden in {auth_mode!r} mode")
        else:
            policy = AuthPolicy.for_mode(auth_mode)
            # Validate (and DNS-pin via the transport below); the base_url
            # itself stays the real URL so Host header + TLS SNI stay correct.
            validate_url_and_get_ip(
                cell.base_url,
                allow_private=policy.allow_private,
                allow_loopback=policy.allow_loopback,
            )
        self.cell = cell
        headers = {"Authorization": f"Bearer {cell.api_key}"} if cell.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=cell.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport
            if transport is not None
            else AsyncSSRFSafeTransport(allow_private=policy.allow_private, allow_loopback=policy.allow_loopback),
        )

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = await self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(502, f"transport error calling {path}: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(resp.status_code, resp.text)
        return resp.json()

    async def embeddings(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        **extra: Any,
    ) -> list[list[float]]:
        """POST {base_url}/embeddings → ordered embedding vectors.

        ``dimensions`` maps to the OpenAI-spec ``dimensions`` field (storage
        width selection, e.g. Cohere embed-v4.0 width) and is omitted when
        ``None``. ``extra`` carries provider-specific body fields verbatim
        (e.g. Cohere ``input_type``); ``model``/``input`` are reserved — the
        cell always wins, an override attempt raises.
        """
        if "model" in extra or "input" in extra:
            raise ValueError("embeddings() extra fields cannot override model/input")
        payload: dict = {"model": self.cell.model, "input": texts, **extra}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        data = await self._post("/embeddings", payload)
        items = sorted(data["data"], key=lambda item: item["index"])
        return [list(item["embedding"]) for item in items]

    async def chat(self, messages: list[dict], **options) -> str:
        """POST {base_url}/chat/completions → assistant message content.

        ``model``/``messages`` are reserved — the cell always wins, an
        override attempt raises.
        """
        if "model" in options or "messages" in options:
            raise ValueError("chat() options cannot override model/messages")
        payload = {"model": self.cell.model, "messages": messages, **options}
        data = await self._post("/chat/completions", payload)
        message = data["choices"][0]["message"]
        content = message.get("content")
        if not content:
            # Reasoning models may emit everything as reasoning (null content)
            # when the token budget is exhausted before any answer text.
            content = message.get("reasoning_content") or ""
        return str(content)

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        """POST {base_url}/rerank → results sorted by relevance score."""
        payload: dict = {"model": self.cell.model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        data = await self._post("/rerank", payload)
        return list(data["results"])

    async def aclose(self) -> None:
        await self._client.aclose()
