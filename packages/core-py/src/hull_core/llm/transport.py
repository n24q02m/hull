"""Unified LLM transport — ONE call surface for chat / embed / rerank.

Library mode (litellm, no proxy). Two former call paths converge here:

* kcore ``infra/llm/dispatch`` (litellm passthrough + api_base SSRF vetting)
* KP ``core/llm._call_model`` (litellm + provider auth params)

``model`` strings carry their litellm provider prefix (``openrouter/...``,
``vertex_ai/...``, ``cohere/...``). OpenRouter is the default provider for
bare names in the DSPy builder; litellm model strings are always passed
through verbatim.

Credentials are never hardcoded: ``api_key``/``api_base`` kwargs win, else
``provider_params()`` consults the ``env_prefix`` namespace (``KLPRISM_`` ->
``KLPRISM_OPENROUTER_API_KEY``; the default ``HULL_`` ->
``HULL_OPENROUTER_API_KEY``; an ``env_prefix`` of ``""`` reads the canonical
``OPENROUTER_API_KEY``), else litellm's own provider env resolution applies.
"""

from __future__ import annotations

import os
from typing import Any

import litellm

from hull_core.http.ssrf import validate_url_and_get_ip, vet_api_base

# Drop unsupported params silently instead of erroring (provider-agnostic).
litellm.drop_params = True


def _env(name: str, env_prefix: str) -> str:
    """Read ``<env_prefix><name>``; empty prefix reads the canonical name."""
    return os.environ.get(f"{env_prefix}{name}", "")


def provider_params(provider: str, *, env_prefix: str = "HULL_") -> dict[str, Any]:
    """Provider-specific auth/routing kwargs derived from env.

    Returns only non-empty values, so callers may merge these under explicit
    kwargs (``{**provider_params(p), "api_key": key}`` keeps ``key``).
    """
    if provider == "openrouter":
        params: dict[str, Any] = {}
        if key := _env("OPENROUTER_API_KEY", env_prefix):
            params["api_key"] = key
        # CF AI Gateway / proxy flip = config-only, zero code change.
        if base := _env("LLM_API_BASE_OPENROUTER", env_prefix):
            params["api_base"] = base
        return params
    return {}


def _provider_of(model: str) -> str:
    """Provider segment of a litellm model id ("" for a bare name)."""
    return model.split("/", 1)[0] if "/" in model else ""


def _vet_base(url: str, *, strict_api_base: bool) -> str:
    """Vet ``api_base``: strict mode always rejects private/loopback targets
    (server deployments); the default follows the ``vet_api_base`` mode policy
    (loopback allowed single-user, ``LLM_API_BASE_ALLOW_PRIVATE`` escape,
    everything blocked under ``PUBLIC_URL``)."""
    if strict_api_base:
        validate_url_and_get_ip(url, allow_private=False, allow_loopback=False)
    else:
        vet_api_base(url)
    return url


def _resolved_kwargs(
    model: str,
    *,
    env_prefix: str,
    api_key: str | None,
    api_base: str | None,
    strict_api_base: bool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge env-derived provider params under explicit kwargs + vet api_base."""
    merged: dict[str, Any] = {**provider_params(_provider_of(model), env_prefix=env_prefix), **kwargs}
    if api_key is not None:
        merged["api_key"] = api_key
    if api_base is not None:
        merged["api_base"] = api_base
    if merged.get("api_base"):
        merged["api_base"] = _vet_base(str(merged["api_base"]), strict_api_base=strict_api_base)
    return merged


async def acompletion(
    *,
    model: str,
    messages: list[dict],
    api_base: str | None = None,
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    strict_api_base: bool = False,
    **kwargs: Any,
) -> Any:
    """Call any chat model; return the litellm ``ModelResponse``.

    Every id goes to ``litellm.acompletion`` with env-derived provider params
    (see ``provider_params``).
    """
    api_kwargs = _resolved_kwargs(
        model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        strict_api_base=strict_api_base,
        kwargs=kwargs,
    )
    return await litellm.acompletion(model=model, messages=messages, **api_kwargs)


def completion(
    *,
    model: str,
    messages: list[dict],
    api_base: str | None = None,
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    strict_api_base: bool = False,
    **kwargs: Any,
) -> Any:
    """Sync sibling of :func:`acompletion`. Do NOT call from an async loop."""
    api_kwargs = _resolved_kwargs(
        model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        strict_api_base=strict_api_base,
        kwargs=kwargs,
    )
    return litellm.completion(model=model, messages=messages, **api_kwargs)


def _reject_stream(kwargs: dict[str, Any], *, use: str) -> None:
    """Refuse ``stream=True`` in a text helper BEFORE any request is sent.

    A streamed call returns a litellm stream wrapper, not a ``ModelResponse``,
    so the text helpers could only fail after the (paid) request went out.
    """
    if kwargs.get("stream"):
        raise ValueError(f"text helpers are non-streaming; call {use}(..., stream=True) to stream")


def _response_text(resp: Any) -> str:
    """Assistant text of a ``ModelResponse`` (``None`` content -> ``""``).

    Anything else (e.g. a stream wrapper) raises ``TypeError`` naming its type
    instead of an opaque ``AttributeError`` on ``.choices``.
    """
    if not isinstance(resp, litellm.ModelResponse):
        raise TypeError(f"expected litellm.ModelResponse, got {type(resp).__qualname__}")
    return resp.choices[0].message.content or ""


async def acompletion_text(
    *,
    model: str,
    messages: list[dict],
    api_base: str | None = None,
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    strict_api_base: bool = False,
    **kwargs: Any,
) -> str:
    """Like :func:`acompletion` but returns the assistant text (empty-safe).

    Non-streaming only: ``stream=True`` raises ``ValueError`` before any
    request is sent (use :func:`acompletion` to stream), and a response that
    is not a litellm ``ModelResponse`` raises ``TypeError``.
    """
    _reject_stream(kwargs, use="acompletion")
    resp = await acompletion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        env_prefix=env_prefix,
        strict_api_base=strict_api_base,
        **kwargs,
    )
    return _response_text(resp)


def completion_text(
    *,
    model: str,
    messages: list[dict],
    api_base: str | None = None,
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    strict_api_base: bool = False,
    **kwargs: Any,
) -> str:
    """Sync sibling of :func:`acompletion_text`. Do NOT call from an async loop.

    Non-streaming only: ``stream=True`` raises ``ValueError`` before any
    request is sent (use :func:`completion` to stream), and a response that
    is not a litellm ``ModelResponse`` raises ``TypeError``.
    """
    _reject_stream(kwargs, use="completion")
    resp = completion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        env_prefix=env_prefix,
        strict_api_base=strict_api_base,
        **kwargs,
    )
    return _response_text(resp)


async def aembedding(
    *,
    model: str,
    input: list[str],
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    strict_api_base: bool = False,
    **kwargs: Any,
) -> Any:
    """Call any litellm embedding model; return the litellm response."""
    api_kwargs = _resolved_kwargs(
        model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        strict_api_base=strict_api_base,
        kwargs=kwargs,
    )
    return await litellm.aembedding(model=model, input=input, dimensions=dimensions, **api_kwargs)


async def arerank(
    *,
    model: str,
    query: str,
    documents: list[Any],
    top_n: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    strict_api_base: bool = False,
    **kwargs: Any,
) -> list[tuple[int, float]]:
    """Rerank documents via any litellm rerank model; return (index, score) sorted desc.

    Provider/model passthrough (e.g. "jina_ai/jina-reranker-m0",
    "cohere/rerank-v3.5"). `documents` may be plain strings (text) or
    provider-specific dicts for multimodal rerank (Jina m0 accepts
    {"text": ...} / {"image": ...}); passed through unchanged. Returns the
    litellm RerankResponse results normalized to (index, relevance_score).
    """
    if not documents:
        return []
    api_kwargs = _resolved_kwargs(
        model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        strict_api_base=strict_api_base,
        kwargs=kwargs,
    )
    resp = await litellm.arerank(model=model, query=query, documents=documents, top_n=top_n, **api_kwargs)
    results = resp.results if hasattr(resp, "results") else resp["results"]
    ranked: list[tuple[int, float]] = []
    for r in results:
        idx = r["index"] if isinstance(r, dict) else r.index
        score = r["relevance_score"] if isinstance(r, dict) else r.relevance_score
        ranked.append((int(idx), float(score)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
