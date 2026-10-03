"""Unified LLM transport — ONE call surface for chat / embed / rerank.

Library mode (litellm, no proxy). Three former call paths converge here:

* kcore ``infra/llm/dispatch`` (litellm passthrough + api_base SSRF vetting)
* KP ``core/llm._call_model`` (litellm + provider auth params)
* KP ``infrastructure/ai/vertex_express`` (Express passthrough for the
  ``vertex_express/`` provider prefix, litellm#21036)

``model`` strings carry their provider prefix (``openrouter/...``,
``vertex_ai/...``, ``cohere/...``, ``vertex_express/...``). OpenRouter is the
default provider for bare names in the DSPy builder; litellm model strings are
always passed through verbatim.

Credentials are never hardcoded: ``api_key``/``api_base`` kwargs win, else the
``env_prefix`` namespace is consulted (``KLPRISM_`` ->
``KLPRISM_OPENROUTER_API_KEY``; the default ``HULL_`` ->
``HULL_OPENROUTER_API_KEY``), else litellm's own provider env resolution
applies. An ``env_prefix`` of ``""`` reads the canonical names directly
(``OPENROUTER_API_KEY``, ``XAI_API_KEY``, ...).

CF AI Gateway BYOK flips are config-only: when both
``<prefix>CF_AI_GATEWAY_URL`` and ``<prefix>CF_AIG_RUN_TOKEN`` are set, xai and
cohere traffic routes through the gateway with the ``cf-aig-authorization``
header; either missing keeps the direct provider endpoint (backward-safe).
"""

from __future__ import annotations

import os
from typing import Any

import litellm

from hull_core.http.ssrf import validate_url_and_get_ip, vet_api_base
from hull_core.llm.vertex_express import (
    VERTEX_EXPRESS_PREFIX,
    acompletion_express,
    completion_express,
)

# Drop unsupported params silently instead of erroring (provider-agnostic).
litellm.drop_params = True

_COHERE_DIRECT_BASE = "https://api.cohere.com"
# Per-op endpoint path (litellm cohere: embed=/v2/embed, rerank=/v1/rerank).
_COHERE_OP_PATH = {"embed": "v2/embed", "rerank": "v1/rerank"}


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
    if provider == "xai":
        params = {}
        if key := _env("XAI_API_KEY", env_prefix):
            params["api_key"] = key
        # CF AI Gateway flip: BYOK routing needs BOTH the gateway base URL and
        # the run-token auth header; either missing keeps grok DIRECT to
        # api.x.ai (backward-safe).
        gateway_url = _env("CF_AI_GATEWAY_URL", env_prefix)
        run_token = _env("CF_AIG_RUN_TOKEN", env_prefix)
        if gateway_url and run_token:
            params["api_base"] = f"{gateway_url.rstrip('/')}/grok/v1"
            params["extra_headers"] = {"cf-aig-authorization": f"Bearer {run_token}"}
        return params
    return {}


def cohere_routing(op: str, *, env_prefix: str = "HULL_") -> dict[str, Any]:
    """litellm kwargs (api_base, api_key, [headers, extra_headers]) for one cohere op.

    With both ``<prefix>CF_AI_GATEWAY_URL`` + ``<prefix>CF_AIG_RUN_TOKEN``
    present, routes via CF AI Gateway BYOK (``cf-aig-authorization`` header);
    otherwise calls api.cohere.com directly. The Cohere key always comes from
    ``<prefix>COHERE_API_KEY`` (litellm attaches it as ``Authorization: Bearer``).

    ``op`` is ``"embed"`` or ``"rerank"`` (per-op endpoint path).
    """
    path = _COHERE_OP_PATH[op]
    params: dict[str, Any] = {}
    if key := _env("COHERE_API_KEY", env_prefix):
        params["api_key"] = key

    gateway_url = _env("CF_AI_GATEWAY_URL", env_prefix)
    run_token = _env("CF_AIG_RUN_TOKEN", env_prefix)
    if gateway_url and run_token:
        params["api_base"] = f"{gateway_url.rstrip('/')}/cohere/{path}"
        header = {"cf-aig-authorization": f"Bearer {run_token}"}
        params["headers"] = header
        params["extra_headers"] = header
    else:
        params["api_base"] = f"{_COHERE_DIRECT_BASE}/{path}"
    return params


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

    ``vertex_express/<model>`` routes through the Express passthrough (litellm
    cannot route it); every other id goes to ``litellm.acompletion`` with
    env-derived provider params (see ``provider_params``).
    """
    if _provider_of(model) == VERTEX_EXPRESS_PREFIX:
        return await acompletion_express(
            model=model, messages=messages, api_key=api_key, env_prefix=env_prefix, **kwargs
        )
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
    if _provider_of(model) == VERTEX_EXPRESS_PREFIX:
        return completion_express(model=model, messages=messages, api_key=api_key, env_prefix=env_prefix, **kwargs)
    api_kwargs = _resolved_kwargs(
        model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        strict_api_base=strict_api_base,
        kwargs=kwargs,
    )
    return litellm.completion(model=model, messages=messages, **api_kwargs)


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
    """Like :func:`acompletion` but returns the assistant text (empty-safe)."""
    resp = await acompletion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        env_prefix=env_prefix,
        strict_api_base=strict_api_base,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


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
    """Sync sibling of :func:`acompletion_text`. Do NOT call from an async loop."""
    resp = completion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        env_prefix=env_prefix,
        strict_api_base=strict_api_base,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


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
