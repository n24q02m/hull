"""Vertex AI Express generateContent adapter (litellm#21036 workaround).

litellm's ``vertex_ai/`` provider requires a service-account / ADC and rejects
the Vertex Express API key (open upstream bug BerriAI/litellm#21036). Gemini
traffic on an Express key therefore routes DIRECT to the Express
``generateContent`` REST endpoint through this thin httpx passthrough -- the
single documented exception to the "everything through litellm" transport
(CF AI Gateway cannot route Express either). Ported from KP
``app/infrastructure/ai/vertex_express.py`` (itself after mcp-core's adapter).

Endpoint (Express is global; no project/location in the path):
    POST https://aiplatform.googleapis.com/v1beta1/publishers/google/models/{model}:generateContent
    header: x-goog-api-key: <VERTEX_EXPRESS_KEY>

NOT the ``?key=`` query param, and ``v1beta1`` (not ``v1``). The response is
shaped into a litellm ``ModelResponse`` so the transport fallback path and the
DSPy ``VertexExpressLM`` consume the same object (``.choices[0].message.content``,
``.usage``, subscriptable ``["choices"]``, per-choice ``finish_reason``).

API key resolution order: explicit ``api_key=`` kwarg, then
``<env_prefix>VERTEX_EXPRESS_KEY`` (default ``HULL_VERTEX_EXPRESS_KEY``;
a product prefix such as ``KLPRISM_`` resolves ``KLPRISM_VERTEX_EXPRESS_KEY``).

Metrics: assign ``finish_reason_observer`` to consume (model, raw Vertex
finishReason); the observer is called best-effort and its exceptions are
swallowed so metrics can never break a call.

Scope: chat/vision ``generateContent`` ONLY (text + inline image input). Vertex
Express exposes no Imagen ``:predict`` REST surface, so image *generation*
stays on the native google-genai SDK in Express client mode.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

import httpx
from litellm import Choices, Message, ModelResponse, Usage

VERTEX_EXPRESS_PREFIX = "vertex_express"

DEFAULT_API_KEY_ENV = "HULL_VERTEX_EXPRESS_KEY"
_BASE = "https://aiplatform.googleapis.com"
_API_VERSION = "v1beta1"
_REQUEST_TIMEOUT_S = 120.0

# Optional metrics hook: (bare model name, raw Vertex finishReason). Set by
# host apps that track provider finish reasons; called inside try/except so a
# metrics failure can never fail the call.
finish_reason_observer: Callable[[str, str], None] | None = None

# OpenAI chat role -> Vertex generateContent role. "system" is hoisted to
# systemInstruction separately; "tool"/"function" are unsupported here and map
# to "user".
_ROLE_MAP = {"user": "user", "assistant": "model"}

# Vertex finishReason enum -> OpenAI finish_reason.
_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}


class VertexExpressError(RuntimeError):
    """Vertex Express returned a non-2xx status or no usable candidate."""


def strip_express_prefix(model: str) -> str:
    """Drop a leading ``vertex_express/`` segment; pass other ids through verbatim."""
    m = model.strip()
    if m.startswith(f"{VERTEX_EXPRESS_PREFIX}/"):
        return m.split("/", 1)[1]
    return m


def _content_parts(content: Any) -> list[dict[str, Any]]:
    """Translate an OpenAI message ``content`` to Vertex ``parts``.

    Accepts a plain string or the vision list-of-blocks form
    (``{"type": "text"|"image_url", ...}``). A ``data:<mime>;base64,<data>``
    image_url becomes an ``inlineData`` part; a remote ``http(s)`` image_url is
    rejected -- callers already inline the media bytes as a data URI before
    dispatch.
    """
    # An assistant/tool message may carry content=None (no text) -- treat as
    # an empty part list rather than iterating None (TypeError).
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append({"text": block.get("text", "")})
        elif btype == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if not url.startswith("data:"):
                raise VertexExpressError(
                    "vertex_express accepts only inline data: image URLs; download remote media before dispatch"
                )
            header, _, data = url.partition(",")
            mime = header.removeprefix("data:").split(";", 1)[0] or "image/png"
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        else:
            raise VertexExpressError(f"unsupported content block type {btype!r}")
    return parts


def messages_to_contents(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """OpenAI ``messages`` -> (Vertex ``contents``, optional systemInstruction)."""
    contents: list[dict[str, Any]] = []
    system: dict[str, Any] | None = None
    for msg in messages:
        role = msg.get("role", "user")
        parts = _content_parts(msg.get("content", ""))
        if role == "system":
            system = {"parts": parts} if system is None else {"parts": system["parts"] + parts}
            continue
        contents.append({"role": _ROLE_MAP.get(role, "user"), "parts": parts})
    return contents, system


def build_express_request(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: str | list[str] | None = None,
    response_format: dict[str, Any] | None = None,
    stream: bool | None = None,
    thinking_budget: int | None = None,
    **_ignored: Any,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build (url, headers, json_body) for an Express generateContent call.

    The api_key goes in the ``x-goog-api-key`` header, never the URL query
    string. Honored OpenAI params (mapped into ``generationConfig``):
    ``temperature``, ``max_tokens`` (-> ``maxOutputTokens``), ``top_p``
    (-> ``topP``), ``stop`` (-> ``stopSequences``), and
    ``response_format={"type": "json_object"}`` (-> ``responseMimeType``).
    ``stream=True`` is REJECTED -- this adapter buffers and cannot stream, so
    silently degrading a streamed request would diverge from the litellm path.
    Any other kwarg (``n``/``logprobs``/``seed``/...) is not supported and
    ignored.

    ``thinking_budget`` (-> ``thinkingConfig.thinkingBudget``) caps the
    REASONING tokens a thinking model may spend. It matters because Gemini
    bills thoughts against ``maxOutputTokens``: left unbounded, thinking
    starves the answer. Measured on gemini-3.5-flash, a 2388-char Chinese
    chapter at ``maxOutputTokens=4096`` -- unbounded: 3932 thought tokens,
    160 answer tokens, finishReason MAX_TOKENS (chapter lost); bounded to
    1024: 2601 answer tokens, finishReason STOP, and ~5s faster. Only
    emitted when not None, so a model without a thinking mode is unaffected.
    """
    if stream:
        raise VertexExpressError(
            "vertex_express does not support streaming (stream=True); use a litellm provider for streamed completions"
        )

    bare = strip_express_prefix(model)
    url = f"{_BASE}/{_API_VERSION}/publishers/google/models/{bare}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    contents, system = messages_to_contents(messages)
    body: dict[str, Any] = {"contents": contents}
    if system is not None:
        body["systemInstruction"] = system

    gen_config: dict[str, Any] = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = max_tokens
    if top_p is not None:
        gen_config["topP"] = top_p
    if stop is not None:
        gen_config["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    if response_format is not None and response_format.get("type") == "json_object":
        gen_config["responseMimeType"] = "application/json"
    if thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if gen_config:
        body["generationConfig"] = gen_config
    return url, headers, body


def _join_parts(parts: list[dict[str, Any]]) -> str | None:
    texts = [p.get("text", "") for p in parts if "text" in p]
    return "".join(texts) if texts else None


def _record_finish_reason(model: str, raw_finish: str) -> None:
    """Report the provider's own finishReason to ``finish_reason_observer``.

    Best-effort: never break a call. Kept hook-shaped (not a direct metrics
    import) so this module stays free of host-app dependencies.
    """
    observer = finish_reason_observer
    if observer is None:
        return
    try:
        observer(model, raw_finish)
    except Exception:  # noqa: BLE001 - a metrics failure must not fail the call
        pass


def translate_response(raw: dict[str, Any], *, model: str) -> ModelResponse:
    """Translate a Vertex generateContent JSON body into a litellm ``ModelResponse``.

    Raises ``VertexExpressError`` when no candidate is present (e.g. a prompt
    blocked by safety filters surfaces only ``promptFeedback.blockReason``). An
    empty-but-present candidate keeps its ``finish_reason`` so callers can log
    WHY the text is empty (MAX_TOKENS / SAFETY), matching the litellm path. The
    RAW Vertex finishReason is stashed on ``_hidden_params`` so an empty-response
    log surfaces the real reason -- litellm's ``Choices`` coerces any non-OpenAI
    value (OTHER/MALFORMED/SPII) to ``"stop"``, which would otherwise mask a
    truncated/blocked response as a clean stop.

    ``MAX_TOKENS`` RAISES rather than returning the partial text. A truncated
    completion is not a usable answer -- for a translation it is half a chapter
    -- and returning it as an ordinary success lets 2xx-only billing middleware
    charge users for a fragment. Failing here routes the call into the same
    no-charge path built for provider errors, and into the caller's fallback
    chain for non-billed callers.
    """
    candidates = raw.get("candidates") or []
    if not candidates:
        feedback = raw.get("promptFeedback", {})
        reason = feedback.get("blockReason", "no candidates returned")
        raise VertexExpressError(f"Vertex Express returned no candidate: {reason}")

    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    content = _join_parts(parts)
    raw_finish = cand.get("finishReason", "STOP")
    finish_reason = _FINISH_MAP.get(raw_finish, "stop")

    _record_finish_reason(strip_express_prefix(model), raw_finish)

    if raw_finish == "MAX_TOKENS":
        usage_meta = raw.get("usageMetadata") or {}
        raise VertexExpressError(
            "Vertex Express truncated the response (finishReason MAX_TOKENS): "
            f"{usage_meta.get('thoughtsTokenCount', 0)} thinking + "
            f"{usage_meta.get('candidatesTokenCount', 0)} answer tokens hit the "
            "maxOutputTokens ceiling. The input is too long to answer within the "
            "output budget -- split it into smaller parts."
        )

    usage_meta = raw.get("usageMetadata") or {}
    usage = Usage(
        prompt_tokens=int(usage_meta.get("promptTokenCount", 0)),
        completion_tokens=int(usage_meta.get("candidatesTokenCount", 0)),
        total_tokens=int(usage_meta.get("totalTokenCount", 0)),
    )

    response = ModelResponse(
        choices=[
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=Message(content=content, role="assistant"),
            )
        ],
        usage=usage,
        model=strip_express_prefix(model),
    )
    # Preserve the raw Vertex finishReason for diagnostics (empty-response log).
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        hidden["vertex_finish_reason"] = raw_finish
    return response


def _resolve_api_key(api_key: str | None, env_prefix: str) -> str:
    key = api_key or os.environ.get(f"{env_prefix}VERTEX_EXPRESS_KEY") or os.environ.get(DEFAULT_API_KEY_ENV)
    if not key:
        raise VertexExpressError(
            f"vertex_express requires an API key (pass api_key= or set {env_prefix}VERTEX_EXPRESS_KEY)"
        )
    return key


# Pooled clients: every LLM call is a keep-alive request to ONE host, so a
# per-call client (one TLS handshake each) is wasteful on hot paths. Reuse one
# sync client process-wide, and one async client per running event loop (an
# httpx.AsyncClient is bound to the loop it was created in, so it cannot be
# shared across loops -- pytest's per-test loops, asyncio.run, etc.).
_sync_client: httpx.Client | None = None
_async_clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(timeout=_REQUEST_TIMEOUT_S)
    return _sync_client


def _get_async_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _async_clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S)
        _async_clients[loop] = client
    # Drop any clients whose loop has since closed (bounded cleanup for tests).
    # NB: event loop uses is_closed() (method), unlike httpx's is_closed property.
    for dead in [lp for lp in _async_clients if lp.is_closed()]:
        _async_clients.pop(dead, None)
    return client


async def acompletion_express(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    **kwargs: Any,
) -> ModelResponse:
    """Async Vertex Express generateContent call (litellm#21036 workaround)."""
    key = _resolve_api_key(api_key, env_prefix)
    url, headers, body = build_express_request(model=model, messages=messages, api_key=key, **kwargs)
    resp = await _get_async_client().post(url, headers=headers, json=body)
    if resp.status_code // 100 != 2:
        raise VertexExpressError(f"Vertex Express HTTP {resp.status_code}: {resp.text[:500]}")
    return translate_response(resp.json(), model=model)


def completion_express(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None = None,
    env_prefix: str = "HULL_",
    **kwargs: Any,
) -> ModelResponse:
    """Sync Vertex Express generateContent call. Do NOT call from an async loop."""
    key = _resolve_api_key(api_key, env_prefix)
    url, headers, body = build_express_request(model=model, messages=messages, api_key=key, **kwargs)
    resp = _get_sync_client().post(url, headers=headers, json=body)
    if resp.status_code // 100 != 2:
        raise VertexExpressError(f"Vertex Express HTTP {resp.status_code}: {resp.text[:500]}")
    return translate_response(resp.json(), model=model)
