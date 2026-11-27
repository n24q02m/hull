"""OpenRouter-first LLM transport over httpx + SSRF vetting.

Provides async and sync completion (including SSE streaming with real USD usage),
embeddings, reranking, and ``/v1/systemone`` calls with:

* SSRF vetting via :mod:`hull_core.http.ssrf` on every custom/gateway base URL
* Strict ZDR default (``provider: {"data_collection": "deny"}``), fail-closed
  when no ZDR endpoint exists
* Cloudflare AI Gateway ``cf-aig-*`` headers with safe defaults
  (``cf-aig-collect-log: false``, ``cf-aig-skip-cache: true`` for private data;
  edge cache only via explicit ``edge_cache=True``)
* Pass-through ``cache_control`` untouched in messages/tools/body
* Gateway circuit breaker falling back directly to
  ``https://openrouter.ai/api/v1`` on gateway failures
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from hull_core.http.ssrf import (
    AsyncSSRFSafeTransport,
    SyncSSRFSafeTransport,
    vet_api_base,
)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
SYSTEMONE_PATH = "/v1/systemone"
_GATEWAY_TRIP_STATUSES = frozenset({429, 500, 502, 503, 504})


class ZDRUnavailableError(RuntimeError):
    """Raised fail-closed when strict ZDR is required but no ZDR endpoint exists."""


class TransportHTTPError(RuntimeError):
    """Raised on non-2xx HTTP responses from the upstream LLM endpoint."""

    def __init__(self, status: int, detail: str, *, url: str = "") -> None:
        super().__init__(f"LLM endpoint returned {status}: {detail[:500]}")
        self.status = status
        self.detail = detail
        self.url = url


@dataclass(frozen=True)
class UsageMetrics:
    """Normalized token, prompt-cache, real USD cost, and ZDR state."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    usd_cost: float | None = None
    zdr_enforced: bool = True


@dataclass(frozen=True)
class Message:
    role: str = "assistant"
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Choice:
    index: int
    message: Message
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResponse:
    id: str
    model: str
    choices: list[Choice]
    usage: dict[str, Any]
    usage_metrics: UsageMetrics
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingData:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class EmbeddingResponse:
    model: str
    data: list[EmbeddingData]
    usage: dict[str, Any]
    usage_metrics: UsageMetrics
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemOneResponse:
    id: str
    model: str
    answers: dict[str, Any]
    usage: dict[str, Any]
    usage_metrics: UsageMetrics
    raw: dict[str, Any] = field(default_factory=dict)


class GatewayCircuitBreaker:
    """Thread-safe circuit breaker from a gateway URL to direct OpenRouter."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        clock: Any = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if self._clock() - self._opened_at >= self.recovery_timeout:
                # Half-open: allow one probe through the gateway.
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = self._clock()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None


_DEFAULT_BREAKER = GatewayCircuitBreaker()


def reset_gateway_circuit_breaker() -> None:
    """Reset the shared module-level CF AI Gateway circuit breaker."""
    _DEFAULT_BREAKER.reset()


def extract_usage_metrics(payload: Mapping[str, Any] | None, *, zdr_enforced: bool = True) -> UsageMetrics:
    """Extract normalized token counts, cached tokens, and OpenRouter USD cost."""
    if not isinstance(payload, Mapping):
        return UsageMetrics(zdr_enforced=zdr_enforced)
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        usage = payload
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    cached_tokens = 0
    if isinstance(details, Mapping) and details.get("cached_tokens") is not None:
        cached_tokens = int(details["cached_tokens"])
    elif usage.get("cached_tokens") is not None:
        cached_tokens = int(usage["cached_tokens"])

    raw_cost = usage.get("cost")
    if raw_cost is None:
        raw_cost = usage.get("total_cost")
    if raw_cost is None:
        raw_cost = payload.get("cost")
    usd_cost = float(raw_cost) if raw_cost is not None else None
    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        usd_cost=usd_cost,
        zdr_enforced=zdr_enforced,
    )


def cf_aig_headers(
    *,
    collect_log: bool = False,
    skip_cache: bool | None = None,
    edge_cache: bool = False,
    cache_ttl: int | None = None,
    custom_cache_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build Cloudflare AI Gateway ``cf-aig-*`` headers with safe defaults.

    Private-data defaults:
    * ``cf-aig-collect-log: "false"`` unless ``collect_log=True``
    * ``cf-aig-skip-cache: "true"`` unless ``edge_cache=True`` (or ``skip_cache=False``)
    """
    headers: dict[str, str] = {}
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    headers.setdefault("cf-aig-collect-log", "true" if collect_log else "false")

    effective_skip_cache = skip_cache if skip_cache is not None else (not edge_cache)
    if effective_skip_cache:
        headers["cf-aig-skip-cache"] = "true"
    else:
        headers["cf-aig-skip-cache"] = "false"
        if cache_ttl is not None:
            headers["cf-aig-cache-ttl"] = str(int(cache_ttl))
        if custom_cache_key:
            headers["cf-aig-custom-cache-key"] = str(custom_cache_key)
    return headers


def _strip_cf_aig_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if not k.lower().startswith("cf-aig-")}


def _is_gateway_url(base_url: str) -> bool:
    return base_url.rstrip("/") != OPENROUTER_API_BASE


def provider_params(provider: str = "openrouter", *, env_prefix: str = "HULL_") -> dict[str, Any]:
    """Resolve provider credentials + optional base URL from ``env_prefix`` / canonical env."""
    if not provider:
        return {}
    key_name = provider.strip().lower()
    if key_name == "openrouter":
        prefixes = (env_prefix, "") if env_prefix else ("",)
        params: dict[str, Any] = {}
        for pfx in prefixes:
            api_key = os.environ.get(f"{pfx}OPENROUTER_API_KEY")
            if api_key:
                params["api_key"] = api_key
                break
        for pfx in prefixes:
            api_base = os.environ.get(f"{pfx}OPENROUTER_API_BASE") or os.environ.get(f"{pfx}AI_GATEWAY_URL")
            if api_base:
                params["api_base"] = api_base.rstrip("/")
                break
        return params
    if key_name == "typesafe":
        prefixes = (env_prefix, "") if env_prefix else ("",)
        params = {}
        for pfx in prefixes:
            api_key = os.environ.get(f"{pfx}TYPESAFE_API_KEY")
            if api_key:
                params["api_key"] = api_key
                break
        for pfx in prefixes:
            api_base = os.environ.get(f"{pfx}TYPESAFE_API_BASE")
            if api_base:
                params["api_base"] = api_base.rstrip("/")
                break
        return params
    return {}


def _provider_of(model: str) -> str:
    if ":" in model and (":" not in model.split("/", 1)[0] or model.startswith(("openrouter:", "typesafe:"))):
        head = model.split(":", 1)[0]
        if head in {"openrouter", "typesafe"}:
            return head
    head, sep, _ = model.partition("/")
    if sep and head in {"openrouter", "typesafe"}:
        return head
    return "openrouter"


def _normalize_openrouter_model(model: str) -> str:
    if model.startswith("openrouter/"):
        return model.removeprefix("openrouter/")
    if model.startswith("openrouter:"):
        return model.removeprefix("openrouter:")
    return model


def _validate_and_apply_zdr(
    payload: dict[str, Any],
    *,
    zdr: bool,
    zdr_available: bool | None,
    zdr_endpoints: Sequence[Any] | None,
) -> bool:
    if not zdr:
        return False
    if zdr_available is False:
        raise ZDRUnavailableError(f"no ZDR endpoint available for model {payload.get('model')!r}")
    if zdr_endpoints is not None:
        has_zdr = False
        for ep in zdr_endpoints:
            if isinstance(ep, str) and ep.strip():
                has_zdr = True
                break
            if isinstance(ep, Mapping) and (ep.get("zdr") is True or "zdr" in str(ep.get("tag", "")).lower()):
                has_zdr = True
                break
        if not has_zdr:
            raise ZDRUnavailableError(f"no ZDR endpoint listed for model {payload.get('model')!r}")

    prov = dict(payload.get("provider") or {})
    dc = prov.get("data_collection")
    if dc is not None and str(dc).lower() != "deny":
        raise ZDRUnavailableError("strict ZDR requires provider.data_collection='deny'")
    prov["data_collection"] = "deny"
    payload["provider"] = prov
    return True


def _check_zdr_error_response(status: int, text: str, model: str | None) -> None:
    lowered = text.lower()
    if status in (400, 404) and (
        "no endpoints found" in lowered
        or "data policy" in lowered
        or "zero data retention" in lowered
        or "zdr" in lowered
    ):
        raise ZDRUnavailableError(f"OpenRouter rejected ZDR request for {model!r}: {text[:300]}")


def _endpoint_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    clean = path if path.startswith("/") else f"/{path}"
    if clean == SYSTEMONE_PATH and base.endswith("/v1"):
        return f"{base}/systemone"
    if base.endswith("/v1") and clean.startswith("/v1/"):
        return f"{base}{clean.removeprefix('/v1')}"
    return f"{base}{clean}"


def _prepare_request(
    *,
    model: str,
    env_prefix: str,
    api_key: str | None,
    api_base: str | None,
    zdr: bool,
    zdr_available: bool | None,
    zdr_endpoints: Sequence[Any] | None,
    collect_log: bool,
    skip_cache: bool | None,
    edge_cache: bool,
    cache_ttl: int | None,
    custom_cache_key: str | None,
    extra_headers: Mapping[str, str] | None,
    extra_body: dict[str, Any],
) -> tuple[str, dict[str, str], dict[str, Any], bool]:
    provider = _provider_of(model)
    resolved = provider_params(provider, env_prefix=env_prefix)
    effective_key = api_key if api_key is not None else resolved.get("api_key", "")
    effective_base = (api_base or resolved.get("api_base") or OPENROUTER_API_BASE).rstrip("/")
    if effective_base != OPENROUTER_API_BASE:
        vet_api_base(effective_base)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if effective_key:
        headers["Authorization"] = f"Bearer {effective_key}"
    if _is_gateway_url(effective_base):
        headers.update(
            cf_aig_headers(
                collect_log=collect_log,
                skip_cache=skip_cache,
                edge_cache=edge_cache,
                cache_ttl=cache_ttl,
                custom_cache_key=custom_cache_key,
                extra_headers=extra_headers,
            )
        )
    elif extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    payload: dict[str, Any] = dict(extra_body)
    payload["model"] = _normalize_openrouter_model(model)
    zdr_enforced = _validate_and_apply_zdr(
        payload,
        zdr=zdr,
        zdr_available=zdr_available,
        zdr_endpoints=zdr_endpoints,
    )
    return effective_base, headers, payload, zdr_enforced


def _parse_completion_response(data: Mapping[str, Any], *, zdr_enforced: bool) -> CompletionResponse:
    raw_dict = dict(data)
    choices: list[Choice] = []
    for idx, ch in enumerate(raw_dict.get("choices") or []):
        msg_raw = ch.get("message") or ch.get("delta") or {}
        msg = Message(
            role=str(msg_raw.get("role") or "assistant"),
            content=msg_raw.get("content"),
            reasoning=msg_raw.get("reasoning"),
            reasoning_content=msg_raw.get("reasoning_content"),
            raw=dict(msg_raw),
        )
        choices.append(
            Choice(
                index=int(ch.get("index", idx)),
                message=msg,
                finish_reason=ch.get("finish_reason"),
                raw=dict(ch),
            )
        )
    usage_raw = dict(raw_dict.get("usage") or {})
    return CompletionResponse(
        id=str(raw_dict.get("id") or ""),
        model=str(raw_dict.get("model") or ""),
        choices=choices,
        usage=usage_raw,
        usage_metrics=extract_usage_metrics(raw_dict, zdr_enforced=zdr_enforced),
        raw=raw_dict,
    )


class AsyncCompletionStream:
    """Async SSE iterator yielding parsed OpenRouter chunks and capturing USD usage."""

    def __init__(
        self,
        lines_factory: Any,
        *,
        zdr_enforced: bool,
        model: str,
    ) -> None:
        self._lines_factory = lines_factory
        self._zdr_enforced = zdr_enforced
        self._model = model
        self._chunks: list[str] = []
        self._usage: UsageMetrics = UsageMetrics(zdr_enforced=zdr_enforced)

    @property
    def usage_metrics(self) -> UsageMetrics:
        return self._usage

    @property
    def collected_text(self) -> str:
        return "".join(self._chunks)

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        async for line in self._lines_factory():
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            chunk = json.loads(body)
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str):
                    self._chunks.append(piece)
            if chunk.get("usage"):
                self._usage = extract_usage_metrics(chunk, zdr_enforced=self._zdr_enforced)
            yield chunk


class SyncCompletionStream:
    """Sync SSE iterator yielding parsed OpenRouter chunks and capturing USD usage."""

    def __init__(
        self,
        lines: Iterator[str],
        close_fn: Any,
        *,
        zdr_enforced: bool,
        model: str,
    ) -> None:
        self._lines = lines
        self._close_fn = close_fn
        self._zdr_enforced = zdr_enforced
        self._model = model
        self._chunks: list[str] = []
        self._usage: UsageMetrics = UsageMetrics(zdr_enforced=zdr_enforced)

    @property
    def usage_metrics(self) -> UsageMetrics:
        return self._usage

    @property
    def collected_text(self) -> str:
        return "".join(self._chunks)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            for line in self._lines:
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                chunk = json.loads(body)
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        self._chunks.append(piece)
                if chunk.get("usage"):
                    self._usage = extract_usage_metrics(chunk, zdr_enforced=self._zdr_enforced)
                yield chunk
        finally:
            self._close_fn()


async def _apost_json(
    *,
    base_url: str,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    circuit_breaker: GatewayCircuitBreaker,
    gateway_fallback: bool,
    transport: httpx.AsyncBaseTransport | None,
    allow_loopback: bool = False,
) -> dict[str, Any]:
    use_gateway = _is_gateway_url(base_url) and gateway_fallback
    targets: list[tuple[str, dict[str, str], bool]] = []
    if use_gateway and circuit_breaker.is_open():
        targets.append((OPENROUTER_API_BASE, _strip_cf_aig_headers(headers), False))
    else:
        targets.append((base_url, headers, use_gateway))
        if use_gateway:
            targets.append((OPENROUTER_API_BASE, _strip_cf_aig_headers(headers), False))

    last_exc: Exception | None = None
    for target_base, target_headers, is_gw in targets:
        url = _endpoint_url(target_base, path)
        tx = transport if transport is not None else AsyncSSRFSafeTransport(allow_loopback=allow_loopback)
        try:
            async with httpx.AsyncClient(transport=tx, timeout=timeout) as client:
                resp = await client.post(url, headers=target_headers, json=payload)
        except httpx.HTTPError as exc:
            last_exc = exc
            if is_gw:
                circuit_breaker.record_failure()
                continue
            raise TransportHTTPError(502, f"transport error: {exc}", url=url) from exc

        if resp.status_code in _GATEWAY_TRIP_STATUSES and is_gw:
            circuit_breaker.record_failure()
            last_exc = TransportHTTPError(resp.status_code, resp.text, url=url)
            continue

        _check_zdr_error_response(resp.status_code, resp.text, payload.get("model"))
        if resp.status_code >= 400:
            raise TransportHTTPError(resp.status_code, resp.text, url=url)
        if is_gw:
            circuit_breaker.record_success()
        return dict(resp.json())

    if isinstance(last_exc, TransportHTTPError):
        raise last_exc
    raise TransportHTTPError(502, f"gateway and fallback failed: {last_exc}")


def _post_json(
    *,
    base_url: str,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    circuit_breaker: GatewayCircuitBreaker,
    gateway_fallback: bool,
    transport: httpx.BaseTransport | None,
    allow_loopback: bool = False,
) -> dict[str, Any]:
    use_gateway = _is_gateway_url(base_url) and gateway_fallback
    targets: list[tuple[str, dict[str, str], bool]] = []
    if use_gateway and circuit_breaker.is_open():
        targets.append((OPENROUTER_API_BASE, _strip_cf_aig_headers(headers), False))
    else:
        targets.append((base_url, headers, use_gateway))
        if use_gateway:
            targets.append((OPENROUTER_API_BASE, _strip_cf_aig_headers(headers), False))

    last_exc: Exception | None = None
    for target_base, target_headers, is_gw in targets:
        url = _endpoint_url(target_base, path)
        tx = transport if transport is not None else SyncSSRFSafeTransport(allow_loopback=allow_loopback)
        try:
            with httpx.Client(transport=tx, timeout=timeout) as client:
                resp = client.post(url, headers=target_headers, json=payload)
        except httpx.HTTPError as exc:
            last_exc = exc
            if is_gw:
                circuit_breaker.record_failure()
                continue
            raise TransportHTTPError(502, f"transport error: {exc}", url=url) from exc

        if resp.status_code in _GATEWAY_TRIP_STATUSES and is_gw:
            circuit_breaker.record_failure()
            last_exc = TransportHTTPError(resp.status_code, resp.text, url=url)
            continue

        _check_zdr_error_response(resp.status_code, resp.text, payload.get("model"))
        if resp.status_code >= 400:
            raise TransportHTTPError(resp.status_code, resp.text, url=url)
        if is_gw:
            circuit_breaker.record_success()
        return dict(resp.json())

    if isinstance(last_exc, TransportHTTPError):
        raise last_exc
    raise TransportHTTPError(502, f"gateway and fallback failed: {last_exc}")


async def acompletion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    env_prefix: str = "HULL_",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 60.0,
    zdr: bool = True,
    zdr_available: bool | None = None,
    zdr_endpoints: Sequence[Any] | None = None,
    collect_log: bool = False,
    skip_cache: bool | None = None,
    edge_cache: bool = False,
    cache_ttl: int | None = None,
    custom_cache_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    circuit_breaker: GatewayCircuitBreaker | None = None,
    gateway_fallback: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> CompletionResponse | AsyncCompletionStream:
    """Async chat completion over OpenRouter (or CF AI Gateway) with SSRF + ZDR."""
    breaker = circuit_breaker or _DEFAULT_BREAKER
    body = {"messages": messages, **kwargs}
    base_url, headers, payload, zdr_enforced = _prepare_request(
        model=model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        zdr=zdr,
        zdr_available=zdr_available,
        zdr_endpoints=zdr_endpoints,
        collect_log=collect_log,
        skip_cache=skip_cache,
        edge_cache=edge_cache,
        cache_ttl=cache_ttl,
        custom_cache_key=custom_cache_key,
        extra_headers=extra_headers,
        extra_body=body,
    )
    if payload.get("stream"):
        payload.setdefault("stream_options", {"include_usage": True})

        async def _iter_lines() -> AsyncIterator[str]:
            use_gw = _is_gateway_url(base_url) and gateway_fallback
            targets: list[tuple[str, dict[str, str], bool]] = []
            if use_gw and breaker.is_open():
                targets.append((OPENROUTER_API_BASE, _strip_cf_aig_headers(headers), False))
            else:
                targets.append((base_url, headers, use_gw))
                if use_gw:
                    targets.append((OPENROUTER_API_BASE, _strip_cf_aig_headers(headers), False))

            last_err: Exception | None = None
            for target_base, target_headers, is_gw in targets:
                url = _endpoint_url(target_base, "/chat/completions")
                tx = transport if transport is not None else AsyncSSRFSafeTransport()
                try:
                    async with httpx.AsyncClient(transport=tx, timeout=timeout) as client:
                        async with client.stream("POST", url, headers=target_headers, json=payload) as resp:
                            if resp.status_code in _GATEWAY_TRIP_STATUSES and is_gw:
                                await resp.aread()
                                breaker.record_failure()
                                last_err = TransportHTTPError(resp.status_code, resp.text, url=url)
                                continue
                            if resp.status_code >= 400:
                                raw_err = (await resp.aread()).decode("utf-8", errors="replace")
                                _check_zdr_error_response(resp.status_code, raw_err, payload.get("model"))
                                raise TransportHTTPError(resp.status_code, raw_err, url=url)
                            if is_gw:
                                breaker.record_success()
                            async for line in resp.aiter_lines():
                                yield line
                            return
                except httpx.HTTPError as exc:
                    last_err = exc
                    if is_gw:
                        breaker.record_failure()
                        continue
                    raise TransportHTTPError(502, f"stream error: {exc}", url=url) from exc
            if isinstance(last_err, TransportHTTPError):
                raise last_err
            raise TransportHTTPError(502, f"gateway stream failed: {last_err}")

        return AsyncCompletionStream(_iter_lines, zdr_enforced=zdr_enforced, model=payload["model"])

    data = await _apost_json(
        base_url=base_url,
        path="/chat/completions",
        headers=headers,
        payload=payload,
        timeout=timeout,
        circuit_breaker=breaker,
        gateway_fallback=gateway_fallback,
        transport=transport,
    )
    return _parse_completion_response(data, zdr_enforced=zdr_enforced)


def completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    env_prefix: str = "HULL_",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 60.0,
    zdr: bool = True,
    zdr_available: bool | None = None,
    zdr_endpoints: Sequence[Any] | None = None,
    collect_log: bool = False,
    skip_cache: bool | None = None,
    edge_cache: bool = False,
    cache_ttl: int | None = None,
    custom_cache_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    circuit_breaker: GatewayCircuitBreaker | None = None,
    gateway_fallback: bool = True,
    transport: httpx.BaseTransport | None = None,
    **kwargs: Any,
) -> CompletionResponse | SyncCompletionStream:
    """Sync sibling of :func:`acompletion`. Do NOT call from an async loop."""
    breaker = circuit_breaker or _DEFAULT_BREAKER
    body = {"messages": messages, **kwargs}
    base_url, headers, payload, zdr_enforced = _prepare_request(
        model=model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        zdr=zdr,
        zdr_available=zdr_available,
        zdr_endpoints=zdr_endpoints,
        collect_log=collect_log,
        skip_cache=skip_cache,
        edge_cache=edge_cache,
        cache_ttl=cache_ttl,
        custom_cache_key=custom_cache_key,
        extra_headers=extra_headers,
        extra_body=body,
    )
    if payload.get("stream"):
        payload.setdefault("stream_options", {"include_usage": True})
        use_gw = _is_gateway_url(base_url) and gateway_fallback
        target_base = OPENROUTER_API_BASE if (use_gw and breaker.is_open()) else base_url
        target_headers = _strip_cf_aig_headers(headers) if target_base != base_url else headers
        url = _endpoint_url(target_base, "/chat/completions")
        tx = transport if transport is not None else SyncSSRFSafeTransport()
        client = httpx.Client(transport=tx, timeout=timeout)
        cm = client.stream("POST", url, headers=target_headers, json=payload)
        resp = cm.__enter__()
        if resp.status_code >= 400:
            raw_err = resp.read().decode("utf-8", errors="replace")
            cm.__exit__(None, None, None)
            client.close()
            _check_zdr_error_response(resp.status_code, raw_err, payload.get("model"))
            raise TransportHTTPError(resp.status_code, raw_err, url=url)

        def _close() -> None:
            cm.__exit__(None, None, None)
            client.close()

        return SyncCompletionStream(resp.iter_lines(), _close, zdr_enforced=zdr_enforced, model=payload["model"])

    data = _post_json(
        base_url=base_url,
        path="/chat/completions",
        headers=headers,
        payload=payload,
        timeout=timeout,
        circuit_breaker=breaker,
        gateway_fallback=gateway_fallback,
        transport=transport,
    )
    return _parse_completion_response(data, zdr_enforced=zdr_enforced)


def _reject_stream(kwargs: dict[str, Any], *, use: str) -> None:
    if kwargs.get("stream"):
        raise ValueError(f"streaming is not supported by *_text; call {use}() instead")


def _extract_content(resp: Any) -> str:
    if not isinstance(resp, CompletionResponse):
        raise TypeError(f"expected CompletionResponse, got {type(resp).__qualname__}")
    if not resp.choices:
        return ""
    return resp.choices[0].message.content or ""


async def acompletion_text(
    *,
    model: str,
    messages: list[dict[str, Any]],
    env_prefix: str = "HULL_",
    **kwargs: Any,
) -> str:
    """Like :func:`acompletion` but returns the assistant text (empty-safe)."""
    _reject_stream(kwargs, use="acompletion")
    resp = await acompletion(model=model, messages=messages, env_prefix=env_prefix, **kwargs)
    return _extract_content(resp)


def completion_text(
    *,
    model: str,
    messages: list[dict[str, Any]],
    env_prefix: str = "HULL_",
    **kwargs: Any,
) -> str:
    """Sync sibling of :func:`acompletion_text`. Do NOT call from an async loop."""
    _reject_stream(kwargs, use="completion")
    resp = completion(model=model, messages=messages, env_prefix=env_prefix, **kwargs)
    return _extract_content(resp)


async def aembedding(
    *,
    model: str,
    input: str | list[str],
    dimensions: int | None = None,
    env_prefix: str = "HULL_",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 60.0,
    zdr: bool = True,
    zdr_available: bool | None = None,
    zdr_endpoints: Sequence[Any] | None = None,
    collect_log: bool = False,
    skip_cache: bool | None = None,
    edge_cache: bool = False,
    cache_ttl: int | None = None,
    custom_cache_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    circuit_breaker: GatewayCircuitBreaker | None = None,
    gateway_fallback: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> EmbeddingResponse:
    """Call OpenRouter embeddings API with SSRF vetting and ZDR enforcement."""
    body: dict[str, Any] = {"input": input, **kwargs}
    if dimensions is not None:
        body["dimensions"] = dimensions
    base_url, headers, payload, zdr_enforced = _prepare_request(
        model=model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        zdr=zdr,
        zdr_available=zdr_available,
        zdr_endpoints=zdr_endpoints,
        collect_log=collect_log,
        skip_cache=skip_cache,
        edge_cache=edge_cache,
        cache_ttl=cache_ttl,
        custom_cache_key=custom_cache_key,
        extra_headers=extra_headers,
        extra_body=body,
    )
    data = await _apost_json(
        base_url=base_url,
        path="/embeddings",
        headers=headers,
        payload=payload,
        timeout=timeout,
        circuit_breaker=circuit_breaker or _DEFAULT_BREAKER,
        gateway_fallback=gateway_fallback,
        transport=transport,
    )
    items = sorted(data.get("data") or [], key=lambda it: int(it.get("index", 0)))
    return EmbeddingResponse(
        model=str(data.get("model") or payload["model"]),
        data=[
            EmbeddingData(index=int(it.get("index", idx)), embedding=list(it["embedding"]))
            for idx, it in enumerate(items)
        ],
        usage=dict(data.get("usage") or {}),
        usage_metrics=extract_usage_metrics(data, zdr_enforced=zdr_enforced),
        raw=data,
    )


async def arerank(
    *,
    model: str,
    query: str,
    documents: list[str],
    top_n: int | None = None,
    env_prefix: str = "HULL_",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 60.0,
    zdr: bool = True,
    zdr_available: bool | None = None,
    zdr_endpoints: Sequence[Any] | None = None,
    collect_log: bool = False,
    skip_cache: bool | None = None,
    edge_cache: bool = False,
    cache_ttl: int | None = None,
    custom_cache_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    circuit_breaker: GatewayCircuitBreaker | None = None,
    gateway_fallback: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> list[tuple[int, float]]:
    """Rerank documents via OpenRouter rerank API; return ``(index, score)`` desc."""
    if not documents:
        return []
    body: dict[str, Any] = {"query": query, "documents": documents, **kwargs}
    if top_n is not None:
        body["top_n"] = top_n
    base_url, headers, payload, _ = _prepare_request(
        model=model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        zdr=zdr,
        zdr_available=zdr_available,
        zdr_endpoints=zdr_endpoints,
        collect_log=collect_log,
        skip_cache=skip_cache,
        edge_cache=edge_cache,
        cache_ttl=cache_ttl,
        custom_cache_key=custom_cache_key,
        extra_headers=extra_headers,
        extra_body=body,
    )
    data = await _apost_json(
        base_url=base_url,
        path="/rerank",
        headers=headers,
        payload=payload,
        timeout=timeout,
        circuit_breaker=circuit_breaker or _DEFAULT_BREAKER,
        gateway_fallback=gateway_fallback,
        transport=transport,
    )
    results = data.get("results") or []
    pairs = [(int(item["index"]), float(item["relevance_score"])) for item in results]
    pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    return pairs


async def asystemone(
    *,
    model: str,
    state: list[dict[str, Any]] | str,
    questions: list[dict[str, Any]] | dict[str, Any],
    env_prefix: str = "HULL_",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 60.0,
    zdr: bool = True,
    zdr_available: bool | None = None,
    zdr_endpoints: Sequence[Any] | None = None,
    collect_log: bool = False,
    skip_cache: bool | None = None,
    edge_cache: bool = False,
    cache_ttl: int | None = None,
    custom_cache_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    circuit_breaker: GatewayCircuitBreaker | None = None,
    gateway_fallback: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> SystemOneResponse:
    """Async ``POST /v1/systemone`` request for the ``decision`` profile."""
    body: dict[str, Any] = {"state": state, "questions": questions, **kwargs}
    base_url, headers, payload, zdr_enforced = _prepare_request(
        model=model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        zdr=zdr,
        zdr_available=zdr_available,
        zdr_endpoints=zdr_endpoints,
        collect_log=collect_log,
        skip_cache=skip_cache,
        edge_cache=edge_cache,
        cache_ttl=cache_ttl,
        custom_cache_key=custom_cache_key,
        extra_headers=extra_headers,
        extra_body=body,
    )
    data = await _apost_json(
        base_url=base_url,
        path=SYSTEMONE_PATH,
        headers=headers,
        payload=payload,
        timeout=timeout,
        circuit_breaker=circuit_breaker or _DEFAULT_BREAKER,
        gateway_fallback=gateway_fallback,
        transport=transport,
    )
    answers = data.get("answers")
    if not isinstance(answers, dict):
        answers = {k: v for k, v in data.items() if k not in {"id", "model", "usage"}}
    return SystemOneResponse(
        id=str(data.get("id") or ""),
        model=str(data.get("model") or payload["model"]),
        answers=dict(answers),
        usage=dict(data.get("usage") or {}),
        usage_metrics=extract_usage_metrics(data, zdr_enforced=zdr_enforced),
        raw=data,
    )


def systemone(
    *,
    model: str,
    state: list[dict[str, Any]] | str,
    questions: list[dict[str, Any]] | dict[str, Any],
    env_prefix: str = "HULL_",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 60.0,
    zdr: bool = True,
    zdr_available: bool | None = None,
    zdr_endpoints: Sequence[Any] | None = None,
    collect_log: bool = False,
    skip_cache: bool | None = None,
    edge_cache: bool = False,
    cache_ttl: int | None = None,
    custom_cache_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    circuit_breaker: GatewayCircuitBreaker | None = None,
    gateway_fallback: bool = True,
    transport: httpx.BaseTransport | None = None,
    **kwargs: Any,
) -> SystemOneResponse:
    """Sync sibling of :func:`asystemone` for ``POST /v1/systemone``."""
    body: dict[str, Any] = {"state": state, "questions": questions, **kwargs}
    base_url, headers, payload, zdr_enforced = _prepare_request(
        model=model,
        env_prefix=env_prefix,
        api_key=api_key,
        api_base=api_base,
        zdr=zdr,
        zdr_available=zdr_available,
        zdr_endpoints=zdr_endpoints,
        collect_log=collect_log,
        skip_cache=skip_cache,
        edge_cache=edge_cache,
        cache_ttl=cache_ttl,
        custom_cache_key=custom_cache_key,
        extra_headers=extra_headers,
        extra_body=body,
    )
    data = _post_json(
        base_url=base_url,
        path=SYSTEMONE_PATH,
        headers=headers,
        payload=payload,
        timeout=timeout,
        circuit_breaker=circuit_breaker or _DEFAULT_BREAKER,
        gateway_fallback=gateway_fallback,
        transport=transport,
    )
    answers = data.get("answers")
    if not isinstance(answers, dict):
        answers = {k: v for k, v in data.items() if k not in {"id", "model", "usage"}}
    return SystemOneResponse(
        id=str(data.get("id") or ""),
        model=str(data.get("model") or payload["model"]),
        answers=dict(answers),
        usage=dict(data.get("usage") or {}),
        usage_metrics=extract_usage_metrics(data, zdr_enforced=zdr_enforced),
        raw=data,
    )
