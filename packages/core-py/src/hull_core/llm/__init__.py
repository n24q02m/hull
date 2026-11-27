"""Unified OpenRouter-first LLM transport (no litellm in default path).

Default calls route directly over ``httpx`` + :mod:`hull_core.http.ssrf` with
strict ZDR, CF AI Gateway ``cf-aig-*`` safety headers, ``cache_control``
pass-through, circuit breaker fallback, and ``/v1/systemone`` support.
"""

from __future__ import annotations

from hull_core.llm.transport import (
    OPENROUTER_API_BASE,
    SYSTEMONE_PATH,
    AsyncCompletionStream,
    Choice,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    GatewayCircuitBreaker,
    Message,
    SyncCompletionStream,
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

__all__ = [
    "OPENROUTER_API_BASE",
    "SYSTEMONE_PATH",
    "AsyncCompletionStream",
    "Choice",
    "CompletionResponse",
    "EmbeddingData",
    "EmbeddingResponse",
    "GatewayCircuitBreaker",
    "Message",
    "SyncCompletionStream",
    "SystemOneResponse",
    "TransportHTTPError",
    "UsageMetrics",
    "ZDRUnavailableError",
    "acompletion",
    "acompletion_text",
    "aembedding",
    "arerank",
    "asystemone",
    "cf_aig_headers",
    "completion",
    "completion_text",
    "extract_usage_metrics",
    "provider_params",
    "reset_gateway_circuit_breaker",
    "systemone",
]
