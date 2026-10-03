"""DSPy LM integration on the unified transport (``[dspy]`` extra).

``VertexExpressLM`` is a ``dspy.LM`` that routes ``vertex_express/`` models
through the Express passthrough (litellm rejects the Express API key,
BerriAI/litellm#21036): ``forward``/``aforward`` bypass litellm and return the
litellm ``ModelResponse`` the adapter produces, so DSPy's own response
processing + history/token accounting (``LM.__call__`` ->
``_process_lm_response`` -> ``update_history``) work unchanged and
``dspy.inspect_history`` keeps reporting.

``build_lm`` is the generic factory: ``vertex_express/`` -> ``VertexExpressLM``,
everything else -> ``dspy.LM`` with ``provider_params`` credentials resolved
from the same ``env_prefix`` namespace as the transport.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dspy

from hull_core.llm.transport import provider_params
from hull_core.llm.vertex_express import (
    VERTEX_EXPRESS_PREFIX,
    acompletion_express,
    completion_express,
)


class VertexExpressLM(dspy.LM):
    """DSPy LM that routes gemini through the Vertex Express passthrough.

    Constructor stores temperature/top_p/max_tokens in ``self.kwargs`` (the
    DSPy convention); ``forward`` merges them onto the per-call kwargs.
    ``output_ceiling`` (optional) is called with the message list to size
    ``max_tokens`` per call when no explicit value was given -- hosts port
    their prompt-scaling policy here rather than this module owning a sizing
    constant (see KP ``dspy_loader.scaled_output_ceiling``).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        env_prefix: str = "HULL_",
        output_ceiling: Callable[[list[dict[str, Any]]], int] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._express_api_key = api_key
        self._env_prefix = env_prefix
        self._output_ceiling = output_ceiling

    def _express_kwargs(self, call_kwargs: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        merged = {**self.kwargs, **call_kwargs}
        # DSPy-internal knobs the Express adapter neither needs nor accepts.
        for key in ("cache", "rollout_id", "num_retries"):
            merged.pop(key, None)
        # An unset max_tokens means "size it to this prompt" when an
        # output_ceiling policy was supplied; an explicit value (constructor
        # or per-call) is the caller's deliberate choice and is left alone.
        if merged.get("max_tokens") is None and self._output_ceiling is not None:
            merged["max_tokens"] = self._output_ceiling(messages)
        return merged

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        messages = messages or [{"role": "user", "content": prompt}]
        return completion_express(
            model=self.model,
            messages=messages,
            api_key=self._express_api_key,
            env_prefix=self._env_prefix,
            **self._express_kwargs(dict(kwargs), messages),
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        messages = messages or [{"role": "user", "content": prompt}]
        return await acompletion_express(
            model=self.model,
            messages=messages,
            api_key=self._express_api_key,
            env_prefix=self._env_prefix,
            **self._express_kwargs(dict(kwargs), messages),
        )


def build_lm(
    model: str,
    *,
    env_prefix: str = "HULL_",
    temperature: float,
    top_p: float,
    max_tokens: int | None = None,
    output_ceiling: Callable[[list[dict[str, Any]]], int] | None = None,
    thinking_budget: int | None = None,
    **kwargs: Any,
) -> dspy.LM:
    """Build a ``dspy.LM`` for a provider-prefixed ``model`` id.

    ``vertex_express/<id>`` -> :class:`VertexExpressLM` (passthrough).
    ``xai/<id>`` -> ``dspy.LM`` on the OpenAI-compatible endpoint
    (``https://api.x.ai/v1`` or the CF AI Gateway when configured), because
    litellm reads ``OPENAI_API_KEY`` for the ``openai/`` prefix, not the xai
    env namespace. Every other id -> ``dspy.LM(f"{model}")`` with
    ``provider_params(provider)`` merged under explicit ``kwargs``.
    """
    # A bare name (no "/" — e.g. "glm-5-flash") is OpenRouter-routed, matching
    # the transport's default-provider convention; a "provider/model" id keeps
    # its own provider (litellm cannot distinguish "z-ai/glm-5" from a real
    # provider prefix, and neither do we — OpenRouter slugs arrive prefixed).
    if "/" in model:
        provider = model.split("/", 1)[0]
    else:
        provider = "openrouter"
        model = f"openrouter/{model}"

    if provider == VERTEX_EXPRESS_PREFIX:
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        return VertexExpressLM(
            model,
            env_prefix=env_prefix,
            output_ceiling=output_ceiling,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "xai":
        # litellm's "openai/" prefix honours api_base + api_key kwargs.
        bare = model.split("/", 1)[1]
        params = provider_params("xai", env_prefix=env_prefix)
        api_base = params.pop("api_base", None) or "https://api.x.ai/v1"
        return dspy.LM(
            f"openai/{bare}",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            api_base=api_base,
            **params,
            **kwargs,
        )

    params = provider_params(provider, env_prefix=env_prefix)
    return dspy.LM(
        model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **params,
        **kwargs,
    )
