"""DSPy LM integration on the unified transport (``[dspy]`` extra).

``build_lm`` is the generic factory: every model id becomes a ``dspy.LM``
(litellm library mode) with ``provider_params`` credentials resolved from
the same ``env_prefix`` namespace as the transport. A bare name (no
provider prefix) is OpenRouter-routed — OpenRouter is the single default
for LM/embed/rerank; any other provider goes through its generic litellm
id (e.g. ``vertex_ai/...``, ``cohere/...``).
"""

from __future__ import annotations

from typing import Any

import dspy

from hull_core.llm.transport import provider_params


def build_lm(
    model: str,
    *,
    env_prefix: str = "HULL_",
    temperature: float,
    top_p: float,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> dspy.LM:
    """Build a ``dspy.LM`` for a provider-prefixed ``model`` id.

    Every id -> ``dspy.LM(f"{model}")`` with ``provider_params(provider)``
    merged under explicit ``kwargs``.
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

    params = provider_params(provider, env_prefix=env_prefix)
    return dspy.LM(
        model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **params,
        **kwargs,
    )
