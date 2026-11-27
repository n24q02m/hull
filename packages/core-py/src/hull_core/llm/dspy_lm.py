"""DSPy LM integration using dspy >= 3.5 native lm15 routing (no litellm)."""

from __future__ import annotations

from typing import Any

import dspy

from hull_core.llm.transport import OPENROUTER_API_BASE, provider_params


def _routed_model(model: str) -> tuple[str, str]:
    """Normalize a model string for dspy >= 3.5 native lm15 routing."""
    if model.startswith(("openrouter/", "openrouter:")):
        return "openrouter", model
    if model.startswith(("typesafe/", "typesafe:")):
        return "typesafe", model
    if "/" not in model and ":" not in model:
        return "openrouter", f"openrouter/{model}"
    head = model.split(":", 1)[0] if ":" in model and "/" not in model.split(":", 1)[0] else model.split("/", 1)[0]
    from dspy._vendor.lm15.router import LITELLM_PROVIDER_PREFIXES

    if head in LITELLM_PROVIDER_PREFIXES:
        return head, model
    return "openrouter", f"openrouter/{model}"


def build_lm(
    model: str,
    *,
    env_prefix: str = "HULL_",
    temperature: float,
    top_p: float,
    max_tokens: int | None = None,
    engine: Any = "lm15",
    zdr: bool = True,
    **kwargs: Any,
) -> dspy.LM:
    """Build a ``dspy.LM`` using dspy >= 3.5 native ``lm15`` routing."""
    provider, routed = _routed_model(model)
    params = provider_params(provider, env_prefix=env_prefix)
    merged: dict[str, Any] = {**params, **kwargs}
    if provider == "openrouter":
        merged.setdefault("api_base", OPENROUTER_API_BASE)
        if zdr:
            extra_body = dict(merged.get("extra_body") or {})
            prov = dict(extra_body.get("provider") or {})
            prov.setdefault("data_collection", "deny")
            extra_body["provider"] = prov
            merged["extra_body"] = extra_body
    return dspy.LM(
        routed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        engine=engine,
        **merged,
    )
