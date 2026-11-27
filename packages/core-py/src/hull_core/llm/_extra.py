"""Optional legacy [litellm] extra helper for hull_core.llm.

Default hull_core.llm transport routes directly through OpenRouter over httpx +
SSRF vetting without litellm. Consumers (wet, crg, mnemo) that explicitly invoke
the legacy litellm compatibility path opt in via ``pip install "n24q02m-hull[litellm]"``.
"""

from __future__ import annotations


def require_litellm_extra() -> None:
    try:
        import litellm  # noqa: F401  # ty: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Legacy litellm transport requires the optional 'litellm' extra; "
            'install it with: pip install "n24q02m-hull[litellm]"'
        ) from exc
