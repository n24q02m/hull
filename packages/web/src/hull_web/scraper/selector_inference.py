"""LLM-based CSS selector inference for autonomous content extraction.

When the scraping agent gets valid HTML but existing selectors fail to extract
meaningful content, this module uses an LLM to analyze the page structure and
infer correct CSS selectors for content, title, and navigation elements.

The LLM is always supplied by the caller as an ``llm_caller`` callable (async
``(prompt, html_content) -> selector dict | raw JSON str``). This module does
no provider auto-detection from the environment, reads no API keys, and ships
no SDK dispatch — hosts that want LLM inference wire their own caller
explicitly. See ``infer_selectors_with_llm``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable

from hull_web.http.url import extract_domain

logger = logging.getLogger(__name__)

LLMCaller = Callable[[str, str], Awaitable[dict[str, str]]]
"""Signature: async (prompt, html_content) -> selector dict."""

# Built-in domain configs for known sites — saves LLM calls
DOMAIN_CONFIGS: dict[str, dict[str, str]] = {
    "ncode.syosetu.com": {
        "content": "#novel_honbun",
        "title": ".novel_title, .novel_subtitle",
        "next_chapter": "a.novelview_pager-next",
    },
    "kakuyomu.jp": {
        "content": ".widget-episodeBody",
        "title": ".widget-episodeTitle",
        "next_chapter": "a[rel='next']",
    },
    "www.pixiv.net": {
        "content": ".novel-content",
        "title": ".work-info__title",
    },
    "mangadex.org": {
        "content": ".md-chapter-page img",
        "title": ".manga-title",
    },
}

# Pre-compile wildcard patterns for fast lookup
_WILDCARD_CONFIGS: list[tuple[re.Pattern[str], dict[str, str]]] = [
    (re.compile(re.escape(pattern).replace(r"\*", r"[^.]*") + r"\Z"), config)
    for pattern, config in DOMAIN_CONFIGS.items()
    if "*" in pattern
]

# Prompt cho LLM infer selectors tu HTML
_INFER_SELECTORS_PROMPT = """\
You are a CSS selector expert. Analyze this HTML and extract the best CSS selectors.

URL: {url}
HTML (truncated to first 5000 chars):
```html
{html_snippet}
```

Return JSON with CSS selectors for:
- "content": the main content area (article body, novel text, manga images)
- "title": the page/chapter title
- "next_chapter": link to next chapter (if pagination exists)

Rules:
- Prefer ID selectors (#id) over class selectors (.class)
- Avoid generic selectors like "div", "p", "span" alone
- For manga/image pages, select the image container
- Return ONLY valid JSON, no explanation

Example response:
{{"content": "#novel_honbun", "title": ".novel_title", "next_chapter": "a.next"}}"""


def get_domain_selectors(url: str) -> dict[str, str] | None:
    """Return built-in selectors for a known domain, or None.

    Logs domain usage for analytics — enabling the Tiered Scraping
    feedback loop (track unknown domains → hardcode popular ones).
    """
    # Performance Optimization: Reusing extract_domain which implements the same
    # fast path string partitioning but includes an LRU cache (~3-4x faster for repeated URLs)
    domain = extract_domain(url).lower()

    selectors: dict[str, str] | None = None

    # Exact match
    if domain in DOMAIN_CONFIGS:
        selectors = DOMAIN_CONFIGS[domain].copy()
        logger.info(
            "domain_selector_hit",
            extra={"domain": domain, "tier": "hardcoded", "url": url},
        )
    else:
        # Wildcard match (e.g. example*.com pattern in DOMAIN_CONFIGS)
        for pattern_re, config in _WILDCARD_CONFIGS:
            if pattern_re.match(domain):
                selectors = config.copy()
                logger.info(
                    "domain_selector_hit",
                    extra={
                        "domain": domain,
                        "tier": "hardcoded_wildcard",
                        "pattern": pattern_re.pattern,
                        "url": url,
                    },
                )
                break

    # Log unknown domain — candidate for future hardcoding
    if selectors is None:
        logger.info(
            "domain_selector_miss",
            extra={"domain": domain, "tier": "unknown", "url": url},
        )

    return selectors


def _build_prompt(url: str, html_content: str) -> str:
    """Build the selector-inference prompt, truncating HTML to first 5000 chars."""
    html_snippet = html_content[:5000]
    return _INFER_SELECTORS_PROMPT.format(url=url, html_snippet=html_snippet)


def _parse_selector_json(text: str) -> dict[str, str]:
    """Parse a JSON response into a whitelisted selector dict."""
    result = json.loads(text or "")
    selectors: dict[str, str] = {}
    if isinstance(result, dict):
        for key in ("content", "title", "next_chapter"):
            value = result.get(key)
            if isinstance(value, str):
                selectors[key] = value
    return selectors


async def infer_selectors_with_llm(
    url: str,
    html_content: str,
    *,
    llm_caller: LLMCaller,
) -> dict[str, str]:
    """Use LLM to infer CSS selectors from HTML structure.

    ``llm_caller`` is required — it is the supported path. The callable
    receives ``(prompt, html_content)`` and returns either a selector dict
    or a raw JSON text. There is no env-based provider fallback.

    Never raises: on any provider error we log and return ``{}`` so that the
    ``ScrapingAgent`` can continue with domain-config and empty selectors.
    """
    prompt = _build_prompt(url, html_content)

    try:
        raw = await llm_caller(prompt, html_content)
    except ImportError as e:
        logger.debug("selector_inference: provider SDK not installed, skipping", extra={"error": str(e)})
        return {}
    except Exception as e:
        logger.warning("LLM selector inference failed", extra={"error": str(e)})
        return {}

    # llm_caller may return a dict directly (already parsed) or raw JSON text.
    if isinstance(raw, str):
        try:
            selectors = _parse_selector_json(raw)
        except json.JSONDecodeError as e:
            logger.warning("LLM selector inference returned invalid JSON", extra={"error": str(e)})
            return {}
    elif isinstance(raw, dict):
        selectors = {k: v for k, v in raw.items() if k in {"content", "title", "next_chapter"} and isinstance(v, str)}
    else:
        logger.warning("LLM selector inference returned unexpected type", extra={"type": str(type(raw))})
        return {}

    # Performance Optimization: Reusing extract_domain which implements the same
    # fast path string partitioning but includes an LRU cache (~3-4x faster for repeated URLs)
    domain = extract_domain(url).lower()
    provider_name = getattr(llm_caller, "__hull_web_provider__", "custom")
    resolved_model = getattr(llm_caller, "__hull_web_model__", None)
    logger.info(
        "domain_selector_inferred",
        extra={
            "domain": domain,
            "tier": "llm_inferred",
            "url": url,
            "selectors": selectors,
            "provider": provider_name,
            "model": resolved_model,
        },
    )
    return selectors


def merge_selectors(
    existing: dict[str, str],
    inferred: dict[str, str],
) -> dict[str, str]:
    """Merge selectors, preferring existing non-empty values."""
    merged = {**inferred}
    for key, value in existing.items():
        if value:  # Existing non-empty takes priority
            merged[key] = value
    return merged
