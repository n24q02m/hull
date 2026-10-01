"""Tests for selector inference utility functions.

Selector inference requires a caller-supplied ``llm_caller`` — provider API
keys in the environment (GEMINI_API_KEY / OPENAI_API_KEY / ...) are
deliberately ignored and no SDK dispatch lives in this module.
"""

import importlib
import json
import sys
from unittest.mock import MagicMock

import pytest

from hull_web.scraper import selector_inference
from hull_web.scraper.selector_inference import (
    infer_selectors_with_llm,
    merge_selectors,
)


def test_merge_selectors_disjoint():
    existing = {"title": ".title"}
    inferred = {"content": "#content"}
    expected = {"title": ".title", "content": "#content"}
    assert merge_selectors(existing, inferred) == expected


def test_merge_selectors_existing_priority():
    existing = {"title": ".existing-title"}
    inferred = {"title": ".inferred-title", "content": "#content"}
    expected = {"title": ".existing-title", "content": "#content"}
    assert merge_selectors(existing, inferred) == expected


def test_merge_selectors_empty_existing_uses_inferred():
    existing = {"title": ""}
    inferred = {"title": ".inferred-title", "content": "#content"}
    expected = {"title": ".inferred-title", "content": "#content"}
    assert merge_selectors(existing, inferred) == expected


def test_merge_selectors_missing_existing_uses_inferred():
    existing = {"content": "#content"}
    inferred = {"title": ".inferred-title"}
    expected = {"content": "#content", "title": ".inferred-title"}
    assert merge_selectors(existing, inferred) == expected


def test_merge_selectors_all_empty():
    assert merge_selectors({}, {}) == {}


def test_merge_selectors_no_inferred():
    existing = {"title": ".title"}
    assert merge_selectors(existing, {}) == {"title": ".title"}


def test_merge_selectors_no_existing():
    inferred = {"title": ".title"}
    assert merge_selectors({}, inferred) == {"title": ".title"}


def test_merge_selectors_no_existing_full():
    inferred = {"title": ".title", "content": "#content", "next_chapter": ".next"}
    assert merge_selectors({}, inferred) == inferred


def test_get_domain_selectors_wildcard(monkeypatch):
    # Verify wildcard-pattern matching infrastructure works correctly + does not
    # leak via subdomain-bypass (e.g. attacker spoofs `attacker.com.<wildcard>.evil.com`).
    # Uses a generic test-fixture wildcard pattern injected via monkeypatch — the
    # built-in DOMAIN_CONFIGS no longer ships site-specific wildcard configs.
    monkeypatch.setitem(sys.modules, "httpx", MagicMock())
    monkeypatch.setitem(sys.modules, "langgraph", MagicMock())
    monkeypatch.setitem(sys.modules, "langgraph.graph", MagicMock())

    import re as _re

    from hull_web.scraper import selector_inference

    fixture_pattern = "testsite*.com"
    fixture_config = {"content": "#main", "title": ".title"}
    monkeypatch.setitem(selector_inference.DOMAIN_CONFIGS, fixture_pattern, fixture_config)
    monkeypatch.setattr(
        selector_inference,
        "_WILDCARD_CONFIGS",
        [
            (
                _re.compile(_re.escape(fixture_pattern).replace(r"\*", r"[^.]*") + r"\Z"),
                fixture_config,
            )
        ],
    )

    from hull_web.scraper.selector_inference import get_domain_selectors

    # Valid matches against the generic wildcard
    assert get_domain_selectors("https://testsite123.com") is not None
    assert get_domain_selectors("https://testsite.com") is not None

    # Exact match works along with wildcard
    monkeypatch.setitem(selector_inference.DOMAIN_CONFIGS, "testsite.com", fixture_config)
    assert get_domain_selectors("https://testsite.com") is not None

    # Invalid matches (verify wildcard-bypass guards still hold)
    assert get_domain_selectors("https://testsite.com.evil.com") is None
    assert get_domain_selectors("https://eviltestsite.com") is None
    assert get_domain_selectors("https://testsite.com.co") is None


# -----------------------------------------------------------------------------
# Env-path removal: llm_caller is required, env keys are ignored.
# -----------------------------------------------------------------------------


_LLM_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "WEB_CORE_LLM_MODEL",
)


def _clear_llm_env(monkeypatch):
    for var in (*_LLM_ENV_VARS, "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)


async def test_infer_requires_llm_caller(monkeypatch):
    """Without an explicit llm_caller there is no inference at all."""
    _clear_llm_env(monkeypatch)
    with pytest.raises(TypeError, match="llm_caller"):
        await infer_selectors_with_llm("https://test-example.com", "<html/>")  # ty: ignore[missing-argument]


async def test_infer_ignores_env_api_keys(monkeypatch):
    """Provider API keys in the env must not enable any provider dispatch."""
    for var in _LLM_ENV_VARS:
        monkeypatch.setenv(var, "dummy")
    with pytest.raises(TypeError, match="llm_caller"):
        await infer_selectors_with_llm("https://test-example.com", "<html/>")  # ty: ignore[missing-argument]


def test_get_domain_selectors_does_not_inject_env_cookies(monkeypatch):
    """Domain cookies are no longer read from WEB_CORE_DOMAIN_COOKIES."""
    monkeypatch.setenv(
        "WEB_CORE_DOMAIN_COOKIES",
        json.dumps({"ncode.syosetu.com": {"session": "abc123"}}),
    )
    selectors = selector_inference.get_domain_selectors("https://ncode.syosetu.com/n1234abc/")
    assert selectors is not None
    assert "cookies" not in selectors


def test_env_cookie_mechanism_removed(monkeypatch):
    """The WEB_CORE_DOMAIN_COOKIES env mechanism no longer exists."""
    monkeypatch.setenv("WEB_CORE_DOMAIN_COOKIES", json.dumps({"d.com": {"a": "b"}}))
    importlib.reload(selector_inference)
    try:
        assert not hasattr(selector_inference, "DOMAIN_COOKIES")
        assert not hasattr(selector_inference, "_load_domain_cookies")
    finally:
        importlib.reload(selector_inference)


def test_env_cookies_do_not_create_domain_selectors(monkeypatch):
    """Cookie-only env domains still return None (no selector match)."""
    monkeypatch.setenv(
        "WEB_CORE_DOMAIN_COOKIES",
        json.dumps({"cookie-only-unknown.com": {"session": "123"}}),
    )
    assert selector_inference.get_domain_selectors("https://cookie-only-unknown.com") is None


# -----------------------------------------------------------------------------
# llm_caller-driven inference.
# -----------------------------------------------------------------------------


async def test_infer_explicit_llm_caller_used():
    async def fake_caller(_prompt, _html):
        return {"content": "#custom", "title": ".t", "next_chapter": "a.n"}

    result = await infer_selectors_with_llm(
        "https://test-example.com",
        "<html/>",
        llm_caller=fake_caller,
    )
    assert result == {"content": "#custom", "title": ".t", "next_chapter": "a.n"}


async def test_infer_llm_caller_returns_json_string():
    async def fake_caller(_prompt, _html):
        return json.dumps({"content": "#x", "title": ".y", "unrelated": "ignored"})

    result = await infer_selectors_with_llm(
        "https://test-example.com",
        "<html/>",
        llm_caller=fake_caller,
    )
    assert result == {"content": "#x", "title": ".y"}


async def test_infer_llm_caller_exception_returns_empty():
    async def boom(_prompt, _html):
        raise RuntimeError("provider down")

    result = await infer_selectors_with_llm(
        "https://test-example.com",
        "<html/>",
        llm_caller=boom,
    )
    assert result == {}


async def test_infer_llm_caller_import_error_returns_empty():
    async def missing_sdk(_prompt, _html):
        raise ImportError("openai not installed")

    result = await infer_selectors_with_llm(
        "https://test-example.com",
        "<html/>",
        llm_caller=missing_sdk,
    )
    assert result == {}


async def test_infer_llm_caller_returns_invalid_json():
    async def fake_caller(_prompt, _html):
        return "invalid { json"

    result = await infer_selectors_with_llm(
        "https://test-example.com",
        "<html/>",
        llm_caller=fake_caller,
    )
    assert result == {}


async def test_infer_llm_caller_returns_unexpected_type():
    async def fake_caller(_prompt, _html):
        return [1, 2, 3]  # Unexpected type

    result = await infer_selectors_with_llm(
        "https://test-example.com",
        "<html/>",
        llm_caller=fake_caller,
    )
    assert result == {}


async def test_infer_llm_caller_returns_raw_json_string():
    async def fake_caller(_prompt, _html):
        # Explicitly return a JSON string to exercise the 'if isinstance(raw, str):' branch
        return '{"content": "#raw", "title": ".raw"}'

    result = await infer_selectors_with_llm(
        "https://test-example.com",
        "<html/>",
        llm_caller=fake_caller,
    )
    assert result == {"content": "#raw", "title": ".raw"}


async def test_infer_domain_extraction_protocol_less():
    async def fake_caller(_prompt, _html):
        return {"content": "#c"}

    # Test with // protocol-less URL
    result = await infer_selectors_with_llm(
        "//test-example.com/path",
        "<html/>",
        llm_caller=fake_caller,
    )
    assert result == {"content": "#c"}


async def test_infer_logs_provider_annotations():
    """Caller-provided __hull_web_*__ annotations surface in the log record."""
    import logging

    async def fake_caller(_prompt, _html):
        return {"content": "#c"}

    # Caller annotations ride on the callable's own __dict__; functions allow it, a class body would not.
    fake_caller.__dict__["__hull_web_provider__"] = "custom-provider"
    fake_caller.__dict__["__hull_web_model__"] = "custom-model"

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)  # ty: ignore[invalid-assignment]  # capture handler

    logger = selector_inference.logger
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        await infer_selectors_with_llm("https://test-example.com", "<html/>", llm_caller=fake_caller)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    inferred = [r for r in records if r.message == "domain_selector_inferred"]
    assert inferred, "expected a domain_selector_inferred record"
    assert inferred[0].provider == "custom-provider"  # ty: ignore[unresolved-attribute]  # extra LogRecord attrs
    assert inferred[0].model == "custom-model"  # ty: ignore[unresolved-attribute]


# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------


def test_parse_selector_json_not_dict():
    assert selector_inference._parse_selector_json("[]") == {}


def test_parse_selector_json_values_not_strings():
    data = {"content": 123, "title": None, "next_chapter": ["abc"]}
    assert selector_inference._parse_selector_json(json.dumps(data)) == {}


def test_get_domain_selectors_completely_unknown_miss(monkeypatch):
    # Ensure we don't match any existing hardcoded domains or wildcards
    monkeypatch.setattr(selector_inference, "DOMAIN_CONFIGS", {})
    monkeypatch.setattr(selector_inference, "_WILDCARD_CONFIGS", [])

    url = "https://unknown.com"
    assert selector_inference.get_domain_selectors(url) is None


def test_get_domain_selectors_real_unknown_no_monkeypatch():
    # Verify that a domain definitely not in DOMAIN_CONFIGS returns None
    # without any monkeypatching of the configs.
    url = "https://this-domain-does-not-exist-12345.com/path"
    assert selector_inference.get_domain_selectors(url) is None


def test_get_domain_selectors_case_insensitivity():
    # Domain matching should be case-insensitive
    url = "HTTPS://NCODE.SYOSETU.COM/123"
    result = selector_inference.get_domain_selectors(url)
    assert result is not None
    assert result["content"] == "#novel_honbun"


def test_get_domain_selectors_empty_url():
    # Empty URL should result in empty domain and no match
    assert selector_inference.get_domain_selectors("") is None


def test_get_domain_selectors_invalid_url():
    # URL that yields empty domain
    assert selector_inference.get_domain_selectors("https://") is None
    assert selector_inference.get_domain_selectors("://") is None
