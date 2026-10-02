"""Tests for InteractOps + provider-parameterized session opener."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from hull_web.browsers.interact import InteractOps, open_interact_session


def _page() -> MagicMock:
    page = MagicMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.goto = AsyncMock()
    page.new_page = AsyncMock()
    return page


async def test_open_interact_session_with_provider():
    page = _page()
    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)

    provider = MagicMock()
    provider.launch = AsyncMock(return_value=browser)
    provider.close = AsyncMock()

    pw, launched, got_page, ops = await open_interact_session("https://example.com", provider=provider)

    assert pw is None  # provider owns the lifecycle
    assert launched is browser
    assert got_page is page
    assert isinstance(ops, InteractOps)
    provider.launch.assert_awaited_once_with(config={"headless": True})
    page.goto.assert_awaited_once_with("https://example.com", timeout=30000)


async def test_interact_ops_click_retry_fallback():
    page = _page()
    page.click.side_effect = [TimeoutError("slow"), None]
    ops = InteractOps(page)

    await ops.click("#btn", timeout_ms=500)

    page.wait_for_selector.assert_awaited_once_with("#btn", timeout=500)
    assert page.click.await_count == 2


async def test_interact_ops_evaluate_passthrough():
    page = _page()
    page.evaluate = AsyncMock(return_value=42)
    ops = InteractOps(page)

    assert await ops.evaluate("1+1") == 42
