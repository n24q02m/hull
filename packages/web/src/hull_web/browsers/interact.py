"""Interactive page ops on a Playwright-compatible page, plus session opener.

Permanent home for ``InteractOps`` (moved from wet's ``interact_ops.py`` per
the author's own handoff note: "once upstream ships it, that module becomes a
re-export shim and eventually deletes").

Design:
    Stateful helper bound to a single Playwright-compatible ``Page``. Each
    method first attempts the native call; on timeout/error falls back to a
    best-effort "wait for selector then retry" path.

The selector-inference (LLM-resolved selector when raw selector is missing)
fallback is a NICE: ``description`` is accepted as an input but resolved with
a deterministic CSS-like heuristic here. A full LLM resolver can replace the
heuristic later without changing this API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class InteractOps:
    """Stateful interactive ops on a Playwright-compatible page.

    Args:
        page: A patchright (Playwright-compatible) ``Page`` instance.
    """

    def __init__(self, page: Any) -> None:
        self._page = page

    async def click(self, selector: str, timeout_ms: int = 10000) -> None:
        """Click an element matched by ``selector``.

        Falls back to ``wait_for(selector)`` + retry once on timeout.
        """
        try:
            await self._page.click(selector, timeout=timeout_ms)
        except Exception as exc:  # pragma: no cover - native exception bubble
            logger.debug("click(%r) primary failed: %s; retrying", selector, exc)
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            await self._page.click(selector, timeout=timeout_ms)

    async def fill(self, selector: str, value: str, timeout_ms: int = 10000) -> None:
        """Fill input matched by ``selector`` with ``value``."""
        try:
            await self._page.fill(selector, value, timeout=timeout_ms)
        except Exception as exc:  # pragma: no cover - native exception bubble
            logger.debug("fill(%r) primary failed: %s; retrying", selector, exc)
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            await self._page.fill(selector, value, timeout=timeout_ms)

    async def submit(self, selector: str, timeout_ms: int = 10000) -> None:
        """Submit a form matched by ``selector``.

        Playwright does not expose ``form.submit`` directly; the canonical
        pattern is ``page.locator(form).evaluate('f => f.submit()')``. We use
        the evaluate path so the action works on forms without a visible
        submit button.
        """
        try:
            locator = self._page.locator(selector)
            await locator.evaluate("(form) => form.submit()", timeout=timeout_ms)
        except Exception as exc:  # pragma: no cover - native exception bubble
            logger.debug("submit(%r) primary failed: %s; retrying", selector, exc)
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            locator = self._page.locator(selector)
            await locator.evaluate("(form) => form.submit()", timeout=timeout_ms)

    async def screenshot(self, full_page: bool = False) -> bytes:
        """Capture a PNG screenshot of the page; returns raw bytes."""
        return await self._page.screenshot(full_page=full_page, type="png")

    async def wait_for(self, selector: str, state: str = "visible", timeout_ms: int = 10000) -> None:
        """Wait for ``selector`` to reach ``state`` (visible / hidden / attached / detached)."""
        await self._page.wait_for_selector(selector, state=state, timeout=timeout_ms)

    async def evaluate(self, expression: str) -> Any:
        """Evaluate a JS expression in the page context.

        Security note: server dispatchers must NOT expose this method to
        external callers — only in-process orchestration may invoke it.
        """
        return await self._page.evaluate(expression)


async def open_interact_session(
    url: str,
    headless: bool = True,
    timeout_ms: int = 30000,
    provider: Any = None,
) -> tuple[Any, Any, Any, InteractOps]:
    """Launch a browser, open ``url``, return handles + ops.

    Returns ``(playwright, browser, page, ops)`` so the caller is responsible
    for the matched ``await browser.close()`` / ``await playwright.stop()``
    lifecycle when no session pool is in use.

    With ``provider=None`` the legacy direct patchright launch is used and
    ``playwright`` is the real started Playwright instance. With an injected
    ``BrowserProvider`` (see :mod:`hull_web.browsers.protocol`) the provider
    owns the lifecycle instead: ``playwright`` is ``None`` and the caller must
    ``await provider.close()`` after ``browser.close()``.
    """
    if provider is not None:
        browser = await provider.launch(config={"headless": headless})
        page = await browser.new_page()
        await page.goto(url, timeout=timeout_ms)
        return None, browser, page, InteractOps(page)

    # Lazy import — patchright + playwright are heavy and only needed
    # when interact actions actually run.
    def _import_async_playwright():
        from patchright.async_api import async_playwright as _ap

        return _ap

    async_playwright = await asyncio.to_thread(_import_async_playwright)
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(url, timeout=timeout_ms)
    return pw, browser, page, InteractOps(page)
