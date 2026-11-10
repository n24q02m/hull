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

Per-request URL guard:
    ``open_interact_session(..., url_guard=guard)`` installs interception
    before the first navigation, so a URL the guard rejects is never
    requested: the initial URL is checked before anything is launched, every
    request Playwright routes (the document and each sub-resource) is checked
    as it is made, and on Chromium every redirect hop is checked through a CDP
    ``Fetch`` interceptor too. That last part is not optional decoration:
    Playwright's route layer deliberately calls a handler only for the first
    request of a redirect chain (microsoft/playwright#34994), so routing alone
    leaves ``302 -> http://127.0.0.1/`` reachable.

    The guard receives the URL as the browser is about to request it, not the
    address it resolves to. A guard that wants DNS-rebinding coverage must
    resolve the hostname on every call -- :func:`hull_web.http.is_safe_url`
    does exactly that, with a pinned resolution cache, and is the intended
    value. A guard that raises is treated as a rejection (fail closed). A
    rejection that stops a navigation surfaces as
    :class:`UrlGuardBlockedError`; one that stops a sub-resource is logged and
    leaves the document alone.

    ``url_guard=None`` (the default) keeps the historical behaviour: no
    interception is installed at all.

    Limits: engines without CDP (the Firefox ``invisible`` provider) get route
    interception only -- the initial URL and sub-resources are still checked,
    redirect hops are not. And no route-layer check can pin the address the
    browser dials for an allowed URL, so a rebinding that happens between the
    guard's own resolution and the browser's connection is outside its reach.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
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


class UrlGuardBlockedError(RuntimeError):
    """A navigation the interact session attempted was rejected by ``url_guard``.

    Attributes:
        url: The rejected URL, exactly as the browser was about to request it.
    """

    def __init__(self, url: str) -> None:
        super().__init__(f"url_guard rejected request to {url}")
        self.url = url


def _guard_allows(url_guard: Callable[[str], bool], url: str) -> bool:
    """Run ``url_guard`` on ``url``; a guard that raises rejects the URL."""
    try:
        return bool(url_guard(url))
    except Exception:
        logger.warning("url_guard raised on %r; treating the URL as rejected", url, exc_info=True)
        return False


def _is_navigation_request(request: Any) -> bool:
    """True when a routed request is the document itself, not a sub-resource."""
    is_navigation_request = getattr(request, "is_navigation_request", None)
    if is_navigation_request is None:
        return False
    try:
        return bool(is_navigation_request())
    except Exception:  # pragma: no cover - defensive: stub request objects
        return False


async def _install_route_guard(
    target: Any,
    url_guard: Callable[[str], bool],
    blocked: list[tuple[str, bool]],
) -> None:
    """Install the per-request guard on a Playwright ``Page``/``BrowserContext``.

    Must run before the first navigation: the handler aborts every request the
    guard rejects and continues every request it allows.
    """

    async def _route_guard(route: Any) -> None:
        request = route.request
        request_url = request.url
        if _guard_allows(url_guard, request_url):
            await route.continue_()
            return
        blocked.append((request_url, _is_navigation_request(request)))
        logger.warning("url_guard rejected browser request: %s", request_url)
        await route.abort("blockedbyclient")

    await target.route("**/*", _route_guard)


async def _install_redirect_guard(
    page: Any,
    url_guard: Callable[[str], bool],
    blocked: list[tuple[str, bool]],
) -> None:
    """Check redirect hops through the CDP ``Fetch`` domain (Chromium only).

    ``route`` handlers never see the request a redirect leads to, so without
    this a ``302`` to a private address reaches it unchecked. Chromium reports
    each hop as a new ``Fetch.requestPaused`` (``redirectedRequestId`` set),
    which can be failed before it leaves. Where CDP is unavailable the guard
    degrades to route interception -- the inherited limit of non-Chromium
    engines -- and says so in the log.
    """
    context = getattr(page, "context", None)
    new_cdp_session = getattr(context, "new_cdp_session", None)
    if new_cdp_session is None:
        logger.debug("browser exposes no CDP session; redirect hops are not guarded")
        return

    session: Any = None

    async def _on_request_paused(params: dict[str, Any]) -> None:
        request = params.get("request") or {}
        request_url = request.get("url", "")
        is_navigation = params.get("resourceType") == "Document" or "redirectedRequestId" in params
        if _guard_allows(url_guard, request_url):
            await session.send("Fetch.continueRequest", {"requestId": params["requestId"]})
            return
        blocked.append((request_url, is_navigation))
        logger.warning("url_guard rejected browser request: %s", request_url)
        await session.send(
            "Fetch.failRequest",
            {"requestId": params["requestId"], "errorReason": "BlockedByClient"},
        )

    try:
        session = await new_cdp_session(page)
        session.on("Fetch.requestPaused", _on_request_paused)
        await session.send(
            "Fetch.enable",
            {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]},
        )
    except Exception as exc:
        logger.warning("CDP redirect guard unavailable; redirect hops are not checked: %s", exc)


async def _guard_page(page: Any, url_guard: Callable[[str], bool] | None) -> list[tuple[str, bool]]:
    """Install the guard for ``page`` and return the blocked-request log it fills."""
    blocked: list[tuple[str, bool]] = []
    if url_guard is None:
        return blocked
    # The context covers popups/frames the page opens later; a provider that
    # hands back a bare page still gets page-scoped routing.
    await _install_route_guard(getattr(page, "context", None) or page, url_guard, blocked)
    await _install_redirect_guard(page, url_guard, blocked)
    return blocked


async def _navigate(page: Any, url: str, timeout_ms: int, blocked: list[tuple[str, bool]]) -> None:
    """``page.goto`` with a guard rejection turned into :class:`UrlGuardBlockedError`."""
    try:
        await page.goto(url, timeout=timeout_ms)
    except Exception as exc:
        blocked_navigation = next((blocked_url for blocked_url, navigation in reversed(blocked) if navigation), None)
        if blocked_navigation is None:
            raise
        raise UrlGuardBlockedError(blocked_navigation) from exc


def _import_async_playwright() -> Any:
    """Import patchright's ``async_playwright`` (heavy, so off the event loop)."""
    from patchright.async_api import async_playwright

    return async_playwright


async def open_interact_session(
    url: str,
    headless: bool = True,
    timeout_ms: int = 30000,
    provider: Any = None,
    url_guard: Callable[[str], bool] | None = None,
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

    Args:
        url: First navigation of the session.
        headless: Launch headless.
        timeout_ms: Navigation timeout.
        provider: Optional :class:`BrowserProvider` that owns the browser.
        url_guard: Optional synchronous predicate consulted for the initial
            URL and for every request the browser then makes -- sub-resources
            and, on Chromium, redirect hops. A rejected initial URL raises
            :class:`UrlGuardBlockedError` before the browser is launched; a
            rejected redirect hop or document raises it instead of navigating,
            and a rejected sub-resource is aborted while the document loads.
            ``None`` installs no interception and keeps the behaviour every
            earlier version had.

    When the direct launch is in use and the guard rejects a URL after the
    browser started, the browser and Playwright instance are closed before
    :class:`UrlGuardBlockedError` propagates; a provider-owned browser is left
    to the caller that owns the provider.
    """
    if url_guard is not None and not _guard_allows(url_guard, url):
        raise UrlGuardBlockedError(url)

    if provider is not None:
        browser = await provider.launch(config={"headless": headless})
        page = await browser.new_page()
        await _navigate(page, url, timeout_ms, await _guard_page(page, url_guard))
        return None, browser, page, InteractOps(page)

    # Lazy import — patchright + playwright are heavy and only needed
    # when interact actions actually run.
    async_playwright = await asyncio.to_thread(_import_async_playwright)
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"],
    )
    try:
        context = await browser.new_context()
        page = await context.new_page()
        await _navigate(page, url, timeout_ms, await _guard_page(page, url_guard))
    except BaseException:
        # A rejected URL is an expected outcome now, so the browser this
        # function launched must not outlive the exception. Provider-owned
        # lifecycles stay the caller's: it still holds the provider.
        with contextlib.suppress(Exception):
            await browser.close()
        with contextlib.suppress(Exception):
            await pw.stop()
        raise
    return pw, browser, page, InteractOps(page)
