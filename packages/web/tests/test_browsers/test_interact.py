"""Tests for InteractOps + provider-parameterized session opener.

The guard tests come in two layers:

* mocked route/CDP objects, which pin the contract the opener installs and the
  error it raises — these are the ones CI runs;
* a real Chromium with two local HTTP servers, which prove the two attacks the
  guard exists for (a redirect to loopback, a sub-resource to a private
  address). They skip when patchright's Chromium is not installed, which is the
  case on a bare CI runner.
"""

from __future__ import annotations

import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hull_web.browsers.interact import InteractOps, UrlGuardBlockedError, open_interact_session


def _page() -> MagicMock:
    page = MagicMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.goto = AsyncMock()
    page.new_page = AsyncMock()
    return page


def _mock_route(url: str, *, navigation: bool = False) -> MagicMock:
    route = MagicMock()
    route.request.url = url
    route.request.is_navigation_request = MagicMock(return_value=navigation)
    route.continue_ = AsyncMock()
    route.abort = AsyncMock()
    return route


def _provider_session() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Provider, browser, page, context and CDP session, wired for the guard."""
    page = _page()
    context = MagicMock()
    context.route = AsyncMock()
    cdp = MagicMock()
    cdp.on = MagicMock()
    cdp.send = AsyncMock()
    context.new_cdp_session = AsyncMock(return_value=cdp)
    page.context = context

    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)
    provider = MagicMock()
    provider.launch = AsyncMock(return_value=browser)
    provider.close = AsyncMock()
    return provider, browser, page, context, cdp


def _route_handler(context: MagicMock) -> Any:
    assert context.route.await_args is not None
    assert context.route.await_args.args[0] == "**/*"
    return context.route.await_args.args[1]


def _cdp_handler(cdp: MagicMock) -> Any:
    assert cdp.on.call_args is not None
    assert cdp.on.call_args.args[0] == "Fetch.requestPaused"
    return cdp.on.call_args.args[1]


def _paused(url: str, *, resource_type: str = "Document", redirected: bool = True) -> dict[str, Any]:
    params: dict[str, Any] = {
        "requestId": f"req-{url}",
        "resourceType": resource_type,
        "request": {"url": url},
    }
    if redirected:
        params["redirectedRequestId"] = "interception-job-1.0"
    return params


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


async def test_url_guard_none_installs_no_interception():
    provider, _browser, page, context, _cdp = _provider_session()

    await open_interact_session("https://example.com", provider=provider)

    context.route.assert_not_awaited()
    context.new_cdp_session.assert_not_awaited()
    page.goto.assert_awaited_once_with("https://example.com", timeout=30000)


async def test_url_guard_rejects_initial_url_before_launch():
    provider, _browser, page, _context, _cdp = _provider_session()
    guard = MagicMock(return_value=False)

    with pytest.raises(UrlGuardBlockedError) as caught:
        await open_interact_session("http://127.0.0.1:8080/admin", provider=provider, url_guard=guard)

    assert caught.value.url == "http://127.0.0.1:8080/admin"
    guard.assert_called_once_with("http://127.0.0.1:8080/admin")
    provider.launch.assert_not_awaited()
    page.goto.assert_not_awaited()


async def test_url_guard_that_raises_is_a_rejection():
    provider, _browser, _page_obj, _context, _cdp = _provider_session()

    def guard(_url: str) -> bool:
        raise RuntimeError("resolver down")

    with pytest.raises(UrlGuardBlockedError):
        await open_interact_session("https://example.com", provider=provider, url_guard=guard)

    provider.launch.assert_not_awaited()


async def test_url_guard_intercepts_before_navigation():
    provider, _browser, page, context, cdp = _provider_session()
    order: list[str] = []
    context.route = AsyncMock(side_effect=lambda *args: order.append("route"))
    cdp.send = AsyncMock(side_effect=lambda *args, **kwargs: order.append("cdp-fetch-enable"))
    page.goto = AsyncMock(side_effect=lambda *args, **kwargs: order.append("goto"))

    await open_interact_session("https://example.com", provider=provider, url_guard=lambda _url: True)

    assert order == ["route", "cdp-fetch-enable", "goto"]
    assert cdp.send.await_args is not None
    assert cdp.send.await_args.args == (
        "Fetch.enable",
        {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]},
    )


async def test_url_guard_route_handler_aborts_and_continues():
    provider, _browser, _page_obj, context, _cdp = _provider_session()
    guard = MagicMock(side_effect=lambda url: "127.0.0.1" not in url)

    await open_interact_session("https://example.com", provider=provider, url_guard=guard)
    handler = _route_handler(context)

    allowed = _mock_route("https://cdn.example.com/app.js")
    await handler(allowed)
    allowed.continue_.assert_awaited_once_with()
    allowed.abort.assert_not_awaited()

    rejected = _mock_route("http://127.0.0.1:8080/secret", navigation=True)
    await handler(rejected)
    rejected.abort.assert_awaited_once_with("blockedbyclient")
    rejected.continue_.assert_not_awaited()


async def test_url_guard_cdp_handler_blocks_redirect_hop():
    provider, _browser, _page_obj, _context, cdp = _provider_session()

    await open_interact_session(
        "https://example.com",
        provider=provider,
        url_guard=lambda url: "127.0.0.1" not in url,
    )
    handler = _cdp_handler(cdp)

    await handler(_paused("http://127.0.0.1:8080/secret"))
    assert cdp.send.await_args is not None
    assert cdp.send.await_args.args == (
        "Fetch.failRequest",
        {"requestId": "req-http://127.0.0.1:8080/secret", "errorReason": "BlockedByClient"},
    )

    await handler(_paused("https://example.com/next", redirected=False))
    assert cdp.send.await_args is not None
    assert cdp.send.await_args.args == ("Fetch.continueRequest", {"requestId": "req-https://example.com/next"})


async def test_url_guard_blocked_redirect_raises_named_error():
    provider, _browser, page, _context, cdp = _provider_session()
    private = "http://127.0.0.1:8080/secret"

    async def goto_dies_like_chromium(_url: str, timeout: int) -> None:
        await _cdp_handler(cdp)(_paused(private))
        raise RuntimeError("Page.goto: net::ERR_BLOCKED_BY_CLIENT")

    page.goto = AsyncMock(side_effect=goto_dies_like_chromium)

    with pytest.raises(UrlGuardBlockedError) as caught:
        await open_interact_session(
            "https://example.com/redirect",
            provider=provider,
            url_guard=lambda url: "127.0.0.1" not in url,
        )

    assert caught.value.url == private
    assert isinstance(caught.value.__cause__, RuntimeError)


async def test_url_guard_leaves_unrelated_goto_failure_alone():
    provider, _browser, page, _context, _cdp = _provider_session()
    page.goto = AsyncMock(side_effect=TimeoutError("page never loaded"))

    with pytest.raises(TimeoutError):
        await open_interact_session("https://example.com", provider=provider, url_guard=lambda _url: True)


async def test_url_guard_survives_engine_without_cdp():
    provider, _browser, page, context, _cdp = _provider_session()
    context.new_cdp_session = AsyncMock(side_effect=AttributeError("no CDP on this engine"))

    await open_interact_session("https://example.com", provider=provider, url_guard=lambda _url: True)

    context.route.assert_awaited_once()
    page.goto.assert_awaited_once_with("https://example.com", timeout=30000)


async def test_url_guard_survives_page_without_cdp_session():
    """A page whose context exposes no CDP at all keeps route interception only."""
    provider, _browser, page, _context, _cdp = _provider_session()
    context = MagicMock(spec=["route"])
    context.route = AsyncMock()
    page.context = context

    await open_interact_session("https://example.com", provider=provider, url_guard=lambda _url: True)

    context.route.assert_awaited_once()
    page.goto.assert_awaited_once_with("https://example.com", timeout=30000)


async def test_url_guard_route_handler_tolerates_odd_request_object():
    provider, _browser, _page_obj, context, _cdp = _provider_session()

    await open_interact_session("https://example.com", provider=provider, url_guard=lambda url: "127.0.0.1" not in url)
    handler = _route_handler(context)

    raising = _mock_route("http://127.0.0.1:8080/secret")
    raising.request.is_navigation_request = MagicMock(side_effect=RuntimeError("not a navigation request"))
    missing = _mock_route("http://127.0.0.1:8080/other")
    missing.request.is_navigation_request = None

    for weird in (raising, missing):
        await handler(weird)
        weird.abort.assert_awaited_once_with("blockedbyclient")


async def test_url_guard_direct_branch_installs_guard():
    page = _page()
    context = MagicMock()
    context.route = AsyncMock()
    cdp = MagicMock()
    cdp.on = MagicMock()
    cdp.send = AsyncMock()
    context.new_cdp_session = AsyncMock(return_value=cdp)
    context.new_page = AsyncMock(return_value=page)
    page.context = context

    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()
    pw = MagicMock()
    pw.chromium.launch = AsyncMock(return_value=browser)
    pw.stop = AsyncMock()

    async_playwright = MagicMock(return_value=MagicMock(start=AsyncMock(return_value=pw)))

    with patch("hull_web.browsers.interact._import_async_playwright", return_value=async_playwright):
        handles = await open_interact_session("https://example.com", url_guard=lambda _url: True)

    assert handles[0] is pw
    context.route.assert_awaited_once()
    cdp.send.assert_awaited_once()
    page.goto.assert_awaited_once_with("https://example.com", timeout=30000)


async def test_url_guard_direct_branch_closes_browser_on_rejection():
    page = _page()
    context = MagicMock()
    context.route = AsyncMock()
    cdp = MagicMock()
    cdp.on = MagicMock()
    cdp.send = AsyncMock()
    context.new_cdp_session = AsyncMock(return_value=cdp)
    context.new_page = AsyncMock(return_value=page)
    page.context = context

    async def goto_blocks_redirect(_url: str, timeout: int) -> None:
        await _cdp_handler(cdp)(_paused("http://127.0.0.1:8080/secret"))
        raise RuntimeError("Page.goto: net::ERR_BLOCKED_BY_CLIENT")

    page.goto = AsyncMock(side_effect=goto_blocks_redirect)

    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()
    pw = MagicMock()
    pw.chromium.launch = AsyncMock(return_value=browser)
    pw.stop = AsyncMock()
    async_playwright = MagicMock(return_value=MagicMock(start=AsyncMock(return_value=pw)))

    with (
        patch("hull_web.browsers.interact._import_async_playwright", return_value=async_playwright),
        pytest.raises(UrlGuardBlockedError),
    ):
        await open_interact_session(
            "https://example.com/redirect",
            url_guard=lambda url: "127.0.0.1" not in url,
        )

    browser.close.assert_awaited_once()
    pw.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# Real Chromium + local servers: the attacks themselves.
# ---------------------------------------------------------------------------


def _chromium_executable() -> str | None:
    """Path to patchright's Chromium, or None when it is not installed."""
    try:
        from patchright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable = playwright.chromium.executable_path
    except Exception:
        return None
    return executable if executable and Path(executable).exists() else None


requires_chromium = pytest.mark.skipif(
    _chromium_executable() is None,
    reason="patchright's Chromium is not installed (bare CI runner); the mocked suite covers the contract",
)


class _Site:
    """A threaded local HTTP server whose routes and hits the test can read."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
        self.hits: list[str] = []

    def route(self, path: str, status: int = 200, headers: dict[str, str] | None = None, body: bytes = b"") -> None:
        self.routes[path] = (status, headers or {}, body)

    @property
    def port(self) -> int:
        return self._port

    @property
    def origin(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def loopback(self) -> str:
        """Same server addressed by literal loopback, which is what a guard rejects."""
        return f"http://127.0.0.1:{self._port}"

    def start(self) -> None:
        site = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                site.hits.append(self.path)
                status, headers, body = site.routes.get(self.path, (404, {}, b"not found"))
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def sites() -> Any:
    """The reachable site and the private one its pages are tricked into calling."""
    public = _Site()
    private = _Site()
    public.start()
    private.start()
    private.route("/secret", body=b"SECRET")
    private.route("/pixel", body=b"pixel")
    try:
        yield public, private
    finally:
        public.stop()
        private.stop()


async def _close(handles: tuple[Any, Any, Any, Any]) -> None:
    playwright, browser, _page, _ops = handles
    with contextlib.suppress(Exception):
        await browser.close()
    with contextlib.suppress(Exception):
        await playwright.stop()


@requires_chromium
async def test_redirect_to_loopback_is_aborted(sites: Any):
    public, private = sites
    blocked = f"{private.loopback}/secret"
    public.route("/redirect", 302, {"Location": blocked})
    guard = MagicMock(side_effect=lambda url: not url.startswith(private.loopback))

    with pytest.raises(UrlGuardBlockedError) as caught:
        await open_interact_session(f"{public.origin}/redirect", timeout_ms=20000, url_guard=guard)

    assert caught.value.url == blocked
    assert public.hits == ["/redirect"]
    assert private.hits == []


@requires_chromium
async def test_private_subresource_is_aborted_document_still_loads(sites: Any):
    public, private = sites
    pixel = f"{private.loopback}/pixel"
    public.route("/doc", body=f"<html><body><h1>loaded</h1><img src='{pixel}'></body></html>".encode())
    guard = MagicMock(side_effect=lambda url: not url.startswith(private.loopback))

    handles = await open_interact_session(f"{public.origin}/doc", timeout_ms=20000, url_guard=guard)
    try:
        _playwright, _browser, page, _ops = handles
        assert await page.inner_text("h1") == "loaded"
    finally:
        await _close(handles)

    assert "/doc" in public.hits
    assert private.hits == []


@requires_chromium
async def test_without_guard_the_redirect_is_followed(sites: Any):
    public, private = sites
    blocked = f"{private.loopback}/secret"
    public.route("/redirect", 302, {"Location": blocked})

    handles = await open_interact_session(f"{public.origin}/redirect", timeout_ms=20000)
    try:
        _playwright, _browser, page, _ops = handles
        assert page.url == blocked
    finally:
        await _close(handles)

    assert private.hits == ["/secret"]


@requires_chromium
async def test_provider_branch_aborts_redirect_to_loopback(sites: Any):
    """The path wet actually uses: a BrowserProvider owns the browser."""
    from hull_web.browsers.patchright import PatchrightProvider

    public, private = sites
    blocked = f"{private.loopback}/secret"
    public.route("/redirect", 302, {"Location": blocked})
    guard = MagicMock(side_effect=lambda url: not url.startswith(private.loopback))
    provider = PatchrightProvider(headless=True)
    try:
        with pytest.raises(UrlGuardBlockedError) as caught:
            await open_interact_session(
                f"{public.origin}/redirect", timeout_ms=20000, provider=provider, url_guard=guard
            )
    finally:
        await provider.close()

    assert caught.value.url == blocked
    assert private.hits == []


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
