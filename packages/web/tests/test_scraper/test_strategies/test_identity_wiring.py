"""Tests for identity wiring on basic_http / tls_spoof / headless strategies."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hull_web.fingerprint import IdentityProfile
from hull_web.scraper.strategies.basic_http import BasicHTTPStrategy
from hull_web.scraper.strategies.headless import HeadlessStrategy
from hull_web.scraper.strategies.tls_spoof import TLSSpoofStrategy


def _identity(impersonate: str = "firefox147") -> IdentityProfile:
    return IdentityProfile(
        seed=99,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
        oscpu="Windows NT 10.0; Win64; x64",
        platform="Win32",
        locale="de-DE",
        timezone_id="Europe/Berlin",
        viewport_width=1600,
        viewport_height=900,
        webgl_vendor="Intel",
        webgl_renderer="Iris Xe",
        impersonate=impersonate,
    )


class TestBasicHTTPIdentity:
    def test_identity_none_keeps_legacy_headers(self):
        strategy = BasicHTTPStrategy()
        assert strategy._headers() == BasicHTTPStrategy.LEGACY_HEADERS

    def test_identity_builds_firefox_headers(self):
        strategy = BasicHTTPStrategy(identity=_identity())
        headers = strategy._headers()
        assert headers["User-Agent"].endswith("Firefox/147.0")
        assert headers["Accept-Language"] == "de-DE,de;q=0.9"
        assert headers["Sec-Fetch-Mode"] == "navigate"

    def test_explicit_headers_win_over_identity(self):
        custom = {"User-Agent": "TestBot"}
        strategy = BasicHTTPStrategy(headers=custom, identity=_identity())
        assert strategy._headers() is custom

    @patch("hull_web.http.client.is_safe_url", return_value=True)
    async def test_fetch_uses_identity_headers(self, _mock_is_safe):
        http = MagicMock()
        response = MagicMock()
        response.text = "<html>ok</html>"
        response.status_code = 200
        response.headers = {"content-type": "text/html"}
        response.url = "https://example.com/"
        http.get = AsyncMock(return_value=response)

        strategy = BasicHTTPStrategy(identity=_identity(), http_client=http)
        result = await strategy.fetch("https://example.com")

        assert result.status_code == 200
        _, kwargs = http.get.call_args
        assert kwargs["headers"]["User-Agent"].endswith("Firefox/147.0")


class TestTLSSpoofIdentity:
    async def test_identity_overrides_impersonate(self):
        captured: dict[str, object] = {}

        class FakeSession:
            async def get(self, url, *, impersonate, **kwargs):
                captured["impersonate"] = impersonate
                resp = MagicMock()
                resp.status_code = 200
                resp.text = "<html>ok</html>"
                resp.headers = {}
                resp.url = url
                return resp

        strategy = TLSSpoofStrategy(identity=_identity("firefox144"), session_factory=FakeSession)
        with patch("hull_web.scraper.strategies.tls_spoof.is_safe_url", return_value=True):
            result = await strategy.fetch("https://example.com")

        assert captured["impersonate"] == "firefox144"
        assert result.metadata["impersonate"] == "firefox144"

    async def test_no_identity_keeps_default_impersonate(self):
        captured: dict[str, object] = {}

        class FakeSession:
            async def get(self, url, *, impersonate, **kwargs):
                captured["impersonate"] = impersonate
                resp = MagicMock()
                resp.status_code = 200
                resp.text = "<html>ok</html>"
                resp.headers = {}
                resp.url = url
                return resp

        strategy = TLSSpoofStrategy(session_factory=FakeSession)
        with patch("hull_web.scraper.strategies.tls_spoof.is_safe_url", return_value=True):
            await strategy.fetch("https://example.com")

        assert captured["impersonate"] == "chrome131"


class TestHeadlessIdentity:
    def _browser_config_mod(self, captured: dict[str, object]):
        mod = types.ModuleType("crawl4ai")

        class BrowserConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class CrawlerRunConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        mod.BrowserConfig = BrowserConfig  # ty: ignore[unresolved-attribute]
        mod.CrawlerRunConfig = CrawlerRunConfig  # ty: ignore[unresolved-attribute]
        return mod

    def test_identity_sets_ua_and_viewport(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}
        monkeypatch.setitem(sys.modules, "crawl4ai", self._browser_config_mod(captured))

        strategy = HeadlessStrategy(identity=_identity())
        strategy._build_browser_config()

        assert str(captured["user_agent"]).endswith("Firefox/147.0")
        assert captured["viewport_width"] == 1600
        assert captured["viewport_height"] == 900

    def test_no_identity_leaves_default(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, object] = {}
        monkeypatch.setitem(sys.modules, "crawl4ai", self._browser_config_mod(captured))

        strategy = HeadlessStrategy()
        strategy._build_browser_config()

        assert "user_agent" not in captured
