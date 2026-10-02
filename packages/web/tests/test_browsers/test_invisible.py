"""Tests for InvisibleProvider, InvisibleStrategy and the CF render client."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hull_web.browsers.cf_render import CFBrowserRenderingClient, CFBrowserRenderingError
from hull_web.browsers.invisible import InvisibleProvider
from hull_web.scraper.strategies.invisible_browser import InvisibleStrategy
from hull_web.scraper.strategies.patchright_browser import PatchrightStrategy


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return None


class TestInvisibleProvider:
    async def test_launch_uses_invisible_playwright(self, monkeypatch: pytest.MonkeyPatch):
        constructed: dict[str, object] = {}
        session = _FakeSession()

        class FakeInvisiblePlaywright:
            def __init__(self, **kwargs):
                constructed.update(kwargs)

            async def __aenter__(self):
                return session

            async def __aexit__(self, *exc):
                return None

        mod = types.ModuleType("invisible_playwright")
        async_api = types.ModuleType("invisible_playwright.async_api")
        async_api.InvisiblePlaywright = FakeInvisiblePlaywright  # ty: ignore[unresolved-attribute]
        mod.async_api = async_api  # ty: ignore[unresolved-attribute]
        monkeypatch.setitem(sys.modules, "invisible_playwright", mod)
        monkeypatch.setitem(sys.modules, "invisible_playwright.async_api", async_api)

        provider = InvisibleProvider(seed=5, binary_path="/opt/stealthfox", headless=True)
        launched = await provider.launch(config={"timezone": "UTC"})

        assert launched is session
        assert constructed["seed"] == 5
        assert constructed["binary_path"] == "/opt/stealthfox"
        assert constructed["headless"] is True
        assert provider.name == "invisible"
        assert provider.supports_arm64 is True

    async def test_close_exits_session(self, monkeypatch: pytest.MonkeyPatch):
        provider = InvisibleProvider()
        session = MagicMock()
        session.__aexit__ = AsyncMock(return_value=None)
        provider._session = session

        await provider.close()

        session.__aexit__.assert_awaited_once()
        assert provider._session is None


class TestInvisibleStrategy:
    def test_subclasses_patchright_and_overrides_provider(self):
        assert issubclass(InvisibleStrategy, PatchrightStrategy)
        strategy = InvisibleStrategy()
        provider, owns = strategy._resolve_provider()
        assert owns is True
        assert provider.name == "invisible"
        assert strategy.name == "invisible"

    def test_injected_provider_wins(self):
        injected = MagicMock()
        injected.name = "invisible"
        strategy = InvisibleStrategy(provider=injected)
        provider, owns = strategy._resolve_provider()
        assert provider is injected
        assert owns is False


class TestCFBrowserRenderingClient:
    def test_requires_credentials(self):
        with pytest.raises(ValueError):
            CFBrowserRenderingClient("", "token")
        with pytest.raises(ValueError):
            CFBrowserRenderingClient("acct", "")

    @patch("hull_web.browsers.cf_render.is_safe_url", return_value=True)
    async def test_render_success(self, _mock_safe):
        http = MagicMock()
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"success": True, "result": "<html>rendered</html>"}
        http.post = AsyncMock(return_value=response)

        client = CFBrowserRenderingClient("acct", "tok", http_client=http)
        html = await client.render("https://spa.example.com")

        assert html == "<html>rendered</html>"
        endpoint, kwargs = http.post.call_args
        assert str(endpoint[0]).endswith("accounts/acct/browser-rendering/content")
        assert kwargs["json"]["url"] == "https://spa.example.com"
        assert kwargs["headers"]["Authorization"] == "Bearer tok"

    @patch("hull_web.browsers.cf_render.is_safe_url", return_value=True)
    async def test_render_api_error_raises(self, _mock_safe):
        http = MagicMock()
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"success": False, "errors": [{"code": 1, "message": "bad"}]}
        http.post = AsyncMock(return_value=response)

        client = CFBrowserRenderingClient("acct", "tok", http_client=http)
        with pytest.raises(CFBrowserRenderingError):
            await client.render("https://spa.example.com")

    async def test_ssrf_blocked(self):
        client = CFBrowserRenderingClient("acct", "tok")
        with pytest.raises(ValueError, match="SSRF"):
            await client.render("http://169.254.169.254/")
