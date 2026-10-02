"""Basic HTTP strategy using httpx with SSRF protection."""

from __future__ import annotations

from typing import Any, ClassVar

from hull_web.fingerprint import IdentityProfile
from hull_web.http.client import is_safe_url, safe_httpx_client
from hull_web.scraper.base import BaseStrategy, ScrapingResult


class BasicHTTPStrategy(BaseStrategy):
    """Fetch pages via httpx with browser-like headers and SSRF protection."""

    name: str = "basic_http"
    LEGACY_HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    # Kept as the default so existing constructor callers keep the exact
    # legacy header set; the identity path supersedes it via ``_headers()``.
    DEFAULT_HEADERS: ClassVar[dict[str, str]] = LEGACY_HEADERS

    def __init__(
        self,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        http_client: Any = None,
        proxy: str | None = None,
        identity: IdentityProfile | None = None,
    ):
        self.timeout = timeout
        self.headers = headers or self.DEFAULT_HEADERS.copy()
        self._explicit_headers = headers is not None
        self.proxy = proxy
        self.identity = identity
        self._http_client = http_client

    def _headers(self) -> dict[str, str]:
        """Firefox-coherent header set when an identity is wired.

        An explicit ``headers`` constructor argument still wins: it is the
        caller's contract, not the identity layer's.
        """
        if self.identity is None or self._explicit_headers:
            return self.headers
        primary = self.identity.locale.split("-")[0]
        return {
            "User-Agent": self.identity.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": f"{self.identity.locale},{primary};q=0.9",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "TE": "trailers",
        }

    async def fetch(self, url: str, selectors: dict[str, Any] | None = None) -> ScrapingResult:
        """Fetch *url* via plain HTTP GET with browser-like headers."""
        if not is_safe_url(url):
            raise ValueError(f"SSRF blocked: {url}")
        # selectors is a nested selector tree, so its values are not all strings.
        cookies: dict[str, str] = {}
        raw_cookies = selectors.get("cookies") if selectors else None
        if isinstance(raw_cookies, dict):
            cookies = {str(name): str(value) for name, value in raw_cookies.items()}
        request_kwargs = {
            "headers": self._headers(),
            "timeout": self.timeout,
            "follow_redirects": True,
            "cookies": cookies,
        }
        if self._http_client is not None:
            response = await self._http_client.get(url, **request_kwargs)
        else:
            client_kwargs: dict[str, Any] = {"timeout": self.timeout}
            if self.proxy is not None:
                client_kwargs["proxy"] = self.proxy
            async with safe_httpx_client(**client_kwargs) as client:
                response = await client.get(url, headers=self._headers(), follow_redirects=True, cookies=cookies)
        return ScrapingResult(
            content=response.text,
            url=str(response.url),
            strategy=self.name,
            status_code=response.status_code,
            metadata={
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(response.text),
                "proxy": self.proxy is not None,
            },
        )
