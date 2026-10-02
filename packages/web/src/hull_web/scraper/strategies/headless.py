"""Headless strategy: Crawl4AI wrapper for JS-rendered pages with stealth mode."""

from __future__ import annotations

import logging
from typing import Any

from hull_web.fingerprint import IdentityProfile
from hull_web.http.client import is_safe_url
from hull_web.scraper.base import BaseStrategy, ScrapingResult

logger = logging.getLogger(__name__)


class HeadlessStrategy(BaseStrategy):
    """Use Crawl4AI headless browser with a stable optional identity."""

    name: str = "headless"

    def __init__(
        self,
        timeout: float = 60.0,
        wait_for: str | None = None,
        stealth: bool = True,
        proxy: str | None = None,
        crawler_factory: Any = None,
        identity: IdentityProfile | None = None,
    ):
        self.timeout = timeout
        self.wait_for = wait_for
        self.stealth = stealth
        self.proxy = proxy
        self.identity = identity
        self._crawler_factory = crawler_factory
        self._locale_warned = False

    def _build_browser_config(self) -> Any:
        """Build a Crawl4AI BrowserConfig with identity and proxy settings."""
        from crawl4ai import BrowserConfig

        config: dict[str, Any] = {
            "headless": True,
            "browser_type": "chromium",
            "enable_stealth": self.stealth,
            "verbose": False,
        }
        if self.identity is not None:
            config.update(
                user_agent=self.identity.user_agent,
                viewport_width=self.identity.viewport_width,
                viewport_height=self.identity.viewport_height,
            )
            # Crawl4AI's BrowserConfig has no locale/timezone fields; the UA is
            # Firefox but the clock/locale are the host's. Log once so the
            # divergence is visible instead of silent.
            if not self._locale_warned:
                self._locale_warned = True
                logger.info(
                    "headless strategy: Crawl4AI has no locale/timezone config; "
                    "identity %s sends Firefox UA but host clock/locale",
                    self.identity.seed,
                )
        browser_config = BrowserConfig(**config)
        if self.proxy is not None:
            browser_config.proxy_config = {"server": self.proxy}
        return browser_config

    def _build_crawler_run_config(self) -> Any:
        """Build a Crawl4AI CrawlerRunConfig with wait and delay settings."""
        from crawl4ai import CrawlerRunConfig

        return CrawlerRunConfig(
            wait_for=self.wait_for or "css:body",
            delay_before_return_html=2.0,
            page_timeout=int(self.timeout * 1000),
            verbose=False,
        )

    async def fetch(self, url: str, selectors: dict[str, str] | None = None) -> ScrapingResult:
        """Fetch *url* via Crawl4AI headless browser rendering."""
        if not is_safe_url(url):
            raise ValueError(f"SSRF blocked: {url}")
        crawler_run_config = self._build_crawler_run_config()
        if self._crawler_factory is not None:
            crawler = self._crawler_factory()
            result = await crawler.arun(url=url, config=crawler_run_config)
        else:
            from crawl4ai import AsyncWebCrawler

            async with AsyncWebCrawler(config=self._build_browser_config()) as crawler:
                result = await crawler.arun(url=url, config=crawler_run_config)
        content = getattr(result, "markdown", "") or getattr(result, "html", "") or ""
        status = getattr(result, "status_code", 200)
        return ScrapingResult(
            content=content,
            url=url,
            strategy=self.name,
            status_code=status,
            metadata={
                "rendered": True,
                "content_length": len(content),
                "wait_for": self.wait_for,
                "stealth": self.stealth,
                "proxy": self.proxy is not None,
            },
        )
