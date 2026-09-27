"""hull-web: shared web infrastructure for search, scraping, HTTP security, and browsers."""

from hull_web.adapters import ChapterImages, ChapterInfo, MangaDexClient, MangaInfo
from hull_web.browsers import (
    BrowserlessClient,
    BrowserProvider,
    PatchrightProvider,
)
from hull_web.http import is_safe_url, is_valid_domain, normalize_url, safe_httpx_client, strip_tracking_params
from hull_web.scraper import ScrapingAgent, StrategyCache
from hull_web.scraper.strategies import RemoteRenderStrategy, RenderClient
from hull_web.search import SearchResult, ensure_searxng, shutdown_searxng

__all__ = [
    "BrowserProvider",
    "BrowserlessClient",
    "ChapterImages",
    "ChapterInfo",
    "MangaDexClient",
    "MangaInfo",
    "PatchrightProvider",
    "RemoteRenderStrategy",
    "RenderClient",
    "ScrapingAgent",
    "SearchResult",
    "StrategyCache",
    "ensure_searxng",
    "is_safe_url",
    "is_valid_domain",
    "normalize_url",
    "safe_httpx_client",
    "shutdown_searxng",
    "strip_tracking_params",
]
