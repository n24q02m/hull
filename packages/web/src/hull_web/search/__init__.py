"""SearXNG search client, models, and runner."""

from hull_web.search.client import search
from hull_web.search.models import SearchError, SearchResult
from hull_web.search.runner import ensure_searxng, shutdown_searxng

__all__ = ["SearchError", "SearchResult", "ensure_searxng", "search", "shutdown_searxng"]
