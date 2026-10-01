"""Base strategy interface and result type for scraping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScrapingResult:
    """Result from a single scraping strategy execution."""

    content: str
    url: str
    strategy: str
    # A strategy that never reached a response (TLS failure, dropped socket)
    # has no status code; the agent treats that as invalid rather than 0.
    status_code: int | None
    metadata: dict[str, object] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for scraping strategies."""

    name: str = "base"

    @abstractmethod
    async def fetch(self, url: str, selectors: dict[str, Any] | None = None) -> ScrapingResult:
        """Fetch content from a URL using this strategy."""
        ...
