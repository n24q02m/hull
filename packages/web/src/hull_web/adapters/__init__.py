"""External API adapters with SSRF-safe HTTP, rate limiting, and typed models."""

from hull_web.adapters.mangadex import ChapterImages, ChapterInfo, MangaDexClient, MangaInfo

__all__ = [
    "ChapterImages",
    "ChapterInfo",
    "MangaDexClient",
    "MangaInfo",
]
