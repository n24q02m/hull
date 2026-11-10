"""Shared helpers for provider key pool tests (imported as ``helpers``)."""

from __future__ import annotations

BASE_NOW = 1_700_000_000.0


class Clock:
    """Injectable wall clock so cooldown/lease windows need no sleeping."""

    def __init__(self, now: float = BASE_NOW) -> None:
        self.value = now

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += seconds
        return self.value


class ProviderFailure(RuntimeError):
    """Stand-in for hull_core.providers.openai_spec.ProviderError."""

    def __init__(
        self,
        status: int | None,
        detail: str = "",
        *,
        retry_after: object = None,
        rate_limit_reset: object = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"provider returned {status}: {detail}")
        self.status = status
        self.detail = detail
        self.retry_after = retry_after
        self.rate_limit_reset = rate_limit_reset
        self.headers = headers
