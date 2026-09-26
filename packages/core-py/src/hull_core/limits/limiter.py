"""Per-user request limiter (spec §4 Q4 — lives in the shared core)."""

from __future__ import annotations

import time
from collections import defaultdict, deque

_WINDOW_SECONDS = 60.0


class SlidingWindowLimiter:
    """In-process sliding-window RPM limiter keyed by principal id.

    Deliberately in-memory: limits guard a single server process. Hosts that
    shard one namespace across processes terminate the limiter at their load
    balancer instead.
    """

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, rpm: int | None, *, now: float | None = None) -> bool:
        """Record a request for ``key`` and report whether it is allowed.

        ``rpm=None`` means unlimited. The request is recorded either way so
        bursts cannot pre-load the window after a limit change.
        """
        current = time.monotonic() if now is None else now
        events = self._events[key]
        cutoff = current - _WINDOW_SECONDS
        while events and events[0] <= cutoff:
            events.popleft()
        events.append(current)
        return rpm is None or len(events) <= rpm
