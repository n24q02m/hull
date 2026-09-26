"""Sliding-window limiter semantics."""

from __future__ import annotations

from hull_core.limits.limiter import SlidingWindowLimiter


def test_unlimited_when_rpm_none() -> None:
    limiter = SlidingWindowLimiter()
    assert all(limiter.check("u", None, now=t) for t in range(1000))


def test_rpm_window() -> None:
    limiter = SlidingWindowLimiter()
    assert limiter.check("u", 3, now=1.0)
    assert limiter.check("u", 3, now=2.0)
    assert limiter.check("u", 3, now=3.0)
    assert not limiter.check("u", 3, now=3.5)
    # events leave the window once they are older than 60s: at t=63 the
    # 1.0 and 2.0 events have expired, leaving room for a fourth
    assert limiter.check("u", 3, now=63.0)


def test_keys_isolated() -> None:
    limiter = SlidingWindowLimiter()
    assert limiter.check("a", 1, now=1.0)
    assert not limiter.check("a", 1, now=1.5)
    assert limiter.check("b", 1, now=1.5)
