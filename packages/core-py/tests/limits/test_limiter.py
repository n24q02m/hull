"""Sliding-window limiter semantics."""

from __future__ import annotations

import sys
import threading

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


def test_concurrent_checks_admit_exactly_rpm() -> None:
    """Token checks run on worker threads, so one key is checked from several threads at once.

    Unguarded, a thread can append its event, lose the GIL, and read the length after
    another thread appended too, refusing a request that was within the limit. The race
    only bites at the limit boundary, so the check runs over many fresh boundaries.
    """
    limiter = SlidingWindowLimiter()
    interval = sys.getswitchinterval()
    # Switch threads as often as possible so an unguarded append/len pair interleaves.
    sys.setswitchinterval(1e-6)
    try:
        for round_no in range(40):
            admitted: list[bool] = []
            start = threading.Barrier(8)

            def worker(key: str = f"u{round_no}") -> None:
                start.wait()
                admitted.extend([limiter.check(key, 40, now=1.0) for _ in range(10)])

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len(admitted) == 80
            assert admitted.count(True) == 40, f"round {round_no}: {admitted.count(True)} admitted, limit 40"
    finally:
        sys.setswitchinterval(interval)
