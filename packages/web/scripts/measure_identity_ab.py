"""E1-c measurement harness — identity off/on A/B across strategy tiers.

Matrix: host x {identity_off, identity_on} x {basic_http, tls_spoof, headless}
  -> per-cell success rate, deepest tier reached, wall-clock p50/p95, peak RSS.
Every attempt is recorded through ``StrategyCache`` (the same instrumentation
production uses), so the JSON also shows the cache-learned per-domain
recommendation order after each cell.

Key metric (v2 section 6): deepest tier reached. If the identity-on arm pulls
the deepest tier DOWN on most hosts (headless -> tls_spoof), the invisible
engine tier is not buying anything and Tier B can be dropped.

Run after the hull identity merge, in an env with ``n24q02m-hull[web,identity]``:
    python scripts/measure_identity_ab.py --reps 2 --out e1c-results.json
Stdout prints the markdown table; the JSON dump lands in --out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------- fixtures
# Difficult hosts exactly as v2 section 6 (Cloudflare, Medium, LinkedIn, X)
# plus one control host that must always pass at basic_http.
FIXTURES: dict[str, dict[str, Any]] = {
    "control_example": {"url": "https://example.org/", "block_markers": []},
    "cloudflare": {
        "url": "https://www.scrapingcourse.com/cloudflare-challenge/",
        "block_markers": ["just a moment", "cf-chl", "attention required"],
    },
    "medium": {
        "url": "https://medium.com/",
        "block_markers": ["sign in", "access denied"],
    },
    "linkedin": {
        "url": "https://www.linkedin.com/jobs/",
        "block_markers": ["authwall", "sign in", "challenge"],
    },
    "x": {
        "url": "https://x.com/explore",
        "block_markers": ["javascript is not available", "log in"],
    },
}

# Escape-tier order (cheap -> expensive). Deepest tier reached = first success.
TIER_ORDER = ["basic_http", "tls_spoof", "headless"]

TEST_SEED = 20261002  # fixed for reproducibility; not a production seed


@dataclass
class CellResult:
    successes: int = 0
    reps: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    deepest_tier: str = "none"  # tier where first success happened
    recommend_after: list[str] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else -1

    @property
    def p95(self) -> float:
        return sorted(self.latencies_ms)[int(0.95 * len(self.latencies_ms)) - 1] if self.latencies_ms else -1


class RssSampler:
    """Peak-RSS sampler; psutil optional (absent -> rss=-1)."""

    def __init__(self) -> None:
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            self._proc = None
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._proc is not None:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            self._stop.wait(0.5)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


# ------------------------------------------------------------ strategy build
def build_strategies(identity_on: bool) -> dict[str, Any]:
    """Fresh strategy instances per cell; identity per A/B arm."""
    from hull_web.fingerprint import build_identity
    from hull_web.scraper.strategies import (
        BasicHTTPStrategy,
        HeadlessStrategy,
        TLSSpoofStrategy,
    )

    identity = build_identity(seed=TEST_SEED) if identity_on else None
    kw = {"identity": identity} if identity is not None else {}
    return {
        "basic_http": BasicHTTPStrategy(**kw),
        "tls_spoof": TLSSpoofStrategy(**kw),
        "headless": HeadlessStrategy(**kw),
    }


def looks_blocked(text: str, markers: list[str]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)


async def run_cell(
    cache: Any,  # hull_web.scraper.cache.StrategyCache
    fixture: dict[str, Any],
    identity_on: bool,
    reps: int,
) -> CellResult:
    """One matrix cell: reps x cheap->expensive escalation, cache-instrumented."""
    from hull_web.scraper.cache import StrategyCache

    assert isinstance(cache, StrategyCache)
    strategies = build_strategies(identity_on)
    url = str(fixture["url"])
    markers: list[str] = list(fixture["block_markers"])
    cell = CellResult()
    for _ in range(reps):
        for tier in TIER_ORDER:  # cheap -> expensive; first success wins
            strat = strategies[tier]
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(strat.fetch(url), timeout=90.0)
                dt_ms = (time.perf_counter() - t0) * 1000
                text = getattr(result, "text", "") or ""
                status = getattr(result, "status_code", getattr(result, "status", 200))
                blocked = looks_blocked(text, markers) or status in (401, 403, 429)
            except Exception:
                dt_ms = (time.perf_counter() - t0) * 1000
                blocked = True
            cell.reps += 1
            cell.latencies_ms.append(dt_ms)
            await cache.record(url, tier, success=not blocked, time_ms=dt_ms)
            if not blocked:
                cell.successes += 1
                cell.deepest_tier = tier
                break
        else:
            cell.deepest_tier = "ALL_FAIL"
    cell.recommend_after = await cache.recommend(url)
    return cell


# ---------------------------------------------------------------- fp diff
def fingerprint_stability_check() -> dict[str, object]:
    """Proof identity is real: same seed -> same profile; different seed -> differs."""
    from hull_web.fingerprint import IdentityProfile, build_identity

    def sig(p: IdentityProfile) -> tuple[Any, ...]:
        # Seed-sensitive: webgl vendor/renderer (hardware persona) + seed itself.
        # user_agent/locale/timezone/viewport are environment-derived (egress IP,
        # installed engine version) and are intentionally seed-invariant.
        return (p.seed, p.webgl_vendor, p.webgl_renderer, p.user_agent, p.impersonate)

    a1, a2 = build_identity(seed=TEST_SEED), build_identity(seed=TEST_SEED)
    b = build_identity(seed=TEST_SEED + 1)
    return {
        "same_seed_reproducible": sig(a1) == sig(a2),
        "diff_seed_differs": sig(a1) != sig(b),
        "a_user_agent": a1.user_agent,
        "b_user_agent": b.user_agent,
        "a_impersonate": a1.impersonate,
        "b_impersonate": b.impersonate,
    }


# ---------------------------------------------------------------- main
async def main_async(reps: int, out: str) -> None:
    from hull_web.scraper.cache import StrategyCache

    cache = StrategyCache()
    results: dict[str, dict[str, CellResult]] = {}
    with RssSampler() as rss:
        for host, fixture in FIXTURES.items():
            results[host] = {}
            for identity_on in (False, True):
                arm = "identity_on" if identity_on else "identity_off"
                print(f"running {host} x {arm} ...", flush=True)
                results[host][arm] = await run_cell(cache, fixture, identity_on, reps)

    fp = fingerprint_stability_check()
    lines = [
        "| host | arm | success | deepest tier | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for host in FIXTURES:
        for arm in ("identity_off", "identity_on"):
            c = results[host][arm]
            lines.append(f"| {host} | {arm} | {c.successes}/{c.reps} | {c.deepest_tier} | {c.p50:.0f} | {c.p95:.0f} |")
    print("\n".join(lines))
    print(f"\npeak_rss_mb={rss.peak / 1e6:.1f}")
    print(f"fp_stability={json.dumps(fp)}")

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "reps": reps,
        "test_seed": TEST_SEED,
        "cells": {h: {a: vars(c) for a, c in arms.items()} for h, arms in results.items()},
        "peak_rss_bytes": rss.peak,
        "fingerprint_stability": fp,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"written: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="e1c-results.json")
    args = ap.parse_args()
    asyncio.run(main_async(args.reps, args.out))


if __name__ == "__main__":
    main()
