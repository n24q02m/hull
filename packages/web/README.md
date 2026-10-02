# hull_web

Ships in the `hull-core` dist as the `[web]` extra:
`pip install "hull-core[web]"`.

Shared web infrastructure for hull consumers: search (SearXNG), scraping
strategies (plain HTTP → TLS-spoof → headless → stealth browser → remote
render), stealth browser providers (Patchright), fingerprinting, and
SSRF-safe HTTP client helpers.

Roles per the hull architecture: scraping / stealth / HTTP-security — this
package intentionally ships no UI or viewer surface.

## Identity A/B measurement (`scripts/measure_identity_ab.py`)

E1-c harness: per host x {identity off, identity on} x {basic_http, tls_spoof,
headless}, records success rate, deepest tier reached, wall-clock p50/p95 and
peak RSS; every attempt flows through `StrategyCache` so the JSON also carries
the cache-learned recommendation order per domain.

```bash
uv sync --extra web --extra identity   # or: pip install "hull-core[web,identity]"
uv run python packages/web/scripts/measure_identity_ab.py --reps 2 --out e1c-results.json
```

Read the `deepest tier` column: if `identity_on` pulls the deepest tier DOWN
on most hosts, the invisible engine tier (extra `[invisible]`) is not buying
anything for those hosts. `fingerprint_stability` in the JSON proves the seed
actually changes the derived person. The fixed `TEST_SEED` is for
reproducibility only — never reuse it in production.
