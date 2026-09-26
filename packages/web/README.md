# hull-web

Shared web infrastructure for hull consumers: search (SearXNG), scraping
strategies (plain HTTP → TLS-spoof → headless → stealth browser → remote
render), stealth browser providers (Patchright), fingerprinting, and
SSRF-safe HTTP client helpers.

Roles per the hull architecture: scraping / stealth / HTTP-security — this
package intentionally ships no UI or viewer surface.
