# hull_core

Shared core for the wet / crg / mnemo self-hosted stack. Base install of the
`hull-core` dist (`pip install hull-core`).

- **Auth** — one mechanism, three modes (`no-auth` / `token` / `multi`),
  `users.toml` (`uid → token_hash + enabled + namespace + allowed_roots +
  limits`), scrypt token hashes, pure-ASGI auth middleware: `401` wrong token,
  `403` disabled user, `429` over limit.
- **Model config** — per-task cells (`embed` / `rerank` / `chat` / `jev_score`),
  each `base_url + api_key + model`, OpenAI-spec HTTP via `httpx`, OpenRouter
  pre-wired as default. Keys are host-only (`HULL_<TASK>_API_KEY` env or
  config).
- **Limiter** — per-user sliding-window RPM.
- **Storage** — `~/.hull/hull.db`, SQLite WAL, namespace-scoped.
- **Server** — `hull server start` serves `http://host:port/mcp` (Streamable
  HTTP); lifecycle lock prevents double binds; `no-auth` refuses non-loopback
  binds.
- **HTTP** — SSRF-safe transports (DNS pinning, mode-aware policy).

```bash
hull config init && hull token hash <secret> && hull server start
```
