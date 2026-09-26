# hull — shared core for the wet / crg / mnemo self-hosted stack

`hull` is the shared monorepo core for three self-hostable products:
**wet** (search), **crg** (code graph), **mnemo** (memory). It owns everything
the three products share: token auth, per-task model configuration, per-user
rate limiting, local SQLite storage, the HTTP MCP server lifecycle, web
scraping/search infrastructure, and a local embedding daemon.

## Packages

| Package | PyPI dist | Purpose |
|---|---|---|
| `packages/core-py` | `hull-core` | Auth (3 modes), per-task model cells, limiter, `~/.hull/` SQLite WAL storage, `server start` + `/mcp` endpoint, CLI, SSRF-safe HTTP, lifecycle lock |
| `packages/embedding-daemon` | `hull-embedding-daemon` | Local ONNX/GGUF embedding server (FastAPI) |
| `packages/web` | `hull-web` | Search (SearXNG), scraping strategies, stealth browsers, fingerprinting, HTTP/SSRF security |

## Runtime model

- Python 3.13. Dev with `uv`, always-on with docker.
- `hull server start` is the only way to run a server; MCP is an HTTP endpoint
  at `http://host:port/mcp` — never a spawned subprocess.
- The CLI is the control plane + consumer. With the server down the CLI tells
  you to start it first; there is no hidden local-core mode.
- Provider calls are plain OpenAI-spec HTTP (`httpx`). OpenRouter is the
  pre-wired default. Each task (`embed`, `rerank`, `chat`, `jev_score`) has its
  own `base_url + api_key + model` cell — change provider by editing config,
  never code.
- Storage is local SQLite (WAL) under `~/.hull/`. Backup/sync is `rclone`
  outside this repo.

## Auth — one mechanism, three modes

Mode is a config state (`[server] auth` in `~/.hull/config.toml`):

1. **`no-auth`** — localhost-only bind enforced, one shared namespace.
2. **`token`** — one shared token (`token_hash` in config), still one shared
   namespace.
3. **`multi`** — `users.toml` maps `uid → {token_hash, enabled, namespace,
   allowed_roots, limits}`; one process serves N users, each isolated to their
   namespace. Admin is host-side only: edit `users.toml`, restart.

Token hashes are scrypt (`hull token hash`). Wrong token → `401`; disabled
user → `403`; over-limit → `429`.

## Quick start

```bash
uv sync
uv run hull config init          # write ~/.hull/config.toml (OpenRouter defaults)
uv run hull token hash mysecret  # mint a token_hash for users.toml / config
uv run hull server start         # serves http://127.0.0.1:8000/mcp
```

## Development

```bash
uv sync
uv run pytest packages/core-py/tests -q
uv run pytest packages/embedding-daemon/tests -q
uv run pytest packages/web/tests -q
```

## License

Apache-2.0
