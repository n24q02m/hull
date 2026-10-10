# hull Handover

Operational handover for [hull](https://github.com/n24q02m/hull) — the shared
monorepo core for the wet / crg / mnemo self-hosted stack. Current stable
line: **n24q02m-hull 0.3.x**.

## Current operation

- Package: `n24q02m-hull` (one PyPI dist). The wheel carries three import
  packages; heavy stacks are opt-in extras:
  - `hull_core` (base) — token auth, per-task model cells, rate limiter,
    `~/.hull/` SQLite WAL storage, `server start` + `/mcp` endpoint, CLI,
    SSRF-safe HTTP, lifecycle lock, `model_selection` picker.
  - `hull_web` (`[web]`) — search (SearXNG), scraping strategies, stealth
    browsers, fingerprinting, HTTP/SSRF security (consumed by wet).
  - `hull_embedding_daemon` (`[embedding]`) — local ONNX/GGUF embedding
    server (FastAPI).
- Runtime: Python 3.13; dev with `uv`, always-on with docker.
- Surfaces: `hull` CLI (control plane + consumer), `hull server start`
  (the only way to run a server — MCP is an HTTP endpoint at
  `http://host:port/mcp`, never a spawned subprocess), `hull-embedding-daemon`
  (HTTP on `127.0.0.1:9800`).
- hull is a library/daemon product, not a packaged MCP server: no registry
  entry. Downstream products (wet, crg, mnemo) consume the library.

## Install

```bash
pip install n24q02m-hull               # hull_core only (crg, mnemo dependency)
pip install "n24q02m-hull[web]"        # + hull_web scraping/browser stack (wet)
pip install "n24q02m-hull[embedding]"  # + hull_embedding_daemon server stack
pip install "n24q02m-hull[mteb]"       # + pyarrow for model_selection MTEB fetchers
pip install "n24q02m-hull[embedding,embedding-gguf]"  # + llama.cpp runtime for *-GGUF models
```

`import hull_web` without the `[web]` extra raises `ImportError` naming the
missing modules and the install command.

## Quick start

```bash
uv sync
uv run hull config init          # write ~/.hull/config.toml (OpenRouter defaults)
uv run hull token hash mysecret  # mint a token_hash for users.toml / config
uv run hull server start         # serves http://127.0.0.1:8000/mcp
```

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

## Embedding daemon

| Route | Purpose |
|-------|---------|
| `GET /health` | liveness + version |
| `GET /models` | public alias table → canonical model names |
| `POST /embed` | `{input: [str], model?, dims?}` → `{data, model, dims}` |
| `POST /rerank` | `{query, documents, model?, top_n?}` → ranked results desc |

- `/embed` honours MRL truncation: `dims` accepts 32..1024 (default 1024).
- Models run through qwen3-embed over the `n24q02m/Qwen3-*` exports
  (`qwen3-0.6b`, `qwen3-0.6b-q4f16`, `qwen3-0.6b-gguf`, `qwen3-rerank-0.6b*`).
  Artifacts lazy-download from Hugging Face into the user cache on first use
  (`HULL_EMBEDDING_CACHE_DIR`).
- Bind `--host/--port` (default `127.0.0.1:9800`); threads `--threads`.
- Backends: ONNX default; CUDA = swap in `onnxruntime-gpu` (never install both
  wheels side by side). GGUF names need `[embedding,embedding-gguf]`, else
  requests fail `503` with the install hint (ONNX unaffected).

## Build, run, and verify

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check .
uv build
```

Use isolated fixtures for server/auth tests; the CLI tells you to start the
server first when it is down — there is no hidden local-core mode.

## Model configuration policy

- Each task (`embed`, `rerank`, `chat`, `jev_score`) has its own
  `base_url + api_key + model` cell; provider calls are plain OpenAI-spec HTTP
  (`httpx`); OpenRouter is the pre-wired transport default. Change provider by
  editing config, never code.
- **No sanctioned default cloud model.** An unconfigured task cell means the
  feature is off (or the caller's local fallback applies); a model must be
  set explicitly per deployment/task. Keys come from the config file or
  `HULL_<TASK>_API_KEY` env vars; keys never select a model.

## Data and safety invariants

- Storage is local SQLite (WAL) under `~/.hull/`; backup/sync is `rclone`
  outside this repo.
- Auth failures are explicit: `401` / `403` / `429`; `multi` mode isolates
  namespaces per user.
- SSRF-safe HTTP in the base stack; web stack keeps scraping/browser work
  behind the `[web]` extra.

## In-flight and rollback

Roll back a source change by reverting the owning commit; config changes take
effect on process restart. Daemon model artifacts are cached per user cache
dir — clear `HULL_EMBEDDING_CACHE_DIR` (or the platform user cache) when
switching artifact exports. Keep graph-free: hull holds no product graph
state; downstream products own their DBs.

## First weeks

1. Run the quick start against a scratch `HOME` (or `~/.hull` sandbox) and
   verify `401/403/429` paths with two tokens in `multi` mode.
2. Exercise `hull-embedding-daemon` `/embed` with `dims` truncation and a GGUF
   request without the extra (expect `503` + hint).
3. Run the full test suite and `uv build`; read the extras matrix against
   `pyproject.toml` before adding a new extra.
