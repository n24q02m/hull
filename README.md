# hull — shared core for the wet / crg / mnemo self-hosted stack

[![CI](https://github.com/n24q02m/hull/actions/workflows/ci.yml/badge.svg)](https://github.com/n24q02m/hull/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/n24q02m/hull/graph/badge.svg)](https://codecov.io/gh/n24q02m/hull)

[![Python](https://img.shields.io/badge/python-3.11--3.15-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

<!-- BEGIN: AUTO-GENERATED-CROSS-PROMO -->
<details>
  <summary><strong>Sister projects from n24q02m</strong> (click to expand)</summary>

| Project                                                                       | Tagline                                                                          | Tag         |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------- |
| [better-semantic-release](https://github.com/n24q02m/better-semantic-release) | Drop-in python-semantic-release fork with built-in release-safety guards (orp... | Tooling     |
| [claude-plugins](https://github.com/n24q02m/claude-plugins)                   | Claude Code plugin marketplace for the n24q02m MCP servers -- install web sea... | Marketplace |
| [crg](https://github.com/n24q02m/crg)                                         | Knowledge graph for token-efficient code reviews -- semantic search and call-... | MCP         |
| [email](https://github.com/n24q02m/email)                                     | IMAP/SMTP email for AI agents -- read, send, organize folders, and manage att... | MCP         |
| [fastretrieval](https://github.com/n24q02m/fastretrieval)                     | Fast multi-model retrieval runtime: ONNX and GGUF embeddings, reranking, and ... | Tooling     |
| [hull](https://github.com/n24q02m/hull)                                       | hull is the shared monorepo core for the wet, crg and mnemo self-hosted stack    | MCP         |
| [jules-task-archiver](https://github.com/n24q02m/jules-task-archiver)         | Chrome Extension for bulk operations on Jules tasks via batchexecute API -- a... | Tooling     |
| [mnemo](https://github.com/n24q02m/mnemo)                                     | Persistent AI memory with hybrid search. Open, free, unlimited.                  | MCP         |
| [notion](https://github.com/n24q02m/notion)                                   | Markdown-first Notion for AI agents -- pages, databases, blocks, and comments... | MCP         |
| [skret](https://github.com/n24q02m/skret)                                     | Secrets without the server.                                                      | CLI         |
| [tacet](https://github.com/n24q02m/tacet)                                     | A self-distilling neuro-symbolic cascade that amortises LLM cost across knowl... | Tooling     |
| [telegram](https://github.com/n24q02m/telegram)                               | Telegram for AI agents -- messages, chats, media, and contacts across both bo... | MCP         |
| [tools-docs](https://github.com/n24q02m/tools-docs)                           | Docs portal for the n24q02m open-source tool stack (skret, MCP plugins, hull,... | Tooling     |
| [wet](https://github.com/n24q02m/wet)                                         | Open-source MCP server for AI agents: web search, content extraction, and lib... | MCP         |
| [workspace](https://github.com/n24q02m/workspace)                             | Google Workspace MCP server (Docs/Drive/Calendar/Gmail/Sheets/Slides/Tasks/Ch... | MCP         |

</details>
<!-- END: AUTO-GENERATED-CROSS-PROMO -->

**hull is the shared monorepo core for the wet, crg and mnemo self-hosted stack.**

It owns everything the three products share — token auth, per-task model
configuration, per-user rate limiting, local SQLite storage, the HTTP MCP server
lifecycle, and web scraping/search infrastructure —
for **wet** (search), **crg** (code graph) and **mnemo** (memory).

## Table of contents

- [Install](#install)
- [Packages](#packages)
- [Runtime model](#runtime-model)
- [Auth — one mechanism, three modes](#auth--one-mechanism-three-modes)
- [Quick start](#quick-start)
- [Development](#development)
- [License](#license)

## Install

hull ships as one PyPI dist, **`n24q02m-hull`**. The wheel carries both python
import packages; the heavy stacks are opt-in extras:

```bash
pip install n24q02m-hull               # hull_core only (crg, mnemo)
pip install "n24q02m-hull[web]"        # + hull_web scraping/browser stack (wet)
pip install "n24q02m-hull[mteb]"       # + pyarrow for model_selection MTEB fetchers
```

`import hull_web` without the `[web]` extra raises `ImportError` naming the
missing modules and the install command.

hull does not run models. The shared embed/rerank HTTP service is
`pip install "fastretrieval[server]"` (`fastretrieval-server`, serving
`/embed` + `/rerank`); hull callers reach it over HTTP.

The TypeScript core ships to npm as **`@n24q02m/hull`**, at the same version as
the PyPI dist (see [packages/core-ts](packages/core-ts/README.md)):

```bash
npm install @n24q02m/hull
```

## Packages

| Source dir | Import package | Extra | Purpose |
|---|---|---|---|
| `packages/core-py` | `hull_core` | (base; MTEB boards need `[mteb]`) | Auth (3 modes), per-task model cells, limiter, `~/.hull/` SQLite WAL storage, `server start` + `/mcp` endpoint, CLI, SSRF-safe HTTP, lifecycle lock, `model_selection` leaderboard picker + runtime registry (refresh → eval-on-change → promote) |
| `packages/web` | `hull_web` | `[web]` | Search (SearXNG), scraping strategies, stealth browsers, fingerprinting, HTTP/SSRF security |
| `packages/core-ts` | `@n24q02m/hull` (npm) | n/a | TypeScript core: MCP Streamable HTTP transport, token-table auth (`token`, `multi`), `buildCli` console builder, encrypted credential storage, lockfiles and cache |

## Runtime model

- Python 3.11 to 3.15 (every minor upstream still supports; CI tests each). Dev with `uv`, always-on with docker.
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
uv run hull config init               # write ~/.hull/config.toml (OpenRouter defaults)
uv run hull config set server.port 8771   # edit one key in place, comments intact
uv run hull config get server.port        # print one key (secrets are not echoed)
printf '%s' "$TOKEN" | uv run hull token hash  # mint a token_hash for users.toml / config
uv run hull status                    # auth mode, bind, DB path
uv run hull doctor                    # python / config / users.toml checks
uv run hull server start              # serves http://127.0.0.1:8000/mcp
```

`hull token hash` reads the token from piped stdin or a hidden prompt; a token
on the command line is refused, because it would land in shell history.
`hull --version` prints the installed version, and `hull <command> -h` lists
that command's flags.

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check .
uv build
```

## License

Apache-2.0
