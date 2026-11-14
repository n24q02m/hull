# hull Handover

Operational handover for [hull](https://github.com/n24q02m/hull) — the shared
monorepo core for the wet / crg / mnemo self-hosted stack. Current stable
line: **n24q02m-hull 0.4.0** (PyPI) / **@n24q02m/hull 0.4.0** (npm).
Next, unreleased: **0.5.0** — add a provider key pool + multi-account registry
(`hull_core.credentials`, `@n24q02m/hull/storage`) and an optional per-request
URL guard on the interact session opener (`hull_web.browsers.open_interact_session`
`url_guard=`), both additive to the 0.4.0 surface.

## Current operation

- Packages:
  - Python: `n24q02m-hull` (PyPI). The wheel carries two import packages; heavy stacks are opt-in extras:
    - `hull_core` (base) — token auth, per-task model cells, **provider key
      pools + named accounts** (`hull_core.credentials`), rate limiter,
      `~/.hull/` SQLite WAL storage, `server start` + `/mcp` endpoint, CLI,
      SSRF-safe HTTP, lifecycle lock, `model_selection` picker.
    - `hull_web` (`[web]`) — search (SearXNG), scraping strategies, stealth
      browsers, fingerprinting, HTTP/SSRF security (consumed by wet).
  - The embed/rerank HTTP service is **not** in this repo: it is
    `fastretrieval[server]` (`fastretrieval-server`, `python -m
    fastretrieval.server`), the dist that owns the model runtime. hull reaches
    it over HTTP like any other OpenAI-spec provider.
  - TypeScript: `@n24q02m/hull` (`packages/core-ts` on npm) — token-table auth (`token`, `multi` with `users.toml`), Streamable HTTP `/mcp` transport, `buildCli` console builder, encrypted `PerPluginStore` storage (`~/.<plugin>/`), the `@n24q02m/hull/storage` key-pool/account mirror over `~/.hull/hull.db`, lockfiles and cache. All relay, OAuth AS/delegated, and CfKvBackend residue is removed from the source tree; the published npm 0.4.0 tarball still carries it (see "Unreleased" below).
- Runtime: Python 3.11 to 3.15 (CI tests each supported minor); dev with `uv`, always-on with docker.
- Surfaces: `hull` CLI (control plane + consumer: `config`, `db`, `token`,
  `status`, `doctor`, `--version`), `hull server start`
  (the only way to run a server — MCP is an HTTP endpoint at
  `http://host:port/mcp`, never a spawned subprocess).
- hull is a library/daemon product, not a packaged MCP server: no registry
  entry. Downstream products (wet, crg, mnemo) consume the library.

## Install

```bash
pip install n24q02m-hull               # hull_core only (crg, mnemo dependency)
pip install "n24q02m-hull[web]"        # + hull_web scraping/browser stack (wet)
pip install "n24q02m-hull[mteb]"       # + pyarrow for model_selection MTEB fetchers
```

`import hull_web` without the `[web]` extra raises `ImportError` naming the
missing modules and the install command.

## Quick start

```bash
uv sync
uv run hull config init               # write ~/.hull/config.toml (OpenRouter defaults)
printf '%s' "$TOKEN" | uv run hull token hash  # mint a token_hash for users.toml / config
uv run hull status                    # auth mode, bind, DB path
uv run hull doctor                    # python / config / users.toml checks
uv run hull server start              # serves http://127.0.0.1:8000/mcp
```

`hull token hash` reads the token from piped stdin or a hidden prompt; a token
on the command line is refused, because it would land in shell history.

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

## Provider credentials — key pools and accounts

Until 0.5.0 a task cell resolved exactly one `api_key`, so one dead key took
the feature down. The cell shape is unchanged; a **pool of ordered
credentials sits under the provider identity** and a cell may select a named
account:

```toml
# ~/.hull/config.toml — unchanged cell shape, optional selectors
[models.embed]
base_url = "https://openrouter.ai/api/v1"
api_key  = "sk-or-v1-..."   # first-run seed only
model    = "openai/text-embedding-3-small"
provider = "openrouter"     # optional, inferred from base_url
account  = "openrouter:work" # optional; default pool when absent
```

- Python API: `hull_core.credentials.CredentialStore` (CRUD + `health` /
  `diagnose` read API), `KeyPool` (rotation), `AccountRegistry` (named
  accounts). TypeScript mirror: `@n24q02m/hull/storage` with the same models,
  policy and table layout.
- Persistence is additive in the existing `~/.hull/hull.db`:
  `provider_accounts` and `provider_credentials` (`CREATE TABLE IF NOT
  EXISTS`; legacy tables untouched). Keys are AES-256-GCM sealed with
  `~/.hull/.credential-key` (32 bytes, `0600`, created once); rows store the
  envelope plus a 4-char hint, never the key.
- `CredentialStore.migrate_model_cells(...)` seeds the pool from the current
  single-key cells (idempotent; one entry per distinct `base_url + api_key`,
  existing order preserved).
- Concurrency: an acquired credential is **leased** (default 60 s) so two
  tasks cannot receive the same key; the lease is released on success,
  failure, or any re-raised error.

Rotation matrix (both languages; `classify_status` in Python /
`classifyStatus` in TS):

| Upstream result | Action |
|---|---|
| `401`, `403` | disable the credential (`disabled=1`), take the next one |
| `429` | cooldown `Retry-After` (seconds or HTTP date), else `rate_limit_reset` (seconds or epoch), clamped to 1 s…1 h, default 60 s; take the next one |
| `5xx` | retry the same credential up to `max5xxRetries` (default 1), then move to the next |
| any other status / no status | re-raise the original error unchanged; credential state is not touched |
| pool empty or all blocked | `PoolExhaustedError` with `total` / `disabled` / `coolingDown` counts and a **redacted** last error (keys never appear in messages, `toString`, `toJSON`, health snapshots, or logs) |

Bounded attempts: `max(1, total × (1 + max5xxRetries))`, so an exhausted pool
fails fast instead of retrying forever. `health()` returns the read-only view
(`lastStatus`, `lastError` redacted, `failureCount`, `cooldownUntil`,
`disabled`, `usable`, `keyHint`) that products consume.

## Embed/rerank service (owned elsewhere)

The shared embedding/rerank HTTP service used to ship here as
`hull_embedding_daemon` (`n24q02m-hull[embedding]`). It now belongs to the dist
that owns the model runtime:

- Install/run: `pip install "fastretrieval[server]"`, then
  `fastretrieval-server` (or `python -m fastretrieval.server`). Default bind
  `127.0.0.1:9800`, same route contract — `GET /health`, `GET /models`,
  `POST /embed` (`{input, model?, dims?}`, `dims` 32..1024), `POST /rerank`
  (`{query, documents, model?, top_n?}`).
- Cache dir: `FASTRETRIEVAL_CACHE_PATH` (was `HULL_EMBEDDING_CACHE_DIR`);
  threads: `FASTRETRIEVAL_THREADS`. GGUF names need `fastretrieval[gguf]`, else
  requests fail `503` with the install hint (ONNX unaffected).
- hull holds no model runtime: callers reach the service over HTTP. A
  reintroduced `import` of `fastretrieval`/`onnxruntime`/`llama_cpp` in
  `packages/*/src` fails `tests/test_embedding_boundary.py`.

## Build, run, and verify

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check .
uv build
```

Use isolated fixtures for server/auth tests; the CLI tells you to start the
server first when it is down — there is no hidden local-core mode.

The embed/rerank service keeps its real-model smoke in its owning repo
(`fastretrieval`, job `server-smoke`); hull's CI no longer downloads models.

## Model configuration policy

- Each task (`embed`, `rerank`, `chat`, `jev_score`) has its own
  `base_url + api_key + model` cell; provider calls are plain OpenAI-spec HTTP
  (`httpx`); OpenRouter is the pre-wired transport default. Change provider by
  editing config, never code.
- **No sanctioned default cloud model.** An unconfigured task cell means the
  feature is off (or the caller's local fallback applies); a model must be
  set explicitly per deployment/task. Keys come from the config file or
  `HULL_<TASK>_API_KEY` env vars; keys never select a model.

## Model candidates snapshot

`model_selection` ranks OpenRouter-servable models per task profile (agentic,
classification, decision, embedding, rerank, translation, …) from public
boards; it is scan-only (no evals, no paid calls) and never writes a model
into any config.

- Workflow `Model Candidates Publish` (`model-candidates.yml`): Mondays 04:41
  UTC or `workflow_dispatch`. It force-pushes `data/model-candidates.json` to
  the `data/model-candidates` branch (the read surface for consumers, not
  `main`) and updates the single sticky issue
  `[model-candidates] weekly leaders (all profiles)`.
- `AA_API_KEY` (free tier) is the only secret; its absence only drops the AA
  source.
- A profile reaches `weak_evidence=false` only with at least two independent
  boards. The `decision` profile counts `jevals`, `jevbench` (decision boards)
  plus `judgemark_v4` (corroborator) and the `tau2_bench_or` / `bfcl` /
  `arena_agent` agentic proxies; `jev_decision_index` (no declared license)
  and `aa_agentic_index` (not provably independent of tau2) stay registered
  but out of the profile. Decisions-segment catalog rows are visible to the
  `decision` profile only.
- Verify after a change: dispatch the workflow, then read
  `tasks.<profile>.candidates` and `source_status.<board>.status` from the
  branch file.

## Data and safety invariants

- Storage is local SQLite (WAL) under `~/.hull/`; backup/sync is `rclone`
  outside this repo.
- Auth failures are explicit: `401` / `403` / `429`; `multi` mode isolates
  namespaces per user.
- SSRF-safe HTTP in the base stack; web stack keeps scraping/browser work
  behind the `[web]` extra.
- Interact sessions take an optional per-request guard (0.5.0, additive):
  `hull_web.browsers.open_interact_session(url, url_guard=guard)` checks the
  initial URL before launching, every routed sub-resource, and — on Chromium —
  every redirect hop through the CDP `Fetch` domain, because Playwright's route
  layer never delivers the request a redirect leads to
  (microsoft/playwright#34994). Pass `hull_web.is_safe_url`, which resolves the
  hostname on every call, to cover DNS rebinding as far as a URL check can. A
  blocked navigation raises `UrlGuardBlockedError` naming the URL; `url_guard`
  absent keeps the pre-0.5.0 behaviour (no interception at all).

## Unreleased: `@n24q02m/hull` removals since 0.4.0

The npm 0.4.0 tarball still ships the relay-era surface. The next npm release
(0.4.1 or whichever version the release tooling picks) drops the public API
listed below; consumers must update as described in the "Consumer bump after the
release" item below. Importers that deep-import these paths must stop
first:

- Public exports removed: `getMode`, `setLocalMode`, `clearMode`, `ServerMode`,
  `acquireSessionLock`, `writeSessionLock`, `releaseSessionLock`, `SessionInfo`,
  `setLockDir`, `markSetupComplete`, `deriveJwtSigningSeed`, and every type of the
  `./schema` subpath (the subpath itself is gone too).
- Also absent from the monorepo build (still in the 0.4.0 tarball): the
  `CfKvBackend` storage backend (`storage` exports only `backendFromEnv`,
  `InMemoryBackend`, `LocalFsBackend`), the `auth/`, `oauth/` and `relay/`
  runtime modules, and the OAuth middleware.
- `buildCli` `doctor` no longer prints a `mode:` line.
- Kept on purpose: the fixed key-derivation inputs `mcp-relay-config`,
  `mcp-relay-export` and `mcp-relay` (HKDF info). Changing them would change
  derived keys and break stored `config.enc` files.
- Consumer check (origin/main of notion, email, workspace, telegram, mnemo,
  wet, crg): none imports a removed symbol or `CfKvBackend`. telegram, mnemo,
  wet and crg are Python and do not depend on `@n24q02m/hull`.
- Consumer bump after the release: notion, email and workspace all declare
  `"@n24q02m/hull": "^0.4.0"`, so any 0.4.x is picked up on the next install.
  Run `bun update @n24q02m/hull`, then type-check and test each. No source change
  is needed (a local pack of this tree passed notion 922/922, email 733/733,
  workspace 334/334). email also already dropped the removed `openBrowser`
  option from its `runHttpServer` call.

## In-flight and rollback

Roll back a source change by reverting the owning commit; config changes take
effect on process restart. Daemon model artifacts are cached per user cache
dir — clear `FASTRETRIEVAL_CACHE_PATH` (or the platform user cache) when
switching artifact exports. Keep graph-free: hull holds no product graph
state; downstream products own their DBs.

## First weeks

1. Run the quick start against a scratch `HOME` (or `~/.hull` sandbox) and
   verify `401/403/429` paths with two tokens in `multi` mode.
2. Exercise `fastretrieval-server` `/embed` with `dims` truncation and a GGUF
   request without `fastretrieval[gguf]` (expect `503` + hint).
3. Run the full test suite and `uv build`; read the extras matrix against
   `pyproject.toml` before adding a new extra.
