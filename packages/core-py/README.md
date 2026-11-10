# hull_core

Shared core for the wet / crg / mnemo self-hosted stack. Base install of the
`n24q02m-hull` dist (`pip install n24q02m-hull`).

- **Auth** — one mechanism, three modes (`no-auth` / `token` / `multi`),
  `users.toml` (`uid → token_hash + enabled + namespace + allowed_roots +
  limits`), scrypt token hashes, pure-ASGI auth middleware: `401` wrong token,
  `403` disabled user, `429` over limit.
- **Model config** — per-task cells (`embed` / `rerank` / `chat` / `jev_score`),
  each `base_url + api_key + model`, OpenAI-spec HTTP via `httpx`, OpenRouter
  pre-wired as default. Keys are host-only (`HULL_<TASK>_API_KEY` env or
  config). Optional `provider` / `account` keys point a cell at a pool entry
  below without changing the cell shape.
- **Provider key pools** — ordered credentials per provider (and per named
  account) with health, cooldown and lease state; 401/403 disables, 429
  cools down, 5xx retries then rotates. See below.
- **Limiter** — per-user sliding-window RPM.
- **Storage** — `~/.hull/hull.db`, SQLite WAL, namespace-scoped.
- **Server** — `hull server start` serves `http://host:port/mcp` (Streamable
  HTTP); lifecycle lock prevents double binds; `no-auth` refuses non-loopback
  binds.
- **HTTP** — SSRF-safe transports (DNS pinning, mode-aware policy).

## Provider key pools and named accounts

One `api_key` per task used to be a single point of failure: a revoked or
rate-limited key took the feature down. `hull_core.credentials` keeps an
ordered pool per provider, rotates on failure, and scopes multi-account
providers by account:

```python
from hull_core.config.settings import load_settings
from hull_core.credentials import CredentialStore

store = CredentialStore()  # ~/.hull/hull.db; key sealed with ~/.hull/.credential-key
store.migrate_model_cells(load_settings().models)  # idempotent seed from existing cells

work = store.accounts().add(provider="openrouter", name="Work", email="ops@example.com")
store.add_credential(
    "openrouter",
    "sk-or-v1-...",
    base_url="https://openrouter.ai/api/v1",
    account_id=work.account_id,  # "openrouter:work"
)

pool = store.pool_for_cell(load_settings().models["embed"])
result = pool.run(lambda credential: call_the_provider(credential))
```

| Upstream | Rotation behaviour |
|---|---|
| `401` / `403` | disable that credential, take the next |
| `429` | cooldown `Retry-After` / `rate_limit_reset` (1 s…1 h, default 60 s), take the next |
| `5xx` | retry the same credential once (`max5xx_retries`), then rotate |
| anything else | re-raise unchanged; pool state untouched |
| empty / all blocked | `PoolExhaustedError` with counts and a redacted last error |

Acquired credentials are leased (default 60 s), so two concurrent tasks never
receive the same key. `store.health(provider)` is the read-only view products
use; keys and their values never appear in health, `repr`, or error text — only
a 4-character hint. Tables (`provider_accounts`, `provider_credentials`) are
created additively next to the existing `~/.hull/` schema.

```bash
hull config init && printf '%s' "$TOKEN" | hull token hash && hull server start
```

`hull token hash` reads the token from stdin or a hidden prompt and refuses a
token on argv (shell history). `hull status` and `hull doctor` report the
instance state; `hull --version` prints the installed version.
