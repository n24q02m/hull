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
- **Skills** — `attach_skills_provider` mounts a skills directory on a FastMCP
  server; a `skills.digest.json` manifest pins the SHA-256 of every skill file,
  and any mismatch mounts nothing. See [Skills integrity](#skills-integrity).

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

## Skills integrity

`hull_core.server.attach_skills_provider(mcp, skills_dir)` mounts a directory
of skills (one folder with a `SKILL.md` per skill) on a FastMCP server. A
`skills.digest.json` manifest at the root of that directory pins the SHA-256 of
every file inside each skill folder:

```bash
hull skills digest ./skills   # write ./skills/skills.digest.json; exit 2 if the tree cannot be hashed
hull skills verify ./skills   # "ok: N files match ..." (exit 0), or one line per problem (exit 1)
```

```python
from fastmcp import FastMCP
from hull_core.server import attach_skills_provider

mcp = FastMCP("my-server")
# Raises SkillIntegrityError and mounts nothing unless every skill file matches.
attach_skills_provider(mcp, "skills", require_digests=True)
```

- The guard is on when the manifest exists, when `digests=` is given (a
  manifest path or a `{path: sha256}` mapping), or when `require_digests=True`.
  With none of these the directory is mounted unchecked, as before. Pass
  `require_digests=True` in production so that deleting the manifest cannot
  switch the check off.
- Strict by default: every file inside a skill folder needs an entry, hidden
  files included. `strict_digests=False` (CLI `--no-strict`) only requires an
  entry for each `SKILL.md`; listed supporting files are still checked.
- Every problem is collected into one `SkillIntegrityError` (a `ValueError`)
  raised before anything is mounted: a digest mismatch, a file with no entry,
  an entry whose file is missing, a missing skills directory, or an invalid
  manifest. `error.problems` lists one `<path>: <reason>` line per file (it is
  empty when the directory or the manifest itself is the problem).
- The bytes the provider loaded are checked as well, so a file swapped between
  the check and the read is never served. With `reload=True` every re-scan is
  verified again, and a re-scan that fails keeps the last verified skills
  serving.
- Symlinks anywhere in the walk are refused while the guard is on. Files
  outside skill folders (the manifest itself included) and hidden folders
  outside skill folders are not part of the check.
- Digests cover exact bytes. Pin line endings for the skills directory in
  `.gitattributes` (for example `skills/** -text`); otherwise a checkout that
  converts line endings fails the check.

The manifest format is shared with `@n24q02m/hull/skills`, so one manifest
verifies in both languages:

```json
{
  "version": 1,
  "files": {
    "hello-skill/SKILL.md": "<64 lowercase hex characters>",
    "hello-skill/references/guide.md": "<64 lowercase hex characters>"
  }
}
```

Keys are relative POSIX paths from the skills directory (no `..` segment, no
leading `/`, no backslash); values are the lowercase hex SHA-256 of each file's
exact bytes. A top-level key other than `version` and `files`, or a `version`
other than `1`, is rejected. `hull skills digest` writes the keys sorted,
indented by two spaces, as UTF-8 with a trailing newline, so regenerating an
unchanged directory produces the same bytes. The same helpers are importable
from `hull_core.server`: `compute_skill_digests`, `write_skill_digests` and
`verify_skill_digests`.
