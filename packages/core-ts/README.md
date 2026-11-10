# @n24q02m/hull

**TypeScript core of hull: MCP Streamable HTTP transport, token-table HTTP auth, encrypted credential storage and a CLI builder for self-hosted MCP servers.**

This package is the TypeScript side of the hull monorepo. It is released in
lockstep with the Python dist `n24q02m-hull`: one hull release tag publishes
the same version to npm and PyPI.

## Install

```bash
npm install @n24q02m/hull
# or
bun add @n24q02m/hull
```

Requirements: Node.js >= 24.21.0. The package is ESM only and ships its own
TypeScript declarations.

## Subpath exports

Import the whole surface from the package root, or a narrower slice from a
subpath:

| Import path | Provides |
|---|---|
| `@n24q02m/hull` | Everything below, plus `buildCli` (console-script subcommand dispatch around a server entry point) and `tryOpenBrowser` |
| `@n24q02m/hull/transport` | `runHttpServer` (local HTTP server with routes), `StreamableHTTPServer` (MCP Streamable HTTP transport), `jsonResponse` |
| `@n24q02m/hull/crypto` | AES `encrypt` / `decrypt`, ECDH `generateKeyPair` / `deriveSharedSecret` / `exportPublicKey` / `importPublicKey`, key derivation `deriveAesKey` |
| `@n24q02m/hull/storage` | `atomicWriteFile`, credential backends (`LocalFsBackend`, `InMemoryBackend`, `backendFromEnv`), encrypted credential read/write/delete and config import/export, `PerPluginStore`, data dir helpers (`dataDir`, `dataDirCandidates`, `legacyDataDirs`), `resolveConfig` (environment variables, then stored config, then defaults), machine identity helpers, and the provider key pool (`CredentialStore`, `KeyPool`, `AccountRegistry`, `SecretCipher`) |
| `@n24q02m/hull/token-auth` | Token-table HTTP auth: `Authenticator`, `hashToken` / `verifyToken`, `loadUsers`, `currentUser`, `ensurePathAllowed`, `guardRequest`, `SlidingWindowLimiter`, `tokenCliHandler` |

## Credential storage layout

`PerPluginStore` keeps one AES-256-GCM encrypted blob per plugin slug (for
example `wet`) under a short, machine-bound data dir:

```text
~/.<plugin>/config.json              single-user credentials
~/.<plugin>/.secret                  single-user machine key (32 bytes, 0600)
~/.<plugin>/subs/<sub>/config.json   HTTP multi-user credentials, one per sub
```

Multi-user blobs use a key derived with scrypt from the `CREDENTIAL_SECRET`
environment variable (salt `<plugin>:<sub>`) instead of `.secret`.
Environment variables take precedence over stored values through
`resolveConfig`, which is how containers override the store.

Credentials written by earlier releases stay readable, so upgrading does not
require signing in again. With the default `LocalFsBackend`, `load()` checks
these dirs in order, and the first one holding a blob decides the result:

1. `~/.<plugin>/` (current)
2. `~/.<plugin>-mcp/`
3. `~/.better-<plugin>-mcp/`

A legacy blob is decrypted with that legacy dir's own key material (its
`.secret`, or the salt `better-<plugin>:<sub>` for the third layout). Legacy
dirs are only read, never written. The next `save()` writes to `~/.<plugin>/`,
and from then on the new copy takes precedence. `clear()` removes the blob
from every location. A corrupt blob is reported as corrupt; the store does
not fall back to an older copy behind it. Injected backends (`InMemoryBackend`,
a `LocalFsBackend` with a custom `baseDir`) are used as the
only source.

## Provider key pools (`~/.hull/hull.db`)

`@n24q02m/hull/storage` mirrors the Python key pool: ordered credentials per
provider (optionally scoped to a named account), encrypted at rest and rotated
on failure. This is separate from `PerPluginStore` above — the pool lives in the
shared hull database, not in `~/.<plugin>/`.

```ts
import { CredentialStore } from '@n24q02m/hull/storage'

const store = new CredentialStore() // ~/.hull/hull.db + ~/.hull/.credential-key
store.migrateModelCells(settings.models) // idempotent seed from existing cells

const work = store.accounts().add({ provider: 'openrouter', name: 'Work', email: 'ops@example.com' })
store.addCredential({
  provider: 'openrouter',
  apiKey: process.env.OPENROUTER_KEY ?? '',
  baseUrl: 'https://openrouter.ai/api/v1',
  accountId: work.accountId // "openrouter:work"
})

const pool = store.poolForCell(settings.models.embed)
const result = await pool.runAsync((credential) => callProvider(credential))
```

| Upstream | Rotation behaviour |
|---|---|
| `401` / `403` | disable that credential, take the next |
| `429` | cooldown `Retry-After` / `rate_limit_reset` (1 s…1 h, default 60 s), take the next |
| `5xx` | retry the same credential once (`max5xxRetries`), then rotate |
| anything else | re-throw unchanged; pool state untouched |
| empty / all blocked | `PoolExhaustedError` with counts and a redacted last error |

Acquired credentials are leased (default 60 s), so concurrent tasks never
receive the same key. `store.health(provider)` is the read-only view; keys and
their values never appear in health, `toString`, `toJSON`, or error text — only
a 4-character hint. Tables (`provider_accounts`, `provider_credentials`) are
created additively with `CREATE TABLE IF NOT EXISTS` in the existing database.

## CLI built-ins

`buildCli(serverName, { serve, extra, version, pluginName })` routes a leading
positional argument to a subcommand. Bare or flag-prefixed argv (for example
`--http`) still starts the server. The built-in subcommands are:

| Subcommand | Behaviour |
|---|---|
| `status` | Same as `config status`: prints configured / not configured (exit 0), or reports a corrupt blob (exit 1). Legacy dirs count. |
| `logout [--yes]` | Same as `config delete`: removes the stored credential from the current and every legacy dir. Without `--yes` it prompts on a TTY; non-interactive use without `--yes` exits 1. |
| `config status` / `config delete [--yes]` | The long forms of the two commands above. |
| `doctor` | Runtime, backend, store dir and config checks. Warns `only legacy store dir found` when credentials still sit in a legacy dir. |
| `token hash` | Mint a token-table hash (see below). |

`auth` is not a built-in. Signing in depends on the server (OAuth device code,
bot token, API key), so each server provides its own `auth`
handler through `extra`. Entries in `extra` also override a built-in of the
same name.

## Token-table HTTP auth

`runHttpServer` accepts an `auth` option with the three modes of core-py
`[server] auth`; when omitted the server runs in `no-auth` mode (loopback only).

| Mode | Config | Behaviour |
|---|---|---|
| `no-auth` | `{ mode: 'no-auth' }` | No Authorization header. Every request runs as `localAuthContext()` (`uid: 'local'`). Refuses to start on a non-loopback `host`. |
| `token` | `{ mode: 'token', tokenHash, rpm? }` | One shared bearer token checked against a `scrypt$...` hash. Caller `uid: 'shared'`, namespace `default`; optional shared per-minute cap. |
| `multi` | `{ mode: 'multi', usersFile?, users? }` | A users.toml table (default `~/.hull/users.toml`). Token maps to a user and namespace; 401 wrong token, 403 disabled user, 429 over the user's `limits.rpm`. |

```ts
import { runHttpServer } from '@n24q02m/hull/transport'
import { currentUser, ensurePathAllowed } from '@n24q02m/hull/token-auth'

await runHttpServer(makeServer, {
  serverName: 'my-mcp',
  host: '0.0.0.0',
  auth: { mode: 'multi', usersFile: '~/.hull/users.toml' }
})

// Inside a tool handler:
const user = currentUser() // { uid, namespace, mode, allowedRoots, rpm, ... }
ensurePathAllowed(user, requestedPath) // throws PathNotAllowedError outside allowed_roots
```

```toml
# users.toml
[users.alice]
token_hash = "scrypt$32768$8$1$<salt-hex>$<dk-hex>"
namespace = "alice"
enabled = true                  # optional, default true
allowed_roots = ["/data/alice"] # optional, default [] (grants no paths)

[users.alice.limits]
rpm = 60                        # optional, default unlimited
```

The guard runs before routing on every path except `/health`, so `/mcp` and
`extraRoutes` handlers both see the caller through `currentUser()`.
Rejections use the core-py JSON body (`{"error": "unauthorized", "detail":
"invalid token"}`) and add `WWW-Authenticate: Bearer` on 401. The config is
validated before the port is bound, so a malformed hash or users.toml fails
at startup. Hashes are interchangeable with core-py (`scrypt`, N=2^15, r=8,
p=1, 16-byte salt, 32-byte key).

Mint a hash with the built-in `token hash` subcommand of any `buildCli`
entry point. It reads the token from piped stdin or a hidden, confirmed
prompt (never from argv) and prints only the hash:

```bash
my-mcp token hash                     # interactive hidden prompt
printf '%s' "$TOKEN" | my-mcp token hash
```

## Example

```ts
import { runHttpServer } from '@n24q02m/hull/transport'
import { encrypt, decrypt } from '@n24q02m/hull/crypto'
import { buildCli, tryOpenBrowser } from '@n24q02m/hull'
```

## Development

```bash
bun install --frozen-lockfile
bun run check   # biome + tsc type-check
bun run test
bun run build   # emits build/
```

## License

Apache-2.0. See [LICENSE](LICENSE).
