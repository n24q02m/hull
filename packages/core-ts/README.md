# @n24q02m/hull

**TypeScript core of hull: MCP Streamable HTTP transport, OAuth 2.1 authorization server, credential storage and relay client for self-hosted MCP servers.**

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
| `@n24q02m/hull/transport` | `runHttpServer` (local HTTP server with routes), `StreamableHTTPServer` (MCP Streamable HTTP transport), `OAuthMiddleware` (bearer JWT verification) |
| `@n24q02m/hull/oauth` | `OAuthProvider` (OAuth 2.1 authorization server), `JWTIssuer`, `verifyPKCE`, `InMemoryAuthCache`, `SqliteUserStore` |
| `@n24q02m/hull/auth` | `createLocalOAuthApp` and `createDelegatedOAuthApp` (credential-form and upstream-OAuth flows), `renderCredentialForm`, `createSessionStore`, `deriveStableSub`, `authorizationServerMetadata` / `protectedResourceMetadata` (well-known documents) |
| `@n24q02m/hull/crypto` | AES `encrypt` / `decrypt`, ECDH `generateKeyPair` / `deriveSharedSecret` / `exportPublicKey` / `importPublicKey`, key derivation `deriveAesKey` / `deriveJwtSigningSeed` |
| `@n24q02m/hull/storage` | `atomicWriteFile`, credential backends (`LocalFsBackend`, `InMemoryBackend`, `CfKvBackend`, `backendFromEnv`), encrypted credential read/write/delete and config import/export, `PerPluginStore`, `resolveConfig` (environment variables, then stored config, then defaults), server mode state, session lock, machine identity helpers |
| `@n24q02m/hull/relay` | Relay client: `createSession`, `generatePassphrase`, `pollForResult`, `sendMessage`, `pollForResponses`, `notifyComplete` |
| `@n24q02m/hull/schema` | Type-only config schema: `ConfigField`, `ConfigMode`, `RelayConfigSchema` and related form layout types |
| `@n24q02m/hull/token-auth` | Token-table HTTP auth: `Authenticator`, `hashToken` / `verifyToken`, `loadUsers`, `currentUser`, `ensurePathAllowed`, `guardRequest`, `SlidingWindowLimiter`, `tokenCliHandler` |

## Token-table HTTP auth

`runHttpServer` accepts an `auth` option with the three modes of core-py
`[server] auth`. It replaces the OAuth/relay stack for that server: `auth` is
mutually exclusive with `relaySchema`, `delegatedOAuth`, `jwtIssuer`,
`authDisabled` and `authScope`, and needs neither a relay nor an OAuth
issuer.

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
import type { RelayConfigSchema } from '@n24q02m/hull/schema'
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
