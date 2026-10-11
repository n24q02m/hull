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
