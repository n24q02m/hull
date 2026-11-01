import { describe, expect, it } from 'vitest'

import type { HttpAuthConfig, HttpRoute } from '../src/index.js'
import * as publicApi from '../src/index.js'

describe('public API surface', () => {
  it('exports the HttpRoute type for consumers registering extra routes', () => {
    // Type-level check: `tsc --noEmit` (via `bun run check`) fails unless
    // HttpRoute is exported from the root entry point with this shape.
    const route: HttpRoute = {
      method: 'GET',
      path: '/accounts/callback',
      handler: (_req, res) => {
        res.writeHead(204)
        res.end()
      }
    }
    expect(route.path).toBe('/accounts/callback')
  })

  it('keeps the helpers published consumers import from the root entry point', () => {
    // better-email-mcp opens OAuth device-code URLs with tryOpenBrowser, and every
    // TS server wires buildCli / runHttpServer from the root. Removing one breaks
    // those consumers at type-check time, so pin them here.
    const exported: Record<string, unknown> = { ...publicApi }
    for (const name of ['buildCli', 'runHttpServer', 'tryOpenBrowser']) {
      expect(typeof exported[name], name).toBe('function')
    }
  })

  it('does not export the removed relay-era surface', () => {
    const exported: Record<string, unknown> = { ...publicApi }
    for (const name of [
      'deriveJwtSigningSeed',
      'markSetupComplete',
      'getMode',
      'setLocalMode',
      'clearMode',
      'acquireSessionLock',
      'writeSessionLock',
      'releaseSessionLock'
    ]) {
      expect(exported[name], name).toBeUndefined()
    }
  })

  it('re-exports the token-table auth API from root', () => {
    const exported: Record<string, unknown> = { ...publicApi }
    for (const name of [
      'hashToken',
      'verifyToken',
      'isValidTokenHash',
      'TokenHashError',
      'loadUsers',
      'findUserByToken',
      'ensurePathAllowed',
      'UsersError',
      'PathNotAllowedError',
      'Authenticator',
      'AuthConfigError',
      'parseBearer',
      'defaultUsersFile',
      'currentUser',
      'localAuthContext',
      'runWithAuthContext',
      'SlidingWindowLimiter',
      'guardRequest',
      'writeAuthRejection',
      'isLoopbackHost',
      'assertNoAuthLoopback',
      'ServerConfigError',
      'tokenCliHandler'
    ]) {
      expect(typeof exported[name], name).toBe('function')
    }
    expect(publicApi.HTTP_AUTH_MODES).toEqual(['no-auth', 'token', 'multi'])
    // Type-level check: the `auth` option type is exported from root.
    const auth: HttpAuthConfig = { mode: 'no-auth' }
    expect(auth.mode).toBe('no-auth')
  })
})
