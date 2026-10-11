/**
 * Token authentication: one mechanism, three modes. TS counterpart of
 * core-py `hull_core.auth.middleware.Authenticator`.
 *
 * - `no-auth`: loopback-only; one shared namespace; no Authorization header.
 * - `token`:   one shared token; still one shared namespace.
 * - `multi`:   users.toml; token -> user -> namespace; 401 wrong token,
 *              403 disabled user, 429 over limit.
 */

import { homedir } from 'node:os'
import { join } from 'node:path'

import type { AuthContext, HttpAuthMode } from './context.js'
import { localAuthContext } from './context.js'
import { SlidingWindowLimiter } from './limiter.js'
import { pyRepr, pyStrip } from './py-compat.js'
import { isValidTokenHash, verifyToken } from './tokens.js'
import { findUserByToken, loadUsers, type TokenUser } from './users.js'

export type { HttpAuthMode } from './context.js'

/** Mode names in core-py `AUTH_MODES` order. */
export const HTTP_AUTH_MODES: readonly HttpAuthMode[] = Object.freeze(['no-auth', 'token', 'multi'])

/** HTTP auth configuration for `runHttpServer({ auth })`. */
export type HttpAuthConfig =
  | { readonly mode: 'no-auth' }
  | {
      readonly mode: 'token'
      /** `scrypt$...` hash of the shared token (mint with `token hash`). */
      readonly tokenHash: string
      /** Optional shared per-minute cap; `null`/absent means unlimited. */
      readonly rpm?: number | null
    }
  | {
      readonly mode: 'multi'
      /** users.toml path; `~` is expanded. Defaults to `~/.hull/users.toml`. */
      readonly usersFile?: string
      /** Pre-loaded table (e.g. from `loadUsers`); takes precedence over `usersFile`. */
      readonly users?: ReadonlyMap<string, TokenUser>
    }

/** Result of authenticating one request. `status === null` means proceed. */
export interface AuthOutcome {
  readonly context: AuthContext | null
  readonly status: number | null
  readonly detail: string
  readonly ok: boolean
}

/** Raised when an `HttpAuthConfig` is unusable (detected before the server binds). */
export class AuthConfigError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'AuthConfigError'
  }
}

/** Default users.toml location, same as core-py: `~/.hull/users.toml`. */
export function defaultUsersFile(): string {
  return join(homedir(), '.hull', 'users.toml')
}

function expandUser(path: string): string {
  if (path === '~') return homedir()
  if (path.startsWith('~/') || path.startsWith('~\\')) return join(homedir(), path.slice(2))
  return path
}

/**
 * Extract the bearer token the way core-py does: split on the first space,
 * case-insensitive `bearer` scheme, Python-`strip()`ed non-empty value.
 */
export function parseBearer(authz: string | null | undefined): string | null {
  if (!authz) return null
  const space = authz.indexOf(' ')
  const scheme = space === -1 ? authz : authz.slice(0, space)
  const value = space === -1 ? '' : pyStrip(authz.slice(space + 1))
  if (scheme.toLowerCase() !== 'bearer' || value.length === 0) return null
  return value
}

function outcome(context: AuthContext | null, status: number | null, detail = ''): AuthOutcome {
  return Object.freeze({ context, status, detail, ok: status === null && context !== null })
}

function validRpm(rpm: unknown): rpm is number | null | undefined {
  return rpm === undefined || rpm === null || (Number.isSafeInteger(rpm) && (rpm as number) >= 1)
}

type ResolvedConfig =
  | { mode: 'no-auth' }
  | { mode: 'token'; tokenHash: string; rpm: number | null }
  | { mode: 'multi'; users: ReadonlyMap<string, TokenUser> }

function resolveConfig(config: HttpAuthConfig): ResolvedConfig {
  const mode = (config as { mode?: unknown }).mode
  if (mode === 'no-auth') return { mode }
  if (mode === 'token') {
    const { tokenHash, rpm } = config as Extract<HttpAuthConfig, { mode: 'token' }>
    if (typeof tokenHash !== 'string' || tokenHash.length === 0) {
      throw new AuthConfigError('[server] auth = "token" requires a non-empty token_hash')
    }
    if (!isValidTokenHash(tokenHash)) {
      throw new AuthConfigError('[server] token_hash is not a scrypt encoding (mint one with `hull token hash`)')
    }
    if (!validRpm(rpm)) throw new AuthConfigError('[server] rpm must be a positive integer')
    return { mode, tokenHash, rpm: rpm ?? null }
  }
  if (mode === 'multi') {
    const { users, usersFile } = config as Extract<HttpAuthConfig, { mode: 'multi' }>
    if (users !== undefined) {
      for (const [uid, user] of users) {
        if (!isValidTokenHash(user.tokenHash)) {
          throw new AuthConfigError(
            `user ${pyRepr(uid)}: token_hash missing or not a scrypt encoding (mint one with \`hull token hash\`)`
          )
        }
      }
      return { mode, users }
    }
    const path = usersFile ? expandUser(usersFile) : defaultUsersFile()
    return { mode, users: loadUsers(path) }
  }
  const shown = typeof mode === 'string' ? pyRepr(mode) : String(mode)
  throw new AuthConfigError(`[server] auth must be one of ('no-auth', 'token', 'multi'), got ${shown}`)
}

/** Resolves bearer tokens to `AuthContext`s according to the configured mode. */
export class Authenticator {
  readonly mode: HttpAuthMode
  readonly limiter: SlidingWindowLimiter
  private readonly config: ResolvedConfig

  /**
   * Validates the whole config up front (unknown mode, empty or malformed
   * token hash, bad rpm, unreadable users.toml) so a broken deployment
   * fails at boot rather than answering 500 per request.
   */
  constructor(config: HttpAuthConfig, options: { limiter?: SlidingWindowLimiter } = {}) {
    this.config = resolveConfig(config)
    this.mode = this.config.mode
    this.limiter = options.limiter ?? new SlidingWindowLimiter()
  }

  /** Number of rows in the users table (`multi` mode), else 0. */
  get userCount(): number {
    return this.config.mode === 'multi' ? this.config.users.size : 0
  }

  async authenticate(authz: string | null | undefined): Promise<AuthOutcome> {
    const config = this.config
    if (config.mode === 'no-auth') return outcome(localAuthContext(), null)

    const token = parseBearer(authz)
    if (token === null) return outcome(null, 401, 'missing or malformed Authorization header')

    if (config.mode === 'token') {
      if (!(await verifyToken(token, config.tokenHash))) return outcome(null, 401, 'invalid token')
      return this.admit({
        uid: 'shared',
        namespace: 'default',
        mode: 'token',
        allowedRoots: Object.freeze([]),
        rpm: config.rpm,
        disabled: false,
        attributes: Object.freeze({})
      })
    }

    const user = await findUserByToken(config.users, token)
    if (user === null) return outcome(null, 401, 'invalid token')
    if (!user.enabled) return outcome(null, 403, `user ${pyRepr(user.uid)} is disabled`)
    return this.admit({
      uid: user.uid,
      namespace: user.namespace,
      mode: 'multi',
      allowedRoots: user.allowedRoots,
      rpm: user.limits.rpm,
      disabled: false,
      attributes: Object.freeze({})
    })
  }

  private admit(ctx: AuthContext): AuthOutcome {
    if (!this.limiter.check(ctx.uid, ctx.rpm)) return outcome(null, 429, 'rate limit exceeded')
    return outcome(Object.freeze(ctx), null)
  }
}
