/**
 * Request-scoped identity for MCP tools.
 *
 * TS counterpart of core-py `hull_core.auth.context`. The HTTP auth guard
 * resolves the caller before the MCP handler runs and binds the resulting
 * `AuthContext` in an `AsyncLocalStorage` (the analogue of Python's
 * contextvar). Tools read it with `currentUser()` and scope every storage
 * access by its `namespace`.
 */

import { AsyncLocalStorage } from 'node:async_hooks'

/** The three HTTP auth modes, mirroring core-py `AUTH_MODES`. */
export type HttpAuthMode = 'no-auth' | 'token' | 'multi'

/** Resolved caller identity attached to one HTTP request. */
export interface AuthContext {
  readonly uid: string
  readonly namespace: string
  readonly mode: HttpAuthMode
  /** Filesystem roots the caller may touch; empty grants nothing. */
  readonly allowedRoots: readonly string[]
  /** Per-minute request cap; `null` means unlimited. */
  readonly rpm: number | null
  readonly disabled: boolean
  readonly attributes: Readonly<Record<string, unknown>>
}

const storage = new AsyncLocalStorage<AuthContext>()

/** Identity used in no-auth mode (and for in-process tool calls). */
export function localAuthContext(): AuthContext {
  return Object.freeze({
    uid: 'local',
    namespace: 'default',
    mode: 'no-auth',
    allowedRoots: Object.freeze([]) as readonly string[],
    rpm: null,
    disabled: false,
    attributes: Object.freeze({})
  })
}

/** The caller's identity; falls back to the local identity when unset. */
export function currentUser(): AuthContext {
  return storage.getStore() ?? localAuthContext()
}

/** Run `fn` with `ctx` bound as the current user for its whole async scope. */
export function runWithAuthContext<T>(ctx: AuthContext, fn: () => T): T {
  return storage.run(ctx, fn)
}
