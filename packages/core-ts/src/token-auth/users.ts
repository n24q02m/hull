/**
 * users.toml: the multi-user token table (auth mode `multi`).
 *
 * Same schema and validation order as core-py `hull_core.auth.users`:
 *
 *     [users.<uid>]
 *     token_hash = "scrypt$..."        # mint with `hull token hash`
 *     enabled = true
 *     namespace = "<uid-or-shared>"    # data isolation root
 *     allowed_roots = ["/data/alice"]  # filesystem roots tools may touch
 *     [users.<uid>.limits]
 *     rpm = 60                         # optional per-minute request cap
 *
 * Admin is host-side only: edit the file, restart the server. There is no
 * admin API and no self-service.
 */

import { readFileSync, realpathSync, statSync } from 'node:fs'
import { basename, dirname, join, resolve, sep } from 'node:path'

import { parse, TomlError } from 'smol-toml'

import { pyRepr } from './py-compat.js'
import { isValidTokenHash, verifyToken } from './tokens.js'

/** Raised when users.toml is missing, malformed, or semantically invalid. */
export class UsersError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'UsersError'
  }
}

/** Raised by `ensurePathAllowed` (core-py raises `PermissionError`). */
export class PathNotAllowedError extends Error {
  readonly code = 'EACCES'

  constructor(message: string) {
    super(message)
    this.name = 'PathNotAllowedError'
  }
}

export interface TokenUserLimits {
  /** Per-minute request cap; `null` means unlimited. */
  readonly rpm: number | null
}

/** One validated `[users.<uid>]` row. */
export interface TokenUser {
  readonly uid: string
  readonly tokenHash: string
  readonly enabled: boolean
  readonly namespace: string
  readonly allowedRoots: readonly string[]
  readonly limits: TokenUserLimits
}

/** Anything carrying a caller's filesystem roots (a `TokenUser` or an `AuthContext`). */
export interface PathScope {
  readonly uid: string
  readonly allowedRoots: readonly string[]
}

type TomlRecord = Record<string, unknown>

function isTable(value: unknown): value is TomlRecord {
  // TOML tables parse to plain (null-prototype) objects; arrays and
  // TomlDate values are excluded by their tags.
  return Object.prototype.toString.call(value) === '[object Object]'
}

function readTomlText(path: string): string {
  let isFile = false
  try {
    isFile = statSync(path).isFile()
  } catch {
    isFile = false
  }
  if (!isFile) throw new UsersError(`users file not found: ${path}`)
  const bytes = readFileSync(path)
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch (err) {
    throw new UsersError(`users file is not valid UTF-8: ${path}`, { cause: err })
  }
}

function field(entry: TomlRecord, key: string, fallback: unknown): unknown {
  return Object.hasOwn(entry, key) ? entry[key] : fallback
}

function parseUser(uid: string, entry: unknown): TokenUser {
  const who = `user ${pyRepr(uid)}`
  if (!isTable(entry)) throw new UsersError(`${who}: expected a table`)

  const tokenHash = field(entry, 'token_hash', undefined)
  if (typeof tokenHash !== 'string' || !isValidTokenHash(tokenHash)) {
    throw new UsersError(`${who}: token_hash missing or not a scrypt encoding (mint one with \`hull token hash\`)`)
  }
  const namespace = field(entry, 'namespace', undefined)
  if (typeof namespace !== 'string' || namespace.length === 0) {
    throw new UsersError(`${who}: namespace must be a non-empty string`)
  }
  const enabled = field(entry, 'enabled', true)
  if (typeof enabled !== 'boolean') throw new UsersError(`${who}: enabled must be a boolean`)

  const roots = field(entry, 'allowed_roots', [])
  if (!Array.isArray(roots) || !roots.every((r) => typeof r === 'string')) {
    throw new UsersError(`${who}: allowed_roots must be a list of strings`)
  }
  const limits = field(entry, 'limits', {})
  if (!isTable(limits)) throw new UsersError(`${who}: limits must be a table`)
  const rpmRaw = field(limits, 'rpm', undefined)
  let rpm: number | null = null
  if (rpmRaw !== undefined) {
    if (typeof rpmRaw !== 'bigint' || rpmRaw < 1n) {
      throw new UsersError(`${who}: limits.rpm must be a positive integer`)
    }
    rpm = Number(rpmRaw)
  }
  return Object.freeze({
    uid,
    tokenHash,
    enabled,
    namespace,
    allowedRoots: Object.freeze([...(roots as string[])]),
    limits: Object.freeze({ rpm })
  })
}

/**
 * Parse and validate users.toml into a `uid -> TokenUser` map.
 *
 * Synchronous on purpose: it runs once at server build time, and each
 * `token_hash` is checked with one full scrypt derivation (like core-py
 * `is_valid_encoding`) so an unusable row fails the boot, not a request.
 */
export function loadUsers(path: string): Map<string, TokenUser> {
  const text = readTomlText(path)
  let raw: TomlRecord
  try {
    raw = parse(text, { integersAsBigInt: true }) as TomlRecord
  } catch (err) {
    const reason = err instanceof TomlError ? err.message : String(err)
    throw new UsersError(`users file is not valid TOML: ${path}: ${reason}`, { cause: err })
  }

  const table = field(raw, 'users', undefined)
  if (!isTable(table) || Object.keys(table).length === 0) {
    throw new UsersError('users file must contain a non-empty [users] table')
  }
  const users = new Map<string, TokenUser>()
  for (const [uid, entry] of Object.entries(table)) users.set(uid, parseUser(uid, entry))
  return users
}

/** Return the first user whose `tokenHash` verifies `token`, else `null`. */
export async function findUserByToken(users: ReadonlyMap<string, TokenUser>, token: string): Promise<TokenUser | null> {
  for (const user of users.values()) {
    if (await verifyToken(token, user.tokenHash)) return user
  }
  return null
}

// Python `Path.resolve()` (non-strict): absolutize, follow symlinks as far
// as the path exists, keep the non-existent tail lexically.
function resolveLoose(target: string): string {
  const absolute = resolve(target)
  try {
    return realpathSync.native(absolute)
  } catch {
    const parent = dirname(absolute)
    if (parent === absolute) return absolute
    return join(resolveLoose(parent), basename(absolute))
  }
}

function comparable(p: string): string {
  return process.platform === 'win32' ? p.toLowerCase() : p
}

function isWithin(resolved: string, root: string): boolean {
  const target = comparable(resolved)
  const base = comparable(root)
  if (target === base) return true
  const prefix = base.endsWith(sep) ? base : base + sep
  return target.startsWith(prefix)
}

/**
 * Enforce the caller's `allowedRoots`. An empty list grants nothing. Both
 * sides are resolved first, so a symlinked escape needs its resolved target
 * to sit under a root too. Throws `PathNotAllowedError` on denial.
 */
export function ensurePathAllowed(scope: PathScope, path: string): void {
  const resolved = resolveLoose(path)
  for (const root of scope.allowedRoots) {
    if (isWithin(resolved, resolveLoose(root))) return
  }
  throw new PathNotAllowedError(`path ${resolved} is outside allowed roots for user ${pyRepr(scope.uid)}`)
}
