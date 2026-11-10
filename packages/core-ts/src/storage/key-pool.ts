/**
 * Credential rotation over one provider pool (TypeScript mirror of
 * `hull_core/credentials/pool.py`).
 *
 * Failure policy:
 *
 * | status        | action                                                    |
 * |---------------|-----------------------------------------------------------|
 * | 401 / 403     | disable the credential and move to the next one            |
 * | 429           | cool it down (`Retry-After` / `rate_limit_reset`, 1s..1h)  |
 * | 5xx           | retry the same credential once, then move to the next one  |
 * | anything else | release the lease and re-raise (request problem, not key)  |
 *
 * The pool is bounded at `credentials * (1 + max5xxRetries)` calls, and the
 * final error carries the redacted last error text — a provider that echoes a
 * key back cannot leak it through hull.
 */

import type { CredentialStore } from './key-pool-store.js'

export const DEFAULT_COOLDOWN_SECONDS = 60
export const MIN_COOLDOWN_SECONDS = 1
export const MAX_COOLDOWN_SECONDS = 3600

const DISABLE_STATUSES = new Set([401, 403])
const COOLDOWN_STATUSES = new Set([429])

/** Host -> provider identity used when config does not name one explicitly. */
export const KNOWN_PROVIDER_HOSTS: Readonly<Record<string, string>> = {
  'openrouter.ai': 'openrouter',
  'api.openai.com': 'openai',
  'api.anthropic.com': 'anthropic',
  'generativelanguage.googleapis.com': 'gemini',
  'api.cohere.com': 'cohere',
  'api.mistral.ai': 'mistral',
  'api.together.xyz': 'together',
  'api.groq.com': 'groq',
  'api.deepseek.com': 'deepseek',
  'api.x.ai': 'xai',
  'api.voyageai.com': 'voyage'
}

/**
 * Derive the provider identity a base URL belongs to. Known hosts map to their
 * short name; anything else becomes the lowercased host with dots replaced by
 * hyphens. An unparseable URL falls back to `default`.
 */
export function providerFromBaseUrl(baseUrl: string): string {
  let host = ''
  try {
    host = new URL(baseUrl).hostname.toLowerCase()
  } catch {
    return 'default'
  }
  if (!host) return 'default'
  for (const [known, provider] of Object.entries(KNOWN_PROVIDER_HOSTS)) {
    if (host === known || host.endsWith(`.${known}`)) return provider
  }
  return host.replaceAll('.', '-')
}

/** One decrypted pool entry. `apiKey` is never part of `toString`/`toJSON`. */
export interface Credential {
  readonly id: number
  readonly provider: string
  readonly accountId: string | null
  readonly label: string | null
  readonly baseUrl: string
  readonly model: string | null
  readonly apiKey: string
  readonly priority: number
  readonly keyHint: string
  readonly masked: string
  toString(): string
  toJSON(): Record<string, unknown>
}

/** Masked state of one pool entry — no key material at all. */
export interface CredentialHealth {
  readonly id: number
  readonly provider: string
  readonly accountId: string | null
  readonly label: string | null
  readonly baseUrl: string
  readonly model: string | null
  readonly keyHint: string
  readonly priority: number
  readonly disabled: boolean
  readonly lastStatus: number | null
  readonly failureCount: number
  readonly cooldownUntil: number
  readonly leaseUntil: number
  readonly cooldownRemaining: number
  readonly usable: boolean
}

/** A named provider identity (one email/workspace per row). */
export interface Account {
  readonly accountId: string
  readonly provider: string
  readonly name: string
  readonly email: string | null
  readonly workspace: string | null
  readonly labels: Record<string, string>
}

/** Outcome of seeding a pool from the pre-pool single-key config. */
export interface MigrationResult {
  readonly provider: string
  readonly added: number
  readonly skipped: number
}

export type FailureAction = 'disable' | 'cooldown' | 'retry' | 'fatal'

export class CredentialPoolError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CredentialPoolError'
  }
}

export class PoolExhaustedError extends CredentialPoolError {
  readonly provider: string
  readonly accountId: string | null
  readonly total: number
  readonly disabled: number
  readonly coolingDown: number
  readonly leased: number
  readonly retryAfter: number | null
  readonly lastError: string | null

  constructor(
    message: string,
    details: {
      provider: string
      accountId?: string | null
      total?: number
      disabled?: number
      coolingDown?: number
      leased?: number
      retryAfter?: number | null
      lastError?: string | null
    }
  ) {
    super(message)
    this.name = 'PoolExhaustedError'
    this.provider = details.provider
    this.accountId = details.accountId ?? null
    this.total = details.total ?? 0
    this.disabled = details.disabled ?? 0
    this.coolingDown = details.coolingDown ?? 0
    this.leased = details.leased ?? 0
    this.retryAfter = details.retryAfter ?? null
    this.lastError = details.lastError ?? null
  }
}

export class CredentialExistsError extends CredentialPoolError {
  constructor(message: string) {
    super(message)
    this.name = 'CredentialExistsError'
  }
}

export class CredentialNotFoundError extends CredentialPoolError {
  constructor(message: string) {
    super(message)
    this.name = 'CredentialNotFoundError'
  }
}

export class AccountError extends CredentialPoolError {
  constructor(message: string) {
    super(message)
    this.name = 'AccountError'
  }
}

/** Map an HTTP status to `disable` / `cooldown` / `retry` / `fatal`. */
export function classifyStatus(status: number | null | undefined): FailureAction {
  if (status === null || status === undefined) return 'fatal'
  if (DISABLE_STATUSES.has(status)) return 'disable'
  if (COOLDOWN_STATUSES.has(status)) return 'cooldown'
  if (status >= 500 && status < 600) return 'retry'
  return 'fatal'
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/** `Retry-After` as delay seconds (numeric form or HTTP-date). */
export function parseRetryAfter(value: unknown, now: number = Date.now() / 1000): number | null {
  if (value === null || value === undefined) return null
  if (isNumber(value)) return Math.max(0, value)
  const text = String(value).trim()
  if (!text) return null
  const numeric = Number(text)
  if (Number.isFinite(numeric)) return Math.max(0, numeric)
  const parsed = Date.parse(text)
  if (Number.isNaN(parsed)) return null
  return Math.max(0, parsed / 1000 - now)
}

/**
 * `rate_limit_reset`/`x-ratelimit-reset` as delay seconds. Providers use two
 * shapes: an epoch second (above 1e9) and a delta already in seconds.
 */
export function parseRateLimitReset(value: unknown, now: number = Date.now() / 1000): number | null {
  if (value === null || value === undefined) return null
  let numeric: number
  if (isNumber(value)) {
    numeric = value
  } else {
    const text = String(value).trim()
    if (!text) return null
    numeric = Number(text)
    if (!Number.isFinite(numeric)) return parseRetryAfter(text, now)
  }
  if (numeric <= 0) return 0
  return numeric > 1_000_000_000 ? Math.max(0, numeric - now) : numeric
}

export interface CooldownOptions {
  retryAfter?: unknown
  rateLimitReset?: unknown
  now?: number
  defaultSeconds?: number
  minimum?: number
  maximum?: number
}

/** Cooldown for a 429: provider hint first, then the default, always clamped. */
export function cooldownSeconds(options: CooldownOptions = {}): number {
  const now = options.now ?? Date.now() / 1000
  const minimum = options.minimum ?? MIN_COOLDOWN_SECONDS
  const maximum = options.maximum ?? MAX_COOLDOWN_SECONDS
  for (const candidate of [
    parseRetryAfter(options.retryAfter, now),
    parseRateLimitReset(options.rateLimitReset, now)
  ]) {
    if (candidate !== null) return Math.min(maximum, Math.max(minimum, candidate))
  }
  return Math.min(maximum, Math.max(minimum, options.defaultSeconds ?? DEFAULT_COOLDOWN_SECONDS))
}

function headerValue(headers: unknown, name: string): unknown {
  if (!headers || typeof headers !== 'object') return undefined
  const getter = (headers as { get?: unknown }).get
  if (typeof getter === 'function') {
    const value = (getter as (key: string) => unknown).call(headers, name)
    if (value !== undefined && value !== null) return value
  }
  const record = headers as Record<string, unknown>
  const wanted = name.toLowerCase()
  for (const [key, value] of Object.entries(record)) {
    if (key.toLowerCase() === wanted && value !== undefined && value !== null) return value
  }
  return undefined
}

export interface KeyPoolOptions {
  accountId?: string | null
  now?: () => number
  max5xxRetries?: number
  leaseSeconds?: number
}

/**
 * Ordered, health-aware rotation over one provider's credentials. A pool is
 * either the provider's default pool (`accountId` null) or one named account's.
 */
export class KeyPool {
  readonly store: CredentialStore
  readonly provider: string
  readonly accountId: string | null
  readonly max5xxRetries: number
  readonly leaseSeconds: number
  private readonly clock: () => number

  constructor(store: CredentialStore, provider: string, options: KeyPoolOptions = {}) {
    if (typeof provider !== 'string' || !provider.trim()) {
      throw new Error('provider must be a non-empty string')
    }
    const retries = options.max5xxRetries ?? 1
    if (retries < 0) throw new Error('max5xxRetries must be >= 0')
    const lease = options.leaseSeconds ?? 60
    if (lease <= 0) throw new Error('leaseSeconds must be > 0')
    this.store = store
    this.provider = provider.trim()
    this.accountId = options.accountId ?? null
    this.max5xxRetries = retries
    this.leaseSeconds = lease
    this.clock = options.now ?? (() => store.now())
  }

  credentials(): Credential[] {
    return this.store.poolCredentials(this.provider, this.accountId)
  }

  health(): CredentialHealth[] {
    return this.store.poolHealth(this.provider, this.accountId)
  }

  toString(): string {
    const total = this.store.diagnose(this.provider, this.accountId).total
    return `KeyPool(provider=${JSON.stringify(this.provider)}, accountId=${JSON.stringify(this.accountId)}, credentials=${total})`
  }

  acquire(options: { exclude?: number[] } = {}): Credential {
    const credential = this.store.leaseFirst(this.provider, this.accountId, {
      now: this.clock(),
      leaseSeconds: this.leaseSeconds,
      exclude: options.exclude ?? []
    })
    if (credential === null) throw this.exhaustedError()
    return credential
  }

  reportSuccess(credential: Credential): void {
    this.store.recordSuccess(credential.id)
  }

  reportFailure(
    credential: Credential,
    status: number | null,
    options: { retryAfter?: unknown; rateLimitReset?: unknown } = {}
  ): FailureAction {
    const action = classifyStatus(status)
    const now = this.clock()
    if (action === 'cooldown') {
      const until =
        now + cooldownSeconds({ retryAfter: options.retryAfter, rateLimitReset: options.rateLimitReset, now })
      this.store.recordFailure(credential.id, status, { now, cooldownUntil: until })
    } else if (action === 'disable' || action === 'retry') {
      this.store.recordFailure(credential.id, status, { now, disable: action === 'disable' })
    } else {
      this.store.releaseLease(credential.id)
    }
    return action
  }

  enable(credentialId: number): void {
    this.store.enable(credentialId)
  }

  /** Replace every stored key of this provider in `text` with `[redacted]`. */
  redact(text: string | null | undefined): string {
    if (!text) return ''
    let result = text
    for (const credential of this.store.credentials(this.provider)) {
      if (credential.apiKey && result.includes(credential.apiKey)) {
        result = result.split(credential.apiKey).join('[redacted]')
      }
    }
    return result
  }

  /** HTTP status carried by a provider error (`ProviderError.status`). */
  static statusOf(error: unknown): number | null {
    if (!error || typeof error !== 'object') return null
    const status = (error as { status?: unknown }).status
    return typeof status === 'number' && Number.isInteger(status) ? status : null
  }

  exhaustedError(lastError?: string | null): PoolExhaustedError {
    const detail = this.store.diagnose(this.provider, this.accountId)
    let message = `no usable ${JSON.stringify(this.provider)} credential`
    if (this.accountId !== null) message += ` for account ${JSON.stringify(this.accountId)}`
    message += ` (total=${detail.total}, disabled=${detail.disabled}, cooling_down=${detail.coolingDown}, leased=${detail.leased}`
    if (detail.retryAfter !== null) message += `, retry in ${Math.round(detail.retryAfter)}s`
    message += ')'
    const redacted = this.redact(lastError)
    if (redacted) message += `; last error: ${redacted.slice(0, 500)}`
    return new PoolExhaustedError(message, {
      provider: this.provider,
      accountId: this.accountId,
      total: detail.total,
      disabled: detail.disabled,
      coolingDown: detail.coolingDown,
      leased: detail.leased,
      retryAfter: detail.retryAfter,
      lastError: redacted || null
    })
  }

  run<T>(call: (credential: Credential) => T, options: { statusOf?: (error: unknown) => number | null } = {}): T {
    const rotation = new Rotation(this, options.statusOf ?? KeyPool.statusOf)
    for (;;) {
      const credential = rotation.nextCredential()
      if (credential === null) throw rotation.exhausted()
      try {
        const result = call(credential)
        this.reportSuccess(credential)
        return result
      } catch (error) {
        if (rotation.failed(credential, error) === 'fatal') throw error
      }
    }
  }

  async runAsync<T>(
    call: (credential: Credential) => Promise<T>,
    options: { statusOf?: (error: unknown) => number | null } = {}
  ): Promise<T> {
    const rotation = new Rotation(this, options.statusOf ?? KeyPool.statusOf)
    for (;;) {
      const credential = rotation.nextCredential()
      if (credential === null) throw rotation.exhausted()
      try {
        const result = await call(credential)
        this.reportSuccess(credential)
        return result
      } catch (error) {
        if (rotation.failed(credential, error) === 'fatal') throw error
      }
    }
  }
}

class Rotation {
  private readonly pool: KeyPool
  private readonly statusOf: (error: unknown) => number | null
  private readonly limit: number
  private attempts = 0
  private readonly retried = new Map<number, number>()
  private readonly skip = new Set<number>()
  lastError: string | null = null

  constructor(pool: KeyPool, statusOf: (error: unknown) => number | null) {
    this.pool = pool
    this.statusOf = statusOf
    const total = pool.store.diagnose(pool.provider, pool.accountId).total
    this.limit = Math.max(1, total * (1 + pool.max5xxRetries))
  }

  nextCredential(): Credential | null {
    if (this.attempts >= this.limit) return null
    let credential: Credential
    try {
      credential = this.pool.acquire({ exclude: [...this.skip] })
    } catch (error) {
      if (error instanceof PoolExhaustedError) return null
      throw error
    }
    this.attempts += 1
    return credential
  }

  failed(credential: Credential, error: unknown): FailureAction {
    const record = (error ?? {}) as Record<string, unknown>
    let retryAfter = record.retryAfter
    let rateLimitReset = record.rateLimitReset
    const headers = record.headers
    if (headers !== undefined && headers !== null) {
      retryAfter ??= headerValue(headers, 'retry-after')
      rateLimitReset ??= headerValue(headers, 'x-ratelimit-reset')
    }
    const action = this.pool.reportFailure(credential, this.statusOf(error), { retryAfter, rateLimitReset })
    this.lastError = this.pool.redact(String(error))
    if (action === 'retry') {
      const count = (this.retried.get(credential.id) ?? 0) + 1
      this.retried.set(credential.id, count)
      if (count > this.pool.max5xxRetries) this.skip.add(credential.id)
    } else if (action === 'disable' || action === 'cooldown') {
      this.skip.add(credential.id)
    }
    return action
  }

  exhausted(): PoolExhaustedError {
    return this.pool.exhaustedError(this.lastError)
  }
}
