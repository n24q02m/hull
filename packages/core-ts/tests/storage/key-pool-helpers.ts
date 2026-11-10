/** Shared helpers for the provider key pool tests. */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach } from 'vitest'
import { CredentialStore } from '../../src/storage/key-pool-store.js'

export const BASE_URL = 'https://openrouter.ai/api/v1'

/** Injectable wall clock so cooldown/lease windows need no sleeping. */
export class Clock {
  value = 1_700_000_000

  now = (): number => this.value

  advance(seconds: number): number {
    this.value += seconds
    return this.value
  }
}

/** Stand-in for a provider HTTP error carrying a status and rate-limit hints. */
export class ProviderFailure extends Error {
  readonly status: number | null
  readonly detail: string
  readonly retryAfter: unknown
  readonly rateLimitReset: unknown
  readonly headers: Record<string, string> | undefined

  constructor(
    status: number | null,
    detail = '',
    options: { retryAfter?: unknown; rateLimitReset?: unknown; headers?: Record<string, string> } = {}
  ) {
    super(`provider returned ${status}: ${detail}`)
    this.name = 'ProviderFailure'
    this.status = status
    this.detail = detail
    this.retryAfter = options.retryAfter
    this.rateLimitReset = options.rateLimitReset
    this.headers = options.headers
  }
}

export function deferred<T = void>(): {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (reason: unknown) => void
} {
  return Promise.withResolvers<T>()
}

/** Per-test store in a temp dir with a controllable clock. */
export class PoolFixture {
  readonly clock = new Clock()
  readonly dir: string
  readonly store: CredentialStore

  constructor() {
    this.dir = mkdtempSync(join(tmpdir(), 'hull-key-pool-'))
    this.store = new CredentialStore(join(this.dir, 'hull.db'), { now: this.clock.now })
  }

  seed(count = 2): number[] {
    return Array.from(
      { length: count },
      (_, index) =>
        this.store.addCredential({
          provider: 'openrouter',
          apiKey: `sk-live-rotation-${String(index).padStart(4, '0')}`,
          baseUrl: BASE_URL,
          label: `c${index}`
        }).id
    )
  }

  close(): void {
    this.store.close()
  }

  cleanup(): void {
    this.close()
    rmSync(this.dir, { recursive: true, force: true })
  }
}

/** Register a fixture that is torn down after each test in the suite. */
export function usePoolFixture(): () => PoolFixture {
  let fixture: PoolFixture
  beforeEach(() => {
    fixture = new PoolFixture()
  })
  afterEach(() => {
    fixture.cleanup()
  })
  return () => fixture
}
