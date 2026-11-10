import { describe, expect, it } from 'vitest'
import {
  type Credential,
  classifyStatus,
  cooldownSeconds,
  DEFAULT_COOLDOWN_SECONDS,
  KeyPool,
  PoolExhaustedError,
  parseRateLimitReset,
  parseRetryAfter,
  providerFromBaseUrl
} from '../../src/storage/key-pool.js'
import { deferred, ProviderFailure, usePoolFixture } from './key-pool-helpers.js'

const fixture = usePoolFixture()

function capture(fn: () => unknown): unknown {
  try {
    fn()
  } catch (error) {
    return error
  }
  throw new Error('expected the call to throw')
}

describe('rotation policy', () => {
  it.each([401, 403])('disables the credential on %i and rotates', (status) => {
    const pool = fixture().store.pool('openrouter')
    const [first, second] = fixture().seed(2)
    const seen: number[] = []
    const result = pool.run((credential) => {
      seen.push(credential.id)
      if (credential.id === first) throw new ProviderFailure(status, 'invalid api key')
      return 'ok'
    })
    expect(result).toBe('ok')
    expect(seen).toEqual([first, second])
    const health = fixture().store.health('openrouter')
    expect(health[0].disabled).toBe(true)
    expect(health[0].lastStatus).toBe(status)
    expect(health[1].lastStatus).toBe(200)
  })

  it('honours a numeric Retry-After on 429', () => {
    const pool = fixture().store.pool('openrouter')
    const [first] = fixture().seed(2)
    pool.run((credential) => {
      if (credential.id === first) throw new ProviderFailure(429, 'rate limited', { retryAfter: '120' })
      return 'ok'
    })
    const health = fixture().store.health('openrouter')[0]
    expect(health.lastStatus).toBe(429)
    expect(health.cooldownUntil).toBeCloseTo(fixture().clock.value + 120)
    expect(health.usable).toBe(false)
  })

  it('honours an HTTP-date Retry-After on 429', () => {
    const pool = fixture().store.pool('openrouter')
    const [first] = fixture().seed(2)
    const header = new Date((fixture().clock.value + 90) * 1000).toUTCString()
    pool.run((credential) => {
      if (credential.id === first) {
        throw new ProviderFailure(429, 'rate limited', { headers: { 'Retry-After': header } })
      }
      return 'ok'
    })
    expect(fixture().store.health('openrouter')[0].cooldownUntil).toBeCloseTo(fixture().clock.value + 90, 0)
  })

  it('honours an epoch rate_limit_reset on 429', () => {
    const pool = fixture().store.pool('openrouter')
    const [first] = fixture().seed(2)
    pool.run((credential) => {
      if (credential.id === first) {
        throw new ProviderFailure(429, 'rate limited', { rateLimitReset: fixture().clock.value + 75 })
      }
      return 'ok'
    })
    expect(fixture().store.health('openrouter')[0].cooldownUntil).toBeCloseTo(fixture().clock.value + 75)
  })

  it('returns a credential to the pool once its cooldown expires', () => {
    const pool = fixture().store.pool('openrouter')
    const [first, second] = fixture().seed(2)
    pool.run((credential) => {
      if (credential.id === first) throw new ProviderFailure(429, 'rate limited', { retryAfter: 30 })
      return 'ok'
    })
    expect(() => pool.acquire({ exclude: [second] })).toThrow(PoolExhaustedError)
    fixture().clock.advance(31)
    expect(pool.acquire().id).toBe(first)
  })

  it('retries a 5xx on the same credential once, then moves on', () => {
    const pool = fixture().store.pool('openrouter')
    const [first, second] = fixture().seed(2)
    const seen: number[] = []
    const result = pool.run((credential) => {
      seen.push(credential.id)
      if (credential.id === first) throw new ProviderFailure(500, 'upstream exploded')
      return 'ok'
    })
    expect(result).toBe('ok')
    expect(seen).toEqual([first, first, second])
    const health = fixture().store.health('openrouter')[0]
    expect(health.failureCount).toBe(2)
    expect(health.lastStatus).toBe(500)
    expect(health.disabled).toBe(false)
  })

  it('bounds attempts instead of looping forever', () => {
    fixture().seed(3)
    const seen: number[] = []
    expect(() =>
      fixture()
        .store.pool('openrouter', { max5xxRetries: 2 })
        .run((credential) => {
          seen.push(credential.id)
          throw new ProviderFailure(503, 'unavailable')
        })
    ).toThrow(PoolExhaustedError)
    expect(seen).toHaveLength(9)
  })

  it('surfaces the last error with every key redacted', () => {
    const [first, second] = fixture().seed(2)
    const keys = [first, second].map((id) => fixture().store.getCredential(id).apiKey)
    const error = capture(() =>
      fixture()
        .store.pool('openrouter')
        .run((credential) => {
          throw new ProviderFailure(401, `invalid api key ${credential.apiKey} (account leak)`)
        })
    ) as PoolExhaustedError
    expect(error).toBeInstanceOf(PoolExhaustedError)
    const rendered = `${String(error)}\n${error.lastError}`
    for (const key of keys) expect(rendered).not.toContain(key)
    expect(rendered).toContain('invalid api key [redacted]')
    expect([error.total, error.disabled, error.coolingDown]).toEqual([2, 2, 0])
  })

  it('reports total=0 for an empty pool', () => {
    const error = capture(() =>
      fixture()
        .store.pool('openrouter')
        .run(() => 'never')
    ) as PoolExhaustedError
    expect(error.message).toContain('total=0')
    expect(error.lastError).toBeNull()
  })

  it('re-raises a non-credential failure and releases the lease', () => {
    const [first] = fixture().seed(2)
    expect(() =>
      fixture()
        .store.pool('openrouter')
        .run(() => {
          throw new ProviderFailure(400, 'malformed request')
        })
    ).toThrow(ProviderFailure)
    expect(fixture().store.pool('openrouter').acquire().id).toBe(first)
    expect(fixture().store.health('openrouter')[0].failureCount).toBe(0)
  })

  it('re-raises a non-HTTP exception and releases the lease', () => {
    const [first] = fixture().seed(2)
    expect(() =>
      fixture()
        .store.pool('openrouter')
        .run(() => {
          throw new Error('bad payload')
        })
    ).toThrow('bad payload')
    expect(fixture().store.pool('openrouter').acquire().id).toBe(first)
  })

  it('rotates through runAsync with the same policy', async () => {
    const [first] = fixture().seed(2)
    const result = await fixture()
      .store.pool('openrouter')
      .runAsync(async (credential) => {
        if (credential.id === first) throw new ProviderFailure(401, 'invalid api key')
        return 'async-ok'
      })
    expect(result).toBe('async-ok')
    expect(fixture().store.health('openrouter')[0].disabled).toBe(true)
  })

  it('leases each credential once and never hands one out twice', () => {
    const [first, second] = fixture().seed(2)
    const pool = fixture().store.pool('openrouter', { leaseSeconds: 30 })
    expect([pool.acquire().id, pool.acquire().id].sort()).toEqual([first, second].sort())
    expect(() => pool.acquire()).toThrow(/leased=2/)
    pool.reportSuccess(fixture().store.getCredential(first))
    expect(pool.acquire().id).toBe(first)
  })
})

describe('secret hygiene', () => {
  it('keeps keys out of toString, JSON and health snapshots', () => {
    const [first] = fixture().seed(2)
    const pool = fixture().store.pool('openrouter')
    const credential: Credential = fixture().store.getCredential(first)
    for (const rendered of [
      String(credential),
      JSON.stringify(credential),
      pool.toString(),
      JSON.stringify(pool.health()),
      JSON.stringify(fixture().store.health())
    ]) {
      expect(rendered).not.toContain(credential.apiKey)
    }
    expect(credential.masked.endsWith('…0000')).toBe(true)
  })
})

describe('concurrent tasks', () => {
  it('never reuses the credential that just got a 429 in another task', () => {
    const [first, second] = fixture().seed(2)
    const pool = fixture().store.pool('openrouter', { leaseSeconds: 30 })
    const taskACredential = pool.acquire()
    const taskBCredential = pool.acquire()
    expect([taskACredential.id, taskBCredential.id]).toEqual([first, second])
    pool.reportFailure(taskACredential, 429, { retryAfter: 60 })
    // The first credential is cooling and the second is still leased by task B:
    // a third attempt cannot grab either one.
    expect(() => pool.acquire()).toThrow(PoolExhaustedError)
    pool.reportSuccess(taskBCredential)
    expect(pool.acquire().id).toBe(second)
  })

  it('gives two simultaneous runAsync tasks distinct credentials', async () => {
    const [first, second] = fixture().seed(2)
    const pool = fixture().store.pool('openrouter', { leaseSeconds: 30 })
    const bAcquired = deferred<number>()
    const releaseA = deferred<void>()
    const taskA = pool.runAsync(async (credential) => {
      expect(credential.id).toBe(first)
      expect(await bAcquired.promise).toBe(second)
      await releaseA.promise
      return 'a'
    })
    const taskB = pool.runAsync(async (credential) => {
      bAcquired.resolve(credential.id)
      return 'b'
    })
    expect(await taskB).toBe('b')
    releaseA.resolve()
    expect(await taskA).toBe('a')
  })
})

describe('policy helpers', () => {
  it.each([
    [null, 'fatal'],
    [400, 'fatal'],
    [401, 'disable'],
    [403, 'disable'],
    [429, 'cooldown'],
    [500, 'retry'],
    [503, 'retry'],
    [599, 'retry'],
    [600, 'fatal']
  ] as const)('classifies %s as %s', (status, action) => {
    expect(classifyStatus(status)).toBe(action)
  })

  it('parses Retry-After forms', () => {
    const now = 1_700_000_000
    expect(parseRetryAfter(null)).toBeNull()
    expect(parseRetryAfter('')).toBeNull()
    expect(parseRetryAfter('12')).toBe(12)
    expect(parseRetryAfter(12)).toBe(12)
    expect(parseRetryAfter('-5')).toBe(0)
    expect(parseRetryAfter('not-a-date')).toBeNull()
    expect(parseRetryAfter(new Date((now + 45) * 1000).toUTCString(), now)).toBeCloseTo(45, 0)
    expect(parseRetryAfter(new Date((now - 45) * 1000).toUTCString(), now)).toBe(0)
  })

  it('parses rate_limit_reset forms', () => {
    const now = 1_700_000_000
    expect(parseRateLimitReset(null)).toBeNull()
    expect(parseRateLimitReset(30)).toBe(30)
    expect(parseRateLimitReset('30')).toBe(30)
    expect(parseRateLimitReset(now + 60, now)).toBeCloseTo(60)
    expect(parseRateLimitReset(0)).toBe(0)
    expect(parseRateLimitReset('garbage')).toBeNull()
  })

  it('prefers Retry-After and clamps the cooldown window', () => {
    const now = 1_700_000_000
    expect(cooldownSeconds({ retryAfter: 20, rateLimitReset: 90, now })).toBe(20)
    expect(cooldownSeconds({ rateLimitReset: 90, now })).toBe(90)
    expect(cooldownSeconds({ now })).toBe(DEFAULT_COOLDOWN_SECONDS)
    expect(cooldownSeconds({ retryAfter: 0, now })).toBe(1)
    expect(cooldownSeconds({ retryAfter: 99_999, now })).toBe(3600)
  })

  it('derives provider identities from base URLs', () => {
    expect(providerFromBaseUrl('https://openrouter.ai/api/v1')).toBe('openrouter')
    expect(providerFromBaseUrl('https://api.openai.com/v1')).toBe('openai')
    expect(providerFromBaseUrl('https://gateway.example.com/v1')).toBe('gateway-example-com')
    expect(providerFromBaseUrl('http://127.0.0.1:9800')).toBe('127-0-0-1')
    expect(providerFromBaseUrl('not a url')).toBe('default')
  })

  it('validates KeyPool configuration', () => {
    const store = fixture().store
    expect(() => new KeyPool(store, '  ')).toThrow(/provider/)
    expect(() => new KeyPool(store, 'openrouter', { max5xxRetries: -1 })).toThrow(/max5xxRetries/)
    expect(() => new KeyPool(store, 'openrouter', { leaseSeconds: 0 })).toThrow(/leaseSeconds/)
  })
})
