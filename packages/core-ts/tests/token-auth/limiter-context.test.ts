import { describe, expect, it } from 'vitest'

import { type AuthContext, currentUser, localAuthContext, runWithAuthContext } from '../../src/token-auth/context.js'
import { SlidingWindowLimiter } from '../../src/token-auth/limiter.js'

describe('SlidingWindowLimiter (core-py parity)', () => {
  it('is unlimited when rpm is null', () => {
    const limiter = new SlidingWindowLimiter()
    for (let t = 0; t < 1000; t++) expect(limiter.check('u', null, t)).toBe(true)
  })

  it('enforces the 60s window', () => {
    const limiter = new SlidingWindowLimiter()
    expect(limiter.check('u', 3, 1.0)).toBe(true)
    expect(limiter.check('u', 3, 2.0)).toBe(true)
    expect(limiter.check('u', 3, 3.0)).toBe(true)
    expect(limiter.check('u', 3, 3.5)).toBe(false)
    // the 1.0 and 2.0 events have expired at t=63
    expect(limiter.check('u', 3, 63.0)).toBe(true)
  })

  it('isolates keys', () => {
    const limiter = new SlidingWindowLimiter()
    expect(limiter.check('a', 1, 1.0)).toBe(true)
    expect(limiter.check('a', 1, 1.5)).toBe(false)
    expect(limiter.check('b', 1, 1.5)).toBe(true)
  })

  it('uses a monotonic clock by default', () => {
    const limiter = new SlidingWindowLimiter()
    expect(limiter.check('clock', 1)).toBe(true)
    expect(limiter.check('clock', 1)).toBe(false)
  })
})

describe('auth context', () => {
  const alice: AuthContext = {
    uid: 'alice',
    namespace: 'alice-ns',
    mode: 'multi',
    allowedRoots: ['/data/alice'],
    rpm: 5,
    disabled: false,
    attributes: {}
  }

  it('local identity matches core-py AuthContext.local()', () => {
    const local = localAuthContext()
    expect(local).toEqual({
      uid: 'local',
      namespace: 'default',
      mode: 'no-auth',
      allowedRoots: [],
      rpm: null,
      disabled: false,
      attributes: {}
    })
    expect(Object.isFrozen(local)).toBe(true)
  })

  it('currentUser falls back to the local identity when unbound', () => {
    expect(currentUser().uid).toBe('local')
  })

  it('binds the identity across awaits and restores it afterwards', async () => {
    const seen = await runWithAuthContext(alice, async () => {
      await new Promise((resolve) => setTimeout(resolve, 1))
      const inner = runWithAuthContext({ ...alice, uid: 'nested' }, () => currentUser().uid)
      return [currentUser().uid, inner]
    })
    expect(seen).toEqual(['alice', 'nested'])
    expect(currentUser().uid).toBe('local')
  })
})
