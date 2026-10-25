import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import {
  AuthConfigError,
  Authenticator,
  defaultUsersFile,
  HTTP_AUTH_MODES,
  type HttpAuthConfig,
  parseBearer
} from '../../src/token-auth/authenticator.js'
import { SlidingWindowLimiter } from '../../src/token-auth/limiter.js'
import { hashToken } from '../../src/token-auth/tokens.js'
import { loadUsers, type TokenUser, UsersError } from '../../src/token-auth/users.js'

let dir: string
let sharedHash: string
let aliceHash: string
let bobHash: string
let carolHash: string
let usersPath: string

beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), 'hull-authn-'))
  ;[sharedHash, aliceHash, bobHash, carolHash] = await Promise.all([
    hashToken('shared-token'),
    hashToken('alice-token'),
    hashToken('bob-token'),
    hashToken('carol-token')
  ])
  usersPath = join(dir, 'users.toml')
  writeFileSync(
    usersPath,
    [
      '[users.alice]',
      `token_hash = "${aliceHash}"`,
      'namespace = "ns-alice"',
      'allowed_roots = ["/data/alice"]',
      '[users.alice.limits]',
      'rpm = 2',
      '',
      '[users.bob]',
      `token_hash = "${bobHash}"`,
      'namespace = "ns-bob"',
      'enabled = false',
      '',
      '[users.carol]',
      `token_hash = "${carolHash}"`,
      'namespace = "ns-carol"',
      ''
    ].join('\n')
  )
})

afterAll(() => {
  rmSync(dir, { recursive: true, force: true })
})

describe('parseBearer (core-py Authenticator._bearer)', () => {
  it.each([
    [undefined, null],
    [null, null],
    ['', null],
    ['Bearer', null],
    ['Bearer ', null],
    ['Bearer    ', null],
    ['Basic abc', null],
    ['Bearerabc', null],
    ['Bearer abc', 'abc'],
    ['bearer abc', 'abc'],
    ['BEARER abc', 'abc'],
    ['Bearer   abc  ', 'abc'],
    ['Bearer a b', 'a b'],
    ['Bearer \tabc\n', 'abc'],
    [' Bearer abc', null]
  ])('parseBearer(%j) -> %j', (input, expected) => {
    expect(parseBearer(input)).toBe(expected)
  })
})

describe('HTTP_AUTH_MODES', () => {
  it('lists the modes in core-py order and is frozen', () => {
    expect(HTTP_AUTH_MODES).toEqual(['no-auth', 'token', 'multi'])
    expect(Object.isFrozen(HTTP_AUTH_MODES)).toBe(true)
  })
})

describe('no-auth mode', () => {
  it('admits every request as the local context without a header', async () => {
    const authn = new Authenticator({ mode: 'no-auth' })
    expect(authn.mode).toBe('no-auth')
    expect(authn.userCount).toBe(0)
    for (const authz of [undefined, 'Bearer anything', 'garbage']) {
      const out = await authn.authenticate(authz)
      expect(out.ok).toBe(true)
      expect(out.status).toBeNull()
      expect(out.context).toMatchObject({ uid: 'local', namespace: 'default', mode: 'no-auth' })
    }
  })
})

describe('token mode', () => {
  it('401s without a bearer and with a wrong token, admits the right one', async () => {
    const authn = new Authenticator({ mode: 'token', tokenHash: sharedHash })
    expect(authn.mode).toBe('token')
    expect(authn.userCount).toBe(0)

    const missing = await authn.authenticate(undefined)
    expect(missing).toEqual({
      context: null,
      status: 401,
      detail: 'missing or malformed Authorization header',
      ok: false
    })
    expect(await authn.authenticate('Basic shared-token')).toMatchObject({ status: 401 })

    const wrong = await authn.authenticate('Bearer nope')
    expect(wrong).toMatchObject({ status: 401, detail: 'invalid token', ok: false, context: null })

    const good = await authn.authenticate('Bearer shared-token')
    expect(good.ok).toBe(true)
    expect(good.detail).toBe('')
    expect(good.context).toEqual({
      uid: 'shared',
      namespace: 'default',
      mode: 'token',
      allowedRoots: [],
      rpm: null,
      disabled: false,
      attributes: {}
    })
    expect(Object.isFrozen(good.context)).toBe(true)
    expect(Object.isFrozen(good)).toBe(true)
  })

  it('applies the shared rpm cap', async () => {
    let now = 0
    const limiter = new SlidingWindowLimiter()
    const original = limiter.check.bind(limiter)
    limiter.check = (key, rpm) => original(key, rpm, now)
    const authn = new Authenticator({ mode: 'token', tokenHash: sharedHash, rpm: 2 }, { limiter })
    expect(authn.limiter).toBe(limiter)
    expect((await authn.authenticate('Bearer shared-token')).ok).toBe(true)
    expect((await authn.authenticate('Bearer shared-token')).ok).toBe(true)
    expect(await authn.authenticate('Bearer shared-token')).toMatchObject({
      status: 429,
      detail: 'rate limit exceeded',
      ok: false
    })
    now = 61
    expect((await authn.authenticate('Bearer shared-token')).context?.rpm).toBe(2)
  })

  it.each<[string, HttpAuthConfig, string]>([
    ['empty hash', { mode: 'token', tokenHash: '' }, '[server] auth = "token" requires a non-empty token_hash'],
    [
      'non-string hash',
      { mode: 'token', tokenHash: undefined as unknown as string },
      '[server] auth = "token" requires a non-empty token_hash'
    ],
    [
      'malformed hash',
      { mode: 'token', tokenHash: 'plaintext' },
      '[server] token_hash is not a scrypt encoding (mint one with `hull token hash`)'
    ],
    ['rpm 0', { mode: 'token', tokenHash: 'PLACEHOLDER', rpm: 0 }, '[server] rpm must be a positive integer'],
    ['rpm -1', { mode: 'token', tokenHash: 'PLACEHOLDER', rpm: -1 }, '[server] rpm must be a positive integer'],
    ['rpm 1.5', { mode: 'token', tokenHash: 'PLACEHOLDER', rpm: 1.5 }, '[server] rpm must be a positive integer'],
    [
      'rpm string',
      { mode: 'token', tokenHash: 'PLACEHOLDER', rpm: '5' as unknown as number },
      '[server] rpm must be a positive integer'
    ]
  ])('rejects a bad config at construction: %s', (_id, config, message) => {
    const resolved =
      'tokenHash' in config && config.tokenHash === 'PLACEHOLDER' ? { ...config, tokenHash: sharedHash } : config
    expect(() => new Authenticator(resolved)).toThrow(new AuthConfigError(message))
  })

  it('accepts a null rpm as unlimited', async () => {
    const authn = new Authenticator({ mode: 'token', tokenHash: sharedHash, rpm: null })
    expect((await authn.authenticate('Bearer shared-token')).context?.rpm).toBeNull()
  })
})

describe('multi mode', () => {
  it('maps each token to its own user and namespace without leaking across users', async () => {
    const authn = new Authenticator({ mode: 'multi', usersFile: usersPath })
    expect(authn.mode).toBe('multi')
    expect(authn.userCount).toBe(3)

    const alice = await authn.authenticate('Bearer alice-token')
    expect(alice.ok).toBe(true)
    expect(alice.context).toEqual({
      uid: 'alice',
      namespace: 'ns-alice',
      mode: 'multi',
      allowedRoots: ['/data/alice'],
      rpm: 2,
      disabled: false,
      attributes: {}
    })
    const carol = await authn.authenticate('Bearer carol-token')
    expect(carol.context).toMatchObject({ uid: 'carol', namespace: 'ns-carol', allowedRoots: [], rpm: null })

    expect(await authn.authenticate('Bearer shared-token')).toMatchObject({ status: 401, detail: 'invalid token' })
    expect(await authn.authenticate(undefined)).toMatchObject({
      status: 401,
      detail: 'missing or malformed Authorization header'
    })
  })

  it('403s a disabled user', async () => {
    const authn = new Authenticator({ mode: 'multi', usersFile: usersPath })
    expect(await authn.authenticate('Bearer bob-token')).toEqual({
      context: null,
      status: 403,
      detail: "user 'bob' is disabled",
      ok: false
    })
  })

  it('rate limits per user, not globally', async () => {
    const authn = new Authenticator({ mode: 'multi', usersFile: usersPath })
    expect((await authn.authenticate('Bearer alice-token')).ok).toBe(true)
    expect((await authn.authenticate('Bearer alice-token')).ok).toBe(true)
    expect(await authn.authenticate('Bearer alice-token')).toMatchObject({ status: 429, detail: 'rate limit exceeded' })
    // carol has no limit and is unaffected by alice's window.
    for (let i = 0; i < 5; i += 1) expect((await authn.authenticate('Bearer carol-token')).ok).toBe(true)
  })

  it('accepts a pre-loaded users map', async () => {
    const users = loadUsers(usersPath)
    const authn = new Authenticator({ mode: 'multi', users })
    expect(authn.userCount).toBe(3)
    expect((await authn.authenticate('Bearer carol-token')).context?.uid).toBe('carol')
  })

  it('validates every hash of a pre-loaded users map', () => {
    const bad: TokenUser = {
      uid: "o'neil",
      tokenHash: 'plaintext',
      enabled: true,
      namespace: 'x',
      allowedRoots: [],
      limits: { rpm: null }
    }
    expect(() => new Authenticator({ mode: 'multi', users: new Map([["o'neil", bad]]) })).toThrow(
      new AuthConfigError(
        `user "o'neil": token_hash missing or not a scrypt encoding (mint one with \`hull token hash\`)`
      )
    )
  })

  it('propagates users.toml errors at construction', () => {
    const absent = join(dir, 'absent.toml')
    expect(() => new Authenticator({ mode: 'multi', usersFile: absent })).toThrow(UsersError)
    expect(() => new Authenticator({ mode: 'multi', usersFile: absent })).toThrow(`users file not found: ${absent}`)
  })

  it('expands a leading ~ in usersFile', () => {
    const rel = join('.hull-test-absent-dir', 'users.toml')
    for (const usersFile of [`~/${rel}`, `~\\${rel}`]) {
      expect(() => new Authenticator({ mode: 'multi', usersFile })).toThrow(
        `users file not found: ${join(homedir(), rel)}`
      )
    }
    expect(() => new Authenticator({ mode: 'multi', usersFile: '~' })).toThrow(`users file not found: ${homedir()}`)
  })

  it('defaults to ~/.hull/users.toml like core-py', () => {
    expect(defaultUsersFile()).toBe(join(homedir(), '.hull', 'users.toml'))
  })

  it.skipIf(existsSync(join(homedir(), '.hull', 'users.toml')))('reads the default users file when unset', () => {
    expect(() => new Authenticator({ mode: 'multi' })).toThrow(`users file not found: ${defaultUsersFile()}`)
  })
})

describe('unknown mode', () => {
  it.each([
    ['oauth', "[server] auth must be one of ('no-auth', 'token', 'multi'), got 'oauth'"],
    [undefined, "[server] auth must be one of ('no-auth', 'token', 'multi'), got undefined"],
    [7, "[server] auth must be one of ('no-auth', 'token', 'multi'), got 7"]
  ])('rejects %j', (mode, message) => {
    const err = (() => {
      try {
        new Authenticator({ mode } as unknown as HttpAuthConfig)
      } catch (e) {
        return e as AuthConfigError
      }
    })()
    expect(err).toBeInstanceOf(AuthConfigError)
    expect(err?.name).toBe('AuthConfigError')
    expect(err?.message).toBe(message)
  })
})
