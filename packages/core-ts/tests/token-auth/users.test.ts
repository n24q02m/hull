import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, parse } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import { hashToken } from '../../src/token-auth/tokens.js'
import {
  ensurePathAllowed,
  findUserByToken,
  loadUsers,
  PathNotAllowedError,
  UsersError
} from '../../src/token-auth/users.js'

let dir: string
let aliceHash: string
let bobHash: string
let counter = 0

function write(content: string | Buffer): string {
  counter += 1
  const path = join(dir, `users-${counter}.toml`)
  writeFileSync(path, content)
  return path
}

beforeAll(async () => {
  dir = realpathSync.native(mkdtempSync(join(tmpdir(), 'hull-users-')))
  aliceHash = await hashToken('alice-token')
  bobHash = await hashToken('bob-token')
})

afterAll(() => {
  rmSync(dir, { recursive: true, force: true })
})

function usersToml(): string {
  return [
    '[users.alice]',
    `token_hash = "${aliceHash}"`,
    'enabled = true',
    'namespace = "alice"',
    'allowed_roots = ["/data/alice"]',
    '',
    '[users.alice.limits]',
    'rpm = 2',
    '',
    '[users.bob]',
    `token_hash = "${bobHash}"`,
    'enabled = false',
    'namespace = "bob"',
    ''
  ].join('\n')
}

describe('loadUsers', () => {
  it('parses the table like core-py', () => {
    const users = loadUsers(write(usersToml()))
    expect([...users.keys()]).toEqual(['alice', 'bob'])
    const alice = users.get('alice')
    expect(alice).toEqual({
      uid: 'alice',
      tokenHash: aliceHash,
      enabled: true,
      namespace: 'alice',
      allowedRoots: ['/data/alice'],
      limits: { rpm: 2 }
    })
    expect(Object.isFrozen(alice)).toBe(true)
    expect(users.get('bob')?.enabled).toBe(false)
    expect(users.get('bob')?.limits.rpm).toBeNull()
    expect(users.get('bob')?.allowedRoots).toEqual([])
  })

  it('defaults enabled to true', () => {
    const users = loadUsers(write(`[users.carol]\ntoken_hash = "${aliceHash}"\nnamespace = "c"\n`))
    expect(users.get('carol')?.enabled).toBe(true)
  })

  it('reports a missing file or a directory as not found', () => {
    const absent = join(dir, 'absent.toml')
    expect(() => loadUsers(absent)).toThrow(new UsersError(`users file not found: ${absent}`))
    expect(() => loadUsers(dir)).toThrow(`users file not found: ${dir}`)
  })

  it('rejects invalid UTF-8', () => {
    const path = write(Buffer.from([0x5b, 0x75, 0xff, 0x5d]))
    expect(() => loadUsers(path)).toThrow(`users file is not valid UTF-8: ${path}`)
  })

  it('rejects invalid TOML with the path', () => {
    const path = write('not [valid toml')
    const err = (() => {
      try {
        loadUsers(path)
      } catch (e) {
        return e as UsersError
      }
    })()
    expect(err).toBeInstanceOf(UsersError)
    expect(err?.message.startsWith(`users file is not valid TOML: ${path}: `)).toBe(true)
    expect(err?.cause).toBeDefined()
  })

  it.each([['[other]\nx = 1\n'], ['users = 1\n'], ['[users]\n'], ['']])(
    'requires a non-empty [users] table (%j)',
    (body) => {
      expect(() => loadUsers(write(body))).toThrow('users file must contain a non-empty [users] table')
    }
  )

  const cases: [string, () => string, string][] = [
    ['entry-not-a-table', () => '[users]\nalice = "scalar"\n', "user 'alice': expected a table"],
    ['entry-array', () => '[users]\nalice = [1]\n', "user 'alice': expected a table"],
    [
      'token-hash-absent',
      () => '[users.alice]\nnamespace = "alice"\n',
      "user 'alice': token_hash missing or not a scrypt encoding (mint one with `hull token hash`)"
    ],
    ['token-hash-not-scrypt', () => '[users.alice]\ntoken_hash = "plaintext"\n', 'not a scrypt encoding'],
    ['token-hash-not-string', () => '[users.alice]\ntoken_hash = 5\n', 'token_hash missing'],
    [
      'namespace-missing',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\n`,
      "user 'alice': namespace must be a non-empty string"
    ],
    [
      'namespace-empty',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = ""\n`,
      'namespace must be a non-empty string'
    ],
    [
      'enabled-not-bool',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = "a"\nenabled = "yes"\n`,
      "user 'alice': enabled must be a boolean"
    ],
    [
      'roots-not-strings',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = "a"\nallowed_roots = [1]\n`,
      "user 'alice': allowed_roots must be a list of strings"
    ],
    [
      'roots-not-list',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = "a"\nallowed_roots = "/x"\n`,
      'allowed_roots must be a list of strings'
    ],
    [
      'limits-not-a-table',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = "a"\nlimits = "rpm=60"\n`,
      "user 'alice': limits must be a table"
    ],
    [
      'rpm-not-positive',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = "a"\n[users.alice.limits]\nrpm = 0\n`,
      "user 'alice': limits.rpm must be a positive integer"
    ],
    [
      'rpm-float',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = "a"\n[users.alice.limits]\nrpm = 1.5\n`,
      'limits.rpm must be a positive integer'
    ],
    [
      'rpm-bool',
      () => `[users.alice]\ntoken_hash = "${aliceHash}"\nnamespace = "a"\n[users.alice.limits]\nrpm = true\n`,
      'limits.rpm must be a positive integer'
    ]
  ]

  it.each(cases)('validates entries: %s', (_id, body, message) => {
    expect(() => loadUsers(write(body()))).toThrow(UsersError)
    expect(() => loadUsers(write(body()))).toThrow(message)
  })

  it('repr-quotes a uid containing a quote like core-py (user "o\'neil")', () => {
    expect(() => loadUsers(write(`[users."o'neil"]\nnamespace = "x"\n`))).toThrow(`user "o'neil": token_hash missing`)
  })
})

describe('findUserByToken', () => {
  it('returns the matching row or null', async () => {
    const users = loadUsers(write(usersToml()))
    expect(await findUserByToken(users, 'alice-token')).toBe(users.get('alice'))
    expect(await findUserByToken(users, 'bob-token')).toBe(users.get('bob'))
    expect(await findUserByToken(users, 'nope')).toBeNull()
  })
})

describe('ensurePathAllowed', () => {
  it('allows the root and paths below it, denies outside', () => {
    const root = join(dir, 'roots', 'alice')
    mkdirSync(root, { recursive: true })
    const user = { uid: 'alice', allowedRoots: [root] }
    expect(() => ensurePathAllowed(user, join(root, 'file.txt'))).not.toThrow()
    expect(() => ensurePathAllowed(user, root)).not.toThrow()
    expect(() => ensurePathAllowed(user, join(root, 'not', 'yet', 'created.txt'))).not.toThrow()
    const outside = join(dir, 'outside.txt')
    expect(() => ensurePathAllowed(user, outside)).toThrow(PathNotAllowedError)
    expect(() => ensurePathAllowed(user, outside)).toThrow(`path ${outside} is outside allowed roots for user 'alice'`)
  })

  it('does not treat a sibling with the same prefix as inside', () => {
    const root = join(dir, 'prefix', 'alice')
    mkdirSync(root, { recursive: true })
    expect(() => ensurePathAllowed({ uid: 'alice', allowedRoots: [root] }, `${root}2`)).toThrow(PathNotAllowedError)
  })

  it('normalizes dot segments before matching', () => {
    const root = join(dir, 'dots', 'alice')
    mkdirSync(root, { recursive: true })
    const traversal = join(root, '..', 'bob', 'secret.txt')
    expect(() => ensurePathAllowed({ uid: 'alice', allowedRoots: [root] }, traversal)).toThrow(PathNotAllowedError)
  })

  it('treats a filesystem root (trailing separator) as containing every path under it', () => {
    const fsRoot = parse(dir).root
    expect(() => ensurePathAllowed({ uid: 'alice', allowedRoots: [fsRoot] }, join(dir, 'anything'))).not.toThrow()
  })

  it('grants nothing for empty allowed roots', () => {
    const err = (() => {
      try {
        ensurePathAllowed({ uid: 'nobody', allowedRoots: [] }, dir)
      } catch (e) {
        return e as PathNotAllowedError
      }
    })()
    expect(err).toBeInstanceOf(PathNotAllowedError)
    expect(err?.code).toBe('EACCES')
    expect(err?.name).toBe('PathNotAllowedError')
  })

  it('follows symlinks, so a link inside the root to the outside is denied', () => {
    const root = join(dir, 'links', 'alice')
    const outside = join(dir, 'links', 'elsewhere')
    mkdirSync(root, { recursive: true })
    mkdirSync(outside, { recursive: true })
    const link = join(root, 'escape')
    // A junction needs no privilege on Windows; elsewhere it is a dir symlink.
    symlinkSync(outside, link, process.platform === 'win32' ? 'junction' : 'dir')
    const user = { uid: 'alice', allowedRoots: [root] }
    expect(() => ensurePathAllowed(user, join(link, 'file.txt'))).toThrow(
      `path ${join(outside, 'file.txt')} is outside allowed roots for user 'alice'`
    )
    // A root reached through a link matches its resolved target.
    expect(() => ensurePathAllowed({ uid: 'alice', allowedRoots: [link] }, join(outside, 'f'))).not.toThrow()
  })

  it.runIf(process.platform === 'win32')('matches case-insensitively on Windows', () => {
    const root = join(dir, 'Case', 'Alice')
    mkdirSync(root, { recursive: true })
    expect(() => ensurePathAllowed({ uid: 'alice', allowedRoots: [root] }, join(root.toUpperCase(), 'x'))).not.toThrow()
  })

  it.runIf(process.platform !== 'win32')('matches case-sensitively on POSIX', () => {
    const root = join(dir, 'case', 'alice')
    mkdirSync(root, { recursive: true })
    expect(() => ensurePathAllowed({ uid: 'alice', allowedRoots: [root] }, join(dir, 'CASE', 'alice', 'x'))).toThrow(
      PathNotAllowedError
    )
  })
})
