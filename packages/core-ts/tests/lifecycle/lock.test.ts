import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  DEFAULT_LOCK_TTL_HOURS,
  refreshLockTimestamp,
  sweepStaleLocks,
  writeLockFile
} from '../../src/lifecycle/lock.js'

let tmp: string

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), 'mcp-core-lock-'))
})

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true })
})

function writeLockWithAge(path: string, pid: number, port: number, ageHours = 0): void {
  const ts = new Date(Date.now() - ageHours * 3600 * 1000).toISOString()
  writeFileSync(path, `${pid}\n${port}\nproxy-token\n${ts}\n`)
}

describe('refreshLockTimestamp', () => {
  it('updates timestamp in 4-line lock', () => {
    const path = join(tmp, 'demo-1234.lock')
    writeLockWithAge(path, 9999, 1234, 10)
    const beforeRaw = readFileSync(path, 'utf-8')
    const beforeTs = beforeRaw.split('\n')[3]

    refreshLockTimestamp(path)

    const afterRaw = readFileSync(path, 'utf-8')
    const afterTs = afterRaw.split('\n')[3]
    expect(new Date(afterTs).getTime()).toBeGreaterThan(new Date(beforeTs).getTime())
  })

  it('silent no-op on legacy 3-line lock', () => {
    const path = join(tmp, 'demo-1234.lock')
    writeFileSync(path, '9999\n1234\ntoken\n')
    expect(() => refreshLockTimestamp(path)).not.toThrow()
    // Untouched
    const content = readFileSync(path, 'utf-8')
    expect(content.startsWith('9999\n1234\ntoken\n')).toBe(true)
  })

  it('silent no-op on missing file', () => {
    const path = join(tmp, 'missing.lock')
    expect(() => refreshLockTimestamp(path)).not.toThrow()
  })

  it('silent no-op on a timestamp that does not parse', () => {
    const path = join(tmp, 'demo-1235.lock')
    writeFileSync(path, '9999\n1235\ntoken\nnot-a-date\n')

    expect(() => refreshLockTimestamp(path)).not.toThrow()
    expect(readFileSync(path, 'utf-8')).toBe('9999\n1235\ntoken\nnot-a-date\n')
  })

  it('silent no-op when the path exists but cannot be read', () => {
    // A directory exists, and reading it as a file fails with EISDIR.
    const path = join(tmp, 'demo-1236.lock')
    mkdirSync(path)

    expect(() => refreshLockTimestamp(path)).not.toThrow()
    expect(lstatSync(path).isDirectory()).toBe(true)
  })
})

describe('sweepStaleLocks', () => {
  it('removes expired lock', () => {
    const p = join(tmp, 'demo-2001.lock')
    writeLockWithAge(p, process.pid, 2001, 25)

    expect(sweepStaleLocks('demo', 24, tmp)).toBe(1)
  })

  it('keeps lock within TTL', () => {
    const p = join(tmp, 'demo-3001.lock')
    writeLockWithAge(p, process.pid, 3001, 1)

    expect(sweepStaleLocks('demo', 24, tmp)).toBe(0)
  })

  it('only targets named server', () => {
    const pDemo = join(tmp, 'demo-4001.lock')
    const pOther = join(tmp, 'other-server-4002.lock')
    writeLockWithAge(pDemo, 999999, 4001, 25)
    writeLockWithAge(pOther, 999999, 4002, 25)

    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, tmp)).toBe(1)
  })

  it('removes legacy 3-line locks', () => {
    const p = join(tmp, 'demo-5001.lock')
    writeFileSync(p, '9999\n5001\ntoken\n')

    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, tmp)).toBe(1)
  })

  it('removes corrupted lock', () => {
    const p = join(tmp, 'demo-6001.lock')
    writeFileSync(p, 'NOT_AN_INT\n1234\nt\n2026-04-28T19:00:00Z\n')

    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, tmp)).toBe(1)
  })

  it('removes a lock whose timestamp does not parse', () => {
    const p = join(tmp, 'demo-6002.lock')
    writeFileSync(p, '9999\n6002\nt\nnot-a-date\n')

    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, tmp)).toBe(1)
    expect(existsSync(p)).toBe(false)
  })

  it('removes a lock entry that cannot be read', () => {
    // A link to a directory fails to read as a file, but the link itself unlinks.
    // 'junction' needs no privilege on Windows; other platforms ignore the type.
    const target = join(tmp, 'elsewhere')
    mkdirSync(target)
    const p = join(tmp, 'demo-6003.lock')
    symlinkSync(target, p, 'junction')

    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, tmp)).toBe(1)
    expect(existsSync(p)).toBe(false)
    expect(existsSync(target)).toBe(true)
  })

  it('skips an unreadable lock entry that cannot be removed either', () => {
    // A real directory: reading fails with EISDIR and unlinking a directory fails too.
    const p = join(tmp, 'demo-6004.lock')
    mkdirSync(p)
    writeLockWithAge(join(tmp, 'demo-6005.lock'), 999999, 6005, 25)

    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, tmp)).toBe(1)
    expect(lstatSync(p).isDirectory()).toBe(true)
  })

  it('returns zero when the locks root is not a directory', () => {
    const root = join(tmp, 'not-a-dir')
    writeFileSync(root, '')

    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, root)).toBe(0)
  })

  it('returns zero when locks dir missing', () => {
    expect(sweepStaleLocks('demo', DEFAULT_LOCK_TTL_HOURS, join(tmp, 'no-such-dir'))).toBe(0)
  })

  it('clears 11-stale-lock pile-up (regression for 2026-04-28 wet-mcp)', () => {
    for (let i = 0; i < 11; i++) {
      writeLockWithAge(join(tmp, `wet-mcp-${50000 + i}.lock`), 999990 + i, 50000 + i, 25)
    }
    expect(sweepStaleLocks('wet-mcp', DEFAULT_LOCK_TTL_HOURS, tmp)).toBe(11)
  })

  it('default TTL is 24h', () => {
    expect(DEFAULT_LOCK_TTL_HOURS).toBe(24)
  })
})

describe('writeLockFile', () => {
  it('writes 4-line padded payload', () => {
    const path = writeLockFile('demo', 12345, 'tok', tmp)
    const raw = readFileSync(path, 'utf-8')
    const lines = raw.replace(/\n+$/, '').replace(/\s+$/, '').split('\n')
    expect(lines[0]).toBe(String(process.pid))
    expect(lines[1]).toBe('12345')
    expect(lines[2]).toBe('tok')
    expect(new Date(lines[3])).toBeInstanceOf(Date)
  })
})
