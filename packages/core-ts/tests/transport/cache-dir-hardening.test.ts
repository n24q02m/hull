import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { atomicWrite } from '../../src/transport/cache.js'

const itPosix = process.platform !== 'win32' ? it : it.skip

describe('tools-cache directory hardening (TOCTOU)', () => {
  let testRoot: string

  beforeEach(() => {
    testRoot = mkdtempSync(join(tmpdir(), 'cache-hardening-'))
  })

  afterEach(() => {
    rmSync(testRoot, { recursive: true, force: true })
  })

  const dirMode = (p: string): number => statSync(p).mode & 0o777

  itPosix('atomicWrite tightens a pre-existing 0o777 cache directory to 0o700', () => {
    const dir = join(testRoot, 'cache')
    mkdirSync(dir, { recursive: true })
    chmodSync(dir, 0o777)

    atomicWrite(join(dir, 'f.json'), '{"ok":true}')

    expect(dirMode(dir)).toBe(0o700)
    expect(readFileSync(join(dir, 'f.json'), 'utf-8')).toBe('{"ok":true}')
  })

  it('atomicWrite re-creates a deleted cache directory on the next write', () => {
    const dir = join(testRoot, 'cache2')
    atomicWrite(join(dir, 'f.json'), '{"v":1}')
    rmSync(dir, { recursive: true, force: true })
    atomicWrite(join(dir, 'f.json'), '{"v":2}')
    expect(readFileSync(join(dir, 'f.json'), 'utf-8')).toBe('{"v":2}')
  })
})
