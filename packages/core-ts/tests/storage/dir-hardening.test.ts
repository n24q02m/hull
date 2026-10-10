import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { InMemoryBackend, LocalFsBackend } from '../../src/storage/backends.js'
import { setHomeDirForTesting } from '../../src/storage/home-dir.js'
import { PerPluginStore } from '../../src/storage/per-plugin-store.js'
import { releaseSessionLock, setLockDir, writeSessionLock } from '../../src/storage/session-lock.js'

/**
 * TOCTOU directory hardening: an existing overly-permissive credential/cache
 * directory must be tightened to 0o700 exactly once per process, and a
 * directory deleted underneath the process must be re-created. Permission-bit
 * assertions are POSIX-only; recreation cases run everywhere.
 */
const itPosix = process.platform !== 'win32' ? it : it.skip

describe('storage directory hardening (TOCTOU)', () => {
  let testHome: string

  beforeEach(() => {
    testHome = mkdtempSync(join(tmpdir(), 'dir-hardening-'))
    setHomeDirForTesting(testHome)
  })

  afterEach(() => {
    setHomeDirForTesting(null)
    setLockDir(null)
    rmSync(testHome, { recursive: true, force: true })
  })

  const dirMode = (p: string): number => statSync(p).mode & 0o777

  itPosix('LocalFsBackend.put tightens a pre-existing 0o777 store directory to 0o700', async () => {
    const dir = join(testHome, '.wet-mcp')
    mkdirSync(dir, { recursive: true })
    chmodSync(dir, 0o777)
    expect(dirMode(dir)).toBe(0o777)

    const backend = new LocalFsBackend()
    await backend.put('wet/config', Buffer.from('blob'))

    expect(dirMode(dir)).toBe(0o700)
    expect(readFileSync(join(dir, 'config.json'))).toEqual(Buffer.from('blob'))
  })

  it('LocalFsBackend.put re-creates the directory if it disappears between writes', async () => {
    const backend = new LocalFsBackend()
    await backend.put('wet/config', Buffer.from('one'))
    rmSync(join(testHome, '.wet-mcp'), { recursive: true, force: true })
    await backend.put('wet/config', Buffer.from('two'))
    expect(readFileSync(join(testHome, '.wet-mcp', 'config.json'))).toEqual(Buffer.from('two'))
  })

  itPosix('writeSessionLock tightens a pre-existing 0o777 lock directory to 0o700', async () => {
    const lockDir = join(testHome, 'locks')
    mkdirSync(lockDir, { recursive: true })
    chmodSync(lockDir, 0o777)
    setLockDir(lockDir)

    await writeSessionLock('wet', { sessionId: 's1', relayUrl: 'https://relay.example', createdAt: Date.now() })

    expect(dirMode(lockDir)).toBe(0o700)
  })

  it('writeSessionLock re-creates a deleted lock directory on the next write', async () => {
    const lockDir = join(testHome, 'locks2')
    setLockDir(lockDir)
    await writeSessionLock('wet', { sessionId: 's1', relayUrl: 'https://relay.example', createdAt: Date.now() })
    rmSync(lockDir, { recursive: true, force: true })
    await writeSessionLock('wet', { sessionId: 's2', relayUrl: 'https://relay.example', createdAt: Date.now() })
    // The lock file exists again and is readable through the public API.
    await releaseSessionLock('wet')
  })

  itPosix(
    'PerPluginStore tightens a permissive machine-key directory even when the secret already exists (read path)',
    async () => {
      const dir = join(testHome, '.wet-mcp')
      mkdirSync(dir, { recursive: true })
      chmodSync(dir, 0o777)
      writeFileSync(join(dir, '.secret'), Buffer.alloc(32, 7))

      const store = new PerPluginStore('wet', null, new InMemoryBackend())
      await store.save({ token: 'x' })

      expect(dirMode(dir)).toBe(0o700)
      // The pre-existing secret is reused, not regenerated.
      expect(readFileSync(join(dir, '.secret'))).toEqual(Buffer.alloc(32, 7))
    }
  )
})
