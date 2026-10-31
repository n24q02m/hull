import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { backendFromEnv, InMemoryBackend, LocalFsBackend } from '../../src/storage/backends.js'
import { setHomeDirForTesting } from '../../src/storage/home-dir.js'
import { PerPluginStore } from '../../src/storage/per-plugin-store.js'

// Mocking node:fs/promises for the atomic-write rename-failure case (Task 4).
// Only `rename` is intercepted, gated on the destination path so unrelated
// gets/puts in this file keep using the real filesystem.
vi.mock('node:fs/promises', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs/promises')>()
  return {
    ...actual,
    rename: vi.fn((oldPath, newPath) => {
      if (typeof newPath === 'string' && newPath.includes('rename-error-trigger')) {
        return Promise.reject(new Error('simulated rename failure'))
      }
      return actual.rename(oldPath, newPath)
    })
  }
})

describe('InMemoryBackend', () => {
  it('put/get/delete roundtrip (absent get returns null)', async () => {
    const backend = new InMemoryBackend()
    expect(await backend.get('k')).toBeNull()
    await backend.put('k', Buffer.from([0, 1, 2]))
    expect(await backend.get('k')).toEqual(Buffer.from([0, 1, 2]))
    await backend.delete('k')
    expect(await backend.get('k')).toBeNull()
  })

  it('delete is idempotent', async () => {
    const backend = new InMemoryBackend()
    await backend.put('k', Buffer.from('data'))
    await backend.delete('k')
    expect(await backend.get('k')).toBeNull()
    await expect(backend.delete('k')).resolves.toBeUndefined()
  })
})

describe('LocalFsBackend', () => {
  let testHome: string

  beforeEach(() => {
    testHome = mkdtempSync(join(tmpdir(), 'backends-test-'))
    setHomeDirForTesting(testHome)
  })

  afterEach(() => {
    setHomeDirForTesting(null)
    rmSync(testHome, { recursive: true, force: true })
  })

  it('writes config blob under <home>/.<plugin>/config.json', async () => {
    const backend = new LocalFsBackend()
    expect(backend.isDefaultLayout).toBe(true)
    await backend.put('wet/config', Buffer.from('blob'))
    const onDisk = readFileSync(join(testHome, '.wet', 'config.json'))
    expect(onDisk).toEqual(Buffer.from('blob'))
    expect(existsSync(join(testHome, '.wet-mcp'))).toBe(false)
  })

  it('honours a custom baseDir and reports a non-default layout', async () => {
    const custom = join(testHome, 'legacy-root')
    const backend = new LocalFsBackend({ baseDir: (plugin) => join(custom, `.${plugin}-mcp`) })
    expect(backend.isDefaultLayout).toBe(false)
    await backend.put('wet/subs/u1/config', Buffer.from('blob'))
    expect(readFileSync(join(custom, '.wet-mcp', 'subs', 'u1', 'config.json'))).toEqual(Buffer.from('blob'))
    expect(await backend.get('wet/subs/u1/config')).toEqual(Buffer.from('blob'))
    await backend.delete('wet/subs/u1/config')
    expect(await backend.get('wet/subs/u1/config')).toBeNull()
    // Traversal checks still apply under a custom base.
    await expect(backend.get('wet/subs/../config')).rejects.toThrow()
  })

  it('sub path get/delete roundtrip', async () => {
    const backend = new LocalFsBackend()
    await backend.put('wet/subs/u1/config', Buffer.from('blob'))
    const onDisk = readFileSync(join(testHome, '.wet', 'subs', 'u1', 'config.json'))
    expect(onDisk).toEqual(Buffer.from('blob'))
    expect(await backend.get('wet/subs/u1/config')).toEqual(Buffer.from('blob'))
    await backend.delete('wet/subs/u1/config')
    expect(await backend.get('wet/subs/u1/config')).toBeNull()
  })

  it('get returns null for absent key', async () => {
    const backend = new LocalFsBackend()
    expect(await backend.get('wet/config')).toBeNull()
  })

  it('rejects traversal in sub component', async () => {
    const backend = new LocalFsBackend()
    await expect(backend.put('wet/subs/../config', Buffer.from('x'))).rejects.toThrow()
    await expect(backend.get('wet/subs/../config')).rejects.toThrow()
    await expect(backend.delete('wet/subs/../config')).rejects.toThrow()
  })

  it('rejects traversal in plugin component', async () => {
    const backend = new LocalFsBackend()
    await expect(backend.put('../config', Buffer.from('x'))).rejects.toThrow()
  })

  it('rejects keys that are not <plugin>/config or <plugin>/subs/<sub>/config', async () => {
    const backend = new LocalFsBackend()
    await expect(backend.get('wet')).rejects.toThrow(/Invalid backend key: wet/)
    await expect(backend.get('wet/other')).rejects.toThrow(/Invalid backend key/)
    await expect(backend.get('wet/subs/u1/other')).rejects.toThrow(/Invalid backend key/)
  })

  it('get and delete rethrow filesystem errors other than ENOENT', async () => {
    const backend = new LocalFsBackend()
    // A directory where the blob file should be: read/unlink fail with
    // EISDIR/EPERM, which must surface instead of reading as "absent".
    mkdirSync(join(testHome, '.wet', 'config.json'), { recursive: true })
    await expect(backend.get('wet/config')).rejects.toThrow()
    await expect(backend.delete('wet/config')).rejects.toThrow()
  })

  it('allows a legitimate dot in sub', async () => {
    const backend = new LocalFsBackend()
    await backend.put('wet/subs/u.1/config', Buffer.from('blob'))
    expect(await backend.get('wet/subs/u.1/config')).toEqual(Buffer.from('blob'))
  })

  it('put leaves no tmp debris and keeps old blob on rename failure', async () => {
    const backend = new LocalFsBackend()
    // Seed the "old" blob directly (bypassing put/rename) so the mocked
    // rename failure below is the only thing exercising the atomic-write path.
    const configDir = join(testHome, '.rename-error-trigger')
    mkdirSync(configDir, { recursive: true })
    writeFileSync(join(configDir, 'config.json'), 'old-blob')

    await expect(backend.put('rename-error-trigger/config', Buffer.from('new-blob'))).rejects.toThrow(
      'simulated rename failure'
    )

    // Old content survives the failed rename.
    expect(readFileSync(join(configDir, 'config.json'), 'utf-8')).toBe('old-blob')
    // No leftover .tmp file in the directory.
    expect(readdirSync(configDir).some((entry) => entry.endsWith('.tmp'))).toBe(false)

    // A successful put still writes the new content.
    await backend.put('demo/config', Buffer.from('new-content'))
    expect(readFileSync(join(testHome, '.demo', 'config.json'), 'utf-8')).toBe('new-content')
  })
})

describe('PerPluginStore with injected backend', () => {
  beforeEach(() => {
    process.env.CREDENTIAL_SECRET = 'test-master'
  })

  afterEach(() => {
    delete process.env.CREDENTIAL_SECRET
  })

  it('stores encrypted blob in the backend and loads it back', async () => {
    const mem = new InMemoryBackend()
    await new PerPluginStore('wet', 'u1', mem).save({ JINA_AI_API_KEY: 'k' })

    const stored = await mem.get('wet/subs/u1/config')
    expect(stored).not.toBeNull()

    // Ciphertext must differ from the plaintext JSON bytes (proves encryption ran).
    const plaintext = Buffer.from(JSON.stringify({ JINA_AI_API_KEY: 'k' }), 'utf-8')
    expect(stored).not.toEqual(plaintext)

    const loaded = await new PerPluginStore('wet', 'u1', mem).load()
    expect(loaded).toEqual({ JINA_AI_API_KEY: 'k' })
  })

  it('clear removes the blob from the backend', async () => {
    const mem = new InMemoryBackend()
    const store = new PerPluginStore('wet', 'u1', mem)
    await store.save({ k: 'v' })
    await store.clear()
    expect(await mem.get('wet/subs/u1/config')).toBeNull()
  })
})

describe('backendFromEnv', () => {
  afterEach(() => {
    delete process.env.MCP_STORAGE_BACKEND
    delete process.env.MCP_KV_BASE_URL
    delete process.env.MCP_KV_TOKEN
  })

  it('default (no env) is LocalFsBackend', () => {
    delete process.env.MCP_STORAGE_BACKEND
    expect(backendFromEnv()).toBeInstanceOf(LocalFsBackend)
  })

  it('throws on an unknown backend kind', () => {
    process.env.MCP_STORAGE_BACKEND = 'bogus'
    expect(() => backendFromEnv()).toThrow()
  })
})
