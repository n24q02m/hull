import { createCipheriv, randomBytes, scryptSync } from 'node:crypto'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { InMemoryBackend, LocalFsBackend } from '../../src/storage/backends.js'
import {
  assertValidPluginName,
  assertValidSub,
  blobPathIn,
  dataDir,
  dataDirCandidates,
  legacyDataDirs
} from '../../src/storage/data-dir.js'
import { setHomeDirForTesting } from '../../src/storage/home-dir.js'
import { credPath, PerPluginStore } from '../../src/storage/per-plugin-store.js'

/**
 * Data-dir layout migration: the store writes `~/.<plugin>/` and reads, in
 * order, `~/.<plugin>/` -> `~/.<plugin>-mcp/` -> `~/.better-<plugin>-mcp/`.
 * Legacy blobs are planted here byte-for-byte the way the pre-migration code
 * wrote them (AES-256-GCM, [IV][ciphertext][tag]) so the fallback is proven
 * against the real legacy format, not against the new writer.
 */

const PLUGIN = 'demo'

let home: string

function encrypt(key: Buffer, payload: Record<string, unknown>): Buffer {
  const iv = randomBytes(12)
  const cipher = createCipheriv('aes-256-gcm', key, iv, { authTagLength: 16 })
  const ct = Buffer.concat([cipher.update(Buffer.from(JSON.stringify(payload), 'utf-8')), cipher.final()])
  return Buffer.concat([iv, ct, cipher.getAuthTag()])
}

function subKey(secret: string, keyPlugin: string, sub: string): Buffer {
  return scryptSync(secret, Buffer.from(`${keyPlugin}:${sub}`, 'utf-8'), 32, { N: 16384, r: 8, p: 1 })
}

/** Plant a single-user blob + its machine key the way the legacy writer did. */
function plantSingle(dirName: string, payload: Record<string, unknown>, secret: Buffer = randomBytes(32)): string {
  const dir = join(home, dirName)
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, '.secret'), secret)
  writeFileSync(join(dir, 'config.json'), encrypt(secret, payload))
  return dir
}

/** Plant a multi-user blob encrypted with the scrypt salt `<keyPlugin>:<sub>`. */
function plantSub(dirName: string, keyPlugin: string, sub: string, payload: Record<string, unknown>): string {
  const dir = join(home, dirName, 'subs', sub)
  mkdirSync(dir, { recursive: true })
  const path = join(dir, 'config.json')
  writeFileSync(path, encrypt(subKey('test-master', keyPlugin, sub), payload))
  return path
}

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'data-dir-'))
  setHomeDirForTesting(home)
})

afterEach(() => {
  vi.restoreAllMocks()
  setHomeDirForTesting(null)
  delete process.env.CREDENTIAL_SECRET
  rmSync(home, { recursive: true, force: true })
})

describe('data-dir layout helpers', () => {
  it('lists candidates in read-precedence order with the legacy key plugin', () => {
    expect(dataDirCandidates(PLUGIN)).toEqual([
      { dir: join(home, '.demo'), keyPlugin: 'demo', legacy: false },
      { dir: join(home, '.demo-mcp'), keyPlugin: 'demo', legacy: true },
      { dir: join(home, '.better-demo-mcp'), keyPlugin: 'better-demo', legacy: true }
    ])
    expect(legacyDataDirs(PLUGIN).map((l) => l.dir)).toEqual([join(home, '.demo-mcp'), join(home, '.better-demo-mcp')])
    expect(dataDir(PLUGIN)).toBe(join(home, '.demo'))
  })

  it('resolves the home dir at call time', () => {
    const other = mkdtempSync(join(tmpdir(), 'data-dir-other-'))
    try {
      setHomeDirForTesting(other)
      expect(dataDir(PLUGIN)).toBe(join(other, '.demo'))
    } finally {
      rmSync(other, { recursive: true, force: true })
    }
  })

  it('builds the blob path for single-user and per-sub', () => {
    expect(blobPathIn('/x', null)).toBe(join('/x', 'config.json'))
    expect(blobPathIn('/x', 'u1')).toBe(join('/x', 'subs', 'u1', 'config.json'))
  })

  it('rejects unsafe plugin names and subs', () => {
    for (const bad of ['', '../x', 'a/b', 'a..b', 'a\\b']) {
      expect(() => assertValidPluginName(bad)).toThrow(/Invalid pluginName/)
      expect(() => dataDirCandidates(bad)).toThrow(/Invalid pluginName/)
      expect(() => assertValidSub(bad)).toThrow(/Invalid sub/)
    }
    expect(() => assertValidPluginName('ok_name-1.2')).not.toThrow()
    expect(() => assertValidSub('oG5FyoFE-RWqI_aciDl4zA')).not.toThrow()
  })

  it('credPath points at the current layout', () => {
    expect(credPath(PLUGIN, null)).toBe(join(home, '.demo', 'config.json'))
    expect(credPath(PLUGIN, 'u1')).toBe(join(home, '.demo', 'subs', 'u1', 'config.json'))
  })
})

describe('PerPluginStore single-user read fallback', () => {
  it('writes the new ~/.<plugin>/ layout, machine key included', async () => {
    const store = new PerPluginStore(PLUGIN)
    await store.save({ k: 'v' })
    expect(existsSync(join(home, '.demo', 'config.json'))).toBe(true)
    expect(existsSync(join(home, '.demo', '.secret'))).toBe(true)
    expect(existsSync(join(home, '.demo-mcp'))).toBe(false)
    expect(await store.load()).toEqual({ k: 'v' })
  })

  it('falls back to ~/.<plugin>-mcp/ decrypted with that directory .secret', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    const store = new PerPluginStore(PLUGIN)
    expect(await store.load()).toEqual({ from: 'mcp' })
    expect(await store.hasStoredBlob()).toBe(true)
  })

  it('falls back to ~/.better-<plugin>-mcp/ decrypted with that directory .secret', async () => {
    plantSingle('.better-demo-mcp', { from: 'better' })
    const store = new PerPluginStore(PLUGIN)
    expect(await store.load()).toEqual({ from: 'better' })
    expect(await store.hasStoredBlob()).toBe(true)
  })

  it('prefers the new dir over both legacy dirs', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    plantSingle('.better-demo-mcp', { from: 'better' })
    const store = new PerPluginStore(PLUGIN)
    await store.save({ from: 'new' })
    expect(await store.load()).toEqual({ from: 'new' })
  })

  it('prefers ~/.<plugin>-mcp/ over ~/.better-<plugin>-mcp/', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    plantSingle('.better-demo-mcp', { from: 'better' })
    expect(await new PerPluginStore(PLUGIN).load()).toEqual({ from: 'mcp' })
  })

  it('a read-only fallback creates nothing: no new dir, no legacy writes', async () => {
    const legacyDir = plantSingle('.demo-mcp', { from: 'mcp' })
    const before = readdirSync(legacyDir).sort()
    expect(await new PerPluginStore(PLUGIN).load()).toEqual({ from: 'mcp' })
    expect(existsSync(join(home, '.demo'))).toBe(false)
    expect(readdirSync(legacyDir).sort()).toEqual(before)
  })

  it('migrates on the next write: new dir gets the blob under its own key, legacy is kept', async () => {
    const legacySecret = randomBytes(32)
    const legacyDir = plantSingle('.demo-mcp', { token: 'old' }, legacySecret)
    const legacyBlob = readFileSync(join(legacyDir, 'config.json'))
    const store = new PerPluginStore(PLUGIN)

    const current = await store.load()
    expect(current).toEqual({ token: 'old' })
    await store.save({ ...current, extra: '1' })

    // Written to the new layout with a freshly generated machine key.
    expect(existsSync(join(home, '.demo', 'config.json'))).toBe(true)
    expect(readFileSync(join(home, '.demo', '.secret'))).not.toEqual(legacySecret)
    // Legacy copy is left untouched (a downgrade still finds its credentials).
    expect(readFileSync(join(legacyDir, 'config.json'))).toEqual(legacyBlob)
    // A fresh store reads the migrated value from the new dir.
    expect(await new PerPluginStore(PLUGIN).load()).toEqual({ token: 'old', extra: '1' })
  })

  it('clear removes the blob from the new dir and every legacy dir', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    plantSingle('.better-demo-mcp', { from: 'better' })
    const store = new PerPluginStore(PLUGIN)
    await store.save({ from: 'new' })

    await store.clear()

    for (const dir of ['.demo', '.demo-mcp', '.better-demo-mcp']) {
      expect(existsSync(join(home, dir, 'config.json')), dir).toBe(false)
    }
    expect(await store.load()).toBeNull()
    expect(await store.hasStoredBlob()).toBe(false)
  })

  it('clear is a no-op when nothing is stored anywhere', async () => {
    const store = new PerPluginStore(PLUGIN)
    await expect(store.clear()).resolves.toBeUndefined()
    expect(await store.hasStoredBlob()).toBe(false)
  })

  it('treats a tampered legacy blob as corrupt: load null + logged, hasStoredBlob true', async () => {
    const dir = plantSingle('.demo-mcp', { from: 'mcp' })
    const blob = readFileSync(join(dir, 'config.json'))
    blob[blob.length - 1] ^= 0xff
    writeFileSync(join(dir, 'config.json'), blob)

    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const store = new PerPluginStore(PLUGIN)
    expect(await store.load()).toBeNull()
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('demo/config'))
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('.demo-mcp'))
    expect(await store.hasStoredBlob()).toBe(true)
  })

  it('does not skip a corrupt higher-precedence legacy blob to resurrect an older one', async () => {
    const dir = plantSingle('.demo-mcp', { from: 'mcp' })
    writeFileSync(join(dir, 'config.json'), Buffer.from('too-short'))
    plantSingle('.better-demo-mcp', { from: 'better' })
    vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(await new PerPluginStore(PLUGIN).load()).toBeNull()
  })

  it('does not fall back past a corrupt blob in the new dir', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    mkdirSync(join(home, '.demo'), { recursive: true })
    writeFileSync(join(home, '.demo', 'config.json'), Buffer.from('too-short'))
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const store = new PerPluginStore(PLUGIN)
    expect(await store.load()).toBeNull()
    expect(await store.hasStoredBlob()).toBe(true)
  })

  it('treats a legacy blob whose .secret is missing as corrupt and does not mint one', async () => {
    const dir = plantSingle('.better-demo-mcp', { from: 'better' })
    rmSync(join(dir, '.secret'))
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const store = new PerPluginStore(PLUGIN)
    expect(await store.load()).toBeNull()
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('.better-demo-mcp'))
    expect(existsSync(join(dir, '.secret'))).toBe(false)
    expect(await store.hasStoredBlob()).toBe(true)
  })

  it('treats a legacy .secret of the wrong length as corrupt instead of throwing', async () => {
    const dir = plantSingle('.demo-mcp', { from: 'mcp' })
    writeFileSync(join(dir, '.secret'), Buffer.alloc(7))
    vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(await new PerPluginStore(PLUGIN).load()).toBeNull()
  })

  it('treats a truncated legacy blob as corrupt', async () => {
    const dir = plantSingle('.demo-mcp', { from: 'mcp' })
    writeFileSync(join(dir, 'config.json'), Buffer.allocUnsafe(28))
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(await new PerPluginStore(PLUGIN).load()).toBeNull()
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('demo/config'))
  })
})

describe('PerPluginStore multi-user (subs/<sub>) read fallback', () => {
  const SUB = 'oG5FyoFE-RWqI_aciDl4zA'

  beforeEach(() => {
    process.env.CREDENTIAL_SECRET = 'test-master'
  })

  it('writes the new ~/.<plugin>/subs/<sub>/ layout', async () => {
    const store = new PerPluginStore(PLUGIN, SUB)
    await store.save({ k: 'v' })
    expect(existsSync(join(home, '.demo', 'subs', SUB, 'config.json'))).toBe(true)
    expect(await store.load()).toEqual({ k: 'v' })
  })

  it('falls back to ~/.<plugin>-mcp/subs/<sub>/ (salt <plugin>:<sub>)', async () => {
    plantSub('.demo-mcp', 'demo', SUB, { from: 'mcp' })
    expect(await new PerPluginStore(PLUGIN, SUB).load()).toEqual({ from: 'mcp' })
  })

  it('falls back to ~/.better-<plugin>-mcp/subs/<sub>/ (salt better-<plugin>:<sub>)', async () => {
    plantSub('.better-demo-mcp', 'better-demo', SUB, { from: 'better' })
    expect(await new PerPluginStore(PLUGIN, SUB).load()).toEqual({ from: 'better' })
  })

  it('a sub blob never leaks to another sub, including through legacy dirs', async () => {
    plantSub('.demo-mcp', 'demo', 'sub-a', { secret: 'for-a' })
    const b = new PerPluginStore(PLUGIN, 'sub-b')
    expect(await b.load()).toBeNull()
    expect(await b.hasStoredBlob()).toBe(false)
  })

  it('migrates on write and clears every location', async () => {
    const legacyPath = plantSub('.better-demo-mcp', 'better-demo', SUB, { token: 'old' })
    const store = new PerPluginStore(PLUGIN, SUB)
    expect(await store.load()).toEqual({ token: 'old' })

    await store.save({ token: 'new' })
    expect(existsSync(join(home, '.demo', 'subs', SUB, 'config.json'))).toBe(true)
    expect(existsSync(legacyPath)).toBe(true)
    expect(await new PerPluginStore(PLUGIN, SUB).load()).toEqual({ token: 'new' })

    plantSub('.demo-mcp', 'demo', SUB, { token: 'mid' })
    await store.clear()
    expect(existsSync(legacyPath)).toBe(false)
    expect(existsSync(join(home, '.demo-mcp', 'subs', SUB, 'config.json'))).toBe(false)
    expect(existsSync(join(home, '.demo', 'subs', SUB, 'config.json'))).toBe(false)
    expect(await store.load()).toBeNull()
  })

  it('treats a legacy sub blob encrypted under another salt as corrupt', async () => {
    // Written with the wrong key plugin for its directory: must not decrypt.
    plantSub('.demo-mcp', 'better-demo', SUB, { from: 'wrong-salt' })
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const store = new PerPluginStore(PLUGIN, SUB)
    expect(await store.load()).toBeNull()
    expect(await store.hasStoredBlob()).toBe(true)
  })

  it('still requires CREDENTIAL_SECRET to read a legacy sub blob', async () => {
    plantSub('.demo-mcp', 'demo', SUB, { from: 'mcp' })
    delete process.env.CREDENTIAL_SECRET
    await expect(new PerPluginStore(PLUGIN, SUB).load()).rejects.toThrow(/CREDENTIAL_SECRET/)
  })
})

describe('legacy fallback applies only to the default local-fs layout', () => {
  it('an injected InMemoryBackend ignores on-disk legacy dirs', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    const store = new PerPluginStore(PLUGIN, null, new InMemoryBackend())
    expect(await store.load()).toBeNull()
    expect(await store.hasStoredBlob()).toBe(false)
    await store.clear()
    // clear on a non-local backend leaves the legacy disk copy alone.
    expect(existsSync(join(home, '.demo-mcp', 'config.json'))).toBe(true)
  })

  it('a LocalFsBackend with a custom baseDir ignores legacy dirs', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    const custom = join(home, 'custom')
    const store = new PerPluginStore(PLUGIN, null, new LocalFsBackend({ baseDir: () => custom }))
    expect(await store.load()).toBeNull()
    await store.save({ from: 'custom' })
    expect(existsSync(join(custom, 'config.json'))).toBe(true)
    expect(await store.load()).toEqual({ from: 'custom' })
  })

  it('an explicitly injected default LocalFsBackend keeps the fallback', async () => {
    plantSingle('.demo-mcp', { from: 'mcp' })
    expect(await new PerPluginStore(PLUGIN, null, new LocalFsBackend()).load()).toEqual({ from: 'mcp' })
  })
})
