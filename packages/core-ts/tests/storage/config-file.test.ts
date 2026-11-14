import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  clearKeyCacheForTesting,
  deleteConfig,
  exportConfig,
  importConfig,
  listConfigs,
  readConfig,
  scheduleReloadExit,
  setConfigPath,
  writeConfig
} from '../../src/storage/config-file.js'
import {
  decryptData,
  deriveFileKey,
  derivePassphraseKey,
  encryptData,
  LEGACY_EXPORT_SALT,
  LEGACY_PBKDF2_ITERATIONS,
  LEGACY_SALT,
  PBKDF2_ITERATIONS
} from '../../src/storage/encryption.js'
import { getMachineId, getUsername } from '../../src/storage/machine-id.js'

let tempDir: string

beforeEach(async () => {
  tempDir = await mkdtemp(join(tmpdir(), 'mcp-test-'))
  setConfigPath(join(tempDir, 'config.enc'))
  clearKeyCacheForTesting()
})

afterEach(async () => {
  setConfigPath(null)
  await rm(tempDir, { recursive: true, force: true })
})

describe('writeConfig + readConfig', () => {
  it('writes and reads a server config', async () => {
    await writeConfig('telegram', { botToken: 'abc123', chatId: '456' })
    const config = await readConfig('telegram')
    expect(config).toEqual({ botToken: 'abc123', chatId: '456' })
  })

  it('returns null for non-existent server', async () => {
    const config = await readConfig('nonexistent')
    expect(config).toBeNull()
  })

  it('returns null when no config file exists', async () => {
    const config = await readConfig('anything')
    expect(config).toBeNull()
  })
})

describe('writeConfig merging', () => {
  it('does not overwrite other servers sections', async () => {
    await writeConfig('telegram', { botToken: 'tok1' })
    await writeConfig('slack', { webhook: 'https://example.com' })

    const telegram = await readConfig('telegram')
    const slack = await readConfig('slack')
    expect(telegram).toEqual({ botToken: 'tok1' })
    expect(slack).toEqual({ webhook: 'https://example.com' })
  })

  it('overwrites same server config on second write', async () => {
    await writeConfig('telegram', { botToken: 'old' })
    await writeConfig('telegram', { botToken: 'new', extra: 'field' })

    const config = await readConfig('telegram')
    expect(config).toEqual({ botToken: 'new', extra: 'field' })
  })
})

describe('deleteConfig', () => {
  it('removes a server section', async () => {
    await writeConfig('telegram', { botToken: 'tok' })
    await writeConfig('slack', { webhook: 'url' })

    await deleteConfig('telegram')

    expect(await readConfig('telegram')).toBeNull()
    expect(await readConfig('slack')).toEqual({ webhook: 'url' })
  })

  it('deletes file when last server removed', async () => {
    await writeConfig('telegram', { botToken: 'tok' })
    await deleteConfig('telegram')

    const { existsSync } = await import('node:fs')
    expect(existsSync(join(tempDir, 'config.enc'))).toBe(false)
  })

  it('no-op for non-existent server', async () => {
    await writeConfig('telegram', { botToken: 'tok' })
    await deleteConfig('nonexistent')
    expect(await readConfig('telegram')).toEqual({ botToken: 'tok' })
  })
})

describe('listConfigs', () => {
  it('returns empty array when no config', async () => {
    expect(await listConfigs()).toEqual([])
  })

  it('returns list of server names', async () => {
    await writeConfig('telegram', { a: '1' })
    await writeConfig('slack', { b: '2' })
    await writeConfig('discord', { c: '3' })

    const names = await listConfigs()
    expect(names.sort()).toEqual(['discord', 'slack', 'telegram'])
  })
})

describe('exportConfig + importConfig', () => {
  it('roundtrip with passphrase', async () => {
    await writeConfig('telegram', { botToken: 'abc' })
    await writeConfig('slack', { webhook: 'url' })

    const exported = await exportConfig('my-secret-passphrase')
    expect(exported).toBeInstanceOf(Buffer)

    // Clear local config
    await deleteConfig('telegram')
    await deleteConfig('slack')
    expect(await listConfigs()).toEqual([])

    // Import back
    await importConfig('my-secret-passphrase', exported)
    expect(await readConfig('telegram')).toEqual({ botToken: 'abc' })
    expect(await readConfig('slack')).toEqual({ webhook: 'url' })
  }, 60000)

  it('wrong passphrase fails to import', async () => {
    await writeConfig('telegram', { botToken: 'abc' })
    const exported = await exportConfig('correct-pass')

    await expect(importConfig('wrong-pass', exported)).rejects.toThrow()
  }, 60000)

  it('import merges into existing config', async () => {
    await writeConfig('local-server', { key: 'local-val' })

    // Create export data from a separate config
    await writeConfig('remote-server', { key: 'remote-val' })
    const exported = await exportConfig('pass')

    // Remove remote, keep local
    await deleteConfig('remote-server')
    expect(await listConfigs()).toEqual(['local-server'])

    // Import should merge
    await importConfig('pass', exported)
    expect(await readConfig('local-server')).toEqual({ key: 'local-val' })
    expect(await readConfig('remote-server')).toEqual({ key: 'remote-val' })
  }, 60000)

  it('imports legacy (unsalted) export data', async () => {
    const store = { version: 1, servers: { legacy: { k: 'v' } } }
    const key = await derivePassphraseKey('pass', LEGACY_EXPORT_SALT, PBKDF2_ITERATIONS)
    const encrypted = await encryptData(key, JSON.stringify(store))

    await importConfig('pass', encrypted)
    expect(await readConfig('legacy')).toEqual({ k: 'v' })
  }, 60000)
})

describe('config validation', () => {
  it('throws on invalid version', async () => {
    const invalidStore = { version: 2, servers: {} }
    const [machineId, username] = await Promise.all([getMachineId(), getUsername()])
    const key = await deriveFileKey(machineId, username, LEGACY_SALT)
    const encrypted = await encryptData(key, JSON.stringify(invalidStore))
    await writeFile(join(tempDir, 'config.enc'), encrypted)

    await expect(readConfig('any')).rejects.toThrow('Invalid config schema')
  })

  it('throws on invalid servers object', async () => {
    const invalidStore = { version: 1, servers: 'not-an-object' }
    const [machineId, username] = await Promise.all([getMachineId(), getUsername()])
    const key = await deriveFileKey(machineId, username, LEGACY_SALT)
    const encrypted = await encryptData(key, JSON.stringify(invalidStore))
    await writeFile(join(tempDir, 'config.enc'), encrypted)

    await expect(readConfig('any')).rejects.toThrow('Invalid config schema')
  })

  it('throws on invalid server config (non-string value)', async () => {
    const invalidStore = { version: 1, servers: { bad: { key: 123 } } }
    const [machineId, username] = await Promise.all([getMachineId(), getUsername()])
    const key = await deriveFileKey(machineId, username, LEGACY_SALT)
    const encrypted = await encryptData(key, JSON.stringify(invalidStore))
    await writeFile(join(tempDir, 'config.enc'), encrypted)

    await expect(readConfig('any')).rejects.toThrow('Invalid config schema')
  })
})

describe('migration', () => {
  it('migrates from legacy unsalted format', async () => {
    const store = { version: 1, servers: { old: { data: 'val' } } }
    const [machineId, username] = await Promise.all([getMachineId(), getUsername()])
    const key = await deriveFileKey(machineId, username, LEGACY_SALT)
    const encrypted = await encryptData(key, JSON.stringify(store))
    await writeFile(join(tempDir, 'config.enc'), encrypted)

    // Read should trigger migration
    const config = await readConfig('old')
    expect(config).toEqual({ data: 'val' })

    // Verify file now has salt prepended (length should increase by 16)
    const { readFile } = await import('node:fs/promises')
    const migratedData = await readFile(join(tempDir, 'config.enc'))
    expect(migratedData.length).toBe(encrypted.length + 16)
  })

  it('migrates a legacy file written with the old PBKDF2 iteration count', async () => {
    const store = { version: 1, servers: { older: { data: 'val' } } }
    const [machineId, username] = await Promise.all([getMachineId(), getUsername()])
    const key = await deriveFileKey(machineId, username, LEGACY_SALT, LEGACY_PBKDF2_ITERATIONS)
    const encrypted = await encryptData(key, JSON.stringify(store))
    await writeFile(join(tempDir, 'config.enc'), encrypted)

    expect(await readConfig('older')).toEqual({ data: 'val' })

    // Rewritten in the salted, current-iteration format, which reads back directly.
    const migrated = await readFile(join(tempDir, 'config.enc'))
    expect(migrated.length).toBe(encrypted.length + 16)
    clearKeyCacheForTesting()
    const current = await deriveFileKey(machineId, username, migrated.subarray(0, 16))
    expect(JSON.parse(await decryptData(current, migrated.subarray(16)))).toEqual(store)
  }, 60000)

  it('rejects a legacy old-iteration file with an invalid schema without rewriting it', async () => {
    const [machineId, username] = await Promise.all([getMachineId(), getUsername()])
    const key = await deriveFileKey(machineId, username, LEGACY_SALT, LEGACY_PBKDF2_ITERATIONS)
    const encrypted = await encryptData(key, JSON.stringify({ version: 2, servers: {} }))
    await writeFile(join(tempDir, 'config.enc'), encrypted)

    await expect(readConfig('any')).rejects.toThrow()
    expect(await readFile(join(tempDir, 'config.enc'))).toEqual(encrypted)
  }, 60000)
})

describe('salted format validation', () => {
  async function writeSalted(store: unknown): Promise<void> {
    const salt = crypto.getRandomValues(new Uint8Array(16))
    const [machineId, username] = await Promise.all([getMachineId(), getUsername()])
    const key = await deriveFileKey(machineId, username, salt)
    const encrypted = await encryptData(key, JSON.stringify(store))
    await writeFile(join(tempDir, 'config.enc'), Buffer.concat([Buffer.from(salt), encrypted]))
  }

  it('rejects a salted file whose schema is invalid', async () => {
    await writeSalted({ version: 2, servers: {} })

    await expect(readConfig('any')).rejects.toThrow()
  }, 60000)

  it('rejects a server entry that is not an object', async () => {
    await writeSalted({ version: 1, servers: { bad: 'not-an-object' } })

    await expect(readConfig('bad')).rejects.toThrow()
  }, 60000)
})

describe('importConfig validation', () => {
  async function exportOf(payload: unknown, passphrase: string): Promise<Buffer> {
    const salt = crypto.getRandomValues(new Uint8Array(16))
    const key = await derivePassphraseKey(passphrase, salt)
    return Buffer.concat([Buffer.from(salt), await encryptData(key, JSON.stringify(payload))])
  }

  it('rejects an export whose payload is not a config object', async () => {
    await expect(importConfig('pass', await exportOf(null, 'pass'))).rejects.toThrow(
      'Invalid config schema in imported data'
    )
    expect(await listConfigs()).toEqual([])
  }, 60000)

  it('imports a salted export written with the old PBKDF2 iteration count', async () => {
    const salt = crypto.getRandomValues(new Uint8Array(16))
    const key = await derivePassphraseKey('pass', salt, LEGACY_PBKDF2_ITERATIONS)
    const store = { version: 1, servers: { moved: { k: 'v' } } }
    const data = Buffer.concat([Buffer.from(salt), await encryptData(key, JSON.stringify(store))])

    await importConfig('pass', data)
    expect(await readConfig('moved')).toEqual({ k: 'v' })
  }, 60000)
})

describe('config directory', () => {
  it('creates a missing parent directory on the first write', async () => {
    const nested = join(tempDir, 'fresh', 'nested', 'config.enc')
    setConfigPath(nested)

    await writeConfig('first', { k: 'v' })

    expect(existsSync(nested)).toBe(true)
    expect(await readConfig('first')).toEqual({ k: 'v' })
  }, 60000)
})

describe('scheduleReloadExit', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  function exitSpy() {
    return vi.spyOn(process, 'exit').mockImplementation((() => undefined) as never)
  }

  it('exits the process a second later outside tests', () => {
    vi.useFakeTimers()
    vi.stubEnv('NODE_ENV', 'production')
    vi.stubEnv('MCP_NO_RELOAD', '')
    const exit = exitSpy()

    scheduleReloadExit()
    vi.advanceTimersByTime(999)
    expect(exit).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(exit).toHaveBeenCalledWith(0)
  })

  it('does nothing when MCP_NO_RELOAD is set', () => {
    vi.useFakeTimers()
    vi.stubEnv('NODE_ENV', 'production')
    vi.stubEnv('MCP_NO_RELOAD', '1')
    const exit = exitSpy()

    scheduleReloadExit()
    vi.advanceTimersByTime(5000)
    expect(exit).not.toHaveBeenCalled()
  })

  it('does nothing under the test runner', () => {
    vi.useFakeTimers()
    vi.stubEnv('NODE_ENV', 'test')
    vi.stubEnv('MCP_NO_RELOAD', '')
    const exit = exitSpy()

    scheduleReloadExit()
    vi.advanceTimersByTime(5000)
    expect(exit).not.toHaveBeenCalled()
  })
})
