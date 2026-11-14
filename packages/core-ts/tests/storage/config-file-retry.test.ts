/**
 * Config writes retry a file another process holds open (`EBUSY`, common on
 * Windows while an antivirus or a second server reads the file), and only that.
 * The OS cannot be made to return EBUSY on demand, so `writeFile` fails on
 * purpose for the first calls and then writes for real.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const failures = vi.hoisted(() => ({ codes: [] as string[], calls: 0 }))

vi.mock('node:fs/promises', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs/promises')>()
  return {
    ...actual,
    writeFile: async (...args: Parameters<typeof actual.writeFile>) => {
      failures.calls += 1
      const code = failures.codes.shift()
      if (code) throw Object.assign(new Error(`${code}: simulated`), { code })
      return actual.writeFile(...args)
    }
  }
})

const { clearKeyCacheForTesting, readConfig, setConfigPath, writeConfig } = await import(
  '../../src/storage/config-file.js'
)

let tempDir: string

beforeEach(async () => {
  tempDir = await mkdtemp(join(tmpdir(), 'mcp-retry-'))
  setConfigPath(join(tempDir, 'config.enc'))
  clearKeyCacheForTesting()
  failures.codes = []
  failures.calls = 0
})

afterEach(async () => {
  setConfigPath(null)
  await rm(tempDir, { recursive: true, force: true })
})

describe('config write retry', () => {
  it('retries a busy file and then writes it', async () => {
    failures.codes = ['EBUSY', 'EBUSY']

    await writeConfig('server', { k: 'v' })

    expect(failures.calls).toBe(3)
    expect(await readConfig('server')).toEqual({ k: 'v' })
  }, 60000)

  it('gives up after three busy attempts', async () => {
    failures.codes = ['EBUSY', 'EBUSY', 'EBUSY']

    await expect(writeConfig('server', { k: 'v' })).rejects.toMatchObject({ code: 'EBUSY' })
    expect(failures.calls).toBe(3)
  }, 60000)

  it('does not retry any other error', async () => {
    failures.codes = ['EACCES']

    await expect(writeConfig('server', { k: 'v' })).rejects.toMatchObject({ code: 'EACCES' })
    expect(failures.calls).toBe(1)
  }, 60000)
})
