import { createCipheriv, randomBytes } from 'node:crypto'
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Interactive `[y/N]` prompt: when `answer` is set, createInterface returns a
// stub that emits that line once; otherwise the real readline is used.
const { promptAnswer } = vi.hoisted(() => ({ promptAnswer: { value: null as string | null } }))
vi.mock('node:readline', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:readline')>()
  return {
    ...actual,
    createInterface: vi.fn((...args: unknown[]) => {
      const answer = promptAnswer.value
      if (answer === null) return (actual.createInterface as (...a: unknown[]) => unknown)(...args)
      return {
        once: (_event: string, cb: (line: string) => void) => queueMicrotask(() => cb(answer)),
        close: vi.fn()
      }
    })
  }
})

// Doctor's writability probe: a store dir whose path contains this trigger is
// reported unwritable (portable; Windows ignores POSIX dir modes).
vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs')>()
  return {
    ...actual,
    accessSync: vi.fn((path: Parameters<typeof actual.accessSync>[0], mode?: number) => {
      if (typeof path === 'string' && path.includes('unwritable-trigger')) {
        throw Object.assign(new Error('EACCES: permission denied'), { code: 'EACCES' })
      }
      return actual.accessSync(path, mode)
    })
  }
})

import {
  LocalFsBackend,
  PerPluginStore,
  setConfigPath,
  setHomeDirForTesting,
  setLockDir,
  writeConfig
} from '../storage/index.js'
import { buildCli } from './build-cli.js'

const SERVER = 'w51-test'

let tmp: string
let logs: string[]
let errs: string[]

/**
 * Plant a single-user credential the way the pre-migration writer did, in a
 * legacy dir under the test home: its own `.secret` + an AES-256-GCM blob.
 */
function plantLegacy(dirName: string, payload: Record<string, unknown>): string {
  const dir = join(tmp, dirName)
  mkdirSync(dir, { recursive: true })
  const secret = randomBytes(32)
  const iv = randomBytes(12)
  const cipher = createCipheriv('aes-256-gcm', secret, iv, { authTagLength: 16 })
  const ct = Buffer.concat([cipher.update(Buffer.from(JSON.stringify(payload), 'utf-8')), cipher.final()])
  writeFileSync(join(dir, '.secret'), secret)
  writeFileSync(join(dir, 'config.json'), Buffer.concat([iv, ct, cipher.getAuthTag()]))
  return dir
}

function capture(): void {
  logs = []
  errs = []
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    logs.push(args.join(' '))
  })
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    errs.push(args.join(' '))
  })
}

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), 'w51-'))
  setHomeDirForTesting(tmp)
  setLockDir(join(tmp, 'locks'))
  setConfigPath(join(tmp, 'config.enc'))
  capture()
})

afterEach(() => {
  vi.restoreAllMocks()
  setHomeDirForTesting(null)
  setLockDir(null)
  setConfigPath(null)
  rmSync(tmp, { recursive: true, force: true })
})

describe('buildCli — serve dispatch', () => {
  it('runs the server with an empty argv on bare invocation', async () => {
    const seen: string[][] = []
    const run = buildCli(SERVER, {
      serve: (argv) => {
        seen.push(argv)
      }
    })
    const rc = await run([])
    expect(seen).toEqual([[]])
    expect(rc).toBe(0)
  })

  it('passes flag-prefixed argv through to serve byte-for-byte', async () => {
    const seen: string[][] = []
    const run = buildCli(SERVER, {
      serve: (argv) => {
        seen.push(argv)
      }
    })
    const rc = await run(['--http', '8080'])
    expect(seen).toEqual([['--http', '8080']])
    expect(rc).toBe(0)
  })

  it('propagates a numeric return code from serve', async () => {
    const run = buildCli(SERVER, { serve: () => 3 })
    expect(await run([])).toBe(3)
  })

  it('propagates a numeric return code from serve on the flag path', async () => {
    const run = buildCli(SERVER, { serve: () => 4 })
    expect(await run(['--http'])).toBe(4)
  })

  it('reads process.argv when called without an argv', async () => {
    const saved = process.argv
    process.argv = ['node', SERVER, 'status']
    try {
      const run = buildCli(SERVER, { serve: () => 0 })
      expect(await run()).toBe(0)
      expect(logs[0]).toBe(`${SERVER}: not configured`)
    } finally {
      process.argv = saved
    }
  })
})

describe('buildCli — help and version', () => {
  it('prints help on -h without touching serve', async () => {
    let served = false
    const run = buildCli(SERVER, {
      serve: () => {
        served = true
      }
    })
    const rc = await run(['-h'])
    expect(rc).toBe(0)
    expect(served).toBe(false)
    expect(logs[0]).toBe(`usage: ${SERVER} [-h] [--version] [<subcommand> ...]`)
    expect(logs).toContain('subcommands: config, doctor, logout, status, token')
  })

  it('prints help on --help', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['--help'])).toBe(0)
    expect(logs.join('\n')).toContain('passed through to the MCP server')
  })

  it('lists extra subcommands sorted in help', async () => {
    const run = buildCli(SERVER, { serve: () => 0, extra: { auth: () => 0 } })
    await run(['-h'])
    expect(logs).toContain('subcommands: auth, config, doctor, logout, status, token')
  })

  it('does not reserve auth as a built-in', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['auth'])).toBe(2)
    expect(errs.join('\n')).toContain(`${SERVER}: unknown subcommand 'auth'`)
  })

  it('prints "<name> <version>" on --version when version is set', async () => {
    const run = buildCli(SERVER, { serve: () => 0, version: '1.2.3' })
    expect(await run(['--version'])).toBe(0)
    expect(logs).toEqual([`${SERVER} 1.2.3`])
  })

  it('prints version on -V', async () => {
    const run = buildCli(SERVER, { serve: () => 0, version: '1.2.3' })
    expect(await run(['-V'])).toBe(0)
    expect(logs).toEqual([`${SERVER} 1.2.3`])
  })

  it('falls through --version to serve when version is unset', async () => {
    const seen: string[][] = []
    const run = buildCli(SERVER, {
      serve: (argv) => {
        seen.push(argv)
      }
    })
    expect(await run(['--version'])).toBe(0)
    expect(seen).toEqual([['--version']])
  })
})

describe('buildCli — unknown subcommand', () => {
  it('reports an unknown subcommand on stderr with rc 2', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    const rc = await run(['bogus'])
    expect(rc).toBe(2)
    expect(errs.join('\n')).toContain(`${SERVER}: unknown subcommand 'bogus'`)
    expect(errs.join('\n')).toContain('expected one of: config, doctor, logout, status, token)')
  })
})

describe('buildCli — token subcommand', () => {
  it('is a reserved built-in that refuses a token on argv', async () => {
    const stderr = vi.spyOn(process.stderr, 'write').mockImplementation(() => true)
    const stdout = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['token', 'hash', 'plaintext'])).toBe(2)
    expect(stdout).not.toHaveBeenCalled()
    expect(stderr.mock.calls.map((c) => String(c[0])).join('')).toContain('refusing a token on argv')
  })

  it('can be overridden through extra', async () => {
    const run = buildCli(SERVER, { serve: () => 0, extra: { token: () => 7 } })
    expect(await run(['token', 'hash'])).toBe(7)
  })
})

describe('buildCli — config subcommand', () => {
  it('reports not configured', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['config', 'status'])).toBe(0)
    expect(logs[0]).toBe(`${SERVER}: not configured`)
  })

  it('reports configured after a payload is saved', async () => {
    await new PerPluginStore(SERVER).save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['config', 'status'])).toBe(0)
    expect(logs[0]).toBe(`${SERVER}: configured (source: file)`)
    // A credential value is never printed by a built-in.
    expect(logs.join('\n')).not.toContain('secret')
  })

  it('sees a single-user credential saved through the unified writeConfig', async () => {
    // Storage-unify contract: a server that persists single-user credentials
    // via mcp-core's public `writeConfig` (keyed by console name) writes to the
    // same per-plugin store `config status` reads (keyed by the derived slug),
    // so the CLI reports "configured" -- the whole point of the unification.
    await writeConfig('better-email-mcp', { EMAIL_CREDENTIALS: 'a@b.com:app-pass' })
    const run = buildCli('better-email-mcp', { serve: () => 0 })
    expect(await run(['config', 'status'])).toBe(0)
    expect(logs[0]).toBe('better-email-mcp: configured (source: file)')
    // A credential value is never printed by a built-in.
    expect(logs.join('\n')).not.toContain('app-pass')
  })

  it('reports a corrupt (undecryptable) config with rc 1', async () => {
    await new LocalFsBackend().put(`${SERVER}/config`, Buffer.from('too-short'))
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['config', 'status'])).toBe(1)
    expect(errs.join('\n')).toContain('config is corrupt')
  })

  it('deletes config with --yes', async () => {
    const store = new PerPluginStore(SERVER)
    await store.save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['config', 'delete', '--yes'])).toBe(0)
    expect(logs[0]).toBe(`${SERVER}: config deleted`)
    expect(await store.load()).toBeNull()
  })

  it('refuses to delete without --yes in non-interactive mode', async () => {
    await new PerPluginStore(SERVER).save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['config', 'delete'])).toBe(1)
    expect(errs.join('\n')).toContain('refusing to delete without --yes')
  })

  it('returns rc 2 on a missing or invalid config action', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['config'])).toBe(2)
    expect(await run(['config', 'bogus'])).toBe(2)
  })
})

describe('buildCli — status subcommand (same semantics as config status)', () => {
  it('reports not configured with rc 0', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['status'])).toBe(0)
    expect(logs[0]).toBe(`${SERVER}: not configured`)
  })

  it('reports configured with rc 0 and never prints a value', async () => {
    await new PerPluginStore(SERVER).save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['status'])).toBe(0)
    expect(logs[0]).toBe(`${SERVER}: configured (source: file)`)
    expect(logs.join('\n')).not.toContain('secret')
  })

  it('reports a corrupt config with rc 1', async () => {
    await new LocalFsBackend().put(`${SERVER}/config`, Buffer.from('too-short'))
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['status'])).toBe(1)
    expect(errs.join('\n')).toContain('config is corrupt')
  })

  it('reports configured for a credential that only exists in a legacy dir', async () => {
    plantLegacy(`.${SERVER}-mcp`, { token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['status'])).toBe(0)
    expect(logs[0]).toBe(`${SERVER}: configured (source: file)`)
  })

  it('can be overridden through extra', async () => {
    const run = buildCli(SERVER, { serve: () => 0, extra: { status: () => 9 } })
    expect(await run(['status'])).toBe(9)
  })
})

describe('buildCli — logout subcommand (same semantics as config delete)', () => {
  it('deletes the stored credential with --yes (rc 0), legacy copies included', async () => {
    const legacyDir = plantLegacy(`.better-${SERVER}-mcp`, { token: 'old' })
    const store = new PerPluginStore(SERVER)
    await store.save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['logout', '--yes'])).toBe(0)
    expect(logs[0]).toBe(`${SERVER}: config deleted`)
    expect(await store.load()).toBeNull()
    expect(await store.hasStoredBlob()).toBe(false)
    expect(existsSync(join(legacyDir, 'config.json'))).toBe(false)
  })

  it('refuses without --yes in non-interactive mode (rc 1) and keeps the credential', async () => {
    const store = new PerPluginStore(SERVER)
    await store.save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['logout'])).toBe(1)
    expect(errs.join('\n')).toContain('refusing to delete without --yes')
    expect(await store.load()).toEqual({ token: 'secret' })
  })

  it('succeeds with --yes even when nothing is stored', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['logout', '--yes'])).toBe(0)
  })
})

describe('buildCli — interactive delete confirmation (TTY)', () => {
  let ttyDescriptor: PropertyDescriptor | undefined
  let prompts: string[]

  beforeEach(() => {
    ttyDescriptor = Object.getOwnPropertyDescriptor(process.stdin, 'isTTY')
    Object.defineProperty(process.stdin, 'isTTY', { value: true, configurable: true })
    prompts = []
    vi.spyOn(process.stderr, 'write').mockImplementation((chunk: string | Uint8Array) => {
      prompts.push(String(chunk))
      return true
    })
  })

  afterEach(() => {
    promptAnswer.value = null
    if (ttyDescriptor) Object.defineProperty(process.stdin, 'isTTY', ttyDescriptor)
    else delete (process.stdin as { isTTY?: boolean }).isTTY
  })

  it('logout deletes after a "yes" answer (rc 0)', async () => {
    promptAnswer.value = ' YES '
    const store = new PerPluginStore(SERVER)
    await store.save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['logout'])).toBe(0)
    expect(prompts.join('')).toContain(`Delete stored config for ${SERVER}? [y/N]`)
    expect(await store.hasStoredBlob()).toBe(false)
  })

  it('config delete aborts on any other answer (rc 1) and keeps the credential', async () => {
    promptAnswer.value = 'n'
    const store = new PerPluginStore(SERVER)
    await store.save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['config', 'delete'])).toBe(1)
    expect(errs.join('\n')).toContain(`${SERVER}: aborted`)
    expect(await store.load()).toEqual({ token: 'secret' })
  })
})

describe('buildCli — doctor subcommand', () => {
  it('runs every check and exits per the aggregate health', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    const rc = await run(['doctor'])
    const all = logs.join('\n')
    expect(all).toMatch(/\[(ok|fail)] node /)
    expect(all).toContain('[ok] credential backend initializes')
    expect(all).toContain('[warn] config: not configured')
    expect(all).toContain('[ok] mode: unset')
    // doctor returns 0 iff no check failed. The runtime-version line is the only
    // check that varies by test runner: `bun run test` runs under Bun, whose
    // process.versions.node is a compat shim that some Bun builds report below
    // the >=24 floor. Assert the aggregate contract, not a fixed exit code.
    expect(rc).toBe(all.includes('[fail]') ? 1 : 0)
  })

  it('returns 1 and reports a corrupt config regardless of runtime', async () => {
    await new LocalFsBackend().put(`${SERVER}/config`, Buffer.from('too-short'))
    const run = buildCli(SERVER, { serve: () => 0 })
    expect(await run(['doctor'])).toBe(1)
    expect(logs.join('\n')).toContain('[fail] config: corrupt')
  })

  it('warns that the store dir does not exist yet when nothing is on disk', async () => {
    const run = buildCli(SERVER, { serve: () => 0 })
    await run(['doctor'])
    expect(logs.join('\n')).toContain(`[warn] store dir does not exist yet: ${join(tmp, `.${SERVER}`)}`)
  })

  it('reports the new store dir as writable once it exists', async () => {
    await new PerPluginStore(SERVER).save({ token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    await run(['doctor'])
    const all = logs.join('\n')
    expect(all).toContain(`[ok] store dir writable: ${join(tmp, `.${SERVER}`)}`)
    expect(all).not.toContain('legacy store dir')
  })

  it('warns when only a legacy store dir exists, naming both dirs', async () => {
    const legacyDir = plantLegacy(`.${SERVER}-mcp`, { token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    await run(['doctor'])
    const all = logs.join('\n')
    expect(all).toContain(`[warn] only legacy store dir found: ${legacyDir}`)
    expect(all).toContain(join(tmp, `.${SERVER}`))
    expect(all).not.toContain('store dir does not exist yet')
    // The legacy credential still decrypts, so config itself is healthy.
    expect(all).toContain('[ok] config: configured')
  })

  it('warns about a better-<plugin>-mcp legacy dir too', async () => {
    const legacyDir = plantLegacy(`.better-${SERVER}-mcp`, { token: 'secret' })
    const run = buildCli(SERVER, { serve: () => 0 })
    await run(['doctor'])
    expect(logs.join('\n')).toContain(`[warn] only legacy store dir found: ${legacyDir}`)
  })

  it('fails when the store dir exists but is not writable', async () => {
    const dir = join(tmp, '.unwritable-trigger')
    mkdirSync(dir, { recursive: true })
    const run = buildCli(SERVER, { serve: () => 0, pluginName: 'unwritable-trigger' })
    expect(await run(['doctor'])).toBe(1)
    expect(logs.join('\n')).toContain(`[fail] store dir not writable: ${dir}`)
  })

  it('fails on a runtime below the supported Node floor', async () => {
    const descriptor = Object.getOwnPropertyDescriptor(process.versions, 'node') as PropertyDescriptor
    Object.defineProperty(process.versions, 'node', { ...descriptor, value: '18.0.0' })
    try {
      const run = buildCli(SERVER, { serve: () => 0 })
      expect(await run(['doctor'])).toBe(1)
      expect(logs.join('\n')).toContain('[fail] node 18.0.0 (expected >=24)')
    } finally {
      Object.defineProperty(process.versions, 'node', descriptor)
    }
  })

  it('reports a broken credential backend without crashing the remaining checks', async () => {
    process.env.MCP_STORAGE_BACKEND = 'bogus'
    try {
      const run = buildCli(SERVER, { serve: () => 0 })
      expect(await run(['doctor'])).toBe(1)
      const all = logs.join('\n')
      expect(all).toContain('[fail] credential backend: Unknown MCP_STORAGE_BACKEND: bogus')
      expect(all).toContain('[warn] config: not checked (credential backend unavailable)')
      // Later checks still run.
      expect(all).toContain('[ok] mode: unset')
    } finally {
      delete process.env.MCP_STORAGE_BACKEND
    }
  })
})

describe('buildCli — pluginName / serverName separation (store key vs display)', () => {
  it('derives pluginName from serverName by stripping a trailing -mcp suffix', async () => {
    // Credentials are saved under the plugin slug ("w51"), which is the
    // console name ("w51-mcp") with the "-mcp" suffix stripped. `config
    // status` must read that same slug, not the literal console name.
    await new PerPluginStore('w51').save({ token: 'secret' })
    const run = buildCli('w51-mcp', { serve: () => 0 })
    expect(await run(['config', 'status'])).toBe(0)
    expect(logs[0]).toBe('w51-mcp: configured (source: file)')
  })

  it('lets an explicit pluginName override the -mcp suffix default', async () => {
    // When the slug is not simply "<name>-mcp" minus "-mcp" (telegram's slug
    // is "telegram", not "better-telegram"), an explicit pluginName pins it.
    await new PerPluginStore('telegram').save({ token: 'secret' })
    const run = buildCli('better-telegram-mcp', { serve: () => 0, pluginName: 'telegram' })
    expect(await run(['config', 'status'])).toBe(0)
    expect(logs[0]).toBe('better-telegram-mcp: configured (source: file)')
  })

  it('deletes the config under the plugin slug store', async () => {
    const store = new PerPluginStore('w51')
    await store.save({ token: 'secret' })
    const run = buildCli('w51-mcp', { serve: () => 0 })
    expect(await run(['config', 'delete', '--yes'])).toBe(0)
    expect(await store.load()).toBeNull()
  })

  it('doctor reads config status under the plugin slug', async () => {
    // Assert on the config-status line only, not the aggregate rc: the
    // runtime-version check is the only line that varies by test runner (see
    // the "runs every check" test above), and is orthogonal to plugin_name.
    await new PerPluginStore('w51').save({ token: 'secret' })
    const run = buildCli('w51-mcp', { serve: () => 0 })
    await run(['doctor'])
    expect(logs.join('\n')).toContain('[ok] config: configured')
  })

  it('doctor probes the store dir under the plugin slug, not the console name', async () => {
    // Saving creates <home>/.w51/. doctor's store-dir probe must point at that
    // path (plugin slug), not <home>/.w51-mcp/ (console name). Assert on that
    // line only, not the aggregate rc (see comment above).
    await new PerPluginStore('w51').save({ token: 'secret' })
    const run = buildCli('w51-mcp', { serve: () => 0 })
    await run(['doctor'])
    const all = logs.join('\n')
    expect(all).toContain(`store dir writable: ${join(tmp, '.w51')}`)
    expect(all).not.toContain('.w51-mcp')
  })
})

describe('buildCli — extra subcommands', () => {
  it('lets an extra subcommand override a built-in', async () => {
    const run = buildCli(SERVER, { serve: () => 0, extra: { doctor: () => 42 } })
    expect(await run(['doctor'])).toBe(42)
  })

  it('passes the remaining argv to an extra handler', async () => {
    const run = buildCli(SERVER, {
      serve: () => 0,
      extra: { auth: (rest) => (rest[0] === 'google' ? 0 : 2) }
    })
    expect(await run(['auth', 'google'])).toBe(0)
    expect(await run(['auth', 'github'])).toBe(2)
  })
})
