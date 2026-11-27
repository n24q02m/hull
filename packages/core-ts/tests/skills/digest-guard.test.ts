/**
 * Paths a real filesystem cannot reach deterministically: a file swapped between the
 * disk check and the SDK's read, a special file on Windows, and entries that cannot be
 * looked up.
 */

import type * as NodeFs from 'node:fs'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

import { McpServer } from '@modelcontextprotocol/server'
import type { SkillFileEntry, SkillFiles } from '@modelcontextprotocol/server/ext/skills'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({
  readdirFails: false,
  lstatDenied: new Set<string>(),
  extraDirents: new Map<string, string>(),
  extraEntries: new Map<string, SkillFileEntry[]>(),
  fakeDirs: new Map<string, SkillFileEntry[]>(),
  reads: new Map<string, Uint8Array>()
}))

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<typeof NodeFs>()
  const readdirSync = (dir: string, options: { withFileTypes: true }): NodeFs.Dirent[] => {
    if (state.readdirFails) throw new Error('denied')
    const entries = actual.readdirSync(dir, options)
    const extra = state.extraDirents.get(dir)
    if (extra === undefined) return entries
    const special = { name: extra, isSymbolicLink: () => false, isDirectory: () => false, isFile: () => false }
    return [...entries, special as unknown as NodeFs.Dirent]
  }
  const lstatSync = ((file: string, options?: NodeFs.StatSyncOptions) => {
    if (state.lstatDenied.has(file)) throw Object.assign(new Error('denied'), { code: 'EACCES' })
    return actual.lstatSync(file, options)
  }) as typeof actual.lstatSync
  return { ...actual, readdirSync, lstatSync }
})

vi.mock('@modelcontextprotocol/server/ext/skills/node', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@modelcontextprotocol/server/ext/skills/node')>()
  const nodeSkillFiles = (root: string): SkillFiles => {
    const inner = actual.nodeSkillFiles(root)
    return {
      list: async (dir) =>
        state.fakeDirs.get(dir) ?? [...(await inner.list(dir)), ...(state.extraEntries.get(dir) ?? [])],
      read: async (path) => state.reads.get(path) ?? inner.read(path)
    }
  }
  return { ...actual, nodeSkillFiles }
})

const { attachSkillsProvider, computeSkillDigests, SkillIntegrityError, verifySkillDigests, writeSkillDigests } =
  await import('../../src/skills/index.js')

const tmpDirs: string[] = []

afterEach(() => {
  state.readdirFails = false
  state.lstatDenied.clear()
  state.extraDirents.clear()
  state.extraEntries.clear()
  state.fakeDirs.clear()
  state.reads.clear()
  for (const dir of tmpDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true })
  }
})

function skillMd(name: string, description = 'Test skill description'): string {
  return `---\nname: ${name}\ndescription: ${description}\n---\n# ${name}\n`
}

function tree(): string {
  const root = resolve(mkdtempSync(join(tmpdir(), 'hull-skills-guard-')))
  tmpDirs.push(root)
  for (const [file, data] of [
    ['hello-skill/SKILL.md', skillMd('hello-skill')],
    ['hello-skill/references/guide.md', '# Guide\n'],
    ['ops/deploy/SKILL.md', skillMd('deploy')]
  ] as const) {
    const full = join(root, ...file.split('/'))
    mkdirSync(dirname(full), { recursive: true })
    writeFileSync(full, data)
  }
  writeSkillDigests(root)
  return root
}

function server(name: string): McpServer {
  return new McpServer({ name, version: '0.1.0' })
}

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text)

describe('skills digest guard on the bytes the SDK reads', () => {
  it('rejects a file swapped after the disk check', async () => {
    const root = tree()
    state.reads.set('hello-skill/SKILL.md', bytes(skillMd('hello-skill', 'Swapped')))
    const mcp = server('race')
    const attached = attachSkillsProvider(mcp, root)
    await expect(attached).rejects.toThrow(SkillIntegrityError)
    await expect(attached).rejects.toThrow('hello-skill/SKILL.md: digest mismatch')
  })

  it('rejects files that appear after the disk check without an entry', async () => {
    const root = tree()
    state.extraEntries.set('hello-skill', [{ name: 'late.md', kind: 'file' }])
    state.reads.set('hello-skill/late.md', bytes('added after the disk check\n'))
    await expect(attachSkillsProvider(server('race-strict'), root)).rejects.toThrow(
      'hello-skill/late.md: no digest entry'
    )
    const lenient = await attachSkillsProvider(server('race-lenient'), root, { strictDigests: false })
    expect(lenient.entries.map((s) => s.uri)).toContain('skill://hello-skill/SKILL.md')

    state.extraEntries.set('', [{ name: 'late-skill', kind: 'directory' }])
    state.fakeDirs.set('late-skill', [{ name: 'SKILL.md', kind: 'file' }])
    state.reads.set('late-skill/SKILL.md', bytes(skillMd('late-skill')))
    await expect(attachSkillsProvider(server('race-lenient-skill'), root, { strictDigests: false })).rejects.toThrow(
      'late-skill/SKILL.md: no digest entry'
    )
  })
})

describe('skills digest walk edge cases', () => {
  it('rejects any entry that is not a regular file', () => {
    const root = tree()
    state.extraDirents.set(join(root, 'hello-skill'), 'socket')
    expect(() => computeSkillDigests(root)).toThrow('hello-skill/socket: not a regular file')
  })

  it('reports an unreadable directory while computing and verifying', () => {
    const root = tree()
    state.readdirFails = true
    expect(() => computeSkillDigests(root)).toThrow(`Skills directory ${root} cannot be read: denied`)
    expect(() => verifySkillDigests(root)).toThrow(`Skills directory ${root} cannot be read: denied`)
  })

  it('reports a listed file that cannot be looked up', () => {
    const root = tree()
    state.lstatDenied.add(join(root, 'hello-skill', 'references', 'guide.md'))
    expect(() => verifySkillDigests(root)).toThrow(`Skills directory ${root} cannot be read: denied`)
  })

  it('keeps the guard on when the manifest cannot be looked up', async () => {
    const root = tree()
    writeFileSync(join(root, 'hello-skill', 'SKILL.md'), skillMd('hello-skill', 'Evil'))
    state.lstatDenied.add(join(root, 'skills.digest.json'))
    await expect(attachSkillsProvider(server('lookup-denied'), root)).rejects.toThrow(
      'hello-skill/SKILL.md: digest mismatch'
    )
  })
})
