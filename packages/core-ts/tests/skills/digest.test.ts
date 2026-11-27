import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { Client, StreamableHTTPClientTransport } from '@modelcontextprotocol/client'
import { McpServer } from '@modelcontextprotocol/server'
import { SkillCatalogError } from '@modelcontextprotocol/server/ext/skills'
import { afterEach, describe, expect, it } from 'vitest'

import * as publicApi from '../../src/index.js'
import {
  attachSkillsProvider,
  computeSkillDigests,
  SKILLS_DIGEST_FILENAME,
  SKILLS_DIGEST_VERSION,
  SkillIntegrityError,
  verifySkillDigests,
  writeSkillDigests
} from '../../src/skills/index.js'
import { type HttpServerHandle, runHttpServer } from '../../src/transport/local-server.js'

// Shared with the core-py suite: the same committed manifest must verify on both sides.
const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), '..', 'fixtures', 'skills-digest')

const tmpDirs: string[] = []
let handle: HttpServerHandle | undefined

afterEach(async () => {
  if (handle) {
    await handle.close()
    handle = undefined
  }
  for (const dir of tmpDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true })
  }
})

function tmp(): string {
  const dir = resolve(mkdtempSync(join(tmpdir(), 'hull-skills-digest-')))
  tmpDirs.push(dir)
  return dir
}

function sha(data: string | Uint8Array): string {
  return createHash('sha256').update(data).digest('hex')
}

function write(file: string, data: string | Uint8Array): string {
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, data)
  return file
}

function skillMd(name: string, description = 'Test skill description'): string {
  return `---\nname: ${name}\ndescription: ${description}\n---\n# ${name}\n\nUse this skill.\n`
}

/** Two skills (one nested), a hidden file inside a skill, and files outside any skill. */
function tree(root: string): string {
  write(join(root, 'hello-skill', 'SKILL.md'), skillMd('hello-skill', 'Greets callers'))
  write(join(root, 'hello-skill', 'references', 'guide.md'), '# Guide\n\nDetails.\n')
  write(join(root, 'hello-skill', '.hidden'), 'hidden but served\n')
  write(join(root, 'ops', 'deploy', 'SKILL.md'), skillMd('deploy'))
  write(join(root, 'README.md'), 'not part of any skill\n')
  write(join(root, '.cache', 'junk', 'SKILL.md'), 'hidden folders are never scanned\n')
  return root
}

/** Returns false when the platform refuses to create symlinks (Windows without the right). */
function trySymlink(target: string, link: string, type: 'file' | 'dir'): boolean {
  try {
    symlinkSync(target, link, type === 'dir' ? 'junction' : 'file')
    return true
  } catch {
    return false
  }
}

function problemsOf(fn: () => unknown): readonly string[] {
  try {
    fn()
  } catch (error) {
    expect(error).toBeInstanceOf(SkillIntegrityError)
    return (error as SkillIntegrityError).problems
  }
  throw new Error('expected a SkillIntegrityError')
}

function server(name: string): McpServer {
  return new McpServer({ name, version: '0.1.0' })
}

const HEX = 'a'.repeat(64)

describe('skills digest manifest generation', () => {
  it('exports the manifest constants and error from the skills module and the root', () => {
    expect(SKILLS_DIGEST_FILENAME).toBe('skills.digest.json')
    expect(SKILLS_DIGEST_VERSION).toBe(1)
    const exported: Record<string, unknown> = { ...publicApi }
    for (const name of ['computeSkillDigests', 'writeSkillDigests', 'verifySkillDigests', 'SkillIntegrityError']) {
      expect(typeof exported[name], name).toBe('function')
    }
    expect(exported.SKILLS_DIGEST_FILENAME).toBe('skills.digest.json')
  })

  it('names the error and keeps the problems list', () => {
    const error = new SkillIntegrityError('boom', ['a/SKILL.md: digest mismatch'])
    expect(error).toBeInstanceOf(Error)
    expect(error.name).toBe('SkillIntegrityError')
    expect(error.problems).toEqual(['a/SKILL.md: digest mismatch'])
    expect(new SkillIntegrityError('no list').problems).toEqual([])
  })

  it('hashes every file inside skill folders, hidden ones included', () => {
    const root = tree(tmp())
    const digests = computeSkillDigests(root)
    expect(digests).toEqual({
      'hello-skill/.hidden': sha('hidden but served\n'),
      'hello-skill/SKILL.md': sha(skillMd('hello-skill', 'Greets callers')),
      'hello-skill/references/guide.md': sha('# Guide\n\nDetails.\n'),
      'ops/deploy/SKILL.md': sha(skillMd('deploy'))
    })
    expect(Object.keys(digests)).toEqual(Object.keys(digests).sort())
  })

  it('includes a skill nested inside another skill', () => {
    const root = tmp()
    write(join(root, 'outer', 'SKILL.md'), skillMd('outer'))
    write(join(root, 'outer', 'inner', 'SKILL.md'), skillMd('inner'))
    expect(Object.keys(computeSkillDigests(root))).toEqual(['outer/SKILL.md', 'outer/inner/SKILL.md'])
  })

  it('ignores a SKILL.md at the top level', () => {
    const root = tmp()
    write(join(root, 'SKILL.md'), skillMd('top'))
    expect(computeSkillDigests(root)).toEqual({})
  })

  it('rejects a missing directory and a file path', () => {
    const root = tmp()
    expect(() => computeSkillDigests(join(root, 'nope'))).toThrow(SkillIntegrityError)
    expect(() => computeSkillDigests(join(root, 'nope'))).toThrow('does not exist')
    const file = write(join(root, 'file.txt'), 'x')
    expect(() => computeSkillDigests(file)).toThrow('not a directory')
  })

  it('rejects a symlink inside a skill', (ctx) => {
    const root = tree(tmp())
    if (!trySymlink(join(root, 'README.md'), join(root, 'hello-skill', 'link.md'), 'file')) ctx.skip()
    expect(problemsOf(() => computeSkillDigests(root))).toEqual(['hello-skill/link.md: symlinks are not allowed'])
  })

  it('rejects a symlinked skill folder', (ctx) => {
    const base = tmp()
    const root = tree(join(base, 'skills'))
    const outside = join(base, 'outside', 'linked')
    write(join(outside, 'SKILL.md'), skillMd('linked'))
    if (!trySymlink(outside, join(root, 'linked'), 'dir')) ctx.skip()
    expect(problemsOf(() => computeSkillDigests(root))).toEqual(['linked: symlinks are not allowed'])
  })

  it('writes the canonical manifest next to the skills', () => {
    const root = tree(tmp())
    const file = writeSkillDigests(root)
    expect(file).toBe(join(root, SKILLS_DIGEST_FILENAME))
    const raw = readFileSync(file, 'utf8')
    expect(raw.endsWith('}\n')).toBe(true)
    expect(raw).not.toContain('\r')
    const payload = JSON.parse(raw)
    expect(Object.keys(payload)).toEqual(['version', 'files'])
    expect(payload.version).toBe(1)
    expect(payload.files).toEqual(computeSkillDigests(root))
  })

  it('writes to a custom path', () => {
    const base = tmp()
    const root = tree(join(base, 'skills'))
    const out = join(base, 'digests.json')
    expect(writeSkillDigests(root, out)).toBe(out)
    expect(JSON.parse(readFileSync(out, 'utf8')).files).toEqual(computeSkillDigests(root))
    expect(() => readFileSync(join(root, SKILLS_DIGEST_FILENAME))).toThrow()
  })
})

describe('skills digest verification', () => {
  it('accepts a valid manifest from the default file, an explicit path, or a mapping', () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    const expected = computeSkillDigests(root)
    expect(verifySkillDigests(root)).toEqual(expected)
    expect(verifySkillDigests(root, join(root, SKILLS_DIGEST_FILENAME))).toEqual(expected)
    expect(verifySkillDigests(root, expected)).toEqual(expected)
  })

  it('rejects one tampered byte', () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    const guide = join(root, 'hello-skill', 'references', 'guide.md')
    const data = readFileSync(guide)
    data[0] = (data[0] as number) ^ 0x01
    writeFileSync(guide, data)
    expect(problemsOf(() => verifySkillDigests(root))).toEqual(['hello-skill/references/guide.md: digest mismatch'])
    expect(() => verifySkillDigests(root)).toThrow('1 skill integrity problem(s)')
  })

  it('rejects a SKILL.md without an entry', () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    write(join(root, 'new-skill', 'SKILL.md'), skillMd('new-skill'))
    expect(problemsOf(() => verifySkillDigests(root))).toEqual(['new-skill/SKILL.md: no digest entry'])
  })

  it('rejects an entry whose file is missing, including below a file', () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    unlinkSync(join(root, 'hello-skill', 'references', 'guide.md'))
    expect(problemsOf(() => verifySkillDigests(root))).toEqual([
      'hello-skill/references/guide.md: listed in the manifest but missing'
    ])
    const digests = { ...computeSkillDigests(root), 'hello-skill/SKILL.md/x': HEX }
    expect(problemsOf(() => verifySkillDigests(root, digests))).toEqual([
      'hello-skill/SKILL.md/x: listed in the manifest but missing'
    ])
  })

  it('rejects an entry that is a directory', () => {
    const root = tree(tmp())
    const digests = { ...computeSkillDigests(root), 'hello-skill/references': '0'.repeat(64) }
    expect(problemsOf(() => verifySkillDigests(root, digests))).toEqual([
      'hello-skill/references: listed in the manifest but not a regular file'
    ])
  })

  it('checks a listed file outside any skill', () => {
    const root = tree(tmp())
    const digests = { ...computeSkillDigests(root), 'README.md': sha('not part of any skill\n') }
    expect(verifySkillDigests(root, digests)).toEqual(digests)
    expect(() => verifySkillDigests(root, { ...digests, 'README.md': '0'.repeat(64) })).toThrow(
      'README.md: digest mismatch'
    )
  })

  it('in strict mode rejects an unlisted supporting file', () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    write(join(root, 'hello-skill', 'scripts', 'run.sh'), 'echo injected\n')
    expect(problemsOf(() => verifySkillDigests(root))).toEqual(['hello-skill/scripts/run.sh: no digest entry'])
  })

  it('in non-strict mode allows unlisted supporting files only', () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    write(join(root, 'hello-skill', 'scripts', 'run.sh'), 'echo extra\n')
    verifySkillDigests(root, undefined, { strict: false })
    write(join(root, 'other', 'SKILL.md'), skillMd('other'))
    writeFileSync(join(root, 'ops', 'deploy', 'SKILL.md'), skillMd('deploy', 'changed'))
    expect(problemsOf(() => verifySkillDigests(root, undefined, { strict: false }))).toEqual([
      'ops/deploy/SKILL.md: digest mismatch',
      'other/SKILL.md: no digest entry'
    ])
  })

  it('reports every problem at once', () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    writeFileSync(join(root, 'hello-skill', '.hidden'), 'changed\n')
    unlinkSync(join(root, 'hello-skill', 'references', 'guide.md'))
    write(join(root, 'added', 'SKILL.md'), skillMd('added'))
    expect(problemsOf(() => verifySkillDigests(root))).toEqual([
      'hello-skill/.hidden: digest mismatch',
      'hello-skill/references/guide.md: listed in the manifest but missing',
      'added/SKILL.md: no digest entry'
    ])
    expect(() => verifySkillDigests(root)).toThrow(`3 skill integrity problem(s) in ${root}:\n- `)
  })

  it('rejects a missing directory and a missing manifest', () => {
    const root = tree(tmp())
    expect(() => verifySkillDigests(join(root, 'nope'), {})).toThrow('Skills directory does not exist')
    expect(() => verifySkillDigests(root)).toThrow('Skills digest manifest not found')
    expect(() => verifySkillDigests(root, join(root, 'elsewhere.json'))).toThrow('Skills digest manifest not found')
  })

  it('rejects a symlink anywhere in the tree', (ctx) => {
    const root = tree(tmp())
    writeSkillDigests(root)
    if (!trySymlink(join(root, 'hello-skill'), join(root, 'alias'), 'dir')) ctx.skip()
    expect(problemsOf(() => verifySkillDigests(root))).toEqual(['alias: symlinks are not allowed'])
  })

  it.each([
    ['{not json', 'not valid JSON'],
    [Buffer.from([0xff, 0xfe]), 'not valid JSON'],
    [Buffer.from([0xef, 0xbb, 0xbf, 0x7b, 0x7d]), 'not valid JSON'],
    ['[]', 'must be a JSON object'],
    ['null', 'must be a JSON object'],
    [JSON.stringify({ version: 2, files: {} }), 'version must be 1'],
    [JSON.stringify({ version: true, files: {} }), 'version must be 1'],
    [JSON.stringify({ version: '1', files: {} }), 'version must be 1'],
    [JSON.stringify({ files: {} }), 'version must be 1'],
    [JSON.stringify({ version: 1 }), 'files must be an object'],
    [JSON.stringify({ version: 1, files: [] }), 'files must be an object'],
    [JSON.stringify({ version: 1, files: null }), 'files must be an object'],
    [JSON.stringify({ version: 1, files: {}, extra: 1 }), "unexpected key 'extra'"],
    [JSON.stringify({ version: 1, files: { 'a/SKILL.md': 'A'.repeat(64) } }), '64 lowercase hex'],
    [JSON.stringify({ version: 1, files: { 'a/SKILL.md': 'ab' } }), '64 lowercase hex'],
    [JSON.stringify({ version: 1, files: { 'a/SKILL.md': 1 } }), '64 lowercase hex'],
    [JSON.stringify({ version: 1, files: { '../x': HEX } }), "invalid path '../x'"],
    [JSON.stringify({ version: 1, files: { '/abs': HEX } }), "invalid path '/abs'"],
    [JSON.stringify({ version: 1, files: { 'a\\b': HEX } }), 'invalid path'],
    [JSON.stringify({ version: 1, files: { 'C:x': HEX } }), "invalid path 'C:x'"],
    [JSON.stringify({ version: 1, files: { 'a//b': HEX } }), "invalid path 'a//b'"],
    [JSON.stringify({ version: 1, files: { 'a/./b': HEX } }), "invalid path 'a/./b'"],
    [JSON.stringify({ version: 1, files: { '': HEX } }), "invalid path ''"],
    [JSON.stringify({ version: 1, files: { 'a\u0000b': HEX } }), 'invalid path']
  ])('rejects an invalid manifest (%s)', (content, reason) => {
    const root = tree(tmp())
    writeFileSync(join(root, SKILLS_DIGEST_FILENAME), content)
    expect(() => verifySkillDigests(root)).toThrow('Invalid skills digest manifest')
    expect(() => verifySkillDigests(root)).toThrow(reason)
  })

  it('accepts the version written as a float, as core-py does', () => {
    const root = tree(tmp())
    const files = computeSkillDigests(root)
    writeFileSync(join(root, SKILLS_DIGEST_FILENAME), `{"version": 1.0, "files": ${JSON.stringify(files)}}`)
    expect(verifySkillDigests(root)).toEqual(files)
  })

  it('treats a __proto__ entry as a plain path', () => {
    const root = tree(tmp())
    const files = computeSkillDigests(root)
    const manifest = `{"version": 1, "files": {"__proto__": "${HEX}", ${JSON.stringify(files).slice(1)}}`
    writeFileSync(join(root, SKILLS_DIGEST_FILENAME), manifest)
    expect(problemsOf(() => verifySkillDigests(root))).toEqual(['__proto__: listed in the manifest but missing'])
  })

  it('rejects an unreadable manifest', () => {
    const root = tree(tmp())
    mkdirSync(join(root, SKILLS_DIGEST_FILENAME))
    expect(() => verifySkillDigests(root)).toThrow('cannot be read')
  })

  it.each([
    [{ 'a/SKILL.md': 'nothex' }, '64 lowercase hex'],
    [{ '../a': HEX }, "invalid path '../a'"],
    [{ 'a/SKILL.md': 1 }, '64 lowercase hex']
  ])('rejects an invalid in-memory mapping (%o)', (mapping, reason) => {
    const root = tree(tmp())
    const digests = mapping as unknown as Record<string, string>
    expect(() => verifySkillDigests(root, digests)).toThrow('Invalid skills digest mapping')
    expect(() => verifySkillDigests(root, digests)).toThrow(reason)
  })
})

describe('attachSkillsProvider with a digest manifest', () => {
  it('keeps the old behaviour when there is no manifest and no option', async () => {
    const root = tree(tmp())
    const catalog = await attachSkillsProvider(server('no-manifest'), root)
    expect(catalog.entries.map((s) => s.uri)).toContain('skill://hello-skill/SKILL.md')
  })

  it('verifies a manifest found in the directory', async () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    const catalog = await attachSkillsProvider(server('auto'), root)
    expect(catalog.entries.map((s) => s.uri)).toEqual(['skill://hello-skill/SKILL.md', 'skill://ops/deploy/SKILL.md'])
  })

  it('attaches nothing when a skill was tampered with', async () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    writeFileSync(join(root, 'hello-skill', 'SKILL.md'), skillMd('hello-skill', 'Evil'))
    const mcp = server('tampered')
    await expect(attachSkillsProvider(mcp, root)).rejects.toThrow(SkillIntegrityError)
    await expect(attachSkillsProvider(mcp, root)).rejects.toThrow('hello-skill/SKILL.md: digest mismatch')
    // One skills extension per server: this only succeeds because nothing was attached.
    await expect(attachSkillsProvider(mcp, tree(tmp()))).resolves.toBeDefined()
  })

  it('attaches nothing when a skill has no entry', async () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    write(join(root, 'rogue', 'SKILL.md'), skillMd('rogue'))
    await expect(attachSkillsProvider(server('unlisted'), root)).rejects.toThrow('rogue/SKILL.md: no digest entry')
  })

  it('rejects an invalid manifest found in the directory', async () => {
    const root = tree(tmp())
    writeFileSync(join(root, SKILLS_DIGEST_FILENAME), '{}')
    await expect(attachSkillsProvider(server('invalid'), root)).rejects.toThrow('Invalid skills digest manifest')
  })

  it('treats a dangling manifest symlink as present', async (ctx) => {
    const base = tmp()
    const root = tree(join(base, 'skills'))
    if (!trySymlink(join(base, 'gone.json'), join(root, SKILLS_DIGEST_FILENAME), 'file')) ctx.skip()
    await expect(attachSkillsProvider(server('dangling'), root)).rejects.toThrow('Skills digest manifest not found')
  })

  it('with requireDigests fails when there is no manifest', async () => {
    const root = tree(tmp())
    await expect(attachSkillsProvider(server('required'), root, { requireDigests: true })).rejects.toThrow(
      'Skills digest manifest not found'
    )
  })

  it('with an explicit option rejects a missing directory or a file path', async () => {
    const base = tmp()
    await expect(attachSkillsProvider(server('gone'), join(base, 'nope'), { requireDigests: true })).rejects.toThrow(
      SkillIntegrityError
    )
    await expect(attachSkillsProvider(server('gone'), join(base, 'nope'), { digests: {} })).rejects.toThrow(
      'Skills directory does not exist'
    )
    const file = write(join(base, 'file.txt'), 'x')
    await expect(attachSkillsProvider(server('file'), file, { digests: {} })).rejects.toThrow('not a directory')
  })

  it('accepts an in-memory mapping', async () => {
    const root = tree(tmp())
    const digests = computeSkillDigests(root)
    const catalog = await attachSkillsProvider(server('mapping'), root, { digests })
    expect(catalog.entries.map((s) => s.uri)).toContain('skill://ops/deploy/SKILL.md')
    const bad = { ...digests, 'ops/deploy/SKILL.md': '0'.repeat(64) }
    await expect(attachSkillsProvider(server('mapping-bad'), root, { digests: bad })).rejects.toThrow(
      'ops/deploy/SKILL.md: digest mismatch'
    )
  })

  it('accepts a manifest path outside the directory', async () => {
    const base = tmp()
    const root = tree(join(base, 'skills'))
    const manifest = writeSkillDigests(root, join(base, 'pinned.json'))
    const catalog = await attachSkillsProvider(server('external'), root, { digests: manifest })
    expect(catalog.entries.map((s) => s.uri)).toContain('skill://hello-skill/SKILL.md')
  })

  it('with strictDigests false allows an unlisted supporting file', async () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    write(join(root, 'hello-skill', 'notes.md'), 'extra\n')
    await expect(attachSkillsProvider(server('strict'), root)).rejects.toThrow('hello-skill/notes.md: no digest entry')
    const catalog = await attachSkillsProvider(server('lenient'), root, { strictDigests: false })
    expect(catalog.entries.map((s) => s.uri)).toContain('skill://hello-skill/SKILL.md')
  })

  it('still passes the catalog options through to the SDK', async () => {
    const root = tree(tmp())
    write(join(root, 'SKILL.md'), skillMd('top'))
    const digests = computeSkillDigests(root)
    // The SDK `strict` turns its root-level SKILL.md warning into an error.
    await expect(attachSkillsProvider(server('sdk-strict'), root, { digests, strict: true })).rejects.toThrow(
      SkillCatalogError
    )
    const catalog = await attachSkillsProvider(server('prefixed'), root, { digests, prefix: 'acme' })
    expect(catalog.entries.map((s) => s.uri)).toContain('skill://acme/hello-skill/SKILL.md')
  })

  it('refuses to serve a file changed after attach', async () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    const catalog = await attachSkillsProvider(server('serve'), root)
    writeFileSync(join(root, 'hello-skill', 'SKILL.md'), skillMd('hello-skill', 'Tampered after start'))

    handle = await runHttpServer(
      () => {
        const mcp = server('serve-http')
        catalog.attach(mcp)
        return mcp
      },
      { serverName: `test-skills-digest-${Date.now()}`, port: 0 }
    )
    const client = new Client({ name: 'skills-digest-client', version: '0.1.0' })
    await client.connect(new StreamableHTTPClientTransport(new URL(`http://${handle.host}:${handle.port}/mcp`)))
    try {
      await expect(client.readResource({ uri: 'skill://hello-skill/SKILL.md' })).rejects.toThrow()
      const deploy = await client.readResource({ uri: 'skill://ops/deploy/SKILL.md' })
      expect(JSON.stringify(deploy.contents)).toContain('Use this skill.')
    } finally {
      await client.close()
    }
  })

  it('rejects a reload that fails the check and keeps the verified catalog', async () => {
    const root = tree(tmp())
    writeSkillDigests(root)
    const catalog = await attachSkillsProvider(server('reload'), root)
    const before = catalog.entries.map((s) => s.uri)

    write(join(root, 'late', 'SKILL.md'), skillMd('late'))
    await expect(catalog.reload()).rejects.toThrow('late/SKILL.md: no digest entry')
    rmSync(join(root, 'late'), { recursive: true })

    writeFileSync(join(root, 'ops', 'deploy', 'SKILL.md'), skillMd('deploy', 'Tampered'))
    await expect(catalog.reload()).rejects.toThrow('ops/deploy/SKILL.md: digest mismatch')
    expect(catalog.entries.map((s) => s.uri)).toEqual(before)
  })
})

describe('cross-language fixture', () => {
  const committed = (): Record<string, string> =>
    JSON.parse(readFileSync(join(FIXTURE, SKILLS_DIGEST_FILENAME), 'utf8')).files

  it('verifies the committed manifest', () => {
    expect(verifySkillDigests(FIXTURE)).toEqual(committed())
  })

  it('generates the committed manifest byte for byte', () => {
    const out = writeSkillDigests(FIXTURE, join(tmp(), 'regenerated.json'))
    expect(readFileSync(out).equals(readFileSync(join(FIXTURE, SKILLS_DIGEST_FILENAME)))).toBe(true)
  })

  it('mounts with the committed manifest', async () => {
    const catalog = await attachSkillsProvider(server('fixture'), FIXTURE)
    expect(catalog.entries.map((s) => s.uri)).toEqual(['skill://hello-skill/SKILL.md', 'skill://ops/deploy/SKILL.md'])
  })
})
