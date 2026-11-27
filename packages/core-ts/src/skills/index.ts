/**
 * MCP 2026-07-28 Skills extension helper for @n24q02m/hull.
 *
 * Wraps `@modelcontextprotocol/server/ext/skills` and
 * `@modelcontextprotocol/server/ext/skills/node` with explicit directory
 * validation before attaching the catalog to an `McpServer`.
 *
 * Optional integrity guard: a `skills.digest.json` manifest pins the SHA-256 of
 * every file inside each skill folder. When the guard is on, a skill whose bytes
 * differ from the manifest, or a `SKILL.md` with no manifest entry, is never
 * attached. The manifest format and the file walk are shared with `hull-core`
 * (core-py), so one manifest verifies in both languages.
 */

import { createHash } from 'node:crypto'
import * as fs from 'node:fs'
import * as path from 'node:path'

import type { McpServer } from '@modelcontextprotocol/server'
import {
  SKILLS_EXTENSION_ID,
  SkillCatalog,
  type SkillCatalogChanges,
  SkillCatalogError,
  type SkillCatalogOptions,
  type SkillFiles,
  type SkillWarning
} from '@modelcontextprotocol/server/ext/skills'
import { loadSkills, nodeSkillFiles } from '@modelcontextprotocol/server/ext/skills/node'

export {
  loadSkills,
  SKILLS_EXTENSION_ID,
  type SkillCatalog,
  type SkillCatalogChanges,
  SkillCatalogError,
  type SkillCatalogOptions,
  type SkillWarning
}

/** Default manifest name, looked up at the root of the skills directory. */
export const SKILLS_DIGEST_FILENAME = 'skills.digest.json'
export const SKILLS_DIGEST_VERSION = 1

/** `{ relative POSIX path: lowercase hex SHA-256 }`, as stored under `files` in the manifest. */
export type SkillDigests = Readonly<Record<string, string>>

/** A manifest path, or the digests themselves. */
export type SkillDigestSource = string | SkillDigests

export interface VerifySkillDigestsOptions {
  /**
   * Require an entry for every file inside a skill folder (default `true`). With
   * `false`, only each `SKILL.md` needs one; listed files are still verified.
   */
  strict?: boolean
}

export interface AttachSkillsOptions extends SkillCatalogOptions {
  /** Manifest path or in-memory digests. Turns the integrity guard on. */
  digests?: SkillDigestSource
  /** Fail when no manifest is found instead of attaching unchecked. Turns the guard on. */
  requireDigests?: boolean
  /** Same as {@link VerifySkillDigestsOptions.strict}; default `true`. */
  strictDigests?: boolean
}

/**
 * A skills directory failed its digest check; nothing from it is attached.
 * `problems` lists one `"<relative path>: <reason>"` line per failing file and is
 * empty when the failure is about the directory or the manifest itself.
 */
export class SkillIntegrityError extends Error {
  readonly problems: readonly string[]

  constructor(message: string, problems: readonly string[] = []) {
    super(message)
    this.name = 'SkillIntegrityError'
    this.problems = Object.freeze([...problems])
  }
}

const SKILL_FILE = 'SKILL.md'
const HEX_DIGEST = /^[0-9a-f]{64}$/
const DRIVE_PREFIX = /^[A-Za-z]:/
const MANIFEST_KEYS = ['version', 'files']

/**
 * Return `{ relative POSIX path: SHA-256 hex }` for every file inside each skill
 * folder. A skill folder is a non-hidden folder below `skillsDir` that holds a
 * `SKILL.md`; every regular file under it counts, hidden ones included. Files
 * outside skill folders are ignored. Any symlink in the walk, or a special file
 * inside a skill, throws {@link SkillIntegrityError}.
 */
export function computeSkillDigests(skillsDir: string): Record<string, string> {
  const root = existingDir(skillsDir)
  const { files, problems } = readingFs(root, () => walk(root))
  if (problems.length > 0) throw integrityError(root, problems)
  return readingFs(root, () => Object.fromEntries(files.map((rel) => [rel, hashFile(root, rel)])))
}

/**
 * Write the canonical manifest for `skillsDir` and return its path. The default
 * location is `<skillsDir>/skills.digest.json`, which sits outside every skill
 * folder and so never lists itself. The output is byte-for-byte what core-py writes.
 */
export function writeSkillDigests(skillsDir: string, output?: string): string {
  const digests = computeSkillDigests(skillsDir)
  const target = output ?? path.join(path.resolve(skillsDir), SKILLS_DIGEST_FILENAME)
  fs.writeFileSync(target, serialize(digests))
  return target
}

/**
 * Check `skillsDir` against a manifest and return the verified digests.
 * `digests` is a manifest path, an in-memory mapping, or omitted for
 * `<skillsDir>/skills.digest.json` (which must then exist). Every entry must be a
 * regular file with a matching digest and every `SKILL.md` must have an entry;
 * see {@link VerifySkillDigestsOptions.strict} for supporting files. All problems
 * are reported in one {@link SkillIntegrityError}.
 */
export function verifySkillDigests(
  skillsDir: string,
  digests?: SkillDigestSource,
  options: VerifySkillDigestsOptions = {}
): Record<string, string> {
  const root = existingDir(skillsDir)
  const expected = expectedDigests(root, digests)
  verifyTree(root, expected, options.strict ?? true)
  return Object.fromEntries(expected)
}

/**
 * Validate `skillsDir` exists as a directory, load all skills into a
 * `SkillCatalog`, attach it to `mcp`, and return the catalog.
 *
 * The integrity guard is on when `digests` is given, when `requireDigests` is
 * set, or when `<skillsDir>/skills.digest.json` exists. Every catalog build (this
 * load and each `reload()`) then verifies the directory against the manifest
 * before reading a skill, and every file the SDK reads (at load, on reload and
 * when serving) must match its digest. A failing load throws
 * {@link SkillIntegrityError} and attaches nothing; a failing `reload()` rejects
 * and the verified catalog keeps serving. With no manifest and no option the
 * directory is attached unchecked, as before. The remaining options go to the SDK.
 */
export async function attachSkillsProvider(
  mcp: McpServer,
  skillsDir: string,
  options: AttachSkillsOptions = {}
): Promise<SkillCatalog> {
  const { digests, requireDigests = false, strictDigests = true, ...catalogOptions } = options
  const trimmed = skillsDir.trim()
  if (!trimmed) {
    throw new Error('skillsDir must be a non-empty directory path')
  }
  const resolved = path.resolve(trimmed)
  const guarded =
    digests !== undefined || requireDigests || manifestPresent(path.join(resolved, SKILLS_DIGEST_FILENAME))

  let catalog: SkillCatalog
  if (guarded) {
    const root = existingDir(resolved)
    catalog = await loadVerified(root, expectedDigests(root, digests), strictDigests, catalogOptions)
  } else {
    if (!fs.existsSync(resolved)) {
      throw new Error(`Skills directory does not exist: ${resolved}`)
    }
    if (!fs.statSync(resolved).isDirectory()) {
      throw new Error(`Skills path is not a directory: ${resolved}`)
    }
    catalog = await loadSkills(resolved, catalogOptions)
  }
  catalog.attach(mcp)
  return catalog
}

async function loadVerified(
  root: string,
  expected: ReadonlyMap<string, string>,
  strict: boolean,
  options: SkillCatalogOptions
): Promise<SkillCatalog> {
  let failure: SkillIntegrityError | undefined
  const files = guardedSkillFiles(root, expected, strict, (error) => {
    failure = error
  })
  try {
    return await SkillCatalog.load(files, options)
  } catch (error) {
    // The SDK wraps adapter errors in SkillCatalogError; surface the integrity failure itself.
    throw failure ?? error
  }
}

/**
 * The SDK adapter over `root`, checked against `expected`. Each catalog build
 * starts by listing the root, so that is where the whole directory is verified;
 * each read is then checked against its digest, which closes the gap between
 * that check and the SDK reading the file.
 */
function guardedSkillFiles(
  root: string,
  expected: ReadonlyMap<string, string>,
  strict: boolean,
  onFailure: (error: SkillIntegrityError) => void
): SkillFiles {
  const inner = nodeSkillFiles(root)
  const fail = (problem: string): never => {
    const error = integrityError(root, [problem])
    onFailure(error)
    throw error
  }
  return {
    async list(dir) {
      if (dir === '') {
        try {
          verifyTree(root, expected, strict)
        } catch (error) {
          onFailure(error as SkillIntegrityError)
          throw error
        }
      }
      return inner.list(dir)
    },
    async read(file) {
      const bytes = await inner.read(file)
      const digest = expected.get(file)
      if (digest === undefined) {
        if (needsEntry(file, strict)) fail(`${file}: no digest entry`)
      } else if (sha256(bytes) !== digest) {
        fail(`${file}: digest mismatch`)
      }
      return bytes
    }
  }
}

function existingDir(skillsDir: string): string {
  const root = path.resolve(skillsDir)
  const stat = fs.statSync(root, { throwIfNoEntry: false })
  if (stat === undefined) throw new SkillIntegrityError(`Skills directory does not exist: ${root}`)
  if (!stat.isDirectory()) throw new SkillIntegrityError(`Skills path is not a directory: ${root}`)
  return root
}

function expectedDigests(root: string, digests: SkillDigestSource | undefined): Map<string, string> {
  if (digests === undefined) return loadManifest(path.join(root, SKILLS_DIGEST_FILENAME))
  if (typeof digests === 'string') return loadManifest(path.resolve(digests))
  return validated(digests, 'Invalid skills digest mapping')
}

function loadManifest(file: string): Map<string, string> {
  let raw: Buffer
  try {
    raw = fs.readFileSync(file)
  } catch (error) {
    if (hasCode(error, 'ENOENT')) throw new SkillIntegrityError(`Skills digest manifest not found: ${file}`)
    throw new SkillIntegrityError(`Skills digest manifest ${file} cannot be read: ${(error as Error).message}`)
  }
  const label = `Invalid skills digest manifest ${file}`
  let payload: unknown
  try {
    // ignoreBOM keeps a byte order mark in the text, so JSON.parse rejects it as core-py does.
    payload = JSON.parse(new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(raw))
  } catch (error) {
    throw new SkillIntegrityError(`${label}: not valid JSON (${(error as Error).message})`)
  }
  if (!isObject(payload)) throw new SkillIntegrityError(`${label}: must be a JSON object`)
  for (const key of Object.keys(payload)) {
    if (!MANIFEST_KEYS.includes(key)) throw new SkillIntegrityError(`${label}: unexpected key '${key}'`)
  }
  if (payload.version !== SKILLS_DIGEST_VERSION) {
    throw new SkillIntegrityError(`${label}: version must be ${SKILLS_DIGEST_VERSION}`)
  }
  if (!isObject(payload.files)) throw new SkillIntegrityError(`${label}: files must be an object`)
  return validated(payload.files, label)
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function validated(entries: Readonly<Record<string, unknown>>, label: string): Map<string, string> {
  const result: [string, string][] = []
  for (const [key, value] of Object.entries(entries)) {
    if (!isRelativePosix(key)) throw new SkillIntegrityError(`${label}: invalid path '${key}'`)
    if (typeof value !== 'string' || !HEX_DIGEST.test(value)) {
      throw new SkillIntegrityError(`${label}: ${key}: digest must be 64 lowercase hex characters`)
    }
    result.push([key, value])
  }
  return new Map(result.sort(([a], [b]) => compareNames(a, b)))
}

function isRelativePosix(key: string): boolean {
  if (!key || key.startsWith('/') || key.includes('\\') || key.includes('\u0000') || DRIVE_PREFIX.test(key)) {
    return false
  }
  return key.split('/').every((part) => part !== '' && part !== '.' && part !== '..')
}

/** Code point order, the same order core-py sorts by. */
function compareNames(a: string, b: string): number {
  return Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8'))
}

function serialize(digests: Readonly<Record<string, string>>): string {
  const files = Object.fromEntries(Object.entries(digests).sort(([a], [b]) => compareNames(a, b)))
  return `${JSON.stringify({ version: SKILLS_DIGEST_VERSION, files }, null, 2)}\n`
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

function hashFile(root: string, rel: string): string {
  return sha256(fs.readFileSync(path.join(root, ...rel.split('/'))))
}

/** Files inside skill folders and problems, both as relative POSIX paths. */
function walk(root: string): { files: string[]; problems: string[] } {
  const files: string[] = []
  const problems: string[] = []
  const visit = (dir: string, prefix: string, inSkill: boolean): void => {
    const entries = fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => compareNames(a.name, b.name))
    for (const entry of entries) {
      const rel = prefix + entry.name
      // Hidden folders such as .git are never scanned for skills.
      if (!inSkill && entry.name.startsWith('.')) continue
      if (entry.isSymbolicLink()) {
        problems.push(`${rel}: symlinks are not allowed`)
      } else if (entry.isDirectory()) {
        const full = path.join(dir, entry.name)
        visit(full, `${rel}/`, inSkill || holdsSkillFile(full))
      } else if (inSkill) {
        if (entry.isFile()) files.push(rel)
        else problems.push(`${rel}: not a regular file`)
      }
    }
  }
  visit(root, '', false)
  return { files: files.sort(compareNames), problems }
}

function holdsSkillFile(dir: string): boolean {
  const stat = fs.lstatSync(path.join(dir, SKILL_FILE), { throwIfNoEntry: false })
  return stat !== undefined && !stat.isDirectory()
}

function needsEntry(rel: string, strict: boolean): boolean {
  return strict || path.posix.basename(rel) === SKILL_FILE
}

function verifyTree(root: string, expected: ReadonlyMap<string, string>, strict: boolean): void {
  const problems: string[] = []
  const { files, problems: walkProblems } = readingFs(root, () => {
    for (const [rel, digest] of expected) {
      const stat = lstatOrMissing(path.join(root, ...rel.split('/')))
      if (stat === undefined) problems.push(`${rel}: listed in the manifest but missing`)
      else if (!stat.isFile()) problems.push(`${rel}: listed in the manifest but not a regular file`)
      else if (hashFile(root, rel) !== digest) problems.push(`${rel}: digest mismatch`)
    }
    return walk(root)
  })
  problems.push(...walkProblems)
  for (const rel of files) {
    if (!expected.has(rel) && needsEntry(rel, strict)) problems.push(`${rel}: no digest entry`)
  }
  if (problems.length > 0) throw integrityError(root, problems)
}

function lstatOrMissing(file: string): fs.Stats | undefined {
  try {
    return fs.lstatSync(file)
  } catch (error) {
    // ENOTDIR: a path that runs through a file, such as `a/SKILL.md/x` on POSIX.
    if (hasCode(error, 'ENOENT') || hasCode(error, 'ENOTDIR')) return undefined
    throw error
  }
}

function hasCode(error: unknown, code: string): boolean {
  return (error as NodeJS.ErrnoException).code === code
}

function readingFs<T>(root: string, run: () => T): T {
  try {
    return run()
  } catch (error) {
    throw new SkillIntegrityError(`Skills directory ${root} cannot be read: ${(error as Error).message}`)
  }
}

function integrityError(root: string, problems: readonly string[]): SkillIntegrityError {
  const lines = problems.map((problem) => `- ${problem}`).join('\n')
  return new SkillIntegrityError(`${problems.length} skill integrity problem(s) in ${root}:\n${lines}`, problems)
}

/**
 * Whether the guard must turn on for `file`; only a definite "not there" says no. A
 * dangling symlink counts as present (and then fails to load), and so does a path that
 * cannot be looked up, so an unreadable manifest never attaches the skills unchecked.
 */
function manifestPresent(file: string): boolean {
  try {
    return lstatOrMissing(file) !== undefined
  } catch {
    return true
  }
}
