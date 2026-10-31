/**
 * Pluggable credential backends.
 *
 * A backend stores and retrieves OPAQUE ciphertext blobs keyed by a string.
 * Encryption and key derivation live in PerPluginStore; backends never see
 * plaintext. This seam lets MCP servers run serverless (e.g. Cloudflare Workers
 * KV) while stdio/VM deployments keep the on-disk layout via LocalFsBackend.
 */

import { existsSync } from 'node:fs'
import { chmod, mkdir, readFile, unlink } from 'node:fs/promises'
import { dirname, join, resolve, sep } from 'node:path'
import { atomicWriteFile } from './atomic-write.js'
import { dataDirName } from './data-dir.js'
import { getHomeDir } from './home-dir.js'

/** Stores opaque ciphertext blobs keyed by a string. */
export interface CredentialBackend {
  get(key: string): Promise<Buffer | null>
  put(key: string, blob: Buffer): Promise<void>
  delete(key: string): Promise<void>
}

/** Map-backed backend, primarily for tests. */
export class InMemoryBackend implements CredentialBackend {
  private store = new Map<string, Buffer>()

  async get(key: string): Promise<Buffer | null> {
    return this.store.get(key) ?? null
  }

  async put(key: string, blob: Buffer): Promise<void> {
    this.store.set(key, blob)
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key)
  }
}

function validateComponent(name: string): string {
  if (!name || name === '.' || name === '..' || name.includes('/') || name.includes('\\') || name.includes('\x00')) {
    throw new Error(`Invalid path component: ${name}`)
  }
  return name
}

/** Default base directory for a plugin: the current `~/.<plugin>` layout. */
function defaultBaseDir(plugin: string): string {
  return join(getHomeDir(), dataDirName(plugin))
}

/**
 * Map a backend key to its on-disk path under `baseDir(plugin)`.
 *
 * With the default layout:
 *
 *   "<plugin>/config"            -> ~/.<plugin>/config.json
 *   "<plugin>/subs/<sub>/config" -> ~/.<plugin>/subs/<sub>/config.json
 */
function keyToPath(key: string, baseDir: (plugin: string) => string): string {
  const slash = key.indexOf('/')
  const plugin = slash === -1 ? key : key.slice(0, slash)
  const rest = slash === -1 ? '' : key.slice(slash + 1)
  validateComponent(plugin)

  const base = baseDir(plugin)
  let path: string
  if (rest === 'config') {
    path = join(base, 'config.json')
  } else if (rest.startsWith('subs/') && rest.endsWith('/config')) {
    const sub = rest.slice('subs/'.length, -'/config'.length)
    validateComponent(sub)
    path = join(base, 'subs', sub, 'config.json')
  } else {
    throw new Error(`Invalid backend key: ${key}`)
  }

  // Defense-in-depth: validateComponent already rejects separators/./../empty/NUL.
  const baseResolved = resolve(base)
  const pathResolved = resolve(path)
  if (pathResolved !== baseResolved && !pathResolved.startsWith(baseResolved + sep)) {
    throw new Error(`Path escapes base: ${key}`)
  }
  return path
}

export interface LocalFsBackendOptions {
  /**
   * Map a (validated) plugin name to the absolute directory its blobs live in.
   * Defaults to the current layout, `~/.<plugin>` (resolved against the home
   * dir at call time). PerPluginStore passes a fixed directory here to read
   * and clear the legacy `~/.<plugin>-mcp` / `~/.better-<plugin>-mcp` layouts.
   */
  baseDir?: (plugin: string) => string
}

/** Stores blobs on local disk, preserving the per-plugin layout. */
export class LocalFsBackend implements CredentialBackend {
  /**
   * True when this backend writes the current `~/.<plugin>` layout (no
   * `baseDir` override). PerPluginStore only consults the legacy on-disk
   * layouts for such a backend; a custom-layout backend is taken as-is.
   */
  readonly isDefaultLayout: boolean
  private readonly baseDir: (plugin: string) => string
  // Directories already initialized in this process: mkdir+chmod run exactly
  // once per directory so an existing overly-permissive directory is tightened
  // (TOCTOU hardening) without re-chmodding on every hot-path write.
  private readonly initDirs = new Set<string>()

  constructor(options: LocalFsBackendOptions = {}) {
    this.isDefaultLayout = options.baseDir === undefined
    this.baseDir = options.baseDir ?? defaultBaseDir
  }

  async get(key: string): Promise<Buffer | null> {
    const path = keyToPath(key, this.baseDir)
    try {
      return await readFile(path)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null
      throw err
    }
  }

  async put(key: string, blob: Buffer): Promise<void> {
    const path = keyToPath(key, this.baseDir)
    const dir = dirname(path)
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true, mode: 0o700 })
    }
    if (!this.initDirs.has(dir)) {
      if (process.platform !== 'win32') {
        await chmod(dir, 0o700)
      }
      this.initDirs.add(dir)
    }
    await atomicWriteFile(path, blob)
  }

  async delete(key: string): Promise<void> {
    const path = keyToPath(key, this.baseDir)
    try {
      await unlink(path)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
    }
  }
}

/** Select a credential backend from the MCP_STORAGE_BACKEND env var. */
export function backendFromEnv(): CredentialBackend {
  const kind = (process.env.MCP_STORAGE_BACKEND ?? 'local').toLowerCase()
  if (kind === 'local' || kind === 'local-fs' || kind === '') {
    return new LocalFsBackend()
  }
  throw new Error(`Unknown MCP_STORAGE_BACKEND: ${kind}`)
}
