/**
 * Per-plugin encrypted credential store.
 *
 * Layout (written by every save; see `data-dir.ts`):
 *   Stdio / HTTP single-user: ~/.<plugin>/config.json
 *   HTTP multi-user:          ~/.<plugin>/subs/<sub>/config.json
 *
 * Read fallback (local-fs default layout only), first blob found wins:
 *   ~/.<plugin>/  ->  ~/.<plugin>-mcp/  ->  ~/.better-<plugin>-mcp/
 * A legacy blob is decrypted with that legacy directory's own key material,
 * and the next save writes it to ~/.<plugin>/ (migrate on write). The legacy
 * copy is left in place; `clear()` removes the blob from every location.
 *
 * Encryption:
 *   Stdio / single-user: AES-256-GCM with machine-bound key persisted at
 *     ~/.<plugin>/.secret (auto-generated, 32 bytes, 0600).
 *   HTTP multi-user:    AES-256-GCM with key derived via scrypt from env
 *     CREDENTIAL_SECRET, salt `<plugin>:<sub>`.
 *
 * Replaces deprecated config-file (shared config.enc) which caused
 * multi-daemon path-drift contention with platformdirs version skew.
 */

import { createCipheriv, createDecipheriv, randomBytes, scryptSync } from 'node:crypto'
import { existsSync } from 'node:fs'
import { chmod, mkdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { atomicWriteFile } from './atomic-write.js'
import { backendFromEnv, type CredentialBackend, LocalFsBackend } from './backends.js'
import {
  assertValidPluginName,
  assertValidSub,
  blobPathIn,
  type DataDirLocation,
  dataDir,
  legacyDataDirs
} from './data-dir.js'

export { setHomeDirForTesting } from './home-dir.js'

// Credential-store directories already initialized in this process. init runs
// even when the secret already exists (read path) so a pre-existing
// overly-permissive directory is tightened exactly once (TOCTOU hardening).
const initDirs = new Set<string>()

/** Path of the credential blob in the current `~/.<plugin>/` layout. */
export function credPath(pluginName: string, sub: string | null): string {
  assertValidPluginName(pluginName)
  if (sub !== null) assertValidSub(sub)
  return blobPathIn(dataDir(pluginName), sub)
}

async function loadOrGenMachineKey(pluginName: string): Promise<Buffer> {
  const dir = dataDir(pluginName)
  const secretPath = join(dir, '.secret')

  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true, mode: 0o700 })
  }
  if (!initDirs.has(dir)) {
    // Explicitly chmod(0o700) to tighten an existing overly-permissive directory
    // (TOCTOU hardening); runs once per directory even on the read path.
    if (process.platform !== 'win32') {
      await chmod(dir, 0o700)
    }
    initDirs.add(dir)
  }

  try {
    return await readFile(secretPath)
  } catch {
    const key = randomBytes(32)
    await atomicWriteFile(secretPath, key)
    return key
  }
}

/**
 * Read-only machine key of a legacy directory. Never creates, chmods or mints
 * anything there: a missing `.secret` means the blob cannot be decrypted.
 */
async function readLegacyMachineKey(dir: string): Promise<Buffer | null> {
  try {
    return await readFile(join(dir, '.secret'))
  } catch {
    return null
  }
}

function deriveMultiUserKey(pluginName: string, sub: string): Buffer {
  const master = process.env.CREDENTIAL_SECRET
  if (!master) {
    throw new Error('CREDENTIAL_SECRET env required for HTTP multi-user mode')
  }
  return scryptSync(master, Buffer.from(`${pluginName}:${sub}`, 'utf-8'), 32, {
    N: 16384,
    r: 8,
    p: 1
  })
}

/** Where a blob was read from: the primary backend, or a legacy directory. */
interface BlobSource {
  backend: CredentialBackend
  /** null for the primary backend (current layout or an injected backend). */
  legacy: DataDirLocation | null
}

export class PerPluginStore {
  readonly credPath: string
  private readonly credKey: string
  private readonly backend: CredentialBackend
  private readonly readsLegacy: boolean

  constructor(
    public pluginName: string,
    public sub: string | null = null,
    backend?: CredentialBackend
  ) {
    this.credPath = credPath(pluginName, sub)
    this.credKey = sub ? `${pluginName}/subs/${sub}/config` : `${pluginName}/config`
    this.backend = backend ?? backendFromEnv()
    // The legacy layouts only exist on local disk, so only a default-layout
    // LocalFsBackend reads them. An injected in-memory/KV/custom-dir backend is
    // taken as the single source of truth.
    this.readsLegacy = this.backend instanceof LocalFsBackend && this.backend.isDefaultLayout
  }

  /**
   * Primary backend first, then the legacy dirs in read-precedence order.
   * Resolved per operation, like the primary LocalFsBackend resolves home.
   */
  private sources(): BlobSource[] {
    const primary: BlobSource = { backend: this.backend, legacy: null }
    if (!this.readsLegacy) return [primary]
    return [
      primary,
      ...legacyDataDirs(this.pluginName).map((loc) => ({
        backend: new LocalFsBackend({ baseDir: () => loc.dir }),
        legacy: loc
      }))
    ]
  }

  private async key(): Promise<Buffer> {
    return this.sub ? deriveMultiUserKey(this.pluginName, this.sub) : loadOrGenMachineKey(this.pluginName)
  }

  /** Key material a legacy directory's writer used; null when it is gone. */
  private async legacyKey(loc: DataDirLocation): Promise<Buffer | null> {
    return this.sub ? deriveMultiUserKey(loc.keyPlugin, this.sub) : readLegacyMachineKey(loc.dir)
  }

  private reportCorrupt(legacy: DataDirLocation | null): null {
    const where = legacy ? ` (legacy dir ${legacy.dir})` : ''
    console.error(
      `[hull PerPluginStore] Credential blob for ${this.credKey}${where} is corrupt or the encryption key changed; treating as not configured (re-run setup to restore)`
    )
    return null
  }

  private async decrypt(blob: Buffer, legacy: DataDirLocation | null): Promise<Record<string, unknown> | null> {
    // Format: [12-byte IV][ciphertext][16-byte GCM tag]
    if (blob.length < 29) return this.reportCorrupt(legacy)
    const key = legacy ? await this.legacyKey(legacy) : await this.key()
    if (key === null) return this.reportCorrupt(legacy)
    const iv = blob.subarray(0, 12)
    const tag = blob.subarray(blob.length - 16)
    const data = blob.subarray(12, blob.length - 16)
    try {
      // createDecipheriv sits inside the try: a legacy `.secret` of the wrong
      // length is unusable key material, i.e. the same "corrupt" outcome.
      const decipher = createDecipheriv('aes-256-gcm', key, iv, { authTagLength: 16 })
      decipher.setAuthTag(tag)
      const plaintext = Buffer.concat([decipher.update(data), decipher.final()])
      return JSON.parse(plaintext.toString('utf-8')) as Record<string, unknown>
    } catch {
      return this.reportCorrupt(legacy)
    }
  }

  /**
   * Load the credential payload. The first location holding a blob decides
   * the result; a corrupt blob there yields null rather than falling through
   * to an older location, so a stale legacy credential is never resurrected
   * behind a newer (damaged) one.
   */
  async load(): Promise<Record<string, unknown> | null> {
    for (const source of this.sources()) {
      const blob = await source.backend.get(this.credKey)
      if (blob !== null) return this.decrypt(blob, source.legacy)
    }
    return null
  }

  /** Encrypt and write to the primary backend (current layout); legacy copies are kept. */
  async save(payload: Record<string, unknown>): Promise<void> {
    const iv = randomBytes(12)
    const cipher = createCipheriv('aes-256-gcm', await this.key(), iv, { authTagLength: 16 })
    const plaintext = Buffer.from(JSON.stringify(payload), 'utf-8')
    const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()])
    const tag = cipher.getAuthTag()
    await this.backend.put(this.credKey, Buffer.concat([iv, ciphertext, tag]))
  }

  /** Delete the blob from the primary backend and from every legacy location. */
  async clear(): Promise<void> {
    for (const source of this.sources()) {
      await source.backend.delete(this.credKey)
    }
  }

  /**
   * Whether a credential blob is stored at all (in any read location),
   * regardless of whether it decrypts. `load()` returns null for both "absent"
   * and "corrupt"; a built-in CLI (`status` / `doctor`) needs to tell the two
   * apart to report "not configured" vs "corrupt (re-run setup)". Reads only
   * the opaque backend blobs, never the plaintext.
   */
  async hasStoredBlob(): Promise<boolean> {
    for (const source of this.sources()) {
      if ((await source.backend.get(this.credKey)) !== null) return true
    }
    return false
  }
}
