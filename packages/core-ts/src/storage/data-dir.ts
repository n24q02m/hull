/**
 * On-disk data directory layout for per-plugin credential storage.
 *
 * Current layout (written by every save):
 *
 *   ~/.<plugin>/config.json              single-user blob
 *   ~/.<plugin>/subs/<sub>/config.json   multi-user (per-sub) blob
 *   ~/.<plugin>/.secret                  single-user machine key (0600)
 *
 * Earlier releases stored the same files under two longer names. They stay
 * READABLE so credentials already on disk keep working with zero re-auth, in
 * this order (newest naming first):
 *
 *   1. ~/.<plugin>/               current
 *   2. ~/.<plugin>-mcp/           legacy, keyed by `<plugin>`
 *   3. ~/.better-<plugin>-mcp/    legacy, keyed by `better-<plugin>`
 *
 * "Keyed by" is the plugin name the legacy writer used for key derivation: the
 * single-user machine key is that directory's own `.secret`, and the multi-user
 * scrypt salt is `<keyPlugin>:<sub>`. A `better-<x>-mcp` directory was written
 * by a server whose plugin slug was `better-<x>`, so its salt carries the
 * `better-` prefix too.
 */

import { join } from 'node:path'
import { getHomeDir } from './home-dir.js'

/** One place a credential blob may live on disk. */
export interface DataDirLocation {
  /** Absolute directory path (resolved against the current home dir). */
  readonly dir: string
  /** Plugin name the writer used for key derivation in this directory. */
  readonly keyPlugin: string
  /** True for the read-only fallback layouts; false for the current one. */
  readonly legacy: boolean
}

// Not global: a /g flag makes .test() stateful across calls (lastIndex), which
// could let a second input bypass validation.
const UNSAFE = /[^a-zA-Z0-9._-]/

/**
 * Validate a plugin name for use as a single path component. Underscore,
 * hyphen and dot are allowed; separators and `..` are not.
 */
export function assertValidPluginName(pluginName: string): void {
  if (!pluginName || UNSAFE.test(pluginName) || pluginName.includes('..')) {
    throw new Error('Invalid pluginName')
  }
}

/**
 * Validate a multi-user sub for use as a single path component. The OAuth AS
 * mints sub = token_urlsafe(), whose base64url alphabet includes `_` and `-`,
 * so both must pass; `/` and `..` stay rejected.
 */
export function assertValidSub(sub: string): void {
  if (!sub || UNSAFE.test(sub) || sub.includes('..')) {
    throw new Error('Invalid sub')
  }
}

/** Directory name (relative to home) of the current layout: `.<plugin>`. */
export function dataDirName(pluginName: string): string {
  return `.${pluginName}`
}

/** Absolute path of the current data directory: `~/.<plugin>`. */
export function dataDir(pluginName: string): string {
  assertValidPluginName(pluginName)
  return join(getHomeDir(), dataDirName(pluginName))
}

/**
 * Every location a plugin's credentials may live in, in read-precedence order:
 * the current `~/.<plugin>/` first, then `~/.<plugin>-mcp/`, then
 * `~/.better-<plugin>-mcp/`.
 */
export function dataDirCandidates(pluginName: string): DataDirLocation[] {
  assertValidPluginName(pluginName)
  const home = getHomeDir()
  return [
    { dir: join(home, dataDirName(pluginName)), keyPlugin: pluginName, legacy: false },
    { dir: join(home, `.${pluginName}-mcp`), keyPlugin: pluginName, legacy: true },
    { dir: join(home, `.better-${pluginName}-mcp`), keyPlugin: `better-${pluginName}`, legacy: true }
  ]
}

/** Only the legacy (read-only fallback) locations, in read-precedence order. */
export function legacyDataDirs(pluginName: string): DataDirLocation[] {
  return dataDirCandidates(pluginName).filter((loc) => loc.legacy)
}

/** Path of the credential blob inside a given data directory. */
export function blobPathIn(dir: string, sub: string | null): string {
  return sub ? join(dir, 'subs', sub, 'config.json') : join(dir, 'config.json')
}
