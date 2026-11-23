/**
 * On-disk data directory layout for per-plugin credential storage.
 *
 * Current layout (written by every save):
 *
 *   ~/.n24q02m/<plugin>/config.json                           single-user blob (default profile)
 *   ~/.n24q02m/<plugin>/profiles/<prof>/config.json           single-user blob (named profile)
 *   ~/.n24q02m/<plugin>/subs/<sub>/config.json                multi-user (per-sub) blob
 *   ~/.n24q02m/<plugin>/subs/<sub>/profiles/<prof>/config.json multi-user named profile blob
 *   ~/.n24q02m/<plugin>/.secret                               single-user machine key (0600)
 *
 * The root directory defaults to `~/.n24q02m` and can be overridden via
 * `N24Q02M_HOME` or `N24Q02M_USE_XDG=1` (which resolves `$XDG_CONFIG_HOME/n24q02m`
 * or `%APPDATA%/n24q02m`).
 *
 * Earlier releases stored the same files directly under `$HOME`. They stay
 * READABLE so credentials already on disk keep working with zero re-auth, in
 * this order (newest naming first):
 *
 *   1. ~/.n24q02m/<plugin>/       current
 *   2. ~/.<plugin>/               legacy (v1 hull), keyed by `<plugin>`
 *   3. ~/.<plugin>-mcp/           legacy, keyed by `<plugin>`
 *   4. ~/.better-<plugin>-mcp/    legacy, keyed by `better-<plugin>`
 */

import { cpSync, existsSync, mkdirSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
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
  if (!pluginName || pluginName === '.' || UNSAFE.test(pluginName) || pluginName.includes('..')) {
    throw new Error('Invalid pluginName')
  }
}

/**
 * Validate a multi-user sub for use as a single path component. Subs are
 * base64url-style tokens (token_urlsafe), whose alphabet includes `_` and `-`,
 * so both must pass; `/` and `..` stay rejected.
 */
export function assertValidSub(sub: string): void {
  if (!sub || sub === '.' || UNSAFE.test(sub) || sub.includes('..')) {
    throw new Error('Invalid sub')
  }
}

/** Validate a Tier-2 profile name for use as a single path component. */
export function assertValidProfile(profile: string): void {
  if (!profile || profile === '.' || UNSAFE.test(profile) || profile.includes('..')) {
    throw new Error('Invalid profile')
  }
}

/**
 * Resolve effective Tier-2 profile: explicit `profile` -> `<PLUGIN>_PROFILE` -> `HULL_PROFILE` -> `'default'`.
 */
export function resolveProfile(
  pluginName: string,
  profile?: string | null,
  env: NodeJS.ProcessEnv = process.env
): string {
  assertValidPluginName(pluginName)
  if (profile !== undefined && profile !== null && profile !== '') {
    assertValidProfile(profile)
    return profile
  }
  const pluginEnvKey = `${pluginName.toUpperCase().replaceAll('-', '_').replaceAll('.', '_')}_PROFILE`
  const fromPlugin = env[pluginEnvKey]?.trim()
  if (fromPlugin) {
    assertValidProfile(fromPlugin)
    return fromPlugin
  }
  const fromHull = env.HULL_PROFILE?.trim()
  if (fromHull) {
    assertValidProfile(fromHull)
    return fromHull
  }
  return 'default'
}

/** Root n24q02m state directory (`~/.n24q02m` by default, or XDG / `N24Q02M_HOME`). */
export function n24q02mBaseDir(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.N24Q02M_HOME?.trim()
  if (override) return override
  const useXdg = env.N24Q02M_USE_XDG?.trim().toLowerCase()
  if (useXdg === '1' || useXdg === 'true' || useXdg === 'yes') {
    const xdg = env.XDG_CONFIG_HOME?.trim() || env.APPDATA?.trim()
    if (xdg) return join(xdg, 'n24q02m')
  }
  return join(getHomeDir(), '.n24q02m')
}

/** Directory name (relative to home) of the current layout: `.n24q02m/<plugin>`. */
export function dataDirName(pluginName: string): string {
  return join('.n24q02m', pluginName)
}

/** Absolute path of the current data directory: `~/.n24q02m/<plugin>`. */
export function dataDir(pluginName: string): string {
  assertValidPluginName(pluginName)
  return join(n24q02mBaseDir(), pluginName)
}

/**
 * Every location a plugin's credentials may live in, in read-precedence order:
 * the current `~/.n24q02m/<plugin>/` first, then `~/.<plugin>/`, then
 * `~/.<plugin>-mcp/`, then `~/.better-<plugin>-mcp/`.
 */
export function dataDirCandidates(pluginName: string): DataDirLocation[] {
  assertValidPluginName(pluginName)
  const home = getHomeDir()
  return [
    { dir: dataDir(pluginName), keyPlugin: pluginName, legacy: false },
    { dir: join(home, `.${pluginName}`), keyPlugin: pluginName, legacy: true },
    { dir: join(home, `.${pluginName}-mcp`), keyPlugin: pluginName, legacy: true },
    { dir: join(home, `.better-${pluginName}-mcp`), keyPlugin: `better-${pluginName}`, legacy: true }
  ]
}

/** Only the legacy (read-only fallback) locations, in read-precedence order. */
export function legacyDataDirs(pluginName: string): DataDirLocation[] {
  return dataDirCandidates(pluginName).filter((loc) => loc.legacy)
}

/** Path of the credential blob inside a given data directory. */
export function blobPathIn(dir: string, sub: string | null, profile?: string | null): string {
  const base = sub ? join(dir, 'subs', sub) : dir
  if (profile !== undefined && profile !== null && profile !== '' && profile !== 'default') {
    assertValidProfile(profile)
    return join(base, 'profiles', profile, 'config.json')
  }
  return join(base, 'config.json')
}

/**
 * One-time non-destructive auto-migration from `~/.<plugin>` to `~/.n24q02m/<plugin>`.
 *
 * If the primary `dataDir(pluginName)` does not exist yet and `~/.<plugin>`
 * exists as a directory, copies the legacy directory tree into `dataDir(pluginName)`
 * without modifying or deleting `~/.<plugin>`, and returns `true`.
 */
export function migrateLegacyDataDir(pluginName: string): boolean {
  assertValidPluginName(pluginName)
  const target = dataDir(pluginName)
  if (existsSync(target)) return false
  const legacyDot = join(getHomeDir(), `.${pluginName}`)
  if (!existsSync(legacyDot) || !statSync(legacyDot).isDirectory()) return false
  mkdirSync(dirname(target), { recursive: true, mode: 0o700 })
  cpSync(legacyDot, target, { recursive: true })
  return true
}
