/**
 * SQLite-backed provider key pool and account registry (`~/.hull/hull.db`).
 *
 * Mirrors `packages/core-py/src/hull_core/credentials/store.py`, including the
 * table DDL and the shared `v1:` secret envelope, so either stack can open a
 * pool the other created. The tables are additive and migration-safe:
 * `CREATE TABLE/INDEX IF NOT EXISTS` only, no `ALTER`, so a database written by
 * an older hull (only the `kv` table) keeps working and older hull ignores the
 * new tables.
 *
 * Scope rules:
 * - A pool is one provider identity; `account_id IS NULL` is that provider's
 *   default pool and a named account has its own pool. Rotation never hops
 *   across accounts.
 * - Entries are ordered by `priority` then insertion order.
 * - Rotation state lives in the database, not process memory, so two tasks (or
 *   two processes) never reuse a credential that is cooling down or leased.
 */

import { mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import Database from 'better-sqlite3'
import { AccountRegistry } from './account-registry.js'
import { getHomeDir } from './home-dir.js'
import {
  type Account,
  AccountError,
  type Credential,
  CredentialExistsError,
  type CredentialHealth,
  CredentialNotFoundError,
  KeyPool,
  type KeyPoolOptions,
  type MigrationResult,
  providerFromBaseUrl
} from './key-pool.js'
import { keyFingerprint, keyHint, SecretCipher } from './key-pool-crypto.js'

export const KEY_POOL_SCHEMA = `
CREATE TABLE IF NOT EXISTS provider_accounts (
    account_id TEXT PRIMARY KEY,
    provider   TEXT NOT NULL,
    name       TEXT NOT NULL,
    email      TEXT,
    workspace  TEXT,
    labels     TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS provider_accounts_name
    ON provider_accounts(provider, name);

CREATE TABLE IF NOT EXISTS provider_credentials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    provider        TEXT NOT NULL,
    account_id      TEXT,
    label           TEXT,
    base_url        TEXT NOT NULL,
    model           TEXT,
    secret          TEXT NOT NULL,
    key_hint        TEXT NOT NULL,
    key_fingerprint TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 0,
    disabled        INTEGER NOT NULL DEFAULT 0,
    last_status     INTEGER,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    cooldown_until  REAL NOT NULL DEFAULT 0,
    lease_until     REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS provider_credentials_identity
    ON provider_credentials(provider, ifnull(account_id, ''), key_fingerprint, base_url);

CREATE INDEX IF NOT EXISTS provider_credentials_pool
    ON provider_credentials(provider, account_id, priority, id);
`

const COLUMNS =
  'id, provider, account_id, label, base_url, model, secret, key_hint, ' +
  'priority, disabled, last_status, failure_count, cooldown_until, lease_until'
const TOUCH = "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"

interface CredentialRow {
  id: number
  provider: string
  account_id: string | null
  label: string | null
  base_url: string
  model: string | null
  secret: string
  key_hint: string
  priority: number
  disabled: number
  last_status: number | null
  failure_count: number
  cooldown_until: number
  lease_until: number
}

export interface AccountRow {
  account_id: string
  provider: string
  name: string
  email: string | null
  workspace: string | null
  labels: string
}

/** A `[models.<task>]`-shaped cell, as products and configs already carry it. */
export interface ModelCellLike {
  readonly baseUrl: string
  readonly apiKey?: string | null
  readonly model?: string | null
  readonly provider?: string | null
  readonly account?: string | null
}

export interface AddCredentialInput {
  readonly provider: string
  readonly apiKey: string
  readonly baseUrl: string
  readonly model?: string | null
  readonly accountId?: string | null
  readonly label?: string | null
  readonly priority?: number | null
}

export interface CredentialStoreOptions {
  readonly cipher?: SecretCipher
  readonly now?: () => number
}

export interface PoolDiagnostics {
  readonly total: number
  readonly disabled: number
  readonly coolingDown: number
  readonly leased: number
  readonly retryAfter: number | null
}

/** The instance database every hull surface already uses: `~/.hull/hull.db`. */
export function defaultHullDbPath(): string {
  return join(getHomeDir(), '.hull', 'hull.db')
}

/** Key file that seals this database's credentials (same directory as the DB). */
export function credentialKeyPath(dbPath: string): string {
  return join(dirname(dbPath), '.credential-key')
}

function required(value: string | null | undefined, field: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${field} must be a non-empty string`)
  return value.trim()
}

function optional(value: string | null | undefined): string | null {
  if (value === null || value === undefined) return null
  const text = value.trim()
  return text || null
}

function makeCredential(row: CredentialRow, apiKey: string): Credential {
  const owner = row.account_id ?? row.label ?? String(row.id)
  const masked = `${row.provider}/${owner}:${row.key_hint}`
  return {
    id: row.id,
    provider: row.provider,
    accountId: row.account_id,
    label: row.label,
    baseUrl: row.base_url,
    model: row.model,
    apiKey,
    priority: row.priority,
    keyHint: row.key_hint,
    masked,
    toString(): string {
      return `Credential(${masked}, base_url=${row.base_url})`
    },
    toJSON(): Record<string, unknown> {
      return {
        id: row.id,
        provider: row.provider,
        accountId: row.account_id,
        label: row.label,
        baseUrl: row.base_url,
        model: row.model,
        priority: row.priority,
        keyHint: row.key_hint
      }
    }
  }
}

/** Ordered provider credential pool plus named accounts, on local SQLite. */
export class CredentialStore {
  readonly path: string
  private readonly db: Database.Database
  private readonly cipher: SecretCipher
  private readonly clock: () => number

  constructor(dbPath: string = defaultHullDbPath(), options: CredentialStoreOptions = {}) {
    this.path = dbPath
    this.clock = options.now ?? (() => Date.now() / 1000)
    this.cipher = options.cipher ?? new SecretCipher(credentialKeyPath(dbPath))
    mkdirSync(dirname(dbPath), { recursive: true, mode: 0o700 })
    this.db = new Database(dbPath)
    this.db.pragma('journal_mode = WAL')
    this.db.pragma('busy_timeout = 5000')
    this.db.exec(KEY_POOL_SCHEMA)
  }

  now(): number {
    return this.clock()
  }

  close(): void {
    this.db.close()
  }

  // -- row mapping ------------------------------------------------------

  private toCredential(row: CredentialRow): Credential {
    return makeCredential(row, this.cipher.decrypt(row.secret))
  }

  private toHealth(row: CredentialRow, now: number): CredentialHealth {
    const cooldownUntil = row.cooldown_until
    const leaseUntil = row.lease_until
    return {
      id: row.id,
      provider: row.provider,
      accountId: row.account_id,
      label: row.label,
      baseUrl: row.base_url,
      model: row.model,
      keyHint: row.key_hint,
      priority: row.priority,
      disabled: Boolean(row.disabled),
      lastStatus: row.last_status,
      failureCount: row.failure_count,
      cooldownUntil,
      leaseUntil,
      cooldownRemaining: Math.max(0, cooldownUntil - now),
      usable: !row.disabled && cooldownUntil <= now && leaseUntil <= now
    }
  }

  private row(credentialId: number): CredentialRow {
    const row = this.db
      .prepare<[number], CredentialRow>(`SELECT ${COLUMNS} FROM provider_credentials WHERE id = ?`)
      .get(credentialId)
    if (!row) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
    return row
  }

  private requireAccount(accountId: string, provider: string): AccountRow {
    const row = this.db
      .prepare<[string], AccountRow>('SELECT * FROM provider_accounts WHERE account_id = ?')
      .get(accountId)
    if (!row) throw new AccountError(`unknown account ${JSON.stringify(accountId)}; add it first`)
    if (row.provider !== provider) {
      throw new AccountError(
        `account ${JSON.stringify(accountId)} belongs to provider ${JSON.stringify(row.provider)}, not ${JSON.stringify(provider)}`
      )
    }
    return row
  }

  // -- writes -----------------------------------------------------------

  addCredential(input: AddCredentialInput): Credential {
    const provider = required(input.provider, 'provider')
    const apiKey = required(input.apiKey, 'apiKey')
    const baseUrl = required(input.baseUrl, 'baseUrl').replace(/\/+$/, '')
    const accountId = optional(input.accountId)
    const label = optional(input.label)
    const model = optional(input.model)
    const secret = this.cipher.encrypt(apiKey)
    const fingerprint = keyFingerprint(apiKey)
    const hint = keyHint(apiKey)
    const write = this.db.transaction((): CredentialRow => {
      if (accountId !== null) this.requireAccount(accountId, provider)
      let priority = input.priority ?? null
      if (priority === null || priority === undefined) {
        const next = this.db
          .prepare<[string, string | null], { next: number }>(
            'SELECT COALESCE(MAX(priority), -1) + 1 AS next FROM provider_credentials WHERE provider = ? AND account_id IS ?'
          )
          .get(provider, accountId)
        priority = next?.next ?? 0
      }
      let result: Database.RunResult
      try {
        result = this.db
          .prepare<[string, string | null, string | null, string, string | null, string, string, string, number]>(
            'INSERT INTO provider_credentials ' +
              '(provider, account_id, label, base_url, model, secret, key_hint, key_fingerprint, priority) ' +
              'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
          )
          .run(provider, accountId, label, baseUrl, model, secret, hint, fingerprint, priority)
      } catch (error) {
        if (error instanceof Error && error.message.includes('UNIQUE')) {
          throw new CredentialExistsError(
            `provider ${JSON.stringify(provider)} already has this credential` +
              (accountId ? ` for account ${JSON.stringify(accountId)}` : '')
          )
        }
        throw error
      }
      return this.row(Number(result.lastInsertRowid))
    })
    return this.toCredential(write.immediate())
  }

  removeCredential(credentialId: number): void {
    const result = this.db.prepare<[number]>('DELETE FROM provider_credentials WHERE id = ?').run(credentialId)
    if (result.changes === 0) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
  }

  setPriority(credentialId: number, priority: number): void {
    const result = this.db
      .prepare<[number, number]>(`UPDATE provider_credentials SET priority = ?, ${TOUCH} WHERE id = ?`)
      .run(priority, credentialId)
    if (result.changes === 0) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
  }

  /** Clear disabled/cooldown/failure state after an operator fixes the key. */
  enable(credentialId: number): void {
    const result = this.db
      .prepare<[number]>(
        `UPDATE provider_credentials SET disabled = 0, last_status = NULL, failure_count = 0, ` +
          `cooldown_until = 0, lease_until = 0, ${TOUCH} WHERE id = ?`
      )
      .run(credentialId)
    if (result.changes === 0) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
  }

  disable(credentialId: number): void {
    const result = this.db
      .prepare<[number]>(`UPDATE provider_credentials SET disabled = 1, lease_until = 0, ${TOUCH} WHERE id = ?`)
      .run(credentialId)
    if (result.changes === 0) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
  }

  /** Clear rotation state for one provider (or every provider); returns rows matched. */
  resetHealth(provider?: string): number {
    const sql =
      'UPDATE provider_credentials SET disabled = 0, last_status = NULL, failure_count = 0, ' +
      `cooldown_until = 0, lease_until = 0, ${TOUCH}`
    const result = provider
      ? this.db.prepare<[string]>(`${sql} WHERE provider = ?`).run(provider)
      : this.db.prepare(sql).run()
    return result.changes
  }

  // -- reads ------------------------------------------------------------

  /** Every decrypted entry (all accounts), ordered by provider/priority/id. */
  credentials(provider?: string): Credential[] {
    const sql = `SELECT ${COLUMNS} FROM provider_credentials`
    const rows = provider
      ? this.db
          .prepare<[string], CredentialRow>(`${sql} WHERE provider = ? ORDER BY provider, priority, id`)
          .all(provider)
      : this.db.prepare<[], CredentialRow>(`${sql} ORDER BY provider, priority, id`).all()
    return rows.map((row) => this.toCredential(row))
  }

  /** The rotation pool for one provider/account scope, in priority order. */
  poolCredentials(provider: string, accountId: string | null = null): Credential[] {
    const rows = this.db
      .prepare<[string, string | null], CredentialRow>(
        `SELECT ${COLUMNS} FROM provider_credentials WHERE provider = ? AND account_id IS ? ORDER BY priority, id`
      )
      .all(required(provider, 'provider'), optional(accountId))
    return rows.map((row) => this.toCredential(row))
  }

  /** Masked state of every entry, including disabled and cooling ones. */
  health(provider?: string): CredentialHealth[] {
    const now = this.clock()
    const sql = `SELECT ${COLUMNS} FROM provider_credentials`
    const rows = provider
      ? this.db
          .prepare<[string], CredentialRow>(`${sql} WHERE provider = ? ORDER BY provider, priority, id`)
          .all(provider)
      : this.db.prepare<[], CredentialRow>(`${sql} ORDER BY provider, priority, id`).all()
    return rows.map((row) => this.toHealth(row, now))
  }

  poolHealth(provider: string, accountId: string | null = null): CredentialHealth[] {
    const now = this.clock()
    const rows = this.db
      .prepare<[string, string | null], CredentialRow>(
        `SELECT ${COLUMNS} FROM provider_credentials WHERE provider = ? AND account_id IS ? ORDER BY priority, id`
      )
      .all(required(provider, 'provider'), optional(accountId))
    return rows.map((row) => this.toHealth(row, now))
  }

  getCredential(credentialId: number): Credential {
    return this.toCredential(this.row(credentialId))
  }

  diagnose(provider: string, accountId: string | null = null): PoolDiagnostics {
    const now = this.clock()
    const rows = this.db
      .prepare<[string, string | null], Pick<CredentialRow, 'disabled' | 'cooldown_until' | 'lease_until'>>(
        'SELECT disabled, cooldown_until, lease_until FROM provider_credentials WHERE provider = ? AND account_id IS ?'
      )
      .all(required(provider, 'provider'), optional(accountId))
    const disabled = rows.filter((row) => row.disabled).length
    const cooling = rows.filter((row) => !row.disabled && row.cooldown_until > now).map((row) => row.cooldown_until)
    const leased = rows.filter((row) => !row.disabled && row.cooldown_until <= now && row.lease_until > now).length
    return {
      total: rows.length,
      disabled,
      coolingDown: cooling.length,
      leased,
      retryAfter: cooling.length ? Math.min(...cooling) - now : null
    }
  }

  // -- rotation state (used by KeyPool) ---------------------------------

  /** Atomically lease the highest-priority usable entry, or return null. */
  leaseFirst(
    provider: string,
    accountId: string | null,
    options: { now: number; leaseSeconds: number; exclude?: number[] }
  ): Credential | null {
    const excluded = [...new Set(options.exclude ?? [])]
    let sql = `SELECT ${COLUMNS} FROM provider_credentials WHERE provider = ? AND account_id IS ? AND disabled = 0 AND cooldown_until <= ? AND lease_until <= ?`
    const params: unknown[] = [required(provider, 'provider'), optional(accountId), options.now, options.now]
    if (excluded.length > 0) {
      sql += ` AND id NOT IN (${excluded.map(() => '?').join(',')})`
      params.push(...excluded)
    }
    sql += ' ORDER BY priority, id LIMIT 1'
    const lease = this.db.transaction((): Credential | null => {
      const row = this.db.prepare<unknown[], CredentialRow>(sql).get(...params)
      if (!row) return null
      this.db
        .prepare<[number, number]>(`UPDATE provider_credentials SET lease_until = ?, ${TOUCH} WHERE id = ?`)
        .run(options.now + options.leaseSeconds, row.id)
      return this.toCredential(row)
    })
    return lease.immediate()
  }

  releaseLease(credentialId: number): void {
    const result = this.db
      .prepare<[number]>(`UPDATE provider_credentials SET lease_until = 0, ${TOUCH} WHERE id = ?`)
      .run(credentialId)
    if (result.changes === 0) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
  }

  recordSuccess(credentialId: number): void {
    const result = this.db
      .prepare<[number]>(
        `UPDATE provider_credentials SET last_status = 200, failure_count = 0, cooldown_until = 0, lease_until = 0, ${TOUCH} WHERE id = ?`
      )
      .run(credentialId)
    if (result.changes === 0) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
  }

  recordFailure(
    credentialId: number,
    status: number | null,
    options: { now?: number; cooldownUntil?: number | null; disable?: boolean } = {}
  ): void {
    const result = this.db
      .prepare<[number | null, number | null, number, number]>(
        'UPDATE provider_credentials SET last_status = ?, failure_count = failure_count + 1, ' +
          'cooldown_until = COALESCE(?, cooldown_until), ' +
          `disabled = CASE WHEN ? THEN 1 ELSE disabled END, lease_until = 0, ${TOUCH} WHERE id = ?`
      )
      .run(status, options.cooldownUntil ?? null, options.disable ? 1 : 0, credentialId)
    if (result.changes === 0) throw new CredentialNotFoundError(`no provider credential with id ${credentialId}`)
  }

  // -- accounts ---------------------------------------------------------

  accounts(): AccountRegistry {
    return new AccountRegistry(this)
  }

  accountRows(): AccountRow[] {
    return this.db.prepare<[], AccountRow>('SELECT * FROM provider_accounts ORDER BY provider, name').all()
  }

  accountRow(accountId: string): AccountRow | undefined {
    return this.db.prepare<[string], AccountRow>('SELECT * FROM provider_accounts WHERE account_id = ?').get(accountId)
  }

  accountRowsByName(name: string): AccountRow[] {
    return this.db.prepare<[string], AccountRow>('SELECT * FROM provider_accounts WHERE name = ?').all(name)
  }

  insertAccount(account: Account): void {
    try {
      this.db
        .prepare<[string, string, string, string | null, string | null, string]>(
          'INSERT INTO provider_accounts (account_id, provider, name, email, workspace, labels) VALUES (?, ?, ?, ?, ?, ?)'
        )
        .run(
          account.accountId,
          account.provider,
          account.name,
          account.email,
          account.workspace,
          JSON.stringify(account.labels)
        )
    } catch (error) {
      if (error instanceof Error && error.message.includes('UNIQUE')) {
        throw new AccountError(
          `account ${JSON.stringify(account.accountId)} or the name ${JSON.stringify(account.name)} ` +
            `for provider ${JSON.stringify(account.provider)} already exists`
        )
      }
      throw error
    }
  }

  updateAccountRow(
    accountId: string,
    values: { name: string; email: string | null; workspace: string | null; labelsJson: string }
  ): void {
    const result = this.db
      .prepare<[string, string | null, string | null, string, string]>(
        `UPDATE provider_accounts SET name = ?, email = ?, workspace = ?, labels = ?, ${TOUCH} WHERE account_id = ?`
      )
      .run(values.name, values.email, values.workspace, values.labelsJson, accountId)
    if (result.changes === 0) throw new AccountError(`unknown account ${JSON.stringify(accountId)}`)
  }

  deleteAccountRow(accountId: string, options: { dropCredentials?: boolean } = {}): void {
    const remove = this.db.transaction((): void => {
      const count = this.db
        .prepare<[string], { n: number }>('SELECT COUNT(*) AS n FROM provider_credentials WHERE account_id = ?')
        .get(accountId)
      const total = count?.n ?? 0
      if (total > 0 && !options.dropCredentials) {
        throw new AccountError(
          `account ${JSON.stringify(accountId)} still has ${total} credential(s); pass dropCredentials: true ` +
            'to delete them with the account'
        )
      }
      this.db.prepare<[string]>('DELETE FROM provider_credentials WHERE account_id = ?').run(accountId)
      const result = this.db.prepare<[string]>('DELETE FROM provider_accounts WHERE account_id = ?').run(accountId)
      if (result.changes === 0) throw new AccountError(`unknown account ${JSON.stringify(accountId)}`)
    })
    remove.immediate()
  }

  // -- pools ------------------------------------------------------------

  pool(provider: string, options: KeyPoolOptions = {}): KeyPool {
    return new KeyPool(this, provider, options)
  }

  /** Resolve a `[models.<task>]`-shaped cell's `provider`/`account` pointers. */
  poolForCell(cell: ModelCellLike, options: KeyPoolOptions = {}): KeyPool {
    const provider = optional(cell.provider) ?? providerFromBaseUrl(cell.baseUrl)
    return this.pool(provider, { ...options, accountId: optional(cell.account) })
  }

  /**
   * Seed the pool from the pre-pool single-key config, idempotently. Every
   * distinct `(baseUrl, apiKey)` becomes one entry in insertion order; running
   * it again skips what is already stored.
   */
  migrateModelCells(
    cells: Iterable<ModelCellLike>,
    options: { provider?: string | null; accountId?: string | null } = {}
  ): MigrationResult {
    const items = [...cells]
    if (items.length === 0) return { provider: options.provider ?? 'default', added: 0, skipped: 0 }
    let provider = options.provider ?? null
    if (provider === null) {
      const keyed = items.filter((cell) => (cell.apiKey ?? '').length > 0)
      const names = new Set(
        (keyed.length > 0 ? keyed : items).map((cell) => cell.provider || providerFromBaseUrl(cell.baseUrl))
      )
      if (names.size !== 1) {
        throw new Error(`model cells span providers [${[...names].sort().join(', ')}]; pass provider explicitly`)
      }
      provider = [...names][0]
    }
    let added = 0
    let skipped = 0
    for (const cell of items) {
      const apiKey = cell.apiKey ?? ''
      if (!apiKey) continue
      const target = options.accountId ?? optional(cell.account)
      try {
        this.addCredential({
          provider,
          apiKey,
          baseUrl: cell.baseUrl,
          model: cell.model ?? null,
          accountId: target
        })
        added += 1
      } catch (error) {
        if (!(error instanceof CredentialExistsError)) throw error
        skipped += 1
      }
    }
    return { provider, added, skipped }
  }
}
