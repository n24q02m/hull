/**
 * Named provider accounts for multi-account pools (TypeScript mirror of
 * `hull_core/credentials/registry.py`).
 *
 * A provider such as OpenRouter can be used through several identities (one
 * per email, workspace or billing owner). Each account owns its own credential
 * pool: rotation never hops between accounts, so a 429 on one account's key
 * cannot silently spend another account's quota.
 *
 * `accountId` is the stable identifier config points at; `name` is the human
 * label and is unique per provider. Credentials can only be attached to an
 * existing account whose provider matches.
 */

import { type Account, AccountError, type CredentialHealth, type KeyPool, type KeyPoolOptions } from './key-pool.js'
import type { AccountRow, CredentialStore } from './key-pool-store.js'

const NON_SLUG = /[^a-z0-9]+/g

function text(value: string | null | undefined, field: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new AccountError(`${field} must be a non-empty string`)
  return value.trim()
}

function optional(value: string | null | undefined): string | null {
  if (value === null || value === undefined) return null
  const trimmed = value.trim()
  return trimmed || null
}

function labelsOf(value: Record<string, unknown> | null | undefined): Record<string, string> {
  if (value === null || value === undefined) return {}
  const result: Record<string, string> = {}
  for (const [key, item] of Object.entries(value)) result[key] = String(item)
  return result
}

function slug(name: string): string {
  const value = name
    .trim()
    .toLowerCase()
    .replace(NON_SLUG, '-')
    .replace(/^-+|-+$/g, '')
  if (!value) throw new AccountError(`cannot derive an account id from name ${JSON.stringify(name)}`)
  return value
}

function rowToAccount(row: AccountRow): Account {
  let parsed: unknown = {}
  try {
    parsed = JSON.parse(row.labels || '{}')
  } catch {
    parsed = {}
  }
  const labels: Record<string, string> = {}
  if (parsed && typeof parsed === 'object') {
    for (const [key, item] of Object.entries(parsed)) labels[key] = String(item)
  }
  return {
    accountId: row.account_id,
    provider: row.provider,
    name: row.name,
    email: row.email,
    workspace: row.workspace,
    labels
  }
}

/** CRUD over `provider_accounts` plus per-account pool resolution. */
export class AccountRegistry {
  readonly store: CredentialStore

  constructor(store: CredentialStore) {
    this.store = store
  }

  /** Register one named account; the default id is `<provider>:<name slug>`. */
  add(input: {
    provider: string
    name: string
    accountId?: string | null
    email?: string | null
    workspace?: string | null
    labels?: Record<string, unknown> | null
  }): Account {
    const provider = text(input.provider, 'provider')
    const name = text(input.name, 'name')
    const accountId = input.accountId ? text(input.accountId, 'accountId') : `${provider}:${slug(name)}`
    const account: Account = {
      accountId,
      provider,
      name,
      email: optional(input.email),
      workspace: optional(input.workspace),
      labels: labelsOf(input.labels)
    }
    this.store.insertAccount(account)
    return account
  }

  /** Resolve an account id first, then a unique name; `null` when absent. */
  maybeGet(ref: string): Account | null {
    const key = text(ref, 'account reference')
    const byId = this.store.accountRow(key)
    if (byId) return rowToAccount(byId)
    const matches = this.store.accountRowsByName(key)
    if (matches.length === 0) return null
    if (matches.length > 1) {
      const providers = [...new Set(matches.map((row) => row.provider))].sort()
      throw new AccountError(
        `account name ${JSON.stringify(key)} is ambiguous across providers [${providers.join(', ')}]; pass the account id`
      )
    }
    return rowToAccount(matches[0])
  }

  get(ref: string): Account {
    const account = this.maybeGet(ref)
    if (account === null) throw new AccountError(`unknown account ${JSON.stringify(ref)}; add it first`)
    return account
  }

  listAccounts(provider?: string): Account[] {
    const accounts = this.store.accountRows().map((row) => rowToAccount(row))
    return provider ? accounts.filter((account) => account.provider === provider) : accounts
  }

  /**
   * Partially update metadata; omitted fields keep their value. `labels: null`
   * clears the labels, `email: null`/`workspace: null` clear those fields. The
   * provider and account id are immutable.
   */
  update(
    ref: string,
    patch: {
      name?: string
      email?: string | null
      workspace?: string | null
      labels?: Record<string, unknown> | null
    } = {}
  ): Account {
    const current = this.get(ref)
    const name = patch.name === undefined ? current.name : text(patch.name, 'name')
    const email = patch.email === undefined ? current.email : optional(patch.email)
    const workspace = patch.workspace === undefined ? current.workspace : optional(patch.workspace)
    const labels = patch.labels === undefined ? current.labels : labelsOf(patch.labels)
    this.store.updateAccountRow(current.accountId, {
      name,
      email,
      workspace,
      labelsJson: JSON.stringify(labels)
    })
    return this.get(current.accountId)
  }

  /** Delete an account; credentials block the delete unless dropped explicitly. */
  remove(ref: string, options: { dropCredentials?: boolean } = {}): void {
    const current = this.get(ref)
    this.store.deleteAccountRow(current.accountId, options)
  }

  /** Masked pool state for one account (safe to log). */
  credentials(ref: string): CredentialHealth[] {
    const account = this.get(ref)
    return this.store.poolHealth(account.provider, account.accountId)
  }

  /** The KeyPool scoped to one account (provider + account id). */
  pool(ref: string, options: KeyPoolOptions = {}): KeyPool {
    const account = this.get(ref)
    return this.store.pool(account.provider, { ...options, accountId: account.accountId })
  }
}
