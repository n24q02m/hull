/**
 * Tier-3 per-plugin named account registry backed by `PerPluginStore`.
 *
 * Mirrors `packages/core-py/src/hull_core/storage/account_registry.py`.
 * Allows a single user (in stdio, token, or multi-user mode, and within any
 * Tier-2 profile) to register and switch between multiple named accounts
 * (e.g. `work` and `personal` Notion workspaces, Telegram bots, or email
 * accounts) under leaf `accounts`.
 */

import type { CredentialBackend } from './backends.js'
import { assertValidProfile } from './data-dir.js'
import { PerPluginStore } from './per-plugin-store.js'

export function assertValidAccountAlias(alias: string): void {
  try {
    assertValidProfile(alias)
  } catch {
    throw new Error('Invalid account alias')
  }
}

export interface NamedAccountRegistryOptions {
  readonly backend?: CredentialBackend
  readonly profile?: string | null
}

interface AccountState {
  readonly active: string | null
  readonly accounts: Record<string, Record<string, unknown>>
}

export class NamedAccountRegistry {
  readonly store: PerPluginStore

  constructor(
    public readonly pluginName: string,
    public readonly sub: string | null = null,
    options?: NamedAccountRegistryOptions
  ) {
    this.store = new PerPluginStore(pluginName, sub, options?.backend, {
      profile: options?.profile,
      subKey: 'accounts'
    })
  }

  private async readState(): Promise<AccountState> {
    const raw = await this.store.load()
    if (!raw) return { active: null, accounts: {} }
    const rawAccounts = raw.accounts
    const accounts: Record<string, Record<string, unknown>> = {}
    if (rawAccounts && typeof rawAccounts === 'object' && !Array.isArray(rawAccounts)) {
      for (const [key, val] of Object.entries(rawAccounts as Record<string, unknown>)) {
        if (val && typeof val === 'object' && !Array.isArray(val)) {
          accounts[key] = { ...(val as Record<string, unknown>) }
        }
      }
    }
    const rawActive = raw.active
    let active = typeof rawActive === 'string' && rawActive in accounts ? rawActive : null
    if (active === null) {
      const keys = Object.keys(accounts)
      if (keys.length > 0) active = keys[0]
    }
    return { active, accounts }
  }

  async listAccounts(): Promise<string[]> {
    const { accounts } = await this.readState()
    return Object.keys(accounts).sort()
  }

  async getActiveAccount(): Promise<string | null> {
    const { active } = await this.readState()
    return active
  }

  async getAccount(alias?: string | null): Promise<Record<string, unknown> | null> {
    const { active, accounts } = await this.readState()
    let target: string | null
    if (alias !== undefined && alias !== null) {
      assertValidAccountAlias(alias)
      target = alias
    } else {
      target = active
    }
    if (target === null) return null
    return accounts[target] ?? null
  }

  async putAccount(
    alias: string,
    credentials: Record<string, unknown>,
    options?: { setActive?: boolean }
  ): Promise<void> {
    assertValidAccountAlias(alias)
    if (!credentials || typeof credentials !== 'object' || Array.isArray(credentials)) {
      throw new Error('Account credentials must be an object')
    }
    const { active, accounts } = await this.readState()
    accounts[alias] = { ...credentials }
    const setActive = options?.setActive ?? true
    const newActive = setActive || active === null ? alias : active
    await this.store.save({ active: newActive, accounts })
  }

  async setActive(alias: string): Promise<void> {
    assertValidAccountAlias(alias)
    const { accounts } = await this.readState()
    if (!(alias in accounts)) {
      throw new Error(`Unknown account alias: ${alias}`)
    }
    await this.store.save({ active: alias, accounts })
  }

  async removeAccount(alias: string): Promise<boolean> {
    assertValidAccountAlias(alias)
    const { active, accounts } = await this.readState()
    if (!(alias in accounts)) return false
    delete accounts[alias]
    const remaining = Object.keys(accounts)
    if (remaining.length === 0) {
      await this.store.clear()
      return true
    }
    const newActive = active !== null && active in accounts ? active : remaining[0]
    await this.store.save({ active: newActive, accounts })
    return true
  }
}
