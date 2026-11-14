/**
 * The refusals of the credential store and account registry: bad input, ids
 * that do not exist, rows sealed by another key, and account rows written by
 * something other than this registry.
 */

import { join } from 'node:path'
import Database from 'better-sqlite3'
import { describe, expect, it } from 'vitest'
import { AccountError, CredentialExistsError, CredentialNotFoundError } from '../../src/storage/key-pool.js'
import { SecretCipher } from '../../src/storage/key-pool-crypto.js'
import { CredentialStore } from '../../src/storage/key-pool-store.js'
import { BASE_URL, usePoolFixture } from './key-pool-helpers.js'

const fixture = usePoolFixture()
const SECRET = 'sk-live-contract-secret-0001'
const MISSING = 987_654

describe('credential store input', () => {
  it.each([
    ['provider', { provider: '  ', apiKey: SECRET, baseUrl: BASE_URL }],
    ['apiKey', { provider: 'openrouter', apiKey: '', baseUrl: BASE_URL }],
    ['baseUrl', { provider: 'openrouter', apiKey: SECRET, baseUrl: undefined as unknown as string }]
  ])('refuses an empty %s', (field, input) => {
    expect(() => fixture().store.addCredential(input)).toThrow(`${field} must be a non-empty string`)
  })

  it('stores a blank label or model as null', () => {
    const credential = fixture().store.addCredential({
      provider: 'openrouter',
      apiKey: SECRET,
      baseUrl: BASE_URL,
      label: '   ',
      model: ''
    })

    expect(credential.label).toBeNull()
    expect(credential.model).toBeNull()
  })

  it('uses the wall clock in seconds when no clock is injected', () => {
    const store = new CredentialStore(join(fixture().dir, 'wall-clock.db'))
    try {
      const before = Date.now() / 1000
      const now = store.now()
      expect(now).toBeGreaterThanOrEqual(before)
      expect(now).toBeLessThanOrEqual(Date.now() / 1000)
    } finally {
      store.close()
    }
  })
})

describe('credential store unknown ids', () => {
  it.each([
    ['getCredential', (store: CredentialStore) => store.getCredential(MISSING)],
    ['enable', (store: CredentialStore) => store.enable(MISSING)],
    ['disable', (store: CredentialStore) => store.disable(MISSING)],
    ['releaseLease', (store: CredentialStore) => store.releaseLease(MISSING)],
    ['recordSuccess', (store: CredentialStore) => store.recordSuccess(MISSING)],
    ['recordFailure', (store: CredentialStore) => store.recordFailure(MISSING, 500)]
  ])('%s refuses an id that does not exist', (_name, call) => {
    fixture().seed(1)

    expect(() => call(fixture().store)).toThrow(CredentialNotFoundError)
    expect(() => call(fixture().store)).toThrow(`no provider credential with id ${MISSING}`)
  })
})

describe('credential store duplicates and keys', () => {
  it('names the account when a key is already in that account', () => {
    const { store } = fixture()
    store.accounts().add({ provider: 'openrouter', name: 'Work', accountId: 'work' })
    store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL, accountId: 'work' })

    expect(() =>
      store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL, accountId: 'work' })
    ).toThrow(new CredentialExistsError('provider "openrouter" already has this credential for account "work"'))
  })

  it('refuses an account that belongs to another provider', () => {
    const { store } = fixture()
    store.accounts().add({ provider: 'anthropic', name: 'Work', accountId: 'work' })

    expect(() =>
      store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL, accountId: 'work' })
    ).toThrow(new AccountError('account "work" belongs to provider "anthropic", not "openrouter"'))
  })

  it('skips rows sealed by another key when checking for a duplicate', () => {
    // Same database, a different key file: the old row cannot be opened, so it
    // cannot be proven a duplicate and does not block the new one.
    const dbPath = join(fixture().dir, 'shared.db')
    const first = new CredentialStore(dbPath, { cipher: new SecretCipher(join(fixture().dir, 'key-a')) })
    first.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
    first.close()

    const second = new CredentialStore(dbPath, { cipher: new SecretCipher(join(fixture().dir, 'key-b')) })
    try {
      const added = second.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
      expect(second.getCredential(added.id).apiKey).toBe(SECRET)
    } finally {
      second.close()
    }
  })

  it('does not swallow a decrypt failure that is not a crypto refusal', () => {
    class FaultyCipher extends SecretCipher {
      broken = false

      override decrypt(envelope: string): string {
        if (this.broken) throw new TypeError('cipher bug')
        return super.decrypt(envelope)
      }
    }
    const cipher = new FaultyCipher(join(fixture().dir, 'key-faulty'))
    const store = new CredentialStore(join(fixture().dir, 'faulty.db'), { cipher })
    try {
      store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
      cipher.broken = true

      expect(() => store.addCredential({ provider: 'openrouter', apiKey: `${SECRET}-2`, baseUrl: BASE_URL })).toThrow(
        new TypeError('cipher bug')
      )
    } finally {
      store.close()
    }
  })
})

describe('account rows', () => {
  it('refuses to update or delete an account that does not exist', () => {
    const { store } = fixture()
    const values = { name: 'Ghost', email: null, workspace: null, labelsJson: '{}' }

    expect(() => store.updateAccountRow('ghost', values)).toThrow(new AccountError('unknown account "ghost"'))
    expect(() => store.deleteAccountRow('ghost')).toThrow(new AccountError('unknown account "ghost"'))
  })

  it('passes on an insert failure that is not a uniqueness clash', () => {
    const { store } = fixture()
    store.close()

    expect(() =>
      store.insertAccount({
        accountId: 'late',
        provider: 'openrouter',
        name: 'Late',
        email: null,
        workspace: null,
        labels: {}
      })
    ).toThrow(/not open/)
  })

  it('reads labels it did not write as an empty table', () => {
    // Another writer (an older build, a hand edit) can leave labels empty, as
    // invalid JSON, or as a JSON value that is not an object.
    const { store } = fixture()
    const registry = store.accounts()
    for (const id of ['blank', 'garbled', 'scalar']) registry.add({ provider: 'openrouter', name: id, accountId: id })
    const raw = new Database(store.path)
    try {
      const set = raw.prepare('UPDATE provider_accounts SET labels = ? WHERE account_id = ?')
      set.run('', 'blank')
      set.run('{not json', 'garbled')
      set.run('"just a string"', 'scalar')
    } finally {
      raw.close()
    }

    expect(registry.get('blank').labels).toEqual({})
    expect(registry.get('garbled').labels).toEqual({})
    expect(registry.get('scalar').labels).toEqual({})
  })
})

describe('account registry input', () => {
  it('refuses an empty provider or name', () => {
    const registry = fixture().store.accounts()

    expect(() => registry.add({ provider: '', name: 'Work' })).toThrow(
      new AccountError('provider must be a non-empty string')
    )
    expect(() => registry.add({ provider: 'openrouter', name: '  ' })).toThrow(
      new AccountError('name must be a non-empty string')
    )
  })

  it('refuses a name that yields no account id', () => {
    expect(() => fixture().store.accounts().add({ provider: 'openrouter', name: '!!!' })).toThrow(
      new AccountError('cannot derive an account id from name "!!!"')
    )
  })

  it('stores blank optional fields as null', () => {
    const account = fixture().store.accounts().add({ provider: 'openrouter', name: 'Blank', email: ' ', workspace: '' })

    expect(account.email).toBeNull()
    expect(account.workspace).toBeNull()
  })

  it('lists the accounts of one provider', () => {
    const registry = fixture().store.accounts()
    registry.add({ provider: 'openrouter', name: 'A' })
    registry.add({ provider: 'anthropic', name: 'B' })

    expect(registry.listAccounts('anthropic').map((account) => account.accountId)).toEqual(['anthropic:b'])
  })

  it('updates the workspace alone and keeps the labels', () => {
    const registry = fixture().store.accounts()
    registry.add({ provider: 'openrouter', name: 'Team', workspace: 'eu', labels: { tier: 'pro' } })

    const updated = registry.update('Team', { workspace: 'us' })

    expect(updated.workspace).toBe('us')
    expect(updated.labels).toEqual({ tier: 'pro' })
  })
})
