import { describe, expect, it } from 'vitest'
import { AccountError, PoolExhaustedError } from '../../src/storage/key-pool.js'
import { BASE_URL, ProviderFailure, usePoolFixture } from './key-pool-helpers.js'

const fixture = usePoolFixture()

describe('account registry', () => {
  it('derives a stable account id and resolves by id or name', () => {
    const registry = fixture().store.accounts()
    const account = registry.add({ provider: 'openrouter', name: 'Work Laptop', email: 'work@example.com' })
    expect(account.accountId).toBe('openrouter:work-laptop')
    expect(registry.get('openrouter:work-laptop')).toEqual(account)
    expect(registry.get('Work Laptop')).toEqual(account)
    expect(registry.listAccounts().map((item) => item.accountId)).toEqual([account.accountId])
  })

  it('round-trips an explicit id, workspace and labels', () => {
    const registry = fixture().store.accounts()
    const account = registry.add({
      provider: 'openrouter',
      name: 'Team',
      accountId: 'team-eu',
      workspace: 'eu-west',
      labels: { tier: 'pro', owner: 'ops' }
    })
    expect(account.accountId).toBe('team-eu')
    const loaded = registry.get('team-eu')
    expect(loaded.labels).toEqual({ tier: 'pro', owner: 'ops' })
    expect(loaded.workspace).toBe('eu-west')
  })

  it('refuses a duplicate name per provider but allows it across providers', () => {
    const registry = fixture().store.accounts()
    registry.add({ provider: 'openrouter', name: 'Work' })
    expect(() => registry.add({ provider: 'openrouter', name: 'Work' })).toThrow(AccountError)
    expect(registry.add({ provider: 'openai', name: 'Work' }).provider).toBe('openai')
  })

  it('rejects an ambiguous name lookup', () => {
    const registry = fixture().store.accounts()
    registry.add({ provider: 'openrouter', name: 'Work' })
    registry.add({ provider: 'openai', name: 'Work' })
    expect(() => registry.get('Work')).toThrow(/ambiguous/)
    expect(() => registry.get('nobody')).toThrow(/unknown account/)
    expect(registry.maybeGet('nobody')).toBeNull()
  })

  it('updates partially and can clear fields', () => {
    const registry = fixture().store.accounts()
    const account = registry.add({ provider: 'openrouter', name: 'Work', email: 'a@b.c', labels: { tier: 'free' } })
    const updated = registry.update(account.accountId, { name: 'Work EU', labels: { tier: 'pro' } })
    expect(updated.name).toBe('Work EU')
    expect(updated.email).toBe('a@b.c')
    expect(updated.labels).toEqual({ tier: 'pro' })
    const cleared = registry.update(account.accountId, { email: null, labels: null })
    expect(cleared.email).toBeNull()
    expect(cleared.labels).toEqual({})
    expect(cleared.workspace).toBeNull()
  })

  it('isolates an account pool from the default pool and other accounts', () => {
    const store = fixture().store
    const registry = store.accounts()
    const work = registry.add({ provider: 'openrouter', name: 'Work' })
    const personal = registry.add({ provider: 'openrouter', name: 'Personal' })
    const workCredential = store.addCredential({
      provider: 'openrouter',
      apiKey: 'sk-live-work-0001',
      baseUrl: BASE_URL,
      accountId: work.accountId
    })
    const homeCredential = store.addCredential({
      provider: 'openrouter',
      apiKey: 'sk-live-home-0002',
      baseUrl: BASE_URL,
      accountId: personal.accountId
    })
    expect(registry.credentials(work.accountId)).toHaveLength(1)
    expect(registry.pool(work.accountId).credentials()).toHaveLength(1)
    const used = registry.pool(work.accountId).run((credential) => credential)
    expect(used.id).toBe(workCredential.id)
    expect(used.id).not.toBe(homeCredential.id)
    expect(used.accountId).toBe(work.accountId)
    expect(fixture().store.poolCredentials('openrouter')).toEqual([])
  })

  it('requires an explicit credential drop before deleting an account', () => {
    const store = fixture().store
    const registry = store.accounts()
    const account = registry.add({ provider: 'openrouter', name: 'Temp' })
    store.addCredential({
      provider: 'openrouter',
      apiKey: 'sk-live-temp-0001',
      baseUrl: BASE_URL,
      accountId: account.accountId
    })
    expect(() => registry.remove(account.accountId)).toThrow(/still has 1 credential/)
    registry.remove(account.accountId, { dropCredentials: true })
    expect(registry.maybeGet(account.accountId)).toBeNull()
    expect(store.poolCredentials('openrouter', account.accountId)).toEqual([])
  })

  it('rotates only inside the account', () => {
    const store = fixture().store
    const registry = store.accounts()
    const work = registry.add({ provider: 'openrouter', name: 'Work' })
    const first = store.addCredential({
      provider: 'openrouter',
      apiKey: 'sk-live-work-0001',
      baseUrl: BASE_URL,
      accountId: work.accountId
    })
    const second = store.addCredential({
      provider: 'openrouter',
      apiKey: 'sk-live-work-0002',
      baseUrl: BASE_URL,
      accountId: work.accountId
    })
    const seen: number[] = []
    const result = registry.pool(work.accountId).run((credential) => {
      seen.push(credential.id)
      if (credential.id === first.id) throw new ProviderFailure(401, 'revoked')
      return 'ok'
    })
    expect(result).toBe('ok')
    expect(seen).toEqual([first.id, second.id])
    expect(() =>
      registry.pool(work.accountId).run(() => {
        throw new ProviderFailure(401, 'revoked again')
      })
    ).toThrow(PoolExhaustedError)
  })
})
