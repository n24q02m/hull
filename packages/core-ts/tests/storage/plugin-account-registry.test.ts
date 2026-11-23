import { describe, expect, it } from 'vitest'
import { InMemoryBackend } from '../../src/storage/backends.js'
import { assertValidAccountAlias, NamedAccountRegistry } from '../../src/storage/plugin-account-registry.js'

describe('NamedAccountRegistry (Tier-3 per-plugin multi-account registry)', () => {
  it('validates account aliases', () => {
    expect(() => assertValidAccountAlias('work-1_a.b')).not.toThrow()
    for (const bad of ['', '.', '..', 'a/b', 'a..b']) {
      expect(() => assertValidAccountAlias(bad)).toThrow(/Invalid account alias/)
    }
  })

  it('supports CRUD and active-account switching', async () => {
    const reg = new NamedAccountRegistry('notion', null, { backend: new InMemoryBackend() })
    expect(await reg.listAccounts()).toEqual([])
    expect(await reg.getActiveAccount()).toBeNull()
    expect(await reg.getAccount()).toBeNull()
    expect(await reg.getAccount('work')).toBeNull()

    await reg.putAccount('work', { token: 'ntn-work' })
    expect(await reg.listAccounts()).toEqual(['work'])
    expect(await reg.getActiveAccount()).toBe('work')
    expect(await reg.getAccount()).toEqual({ token: 'ntn-work' })
    expect(await reg.getAccount('work')).toEqual({ token: 'ntn-work' })

    await reg.putAccount('personal', { token: 'ntn-personal' }, { setActive: false })
    expect(await reg.listAccounts()).toEqual(['personal', 'work'])
    expect(await reg.getActiveAccount()).toBe('work')
    expect(await reg.getAccount('personal')).toEqual({ token: 'ntn-personal' })

    await reg.setActive('personal')
    expect(await reg.getActiveAccount()).toBe('personal')
    expect(await reg.getAccount()).toEqual({ token: 'ntn-personal' })

    await expect(reg.setActive('missing')).rejects.toThrow(/Unknown account alias/)
    await expect(reg.putAccount('bad', null as unknown as Record<string, unknown>)).rejects.toThrow(
      /Account credentials must be an object/
    )

    expect(await reg.removeAccount('missing')).toBe(false)
    expect(await reg.removeAccount('personal')).toBe(true)
    expect(await reg.getActiveAccount()).toBe('work')
    expect(await reg.listAccounts()).toEqual(['work'])

    expect(await reg.removeAccount('work')).toBe(true)
    expect(await reg.listAccounts()).toEqual([])
    expect(await reg.getActiveAccount()).toBeNull()
    expect(await reg.store.hasStoredBlob()).toBe(false)
  })

  it('recovers when stored active alias is missing from accounts', async () => {
    const reg = new NamedAccountRegistry('notion', 'sub-1', {
      backend: new InMemoryBackend(),
      profile: 'work'
    })
    process.env.CREDENTIAL_SECRET = 'test-master'
    try {
      await reg.store.save({ active: 'ghost', accounts: { alpha: { k: 'v' }, invalid: 'skip' } })
      expect(await reg.getActiveAccount()).toBe('alpha')
      expect(await reg.listAccounts()).toEqual(['alpha'])
      expect(await reg.getAccount()).toEqual({ k: 'v' })
    } finally {
      delete process.env.CREDENTIAL_SECRET
    }
  })
})
