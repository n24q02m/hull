import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import Database from 'better-sqlite3'
import { afterEach, describe, expect, it } from 'vitest'
import { AccountError, CredentialExistsError, CredentialNotFoundError } from '../../src/storage/key-pool.js'
import {
  CredentialStore,
  credentialKeyPath,
  defaultHullDbPath,
  KEY_POOL_SCHEMA
} from '../../src/storage/key-pool-store.js'
import { setHomeDirForTesting } from '../../src/storage/per-plugin-store.js'
import { BASE_URL, usePoolFixture } from './key-pool-helpers.js'

const fixture = usePoolFixture()
const SECRET = 'sk-live-store-secret-0001'

afterEach(() => {
  setHomeDirForTesting(null)
})

describe('schema and encryption at rest', () => {
  it('adds tables without disturbing an older database', () => {
    const path = join(fixture().dir, 'legacy.db')
    const legacy = new Database(path)
    legacy.exec('CREATE TABLE kv (namespace TEXT, key TEXT, value TEXT)')
    legacy.prepare('INSERT INTO kv VALUES (?, ?, ?)').run('old', 'k', 'v')
    legacy.close()
    const first = new CredentialStore(path, { now: fixture().clock.now })
    first.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
    first.close()
    const second = new CredentialStore(path, { now: fixture().clock.now })
    try {
      expect(second.credentials().map((item) => item.apiKey)).toEqual([SECRET])
    } finally {
      second.close()
    }
    const raw = new Database(path)
    try {
      expect(raw.prepare('SELECT value FROM kv').get()).toEqual({ value: 'v' })
      const tables = raw
        .prepare<[], { name: string }>("SELECT name FROM sqlite_master WHERE type='table'")
        .all()
        .map((row) => row.name)
      expect(tables).toEqual(expect.arrayContaining(['kv', 'provider_accounts', 'provider_credentials']))
    } finally {
      raw.close()
    }
    expect(KEY_POOL_SCHEMA).toContain('CREATE TABLE IF NOT EXISTS provider_accounts')
  })

  it('stores an encrypted envelope, never the key itself', () => {
    const store = fixture().store
    const credential = store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
    expect(credential.apiKey).toBe(SECRET)
    expect(credential.keyHint).toBe('…0001')
    const raw = new Database(join(fixture().dir, 'hull.db'))
    try {
      const row = raw.prepare<[], { secret: string }>('SELECT secret FROM provider_credentials').get()
      expect(row).toBeDefined()
      if (!row) throw new Error('credential row missing')
      expect(row.secret.startsWith('v1:')).toBe(true)
      expect(row.secret).not.toContain(SECRET)
    } finally {
      raw.close()
    }
    for (const file of ['hull.db', 'hull.db-wal']) {
      let bytes: Buffer
      try {
        bytes = readFileSync(join(fixture().dir, file))
      } catch {
        continue
      }
      expect(bytes.includes(Buffer.from(SECRET))).toBe(false)
    }
    expect(store.getCredential(credential.id).apiKey).toBe(SECRET)
  })
})

describe('credential CRUD', () => {
  it('refuses duplicates and orders by priority', () => {
    const store = fixture().store
    const first = store.addCredential({ provider: 'openrouter', apiKey: 'key-one-0001', baseUrl: BASE_URL })
    const second = store.addCredential({
      provider: 'openrouter',
      apiKey: 'key-two-0002',
      baseUrl: BASE_URL,
      label: 'backup'
    })
    expect([first.priority, second.priority]).toEqual([0, 1])
    expect(store.poolCredentials('openrouter').map((item) => item.id)).toEqual([first.id, second.id])
    expect(() =>
      store.addCredential({ provider: 'openrouter', apiKey: 'key-one-0001', baseUrl: `${BASE_URL}/v2` })
    ).toThrow(CredentialExistsError)
    store.setPriority(first.id, 9)
    expect(store.poolCredentials('openrouter').map((item) => item.id)).toEqual([second.id, first.id])
    store.removeCredential(second.id)
    expect(() => store.removeCredential(second.id)).toThrow(CredentialNotFoundError)
    expect(() => store.setPriority(second.id, 1)).toThrow(CredentialNotFoundError)
  })

  it('reports health and clears it on enable/reset', () => {
    const store = fixture().store
    const credential = store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
    store.recordFailure(credential.id, 500, { now: fixture().clock.value })
    store.recordFailure(credential.id, 503, { now: fixture().clock.value })
    let health = store.health('openrouter')[0]
    expect([health.disabled, health.lastStatus, health.failureCount]).toEqual([false, 503, 2])
    expect(health.usable).toBe(true)
    expect(JSON.stringify(health)).not.toContain(SECRET)
    store.disable(credential.id)
    expect(store.health()[0].usable).toBe(false)
    store.enable(credential.id)
    health = store.health()[0]
    expect([health.disabled, health.lastStatus, health.failureCount]).toEqual([false, null, 0])
  })

  it('keeps a cooldown until it expires', () => {
    const store = fixture().store
    const credential = store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
    store.recordFailure(credential.id, 429, {
      now: fixture().clock.value,
      cooldownUntil: fixture().clock.value + 30
    })
    const health = store.poolHealth('openrouter')[0]
    expect(health.usable).toBe(false)
    expect(health.cooldownRemaining).toBeCloseTo(30)
    expect(store.leaseFirst('openrouter', null, { now: fixture().clock.value, leaseSeconds: 60 })).toBeNull()
    fixture().clock.advance(31)
    const leased = store.leaseFirst('openrouter', null, { now: fixture().clock.value, leaseSeconds: 60 })
    expect(leased?.id).toBe(credential.id)
  })

  it('counts every blocking state in diagnose', () => {
    const store = fixture().store
    const first = store.addCredential({ provider: 'openrouter', apiKey: 'key-a-0001', baseUrl: BASE_URL })
    const second = store.addCredential({ provider: 'openrouter', apiKey: 'key-b-0002', baseUrl: BASE_URL })
    store.disable(first.id)
    store.recordFailure(second.id, 429, {
      now: fixture().clock.value,
      cooldownUntil: fixture().clock.value + 10
    })
    expect(store.diagnose('openrouter')).toEqual({
      total: 2,
      disabled: 1,
      coolingDown: 1,
      leased: 0,
      retryAfter: 10
    })
    expect(store.diagnose('missing').total).toBe(0)
  })

  it('scopes resetHealth to a provider', () => {
    const store = fixture().store
    const first = store.addCredential({ provider: 'openrouter', apiKey: 'key-a-0001', baseUrl: BASE_URL })
    const second = store.addCredential({
      provider: 'openai',
      apiKey: 'key-b-0002',
      baseUrl: 'https://api.openai.com/v1'
    })
    store.disable(first.id)
    store.disable(second.id)
    expect(store.resetHealth('openrouter')).toBe(1)
    expect(store.health('openrouter')[0].disabled).toBe(false)
    expect(store.health('openai')[0].disabled).toBe(true)
    expect(store.resetHealth()).toBe(2)
    expect(store.health('openai')[0].disabled).toBe(false)
  })
})

describe('pool resolution and migration', () => {
  it('resolves a cell to its provider/account pool', () => {
    const store = fixture().store
    const account = store.accounts().add({ provider: 'openrouter', name: 'Work' })
    store.addCredential({
      provider: 'openrouter',
      apiKey: SECRET,
      baseUrl: BASE_URL,
      accountId: account.accountId
    })
    const scoped = store.poolForCell({
      baseUrl: BASE_URL,
      apiKey: SECRET,
      model: 'm',
      account: account.accountId
    })
    expect(scoped.credentials()).toHaveLength(1)
    expect(scoped.accountId).toBe(account.accountId)
    const inferred = store.poolForCell({ baseUrl: 'https://api.openai.com/v1', model: 'm' })
    expect(inferred.provider).toBe('openai')
    expect(inferred.accountId).toBeNull()
  })

  it('migrates a single-key config into one entry, idempotently', () => {
    const store = fixture().store
    const cells = ['embed', 'rerank', 'chat', 'jev_score'].map((task) => ({
      task,
      baseUrl: BASE_URL,
      apiKey: SECRET,
      model: 'model-x'
    }))
    expect(store.migrateModelCells(cells)).toEqual({ provider: 'openrouter', added: 1, skipped: 3 })
    expect(store.pool('openrouter').credentials()[0].apiKey).toBe(SECRET)
    expect(store.migrateModelCells(cells)).toEqual({ provider: 'openrouter', added: 0, skipped: 4 })
    expect(store.poolCredentials('openrouter')).toHaveLength(1)
  })

  it('keeps distinct keys as ordered entries', () => {
    const store = fixture().store
    const result = store.migrateModelCells([
      { baseUrl: BASE_URL, apiKey: 'key-a-0001' },
      { baseUrl: 'https://openrouter.ai/api/v2', apiKey: 'key-b-0002' },
      { baseUrl: BASE_URL, apiKey: 'key-c-0003' }
    ])
    expect(result).toEqual({ provider: 'openrouter', added: 3, skipped: 0 })
    expect(store.poolCredentials('openrouter').map((item) => item.apiKey)).toEqual([
      'key-a-0001',
      'key-b-0002',
      'key-c-0003'
    ])
  })

  it('treats an empty or keyless config as a no-op', () => {
    const store = fixture().store
    expect(store.migrateModelCells([{ baseUrl: BASE_URL }])).toEqual({
      provider: 'openrouter',
      added: 0,
      skipped: 0
    })
    expect(store.migrateModelCells([])).toEqual({ provider: 'default', added: 0, skipped: 0 })
  })

  it('requires an explicit provider when keyed cells span providers', () => {
    const store = fixture().store
    const cells = [
      { baseUrl: BASE_URL, apiKey: 'key-a-0001' },
      { baseUrl: 'https://api.openai.com/v1', apiKey: 'key-b-0002' }
    ]
    expect(() => store.migrateModelCells(cells)).toThrow(/span providers/)
    expect(store.migrateModelCells(cells, { provider: 'gateway' })).toEqual({
      provider: 'gateway',
      added: 2,
      skipped: 0
    })
  })

  it('migrates into an existing account only', () => {
    const store = fixture().store
    expect(() => store.migrateModelCells([{ baseUrl: BASE_URL, apiKey: SECRET, account: 'work' }])).toThrow(
      AccountError
    )
    const account = store.accounts().add({ provider: 'openrouter', name: 'Work' })
    const result = store.migrateModelCells([{ baseUrl: BASE_URL, apiKey: SECRET, account: account.accountId }])
    expect(result.added).toBe(1)
    expect(store.poolCredentials('openrouter')).toEqual([])
    expect(store.poolCredentials('openrouter', account.accountId)).toHaveLength(1)
  })
})

describe('default paths', () => {
  it('follow the hull home directory and create it on first use', () => {
    setHomeDirForTesting(fixture().dir)
    const path = defaultHullDbPath()
    expect(path).toBe(join(fixture().dir, '.hull', 'hull.db'))
    expect(credentialKeyPath(path)).toBe(join(fixture().dir, '.hull', '.credential-key'))
    const store = new CredentialStore()
    try {
      store.addCredential({ provider: 'openrouter', apiKey: SECRET, baseUrl: BASE_URL })
      expect(store.credentials()).toHaveLength(1)
    } finally {
      store.close()
    }
    expect(existsSync(path)).toBe(true)
    expect(() => readFileSync(join(fixture().dir, '.hull', '.credential-key'))).not.toThrow()
  })
})
