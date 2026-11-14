/**
 * A refused lease explains itself with the counts read in the same transaction.
 *
 * `PoolExhaustedError` carries the pool counts so a caller can tell "every key
 * is leased by another in-flight call, wait" from "every key is disabled or
 * cooling, give up". Those counts used to come from a second read after the
 * refused lease committed: a holder in another process that released its lease
 * in between made the error report `leased=0` for a pool that was busy, and a
 * caller waiting on leases gave up. Without exclusions a refusal means every
 * entry is disabled, cooling down or leased, so the counts must add up to the
 * total. Mirrors `packages/core-py/tests/credentials/test_exhaustion_snapshot.py`.
 */

import { type ChildProcess, spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { type KeyPool, PoolExhaustedError } from '../../src/storage/key-pool.js'
import { CredentialStore } from '../../src/storage/key-pool-store.js'
import { usePoolFixture } from './key-pool-helpers.js'

const fixture = usePoolFixture()

// Each refusal is one chance for the holder to release between the refused
// lease and the counts; the old code lost about one in a thousand of them.
const REFUSALS = 10_000
// SQLite's busy handler is not fair: the holder can starve for a while behind
// this process's writes, so the loop runs to a refusal count, not a round count.
const DEADLINE_MS = 90_000
const HOLDER = fileURLToPath(new URL('../fixtures/lease-holder.mjs', import.meta.url))

type EntryPoint = (pool: KeyPool) => Promise<void>

const ENTRY_POINTS: Record<string, EntryPoint> = {
  acquire: async (pool) => {
    pool.reportSuccess(pool.acquire())
  },
  run: async (pool) => {
    pool.run(() => undefined)
  },
  runAsync: async (pool) => {
    await pool.runAsync(async () => undefined)
  }
}

function unexplained(error: PoolExhaustedError): boolean {
  return error.disabled + error.coolingDown + error.leased !== error.total
}

const holders: ChildProcess[] = []

afterEach(() => {
  for (const holder of holders.splice(0)) holder.kill('SIGTERM')
})

/** Start the other process and wait until its own connection is open. */
async function startHolder(dbPath: string, credentialId: number, now: number): Promise<ChildProcess> {
  const holder = spawn(
    process.execPath,
    [HOLDER, dbPath, String(credentialId), String(now), String(DEADLINE_MS + 30_000)],
    {
      stdio: ['ignore', 'pipe', 'inherit']
    }
  )
  holders.push(holder)
  await new Promise<void>((resolve, reject) => {
    holder.once('error', reject)
    holder.once('exit', (code) => reject(new Error(`lease holder exited early with ${code}`)))
    holder.stdout?.on('data', (chunk: Buffer) => {
      if (chunk.toString().includes('ready')) resolve()
    })
  })
  return holder
}

function stopped(holder: ChildProcess): Promise<void> {
  return new Promise((resolve) => {
    if (holder.exitCode !== null) {
      resolve()
      return
    }
    holder.once('exit', () => resolve())
    holder.kill('SIGTERM')
  })
}

describe('a refusal under cross-process contention', () => {
  it.each(Object.keys(ENTRY_POINTS))(
    'counts the lease that caused it (%s)',
    async (entry) => {
      const { store, clock } = fixture()
      const [id] = fixture().seed(1)
      const holder = await startHolder(store.path, id, clock.value)
      const pool = store.pool('openrouter')
      const refusals: PoolExhaustedError[] = []
      const deadline = Date.now() + DEADLINE_MS
      while (refusals.length < REFUSALS && Date.now() < deadline) {
        try {
          await ENTRY_POINTS[entry](pool)
        } catch (error) {
          if (!(error instanceof PoolExhaustedError)) throw error
          refusals.push(error)
        }
      }
      await stopped(holder)

      // The other process really held the key: otherwise nothing was tested.
      expect(refusals.length).toBe(REFUSALS)
      expect(refusals.filter(unexplained).map((error) => error.message)).toEqual([])
    },
    DEADLINE_MS + 30_000
  )
})

/**
 * Release the holder's lease right after the pool's lease attempt is refused:
 * the moment another process can act before anything else the pool reads.
 */
function releaseAfterRefusal(store: CredentialStore, holder: CredentialStore, credentialId: number): void {
  const methods = store as unknown as Record<string, (...args: unknown[]) => unknown>
  for (const name of ['leaseFirst', 'leaseFirstOrDiagnose']) {
    const real = methods[name]
    if (typeof real !== 'function') continue
    methods[name] = (...args: unknown[]): unknown => {
      const result = real.apply(store, args)
      const refused = Array.isArray(result) ? result[0] === null : result === null
      if (refused) holder.releaseLease(credentialId)
      return result
    }
  }
}

describe('a release right after the refusal', () => {
  it.each(Object.keys(ENTRY_POINTS))('does not turn a busy pool into an unexplained exhaustion (%s)', async (entry) => {
    const { store } = fixture()
    const [id] = fixture().seed(1)
    // A second store on the same file is a second connection, as another process has.
    const holder = new CredentialStore(store.path, { now: store.now.bind(store) })
    try {
      holder.pool('openrouter').acquire()
      releaseAfterRefusal(store, holder, id)

      const error = await ENTRY_POINTS[entry](store.pool('openrouter')).then(
        () => null,
        (caught: unknown) => caught
      )

      expect(error).toBeInstanceOf(PoolExhaustedError)
      const exhausted = error as PoolExhaustedError
      expect([exhausted.total, exhausted.leased]).toEqual([1, 1])
      expect(store.diagnose('openrouter').leased).toBe(0)
    } finally {
      holder.close()
    }
  })
})

describe('leaseFirstOrDiagnose', () => {
  it('leases a usable entry and returns no counts', () => {
    const { store, clock } = fixture()
    const [id] = fixture().seed(1)

    const [credential, detail] = store.leaseFirstOrDiagnose('openrouter', null, {
      now: clock.value,
      leaseSeconds: 60
    })

    expect(credential?.id).toBe(id)
    expect(detail).toBeNull()
    expect(store.diagnose('openrouter').leased).toBe(1)
  })

  it('returns the counts behind a refusal', () => {
    const { store, clock } = fixture()
    const [leased, cooling, disabled] = fixture().seed(3)
    expect(store.leaseFirst('openrouter', null, { now: clock.value, leaseSeconds: 60 })?.id).toBe(leased)
    store.recordFailure(cooling, 429, { now: clock.value, cooldownUntil: clock.value + 30 })
    store.disable(disabled)

    const [credential, detail] = store.leaseFirstOrDiagnose('openrouter', null, {
      now: clock.value,
      leaseSeconds: 60
    })

    expect(credential).toBeNull()
    expect(detail).toEqual({ total: 3, disabled: 1, coolingDown: 1, leased: 1, retryAfter: 30 })
  })
})

describe('exhaustedError', () => {
  it('reads live counts when there is no refusal to report', () => {
    const { store } = fixture()
    const [id] = fixture().seed(1)
    store.disable(id)

    const error = store.pool('openrouter').exhaustedError('upstream said no')

    expect([error.total, error.disabled, error.coolingDown, error.leased]).toEqual([1, 1, 0, 0])
    expect(error.lastError).toBe('upstream said no')
  })
})

describe('static guard', () => {
  it('leases through the call that reports its own refusal', () => {
    // A pool that leases with `leaseFirst` must read the counts afterwards,
    // which is the race this file pins down.
    const source = readFileSync(
      join(fileURLToPath(new URL('../../src/storage/', import.meta.url)), 'key-pool.ts'),
      'utf8'
    )
    expect(source).not.toMatch(/\.leaseFirst\(/)
    expect(source).toMatch(/\.leaseFirstOrDiagnose\(/)
  })
})
