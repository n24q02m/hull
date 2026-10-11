/**
 * Generate the TypeScript -> Python token-hash parity fixture.
 *
 * Writes `packages/core-py/tests/auth/fixtures/ts_token_hash_vectors.json`:
 * scrypt token hashes minted by core-ts `hashToken` (plus non-default
 * parameter sets built with node:crypto directly). core-py
 * `hull_core.auth.tokens.verify_token` must accept every entry and reject
 * the wrong-token variants.
 *
 * Run from packages/core-ts:
 *
 *     bun scripts/gen-token-hash-vectors.ts
 *
 * Each run mints fresh random salts; only regenerate when the vector set
 * changes. The tokens are test values, never real credentials.
 */

import { scryptSync } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { hashToken, isValidTokenHash, verifyToken } from '../src/token-auth/tokens.js'

const OUT = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  'core-py',
  'tests',
  'auth',
  'fixtures',
  'ts_token_hash_vectors.json'
)

const DEFAULT_TOKENS: Record<string, string> = {
  ascii: 's3cret-token',
  empty: '',
  'inner-spaces': 'two words here',
  unicode: 'm\u1eadt-kh\u1ea9u-Vi\u1ec7t-\u{1f511}',
  long: 'y'.repeat(1024)
}

interface Custom {
  name: string
  token: string
  n: number
  r: number
  p: number
  dklen: number
  salt: string
}

const CUSTOM: Custom[] = [
  { name: 'n16384-r8-p1', token: 'custom-a', n: 16384, r: 8, p: 1, dklen: 32, salt: '11'.repeat(16) },
  { name: 'n1024-r4-p2', token: 'custom-b', n: 1024, r: 4, p: 2, dklen: 32, salt: '5a'.repeat(16) },
  { name: 'dklen16', token: 'custom-c', n: 2048, r: 8, p: 1, dklen: 16, salt: 'e1'.repeat(8) },
  { name: 'dklen64', token: 'custom-d', n: 2048, r: 8, p: 1, dklen: 64, salt: '1e'.repeat(24) }
]

interface Vector {
  name: string
  token: string
  encoded: string
}

async function main(): Promise<void> {
  const valid: Vector[] = []
  for (const [name, token] of Object.entries(DEFAULT_TOKENS))
    valid.push({ name, token, encoded: await hashToken(token) })
  for (const c of CUSTOM) {
    const salt = Buffer.from(c.salt, 'hex')
    const dk = scryptSync(Buffer.from(c.token, 'utf8'), salt, c.dklen, {
      N: c.n,
      r: c.r,
      p: c.p,
      maxmem: 64 * 1024 * 1024
    })
    const encoded = ['scrypt', c.n, c.r, c.p, salt.toString('hex'), dk.toString('hex')].join('$')
    valid.push({ name: c.name, token: c.token, encoded })
  }
  for (const v of valid) {
    if (!(await verifyToken(v.token, v.encoded)) || !isValidTokenHash(v.encoded)) {
      throw new Error(`self-check failed for ${v.name}`)
    }
  }
  const wrong: Vector[] = []
  for (const v of valid) {
    const token = `${v.token}-wrong`
    if (await verifyToken(token, v.encoded)) throw new Error(`wrong token verified for ${v.name}`)
    wrong.push({ name: v.name, token, encoded: v.encoded })
  }
  const fixture = {
    _comment:
      'TS-minted scrypt token hashes (@n24q02m/hull token-auth) that core-py must verify; ' +
      'regenerate with `bun scripts/gen-token-hash-vectors.ts` in packages/core-ts. Test tokens only.',
    valid,
    wrong_token: wrong
  }
  // Escape non-ASCII like Python json.dumps(ensure_ascii=True) so the file stays ASCII.
  const text = JSON.stringify(fixture, null, 2).replace(
    /[\u007f-\uffff]/g,
    (ch) => `\\u${ch.charCodeAt(0).toString(16).padStart(4, '0')}`
  )
  mkdirSync(dirname(OUT), { recursive: true })
  writeFileSync(OUT, `${text}\n`, 'utf8')
  console.log(`wrote ${OUT}`)
}

await main()
