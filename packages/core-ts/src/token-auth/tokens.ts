/**
 * Bearer token hashing and verification for the HTTP token table.
 *
 * Byte-compatible with core-py `hull_core.auth.tokens`: tokens never reach
 * disk in plaintext; the host stores a scrypt hash in `users.toml` /
 * `config.toml` and verifies presented bearer tokens against it. Format:
 *
 *     scrypt$<n>$<r>$<p>$<salt-hex>$<derived-key-hex>
 *
 * Defaults match Python exactly (N=2^15, r=8, p=1, 32-byte key, 16-byte
 * salt, 64 MiB scrypt memory ceiling), so a hash minted by either language
 * verifies in the other.
 */

import { randomBytes, type ScryptOptions, scrypt, scryptSync, timingSafeEqual } from 'node:crypto'

import { pyPrefix, pyRepr } from './py-compat.js'

const SCRYPT_N = 2 ** 15
const SCRYPT_R = 8
const SCRYPT_P = 1
const DKLEN = 32
const SALT_BYTES = 16
const PREFIX = 'scrypt'
// 128 * N * r = 32 MiB working memory; grant the same explicit 64 MiB
// ceiling core-py passes to hashlib.scrypt.
const MAXMEM = 64 * 1024 * 1024

const DECIMAL = /^[0-9]+$/
const HEX = /^(?:[0-9a-fA-F]{2})+$/

/** Raised when an encoded token hash is malformed. */
export class TokenHashError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'TokenHashError'
  }
}

interface ParsedHash {
  readonly n: number
  readonly r: number
  readonly p: number
  readonly salt: Buffer
  readonly expected: Buffer
}

function malformed(encoded: unknown, cause?: unknown): TokenHashError {
  const shown = typeof encoded === 'string' ? pyRepr(pyPrefix(encoded, 24)) : String(encoded)
  return new TokenHashError(`malformed token hash encoding: ${shown}`, cause === undefined ? undefined : { cause })
}

function parseInteger(text: string): number | null {
  if (!DECIMAL.test(text)) return null
  const value = Number(text)
  return Number.isSafeInteger(value) ? value : null
}

function parseHash(encoded: unknown): ParsedHash {
  if (typeof encoded !== 'string') throw malformed(encoded)
  const parts = encoded.split('$')
  if (parts.length !== 6) throw malformed(encoded)
  const [prefix, nText, rText, pText, saltHex, dkHex] = parts
  const n = parseInteger(nText)
  const r = parseInteger(rText)
  const p = parseInteger(pText)
  if (prefix !== PREFIX || n === null || r === null || p === null) throw malformed(encoded)
  // scrypt requires N to be a power of two greater than one, r and p >= 1.
  if (n < 2 || (n & (n - 1)) !== 0 || n > 2 ** 30 || r < 1 || p < 1) throw malformed(encoded)
  if (!HEX.test(saltHex) || !HEX.test(dkHex)) throw malformed(encoded)
  return { n, r, p, salt: Buffer.from(saltHex, 'hex'), expected: Buffer.from(dkHex, 'hex') }
}

function scryptOptions(parsed: Pick<ParsedHash, 'n' | 'r' | 'p'>): ScryptOptions {
  return { N: parsed.n, r: parsed.r, p: parsed.p, maxmem: MAXMEM }
}

function scryptAsync(password: Buffer, salt: Buffer, keylen: number, options: ScryptOptions): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    scrypt(password, salt, keylen, options, (err, derived) => {
      if (err) reject(err)
      else resolve(derived)
    })
  })
}

/** Hash a bearer token into the storable `scrypt$...` encoding. */
export async function hashToken(token: string): Promise<string> {
  const salt = randomBytes(SALT_BYTES)
  const dk = await scryptAsync(
    Buffer.from(token, 'utf8'),
    salt,
    DKLEN,
    scryptOptions({ n: SCRYPT_N, r: SCRYPT_R, p: SCRYPT_P })
  )
  return [PREFIX, String(SCRYPT_N), String(SCRYPT_R), String(SCRYPT_P), salt.toString('hex'), dk.toString('hex')].join(
    '$'
  )
}

/**
 * Constant-time verify a presented token against an encoded hash.
 *
 * Rejects with `TokenHashError` when `encoded` is malformed (never resolves
 * `true` for a hash it cannot parse); resolves `false` on a mismatch.
 */
export async function verifyToken(token: string, encoded: string): Promise<boolean> {
  const parsed = parseHash(encoded)
  let actual: Buffer
  try {
    actual = await scryptAsync(Buffer.from(token, 'utf8'), parsed.salt, parsed.expected.length, scryptOptions(parsed))
  } catch (err) {
    throw malformed(encoded, err)
  }
  return timingSafeEqual(actual, parsed.expected)
}

/**
 * True when `encoded` parses as a scrypt token hash whose parameters the
 * local scrypt accepts. Like core-py `is_valid_encoding`, it runs one full
 * derivation so parameters that only fail inside scrypt (memory ceiling)
 * are caught at load time instead of on the first request. Never throws.
 */
export function isValidTokenHash(encoded: string): boolean {
  try {
    const parsed = parseHash(encoded)
    scryptSync(Buffer.alloc(0), parsed.salt, parsed.expected.length, scryptOptions(parsed))
    return true
  } catch {
    return false
  }
}
