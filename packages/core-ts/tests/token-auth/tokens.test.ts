import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

import { hashToken, isValidTokenHash, TokenHashError, verifyToken } from '../../src/token-auth/tokens.js'

interface Vector {
  name: string
  token: string
  encoded: string
}

interface Fixture {
  valid: Vector[]
  wrong_token: Vector[]
  malformed: { encoded: string; message: string }[]
}

// Minted by core-py (scripts/gen_token_hash_vectors.py at the repo root).
const fixture = JSON.parse(
  readFileSync(fileURLToPath(new URL('../fixtures/token-hash-vectors.json', import.meta.url)), 'utf-8')
) as Fixture

const TAIL = '00112233445566778899aabbccddeeff'

describe('Python-minted token hash vectors', () => {
  it('covers the expected parameter sets', () => {
    const names = fixture.valid.map((v) => v.name)
    expect(names).toEqual(
      expect.arrayContaining([
        'ascii',
        'empty',
        'unicode',
        'long',
        'n1024-r4-p2',
        'dklen16',
        'dklen64',
        'uppercase-hex'
      ])
    )
    expect(fixture.malformed.length).toBeGreaterThan(10)
  })

  it.each(fixture.valid.map((v) => [v.name, v]))('verifies %s', async (_name, v) => {
    expect(await verifyToken(v.token, v.encoded)).toBe(true)
    expect(isValidTokenHash(v.encoded)).toBe(true)
  })

  it.each(fixture.wrong_token.map((v) => [v.name, v]))('rejects the wrong token for %s', async (_name, v) => {
    expect(await verifyToken(v.token, v.encoded)).toBe(false)
  })

  it.each(fixture.malformed.map((m) => [m.encoded, m.message]))(
    'raises TokenHashError for %j with the core-py message',
    async (encoded, message) => {
      const err = await verifyToken('x', encoded).catch((e: unknown) => e)
      expect(err).toBeInstanceOf(TokenHashError)
      expect((err as Error).message).toBe(message)
      expect((err as Error).name).toBe('TokenHashError')
      expect(isValidTokenHash(encoded)).toBe(false)
    }
  )
})

describe('hashToken', () => {
  it('emits the core-py default encoding and round-trips', async () => {
    const encoded = await hashToken('s3cret-token')
    expect(encoded).toMatch(/^scrypt\$32768\$8\$1\$[0-9a-f]{32}\$[0-9a-f]{64}$/)
    expect(await verifyToken('s3cret-token', encoded)).toBe(true)
    expect(await verifyToken('wrong', encoded)).toBe(false)
    expect(isValidTokenHash(encoded)).toBe(true)
  })

  it('uses a fresh salt per hash', async () => {
    expect(await hashToken('same')).not.toBe(await hashToken('same'))
  })
})

describe('strict parsing (core-ts rejects what core-py would only fail inside scrypt)', () => {
  it.each([
    [`scrypt$ 8$8$1$${TAIL}$${TAIL}`, 'whitespace inside an integer'],
    [`scrypt$1_024$8$1$${TAIL}$${TAIL}`, 'underscore digit separator'],
    [`scrypt$-2$8$1$${TAIL}$${TAIL}`, 'negative n'],
    [`scrypt$3$8$1$${TAIL}$${TAIL}`, 'n not a power of two'],
    [`scrypt$1$8$1$${TAIL}$${TAIL}`, 'n = 1'],
    [`scrypt$2147483648$8$1$${TAIL}$${TAIL}`, 'n above 2^30'],
    [`scrypt$1024$0$1$${TAIL}$${TAIL}`, 'r = 0'],
    [`scrypt$1024$8$0$${TAIL}$${TAIL}`, 'p = 0'],
    [`scrypt$99999999999999999999$8$1$${TAIL}$${TAIL}`, 'unsafe integer'],
    [`scrypt$1024$8$1$00 11$${TAIL}`, 'whitespace inside hex']
  ])('rejects %j (%s)', async (encoded) => {
    await expect(verifyToken('x', encoded)).rejects.toBeInstanceOf(TokenHashError)
    expect(isValidTokenHash(encoded)).toBe(false)
  })

  it('wraps a scrypt failure (memory ceiling) as TokenHashError with the cause', async () => {
    // 128 * 2^20 * 8 = 1 GiB working memory, above the 64 MiB ceiling.
    const encoded = `scrypt$1048576$8$1$${TAIL}$${TAIL}`
    const err = (await verifyToken('x', encoded).catch((e: unknown) => e)) as TokenHashError
    expect(err).toBeInstanceOf(TokenHashError)
    expect(err.message).toBe("malformed token hash encoding: 'scrypt$1048576$8$1$00112'")
    expect(err.cause).toBeInstanceOf(Error)
    expect(isValidTokenHash(encoded)).toBe(false)
  })

  it('rejects a non-string encoding instead of crashing', async () => {
    await expect(verifyToken('x', 42 as unknown as string)).rejects.toThrow('malformed token hash encoding: 42')
    expect(isValidTokenHash(undefined as unknown as string)).toBe(false)
  })
})
