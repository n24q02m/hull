import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  CredentialCryptoError,
  decryptSecret,
  encryptSecret,
  keyHint,
  loadOrCreateKey,
  SECRET_KEY_LENGTH
} from '../../src/storage/key-pool-crypto.js'

const VECTOR_KEY = Buffer.from(Array.from({ length: 32 }, (_, index) => index))
const VECTOR_NONCE = Buffer.from(Array.from({ length: 12 }, (_, index) => index))
const VECTOR_PLAINTEXT = 'sk-hull-vector-0123456789'
const VECTOR_ENVELOPE = 'v1:AAECAwQFBgcICQoL:NGn7c7CJrjb7JPT/3ptVXbLktADFTWhEAcYamPqU91OPf2S7BjLOftk='

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'hull-pool-crypto-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

describe('credential secret envelope', () => {
  it('round-trips through the versioned envelope', () => {
    const envelope = encryptSecret(VECTOR_KEY, 'sk-roundtrip-' + '1234')
    const [prefix, nonce, payload] = envelope.split(':')
    expect(prefix).toBe('v1')
    expect(Buffer.from(nonce, 'base64')).toHaveLength(12)
    expect(Buffer.from(payload, 'base64').length).toBeGreaterThan(16)
    expect(decryptSecret(VECTOR_KEY, envelope)).toBe('sk-roundtrip-1234')
  })

  it('matches the shared vector pinned by core-py', () => {
    expect(encryptSecret(VECTOR_KEY, VECTOR_PLAINTEXT, VECTOR_NONCE)).toBe(VECTOR_ENVELOPE)
    expect(decryptSecret(VECTOR_KEY, VECTOR_ENVELOPE)).toBe(VECTOR_PLAINTEXT)
  })

  it.each([
    'not-an-envelope',
    'v2:AAECAwQFBgcICQoL:NGn7c7CJrjb7JPT/3ptVXbLktADFTWhEAcYamPqU91OPf2S7BjLOftk=',
    'v1:@@@:@@@',
    'v1:AAECAwQFBgcICQoL:AAAA'
  ])('rejects a malformed envelope without echoing it (%s)', (envelope) => {
    expect(() => decryptSecret(VECTOR_KEY, envelope)).toThrow(CredentialCryptoError)
    try {
      decryptSecret(VECTOR_KEY, envelope)
    } catch (error) {
      expect(String(error)).not.toContain(VECTOR_PLAINTEXT)
      expect(String(error)).not.toContain(envelope)
    }
  })

  it('reports a wrong key as corrupt without leaking the plaintext', () => {
    const envelope = encryptSecret(VECTOR_KEY, VECTOR_PLAINTEXT)
    expect(() => decryptSecret(Buffer.alloc(32), envelope)).toThrow(/corrupt or the key changed/)
    try {
      decryptSecret(Buffer.alloc(32), envelope)
    } catch (error) {
      expect(String(error)).not.toContain(VECTOR_PLAINTEXT)
    }
  })

  it('rejects a bad key length and nonce', () => {
    expect(() => encryptSecret(Buffer.from('short'), 'value')).toThrow(CredentialCryptoError)
    expect(() => encryptSecret(VECTOR_KEY, 'value', Buffer.from('short'))).toThrow(/nonce must be exactly 12 bytes/)
  })
})

describe('credential key file', () => {
  it('is created once, stays 32 bytes and is reused', () => {
    const path = join(dir, 'nested', '.credential-key')
    const first = loadOrCreateKey(path)
    const second = loadOrCreateKey(path)
    expect(first.equals(second)).toBe(true)
    expect(first).toHaveLength(SECRET_KEY_LENGTH)
    expect(readFileSync(path).equals(first)).toBe(true)
  })

  it('never replaces a foreign key file', () => {
    const path = join(dir, '.credential-key')
    writeFileSync(path, Buffer.from('0123456789abcdef'))
    expect(() => loadOrCreateKey(path)).toThrow(/will not overwrite/)
    expect(readFileSync(path).toString()).toBe('0123456789abcdef')
  })
})

describe('key hint', () => {
  it('never exposes the key itself', () => {
    const apiKey = 'sk-live-super-secret-abcd'
    expect(keyHint(apiKey)).toBe('…abcd')
    expect(keyHint(apiKey)).not.toContain(apiKey)
    expect(keyHint('short')).toBe('…')
  })
})
