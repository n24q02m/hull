/**
 * Encryption at rest for pooled provider credentials.
 *
 * The provider pool lives in `~/.hull/hull.db`, so every API key is sealed
 * with AES-256-GCM under a 32-byte machine key at `~/.hull/.credential-key`
 * (created 0600 on first use, never overwritten). Envelope:
 *
 *   v1:<base64 nonce>:<base64 ciphertext || 16-byte tag>
 *
 * `packages/core-py/src/hull_core/credentials/crypto.py` writes and reads the
 * same envelope with the same key file, so either stack can open a pool the
 * other created; both test suites pin one shared vector.
 */

import { createCipheriv, createDecipheriv, createHmac, randomBytes } from 'node:crypto'
import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from 'node:fs'
import { dirname } from 'node:path'

export const SECRET_KEY_LENGTH = 32
export const SECRET_NONCE_LENGTH = 12
const ENVELOPE_PREFIX = 'v1'
const TAG_LENGTH = 16

export class CredentialCryptoError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CredentialCryptoError'
  }
}

function checkedKey(key: Buffer): Buffer {
  if (key.length !== SECRET_KEY_LENGTH) {
    throw new CredentialCryptoError(
      `credential key must be exactly ${SECRET_KEY_LENGTH} raw bytes (found ${key.length}); ` +
        'hull will not overwrite or delete it. Move it aside to mint a fresh key, then re-add the ' +
        'provider credentials — the old envelopes cannot be recovered without the original key.'
    )
  }
  return key
}

/**
 * Read the 32-byte credential key, minting it 0600 on first use.
 *
 * Creation is exclusive (`wx`): two processes racing on the first use converge
 * on the winner's key instead of the later write winning. A pre-existing file
 * of any other length is refused, never replaced.
 */
export function loadOrCreateKey(path: string): Buffer {
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
  let fd: number
  try {
    fd = openSync(path, 'wx', 0o600)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
    return checkedKey(readFileSync(path))
  }
  const key = randomBytes(SECRET_KEY_LENGTH)
  try {
    writeSync(fd, key)
    fsyncSync(fd)
  } finally {
    closeSync(fd)
  }
  return key
}

/** Seal `plaintext` into a `v1:<nonce>:<ciphertext+tag>` envelope. */
export function encryptSecret(
  key: Buffer,
  plaintext: string,
  nonce: Buffer = randomBytes(SECRET_NONCE_LENGTH)
): string {
  checkedKey(key)
  if (nonce.length !== SECRET_NONCE_LENGTH) {
    throw new Error(`nonce must be exactly ${SECRET_NONCE_LENGTH} bytes`)
  }
  const cipher = createCipheriv('aes-256-gcm', key, nonce)
  const ciphertext = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()])
  const payload = Buffer.concat([ciphertext, cipher.getAuthTag()])
  return `${ENVELOPE_PREFIX}:${nonce.toString('base64')}:${payload.toString('base64')}`
}

/** Open an envelope produced by {@link encryptSecret}; never echoes it. */
export function decryptSecret(key: Buffer, envelope: string): string {
  checkedKey(key)
  const parts = envelope.split(':')
  if (parts.length !== 3 || parts[0] !== ENVELOPE_PREFIX) {
    throw new CredentialCryptoError('credential secret is not a v1 hull envelope')
  }
  const nonce = Buffer.from(parts[1], 'base64')
  const payload = Buffer.from(parts[2], 'base64')
  // Node's base64 decoder is lenient; round-tripping rejects `@@@`-style junk.
  if (
    nonce.toString('base64') !== parts[1] ||
    payload.toString('base64') !== parts[2] ||
    nonce.length !== SECRET_NONCE_LENGTH ||
    payload.length <= TAG_LENGTH
  ) {
    throw new CredentialCryptoError('credential secret envelope is corrupt')
  }
  try {
    const decipher = createDecipheriv('aes-256-gcm', key, nonce)
    decipher.setAuthTag(payload.subarray(payload.length - TAG_LENGTH))
    const plaintext = Buffer.concat([
      decipher.update(payload.subarray(0, payload.length - TAG_LENGTH)),
      decipher.final()
    ])
    return plaintext.toString('utf8')
  } catch {
    // Wrong key or damaged row: the message carries neither envelope nor plaintext.
    throw new CredentialCryptoError('credential secret is corrupt or the key changed')
  }
}

/** Display-only tail of a key (`…abcd`); never enough to reconstruct it. */
export function keyHint(apiKey: string): string {
  return apiKey.length >= 8 ? `…${apiKey.slice(-4)}` : '…'
}

/**
 * Keyed HMAC-SHA256 fingerprint for duplicate detection, never for auth.
 *
 * An unkeyed digest of a provider key is offline-crackable and leaks equality
 * across machines; HMAC under the local `.credential-key` keeps duplicate
 * detection while making the digest useless without that key.
 */
export function keyFingerprint(key: Buffer, apiKey: string): string {
  return createHmac('sha256', checkedKey(key)).update(apiKey, 'utf8').digest('hex')
}

/** Lazy key-file holder used by {@link CredentialStore}. */
export class SecretCipher {
  readonly keyPath: string
  private cachedKey: Buffer | null = null

  constructor(keyPath: string) {
    this.keyPath = keyPath
  }

  get key(): Buffer {
    if (this.cachedKey === null) this.cachedKey = loadOrCreateKey(this.keyPath)
    return this.cachedKey
  }

  fingerprint(plaintext: string): string {
    return keyFingerprint(this.key, plaintext)
  }

  encrypt(plaintext: string): string {
    return encryptSecret(this.key, plaintext)
  }

  decrypt(envelope: string): string {
    return decryptSecret(this.key, envelope)
  }
}
