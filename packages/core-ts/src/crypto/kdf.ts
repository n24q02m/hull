const encoder = new TextEncoder()

// HKDF info label inherited from the former relay handshake. It is an input to
// the derived key, so changing it changes every key (see the cross-language
// vectors in tests/fixtures/crypto-vectors.json); the value must stay.
const INFO = encoder.encode('mcp-relay')

export async function deriveAesKey(sharedSecret: ArrayBuffer, passphrase: string): Promise<CryptoKey> {
  const salt = encoder.encode(passphrase)
  const keyMaterial = await crypto.subtle.importKey('raw', sharedSecret, 'HKDF', false, ['deriveKey'])

  return crypto.subtle.deriveKey(
    { name: 'HKDF', hash: 'SHA-256', salt, info: INFO },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  )
}
