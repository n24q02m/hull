import { readdirSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

import * as tokenAuth from '../../src/token-auth/index.js'

const SRC = fileURLToPath(new URL('../../src/token-auth/', import.meta.url))

describe('token-auth subpath export (./token-auth)', () => {
  const pkg = JSON.parse(readFileSync(fileURLToPath(new URL('../../package.json', import.meta.url)), 'utf-8')) as {
    exports: Record<string, { import: string; types: string }>
    dependencies: Record<string, string>
  }

  it('points ./token-auth at the barrel', () => {
    expect(pkg.exports['./token-auth']).toEqual({
      import: './build/token-auth/index.js',
      types: './build/token-auth/index.d.ts'
    })
  })

  it('declares the TOML parser as a runtime dependency', () => {
    expect(pkg.dependencies['smol-toml']).toBeDefined()
  })

  it('the barrel exposes the consumer API', () => {
    expect(typeof tokenAuth.Authenticator).toBe('function')
    expect(typeof tokenAuth.hashToken).toBe('function')
    expect(typeof tokenAuth.currentUser).toBe('function')
    expect(typeof tokenAuth.guardRequest).toBe('function')
    expect(typeof tokenAuth.tokenCliHandler).toBe('function')
  })
})

describe('token-auth import isolation', () => {
  // Token mode must work with no relay/OAuth/transport machinery loaded, so
  // the module may only import node builtins, smol-toml and its siblings.
  const files = readdirSync(SRC).filter((f) => f.endsWith('.ts'))

  it('has the expected module set', () => {
    expect(files.sort()).toEqual([
      'authenticator.ts',
      'cli.ts',
      'context.ts',
      'http-guard.ts',
      'index.ts',
      'limiter.ts',
      'loopback.ts',
      'py-compat.ts',
      'tokens.ts',
      'users.ts'
    ])
  })

  it.each(files)('%s imports only node builtins, smol-toml and siblings', (file) => {
    const text = readFileSync(`${SRC}${file}`, 'utf-8')
    const specifiers = [...text.matchAll(/\bfrom\s+'([^']+)'/g)].map((m) => m[1])
    for (const spec of specifiers) {
      const allowed = spec.startsWith('node:') || spec === 'smol-toml' || /^\.\/[a-z-]+\.js$/.test(spec)
      expect(allowed, `${file} imports ${spec}`).toBe(true)
    }
  })
})
