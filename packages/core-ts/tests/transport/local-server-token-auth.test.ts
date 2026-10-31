/**
 * Integration tests for `runHttpServer({ auth })` (token-table HTTP auth).
 * Starts a real HTTP server per test and drives it with `fetch` and the MCP
 * SDK Streamable HTTP client.
 */
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'

import { AuthConfigError } from '../../src/token-auth/authenticator.js'
import { currentUser } from '../../src/token-auth/context.js'
import { ServerConfigError } from '../../src/token-auth/loopback.js'
import { hashToken } from '../../src/token-auth/tokens.js'
import type { HttpRoute } from '../../src/transport/local-server.js'
import { type HttpServerHandle, runHttpServer } from '../../src/transport/local-server.js'

function makeMcpServer(): McpServer {
  const server = new McpServer({ name: 'test', version: '0.1.0' })
  server.registerTool('whoami', { description: 'echo the caller identity' }, async () => {
    // Cross an await so the identity must survive the async hop.
    await new Promise((resolve) => setTimeout(resolve, 1))
    const user = currentUser()
    return { content: [{ type: 'text', text: JSON.stringify({ uid: user.uid, namespace: user.namespace }) }] }
  })
  return server
}

const whoamiRoute: HttpRoute = {
  method: 'GET',
  path: '/whoami',
  handler: (_req, res) => {
    const user = currentUser()
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ uid: user.uid, namespace: user.namespace, mode: user.mode }))
  }
}

let dir: string
let sharedHash: string
let usersPath: string
let handle: HttpServerHandle | undefined

beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), 'hull-http-auth-'))
  const [shared, alice, bob, carol] = await Promise.all([
    hashToken('shared-token'),
    hashToken('alice-token'),
    hashToken('bob-token'),
    hashToken('carol-token')
  ])
  sharedHash = shared
  usersPath = join(dir, 'users.toml')
  writeFileSync(
    usersPath,
    [
      '[users.alice]',
      `token_hash = "${alice}"`,
      'namespace = "ns-alice"',
      '',
      '[users.bob]',
      `token_hash = "${bob}"`,
      'namespace = "ns-bob"',
      'enabled = false',
      '',
      '[users.carol]',
      `token_hash = "${carol}"`,
      'namespace = "ns-carol"',
      '[users.carol.limits]',
      'rpm = 1',
      ''
    ].join('\n')
  )
})

afterAll(() => {
  rmSync(dir, { recursive: true, force: true })
})

afterEach(async () => {
  await handle?.close()
  handle = undefined
})

function base(): string {
  if (!handle) throw new Error('no server')
  return `http://${handle.host}:${handle.port}`
}

async function callWhoami(token: string): Promise<unknown> {
  const client = new Client({ name: 'token-auth-test', version: '0.0.0' })
  const transport = new StreamableHTTPClientTransport(new URL(`${base()}/mcp`), {
    requestInit: { headers: { authorization: `Bearer ${token}` } }
  })
  await client.connect(transport)
  try {
    const result = await client.callTool({ name: 'whoami', arguments: {} })
    const [first] = result.content as { type: string; text: string }[]
    return JSON.parse(first.text)
  } finally {
    await client.close()
  }
}

describe('runHttpServer auth: token mode', () => {
  it('rejects /mcp without or with a wrong token, with the core-py body and challenge', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-token-auth-${Date.now()}`,
      port: 0,
      auth: { mode: 'token', tokenHash: sharedHash }
    })
    const attempts: Record<string, string>[] = [{}, { authorization: 'Bearer wrong' }]
    for (const headers of attempts) {
      const res = await fetch(`${base()}/mcp`, { method: 'POST', headers, body: '{}' })
      expect(res.status).toBe(401)
      expect(res.headers.get('www-authenticate')).toBe('Bearer')
      expect(res.headers.get('content-type')).toBe('application/json')
      const body = await res.text()
      expect(body.startsWith('{"error": "unauthorized", "detail": ')).toBe(true)
    }
  })

  it('serves MCP for the right token and exposes the caller to tools via currentUser()', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-token-auth-mcp-${Date.now()}`,
      port: 0,
      auth: { mode: 'token', tokenHash: sharedHash }
    })
    expect(await callWhoami('shared-token')).toEqual({ uid: 'shared', namespace: 'default' })
  })

  it('keeps /health open and guards extra routes', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-token-auth-routes-${Date.now()}`,
      port: 0,
      auth: { mode: 'token', tokenHash: sharedHash },
      extraRoutes: [whoamiRoute]
    })
    const health = await fetch(`${base()}/health`)
    expect(health.status).toBe(200)
    expect((await health.json()) as unknown).toMatchObject({ status: 'ok' })

    expect((await fetch(`${base()}/whoami`)).status).toBe(401)
    const ok = await fetch(`${base()}/whoami`, { headers: { authorization: 'Bearer shared-token' } })
    expect(await ok.json()).toEqual({ uid: 'shared', namespace: 'default', mode: 'token' })

    // Unknown paths are guarded too, then fall through to the 404.
    expect((await fetch(`${base()}/nope`)).status).toBe(401)
    expect((await fetch(`${base()}/nope`, { headers: { authorization: 'Bearer shared-token' } })).status).toBe(404)
  })

  it('validates the config before binding', async () => {
    await expect(
      runHttpServer(makeMcpServer, { serverName: 'test-token-bad', port: 0, auth: { mode: 'token', tokenHash: '' } })
    ).rejects.toBeInstanceOf(AuthConfigError)
  })
})

describe('runHttpServer auth: multi mode', () => {
  it('maps each user to their namespace, 403s disabled users and 429s over the limit', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-multi-auth-${Date.now()}`,
      port: 0,
      auth: { mode: 'multi', usersFile: usersPath },
      extraRoutes: [whoamiRoute]
    })
    expect(await callWhoami('alice-token')).toEqual({ uid: 'alice', namespace: 'ns-alice' })

    const bob = await fetch(`${base()}/whoami`, { headers: { authorization: 'Bearer bob-token' } })
    expect(bob.status).toBe(403)
    expect(bob.headers.get('www-authenticate')).toBeNull()
    expect(await bob.text()).toBe(`{"error": "forbidden", "detail": "user 'bob' is disabled"}`)

    const carol = { headers: { authorization: 'Bearer carol-token' } }
    const first = await fetch(`${base()}/whoami`, carol)
    expect(await first.json()).toEqual({ uid: 'carol', namespace: 'ns-carol', mode: 'multi' })
    const second = await fetch(`${base()}/whoami`, carol)
    expect(second.status).toBe(429)
    expect(await second.text()).toBe('{"error": "rate limit exceeded", "detail": "rate limit exceeded"}')

    expect((await fetch(`${base()}/whoami`, { headers: { authorization: 'Bearer shared-token' } })).status).toBe(401)
  })
})

describe('runHttpServer auth: no-auth mode', () => {
  it('serves loopback binds as the local identity', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-noauth-${Date.now()}`,
      port: 0,
      auth: { mode: 'no-auth' },
      extraRoutes: [whoamiRoute]
    })
    const res = await fetch(`${base()}/whoami`)
    expect(await res.json()).toEqual({ uid: 'local', namespace: 'default', mode: 'no-auth' })
    expect(await callWhoami('ignored')).toEqual({ uid: 'local', namespace: 'default' })
  })

  it.each(['0.0.0.0', '::', '192.168.1.10'])('refuses the non-loopback host %s before binding', async (host) => {
    await expect(
      runHttpServer(makeMcpServer, { serverName: 'test-noauth-public', port: 0, host, auth: { mode: 'no-auth' } })
    ).rejects.toBeInstanceOf(ServerConfigError)
  })
})
