/**
 * Integration tests for the ``runHttpServer`` entry point.
 */
import { existsSync } from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { McpServer } from '@modelcontextprotocol/server'
import { afterEach, describe, expect, it } from 'vitest'

import { type HttpServerHandle, runHttpServer } from '../../src/transport/local-server.js'

function makeMcpServer(): McpServer {
  return new McpServer({ name: 'test', version: '0.1.0' })
}

let handle: HttpServerHandle | undefined

afterEach(async () => {
  if (handle) {
    await handle.close()
    handle = undefined
  }
})

describe('runHttpServer', () => {
  it('serves /health without auth', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-srv-${Date.now()}`,
      port: 0
    })

    const resp = await fetch(`http://${handle.host}:${handle.port}/health`)
    expect(resp.status).toBe(200)
    const body = (await resp.json()) as { status: string; server: string }
    expect(body.status).toBe('ok')
  })

  it('returns 404 for unknown routes', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-srv-${Date.now()}`,
      port: 0
    })

    const resp = await fetch(`http://${handle.host}:${handle.port}/unknown-route`)
    expect(resp.status).toBe(404)
    const body = (await resp.json()) as { error: string }
    expect(body.error).toBe('not_found')
  })

  it('creates and cleans up lifecycle lock file', async () => {
    const serverName = `test-lock-${Date.now()}`
    handle = await runHttpServer(makeMcpServer, {
      serverName,
      port: 0
    })

    const lockPath = path.join(os.homedir(), '.config', 'mcp', 'locks', `${serverName}-${handle.port}.lock`)
    expect(existsSync(lockPath)).toBe(true)

    await handle.close()
    expect(existsSync(lockPath)).toBe(false)
    handle = undefined
  })

  it('serves MCP 2026-07-28 CORS headers and handles OPTIONS preflight when configured', async () => {
    handle = await runHttpServer(makeMcpServer, {
      serverName: `test-cors-${Date.now()}`,
      port: 0,
      corsAllowOrigins: ['https://app.example']
    })

    const preflight = await fetch(`http://${handle.host}:${handle.port}/mcp`, {
      method: 'OPTIONS',
      headers: { origin: 'https://app.example' }
    })
    expect(preflight.status).toBe(204)
    expect(preflight.headers.get('access-control-allow-origin')).toBe('https://app.example')
    expect(preflight.headers.get('access-control-allow-headers')).toContain('Mcp-Method')
    expect(preflight.headers.get('access-control-allow-headers')).toContain('Mcp-Name')

    const disallowed = await fetch(`http://${handle.host}:${handle.port}/health`, {
      headers: { origin: 'https://evil.example' }
    })
    expect(disallowed.status).toBe(200)
    expect(disallowed.headers.get('access-control-allow-origin')).toBeNull()
  })
})
