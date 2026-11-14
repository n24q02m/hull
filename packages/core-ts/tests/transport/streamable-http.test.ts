import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { afterEach, describe, expect, it } from 'vitest'

import { StreamableHTTPServer } from '../../src/transport/streamable-http.js'

describe('StreamableHTTPServer', () => {
  it('exposes host and port properties', () => {
    const server = new McpServer({ name: 'test', version: '0.0.0' })
    const http = new StreamableHTTPServer({
      server,
      host: '127.0.0.1',
      port: 9876
    })
    expect(http.host).toBe('127.0.0.1')
    expect(http.port).toBe(9876)
  })

  it('defaults host to 127.0.0.1 when not specified', () => {
    const server = new McpServer({ name: 'test', version: '0.0.0' })
    const http = new StreamableHTTPServer({ server, port: 9000 })
    expect(http.host).toBe('127.0.0.1')
  })

  it('exposes the underlying transport for advanced use', () => {
    const server = new McpServer({ name: 'test', version: '0.0.0' })
    const http = new StreamableHTTPServer({ server, port: 9000 })
    expect(http.transport).toBeDefined()
  })

  it('connect() is idempotent — multiple calls do not re-connect', async () => {
    const server = new McpServer({ name: 'test', version: '0.0.0' })
    const http = new StreamableHTTPServer({ server, port: 9000 })
    await http.connect()
    await http.connect()
    // Should not throw; second call is a no-op.
    await http.close()
  })
})

describe('StreamableHTTPServer.handleRequest', () => {
  const INITIALIZE = {
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: { protocolVersion: '2025-11-25', capabilities: {}, clientInfo: { name: 'probe', version: '0.0.0' } }
  }

  let listener: Server | null = null

  afterEach(async () => {
    const open = listener
    listener = null
    if (open) await new Promise<void>((resolve) => open.close(() => resolve()))
  })

  /** Serve `mcp` on a loopback port and POST one MCP `initialize`. */
  async function initialize(mcp: StreamableHTTPServer): Promise<{ status: number; body: string }> {
    await mcp.connect()
    listener = createServer((req, res) => {
      void mcp.handleRequest(req, res)
    })
    const server = listener
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()))
    const { port } = server.address() as AddressInfo
    const response = await fetch(`http://127.0.0.1:${port}/mcp`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json, text/event-stream' },
      body: JSON.stringify(INITIALIZE)
    })
    return { status: response.status, body: await response.text() }
  }

  it('answers MCP requests when no auth middleware is set', async () => {
    const mcp = new StreamableHTTPServer({ server: new McpServer({ name: 'open', version: '1.2.3' }), port: 0 })
    try {
      const { status, body } = await initialize(mcp)

      expect(status).toBe(200)
      expect(body).toContain('"serverInfo":{"name":"open","version":"1.2.3"}')
    } finally {
      await mcp.close()
    }
  })

  it('passes the request to the transport once the middleware allows it', async () => {
    const seen: string[] = []
    const mcp = new StreamableHTTPServer({
      server: new McpServer({ name: 'guarded', version: '1.0.0' }),
      port: 0,
      authMiddleware: async (req) => {
        seen.push(req.method ?? '')
        return true
      }
    })
    try {
      const { status, body } = await initialize(mcp)

      expect(seen).toEqual(['POST'])
      expect(status).toBe(200)
      expect(body).toContain('"name":"guarded"')
    } finally {
      await mcp.close()
    }
  })

  it('stops at the middleware when it refuses, leaving its own response', async () => {
    const mcp = new StreamableHTTPServer({
      server: new McpServer({ name: 'guarded', version: '1.0.0' }),
      port: 0,
      authMiddleware: (_req, res) => {
        res.writeHead(401, { 'content-type': 'application/json', 'www-authenticate': 'Bearer' })
        res.end('{"error":"unauthorized"}')
        return false
      }
    })
    try {
      const { status, body } = await initialize(mcp)

      expect(status).toBe(401)
      expect(body).toBe('{"error":"unauthorized"}')
    } finally {
      await mcp.close()
    }
  })
})
