/**
 * `runHttpServer` over real loopback HTTP: the `/mcp` refusals, a full session
 * lifecycle, the request fallbacks and the lock-file upkeep.
 *
 * HOME and USERPROFILE point at a temp dir so the lock files land there and not
 * in the account running the suite.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { McpServer } from '@modelcontextprotocol/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type HttpRoute, type HttpServerHandle, runHttpServer } from '../../src/transport/local-server.js'

const INITIALIZE = {
  jsonrpc: '2.0',
  id: 1,
  method: 'initialize',
  params: { protocolVersion: '2025-11-25', capabilities: {}, clientInfo: { name: 'probe', version: '0.0.0' } }
}

const MCP_HEADERS = { 'content-type': 'application/json', accept: 'application/json, text/event-stream' }

let home: string
let handle: HttpServerHandle | undefined

function makeMcpServer(): McpServer {
  return new McpServer({ name: 'session-test', version: '0.1.0' })
}

function locksDir(): string {
  return join(home, '.config', 'mcp', 'locks')
}

async function start(
  options: { extraRoutes?: HttpRoute[]; factory?: () => McpServer; serverName?: string } = {}
): Promise<HttpServerHandle> {
  handle = await runHttpServer(options.factory ?? makeMcpServer, {
    serverName: options.serverName ?? `session-test-${Date.now()}`,
    port: 0,
    host: '127.0.0.1',
    extraRoutes: options.extraRoutes
  })
  return handle
}

function mcpUrl(h: HttpServerHandle): string {
  return `http://${h.host}:${h.port}/mcp`
}

async function rpcError(response: Response): Promise<{ status: number; code: number; message: string }> {
  const body = (await response.json()) as { jsonrpc: string; id: string; error: { code: number; message: string } }
  expect(body.jsonrpc).toBe('2.0')
  expect(body.id).toBe('server-error')
  return { status: response.status, code: body.error.code, message: body.error.message }
}

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'mcp-local-server-'))
  vi.stubEnv('HOME', home)
  vi.stubEnv('USERPROFILE', home)
})

afterEach(async () => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  const open = handle
  handle = undefined
  if (open) await open.close()
  vi.unstubAllEnvs()
  rmSync(home, { recursive: true, force: true })
})

describe('runHttpServer /mcp refusals', () => {
  it('binds the host it is given', async () => {
    const h = await start()

    expect(h.host).toBe('127.0.0.1')
    expect(h.port).toBeGreaterThan(0)
  })

  it('picks a free port when none is given', async () => {
    handle = await runHttpServer(makeMcpServer, { serverName: `session-test-${Date.now()}` })

    expect(handle.port).toBeGreaterThan(0)
    const response = await fetch(`http://${handle.host}:${handle.port}/health`)
    expect(response.status).toBe(200)
  })

  it('answers 404 for a session id it does not know', async () => {
    const h = await start()

    const response = await fetch(mcpUrl(h), {
      method: 'POST',
      headers: { ...MCP_HEADERS, 'mcp-session-id': 'no-such-session' },
      body: JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list' })
    })

    expect(await rpcError(response)).toEqual({ status: 404, code: -32600, message: 'Session not found' })
  })

  it('answers 400 to a non-POST request without a session id', async () => {
    const h = await start()

    const response = await fetch(mcpUrl(h), { method: 'GET', headers: { accept: 'text/event-stream' } })

    expect(await rpcError(response)).toEqual({
      status: 400,
      code: -32600,
      message: 'Bad Request: Mcp-Session-Id header is required'
    })
  })

  it('answers 413 to a bootstrap body over 4 MiB', async () => {
    const h = await start()

    const response = await fetch(mcpUrl(h), {
      method: 'POST',
      headers: MCP_HEADERS,
      body: 'x'.repeat(4 * 1024 * 1024 + 1)
    })

    expect(await rpcError(response)).toEqual({ status: 413, code: -32600, message: 'Request body too large' })
  })

  it('answers a JSON-RPC parse error to a body that is not JSON', async () => {
    const h = await start()

    const response = await fetch(mcpUrl(h), { method: 'POST', headers: MCP_HEADERS, body: '{"jsonrpc":' })

    expect(await rpcError(response)).toEqual({ status: 400, code: -32700, message: 'Parse error' })
  })

  it('answers 400 to a first request that is not initialize', async () => {
    const h = await start()

    const response = await fetch(mcpUrl(h), {
      method: 'POST',
      headers: MCP_HEADERS,
      body: JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list' })
    })

    expect(await rpcError(response)).toEqual({
      status: 400,
      code: -32600,
      message: 'Bad Request: no valid session ID provided'
    })
  })
})

describe('runHttpServer session lifecycle', () => {
  it('closes a session on DELETE, even when its server fails to close', async () => {
    const closes: string[] = []
    const factory = (): McpServer => {
      const server = makeMcpServer()
      const realClose = server.close.bind(server)
      server.close = async () => {
        closes.push('close')
        await realClose()
        throw new Error('close failed')
      }
      return server
    }
    const h = await start({ factory })

    const init = await fetch(mcpUrl(h), { method: 'POST', headers: MCP_HEADERS, body: JSON.stringify(INITIALIZE) })
    expect(init.status).toBe(200)
    await init.text()
    const sessionId = init.headers.get('mcp-session-id')
    expect(sessionId).toBeTruthy()
    const session = { ...MCP_HEADERS, 'mcp-session-id': sessionId as string, 'mcp-protocol-version': '2025-11-25' }

    const deleted = await fetch(mcpUrl(h), { method: 'DELETE', headers: session })
    expect(deleted.status).toBe(200)
    await vi.waitFor(() => expect(closes).toEqual(['close']))

    const after = await fetch(mcpUrl(h), {
      method: 'POST',
      headers: session,
      body: JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list' })
    })
    expect(await rpcError(after)).toEqual({ status: 404, code: -32600, message: 'Session not found' })
  })
})

describe('runHttpServer request handling', () => {
  it('serves a request that carries no Host header', async () => {
    // HTTP/1.0 lets a client omit Host, which fetch never does.
    const h = await start()

    const raw = await new Promise<string>((resolve, reject) => {
      const socket = connect(h.port, h.host, () => socket.end('GET /health HTTP/1.0\r\n\r\n'))
      let data = ''
      socket.setEncoding('utf8')
      socket.on('data', (chunk: string) => {
        data += chunk
      })
      socket.on('end', () => resolve(data))
      socket.on('error', reject)
    })

    expect(raw.split('\r\n')[0]).toMatch(/^HTTP\/1\.[01] 200 OK$/)
    expect(raw).toContain('"status":"ok"')
  })

  it('leaves a response alone when a route fails after sending it', async () => {
    const h = await start({
      extraRoutes: [
        {
          method: 'GET',
          path: '/sent-then-fails',
          handler: (_req, res) => {
            res.writeHead(202, { 'content-type': 'text/plain' })
            res.end('sent')
            throw new Error('after the response')
          }
        }
      ]
    })

    const response = await fetch(`http://${h.host}:${h.port}/sent-then-fails`)

    expect(response.status).toBe(202)
    expect(await response.text()).toBe('sent')
  })
})

describe('runHttpServer lock upkeep', () => {
  it('sweeps stale locks of the same server at start and says so', async () => {
    const serverName = `sweep-test-${Date.now()}`
    mkdirSync(locksDir(), { recursive: true })
    const stale = new Date(Date.now() - 25 * 3600 * 1000).toISOString()
    writeFileSync(join(locksDir(), `${serverName}-1.lock`), `999999\n1\n\n${stale}\n`)
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {})

    await start({ serverName })

    expect(errors).toHaveBeenCalledWith(`[runHttpServer] cleaned 1 stale lock(s) for ${serverName}`)
  })

  it('refreshes its lock timestamp every hour', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
    const serverName = `refresh-test-${Date.now()}`
    const h = await start({ serverName })
    const lockPath = join(locksDir(), `${serverName}-${h.port}.lock`)
    const lines = readFileSync(lockPath, 'utf-8').split('\n')
    const old = '2020-01-01T00:00:00.000Z'
    writeFileSync(lockPath, `${lines[0]}\n${lines[1]}\n${lines[2]}\n${old}\n`)

    vi.advanceTimersByTime(3600 * 1000)

    const refreshed = readFileSync(lockPath, 'utf-8').split('\n')[3].trim()
    expect(refreshed).not.toBe(old)
    expect(Date.now() - new Date(refreshed).getTime()).toBeLessThan(60_000)
  })
})
