/**
 * Local MCP server entry point combining token-table auth + Streamable HTTP transport.
 *
 * Composes:
 *  1. Token-table auth (no-auth / token / multi) via Authenticator + guardRequest
 *  2. MCP Streamable HTTP transport -- serves /mcp with per-session management
 *  3. /health endpoint -- liveness probe
 *
 * This is the TypeScript counterpart of core-py's `hull_core.server.http`.
 */

import { randomUUID } from 'node:crypto'
import * as fs from 'node:fs'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import * as os from 'node:os'
import * as path from 'node:path'

import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js'
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types.js'
import { refreshLockTimestamp, sweepStaleLocks, writeLockFile } from '../lifecycle/lock.js'
import { Authenticator, type HttpAuthConfig } from '../token-auth/authenticator.js'
import { guardRequest } from '../token-auth/http-guard.js'
import { assertNoAuthLoopback } from '../token-auth/loopback.js'

export function jsonResponse(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload)
  })
  res.end(payload)
}

/**
 * A route a consumer registers on the same port ``runHttpServer`` serves.
 *
 * The built-in ``/mcp`` and ``/health`` routes are matched FIRST and cannot be
 * overridden -- a consumer cannot accidentally shadow the MCP transport or the
 * liveness probe.
 */
export interface HttpRoute {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE'
  path: string
  handler: (req: IncomingMessage, res: ServerResponse) => Promise<void> | void
}

export interface RunHttpServerOptions {
  /** Identifier used for credential storage and lifecycle locks. */
  serverName: string
  /** 0 = auto-find a free port. Default: 0. */
  port?: number
  /** Host to bind. Default '127.0.0.1'. */
  host?: string
  /** Extra HTTP routes served on the same port. */
  extraRoutes?: HttpRoute[]
  /**
   * Token-table HTTP auth, the core-py `[server] auth` modes:
   *
   *  - `{ mode: 'no-auth' }` (default) -- no Authorization header; loopback only.
   *  - `{ mode: 'token', tokenHash, rpm? }` -- one shared bearer token.
   *  - `{ mode: 'multi', usersFile?, users? }` -- users.toml table.
   */
  auth?: HttpAuthConfig
}

export interface HttpServerHandle {
  /** Actual TCP port bound. Non-zero even when ``options.port`` was 0. */
  port: number
  /** Host bound. */
  host: string
  /** Cleanly close transport + http server. */
  close: () => Promise<void>
}

/**
 * Start an HTTP MCP server on ``host:port``.
 */
export async function runHttpServer(
  serverFactory: () => McpServer,
  options: RunHttpServerOptions
): Promise<HttpServerHandle> {
  const host = options.host ?? '127.0.0.1'
  const wantedPort = options.port ?? 0

  const authConfig: HttpAuthConfig = options.auth ?? { mode: 'no-auth' }
  const tokenAuth = new Authenticator(authConfig)
  if (tokenAuth.mode === 'no-auth') {
    assertNoAuthLoopback(host)
  }

  const transports = new Map<string, StreamableHTTPServerTransport>()
  const servers = new Map<string, McpServer>()

  /** JSON-RPC error envelope, shaped like core-py's session manager replies. */
  function jsonRpcError(res: ServerResponse, status: number, code: number, message: string): void {
    jsonResponse(res, status, {
      jsonrpc: '2.0',
      id: 'server-error',
      error: { code, message }
    })
  }

  async function readBootstrapBody(
    req: IncomingMessage
  ): Promise<{ ok: true; body: unknown } | { ok: false; status: number; code: number; message: string }> {
    const MAX_BYTES = 4 * 1024 * 1024
    const chunks: Buffer[] = []
    let size = 0
    for await (const chunk of req) {
      const buf = chunk as Buffer
      size += buf.length
      if (size > MAX_BYTES) {
        return { ok: false, status: 413, code: -32600, message: 'Request body too large' }
      }
      chunks.push(buf)
    }
    try {
      return { ok: true, body: JSON.parse(Buffer.concat(chunks).toString('utf8')) }
    } catch {
      return { ok: false, status: 400, code: -32700, message: 'Parse error' }
    }
  }

  async function handleSessionRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const sessionHeader = req.headers['mcp-session-id']
    const incomingSessionId = Array.isArray(sessionHeader) ? sessionHeader[0] : sessionHeader

    if (incomingSessionId) {
      const known = transports.get(incomingSessionId)
      if (!known) {
        jsonRpcError(res, 404, -32600, 'Session not found')
        return
      }
      await known.handleRequest(req, res)
      return
    }

    if (req.method !== 'POST') {
      jsonRpcError(res, 400, -32600, 'Bad Request: Mcp-Session-Id header is required')
      return
    }

    const parsed = await readBootstrapBody(req)
    if (!parsed.ok) {
      jsonRpcError(res, parsed.status, parsed.code, parsed.message)
      return
    }
    if (!isInitializeRequest(parsed.body)) {
      jsonRpcError(res, 400, -32600, 'Bad Request: no valid session ID provided')
      return
    }

    const server = serverFactory()
    let transport: StreamableHTTPServerTransport
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      onsessioninitialized: (sessionId) => {
        transports.set(sessionId, transport)
        servers.set(sessionId, server)
      },
      onsessionclosed: (sessionId) => {
        transports.delete(sessionId)
        servers.delete(sessionId)
        server.close().catch(() => {
          /* best-effort cleanup */
        })
      }
    })
    await server.connect(transport)
    await transport.handleRequest(req, res, parsed.body)
  }

  async function dispatch(req: IncomingMessage, res: ServerResponse, pathname: string): Promise<void> {
    if (pathname === '/mcp') {
      await handleSessionRequest(req, res)
      return
    }

    if (options.extraRoutes) {
      for (const route of options.extraRoutes) {
        if (route.path === pathname && route.method === req.method) {
          await route.handler(req, res)
          return
        }
      }
    }

    jsonResponse(res, 404, { error: 'not_found' })
  }

  const handler = async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`)
    const pathname = url.pathname

    if (pathname === '/health') {
      jsonResponse(res, 200, { status: 'ok', server: options.serverName })
      return
    }

    await guardRequest(tokenAuth, req, res, () => dispatch(req, res, pathname))
  }

  const httpServer: Server = createServer((req, res) => {
    handler(req, res).catch(() => {
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ error: 'internal_error' }))
      }
    })
  })

  await new Promise<void>((resolve, reject) => {
    httpServer.once('error', reject)
    httpServer.listen(wantedPort, host, () => {
      httpServer.removeListener('error', reject)
      resolve()
    })
  })

  const addr = httpServer.address() as AddressInfo
  const actualPort = addr.port

  const swept = sweepStaleLocks(options.serverName)
  if (swept > 0) {
    console.error(`[runHttpServer] cleaned ${swept} stale lock(s) for ${options.serverName}`)
  }

  const lockDirPath = path.join(os.homedir(), '.config', 'mcp', 'locks')
  fs.mkdirSync(lockDirPath, { recursive: true, mode: 0o700 })
  if (process.platform !== 'win32') {
    fs.chmodSync(lockDirPath, 0o700)
  }
  const lockFile = writeLockFile(options.serverName, actualPort, '', lockDirPath)

  const lockRefreshInterval = setInterval(() => refreshLockTimestamp(lockFile), 3600 * 1000)
  if (typeof lockRefreshInterval.unref === 'function') {
    lockRefreshInterval.unref()
  }

  return {
    port: actualPort,
    host,
    close: async () => {
      for (const transport of transports.values()) {
        try {
          await transport.close()
        } catch {
          /* best-effort cleanup */
        }
      }
      for (const server of servers.values()) {
        try {
          await server.close()
        } catch {
          /* best-effort cleanup */
        }
      }
      transports.clear()
      servers.clear()

      if (lockRefreshInterval !== null) {
        clearInterval(lockRefreshInterval)
      }

      return new Promise<void>((resolve, reject) => {
        httpServer.close(async (err) => {
          if (lockFile) {
            try {
              await fs.promises.unlink(lockFile)
            } catch {
              /* best-effort cleanup */
            }
          }
          if (err) {
            reject(err)
          } else {
            resolve()
          }
        })
      })
    }
  }
}
