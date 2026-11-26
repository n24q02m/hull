/**
 * Streamable HTTP 2026-07-28 transport wrapper for @n24q02m/hull.
 *
 * Thin convenience wrapper around `@modelcontextprotocol/node`'s
 * `NodeStreamableHTTPServerTransport` that pairs it with an `McpServer`
 * instance, optional Bearer validation middleware, and optional MCP CORS headers.
 *
 * The SDK already implements the full 2025-11-25 transport spec
 * (session management, SSE streaming, stateful / stateless modes),
 * so this wrapper focuses on the mcp-core ergonomics:
 *
 *  - one-call `connect()` that wires server <-> transport
 *  - `handleRequest(req, res)` that applies the auth middleware
 *    first and short-circuits with 401 on missing / invalid token
 *  - explicit `host` / `port` properties for lifecycle management
 *    (the actual port bind is left to the caller's http.Server)
 */

import type { IncomingMessage, ServerResponse } from 'node:http'

import {
  NodeStreamableHTTPServerTransport,
  type StreamableHTTPServerTransportOptions
} from '@modelcontextprotocol/node'
import type { McpServer } from '@modelcontextprotocol/server'

import { applyMcpCorsHeaders } from '../extensions/index.js'

export interface StreamableHTTPServerOptions {
  /** MCP server instance that handles requests */
  server: McpServer
  /** Host to advertise (used for logging / display only) */
  host?: string
  /** Port to advertise (used for logging / display only) */
  port: number
  /** Optional middleware for request authorization */
  authMiddleware?: (req: IncomingMessage, res: ServerResponse) => Promise<boolean> | boolean
  /** Optional CORS allowed origins (enables MCP 2026-07-28 headers + OPTIONS preflight) */
  corsAllowOrigins?: readonly string[]
  /** Transport options forwarded to NodeStreamableHTTPServerTransport */
  transportOptions?: StreamableHTTPServerTransportOptions
}

export class StreamableHTTPServer {
  readonly host: string
  readonly port: number
  private readonly _server: McpServer
  private readonly _auth?: (req: IncomingMessage, res: ServerResponse) => Promise<boolean> | boolean
  private readonly _corsAllowOrigins?: readonly string[]
  private readonly _transport: NodeStreamableHTTPServerTransport
  private _connected = false

  constructor(options: StreamableHTTPServerOptions) {
    this._server = options.server
    this.host = options.host ?? '127.0.0.1'
    this.port = options.port
    this._auth = options.authMiddleware
    this._corsAllowOrigins = options.corsAllowOrigins
    this._transport = new NodeStreamableHTTPServerTransport(
      options.transportOptions ?? { sessionIdGenerator: undefined }
    )
  }

  /** Connect the MCP server to the transport. Call once before handleRequest. */
  async connect(): Promise<void> {
    if (this._connected) return
    await this._server.connect(this._transport)
    this._connected = true
  }

  /**
   * Handle an incoming HTTP request. Applies CORS headers (and preflight)
   * and auth validation first if configured, then delegates to the SDK transport.
   */
  async handleRequest(req: IncomingMessage, res: ServerResponse, parsedBody?: unknown): Promise<void> {
    if (this._corsAllowOrigins?.length && applyMcpCorsHeaders(req, res, this._corsAllowOrigins)) {
      return
    }
    if (this._auth) {
      const allowed = await this._auth(req, res)
      if (!allowed) return
    }
    await this._transport.handleRequest(req, res, parsedBody)
  }

  /** Expose the underlying transport for advanced use cases. */
  get transport(): NodeStreamableHTTPServerTransport {
    return this._transport
  }

  /** Close the transport and drain any active sessions. */
  async close(): Promise<void> {
    await this._transport.close()
    this._connected = false
  }
}
