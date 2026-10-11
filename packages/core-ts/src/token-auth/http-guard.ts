/**
 * node:http guard that authenticates a request and binds the identity,
 * the counterpart of core-py `hull_core.auth.asgi.HullAuthMiddleware`.
 *
 * Rejections are byte-identical to core-py: a JSON body
 * `{"error": <status text>, "detail": <detail>}` (Python `json.dumps`
 * formatting), `content-type: application/json`, an explicit
 * `content-length`, and `www-authenticate: Bearer` on 401.
 */

import type { IncomingMessage, ServerResponse } from 'node:http'
import type { AuthOutcome } from './authenticator.js'
import { runWithAuthContext } from './context.js'
import { pyJsonDumps } from './py-compat.js'

/** Status -> `error` field, same table as core-py `_STATUS_TEXT`. */
export const AUTH_STATUS_TEXT: Readonly<Record<number, string>> = Object.freeze({
  401: 'unauthorized',
  403: 'forbidden',
  429: 'rate limit exceeded',
  500: 'server misconfiguration'
})

/** Minimal authenticator surface the guard needs. */
export interface RequestAuthenticator {
  authenticate(authz: string | null | undefined): Promise<AuthOutcome>
}

/** Write a core-py-compatible JSON auth rejection. */
export function writeAuthRejection(res: ServerResponse, status: number, detail: string): void {
  const body = Buffer.from(pyJsonDumps({ error: AUTH_STATUS_TEXT[status] ?? 'error', detail }), 'utf8')
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(body.length)
  }
  if (status === 401) headers['www-authenticate'] = 'Bearer'
  res.writeHead(status, headers)
  res.end(body)
}

/**
 * Authenticate `req`; on success run `next` with the resolved `AuthContext`
 * bound for its whole async scope (so `currentUser()` works inside MCP tool
 * handlers), otherwise write the rejection and never call `next`.
 */
export async function guardRequest(
  authenticator: RequestAuthenticator,
  req: IncomingMessage,
  res: ServerResponse,
  next: () => Promise<void>
): Promise<void> {
  const result = await authenticator.authenticate(req.headers.authorization)
  if (!result.ok) {
    writeAuthRejection(res, result.status ?? 500, result.detail)
    return
  }
  const context = result.context
  if (context === null) {
    // ok=true always carries a context; fail closed rather than bind a
    // null identity for the rest of the request.
    writeAuthRejection(res, 500, 'authenticator returned no context')
    return
  }
  await runWithAuthContext(context, next)
}
