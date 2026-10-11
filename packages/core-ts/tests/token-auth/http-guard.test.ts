import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'

import type { AuthOutcome } from '../../src/token-auth/authenticator.js'
import { Authenticator } from '../../src/token-auth/authenticator.js'
import { type AuthContext, currentUser } from '../../src/token-auth/context.js'
import {
  AUTH_STATUS_TEXT,
  guardRequest,
  type RequestAuthenticator,
  writeAuthRejection
} from '../../src/token-auth/http-guard.js'
import { hashToken } from '../../src/token-auth/tokens.js'

let server: Server | undefined

afterEach(async () => {
  if (server) await new Promise<void>((resolve) => server?.close(() => resolve()))
  server = undefined
})

async function serve(handler: (req: IncomingMessage, res: ServerResponse) => Promise<void> | void): Promise<string> {
  server = createServer((req, res) => {
    Promise.resolve(handler(req, res)).catch((err: unknown) => {
      res.writeHead(599)
      res.end(String(err))
    })
  })
  await new Promise<void>((resolve) => server?.listen(0, '127.0.0.1', resolve))
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`
}

function fixed(result: Partial<AuthOutcome>): RequestAuthenticator {
  return {
    authenticate: async () => ({ context: null, status: null, detail: '', ok: false, ...result })
  }
}

function guarded(authn: RequestAuthenticator, seen: { ctx?: AuthContext; called: number }) {
  return (req: IncomingMessage, res: ServerResponse) =>
    guardRequest(authn, req, res, async () => {
      seen.called += 1
      await new Promise((resolve) => setTimeout(resolve, 5))
      seen.ctx = currentUser()
      res.writeHead(200, { 'content-type': 'text/plain' })
      res.end('ok')
    })
}

describe('AUTH_STATUS_TEXT', () => {
  it('matches core-py _STATUS_TEXT', () => {
    expect(AUTH_STATUS_TEXT).toEqual({
      401: 'unauthorized',
      403: 'forbidden',
      429: 'rate limit exceeded',
      500: 'server misconfiguration'
    })
    expect(Object.isFrozen(AUTH_STATUS_TEXT)).toBe(true)
  })
})

describe('guardRequest over node:http', () => {
  it('401s with the exact core-py body and a Bearer challenge', async () => {
    const authn = new Authenticator({ mode: 'token', tokenHash: await hashToken('right') })
    const seen = { called: 0 }
    const url = await serve(guarded(authn, seen))

    const res = await fetch(url, { headers: { authorization: 'Bearer wrong' } })
    expect(res.status).toBe(401)
    expect(res.headers.get('content-type')).toBe('application/json')
    expect(res.headers.get('www-authenticate')).toBe('Bearer')
    const body = await res.text()
    expect(body).toBe('{"error": "unauthorized", "detail": "invalid token"}')
    expect(res.headers.get('content-length')).toBe(String(Buffer.byteLength(body)))

    const missing = await fetch(url)
    expect(missing.status).toBe(401)
    expect(await missing.text()).toBe(
      '{"error": "unauthorized", "detail": "missing or malformed Authorization header"}'
    )
    expect(seen.called).toBe(0)
  })

  it('binds the context for the whole async scope of next, then unbinds it', async () => {
    const authn = new Authenticator({ mode: 'token', tokenHash: await hashToken('right') })
    const seen: { ctx?: AuthContext; called: number } = { called: 0 }
    const url = await serve(guarded(authn, seen))

    const res = await fetch(url, { headers: { authorization: 'Bearer right' } })
    expect(res.status).toBe(200)
    expect(await res.text()).toBe('ok')
    expect(seen.called).toBe(1)
    expect(seen.ctx).toMatchObject({ uid: 'shared', namespace: 'default', mode: 'token' })
    // Outside the request scope the local identity is back.
    expect(currentUser().uid).toBe('local')
  })

  it.each([
    [403, "user 'bob' is disabled", `{"error": "forbidden", "detail": "user 'bob' is disabled"}`],
    [429, 'rate limit exceeded', '{"error": "rate limit exceeded", "detail": "rate limit exceeded"}']
  ])('%i rejections carry no challenge header', async (status, detail, body) => {
    const seen = { called: 0 }
    const url = await serve(guarded(fixed({ status, detail }), seen))
    const res = await fetch(url, { headers: { authorization: 'Bearer x' } })
    expect(res.status).toBe(status)
    expect(res.headers.get('www-authenticate')).toBeNull()
    expect(await res.text()).toBe(body)
    expect(seen.called).toBe(0)
  })

  it('fails closed with 500 when a non-ok outcome has no status', async () => {
    const seen = { called: 0 }
    const url = await serve(guarded(fixed({ status: null, detail: 'odd' }), seen))
    const res = await fetch(url)
    expect(res.status).toBe(500)
    expect(await res.text()).toBe('{"error": "server misconfiguration", "detail": "odd"}')
    expect(seen.called).toBe(0)
  })

  it('fails closed with 500 when an ok outcome carries no context', async () => {
    const seen = { called: 0 }
    const url = await serve(guarded(fixed({ ok: true, context: null }), seen))
    const res = await fetch(url)
    expect(res.status).toBe(500)
    expect(await res.text()).toBe('{"error": "server misconfiguration", "detail": "authenticator returned no context"}')
    expect(seen.called).toBe(0)
  })
})

describe('writeAuthRejection', () => {
  it("falls back to 'error' for an unknown status and escapes non-ASCII like json.dumps", async () => {
    const url = await serve((_req, res) => writeAuthRejection(res, 418, 'caf\u00e9'))
    const res = await fetch(url)
    expect(res.status).toBe(418)
    expect(res.headers.get('www-authenticate')).toBeNull()
    expect(await res.text()).toBe('{"error": "error", "detail": "caf\\u00e9"}')
  })
})
