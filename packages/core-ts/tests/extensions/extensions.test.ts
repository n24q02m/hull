import type { IncomingMessage, ServerResponse } from 'node:http'

import {
  CLIENT_CAPABILITIES_META_KEY,
  PROTOCOL_VERSION_META_KEY,
  SdkError,
  SdkErrorCode,
  type ServerContext
} from '@modelcontextprotocol/server'
import { describe, expect, it, vi } from 'vitest'
import { z } from 'zod'

import {
  applyMcpCorsHeaders,
  buildFilePayload,
  elicitWithFallback,
  FILES_EXTENSION_ID,
  MCP_2026_07_28_HEADERS,
  MCP_CORS_ALLOW_HEADERS,
  MCP_CORS_EXPOSE_HEADERS
} from '../../src/extensions/index.js'

function mockRes() {
  const headers: Record<string, string> = {}
  let statusCode = 0
  let ended = false
  const res = {
    setHeader(name: string, value: string) {
      headers[name] = value
    },
    writeHead(status: number) {
      statusCode = status
    },
    end() {
      ended = true
    }
  } as unknown as ServerResponse
  return { res, headers, getStatus: () => statusCode, isEnded: () => ended }
}

function makeCtx(overrides: Partial<ServerContext['mcpReq']> = {}): ServerContext {
  const controller = new AbortController()
  return {
    mcpReq: {
      id: 1,
      method: 'tools/call',
      signal: controller.signal,
      send: vi.fn(),
      notify: vi.fn(),
      requestState: vi.fn(),
      log: vi.fn(),
      elicitInput: vi.fn(),
      requestSampling: vi.fn(),
      ...overrides
    }
  } as unknown as ServerContext
}

describe('MCP 2026-07-28 headers & CORS helper', () => {
  it('exports header constants', () => {
    expect(MCP_2026_07_28_HEADERS).toEqual(['Mcp-Method', 'Mcp-Name', 'Mcp-Protocol-Version', 'Mcp-Session-Id'])
    expect(MCP_CORS_ALLOW_HEADERS).toContain('Authorization')
    expect(MCP_CORS_EXPOSE_HEADERS).toContain('Mcp-Session-Id')
  })

  it('handles empty origins, wildcard, exact origin array header, mismatch, and OPTIONS preflight', () => {
    const empty = mockRes()
    expect(applyMcpCorsHeaders({ headers: {}, method: 'POST' } as IncomingMessage, empty.res, [])).toBe(false)
    expect(empty.headers).toEqual({})

    const wildcard = mockRes()
    expect(
      applyMcpCorsHeaders(
        { headers: { origin: 'https://any.example' }, method: 'OPTIONS' } as IncomingMessage,
        wildcard.res,
        ['*']
      )
    ).toBe(true)
    expect(wildcard.headers['Access-Control-Allow-Origin']).toBe('*')
    expect(wildcard.getStatus()).toBe(204)
    expect(wildcard.isEnded()).toBe(true)

    const arrayOrigin = mockRes()
    expect(
      applyMcpCorsHeaders(
        { headers: { origin: ['https://allowed.example'] as unknown as string }, method: 'POST' } as IncomingMessage,
        arrayOrigin.res,
        ['https://allowed.example']
      )
    ).toBe(false)
    expect(arrayOrigin.headers['Access-Control-Allow-Origin']).toBe('https://allowed.example')
    expect(arrayOrigin.headers.Vary).toBe('Origin')

    const denied = mockRes()
    applyMcpCorsHeaders(
      { headers: { origin: 'https://blocked.example' }, method: 'POST' } as IncomingMessage,
      denied.res,
      ['https://allowed.example']
    )
    expect(denied.headers['Access-Control-Allow-Origin']).toBeUndefined()
  })
})

describe('buildFilePayload', () => {
  it('builds text and binary payloads and rejects invalid inputs', () => {
    expect(FILES_EXTENSION_ID).toBe('io.modelcontextprotocol/files')

    const textPayload = buildFilePayload('hello world', 'greeting.txt')
    expect(textPayload).toEqual({
      extension: FILES_EXTENSION_ID,
      uri: 'file:///greeting.txt',
      name: 'greeting.txt',
      mimeType: 'text/plain; charset=utf-8',
      sizeBytes: 11,
      text: 'hello world'
    })

    const bytes = new Uint8Array([1, 2, 3, 4])
    const binPayload = buildFilePayload(bytes, 'blob.bin', { mimeType: 'application/x-bin', uri: 'file:///custom' })
    expect(binPayload).toEqual({
      extension: FILES_EXTENSION_ID,
      uri: 'file:///custom',
      name: 'blob.bin',
      mimeType: 'application/x-bin',
      sizeBytes: 4,
      blob: Buffer.from(bytes).toString('base64')
    })

    expect(() => buildFilePayload('abc', '   ')).toThrow('non-empty')
    expect(() => buildFilePayload(123 as unknown as string, 'bad.txt')).toThrow(TypeError)
  })
})

describe('elicitWithFallback', () => {
  const schema = z.object({ confirm: z.boolean() })
  const jsonSchema = {
    type: 'object' as const,
    properties: { confirm: { type: 'boolean' as const } },
    required: ['confirm']
  }

  it('handles 2026-07-28 MRTR inputResponses (accept with/without schema, invalid schema re-issue, decline, cancel)', async () => {
    const acceptedSchemaCtx = makeCtx({
      inputResponses: { confirm: { action: 'accept', content: { confirm: true } } }
    })
    const okSchema = await elicitWithFallback(acceptedSchemaCtx, {
      key: 'confirm',
      message: 'Deploy?',
      requestedSchema: schema,
      schema
    })
    expect(okSchema).toEqual({ status: 'accepted', data: { confirm: true }, source: 'mrtr' })

    const invalidSchemaCtx = makeCtx({
      inputResponses: { confirm: { action: 'accept', content: { confirm: 'not-bool' } } }
    })
    const reissued = await elicitWithFallback(invalidSchemaCtx, {
      key: 'confirm',
      message: 'Deploy?',
      requestedSchema: schema,
      schema
    })
    expect(reissued.status).toBe('input_required')

    const acceptedRawCtx = makeCtx({
      inputResponses: { confirm: { action: 'accept' } }
    })
    const okRaw = await elicitWithFallback<{ confirm?: boolean }>(acceptedRawCtx, {
      key: 'confirm',
      message: 'Deploy?',
      requestedSchema: jsonSchema
    })
    expect(okRaw).toEqual({ status: 'accepted', data: {}, source: 'mrtr' })

    const declinedCtx = makeCtx({ inputResponses: { confirm: { action: 'decline' } } })
    expect(
      await elicitWithFallback(declinedCtx, {
        key: 'confirm',
        message: 'Deploy?',
        requestedSchema: schema,
        defaultValue: { confirm: false }
      })
    ).toEqual({ status: 'declined', data: { confirm: false } })

    const cancelledCtx = makeCtx({ inputResponses: { confirm: { action: 'cancel' } } })
    expect(
      await elicitWithFallback(cancelledCtx, {
        key: 'confirm',
        message: 'Deploy?',
        requestedSchema: schema,
        defaultValue: { confirm: false }
      })
    ).toEqual({ status: 'cancelled', data: { confirm: false } })
  })

  it('handles 2026-07-28 envelope MRTR issuance and unsupported capability fallback', async () => {
    const fallbackCtx = makeCtx({
      envelope: {
        [PROTOCOL_VERSION_META_KEY]: '2026-07-28',
        [CLIENT_CAPABILITIES_META_KEY]: {}
      }
    })
    const fallback = await elicitWithFallback(fallbackCtx, {
      key: 'confirm',
      message: 'Deploy?',
      requestedSchema: schema,
      defaultValue: { confirm: false }
    })
    expect(fallback).toEqual({ status: 'fallback', data: { confirm: false }, reason: 'elicitation_unsupported' })

    const modernCtx = makeCtx({
      envelope: {
        [PROTOCOL_VERSION_META_KEY]: '2026-07-28',
        [CLIENT_CAPABILITIES_META_KEY]: { elicitation: {} }
      }
    })
    const issued = await elicitWithFallback(modernCtx, {
      key: 'confirm',
      message: 'Deploy?',
      requestedSchema: schema,
      requestState: 'opaque-state'
    })
    expect(issued.status).toBe('input_required')
    if (issued.status === 'input_required') {
      expect(issued.result.requestState).toBe('opaque-state')
    }
  })

  it('handles legacy elicitInput outcomes and SdkError fallbacks', async () => {
    const acceptCtx = makeCtx({
      elicitInput: vi.fn().mockResolvedValue({ action: 'accept', content: { confirm: true } })
    })
    expect(
      await elicitWithFallback(acceptCtx, { key: 'confirm', message: 'Deploy?', requestedSchema: schema })
    ).toEqual({ status: 'accepted', data: { confirm: true }, source: 'legacy' })

    const emptyAcceptCtx = makeCtx({
      elicitInput: vi.fn().mockResolvedValue({ action: 'accept' })
    })
    expect(
      await elicitWithFallback(emptyAcceptCtx, { key: 'confirm', message: 'Deploy?', requestedSchema: jsonSchema })
    ).toEqual({ status: 'accepted', data: {}, source: 'legacy' })

    const declineCtx = makeCtx({
      elicitInput: vi.fn().mockResolvedValue({ action: 'decline' })
    })
    expect(
      await elicitWithFallback(declineCtx, { key: 'confirm', message: 'Deploy?', requestedSchema: jsonSchema })
    ).toEqual({ status: 'declined', data: undefined })

    const cancelCtx = makeCtx({
      elicitInput: vi.fn().mockResolvedValue({ action: 'cancel' })
    })
    expect(
      await elicitWithFallback(cancelCtx, { key: 'confirm', message: 'Deploy?', requestedSchema: jsonSchema })
    ).toEqual({ status: 'cancelled', data: undefined })

    const methodNotSupportedCtx = makeCtx({
      elicitInput: vi.fn().mockRejectedValue(new SdkError(SdkErrorCode.MethodNotSupportedByProtocolVersion, '2026'))
    })
    const mrtrFallback = await elicitWithFallback(methodNotSupportedCtx, {
      key: 'confirm',
      message: 'Deploy?',
      requestedSchema: schema
    })
    expect(mrtrFallback.status).toBe('input_required')

    const capUnsupportedCtx = makeCtx({
      elicitInput: vi.fn().mockRejectedValue(new SdkError(SdkErrorCode.CapabilityNotSupported, 'no elicit'))
    })
    expect(
      await elicitWithFallback(capUnsupportedCtx, {
        key: 'confirm',
        message: 'Deploy?',
        requestedSchema: schema,
        defaultValue: { confirm: false }
      })
    ).toEqual({ status: 'fallback', data: { confirm: false }, reason: SdkErrorCode.CapabilityNotSupported })

    const rethrowCtx = makeCtx({
      elicitInput: vi.fn().mockRejectedValue(new Error('unexpected failure'))
    })
    await expect(
      elicitWithFallback(rethrowCtx, {
        key: 'confirm',
        message: 'Deploy?',
        requestedSchema: schema,
        defaultValue: { confirm: false }
      })
    ).rejects.toThrow('unexpected failure')
  })
})
