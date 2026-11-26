/**
 * MCP 2026-07-28 + Anthropic Nov 2026 extensions for @n24q02m/hull:
 *  - MCP 2026-07-28 HTTP header constants and CORS preflight/header helper
 *  - `io.modelcontextprotocol/files` payload builder
 *  - `elicitWithFallback` supporting 2026-07-28 MRTR (`inputRequired`/`acceptedContent`),
 *    legacy `2025-11-25` push `elicitInput`, and headless default fallback
 */

import type { IncomingMessage, ServerResponse } from 'node:http'

import {
  acceptedContent,
  CLIENT_CAPABILITIES_META_KEY,
  type ElicitInputParams,
  type ElicitRequestFormParams,
  type InputRequiredResult,
  inputRequired,
  inputResponse,
  PROTOCOL_VERSION_META_KEY,
  SdkError,
  SdkErrorCode,
  type ServerContext,
  type StandardSchemaV1,
  type StandardSchemaWithJSON
} from '@modelcontextprotocol/server'

export const MCP_2026_07_28_HEADERS = ['Mcp-Method', 'Mcp-Name', 'Mcp-Protocol-Version', 'Mcp-Session-Id'] as const

export const MCP_CORS_ALLOW_HEADERS = ['Authorization', 'Content-Type', 'Accept', ...MCP_2026_07_28_HEADERS] as const

export const MCP_CORS_EXPOSE_HEADERS = ['Mcp-Session-Id', 'Mcp-Protocol-Version'] as const

export const FILES_EXTENSION_ID = 'io.modelcontextprotocol/files'

export interface FilePayload {
  extension: typeof FILES_EXTENSION_ID
  uri: string
  name: string
  mimeType: string
  sizeBytes: number
  text?: string
  blob?: string
}

export interface BuildFilePayloadOptions {
  mimeType?: string
  uri?: string
}

/**
 * Apply MCP 2026-07-28 CORS headers when `allowOrigins` is non-empty and
 * handle `OPTIONS` preflight requests directly (`true` = response sent).
 */
export function applyMcpCorsHeaders(
  req: IncomingMessage,
  res: ServerResponse,
  allowOrigins: readonly string[]
): boolean {
  if (allowOrigins.length === 0) {
    return false
  }
  const rawOrigin = req.headers.origin
  const origin = Array.isArray(rawOrigin) ? rawOrigin[0] : rawOrigin
  const allowAll = allowOrigins.includes('*')

  if (allowAll) {
    res.setHeader('Access-Control-Allow-Origin', '*')
  } else if (origin && allowOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin)
    res.setHeader('Vary', 'Origin')
  }

  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', MCP_CORS_ALLOW_HEADERS.join(', '))
  res.setHeader('Access-Control-Expose-Headers', MCP_CORS_EXPOSE_HEADERS.join(', '))

  if (req.method === 'OPTIONS') {
    res.writeHead(204)
    res.end()
    return true
  }
  return false
}

/**
 * Build an `io.modelcontextprotocol/files` payload from text or binary bytes.
 */
export function buildFilePayload(
  data: string | Uint8Array,
  name: string,
  options?: BuildFilePayloadOptions
): FilePayload {
  const cleanName = name.trim()
  if (!cleanName) {
    throw new Error('File name must be non-empty')
  }
  const uri = options?.uri ?? `file:///${encodeURIComponent(cleanName)}`

  if (typeof data === 'string') {
    return {
      extension: FILES_EXTENSION_ID,
      uri,
      name: cleanName,
      mimeType: options?.mimeType ?? 'text/plain; charset=utf-8',
      sizeBytes: Buffer.byteLength(data, 'utf8'),
      text: data
    }
  }

  if (data instanceof Uint8Array) {
    return {
      extension: FILES_EXTENSION_ID,
      uri,
      name: cleanName,
      mimeType: options?.mimeType ?? 'application/octet-stream',
      sizeBytes: data.byteLength,
      blob: Buffer.from(data).toString('base64')
    }
  }

  throw new TypeError('File payload data must be a string or Uint8Array')
}

export type ElicitationOutcome<T extends Record<string, unknown>> =
  | { status: 'accepted'; data: T; source: 'mrtr' | 'legacy' }
  | { status: 'declined'; data: T | undefined }
  | { status: 'cancelled'; data: T | undefined }
  | { status: 'fallback'; data: T; reason: string }
  | { status: 'input_required'; result: InputRequiredResult }

export interface ElicitWithFallbackOptions<T extends Record<string, unknown>> {
  key: string
  message: string
  requestedSchema: ElicitInputParams['requestedSchema']
  schema?: StandardSchemaV1<unknown, T>
  defaultValue?: T
  requestState?: string
  preferMultiRoundTrip?: boolean
}

function isStandardJsonSchema(schema: ElicitInputParams['requestedSchema']): schema is StandardSchemaWithJSON {
  const std = '~standard' in schema ? (schema as { '~standard': unknown })['~standard'] : undefined
  return typeof std === 'object' && std !== null && 'jsonSchema' in std
}

function toLegacyFormSchema(schema: ElicitInputParams['requestedSchema']): ElicitRequestFormParams['requestedSchema'] {
  if (isStandardJsonSchema(schema)) {
    return schema['~standard'].jsonSchema.input({
      target: 'draft-07'
    }) as ElicitRequestFormParams['requestedSchema']
  }
  return schema
}

function makeInputRequiredResult<T extends Record<string, unknown>>(
  options: ElicitWithFallbackOptions<T>
): InputRequiredResult {
  return inputRequired({
    inputRequests: {
      [options.key]: inputRequired.elicit({
        message: options.message,
        requestedSchema: options.requestedSchema
      })
    },
    ...(options.requestState !== undefined ? { requestState: options.requestState } : {})
  })
}

function hasModernProtocolEnvelope(ctx: ServerContext): boolean {
  const envelope = ctx.mcpReq.envelope as Record<string, unknown> | undefined
  const version = envelope?.[PROTOCOL_VERSION_META_KEY]
  return typeof version === 'string' && version >= '2026-07-28'
}

function clientDeclaresNoElicitation(ctx: ServerContext): boolean {
  const envelope = ctx.mcpReq.envelope as Record<string, unknown> | undefined
  const caps = envelope?.[CLIENT_CAPABILITIES_META_KEY]
  return Boolean(caps && typeof caps === 'object' && !('elicitation' in caps && caps.elicitation))
}

/**
 * Request structured user input across MCP 2026-07-28 (multi-round-trip
 * `inputRequired` / `acceptedContent`), legacy 2025 push (`elicitInput`),
 * and headless/unsupported clients (`defaultValue` fallback).
 */
export async function elicitWithFallback<T extends Record<string, unknown>>(
  ctx: ServerContext,
  options: ElicitWithFallbackOptions<T>
): Promise<ElicitationOutcome<T>> {
  const view = inputResponse(ctx.mcpReq.inputResponses, options.key)
  if (view.kind === 'elicit') {
    if (view.action === 'decline') {
      return { status: 'declined', data: options.defaultValue }
    }
    if (view.action === 'cancel') {
      return { status: 'cancelled', data: options.defaultValue }
    }
    if (options.schema) {
      const validated = acceptedContent(ctx.mcpReq.inputResponses, options.key, options.schema)
      if (validated !== undefined) {
        return { status: 'accepted', data: validated, source: 'mrtr' }
      }
      return { status: 'input_required', result: makeInputRequiredResult(options) }
    }
    return { status: 'accepted', data: (view.content ?? {}) as T, source: 'mrtr' }
  }

  if (options.preferMultiRoundTrip || hasModernProtocolEnvelope(ctx)) {
    if (clientDeclaresNoElicitation(ctx) && options.defaultValue !== undefined) {
      return { status: 'fallback', data: options.defaultValue, reason: 'elicitation_unsupported' }
    }
    return { status: 'input_required', result: makeInputRequiredResult(options) }
  }

  try {
    const result = await ctx.mcpReq.elicitInput({
      mode: 'form',
      message: options.message,
      requestedSchema: toLegacyFormSchema(options.requestedSchema)
    })
    if (result.action === 'accept') {
      return { status: 'accepted', data: (result.content ?? {}) as T, source: 'legacy' }
    }
    if (result.action === 'decline') {
      return { status: 'declined', data: options.defaultValue }
    }
    return { status: 'cancelled', data: options.defaultValue }
  } catch (err) {
    if (err instanceof SdkError && err.code === SdkErrorCode.MethodNotSupportedByProtocolVersion) {
      return { status: 'input_required', result: makeInputRequiredResult(options) }
    }
    if (
      options.defaultValue !== undefined &&
      err instanceof SdkError &&
      err.code === SdkErrorCode.CapabilityNotSupported
    ) {
      return { status: 'fallback', data: options.defaultValue, reason: err.code }
    }
    throw err
  }
}
