/**
 * Loopback detection for the `no-auth` bind guard, mirroring core-py
 * `hull_core.server.run._is_loopback`: the literal names `localhost`,
 * `127.0.0.1`, `::1`, or any IPv4 address in 127.0.0.0/8 written in the
 * classic `inet_aton` forms (`127.1`, `0x7f.1`, `017700000001`, ...).
 */

import { pyRepr } from './py-compat.js'

const LOOPBACK_NAMES: ReadonlySet<string> = new Set(['localhost', '127.0.0.1', '::1'])

/** Invalid server start configuration (core-py `ServerConfigError`). */
export class ServerConfigError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ServerConfigError'
  }
}

function parsePart(part: string): number | null {
  if (/^0[xX][0-9a-fA-F]+$/.test(part)) return Number.parseInt(part.slice(2), 16)
  if (/^0[0-7]*$/.test(part)) return part.length === 1 ? 0 : Number.parseInt(part.slice(1), 8)
  if (/^[1-9][0-9]*$/.test(part)) return Number.parseInt(part, 10)
  return null
}

/** Strict BSD `inet_aton`: returns the 32-bit address, or `null` when invalid. */
export function inetAton(host: string): number | null {
  const parts = host.split('.')
  if (parts.length < 1 || parts.length > 4) return null
  const values: number[] = []
  for (const part of parts) {
    const value = parsePart(part)
    if (value === null) return null
    values.push(value)
  }
  const last = values.length - 1
  for (let i = 0; i < last; i++) if (values[i] > 0xff) return null
  if (values[last] >= 2 ** (8 * (4 - last))) return null
  let address = 0
  for (let i = 0; i < last; i++) address += values[i] * 2 ** (8 * (3 - i))
  return address + values[last]
}

/** True when `host` is a loopback bind address under core-py's rules. */
export function isLoopbackHost(host: string): boolean {
  if (LOOPBACK_NAMES.has(host)) return true
  const address = inetAton(host)
  return address !== null && Math.floor(address / 2 ** 24) === 0x7f
}

/** Throw core-py's `ServerConfigError` when `no-auth` would bind beyond loopback. */
export function assertNoAuthLoopback(host: string): void {
  if (isLoopbackHost(host)) return
  throw new ServerConfigError(
    `auth = 'no-auth' only permits loopback binds, refusing host ${pyRepr(host)} ` +
      "(set [server] auth to 'token' or 'multi' for a shared listener)"
  )
}
