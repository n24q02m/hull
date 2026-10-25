import { describe, expect, it } from 'vitest'

import { assertNoAuthLoopback, inetAton, isLoopbackHost, ServerConfigError } from '../../src/token-auth/loopback.js'

describe('inetAton', () => {
  // Values agree with CPython socket.inet_aton on both glibc and Windows.
  it.each([
    ['127.0.0.1', 0x7f000001],
    ['127.1', 0x7f000001],
    ['0x7f.1', 0x7f000001],
    ['0X7F.0.0.1', 0x7f000001],
    ['0x7f.0x0.0.0x1', 0x7f000001],
    ['017700000001', 0x7f000001],
    ['0177.0.0.1', 0x7f000001],
    ['2130706433', 0x7f000001],
    ['0x7f000001', 0x7f000001],
    ['127.255.255.255', 0x7fffffff],
    ['128.0.0.1', 0x80000001],
    ['1.2.3', 0x01020003],
    ['0.0.0.0', 0],
    ['0', 0],
    ['00', 0]
  ])('parses %s', (host, expected) => {
    expect(inetAton(host)).toBe(expected)
  })

  it.each(['127.0.0.256', '127.256.1', '', '1.2.3.4.5', 'a.b', '08', '127..1', '0x', '4294967296', '127.0.0.1 ', '-1'])(
    'rejects %j',
    (host) => {
      expect(inetAton(host)).toBeNull()
    }
  )
})

describe('isLoopbackHost', () => {
  it.each(['localhost', '127.0.0.1', '::1', '127.5.6.7', '127.1', '0x7f.1'])('%s is loopback', (host) => {
    expect(isLoopbackHost(host)).toBe(true)
  })

  it.each(['0.0.0.0', '::', '128.0.0.1', 'LOCALHOST', 'example.com', '10.0.0.1', '::ffff:127.0.0.1'])(
    '%s is not loopback',
    (host) => {
      expect(isLoopbackHost(host)).toBe(false)
    }
  )
})

describe('assertNoAuthLoopback', () => {
  it('accepts loopback hosts', () => {
    expect(() => assertNoAuthLoopback('127.0.0.1')).not.toThrow()
  })

  it("throws core-py's ServerConfigError message for a shared bind", () => {
    expect(() => assertNoAuthLoopback('0.0.0.0')).toThrow(ServerConfigError)
    expect(() => assertNoAuthLoopback('0.0.0.0')).toThrow(
      "auth = 'no-auth' only permits loopback binds, refusing host '0.0.0.0' " +
        "(set [server] auth to 'token' or 'multi' for a shared listener)"
    )
  })
})
