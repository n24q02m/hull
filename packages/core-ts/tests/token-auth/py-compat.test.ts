import { describe, expect, it } from 'vitest'

import { pyJsonDumps, pyPrefix, pyRepr, pyStrip } from '../../src/token-auth/py-compat.js'

// Expected values were produced by CPython 3.13 (`repr`, `str.strip`,
// `json.dumps`) so the TS output is byte-identical to core-py's.
describe('pyRepr', () => {
  it.each([
    ['plain', "'plain'"],
    ["it's", `"it's"`],
    ['say "hi"', `'say "hi"'`],
    [`both ' and "`, `'both \\' and "'`],
    ['back\\slash', "'back\\\\slash'"],
    ['tab\tnl\ncr\r', "'tab\\tnl\\ncr\\r'"],
    ['\u0000\u0007\u001b\u007f', "'\\x00\\x07\\x1b\\x7f'"],
    ['\u0085\u00a0\u00ad', "'\\x85\\xa0\\xad'"],
    ['\u200b\u2028\u2029\u3000', "'\\u200b\\u2028\\u2029\\u3000'"],
    ['\ue000\u{f0000}\u{10fffd}', "'\\ue000\\U000f0000\\U0010fffd'"],
    ['m\u1eadt \u{1f511}', "'m\u1eadt \u{1f511}'"],
    ['\ud800', "'\\ud800'"]
  ])('repr(%j)', (input, expected) => {
    expect(pyRepr(input)).toBe(expected)
  })
})

describe('pyStrip', () => {
  it.each([
    ['  x  ', 'x'],
    ['\u001c\u001dx\u001e\u001f', 'x'],
    ['\u0085x\u3000', 'x'],
    ['\ufeffx\ufeff', '\ufeffx\ufeff'],
    ['\u200bx', '\u200bx'],
    ['\t\n\u000b\f\rx', 'x'],
    ['', ''],
    ['   ', '']
  ])('strip(%j)', (input, expected) => {
    expect(pyStrip(input)).toBe(expected)
  })
})

describe('pyPrefix', () => {
  it('slices by code point, not UTF-16 unit', () => {
    expect(pyPrefix('\u{1f511}\u{1f511}abc', 2)).toBe('\u{1f511}\u{1f511}')
    expect(pyPrefix('ab', 24)).toBe('ab')
  })
})

describe('pyJsonDumps', () => {
  it('matches json.dumps for a plain map', () => {
    expect(pyJsonDumps({ error: 'unauthorized', detail: 'invalid token' })).toBe(
      '{"error": "unauthorized", "detail": "invalid token"}'
    )
  })

  it('escapes non-ASCII and control characters like ensure_ascii=True', () => {
    const detail = `user 'b\u1ec7' is disabled \u{1f511} \u007f \u0001 "q" \\`
    expect(pyJsonDumps({ error: 'forbidden', detail })).toBe(
      `{"error": "forbidden", "detail": "user 'b\\u1ec7' is disabled \\ud83d\\udd11 \\u007f \\u0001 \\"q\\" \\\\"}`
    )
  })
})
