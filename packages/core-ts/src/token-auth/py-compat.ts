/**
 * Small helpers that reproduce Python string semantics the core-py auth stack
 * leaks into its wire output and error messages, so the TS port emits the
 * same bytes: `repr()` of a str, `str.strip()`, code-point slicing, and
 * `json.dumps()` of a flat string map.
 *
 * Internal to `token-auth`; not part of the public API.
 */

// Every code point for which Python's `str.isspace()` is true. JS `trim()`
// differs: it strips U+FEFF but not U+001C..U+001F or U+0085.
const PY_WHITESPACE: ReadonlySet<number> = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
  0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000
])

function isPySpace(ch: string): boolean {
  return PY_WHITESPACE.has(ch.codePointAt(0) as number)
}

/** Python `str.strip()` with no argument. */
export function pyStrip(value: string): string {
  const chars = Array.from(value)
  let start = 0
  let end = chars.length
  while (start < end && isPySpace(chars[start])) start++
  while (end > start && isPySpace(chars[end - 1])) end--
  return chars.slice(start, end).join('')
}

/** Python `s[:n]` (code points, not UTF-16 units). */
export function pyPrefix(value: string, n: number): string {
  return Array.from(value).slice(0, n).join('')
}

// Python's `str.isprintable()` is false for categories Cc, Cf, Cs, Co, Cn,
// Zl, Zp and Zs, except the ASCII space.
const NON_PRINTABLE = /^[\p{C}\p{Z}]$/u

/**
 * Python `repr()` of a str: single quotes unless the text contains a single
 * quote and no double quote; backslash, the chosen quote, and non-printable
 * characters escaped the way CPython escapes them.
 */
export function pyRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'"
  let out = quote
  for (const ch of value) {
    const code = ch.codePointAt(0) as number
    if (ch === quote || ch === '\\') out += `\\${ch}`
    else if (ch === '\t') out += '\\t'
    else if (ch === '\n') out += '\\n'
    else if (ch === '\r') out += '\\r'
    else if (ch !== ' ' && NON_PRINTABLE.test(ch)) {
      if (code < 0x100) out += `\\x${code.toString(16).padStart(2, '0')}`
      else if (code < 0x10000) out += `\\u${code.toString(16).padStart(4, '0')}`
      else out += `\\U${code.toString(16).padStart(8, '0')}`
    } else out += ch
  }
  return out + quote
}

/**
 * Python `json.dumps(obj)` for a flat `str -> str` map with default settings
 * (`ensure_ascii=True`, separators `', '` and `': '`).
 */
export function pyJsonDumps(obj: Readonly<Record<string, string>>): string {
  const encode = (s: string): string =>
    JSON.stringify(s).replace(/[^\x20-\x7e]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`)
  const items = Object.entries(obj).map(([k, v]) => `${encode(k)}: ${encode(v)}`)
  return `{${items.join(', ')}}`
}
