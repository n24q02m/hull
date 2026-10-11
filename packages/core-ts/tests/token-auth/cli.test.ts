import { EventEmitter } from 'node:events'
import { describe, expect, it } from 'vitest'

import { tokenCliHandler } from '../../src/token-auth/cli.js'
import { verifyToken } from '../../src/token-auth/tokens.js'

const HASH = /^scrypt\$32768\$8\$1\$[0-9a-f]{32}\$[0-9a-f]{64}\n$/

class Sink {
  text = ''
  write(chunk: string): boolean {
    this.text += chunk
    return true
  }
}

class FakeStdin extends EventEmitter {
  rawModes: boolean[] = []
  encoding: string | undefined
  paused = false
  resumed = 0

  constructor(readonly isTTY: boolean) {
    super()
  }

  setRawMode(mode: boolean): this {
    this.rawModes.push(mode)
    return this
  }

  setEncoding(encoding: string): this {
    this.encoding = encoding
    return this
  }

  resume(): this {
    this.resumed += 1
    return this
  }

  pause(): this {
    this.paused = true
    return this
  }
}

/** Run the handler and feed `chunks` (or an error/end) once it is listening. */
async function run(
  argv: string[],
  options: { tty?: boolean; chunks?: (string | Buffer)[]; end?: boolean; error?: Error } = {}
) {
  const stdin = new FakeStdin(options.tty ?? false)
  const stdout = new Sink()
  const stderr = new Sink()
  const pending = Promise.resolve(tokenCliHandler({ stdin, stdout, stderr })(argv))
  for (const chunk of options.chunks ?? []) {
    await new Promise((resolve) => setImmediate(resolve))
    stdin.emit('data', chunk)
  }
  await new Promise((resolve) => setImmediate(resolve))
  if (options.error) stdin.emit('error', options.error)
  else if (options.end ?? !options.tty) stdin.emit('end')
  const rc = await pending
  return { rc, stdout: stdout.text, stderr: stderr.text, stdin }
}

describe('token hash argv handling', () => {
  it.each([[[]], [['verify']], [['--help']]])('prints usage and returns 2 for %j', async (argv) => {
    const out = await run(argv)
    expect(out.rc).toBe(2)
    expect(out.stdout).toBe('')
    expect(out.stderr).toBe('usage: token hash  (reads the token from stdin or a hidden prompt; never from argv)\n')
  })

  it('refuses a token passed on argv', async () => {
    const out = await run(['hash', 'plaintext-token'])
    expect(out.rc).toBe(2)
    expect(out.stdout).toBe('')
    expect(out.stderr).toBe(
      'token hash: refusing a token on argv (it would be logged)\n' +
        'usage: token hash  (reads the token from stdin or a hidden prompt; never from argv)\n'
    )
  })
})

describe('token hash from a pipe', () => {
  it('hashes the single line and prints only the hash', async () => {
    const out = await run(['hash'], { chunks: ['piped-', Buffer.from('token\n')] })
    expect(out.rc).toBe(0)
    expect(out.stderr).toBe('')
    expect(out.stdout).toMatch(HASH)
    expect(await verifyToken('piped-token', out.stdout.trim())).toBe(true)
    expect(out.stdin.resumed).toBe(1)
  })

  it('strips one CRLF and accepts no trailing newline', async () => {
    const crlf = await run(['hash'], { chunks: ['crlf-token\r\n'] })
    expect(crlf.rc).toBe(0)
    expect(await verifyToken('crlf-token', crlf.stdout.trim())).toBe(true)
    const bare = await run(['hash'], { chunks: ['bare-token'] })
    expect(await verifyToken('bare-token', bare.stdout.trim())).toBe(true)
  })

  it('keeps non-ASCII tokens byte-exact', async () => {
    const token = 'm\u1eadt-kh\u1ea9u-\u{1f511}'
    const out = await run(['hash'], { chunks: [Buffer.from(`${token}\n`, 'utf8')] })
    expect(out.rc).toBe(0)
    expect(await verifyToken(token, out.stdout.trim())).toBe(true)
  })

  it.each([
    [['a\nb\n'], 'token hash: expected a single line on stdin\n'],
    [['a\rb'], 'token hash: expected a single line on stdin\n'],
    [['a\n\n'], 'token hash: expected a single line on stdin\n'],
    [['\n'], 'token hash: empty token\n'],
    [[], 'token hash: empty token\n'],
    [[' padded\n'], 'token hash: token must not start or end with whitespace\n'],
    [['padded\t\n'], 'token hash: token must not start or end with whitespace\n']
  ])('rejects %j', async (chunks, message) => {
    const out = await run(['hash'], { chunks })
    expect(out.rc).toBe(1)
    expect(out.stdout).toBe('')
    expect(out.stderr).toBe(message)
  })

  it('rejects invalid UTF-8', async () => {
    const out = await run(['hash'], { chunks: [Buffer.from([0x61, 0xff, 0x0a])] })
    expect(out.rc).toBe(1)
    expect(out.stdout).toBe('')
    expect(out.stderr).toMatch(/^token hash: could not read token: /)
  })

  it('reports a stream error', async () => {
    const out = await run(['hash'], { chunks: ['partial'], error: new Error('EIO') })
    expect(out.rc).toBe(1)
    expect(out.stderr).toBe('token hash: could not read token: EIO\n')
  })
})

describe('token hash from a TTY prompt', () => {
  it('prompts twice on stderr, hides input and hashes a confirmed token', async () => {
    const out = await run(['hash'], { tty: true, chunks: ['tty-', 'token\r', 'tty-token\r'] })
    expect(out.rc).toBe(0)
    expect(out.stderr).toBe('Token: \nRepeat token: \n')
    expect(out.stdout).toMatch(HASH)
    expect(await verifyToken('tty-token', out.stdout.trim())).toBe(true)
    expect(out.stdin.rawModes).toEqual([true, false])
    expect(out.stdin.encoding).toBe('utf8')
    expect(out.stdin.paused).toBe(true)
  })

  it('keeps type-ahead for the second prompt and honours backspace', async () => {
    const out = await run(['hash'], { tty: true, chunks: ['abx\u007fc\nab\bbc\n'] })
    expect(out.rc).toBe(0)
    expect(await verifyToken('abc', out.stdout.trim())).toBe(true)
  })

  it('treats Ctrl-D after text as the end of the line', async () => {
    const out = await run(['hash'], { tty: true, chunks: ['tok\u0004', 'tok\u0004'] })
    expect(out.rc).toBe(0)
    expect(await verifyToken('tok', out.stdout.trim())).toBe(true)
  })

  it('fails on a mismatch', async () => {
    const out = await run(['hash'], { tty: true, chunks: ['one\r', 'two\r'] })
    expect(out.rc).toBe(1)
    expect(out.stdout).toBe('')
    expect(out.stderr).toBe('Token: \nRepeat token: \ntoken hash: tokens do not match\n')
    expect(out.stdin.rawModes).toEqual([true, false])
  })

  it.each([
    [['abc\u0003'], 'token hash: aborted (interrupted)\n'],
    [['\u0004'], 'token hash: aborted (end of input)\n'],
    [['abc\r\u0003'], 'token hash: aborted (interrupted)\n']
  ])('aborts with 130 on %j', async (chunks, message) => {
    const out = await run(['hash'], { tty: true, chunks })
    expect(out.rc).toBe(130)
    expect(out.stdout).toBe('')
    expect(out.stderr.endsWith(message)).toBe(true)
    expect(out.stdin.rawModes).toEqual([true, false])
  })

  it('aborts with 130 when the TTY ends mid-prompt', async () => {
    const out = await run(['hash'], { tty: true, chunks: ['abc'], end: true })
    expect(out.rc).toBe(130)
    expect(out.stderr).toBe('Token: \ntoken hash: aborted (end of input)\n')
  })

  it('reports a TTY read error', async () => {
    const out = await run(['hash'], { tty: true, error: new Error('EBADF') })
    expect(out.rc).toBe(1)
    expect(out.stderr).toBe('Token: \ntoken hash: could not read token: EBADF\n')
  })

  it('applies the same empty and whitespace checks', async () => {
    const empty = await run(['hash'], { tty: true, chunks: ['\r\r'] })
    expect(empty.rc).toBe(1)
    expect(empty.stderr.endsWith('token hash: empty token\n')).toBe(true)
    const padded = await run(['hash'], { tty: true, chunks: ['x \r', 'x \r'] })
    expect(padded.rc).toBe(1)
    expect(padded.stderr.endsWith('token hash: token must not start or end with whitespace\n')).toBe(true)
  })
})

describe('token hash defaults', () => {
  it('builds a handler bound to the process streams without touching them', () => {
    expect(typeof tokenCliHandler()).toBe('function')
  })
})
