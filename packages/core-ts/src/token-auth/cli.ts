/**
 * `token hash` CLI subcommand: mint a `scrypt$...` hash for `token` mode
 * (`tokenHash`) or a users.toml row (`token_hash`).
 *
 * Unlike core-py `hull token hash <token>`, the plaintext is never taken
 * from argv (it would land in shell history and the process table). It is
 * read from piped stdin, or from a hidden, confirmed TTY prompt. Only the
 * hash is printed to stdout; prompts and errors go to stderr.
 */

import { pyStrip } from './py-compat.js'
import { hashToken } from './tokens.js'

/** Same shape as `CliHandler` in `cli/build-cli.ts` (kept local to avoid a dependency). */
export type TokenCliHandler = (argv: string[]) => number | Promise<number>

export interface TokenCliInput {
  isTTY?: boolean
  setRawMode?(mode: boolean): unknown
  setEncoding?(encoding: BufferEncoding): unknown
  on(event: 'data', listener: (chunk: string | Buffer) => void): unknown
  on(event: 'end' | 'close', listener: () => void): unknown
  on(event: 'error', listener: (err: Error) => void): unknown
  removeListener(event: 'data', listener: (chunk: string | Buffer) => void): unknown
  removeListener(event: 'end' | 'close', listener: () => void): unknown
  removeListener(event: 'error', listener: (err: Error) => void): unknown
  resume?(): unknown
  pause?(): unknown
}

export interface TokenCliOutput {
  write(chunk: string): unknown
}

export interface TokenCliIO {
  stdin?: TokenCliInput
  stdout?: TokenCliOutput
  stderr?: TokenCliOutput
}

const USAGE = 'usage: token hash  (reads the token from stdin or a hidden prompt; never from argv)'

class InputAborted extends Error {}

function readAll(stdin: TokenCliInput): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    const onData = (chunk: string | Buffer): void => {
      chunks.push(typeof chunk === 'string' ? Buffer.from(chunk, 'utf8') : chunk)
    }
    const cleanup = (): void => {
      stdin.removeListener('data', onData)
      stdin.removeListener('end', onEnd)
      stdin.removeListener('error', onError)
    }
    const onEnd = (): void => {
      cleanup()
      try {
        resolve(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks)))
      } catch (err) {
        reject(err)
      }
    }
    const onError = (err: Error): void => {
      cleanup()
      reject(err)
    }
    stdin.on('data', onData)
    stdin.on('end', onEnd)
    stdin.on('error', onError)
    stdin.resume?.()
  })
}

/** Reads hidden lines from a raw-mode TTY; keeps type-ahead between prompts. */
class HiddenReader {
  private pending: string[] = []

  constructor(
    private readonly stdin: TokenCliInput,
    private readonly stderr: TokenCliOutput
  ) {}

  async read(prompt: string): Promise<string> {
    this.stderr.write(prompt)
    const typed: string[] = []
    const consume = (chars: string[]): 'done' | 'more' => {
      while (chars.length > 0) {
        const ch = chars.shift() as string
        if (ch === '\r' || ch === '\n') {
          this.pending = chars
          return 'done'
        }
        if (ch === '\u0003') throw new InputAborted('interrupted')
        if (ch === '\u0004') {
          if (typed.length === 0) throw new InputAborted('end of input')
          this.pending = chars
          return 'done'
        }
        if (ch === '\u007f' || ch === '\b') typed.pop()
        else typed.push(ch)
      }
      return 'more'
    }
    try {
      if (consume(this.pending) === 'done') return typed.join('')
      return await new Promise<string>((resolve, reject) => {
        const cleanup = (): void => {
          this.stdin.removeListener('data', onData)
          this.stdin.removeListener('end', onEnd)
          this.stdin.removeListener('error', onError)
        }
        const onData = (chunk: string | Buffer): void => {
          try {
            if (consume(Array.from(chunk.toString())) === 'done') {
              cleanup()
              resolve(typed.join(''))
            }
          } catch (err) {
            cleanup()
            reject(err)
          }
        }
        const onEnd = (): void => {
          cleanup()
          reject(new InputAborted('end of input'))
        }
        const onError = (err: Error): void => {
          cleanup()
          reject(err)
        }
        this.stdin.on('data', onData)
        this.stdin.on('end', onEnd)
        this.stdin.on('error', onError)
        this.stdin.resume?.()
      })
    } finally {
      this.stderr.write('\n')
    }
  }
}

async function promptToken(stdin: TokenCliInput, stderr: TokenCliOutput): Promise<string | null> {
  stdin.setEncoding?.('utf8')
  stdin.setRawMode?.(true)
  try {
    const reader = new HiddenReader(stdin, stderr)
    const first = await reader.read('Token: ')
    const second = await reader.read('Repeat token: ')
    if (first !== second) {
      stderr.write('token hash: tokens do not match\n')
      return null
    }
    return first
  } finally {
    stdin.setRawMode?.(false)
    stdin.pause?.()
  }
}

function fromPipe(text: string, stderr: TokenCliOutput): string | null {
  let token = text
  if (token.endsWith('\r\n')) token = token.slice(0, -2)
  else if (token.endsWith('\n')) token = token.slice(0, -1)
  if (/[\r\n]/.test(token)) {
    stderr.write('token hash: expected a single line on stdin\n')
    return null
  }
  return token
}

function checkToken(token: string, stderr: TokenCliOutput): boolean {
  if (token.length === 0) {
    stderr.write('token hash: empty token\n')
    return false
  }
  // The guard strips the bearer value, so a padded token could never match.
  if (pyStrip(token) !== token) {
    stderr.write('token hash: token must not start or end with whitespace\n')
    return false
  }
  return true
}

/**
 * Build the `token` subcommand handler (`token hash`). Built into
 * `buildCli` under the name `token`; `io` is injectable for tests.
 */
export function tokenCliHandler(io: TokenCliIO = {}): TokenCliHandler {
  return async (argv) => {
    const stdin = io.stdin ?? (process.stdin as unknown as TokenCliInput)
    const stdout = io.stdout ?? process.stdout
    const stderr = io.stderr ?? process.stderr
    if (argv[0] !== 'hash' || argv.length > 1) {
      if (argv[0] === 'hash') stderr.write('token hash: refusing a token on argv (it would be logged)\n')
      stderr.write(`${USAGE}\n`)
      return 2
    }
    let token: string | null
    try {
      token = stdin.isTTY ? await promptToken(stdin, stderr) : fromPipe(await readAll(stdin), stderr)
    } catch (err) {
      if (err instanceof InputAborted) {
        stderr.write(`token hash: aborted (${err.message})\n`)
        return 130
      }
      stderr.write(`token hash: could not read token: ${(err as Error).message}\n`)
      return 1
    }
    if (token === null || !checkToken(token, stderr)) return 1
    stdout.write(`${await hashToken(token)}\n`)
    return 0
  }
}
