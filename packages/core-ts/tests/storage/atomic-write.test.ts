import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { atomicWriteFile } from '../../src/storage/atomic-write.js'

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'mcp-atomic-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

describe('atomicWriteFile', () => {
  it('replaces the file and leaves no temp file behind', async () => {
    const path = join(dir, 'blob.enc')
    writeFileSync(path, 'old')

    await atomicWriteFile(path, Buffer.from('new'))

    expect(readFileSync(path, 'utf-8')).toBe('new')
    expect(readdirSync(dir)).toEqual(['blob.enc'])
  })

  it('reports the open failure when the temp file cannot be created', async () => {
    // No parent directory: opening the temp file fails, so there is no temp
    // file to clean up and that cleanup failure must not hide the real error.
    const path = join(dir, 'missing-parent', 'blob.enc')

    await expect(atomicWriteFile(path, Buffer.from('data'))).rejects.toMatchObject({ code: 'ENOENT' })
    expect(existsSync(join(dir, 'missing-parent'))).toBe(false)
  })
})
