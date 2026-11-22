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

  it('serializes concurrent writes to the same path without temp-file collisions', async () => {
    const path = join(dir, 'blob.enc')

    await Promise.all(Array.from({ length: 8 }, (_, i) => atomicWriteFile(path, Buffer.from(`v-${i}`))))

    expect(readFileSync(path, 'utf-8')).toBe('v-7')
    expect(readdirSync(dir)).toEqual(['blob.enc'])
  })

  it('keeps the write queue usable after an earlier failure for the same path', async () => {
    const badPath = join(dir, 'missing', 'blob.enc')
    await expect(atomicWriteFile(badPath, Buffer.from('bad'))).rejects.toMatchObject({ code: 'ENOENT' })

    const path = join(dir, 'blob.enc')
    await atomicWriteFile(path, Buffer.from('recovered'))
    expect(readFileSync(path, 'utf-8')).toBe('recovered')
  })
})
