import { randomBytes } from 'node:crypto'
import { open, rename, unlink } from 'node:fs/promises'
import { resolve } from 'node:path'

const pendingWrites = new Map<string, Promise<void>>()

async function writeOnce(path: string, data: Buffer): Promise<void> {
  const tmp = `${path}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`
  try {
    const handle = await open(tmp, 'w', 0o600)
    try {
      await handle.writeFile(data)
      await handle.sync()
    } finally {
      await handle.close()
    }
    await rename(tmp, path)
  } catch (err) {
    await unlink(tmp).catch(() => {})
    throw err
  }
}

/**
 * Write data atomically: unique tmp file + fsync + rename, serialized per path.
 * A torn credential write decrypts to garbage and reads back as "not configured";
 * a unique per-write temp path plus per-path serialization prevents concurrent
 * refreshes from racing on the same `.tmp` file.
 */
export async function atomicWriteFile(path: string, data: Buffer): Promise<void> {
  const key = resolve(path)
  const prev = pendingWrites.get(key) ?? Promise.resolve()
  const current = prev.then(() => writeOnce(path, data))
  const tail = current.then(
    () => undefined,
    () => undefined
  )
  pendingWrites.set(key, tail)
  try {
    await current
  } finally {
    if (pendingWrites.get(key) === tail) {
      pendingWrites.delete(key)
    }
  }
}
