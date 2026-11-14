/**
 * Every machine-id source, whichever OS runs the suite.
 *
 * `machine-id.test.ts` exercises the real host, so it only ever reaches the
 * branch of the platform it runs on. Here the OS sources are stubbed and
 * `process.platform` is set per test, so the Linux, macOS, Windows and
 * hostname+MAC fallback paths are all checked on every runner.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const os = vi.hoisted(() => ({
  readFile: vi.fn(),
  execFile: vi.fn(),
  hostname: vi.fn(),
  networkInterfaces: vi.fn(),
  userInfo: vi.fn()
}))

vi.mock('node:fs/promises', async (importOriginal) => ({
  ...(await importOriginal<typeof import('node:fs/promises')>()),
  readFile: os.readFile
}))

vi.mock('node:child_process', async (importOriginal) => {
  const { promisify } = await import('node:util')
  // `promisify(execFile)` uses the `promisify.custom` form, which resolves `{ stdout, stderr }`.
  const execFile = Object.assign(
    () => {
      throw new Error('machine-id only uses the promisified execFile')
    },
    { [promisify.custom]: os.execFile }
  )
  return { ...(await importOriginal<typeof import('node:child_process')>()), execFile }
})

vi.mock('node:os', async (importOriginal) => ({
  ...(await importOriginal<typeof import('node:os')>()),
  hostname: os.hostname,
  networkInterfaces: os.networkInterfaces,
  userInfo: os.userInfo
}))

const { clearMachineIdCacheForTesting, getMachineId, getUsername } = await import('../../src/storage/machine-id.js')

const realPlatform = Object.getOwnPropertyDescriptor(process, 'platform') as PropertyDescriptor

function onPlatform(platform: NodeJS.Platform): void {
  Object.defineProperty(process, 'platform', { value: platform, configurable: true })
}

function nic(mac: string, internal = false): { mac: string; internal: boolean } {
  return { mac, internal }
}

beforeEach(() => {
  clearMachineIdCacheForTesting()
  for (const source of Object.values(os)) source.mockReset()
  os.hostname.mockReturnValue('build-host')
  os.networkInterfaces.mockReturnValue({})
})

afterEach(() => {
  Object.defineProperty(process, 'platform', realPlatform)
  vi.unstubAllEnvs()
})

describe('getMachineId on Linux', () => {
  it('reads /etc/machine-id without the trailing newline', async () => {
    onPlatform('linux')
    os.readFile.mockResolvedValue('4c4c4544004d3610\n')

    expect(await getMachineId()).toBe('4c4c4544004d3610')
    expect(os.readFile).toHaveBeenCalledWith('/etc/machine-id', 'utf-8')
  })

  it('reads the source once and serves the cached id afterwards', async () => {
    onPlatform('linux')
    os.readFile.mockResolvedValue('first-id\n')

    expect(await getMachineId()).toBe('first-id')
    os.readFile.mockResolvedValue('second-id\n')
    expect(await getMachineId()).toBe('first-id')
    expect(os.readFile).toHaveBeenCalledTimes(1)
  })

  it('falls back to hostname and MAC when the file cannot be read', async () => {
    onPlatform('linux')
    os.readFile.mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    os.networkInterfaces.mockReturnValue({ eth0: [nic('aa:bb:cc:dd:ee:ff')] })

    expect(await getMachineId()).toBe('build-host-aa:bb:cc:dd:ee:ff')
  })
})

describe('getMachineId on macOS', () => {
  it('takes IOPlatformUUID from ioreg', async () => {
    onPlatform('darwin')
    os.execFile.mockResolvedValue({
      stdout: '  "IOPlatformSerialNumber" = "C02XYZ"\n  "IOPlatformUUID" = "564D1E2A-8F0B-4C1D-9E3F-1A2B3C4D5E6F"\n',
      stderr: ''
    })

    expect(await getMachineId()).toBe('564D1E2A-8F0B-4C1D-9E3F-1A2B3C4D5E6F')
    expect(os.execFile).toHaveBeenCalledWith('ioreg', ['-rd1', '-c', 'IOPlatformExpertDevice'])
  })

  it('falls back when ioreg prints no UUID', async () => {
    onPlatform('darwin')
    os.execFile.mockResolvedValue({ stdout: '+-o Root  <class IORegistryEntry>\n', stderr: '' })
    os.networkInterfaces.mockReturnValue({ en0: [nic('11:22:33:44:55:66')] })

    expect(await getMachineId()).toBe('build-host-11:22:33:44:55:66')
  })
})

describe('getMachineId on Windows', () => {
  it('takes MachineGuid from the registry', async () => {
    onPlatform('win32')
    os.execFile.mockResolvedValue({
      stdout:
        '\r\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\r\n    MachineGuid    REG_SZ    0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0\r\n',
      stderr: ''
    })

    expect(await getMachineId()).toBe('0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0')
    expect(os.execFile).toHaveBeenCalledWith('reg', [
      'query',
      'HKLM\\SOFTWARE\\Microsoft\\Cryptography',
      '/v',
      'MachineGuid'
    ])
  })

  it('falls back when reg query fails', async () => {
    onPlatform('win32')
    os.execFile.mockRejectedValue(new Error('ERROR: The system was unable to find the specified registry key'))
    os.networkInterfaces.mockReturnValue({ Ethernet: [nic('de:ad:be:ef:00:01')] })

    expect(await getMachineId()).toBe('build-host-de:ad:be:ef:00:01')
  })

  it('falls back when reg query prints no MachineGuid', async () => {
    onPlatform('win32')
    os.execFile.mockResolvedValue({
      stdout: '\r\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\r\n',
      stderr: ''
    })

    expect(await getMachineId()).toBe('build-host-unknown')
  })
})

describe('getMachineId fallback', () => {
  it('goes straight to hostname and MAC on a platform with no id source', async () => {
    onPlatform('freebsd')
    os.networkInterfaces.mockReturnValue({ em0: [nic('02:00:00:00:00:01')] })

    expect(await getMachineId()).toBe('build-host-02:00:00:00:00:01')
    expect(os.readFile).not.toHaveBeenCalled()
    expect(os.execFile).not.toHaveBeenCalled()
  })

  it('skips loopback, all-zero and empty interfaces for the first real MAC', async () => {
    onPlatform('freebsd')
    os.networkInterfaces.mockReturnValue({
      lo: [nic('00:00:00:00:00:00', true)],
      missing: undefined,
      tunnel: [nic('00:00:00:00:00:00')],
      eth0: [nic('aa:00:00:00:00:01'), nic('aa:00:00:00:00:02')],
      eth1: [nic('bb:00:00:00:00:01')]
    })

    expect(await getMachineId()).toBe('build-host-aa:00:00:00:00:01')
  })

  it('uses "unknown" when no interface has a usable MAC', async () => {
    onPlatform('freebsd')
    os.networkInterfaces.mockReturnValue({ lo: [nic('00:00:00:00:00:00', true)] })

    expect(await getMachineId()).toBe('build-host-unknown')
  })
})

describe('getUsername fallback', () => {
  function userInfoFails(): void {
    os.userInfo.mockImplementation(() => {
      throw Object.assign(new Error('no passwd entry'), { code: 'ENOENT' })
    })
  }

  it('returns the OS account name when userInfo works', () => {
    os.userInfo.mockReturnValue({ username: 'builder', uid: 1000, gid: 1000, shell: null, homedir: '/home/builder' })

    expect(getUsername()).toBe('builder')
  })

  it('uses USER when userInfo throws', () => {
    userInfoFails()
    vi.stubEnv('USER', 'env-user')
    vi.stubEnv('USERNAME', 'windows-user')

    expect(getUsername()).toBe('env-user')
  })

  it('uses USERNAME when USER is unset', () => {
    userInfoFails()
    vi.stubEnv('USER', undefined)
    vi.stubEnv('USERNAME', 'windows-user')

    expect(getUsername()).toBe('windows-user')
  })

  it('returns "unknown" when neither variable is set', () => {
    userInfoFails()
    vi.stubEnv('USER', undefined)
    vi.stubEnv('USERNAME', undefined)

    expect(getUsername()).toBe('unknown')
  })
})
