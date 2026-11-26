import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Client, StreamableHTTPClientTransport } from '@modelcontextprotocol/client'
import { McpServer } from '@modelcontextprotocol/server'
import { afterEach, describe, expect, it } from 'vitest'

import { attachSkillsProvider, SKILLS_EXTENSION_ID } from '../../src/skills/index.js'
import { type HttpServerHandle, runHttpServer } from '../../src/transport/local-server.js'

let tmpDir: string | undefined
let handle: HttpServerHandle | undefined

afterEach(async () => {
  if (handle) {
    await handle.close()
    handle = undefined
  }
  if (tmpDir) {
    rmSync(tmpDir, { recursive: true, force: true })
    tmpDir = undefined
  }
})

describe('skills provider', () => {
  it('exports the MCP skills extension identifier', () => {
    expect(SKILLS_EXTENSION_ID).toBe('io.modelcontextprotocol/skills')
  })

  it('rejects empty, missing, and non-directory skills paths', async () => {
    const server = new McpServer({ name: 'skills-err', version: '0.1.0' })
    await expect(attachSkillsProvider(server, '   ')).rejects.toThrow('non-empty directory path')

    tmpDir = mkdtempSync(join(tmpdir(), 'hull-skills-err-'))
    const missing = join(tmpDir, 'does-not-exist')
    await expect(attachSkillsProvider(server, missing)).rejects.toThrow('does not exist')

    const filePath = join(tmpDir, 'not-a-dir.txt')
    writeFileSync(filePath, 'hello')
    await expect(attachSkillsProvider(server, filePath)).rejects.toThrow('not a directory')
  })

  it('loads skills from disk, attaches to McpServer, and serves via Client over HTTP', async () => {
    tmpDir = mkdtempSync(join(tmpdir(), 'hull-skills-ok-'))
    const skillDir = join(tmpDir, 'hello-skill')
    mkdirSync(skillDir, { recursive: true })
    writeFileSync(
      join(skillDir, 'SKILL.md'),
      [
        '---',
        'name: hello-skill',
        'description: Friendly greeting skill',
        '---',
        '',
        '# Hello Skill',
        'Say hi cleanly.'
      ].join('\n')
    )

    const resolvedSkillsDir = tmpDir
    const directServer = new McpServer({ name: 'direct-skills', version: '0.1.0' })
    const catalog = await attachSkillsProvider(directServer, resolvedSkillsDir)
    expect(catalog.entries.map((s) => s.frontmatter.name)).toContain('hello-skill')

    const skillUri = catalog.entries[0]?.uri
    expect(skillUri).toBeDefined()

    handle = await runHttpServer(
      () => {
        const mcp = new McpServer({ name: 'direct-skills-http', version: '0.1.0' })
        catalog.attach(mcp)
        return mcp
      },
      { serverName: `test-skills-client-${Date.now()}`, port: 0 }
    )

    const client = new Client({ name: 'skills-client', version: '0.1.0' })
    const transport = new StreamableHTTPClientTransport(new URL(`http://${handle.host}:${handle.port}/mcp`))
    await client.connect(transport)
    try {
      const read = await client.readResource({ uri: skillUri as string })
      expect(JSON.stringify(read.contents)).toContain('Say hi cleanly.')
    } finally {
      await client.close()
    }
  })
})
