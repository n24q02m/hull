/**
 * MCP 2026-07-28 Skills extension helper for @n24q02m/hull.
 *
 * Wraps `@modelcontextprotocol/server/ext/skills` and
 * `@modelcontextprotocol/server/ext/skills/node` with explicit directory
 * validation before attaching the catalog to an `McpServer`.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import type { McpServer } from '@modelcontextprotocol/server'
import {
  SKILLS_EXTENSION_ID,
  type SkillCatalog,
  type SkillCatalogChanges,
  SkillCatalogError,
  type SkillCatalogOptions,
  type SkillWarning
} from '@modelcontextprotocol/server/ext/skills'
import { loadSkills } from '@modelcontextprotocol/server/ext/skills/node'

export {
  loadSkills,
  SKILLS_EXTENSION_ID,
  type SkillCatalog,
  type SkillCatalogChanges,
  SkillCatalogError,
  type SkillCatalogOptions,
  type SkillWarning
}

/**
 * Validate `skillsDir` exists as a directory, load all skills into a
 * `SkillCatalog`, attach it to `mcp`, and return the catalog.
 */
export async function attachSkillsProvider(
  mcp: McpServer,
  skillsDir: string,
  options?: SkillCatalogOptions
): Promise<SkillCatalog> {
  const trimmed = skillsDir.trim()
  if (!trimmed) {
    throw new Error('skillsDir must be a non-empty directory path')
  }
  const resolved = path.resolve(trimmed)
  if (!fs.existsSync(resolved)) {
    throw new Error(`Skills directory does not exist: ${resolved}`)
  }
  if (!fs.statSync(resolved).isDirectory()) {
    throw new Error(`Skills path is not a directory: ${resolved}`)
  }

  const catalog = await loadSkills(resolved, options)
  catalog.attach(mcp)
  return catalog
}
