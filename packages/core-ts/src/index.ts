// Console-script CLI builder shared by every TS MCP server (config/doctor/logout/status/token built-ins).
export { type BuildCliOptions, buildCli, type CliHandler, type ServeFn } from './cli/build-cli.js'
export * from './crypto/index.js'
export { exportConfig, importConfig, listConfigs, scheduleReloadExit } from './storage/config-file.js'
// Single-user credential read/write/delete route through the unified
// per-plugin store (with legacy config.enc read fallback).
export {
  deleteStoredConfig as deleteConfig,
  readStoredConfig as readConfig,
  writeStoredConfig as writeConfig
} from './storage/credential-store.js'
export * from './storage/resolver.js'
// HTTP token-table auth (no-auth / token / multi users.toml), the TS port of
// core-py hull_core.auth; wired into runHttpServer via its `auth` option.
export * from './token-auth/index.js'
// HTTP MCP server entry point (Streamable HTTP transport on host:port)
export {
  type HttpRoute,
  type HttpServerHandle,
  type RunHttpServerOptions,
  runHttpServer
} from './transport/local-server.js'
// Cross-platform best-effort browser opener (WSL-aware, headless/CI-guarded).
export { tryOpenBrowser } from './util/browser.js'
