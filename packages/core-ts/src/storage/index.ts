export { AccountRegistry } from './account-registry.js'
export { atomicWriteFile } from './atomic-write.js'
export {
  backendFromEnv,
  type CredentialBackend,
  InMemoryBackend,
  LocalFsBackend,
  type LocalFsBackendOptions
} from './backends.js'
export { exportConfig, importConfig, listConfigs, setConfigPath } from './config-file.js'
// Single-user credential read/write/delete route through the unified
// per-plugin store (with legacy config.enc read fallback), so downstream
// servers and the CLI built-ins converge on the same on-disk location.
export {
  deleteStoredConfig as deleteConfig,
  readStoredConfig as readConfig,
  writeStoredConfig as writeConfig
} from './credential-store.js'
export { type DataDirLocation, dataDir, dataDirCandidates, legacyDataDirs } from './data-dir.js'
export { decryptData, deriveFileKey, derivePassphraseKey, encryptData } from './encryption.js'
export {
  AccountError,
  type CooldownOptions,
  type Credential,
  CredentialExistsError,
  type CredentialHealth,
  CredentialNotFoundError,
  CredentialPoolError,
  classifyStatus,
  cooldownSeconds,
  DEFAULT_COOLDOWN_SECONDS,
  type FailureAction,
  KeyPool,
  type KeyPoolOptions,
  KNOWN_PROVIDER_HOSTS,
  MAX_COOLDOWN_SECONDS,
  MIN_COOLDOWN_SECONDS,
  type MigrationResult,
  PoolExhaustedError,
  parseRateLimitReset,
  parseRetryAfter,
  providerFromBaseUrl
} from './key-pool.js'
export {
  CredentialCryptoError,
  decryptSecret,
  encryptSecret,
  keyFingerprint,
  keyHint,
  loadOrCreateKey,
  SECRET_KEY_LENGTH,
  SECRET_NONCE_LENGTH,
  SecretCipher
} from './key-pool-crypto.js'
export {
  type AccountRow,
  type AddCredentialInput,
  CredentialStore,
  type CredentialStoreOptions,
  credentialKeyPath,
  defaultHullDbPath,
  KEY_POOL_SCHEMA,
  type ModelCellLike,
  type PoolDiagnostics
} from './key-pool-store.js'
export { getMachineId, getUsername } from './machine-id.js'
export { credPath, PerPluginStore, setHomeDirForTesting } from './per-plugin-store.js'
export { type ConfigSource, type ResolvedConfig, resolveConfig } from './resolver.js'
