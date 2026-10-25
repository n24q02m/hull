/**
 * HTTP token-table auth (`no-auth` / `token` / `multi`), the TS port of
 * core-py `hull_core.auth` + `hull_core.limits`. Self-contained: nothing
 * here imports the relay, OAuth, or transport modules.
 */

export {
  AuthConfigError,
  Authenticator,
  type AuthOutcome,
  defaultUsersFile,
  HTTP_AUTH_MODES,
  type HttpAuthConfig,
  parseBearer
} from './authenticator.js'
export { type TokenCliHandler, type TokenCliIO, tokenCliHandler } from './cli.js'
export { type AuthContext, currentUser, type HttpAuthMode, localAuthContext, runWithAuthContext } from './context.js'
export { AUTH_STATUS_TEXT, guardRequest, type RequestAuthenticator, writeAuthRejection } from './http-guard.js'
export { SlidingWindowLimiter } from './limiter.js'
export { assertNoAuthLoopback, inetAton, isLoopbackHost, ServerConfigError } from './loopback.js'
export { hashToken, isValidTokenHash, TokenHashError, verifyToken } from './tokens.js'
export {
  ensurePathAllowed,
  findUserByToken,
  loadUsers,
  PathNotAllowedError,
  type PathScope,
  type TokenUser,
  type TokenUserLimits,
  UsersError
} from './users.js'
