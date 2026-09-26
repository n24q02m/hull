"""Token authentication: one mechanism, three modes (spec §4).

Mode is a config state, resolved by the host in ``config.toml``:

- ``no-auth`` — localhost-only; one shared namespace; no Authorization header.
- ``token``   — one shared token; still one shared namespace.
- ``multi``   — users.toml; token → user → namespace; 401 wrong token,
  403 disabled user, 429 over limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from hull_core.auth.context import AuthContext
from hull_core.auth.tokens import verify_token
from hull_core.auth.users import User, UsersError, find_user_by_token
from hull_core.config.settings import HullSettings
from hull_core.limits.limiter import SlidingWindowLimiter


@dataclass(frozen=True)
class AuthOutcome:
    """Result of authenticating one request."""

    context: AuthContext | None = None
    status: int | None = None  # None = proceed; 401/403 = reject
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is None and self.context is not None


class Authenticator:
    """Resolves bearer tokens to AuthContexts according to the configured mode."""

    def __init__(
        self,
        settings: HullSettings,
        users: dict[str, User] | None = None,
        limiter: SlidingWindowLimiter | None = None,
    ) -> None:
        self._settings = settings
        self._mode = settings.server.auth
        self._users = users
        self._limiter = limiter if limiter is not None else SlidingWindowLimiter()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def limiter(self) -> SlidingWindowLimiter:
        return self._limiter

    @staticmethod
    def _bearer(authz: str | None) -> str | None:
        if not authz:
            return None
        scheme, _, value = authz.partition(" ")
        if scheme.lower() != "bearer" or not value.strip():
            return None
        return value.strip()

    def authenticate(self, authz: str | None) -> AuthOutcome:
        if self._mode == "no-auth":
            return AuthOutcome(context=AuthContext.local())

        token = self._bearer(authz)
        if token is None:
            return AuthOutcome(status=401, detail="missing or malformed Authorization header")

        if self._mode == "token":
            token_hash = self._settings.server.token_hash
            if token_hash is None or not verify_token(token, token_hash):
                return AuthOutcome(status=401, detail="invalid token")
            return self._admit(
                AuthContext(uid="shared", namespace="default", mode="token", rpm=self._settings.server.rpm)
            )

        if self._mode == "multi":
            if self._users is None:
                raise UsersError("multi mode requires the users table (load users.toml first)")
            user = find_user_by_token(self._users, token)
            if user is None:
                return AuthOutcome(status=401, detail="invalid token")
            if not user.enabled:
                return AuthOutcome(status=403, detail=f"user {user.uid!r} is disabled")
            return self._admit(
                AuthContext(
                    uid=user.uid,
                    namespace=user.namespace,
                    mode="multi",
                    allowed_roots=user.allowed_roots,
                    rpm=user.limits.rpm,
                )
            )

        return AuthOutcome(status=500, detail=f"unknown auth mode {self._mode!r}")

    def _admit(self, ctx: AuthContext) -> AuthOutcome:
        if not self._limiter.check(ctx.uid, ctx.rpm):
            return AuthOutcome(status=429, detail="rate limit exceeded")
        return AuthOutcome(context=ctx)
