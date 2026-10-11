"""Token authentication: users.toml, three modes, ASGI middleware, identity context."""

from hull_core.auth.asgi import HullAuthMiddleware
from hull_core.auth.bundled_client import (
    BundledClientSpec,
    ResolvedClient,
    resolve_bundled_client,
    token_client_mismatch,
)
from hull_core.auth.context import AuthContext, current_user, reset_current_user, set_current_user
from hull_core.auth.middleware import Authenticator, AuthOutcome
from hull_core.auth.tokens import TokenHashError, hash_token, verify_token
from hull_core.auth.users import (
    User,
    UserLimits,
    UsersError,
    ensure_path_allowed,
    find_user_by_token,
    load_users,
)

__all__ = [
    "AuthContext",
    "AuthOutcome",
    "Authenticator",
    "BundledClientSpec",
    "HullAuthMiddleware",
    "ResolvedClient",
    "TokenHashError",
    "User",
    "UserLimits",
    "UsersError",
    "current_user",
    "ensure_path_allowed",
    "find_user_by_token",
    "hash_token",
    "load_users",
    "reset_current_user",
    "resolve_bundled_client",
    "set_current_user",
    "token_client_mismatch",
    "verify_token",
]
