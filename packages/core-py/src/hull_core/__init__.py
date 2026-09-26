"""hull-core: shared auth, model config, limiter, storage, and MCP server core."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hull-core")
except PackageNotFoundError:  # package not installed (e.g. running from source tree)
    __version__ = "0.0.0+unknown"

from hull_core.auth.context import AuthContext, current_user
from hull_core.auth.middleware import Authenticator
from hull_core.auth.tokens import hash_token, verify_token
from hull_core.auth.users import User, UserLimits, UsersError, load_users
from hull_core.config.models import ModelCell, resolve_model_cells
from hull_core.config.settings import HullSettings, ServerSettings, load_settings
from hull_core.limits.limiter import SlidingWindowLimiter
from hull_core.storage.sqlite import HullDatabase

__all__ = [
    "AuthContext",
    "Authenticator",
    "HullDatabase",
    "HullSettings",
    "ModelCell",
    "ServerSettings",
    "SlidingWindowLimiter",
    "User",
    "UserLimits",
    "UsersError",
    "current_user",
    "hash_token",
    "load_settings",
    "load_users",
    "resolve_model_cells",
    "verify_token",
]
