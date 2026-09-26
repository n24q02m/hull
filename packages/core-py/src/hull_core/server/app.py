"""HTTP MCP server: ``/mcp`` endpoint behind token auth.

Mount pattern (ported from the mcp-core transport): ``FastMCP.http_app`` is
the supported way to get a Starlette app pre-configured for Streamable HTTP —
middleware is attached at construction time. The auth middleware is pure ASGI
so the per-request identity contextvar reaches tool code in the same task.

The skeleton ships three demo tools proving the §4 isolation contract:
``namespace_info``, ``kv_put``, ``kv_get`` — storage is namespaced by the
authenticated caller, so two tokens with different namespaces never see each
other's keys within one process.
"""

from __future__ import annotations

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware

from hull_core.auth.asgi import HullAuthMiddleware
from hull_core.auth.context import current_user
from hull_core.auth.middleware import Authenticator
from hull_core.auth.users import User, load_users
from hull_core.config.settings import HullSettings, load_settings
from hull_core.limits.limiter import SlidingWindowLimiter
from hull_core.storage.sqlite import HullDatabase


def build_mcp(database: HullDatabase) -> FastMCP:
    mcp: FastMCP = FastMCP(name="hull")

    @mcp.tool
    def namespace_info() -> dict:
        """Report the authenticated caller's identity and namespace."""
        user = current_user()
        return {"uid": user.uid, "namespace": user.namespace, "mode": user.mode}

    @mcp.tool
    def kv_put(key: str, value: str) -> dict:
        """Store a string under the caller's namespace."""
        user = current_user()
        database.kv_put(user.namespace, key, value)
        return {"ok": True, "namespace": user.namespace, "key": key}

    @mcp.tool
    def kv_get(key: str) -> dict:
        """Read a string from the caller's namespace (absent → null)."""
        user = current_user()
        return {"namespace": user.namespace, "key": key, "value": database.kv_get(user.namespace, key)}

    return mcp


def load_users_for(settings: HullSettings) -> dict[str, User] | None:
    if settings.server.auth != "multi":
        return None
    if settings.server.users_file is None:
        raise RuntimeError("multi mode requires [server] users_file")
    return load_users(settings.server.users_file)


def build_app(
    settings: HullSettings | None = None,
    *,
    database: HullDatabase | None = None,
    limiter: SlidingWindowLimiter | None = None,
) -> Starlette:
    """Construct the authenticated MCP Starlette app (testable, no port bind)."""
    settings = settings if settings is not None else load_settings()
    database = database if database is not None else HullDatabase()
    users = load_users_for(settings)
    authenticator = Authenticator(settings, users=users, limiter=limiter or SlidingWindowLimiter())
    mcp = build_mcp(database)
    return mcp.http_app(
        path="/mcp",
        transport="streamable-http",
        middleware=[Middleware(HullAuthMiddleware, authenticator=authenticator)],
    )
