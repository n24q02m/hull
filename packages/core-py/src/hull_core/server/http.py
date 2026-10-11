"""HTTP runner for consumer MCP servers (telegram, ...) built on hull auth.

A consumer hands over its own ``FastMCP`` instance; hull mounts it at ``/mcp``
(Streamable HTTP) behind :class:`~hull_core.auth.asgi.HullAuthMiddleware` and
adds an unauthenticated ``GET /health`` liveness probe. No relay, OAuth AS or
hosted mode exists here: the three hull auth modes (``no-auth`` / ``token`` /
``multi``) from ``HullSettings`` are the whole story, and ``no-auth`` refuses
any non-loopback bind exactly like ``hull server start``.
"""

from __future__ import annotations

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from hull_core.auth.asgi import HullAuthMiddleware
from hull_core.auth.middleware import Authenticator
from hull_core.auth.users import User
from hull_core.config.settings import HullSettings
from hull_core.limits.limiter import SlidingWindowLimiter
from hull_core.server.app import load_users_for
from hull_core.server.run import ServerConfigError, _is_loopback

HEALTH_PATH = "/health"


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def build_http_app(
    mcp: FastMCP,
    settings: HullSettings,
    *,
    users: dict[str, User] | None = None,
    limiter: SlidingWindowLimiter | None = None,
) -> Starlette:
    """Mount ``mcp`` at ``/mcp`` behind hull auth, plus a public ``GET /health``."""
    authenticator = Authenticator(settings, users, limiter=limiter or SlidingWindowLimiter())
    app = mcp.http_app(
        path="/mcp",
        transport="streamable-http",
        middleware=[Middleware(HullAuthMiddleware, authenticator=authenticator, public_paths=(HEALTH_PATH,))],
    )
    app.router.routes.append(Route(HEALTH_PATH, _health, methods=["GET"]))
    return app


async def run_http_server(
    mcp: FastMCP,
    settings: HullSettings,
    *,
    host: str | None = None,
    port: int | None = None,
    users: dict[str, User] | None = None,
) -> None:
    """Serve ``mcp`` over HTTP until cancelled (uvicorn, inside the caller's loop)."""
    bind_host = host or settings.server.host
    bind_port = port or settings.server.port
    if settings.server.auth == "no-auth" and not _is_loopback(bind_host):
        raise ServerConfigError(
            f"auth = 'no-auth' only permits loopback binds, refusing host {bind_host!r} "
            "(set [server] auth to 'token' or 'multi' for a shared listener)"
        )
    if settings.server.auth == "multi" and users is None:
        users = load_users_for(settings)
    app = build_http_app(mcp, settings, users=users)
    server = uvicorn.Server(uvicorn.Config(app, host=bind_host, port=bind_port, log_level="warning"))
    await server.serve()
