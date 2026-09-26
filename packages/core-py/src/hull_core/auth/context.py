"""Request-scoped identity for MCP tools.

The pure-ASGI auth middleware resolves the caller before the MCP endpoint
runs and stores the resulting :class:`AuthContext` in a contextvar. Tools read
it with :func:`current_user` and scope every storage access by its namespace.
A pure-ASGI middleware (not Starlette's BaseHTTPMiddleware) is used so the
contextvar propagates into the endpoint task without anyio task-group copies.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field

_CURRENT: ContextVar["AuthContext | None"] = ContextVar("hull_current_user", default=None)


@dataclass(frozen=True)
class AuthContext:
    """Resolved caller identity attached to one HTTP request."""

    uid: str
    namespace: str
    mode: str
    allowed_roots: tuple[str, ...] = ()
    rpm: int | None = None
    disabled: bool = False
    attributes: dict = field(default_factory=dict)

    @staticmethod
    def local() -> "AuthContext":
        """Identity used in no-auth mode (and for in-process tool calls)."""
        return AuthContext(uid="local", namespace="default", mode="no-auth")


def set_current_user(ctx: AuthContext) -> Token:
    return _CURRENT.set(ctx)


def reset_current_user(token: Token) -> None:
    _CURRENT.reset(token)


def current_user() -> AuthContext:
    """The caller's identity; falls back to the local identity when unset."""
    ctx = _CURRENT.get()
    return ctx if ctx is not None else AuthContext.local()
