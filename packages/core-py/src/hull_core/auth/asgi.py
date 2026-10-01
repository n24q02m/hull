"""Pure-ASGI middleware that authenticates the request and binds the identity.

Runs before the MCP endpoint (passed through FastMCP's ``http_app`` middleware
list) and publishes :class:`AuthContext` via the module contextvar so tools
can scope storage access per user. Rejections are JSON:
``401`` (missing/invalid token), ``403`` (disabled user), ``429`` (over limit).
"""

from __future__ import annotations

import json

from hull_core.auth.context import set_current_user, reset_current_user
from hull_core.auth.middleware import Authenticator

_STATUS_TEXT = {
    401: "unauthorized",
    403: "forbidden",
    429: "rate limit exceeded",
    500: "server misconfiguration",
}


class HullAuthMiddleware:
    """ASGI middleware: authenticate → bind contextvar → forward, or reject."""

    def __init__(self, app, authenticator: Authenticator) -> None:
        self.app = app
        self.authenticator = authenticator

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        authz = None
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                authz = value.decode("latin-1")
                break

        outcome = self.authenticator.authenticate(authz)
        if not outcome.ok:
            await self._reject(send, outcome.status or 500, outcome.detail)
            return

        context = outcome.context
        if context is None:
            # ok=True always carries a context; fail closed rather than bind a
            # null identity into the contextvar for the rest of the request.
            await self._reject(send, 500, "authenticator returned no context")
            return

        token = set_current_user(context)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_current_user(token)

    async def _reject(self, send, status: int, detail: str) -> None:  # noqa: ANN001
        body = json.dumps({"error": _STATUS_TEXT.get(status, "error"), "detail": detail}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if status == 401:
            headers.append((b"www-authenticate", b"Bearer"))
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body})
