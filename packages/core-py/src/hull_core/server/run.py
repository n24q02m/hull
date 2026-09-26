"""Blocking server runner for ``hull server start``.

Acquires the per-(name, port) lifecycle lock BEFORE binding so two instances
cannot race the same port, then hands the app to uvicorn. In ``no-auth`` mode
a non-loopback bind is refused: an unauthenticated listener must never leave
localhost.
"""

from __future__ import annotations

import socket
from pathlib import Path

import uvicorn

from hull_core.config.settings import HullSettings, load_settings
from hull_core.lifecycle.lock import LifecycleLock
from hull_core.server.app import build_app

_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


class ServerConfigError(ValueError):
    """Invalid server start configuration."""


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return int(socket.inet_aton(host).hex(), 16) & 0xFF000000 == 0x7F000000
    except OSError:
        return False


def run_server(
    settings: HullSettings | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    lock_root: Path | None = None,
) -> None:
    """Blocking entry point: acquire lock, serve until interrupted."""
    settings = settings if settings is not None else load_settings()
    bind_host = host or settings.server.host
    bind_port = port or settings.server.port

    if settings.server.auth == "no-auth" and not _is_loopback(bind_host):
        raise ServerConfigError(
            f"auth = 'no-auth' only permits loopback binds, refusing host {bind_host!r} "
            "(set [server] auth to 'token' or 'multi' for a shared listener)"
        )

    lock = LifecycleLock("hull", bind_port, root=lock_root)
    with lock:
        app = build_app(settings)
        uvicorn.run(app, host=bind_host, port=bind_port, log_level="info")
