"""Persisted Telethon StringSession seam with save-on-change.

``StringSessionStore``
    Wraps :class:`~hull_core.storage.per_plugin_store.PerPluginStore` (leaf
    ``session``) to persist a Telethon session string as an encrypted
    ``{"session": str}`` blob in the configured credential backend. Never
    imports telethon.

``SaveOnChangeStringSession``
    A Telethon ``StringSession`` subclass whose ``save()`` pushes the
    serialized string to a ``sink`` callback, so the caller can persist every
    auth-state change (e.g. into ``StringSessionStore``) without polling.

Telethon gating
    telethon is a Telegram-only dependency and hull never declares it. The
    subclass is built on first attribute access (module ``__getattr__``), so
    importing this module, or ``hull_core.storage``, works without telethon;
    touching ``SaveOnChangeStringSession`` without it raises ``ImportError``
    with an install hint instead of handing back an unusable class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hull_core.storage.backends import CredentialBackend
from hull_core.storage.per_plugin_store import PerPluginStore

# SaveOnChangeStringSession is provided lazily by the module __getattr__ below.
__all__ = ["SaveOnChangeStringSession", "StringSessionStore"]  # noqa: F822

TELETHON_INSTALL_HINT = (
    "SaveOnChangeStringSession requires telethon, which hull does not install. "
    "Add `telethon` to your server's dependencies (e.g. `uv add telethon`)."
)


class StringSessionStore:
    """Encrypted store for a single Telethon session string."""

    def __init__(self, plugin: str, sub: str | None = None, *, backend: CredentialBackend | None = None) -> None:
        self._store = PerPluginStore(plugin, sub, backend=backend, sub_key="session")

    @property
    def store(self) -> PerPluginStore:
        return self._store

    def load(self) -> str | None:
        """Return the stored session string, or ``None`` when absent/undecryptable."""
        payload = self._store.load()
        if not payload:
            return None
        return payload.get("session")

    def save(self, session_str: str) -> None:
        """Persist ``{"session": session_str}``."""
        self._store.save({"session": session_str})

    def clear(self) -> None:
        """Delete the stored session blob (every location)."""
        self._store.clear()


_save_on_change_cls: type | None = None


def _build_save_on_change_cls() -> type:
    try:
        from telethon.sessions import StringSession  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(TELETHON_INSTALL_HINT) from exc

    class SaveOnChangeStringSession(StringSession):
        """Telethon StringSession that pushes its serialized string to a sink on save."""

        def __init__(self, session_str: str | None = None, sink: Callable[[str], None] | None = None) -> None:
            if session_str is None:
                super().__init__()
            else:
                super().__init__(session_str)
            self._sink = sink

        def save(self) -> str:
            result = super().save()
            if self._sink is not None:
                self._sink(result)
            return result

    SaveOnChangeStringSession.__module__ = __name__
    return SaveOnChangeStringSession


def __getattr__(name: str) -> Any:
    if name == "SaveOnChangeStringSession":
        global _save_on_change_cls
        if _save_on_change_cls is None:
            _save_on_change_cls = _build_save_on_change_cls()
        return _save_on_change_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
