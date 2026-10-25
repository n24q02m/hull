"""Pluggable byte-blob backends for per-plugin credential storage.

Mirrors ``packages/core-ts/src/storage/backends.ts``. Every backend stores
opaque (already encrypted) blobs under a slash-separated key::

    <plugin>/config                    -> config.json
    <plugin>/subs/<sub>/config         -> subs/<sub>/config.json
    <plugin>/tokens/<provider>         -> tokens/<provider>.json
    <plugin>/subs/<sub>/tokens/<prov>  -> subs/<sub>/tokens/<prov>.json
    <plugin>/session                   -> session.json
    <plugin>/subs/<sub>/session        -> subs/<sub>/session.json

(paths relative to the local backend's base directory, ``~/.<plugin>`` by
default). The ``tokens`` and ``session`` leaves extend the TS mapping for the
Python consumers (OAuth token caches, Telethon string sessions).

Backend selection (:func:`backend_from_env`) reads ``MCP_STORAGE_BACKEND``:
``local`` / ``local-fs`` / unset -> :class:`LocalFsBackend`; ``memory`` /
``in-memory`` -> :class:`InMemoryBackend`. The remote ``cf-kv`` backend of
mcp-core is intentionally not ported (no hosted storage after de-hosting).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from hull_core.storage.data_dir import data_dir

# Module-level so tests can exercise the POSIX permission path on Windows.
_POSIX = os.name != "nt"


@runtime_checkable
class CredentialBackend(Protocol):
    """Opaque blob store keyed by ``<plugin>/...`` strings."""

    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, blob: bytes) -> None: ...

    def delete(self, key: str) -> None: ...


class InMemoryBackend:
    """Process-local backend (tests, ephemeral servers)."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def put(self, key: str, blob: bytes) -> None:
        self._store[key] = bytes(blob)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


def _validate_component(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"Invalid path component: {name!r}")
    return name


def _relative_path(key: str) -> tuple[str, Path]:
    """Split ``key`` into (plugin, path relative to the plugin base dir)."""
    plugin, _, rest = key.partition("/")
    _validate_component(plugin)
    parts = rest.split("/") if rest else []
    sub_prefix: tuple[str, ...] = ()
    if len(parts) >= 3 and parts[0] == "subs":
        sub_prefix = ("subs", _validate_component(parts[1]))
        parts = parts[2:]
    if parts == ["config"]:
        leaf: tuple[str, ...] = ("config.json",)
    elif parts == ["session"]:
        leaf = ("session.json",)
    elif len(parts) == 2 and parts[0] == "tokens":
        leaf = ("tokens", f"{_validate_component(parts[1])}.json")
    else:
        raise ValueError(f"Invalid backend key: {key!r}")
    return plugin, Path(*sub_prefix, *leaf)


def tighten_dir(directory: Path) -> None:
    """Force ``directory`` to 0700 on POSIX (no-op on Windows, which has no mode bits)."""
    if _POSIX:
        os.chmod(directory, 0o700)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically with mode 0600.

    The parent directory is created 0700 (and tightened to 0700 when it already
    exists, on POSIX). The bytes go to ``<path>.tmp`` in the same directory,
    are fsynced, then ``os.replace``d over the target, so a crash never leaves
    a truncated blob behind.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tighten_dir(parent)
    tmp = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class LocalFsBackend:
    """Stores blobs on local disk under a per-plugin base directory.

    ``base_dir`` overrides the default ``~/.<plugin>`` base: a fixed path, or a
    zero-arg callable resolved per operation. PerPluginStore reads the legacy
    layouts only through a default-layout backend (``is_default_layout``).
    """

    def __init__(self, base_dir: Path | Callable[[], Path] | None = None) -> None:
        self._base_dir = base_dir

    @property
    def is_default_layout(self) -> bool:
        return self._base_dir is None

    def _base_for(self, plugin: str) -> Path:
        base = self._base_dir
        if base is None:
            return data_dir(plugin)
        if isinstance(base, Path):
            return base
        return Path(base())

    def path_for(self, key: str) -> Path:
        """On-disk path of ``key`` (validated, guaranteed inside the base dir)."""
        plugin, relative = _relative_path(key)
        base = self._base_for(plugin)
        path = base / relative
        base_resolved = base.resolve()
        path_resolved = path.resolve()
        if path_resolved != base_resolved and not path_resolved.is_relative_to(base_resolved):
            raise ValueError(f"Path escapes base: {key!r}")
        return path

    def get(self, key: str) -> bytes | None:
        try:
            return self.path_for(key).read_bytes()
        except FileNotFoundError:
            return None

    def put(self, key: str, blob: bytes) -> None:
        atomic_write_bytes(self.path_for(key), blob)

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)


def backend_from_env(env: Mapping[str, str] | None = None) -> CredentialBackend:
    """Select a backend from ``MCP_STORAGE_BACKEND`` (default: local disk).

    Only local storage exists after de-hosting: ``local`` / ``local-fs`` /
    unset -> :class:`LocalFsBackend`, ``memory`` / ``in-memory`` ->
    :class:`InMemoryBackend` (tests, ephemeral runs). Any other value,
    including the removed ``cf-kv``, raises instead of silently falling back.
    """
    env = os.environ if env is None else env
    kind = env.get("MCP_STORAGE_BACKEND", "local").strip().lower()
    if kind in ("", "local", "local-fs"):
        return LocalFsBackend()
    if kind in ("memory", "in-memory"):
        return InMemoryBackend()
    if kind == "cf-kv":
        raise ValueError(
            "MCP_STORAGE_BACKEND=cf-kv is no longer supported: hull stores credentials on local disk only "
            "(unset MCP_STORAGE_BACKEND or set it to 'local')"
        )
    raise ValueError(f"Unknown MCP_STORAGE_BACKEND: {kind!r} (expected 'local', 'local-fs', 'memory' or 'in-memory')")
