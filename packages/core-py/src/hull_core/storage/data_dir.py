"""On-disk data directory layout for per-plugin credential storage.

Mirrors ``packages/core-ts/src/storage/data-dir.ts``. Current layout (written
by every save)::

    ~/.<plugin>/config.json              single-user blob
    ~/.<plugin>/subs/<sub>/config.json   multi-user (per-sub) blob
    ~/.<plugin>/.secret                  single-user machine key (0600)

Earlier releases stored the same files under two longer names. They stay
READABLE (never written) so credentials already on disk keep working, in this
order (newest naming first):

    1. ~/.<plugin>/               current
    2. ~/.<plugin>-mcp/           legacy, keyed by ``<plugin>``
    3. ~/.better-<plugin>-mcp/    legacy, keyed by ``better-<plugin>``

"Keyed by" is the plugin name the legacy writer used for key derivation: the
single-user machine key is that directory's own ``.secret`` and the multi-user
salt is ``<key_plugin>:<sub>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]")


def _is_unsafe(value: str) -> bool:
    return not value or value in (".", "..") or ".." in value or _UNSAFE.search(value) is not None


def validate_plugin(plugin: str) -> str:
    """Validate a plugin name for use as one path component; return it unchanged."""
    if not isinstance(plugin, str) or _is_unsafe(plugin):
        raise ValueError("Invalid plugin name")
    return plugin


def validate_sub(sub: str) -> str:
    """Validate a multi-user sub (``token_urlsafe`` alphabet allowed); return it unchanged."""
    if not isinstance(sub, str) or _is_unsafe(sub):
        raise ValueError("Invalid sub")
    return sub


@dataclass(frozen=True)
class DataDirCandidate:
    """One place a plugin's credential blob may live on disk."""

    dir: Path
    key_plugin: str
    legacy: bool


def data_dir(plugin: str) -> Path:
    """Current data directory: ``~/.<plugin>``."""
    return Path.home() / f".{validate_plugin(plugin)}"


def data_dir_candidates(plugin: str) -> list[DataDirCandidate]:
    """Every data directory for ``plugin`` in read-precedence order."""
    validate_plugin(plugin)
    home = Path.home()
    return [
        DataDirCandidate(dir=home / f".{plugin}", key_plugin=plugin, legacy=False),
        DataDirCandidate(dir=home / f".{plugin}-mcp", key_plugin=plugin, legacy=True),
        DataDirCandidate(dir=home / f".better-{plugin}-mcp", key_plugin=f"better-{plugin}", legacy=True),
    ]


def legacy_data_dirs(plugin: str) -> list[DataDirCandidate]:
    """Only the read-only legacy directories, in read-precedence order."""
    return [candidate for candidate in data_dir_candidates(plugin) if candidate.legacy]


def blob_path_in(dir: Path, sub: str | None = None) -> Path:  # noqa: A002
    """Path of the credential blob inside ``dir``."""
    if sub is None:
        return dir / "config.json"
    return dir / "subs" / validate_sub(sub) / "config.json"
