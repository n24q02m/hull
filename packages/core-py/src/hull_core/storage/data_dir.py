"""On-disk data directory layout for per-plugin credential storage.

Mirrors ``packages/core-ts/src/storage/data-dir.ts``. Current layout (written
by every save)::

    ~/.n24q02m/<plugin>/config.json                           single-user blob (default profile)
    ~/.n24q02m/<plugin>/profiles/<prof>/config.json           single-user blob (named profile)
    ~/.n24q02m/<plugin>/subs/<sub>/config.json                multi-user (per-sub) blob
    ~/.n24q02m/<plugin>/subs/<sub>/profiles/<prof>/config.json multi-user named profile blob
    ~/.n24q02m/<plugin>/.secret                               single-user machine key (0600)

The base directory defaults to ``~/.n24q02m`` and can be overridden via
``N24Q02M_HOME`` or ``N24Q02M_USE_XDG=1`` (which uses ``$XDG_CONFIG_HOME/n24q02m``
or ``%APPDATA%/n24q02m``).

Earlier releases stored the same files directly under ``$HOME``. They stay
READABLE (never written) so credentials already on disk keep working, in this
order (newest naming first):

    1. ~/.n24q02m/<plugin>/       current
    2. ~/.<plugin>/               legacy (v1 hull), keyed by ``<plugin>``
    3. ~/.<plugin>-mcp/           legacy, keyed by ``<plugin>``
    4. ~/.better-<plugin>-mcp/    legacy, keyed by ``better-<plugin>``

"Keyed by" is the plugin name the legacy writer used for key derivation: the
single-user machine key is that directory's own ``.secret`` and the multi-user
salt is ``<key_plugin>:<sub>``.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
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


def validate_profile(profile: str) -> str:
    """Validate a Tier-2 profile name for use as one path component; return it unchanged."""
    if not isinstance(profile, str) or _is_unsafe(profile):
        raise ValueError("Invalid profile")
    return profile


def resolve_profile(
    plugin: str,
    profile: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve effective Tier-2 profile: explicit arg -> ``<PLUGIN>_PROFILE`` -> ``HULL_PROFILE`` -> ``'default'``."""
    validate_plugin(plugin)
    if profile is not None and profile != "":
        return validate_profile(profile)
    environ = os.environ if env is None else env
    plugin_env = f"{plugin.upper().replace('-', '_').replace('.', '_')}_PROFILE"
    from_plugin = environ.get(plugin_env, "").strip()
    if from_plugin:
        return validate_profile(from_plugin)
    from_hull = environ.get("HULL_PROFILE", "").strip()
    if from_hull:
        return validate_profile(from_hull)
    return "default"


def n24q02m_base_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the root n24q02m state directory (``~/.n24q02m`` by default)."""
    environ = os.environ if env is None else env
    override = environ.get("N24Q02M_HOME", "").strip()
    if override:
        return Path(override)
    if environ.get("N24Q02M_USE_XDG", "").strip().lower() in ("1", "true", "yes"):
        xdg = environ.get("XDG_CONFIG_HOME", "").strip() or environ.get("APPDATA", "").strip()
        if xdg:
            return Path(xdg) / "n24q02m"
    return Path.home() / ".n24q02m"


@dataclass(frozen=True)
class DataDirCandidate:
    """One place a plugin's credential blob may live on disk."""

    dir: Path
    key_plugin: str
    legacy: bool


def data_dir(plugin: str) -> Path:
    """Current data directory: ``~/.n24q02m/<plugin>`` (or XDG/N24Q02M_HOME override)."""
    return n24q02m_base_dir() / validate_plugin(plugin)


def data_dir_candidates(plugin: str) -> list[DataDirCandidate]:
    """Every data directory for ``plugin`` in read-precedence order."""
    validate_plugin(plugin)
    home = Path.home()
    return [
        DataDirCandidate(dir=data_dir(plugin), key_plugin=plugin, legacy=False),
        DataDirCandidate(dir=home / f".{plugin}", key_plugin=plugin, legacy=True),
        DataDirCandidate(dir=home / f".{plugin}-mcp", key_plugin=plugin, legacy=True),
        DataDirCandidate(dir=home / f".better-{plugin}-mcp", key_plugin=f"better-{plugin}", legacy=True),
    ]


def legacy_data_dirs(plugin: str) -> list[DataDirCandidate]:
    """Only the read-only legacy directories, in read-precedence order."""
    return [candidate for candidate in data_dir_candidates(plugin) if candidate.legacy]


def blob_path_in(dir: Path, sub: str | None = None, profile: str | None = None) -> Path:  # noqa: A002
    """Path of the credential blob inside ``dir``."""
    base = dir if sub is None else dir / "subs" / validate_sub(sub)
    if profile is not None and profile != "" and validate_profile(profile) != "default":
        base = base / "profiles" / profile
    return base / "config.json"


def migrate_legacy_data_dir(plugin: str) -> bool:
    """One-time non-destructive auto-migration from ``~/.<plugin>`` to ``~/.n24q02m/<plugin>``.

    When the target ``data_dir(plugin)`` does not exist yet and ``~/.<plugin>``
    exists as a directory, copies the legacy directory tree into ``data_dir(plugin)``
    while leaving ``~/.<plugin>`` untouched, and returns ``True``.
    """
    validate_plugin(plugin)
    target = data_dir(plugin)
    if target.exists():
        return False
    legacy_dot = Path.home() / f".{plugin}"
    if not legacy_dot.is_dir():
        return False
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copytree(legacy_dot, target)
    return True
