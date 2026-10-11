"""Storage: local SQLite (WAL) under ``~/.hull/``, per-plugin encrypted credentials, legacy config.enc reader."""

from typing import Any

from hull_core.storage.backends import (
    CredentialBackend,
    InMemoryBackend,
    LocalFsBackend,
    backend_from_env,
)
from hull_core.storage.data_dir import (
    DataDirCandidate,
    blob_path_in,
    data_dir,
    data_dir_candidates,
    legacy_data_dirs,
    validate_plugin,
    validate_sub,
)
from hull_core.storage.legacy_config_file import delete_legacy_config, read_legacy_config
from hull_core.storage.per_plugin_store import MachineKeyError, PerPluginStore
from hull_core.storage.sqlite import HullDatabase
from hull_core.storage.string_session_store import StringSessionStore

__all__ = [
    "CredentialBackend",
    "DataDirCandidate",
    "HullDatabase",
    "InMemoryBackend",
    "LocalFsBackend",
    "MachineKeyError",
    "PerPluginStore",
    "SaveOnChangeStringSession",
    "StringSessionStore",
    "backend_from_env",
    "blob_path_in",
    "data_dir",
    "data_dir_candidates",
    "delete_legacy_config",
    "legacy_data_dirs",
    "read_legacy_config",
    "validate_plugin",
    "validate_sub",
]


def __getattr__(name: str) -> Any:
    # SaveOnChangeStringSession needs telethon; resolve it lazily so importing
    # hull_core.storage never requires that Telegram-only dependency.
    if name == "SaveOnChangeStringSession":
        from hull_core.storage import string_session_store

        return string_session_store.SaveOnChangeStringSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
