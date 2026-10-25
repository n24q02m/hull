"""Read-only access to mcp-core's shared encrypted ``config.enc``.

Servers that moved from ``n24q02m-mcp-core`` to hull keep their users' saved
credentials (zero re-auth) by reading their entry from the old shared store
and re-saving it into their own :class:`~hull_core.storage.PerPluginStore`.
This module is the port of the decrypt half of mcp-core's
``mcp_core.storage.config_file``; hull never creates this file or adds
entries to it. The only write is :func:`delete_legacy_config`, which drops one
server's entry once it has been migrated and keeps everyone else's.

File location (unchanged from mcp-core)::

    platformdirs.user_config_dir("mcp", appauthor=False) / "config.enc"

Plaintext is JSON ``{"version": 1, "servers": {"<server>": {"KEY": "value"}}}``.

On-disk formats, tried in this order (same as mcp-core's ``_load_store``):

1. current: ``[16-byte salt][12-byte IV][AES-256-GCM ciphertext + tag]``,
   key = ``PBKDF2-HMAC-SHA256("<machine_id>:<username>", salt, 600_000)``
2. legacy: ``[12-byte IV][ciphertext + tag]``, salt ``b"mcp-relay-config"``,
   600 000 iterations
3. legacy: same, 100 000 iterations

``machine_id`` / ``username`` reproduce mcp-core's ``machine_id`` module
exactly (``/etc/machine-id``, macOS ``IOPlatformUUID``, Windows
``MachineGuid``, else ``<hostname>-<first MAC>``), because the key is bound to
them. Nothing here logs or returns key material; undecryptable data reads as
"no entry".
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeGuard

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from platformdirs import user_config_dir

from hull_core.storage.backends import atomic_write_bytes

LEGACY_SALT = b"mcp-relay-config"
PBKDF2_ITERATIONS = 600_000
LEGACY_PBKDF2_ITERATIONS = 100_000
_SALT_SIZE = 16
_IV_SIZE = 12
_KEY_LEN = 32


def default_legacy_config_path() -> Path:
    """Where mcp-core kept the shared store on this OS."""
    return Path(user_config_dir("mcp", appauthor=False)) / "config.enc"


def _first_mac() -> str:
    try:
        mac_int = uuid.getnode()
        # A random node id (no real NIC found) has the multicast bit set.
        if (mac_int >> 40) & 1:
            return "unknown"
        mac_str = ":".join(f"{(mac_int >> (8 * i)) & 0xFF:02x}" for i in range(5, -1, -1))
        if mac_str == "00:00:00:00:00:00":
            return "unknown"
        return mac_str
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def machine_id() -> str:
    """mcp-core's stable machine identifier (the file key is bound to it)."""
    plat = sys.platform
    try:
        if plat.startswith("linux"):
            with open("/etc/machine-id") as handle:
                return handle.read().strip()
        if plat == "darwin":
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True, timeout=5
            )
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', result.stdout)
            if match:
                return match.group(1)
        if plat == "win32":
            result = subprocess.run(
                ["reg", "query", r"HKLM\SOFTWARE\Microsoft\Cryptography", "/v", "MachineGuid"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            match = re.search(r"MachineGuid\s+REG_SZ\s+(\S+)", result.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass
    return f"{socket.gethostname()}-{_first_mac()}"


def username() -> str:
    """mcp-core's username lookup (``getpass`` then ``USER`` / ``USERNAME``)."""
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


def _file_key(salt: bytes, iterations: int) -> bytes:
    material = f"{machine_id()}:{username()}".encode()
    return hashlib.pbkdf2_hmac("sha256", material, salt, iterations, dklen=_KEY_LEN)


def _decrypt_json(key: bytes, data: bytes) -> Any:
    plaintext = AESGCM(key).decrypt(data[:_IV_SIZE], data[_IV_SIZE:], None)
    return json.loads(plaintext.decode("utf-8"))


def _valid_store(store: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(store, dict) and isinstance(store.get("servers"), dict)


def _load(path: Path) -> tuple[dict[str, Any], bytes] | None:
    """Decrypt the store; returns ``(store, salt to re-encrypt with)`` or ``None``."""
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if len(data) >= _SALT_SIZE + _IV_SIZE:
        salt = data[:_SALT_SIZE]
        try:
            store = _decrypt_json(_file_key(salt, PBKDF2_ITERATIONS), data[_SALT_SIZE:])
        except Exception:
            store = None
        if _valid_store(store):
            return store, salt

    for iterations in (PBKDF2_ITERATIONS, LEGACY_PBKDF2_ITERATIONS):
        try:
            store = _decrypt_json(_file_key(LEGACY_SALT, iterations), data)
        except Exception:
            continue
        if _valid_store(store):
            # A legacy-format file is rewritten in the current format with a
            # fresh salt, as mcp-core's own auto-migration would have done.
            return store, os.urandom(_SALT_SIZE)
    return None


def read_legacy_config(server_name: str, *, path: Path | None = None) -> dict[str, str] | None:
    """Return ``server_name``'s entry from mcp-core's ``config.enc``, or ``None``.

    ``None`` covers a missing file, a missing entry and any undecryptable or
    malformed content; this never raises on bad data and never writes.
    """
    loaded = _load(path or default_legacy_config_path())
    if loaded is None:
        return None
    entry = loaded[0]["servers"].get(server_name)
    return entry if isinstance(entry, dict) else None


def delete_legacy_config(server_name: str, *, path: Path | None = None) -> bool:
    """Remove only ``server_name``'s entry; return whether one was removed.

    The remaining entries are re-encrypted in the current format so other
    servers keep their credentials. When no entry remains the file is deleted
    (mcp-core's ``delete_config`` behaviour). A missing or undecryptable file
    is left untouched and reported as ``False``.
    """
    config_path = path or default_legacy_config_path()
    loaded = _load(config_path)
    if loaded is None:
        return False
    store, salt = loaded
    if server_name not in store["servers"]:
        return False
    del store["servers"][server_name]
    if not store["servers"]:
        config_path.unlink(missing_ok=True)
        return True
    iv = os.urandom(_IV_SIZE)
    ciphertext = AESGCM(_file_key(salt, PBKDF2_ITERATIONS)).encrypt(iv, json.dumps(store).encode("utf-8"), None)
    atomic_write_bytes(config_path, salt + iv + ciphertext)
    return True
