"""Per-plugin encrypted credential store (Python port of ``per-plugin-store.ts``).

Layout (written by every save; see :mod:`hull_core.storage.data_dir`)::

    single-user: ~/.<plugin>/<leaf>.json
    multi-user:  ~/.<plugin>/subs/<sub>/<leaf>.json

``leaf`` is ``config`` unless the caller passes ``sub_key`` (e.g. ``session``
for :class:`~hull_core.storage.string_session_store.StringSessionStore`).

Read fallback (default-layout :class:`LocalFsBackend` only), first blob found
wins: ``~/.<plugin>/`` -> ``~/.<plugin>-mcp/`` -> ``~/.better-<plugin>-mcp/``.
A legacy blob is decrypted with that directory's own key material and the
next :meth:`PerPluginStore.save` writes the current layout (migrate on write).
Legacy copies are never written; :meth:`PerPluginStore.clear` removes the blob
from every location.

Blob format (identical across hull TS, hull py and the old mcp-core py)::

    [12-byte nonce][AES-256-GCM ciphertext][16-byte tag]

Keys:

* single-user: 32 raw random bytes at ``<data dir>/.secret`` (0600). The
  current dir's secret is generated on first use; a legacy dir's secret is
  only ever read.
* multi-user (current layout): ``scrypt(CREDENTIAL_SECRET, salt="<plugin>:<sub>",
  N=16384, r=8, p=1, dklen=32)``, matching hull TS.
* multi-user (legacy dirs): PBKDF2-HMAC-SHA256, 600 000 iterations, salt
  ``"<key_plugin>:<sub>"``, which is what the old mcp-core Python store wrote.

Foreign ``.secret`` files (fail loudly, never overwrite)
    The current dir's ``.secret`` is used only when it is exactly 32 bytes.
    Any other existing content (empty, hex text such as the 64-char master
    secret of mcp-core's old ``credential_store``, any other length) raises
    :class:`MachineKeyError` naming the path, from both ``load()`` and
    ``save()``; hull never decrypts with it, never regenerates over it and
    never deletes it, because replacing a key silently orphans every blob
    encrypted under it. The operator (or the consumer's own migration code)
    must move the foreign file aside. A 32-byte file cannot be told apart
    from a hull key and is used as one. Generation is create-if-absent
    (hard link of a fully written temp file), so two processes racing on
    first use converge on one key instead of the later write winning.
    A legacy dir's ``.secret`` of the wrong length only makes that legacy
    blob read as corrupt (``load()`` -> ``None``); legacy dirs are read-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hull_core.storage.backends import (
    CredentialBackend,
    LocalFsBackend,
    atomic_write_bytes,
    backend_from_env,
    tighten_dir,
)
from hull_core.storage.data_dir import DataDirCandidate, data_dir, legacy_data_dirs, validate_plugin, validate_sub

_NONCE_LEN = 12
_TAG_LEN = 16
_MIN_BLOB_LEN = _NONCE_LEN + _TAG_LEN + 1
_KEY_LEN = 32
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_LEGACY_PBKDF2_ITERATIONS = 600_000
_SECRET_FILE = ".secret"
_MASTER_ENV = "CREDENTIAL_SECRET"


class MachineKeyError(RuntimeError):
    """The current data dir holds a ``.secret`` that is not a 32-byte hull machine key."""


def _master_secret() -> str:
    master = os.environ.get(_MASTER_ENV, "")
    if not master:
        raise RuntimeError(f"{_MASTER_ENV} env required for HTTP multi-user mode")
    return master


def derive_multi_user_key(plugin: str, sub: str) -> bytes:
    """Current multi-user key: scrypt over ``CREDENTIAL_SECRET`` (hull TS parity)."""
    return hashlib.scrypt(
        _master_secret().encode("utf-8"),
        salt=f"{plugin}:{sub}".encode("utf-8"),
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LEN,
    )


def derive_legacy_multi_user_key(key_plugin: str, sub: str) -> bytes:
    """Old mcp-core Python multi-user key: PBKDF2-HMAC-SHA256, 600k iterations."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        _master_secret().encode("utf-8"),
        f"{key_plugin}:{sub}".encode("utf-8"),
        _LEGACY_PBKDF2_ITERATIONS,
        dklen=_KEY_LEN,
    )


# Current-layout directories already tightened in this process: an existing
# overly-permissive dir is chmod-ed to 0700 once, even on the read path.
_tightened_dirs: set[Path] = set()


def _checked_machine_key(secret_path: Path, key: bytes) -> bytes:
    if len(key) != _KEY_LEN:
        raise MachineKeyError(
            f"{secret_path} is not a hull machine key (expected exactly {_KEY_LEN} raw bytes, found {len(key)}). "
            "hull will not use, overwrite or delete it. If the file belongs to another tool (e.g. the hex "
            "master secret of mcp-core's old credential_store), move it out of this directory; if it was "
            "hull's own key, credentials encrypted with it cannot be recovered and the file must be removed "
            "before running setup again."
        )
    return key


def _create_machine_key(secret_path: Path) -> bytes:
    """Create ``secret_path`` with a fresh key unless it already exists; return the winning key."""
    key = secrets.token_bytes(_KEY_LEN)
    tmp = secret_path.with_name(f"{_SECRET_FILE}.{secrets.token_hex(8)}.tmp")
    atomic_write_bytes(tmp, key)
    try:
        try:
            # link() never replaces an existing file: if another process won
            # the race, its key stays and ours is discarded.
            os.link(tmp, secret_path)
        except FileExistsError:
            return _checked_machine_key(secret_path, secret_path.read_bytes())
        except OSError:
            # Filesystem without hard links: exclusive create, single write.
            try:
                fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
            except FileExistsError:
                return _checked_machine_key(secret_path, secret_path.read_bytes())
            with os.fdopen(fd, "wb") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
        return key
    finally:
        tmp.unlink(missing_ok=True)


def _load_or_generate_machine_key(directory: Path) -> bytes:
    secret_path = directory / _SECRET_FILE
    if directory not in _tightened_dirs and directory.is_dir():
        tighten_dir(directory)
        _tightened_dirs.add(directory)
    try:
        return _checked_machine_key(secret_path, secret_path.read_bytes())
    except FileNotFoundError:
        return _create_machine_key(secret_path)


def _read_legacy_machine_key(directory: Path) -> bytes | None:
    try:
        return (directory / _SECRET_FILE).read_bytes()
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class _BlobSource:
    backend: CredentialBackend
    legacy: DataDirCandidate | None


class PerPluginStore:
    """Encrypted JSON-dict store for one plugin (and optionally one multi-user sub)."""

    def __init__(
        self,
        plugin: str,
        sub: str | None = None,
        *,
        backend: CredentialBackend | None = None,
        sub_key: str | None = None,
    ) -> None:
        self.plugin = validate_plugin(plugin)
        self.sub = validate_sub(sub) if sub is not None else None
        leaf = sub_key or "config"
        self.cred_key = f"{plugin}/subs/{sub}/{leaf}" if self.sub else f"{plugin}/{leaf}"
        self._backend = backend if backend is not None else backend_from_env()
        # Legacy layouts exist only on local disk under the default layout; an
        # injected in-memory/KV/custom-dir backend is the single source of truth.
        self._reads_legacy = isinstance(self._backend, LocalFsBackend) and self._backend.is_default_layout

    @property
    def backend(self) -> CredentialBackend:
        return self._backend

    @property
    def cred_path(self) -> Path | None:
        """On-disk path of the primary blob, or ``None`` for a non-local backend."""
        if isinstance(self._backend, LocalFsBackend):
            return self._backend.path_for(self.cred_key)
        return None

    def _sources(self) -> list[_BlobSource]:
        primary = _BlobSource(self._backend, None)
        if not self._reads_legacy:
            return [primary]
        return [primary] + [
            _BlobSource(LocalFsBackend(base_dir=candidate.dir), candidate)
            for candidate in legacy_data_dirs(self.plugin)
        ]

    def _key(self) -> bytes:
        if self.sub:
            return derive_multi_user_key(self.plugin, self.sub)
        return _load_or_generate_machine_key(data_dir(self.plugin))

    def _legacy_key(self, candidate: DataDirCandidate) -> bytes | None:
        if self.sub:
            return derive_legacy_multi_user_key(candidate.key_plugin, self.sub)
        return _read_legacy_machine_key(candidate.dir)

    def _report_corrupt(self, legacy: DataDirCandidate | None) -> None:
        where = f" (legacy dir {legacy.dir})" if legacy else ""
        print(
            f"[hull PerPluginStore] Credential blob for {self.cred_key}{where} is corrupt or the encryption key "
            "changed; treating as not configured (re-run setup to restore)",
            file=sys.stderr,
        )

    def _decrypt(self, blob: bytes, legacy: DataDirCandidate | None) -> dict[str, Any] | None:
        if len(blob) < _MIN_BLOB_LEN:
            self._report_corrupt(legacy)
            return None
        key = self._legacy_key(legacy) if legacy else self._key()
        if key is None:
            self._report_corrupt(legacy)
            return None
        try:
            # AESGCM() sits inside the try: a legacy .secret of the wrong length
            # is unusable key material, i.e. the same "corrupt" outcome.
            plaintext = AESGCM(key).decrypt(blob[:_NONCE_LEN], blob[_NONCE_LEN:], None)
            payload = json.loads(plaintext.decode("utf-8"))
        except Exception:
            self._report_corrupt(legacy)
            return None
        if not isinstance(payload, dict):
            self._report_corrupt(legacy)
            return None
        return payload

    def load(self) -> dict[str, Any] | None:
        """Decrypt the payload; the first location holding a blob decides the result.

        A corrupt blob there yields ``None`` rather than falling through to an
        older location, so a stale legacy credential is never resurrected
        behind a newer (damaged) one.
        """
        for source in self._sources():
            blob = source.backend.get(self.cred_key)
            if blob is not None:
                return self._decrypt(blob, source.legacy)
        return None

    def save(self, payload: dict[str, Any]) -> None:
        """Encrypt and write to the primary backend; legacy copies are kept."""
        nonce = secrets.token_bytes(_NONCE_LEN)
        ciphertext = AESGCM(self._key()).encrypt(nonce, json.dumps(payload).encode("utf-8"), None)
        self._backend.put(self.cred_key, nonce + ciphertext)

    def clear(self) -> None:
        """Delete the blob from the primary backend and every legacy location."""
        for source in self._sources():
            source.backend.delete(self.cred_key)

    def has_stored_blob(self) -> bool:
        """Whether any read location holds a blob (decryptable or not); never decrypts."""
        return any(source.backend.get(self.cred_key) is not None for source in self._sources())

    def blob_locations(self) -> list[Path]:
        """Local paths currently holding a blob (empty for non-local backends)."""
        found: list[Path] = []
        for source in self._sources():
            backend = source.backend
            if isinstance(backend, LocalFsBackend) and backend.get(self.cred_key) is not None:
                found.append(backend.path_for(self.cred_key))
        return found
