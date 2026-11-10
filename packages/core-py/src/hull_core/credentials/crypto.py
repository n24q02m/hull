"""Encryption at rest for pooled provider credentials.

The provider pool lives in the local ``~/.hull/hull.db`` SQLite database, so the
API keys must not sit there as plaintext. Every key is sealed with AES-256-GCM
under a 32-byte machine key at ``~/.hull/.credential-key`` (created 0600 on
first use, never overwritten). Envelope::

    v1:<base64 nonce>:<base64 ciphertext || 16-byte tag>

``packages/core-ts/src/storage/key-pool-crypto.ts`` writes and reads the same
envelope with the same key file, so either stack can open a pool the other
created; both test suites pin one shared vector.

Only the envelope ever reaches the database. The key hint (last four
characters) and the keyed HMAC-SHA256 fingerprint are derived values stored alongside it
for display and duplicate detection; neither can be turned back into the key.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_LENGTH = 32
NONCE_LENGTH = 12
_ENVELOPE_PREFIX = "v1"
_TAG_LENGTH = 16


class CredentialCryptoError(RuntimeError):
    """The credential key is unusable or an envelope cannot be decrypted."""


def _default_key_path() -> Path:
    # Imported lazily so tests that monkeypatch ``default_config_dir`` (and
    # callers that pass an explicit path) are honoured.
    from hull_core.config.settings import default_config_dir

    return default_config_dir() / ".credential-key"


def _checked_key(path: Path, key: bytes) -> bytes:
    if len(key) != KEY_LENGTH:
        raise CredentialCryptoError(
            f"{path} is not a hull credential key (expected {KEY_LENGTH} raw bytes, found {len(key)}); "
            "hull will not overwrite or delete it. Move it aside to mint a fresh key, then re-add the "
            "provider credentials — the old envelopes cannot be recovered without the original key."
        )
    return key


def load_or_create_key(path: Path | None = None) -> bytes:
    """Read the 32-byte credential key, minting it 0600 on first use.

    Creation is exclusive (``O_CREAT | O_EXCL``): two processes racing on the
    first use converge on the winner's key instead of the later write winning.
    A pre-existing file of any other length is refused, never replaced.
    """
    path = Path(path) if path is not None else _default_key_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError:
        return _checked_key(path, path.read_bytes())
    key = secrets.token_bytes(KEY_LENGTH)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def encrypt_secret(key: bytes, plaintext: str, *, nonce: bytes | None = None) -> str:
    """Seal ``plaintext`` into a ``v1:<nonce>:<ciphertext+tag>`` envelope."""
    _checked_key(Path("<memory>"), key)
    nonce = secrets.token_bytes(NONCE_LENGTH) if nonce is None else nonce
    if len(nonce) != NONCE_LENGTH:
        raise ValueError(f"nonce must be exactly {NONCE_LENGTH} bytes")
    sealed = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return ":".join(
        (
            _ENVELOPE_PREFIX,
            base64.b64encode(nonce).decode("ascii"),
            base64.b64encode(sealed).decode("ascii"),
        )
    )


def decrypt_secret(key: bytes, envelope: str) -> str:
    """Open an envelope produced by :func:`encrypt_secret` (never echoes it)."""
    _checked_key(Path("<memory>"), key)
    parts = envelope.split(":")
    if len(parts) != 3 or parts[0] != _ENVELOPE_PREFIX:
        raise CredentialCryptoError("credential secret is not a v1 hull envelope")
    try:
        nonce = base64.b64decode(parts[1], validate=True)
        sealed = base64.b64decode(parts[2], validate=True)
    except (binascii.Error, ValueError):
        raise CredentialCryptoError("credential secret envelope is corrupt") from None
    if len(nonce) != NONCE_LENGTH or len(sealed) <= _TAG_LENGTH:
        raise CredentialCryptoError("credential secret envelope is corrupt")
    try:
        return AESGCM(key).decrypt(nonce, sealed, None).decode("utf-8")
    except Exception:
        # InvalidTag / UnicodeDecodeError: a wrong key or a damaged row. The
        # message deliberately carries neither the envelope nor the plaintext.
        raise CredentialCryptoError("credential secret is corrupt or the key changed") from None


def key_hint(api_key: str) -> str:
    """Display-only tail of a key (``…abcd``); never enough to reconstruct it."""
    return "…" + api_key[-4:] if len(api_key) >= 8 else "…"


def key_fingerprint(key: bytes, api_key: str) -> str:
    """Keyed HMAC-SHA256 fingerprint for duplicate detection, never for auth.

    Unkeyed digests of a provider key are offline-crackable and leak equality
    across machines; HMAC under the local ``.credential-key`` keeps duplicate
    detection while making the digest useless without that key.
    """
    _checked_key(Path("<memory>"), key)
    return hmac.new(key, api_key.encode("utf-8"), hashlib.sha256).hexdigest()


class SecretCipher:
    """Lazy key-file holder used by :class:`~hull_core.credentials.store.CredentialStore`."""

    def __init__(self, key_path: Path | None = None) -> None:
        self.key_path = Path(key_path) if key_path is not None else _default_key_path()
        self._key: bytes | None = None

    @property
    def key(self) -> bytes:
        if self._key is None:
            self._key = load_or_create_key(self.key_path)
        return self._key

    def fingerprint(self, plaintext: str) -> str:
        return key_fingerprint(self.key, plaintext)

    def encrypt(self, plaintext: str) -> str:
        return encrypt_secret(self.key, plaintext)

    def decrypt(self, envelope: str) -> str:
        return decrypt_secret(self.key, envelope)
