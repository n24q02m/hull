"""Token hashing and verification (host-side credential material).

Tokens never travel to disk in plaintext: the host stores a scrypt hash in
``users.toml`` / ``config.toml`` and verifies presented bearer tokens against
it. Format::

    scrypt$<n>$<r>$<p>$<salt-hex>$<derived-key-hex>
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32
_PREFIX = "scrypt"
# 128 * N * r = 32 MiB working memory; OpenSSL's default cap is lower on some
# builds, so grant an explicit 64 MiB ceiling.
_MAXMEM = 64 * 1024 * 1024


class TokenHashError(ValueError):
    """Raised when an encoded token hash is malformed."""


def hash_token(token: str) -> str:
    """Hash a bearer token into the storable ``scrypt$...`` encoding."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        token.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
        maxmem=_MAXMEM,
    )
    return "$".join(
        [
            _PREFIX,
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            salt.hex(),
            dk.hex(),
        ]
    )


def verify_token(token: str, encoded: str) -> bool:
    """Constant-time verify a presented token against an encoded hash."""
    try:
        prefix, n_s, r_s, p_s, salt_hex, dk_hex = encoded.split("$")
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
    except (ValueError, AttributeError) as exc:
        raise TokenHashError(f"malformed token hash encoding: {encoded[:24]!r}") from exc
    if prefix != _PREFIX or not salt or not expected:
        raise TokenHashError(f"malformed token hash encoding: {encoded[:24]!r}")
    actual = hashlib.scrypt(
        token.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected),
        maxmem=_MAXMEM,
    )
    return hmac.compare_digest(actual, expected)


def is_valid_encoding(encoded: str) -> bool:
    """True when ``encoded`` parses as a scrypt token hash."""
    try:
        verify_token("", encoded)
    except TokenHashError:
        return False
    return True
