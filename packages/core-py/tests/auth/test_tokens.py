"""scrypt token hashing round-trips and malformed-encoding rejection."""

from __future__ import annotations

import pytest

from hull_core.auth.tokens import TokenHashError, hash_token, is_valid_encoding, verify_token


def test_hash_verify_roundtrip() -> None:
    encoded = hash_token("s3cret-token")
    assert encoded.startswith("scrypt$")
    assert verify_token("s3cret-token", encoded)
    assert not verify_token("wrong", encoded)


def test_unique_salts_same_token() -> None:
    assert hash_token("same") != hash_token("same")


def test_malformed_encoding_raises() -> None:
    with pytest.raises(TokenHashError):
        verify_token("x", "not-a-hash")
    with pytest.raises(TokenHashError):
        verify_token("x", "scrypt$bad$bits")


def test_is_valid_encoding() -> None:
    assert is_valid_encoding(hash_token("t"))
    assert not is_valid_encoding("scrypt$x$y")
    assert not is_valid_encoding("plaintext")
