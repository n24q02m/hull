"""Credential secret envelope: round-trip, shared vector, key-file safety."""

from __future__ import annotations

import base64

import pytest

from hull_core.credentials.crypto import (
    KEY_LENGTH,
    CredentialCryptoError,
    decrypt_secret,
    encrypt_secret,
    key_fingerprint,
    key_hint,
    load_or_create_key,
)

VECTOR_KEY = bytes(range(32))
VECTOR_NONCE = bytes(range(12))
VECTOR_PLAINTEXT = "sk-hull-vector-" + "0123456789"
VECTOR_ENVELOPE = "v1:AAECAwQFBgcICQoL:NGn7c7CJrjb7JPT/3ptVXbLktADFTWhEAcYamPqU91OPf2S7BjLOftk="


def test_roundtrip_uses_versioned_envelope() -> None:
    envelope = encrypt_secret(VECTOR_KEY, "sk-roundtrip-" + "1234")
    prefix, nonce, payload = envelope.split(":")
    assert prefix == "v1"
    assert len(base64.b64decode(nonce)) == 12
    assert decrypt_secret(VECTOR_KEY, envelope) == "sk-roundtrip-1234"
    # ciphertext + 16-byte tag
    assert len(base64.b64decode(payload)) > 16


def test_shared_vector_matches_typescript() -> None:
    # packages/core-ts/tests/storage/key-pool-crypto.test.ts pins the same
    # bytes, and encrypts them to the same envelope: either stack can open a
    # pool the other one wrote.
    assert encrypt_secret(VECTOR_KEY, VECTOR_PLAINTEXT, nonce=VECTOR_NONCE) == VECTOR_ENVELOPE
    assert decrypt_secret(VECTOR_KEY, VECTOR_ENVELOPE) == VECTOR_PLAINTEXT


@pytest.mark.parametrize(
    "envelope",
    [
        "not-an-envelope",
        "v2:AAECAwQFBgcICQoL:NGn7c7CJrjb7JPT/3ptVXbLktADFTWhEAcYamPqU91OPf2S7BjLOftk=",
        "v1:@@@:@@@",
        "v1:AAECAwQFBgcICQoL:AAAA",
    ],
)
def test_malformed_envelopes_raise_without_echoing(envelope: str) -> None:
    with pytest.raises(CredentialCryptoError) as excinfo:
        decrypt_secret(VECTOR_KEY, envelope)
    assert VECTOR_PLAINTEXT not in str(excinfo.value)
    assert envelope not in str(excinfo.value)


def test_wrong_key_is_reported_as_corrupt() -> None:
    envelope = encrypt_secret(VECTOR_KEY, VECTOR_PLAINTEXT)
    with pytest.raises(CredentialCryptoError, match="corrupt or the key changed") as excinfo:
        decrypt_secret(bytes(32), envelope)
    assert VECTOR_PLAINTEXT not in str(excinfo.value)


def test_bad_key_length_and_nonce_rejected() -> None:
    with pytest.raises(CredentialCryptoError, match="not a hull credential key"):
        encrypt_secret(b"short", "value")
    with pytest.raises(ValueError, match="nonce must be exactly 12 bytes"):
        encrypt_secret(VECTOR_KEY, "value", nonce=b"short")


def test_key_file_is_created_once_and_stays_32_bytes(tmp_path) -> None:
    path = tmp_path / "nested" / ".credential-key"
    first = load_or_create_key(path)
    second = load_or_create_key(path)
    assert first == second
    assert len(first) == KEY_LENGTH
    assert path.read_bytes() == first


def test_foreign_key_file_is_never_replaced(tmp_path) -> None:
    path = tmp_path / ".credential-key"
    path.write_bytes(b"0123456789abcdef")  # 16 bytes, e.g. another tool's key
    with pytest.raises(CredentialCryptoError, match="will not overwrite"):
        load_or_create_key(path)
    assert path.read_bytes() == b"0123456789abcdef"


def test_hint_and_fingerprint_never_expose_the_key() -> None:
    api_key = "sk-live-super-secret-abcd"
    assert key_hint(api_key) == "…abcd"
    assert api_key not in key_hint(api_key)
    assert key_hint("short") == "…"
    assert key_fingerprint(api_key) == key_fingerprint(api_key)
    assert api_key not in key_fingerprint(api_key)
    assert len(key_fingerprint(api_key)) == 64
