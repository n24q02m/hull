"""scrypt token hashing round-trips and malformed-encoding rejection."""

from __future__ import annotations

import json
from pathlib import Path

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


# Hashes minted by core-ts (@n24q02m/hull token-auth); regenerate with
# `bun scripts/gen-token-hash-vectors.ts` in packages/core-ts.
_TS_VECTORS = json.loads((Path(__file__).parent / "fixtures" / "ts_token_hash_vectors.json").read_text("utf-8"))


@pytest.mark.parametrize("case", _TS_VECTORS["valid"], ids=lambda c: c["name"])
def test_verifies_ts_minted_hash(case: dict[str, str]) -> None:
    assert verify_token(case["token"], case["encoded"])
    assert is_valid_encoding(case["encoded"])


@pytest.mark.parametrize("case", _TS_VECTORS["wrong_token"], ids=lambda c: c["name"])
def test_rejects_wrong_token_for_ts_minted_hash(case: dict[str, str]) -> None:
    assert not verify_token(case["token"], case["encoded"])


def test_ts_vector_set_is_complete() -> None:
    names = {c["name"] for c in _TS_VECTORS["valid"]}
    assert {"ascii", "empty", "unicode", "long", "n1024-r4-p2", "dklen16", "dklen64"} <= names
    assert len(_TS_VECTORS["wrong_token"]) == len(_TS_VECTORS["valid"])
