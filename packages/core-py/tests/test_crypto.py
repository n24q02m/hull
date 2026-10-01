"""Constant-time comparison: equality by value, no early exit on length."""

from __future__ import annotations

import pytest

from hull_core.crypto import timing_safe_equal


def test_identical_bytes_match() -> None:
    assert timing_safe_equal(b"secret-token", b"secret-token") is True


def test_same_length_mismatch_is_rejected() -> None:
    assert timing_safe_equal(b"secret-token", b"secret-tokes") is False


def test_length_mismatch_is_rejected() -> None:
    """The length leak is the whole reason this helper exists: never equal."""
    assert timing_safe_equal("", "x") is False, "empty is not a wildcard prefix"
    assert timing_safe_equal(b"short", b"a-much-longer-secret") is False


def test_str_inputs_are_compared_as_utf8() -> None:
    assert timing_safe_equal("mật-khẩu", "mật-khẩu") is True
    assert timing_safe_equal("mật-khẩu", "mật-khẩ") is False


def test_str_and_bytes_of_the_same_value_match() -> None:
    assert timing_safe_equal("plain-token", b"plain-token") is True
    assert timing_safe_equal(b"plain-token", "plain-token") is True


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param("", "", id="both-empty"),
        pytest.param("héllo", b"h\xc3\xa9llo", id="utf8-bytes"),
    ],
)
def test_mixed_encodings_agree(left: bytes | str, right: bytes | str) -> None:
    assert timing_safe_equal(left, right) is True
