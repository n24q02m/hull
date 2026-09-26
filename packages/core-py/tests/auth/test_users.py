"""users.toml parsing, validation, token lookup, allowed_roots enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from hull_core.auth.tokens import hash_token
from hull_core.auth.users import (
    UsersError,
    ensure_path_allowed,
    find_user_by_token,
    load_users,
)
from hull_core.auth.context import AuthContext

ALICE_HASH = hash_token("alice-token")
BOB_HASH = hash_token("bob-token")

USERS_TOML = f"""\
[users.alice]
token_hash = "{ALICE_HASH}"
enabled = true
namespace = "alice"
allowed_roots = ["/data/alice"]

[users.alice.limits]
rpm = 2

[users.bob]
token_hash = "{BOB_HASH}"
enabled = false
namespace = "bob"
"""


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "users.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_users_parses_table(tmp_path: Path) -> None:
    users = load_users(_write(tmp_path, USERS_TOML))
    assert set(users) == {"alice", "bob"}
    alice = users["alice"]
    assert alice.namespace == "alice"
    assert alice.enabled is True
    assert alice.allowed_roots == ("/data/alice",)
    assert alice.limits.rpm == 2
    assert users["bob"].enabled is False
    assert users["bob"].limits.rpm is None


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UsersError, match="not found"):
        load_users(tmp_path / "absent.toml")


def test_missing_users_table_raises(tmp_path: Path) -> None:
    with pytest.raises(UsersError, match=r"\[users\]"):
        load_users(_write(tmp_path, "[other]\nx = 1\n"))


def test_bad_token_hash_raises(tmp_path: Path) -> None:
    content = '[users.eve]\ntoken_hash = "plaintext"\nenabled = true\nnamespace = "eve"\n'
    with pytest.raises(UsersError, match="token_hash"):
        load_users(_write(tmp_path, content))


def test_missing_namespace_raises(tmp_path: Path) -> None:
    content = f'[users.eve]\ntoken_hash = "{hash_token("t")}"\nenabled = true\n'
    with pytest.raises(UsersError, match="namespace"):
        load_users(_write(tmp_path, content))


def test_find_user_by_token(tmp_path: Path) -> None:
    users = load_users(_write(tmp_path, USERS_TOML))
    assert find_user_by_token(users, "alice-token") is users["alice"]
    assert find_user_by_token(users, "nope") is None


def test_ensure_path_allowed(tmp_path: Path) -> None:
    root = tmp_path / "roots" / "alice"
    root.mkdir(parents=True)
    user = AuthContext(uid="alice", namespace="alice", mode="multi", allowed_roots=(str(root),))
    ensure_path_allowed(user, root / "file.txt")
    ensure_path_allowed(user, root)
    with pytest.raises(PermissionError):
        ensure_path_allowed(user, tmp_path / "outside.txt")


def test_empty_allowed_roots_grants_nothing(tmp_path: Path) -> None:
    user = AuthContext(uid="nobody", namespace="n", mode="multi")
    with pytest.raises(PermissionError):
        ensure_path_allowed(user, tmp_path)
