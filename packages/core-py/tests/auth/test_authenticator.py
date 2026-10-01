"""Authenticator mode matrix: 401 wrong token, 403 disabled, 429 over limit."""

from __future__ import annotations

from pathlib import Path

import pytest

from hull_core.auth.context import AuthContext
from hull_core.auth.middleware import Authenticator
from hull_core.auth.tokens import hash_token
from hull_core.auth.users import load_users
from hull_core.config.settings import HullSettings, ServerSettings
from hull_core.limits.limiter import SlidingWindowLimiter

ALICE_HASH = hash_token("alice-token")
BOB_HASH = hash_token("bob-token")

USERS_TOML = f"""\
[users.alice]
token_hash = "{ALICE_HASH}"
enabled = true
namespace = "alice"

[users.alice.limits]
rpm = 2

[users.bob]
token_hash = "{BOB_HASH}"
enabled = false
namespace = "bob"
"""


def _settings(tmp_path: Path, mode: str, **kw) -> HullSettings:
    return HullSettings(
        config_dir=tmp_path,
        server=ServerSettings(auth=mode, **kw),
    )


def _multi(tmp_path: Path) -> tuple[HullSettings, dict]:
    users_file = tmp_path / "users.toml"
    users_file.write_text(USERS_TOML, encoding="utf-8")
    settings = _settings(tmp_path, "multi", users_file=users_file)
    return settings, load_users(users_file)


def test_no_auth_admits_everything(tmp_path: Path) -> None:
    outcome = Authenticator(_settings(tmp_path, "no-auth")).authenticate(None)
    assert outcome.ok
    assert outcome.context == AuthContext.local()
    assert outcome.context.namespace == "default"


def test_token_mode_wrong_token_401(tmp_path: Path) -> None:
    auth = Authenticator(_settings(tmp_path, "token", token_hash=hash_token("real")))
    assert auth.authenticate("Bearer wrong").status == 401
    assert auth.authenticate(None).status == 401
    assert auth.authenticate("Basic dXNlcjpwYXNz").status == 401


def test_token_mode_correct_token_shared_namespace(tmp_path: Path) -> None:
    auth = Authenticator(_settings(tmp_path, "token", token_hash=hash_token("real")))
    outcome = auth.authenticate("Bearer real")
    assert outcome.ok
    assert outcome.context is not None
    assert outcome.context.namespace == "default"
    assert outcome.context.mode == "token"


def test_multi_unknown_token_401(tmp_path: Path) -> None:
    settings, users = _multi(tmp_path)
    outcome = Authenticator(settings, users=users).authenticate("Bearer ghost")
    assert outcome.status == 401


def test_multi_disabled_user_403(tmp_path: Path) -> None:
    settings, users = _multi(tmp_path)
    outcome = Authenticator(settings, users=users).authenticate("Bearer bob-token")
    assert outcome.status == 403


def test_multi_valid_user_namespace(tmp_path: Path) -> None:
    settings, users = _multi(tmp_path)
    outcome = Authenticator(settings, users=users).authenticate("Bearer alice-token")
    assert outcome.ok
    assert outcome.context is not None
    assert outcome.context.uid == "alice"
    assert outcome.context.namespace == "alice"


def test_multi_rpm_limit_429(tmp_path: Path) -> None:
    settings, users = _multi(tmp_path)
    auth = Authenticator(settings, users=users, limiter=SlidingWindowLimiter())
    assert auth.authenticate("Bearer alice-token").ok
    assert auth.authenticate("Bearer alice-token").ok
    assert auth.authenticate("Bearer alice-token").status == 429
    # bob (disabled) is still 403 — limiter for alice doesn't leak across users
    assert auth.authenticate("Bearer bob-token").status == 403


def test_multi_requires_users_table(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "multi")
    with pytest.raises(Exception, match="users"):
        Authenticator(settings).authenticate("Bearer alice-token")
