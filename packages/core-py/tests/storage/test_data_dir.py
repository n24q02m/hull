"""Data directory layout: validation, current + legacy candidates, profiles, migration."""

from __future__ import annotations

from pathlib import Path

import pytest

from hull_core.storage.data_dir import (
    DataDirCandidate,
    blob_path_in,
    data_dir,
    data_dir_candidates,
    legacy_data_dirs,
    migrate_legacy_data_dir,
    n24q02m_base_dir,
    resolve_profile,
    validate_plugin,
    validate_profile,
    validate_sub,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("N24Q02M_HOME", raising=False)
    monkeypatch.delenv("N24Q02M_USE_XDG", raising=False)
    monkeypatch.delenv("HULL_PROFILE", raising=False)
    monkeypatch.delenv("TELEGRAM_PROFILE", raising=False)
    return tmp_path


@pytest.mark.parametrize("name", ["telegram", "better-telegram", "a.b_c-1", "X"])
def test_validate_plugin_accepts_safe_names(name: str) -> None:
    assert validate_plugin(name) == name


@pytest.mark.parametrize("name", ["", ".", "..", "a..b", "a/b", "a\\b", "a b", "a:b", "tele\x00gram"])
def test_validate_plugin_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid plugin name"):
        validate_plugin(name)


def test_validate_plugin_rejects_non_str() -> None:
    with pytest.raises(ValueError, match="Invalid plugin name"):
        validate_plugin(123)  # ty: ignore[invalid-argument-type]


def test_validate_sub_accepts_token_urlsafe_alphabet() -> None:
    assert validate_sub("Ab-_09xyz") == "Ab-_09xyz"


@pytest.mark.parametrize("sub", ["", "..", "../etc", "a/b", "a b"])
def test_validate_sub_rejects_unsafe(sub: str) -> None:
    with pytest.raises(ValueError, match="Invalid sub"):
        validate_sub(sub)


def test_validate_sub_rejects_non_str() -> None:
    with pytest.raises(ValueError, match="Invalid sub"):
        validate_sub(None)  # ty: ignore[invalid-argument-type]


def test_validate_profile_accepts_and_rejects() -> None:
    assert validate_profile("work-1_a.b") == "work-1_a.b"
    for bad in ("", ".", "..", "a..b", "a/b", "a b"):
        with pytest.raises(ValueError, match="Invalid profile"):
            validate_profile(bad)
    with pytest.raises(ValueError, match="Invalid profile"):
        validate_profile(42)  # ty: ignore[invalid-argument-type]


def test_resolve_profile_precedence(home: Path) -> None:
    env = {"TELEGRAM_PROFILE": "from-plugin", "HULL_PROFILE": "from-hull"}
    assert resolve_profile("telegram", "explicit", env=env) == "explicit"
    assert resolve_profile("telegram", None, env=env) == "from-plugin"
    assert resolve_profile("telegram", "", env={"HULL_PROFILE": "from-hull"}) == "from-hull"
    assert resolve_profile("telegram", None, env={}) == "default"
    with pytest.raises(ValueError, match="Invalid profile"):
        resolve_profile("telegram", None, env={"HULL_PROFILE": "../evil"})


def test_n24q02m_base_dir_overrides(home: Path) -> None:
    assert n24q02m_base_dir({}) == home / ".n24q02m"
    assert n24q02m_base_dir({"N24Q02M_HOME": str(home / "custom-home")}) == home / "custom-home"
    assert n24q02m_base_dir({"N24Q02M_USE_XDG": "1", "XDG_CONFIG_HOME": str(home / "xdg")}) == home / "xdg" / "n24q02m"
    assert (
        n24q02m_base_dir({"N24Q02M_USE_XDG": "true", "APPDATA": str(home / "appdata")}) == home / "appdata" / "n24q02m"
    )
    assert n24q02m_base_dir({"N24Q02M_USE_XDG": "1"}) == home / ".n24q02m"


def test_data_dir_is_under_n24q02m_home(home: Path) -> None:
    assert data_dir("telegram") == home / ".n24q02m" / "telegram"


def test_data_dir_validates(home: Path) -> None:
    with pytest.raises(ValueError):
        data_dir("../x")


def test_candidates_in_read_precedence_order(home: Path) -> None:
    assert data_dir_candidates("telegram") == [
        DataDirCandidate(dir=home / ".n24q02m" / "telegram", key_plugin="telegram", legacy=False),
        DataDirCandidate(dir=home / ".telegram", key_plugin="telegram", legacy=True),
        DataDirCandidate(dir=home / ".telegram-mcp", key_plugin="telegram", legacy=True),
        DataDirCandidate(dir=home / ".better-telegram-mcp", key_plugin="better-telegram", legacy=True),
    ]


def test_candidates_validate(home: Path) -> None:
    with pytest.raises(ValueError):
        data_dir_candidates("a/b")


def test_legacy_dirs_only(home: Path) -> None:
    assert [c.dir.name for c in legacy_data_dirs("telegram")] == [
        ".telegram",
        ".telegram-mcp",
        ".better-telegram-mcp",
    ]
    assert all(c.legacy for c in legacy_data_dirs("telegram"))


def test_blob_path_in(tmp_path: Path) -> None:
    assert blob_path_in(tmp_path) == tmp_path / "config.json"
    assert blob_path_in(tmp_path, profile="default") == tmp_path / "config.json"
    assert blob_path_in(tmp_path, profile="work") == tmp_path / "profiles" / "work" / "config.json"
    assert blob_path_in(tmp_path, "u1") == tmp_path / "subs" / "u1" / "config.json"
    assert (
        blob_path_in(tmp_path, "u1", profile="work") == tmp_path / "subs" / "u1" / "profiles" / "work" / "config.json"
    )
    with pytest.raises(ValueError, match="Invalid sub"):
        blob_path_in(tmp_path, "../u1")
    with pytest.raises(ValueError, match="Invalid profile"):
        blob_path_in(tmp_path, profile="../bad")


def test_migrate_legacy_data_dir_non_destructive(home: Path) -> None:
    assert migrate_legacy_data_dir("telegram") is False
    legacy = home / ".telegram"
    (legacy / "subs" / "u1").mkdir(parents=True)
    (legacy / ".secret").write_bytes(b"s" * 32)
    (legacy / "config.json").write_bytes(b"c1")
    (legacy / "subs" / "u1" / "config.json").write_bytes(b"c2")

    assert migrate_legacy_data_dir("telegram") is True
    target = home / ".n24q02m" / "telegram"
    assert (target / ".secret").read_bytes() == b"s" * 32
    assert (target / "config.json").read_bytes() == b"c1"
    assert (target / "subs" / "u1" / "config.json").read_bytes() == b"c2"
    # Legacy directory remains untouched.
    assert (legacy / "config.json").read_bytes() == b"c1"
    # Second invocation is a no-op because target now exists.
    assert migrate_legacy_data_dir("telegram") is False
