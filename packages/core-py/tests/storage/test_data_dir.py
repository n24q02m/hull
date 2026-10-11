"""Data directory layout: validation, current + legacy candidates, blob paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from hull_core.storage.data_dir import (
    DataDirCandidate,
    blob_path_in,
    data_dir,
    data_dir_candidates,
    legacy_data_dirs,
    validate_plugin,
    validate_sub,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
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


def test_data_dir_is_dot_plugin_under_home(home: Path) -> None:
    assert data_dir("telegram") == home / ".telegram"


def test_data_dir_validates(home: Path) -> None:
    with pytest.raises(ValueError):
        data_dir("../x")


def test_candidates_in_read_precedence_order(home: Path) -> None:
    assert data_dir_candidates("telegram") == [
        DataDirCandidate(dir=home / ".telegram", key_plugin="telegram", legacy=False),
        DataDirCandidate(dir=home / ".telegram-mcp", key_plugin="telegram", legacy=True),
        DataDirCandidate(dir=home / ".better-telegram-mcp", key_plugin="better-telegram", legacy=True),
    ]


def test_candidates_validate(home: Path) -> None:
    with pytest.raises(ValueError):
        data_dir_candidates("a/b")


def test_legacy_dirs_only(home: Path) -> None:
    assert [c.dir.name for c in legacy_data_dirs("telegram")] == [".telegram-mcp", ".better-telegram-mcp"]
    assert all(c.legacy for c in legacy_data_dirs("telegram"))


def test_blob_path_in(tmp_path: Path) -> None:
    assert blob_path_in(tmp_path) == tmp_path / "config.json"
    assert blob_path_in(tmp_path, "u1") == tmp_path / "subs" / "u1" / "config.json"
    with pytest.raises(ValueError, match="Invalid sub"):
        blob_path_in(tmp_path, "../u1")
