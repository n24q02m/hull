"""Settings loading: defaults, mode validation, template write."""

from __future__ import annotations

from pathlib import Path

import pytest

from hull_core.config.settings import (
    CONFIG_TEMPLATE,
    ConfigError,
    default_config_dir,
    load_settings,
    write_default_config,
)


def test_defaults_when_config_absent(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    assert settings.server.auth == "no-auth"
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8000
    assert settings.server.token_hash is None
    assert settings.server.users_file is None


def test_invalid_auth_mode_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[server]\nauth = "oauth-magic"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="auth"):
        load_settings(tmp_path)


def test_token_mode_requires_hash(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[server]\nauth = "token"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="token_hash"):
        load_settings(tmp_path)


def test_token_mode_with_hash(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[server]\nauth = "token"\ntoken_hash = "scrypt$1$1$1$ab$cd"\n', encoding="utf-8"
    )
    assert load_settings(tmp_path).server.token_hash == "scrypt$1$1$1$ab$cd"


def test_multi_mode_defaults_users_file(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[server]\nauth = "multi"\n', encoding="utf-8")
    assert load_settings(tmp_path).server.users_file == tmp_path / "users.toml"


def test_write_default_config_roundtrip(tmp_path: Path) -> None:
    path = write_default_config(tmp_path)
    assert path == tmp_path / "config.toml"
    assert CONFIG_TEMPLATE in path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_default_config(tmp_path)
    write_default_config(tmp_path, force=True)
    settings = load_settings(tmp_path)
    assert settings.server.auth == "no-auth"
    assert set(settings.models) >= {"embed", "rerank", "chat", "jev_score"}


def test_invalid_toml_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("not [valid toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="TOML"):
        load_settings(tmp_path)


def test_default_config_dir_under_home() -> None:
    assert default_config_dir().name == ".hull"
