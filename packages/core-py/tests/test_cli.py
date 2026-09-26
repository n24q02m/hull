"""CLI: token hashing roundtrip + config init behavior."""

from __future__ import annotations

from pathlib import Path

from hull_core.auth.tokens import verify_token
from hull_core.cli import main


def test_token_hash_outputs_verifiable_encoding(capsys) -> None:  # noqa: ANN001
    assert main(["token", "hash", "my-plain-token"]) == 0
    encoded = capsys.readouterr().out.strip()
    assert verify_token("my-plain-token", encoded)


def test_config_init_writes_template(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("hull_core.config.settings.default_config_dir", lambda: tmp_path)
    assert main(["config", "init"]) == 0
    assert (tmp_path / "config.toml").is_file()
    assert main(["config", "init"]) == 2  # refuses clobber
    assert main(["config", "init", "--force"]) == 0


def test_config_path_prints_location(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr("hull_core.config.settings.default_config_dir", lambda: tmp_path)
    assert main(["config", "path"]) == 0
    assert "config.toml" in capsys.readouterr().out


def test_db_path_prints_location(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr("hull_core.config.settings.default_config_dir", lambda: tmp_path)
    assert main(["db", "path"]) == 0
    assert "hull.db" in capsys.readouterr().out
