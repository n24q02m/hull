"""CLI: token hashing roundtrip + config init behavior."""

from __future__ import annotations

from pathlib import Path

from hull_core.auth.tokens import verify_token
from hull_core.cli import main
from hull_core.server.run import ServerConfigError


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
    monkeypatch.setattr("hull_core.cli.default_config_dir", lambda: tmp_path)
    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "config.toml")


def test_config_show_prints_template_when_file_absent(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr("hull_core.cli.default_config_dir", lambda: tmp_path)
    assert main(["config", "show"]) == 0
    assert 'auth = "no-auth"' in capsys.readouterr().out


def test_config_show_prefers_the_file_on_disk(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    (tmp_path / "config.toml").write_text("[server]\nport = 9999\n", encoding="utf-8")
    monkeypatch.setattr("hull_core.cli.default_config_dir", lambda: tmp_path)
    assert main(["config", "show"]) == 0
    assert capsys.readouterr().out == "[server]\nport = 9999\n"


def test_server_start_forwards_bind_args_and_swallows_interrupt(monkeypatch) -> None:  # noqa: ANN001
    seen: dict = {}

    def fake_run(**kwargs):  # noqa: ANN003
        seen.update(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr("hull_core.cli.run_server", fake_run)
    assert main(["server", "start", "--host", "127.0.0.1", "--port", "9123"]) == 0
    assert seen == {"host": "127.0.0.1", "port": 9123}


def test_server_start_reports_config_error_as_exit_2(monkeypatch, capsys) -> None:  # noqa: ANN001
    def fake_run(**_kwargs):  # noqa: ANN003
        raise ServerConfigError("auth = 'no-auth' only permits loopback binds")

    monkeypatch.setattr("hull_core.cli.run_server", fake_run)
    assert main(["server", "start", "--host", "0.0.0.0"]) == 2
    assert "only permits loopback" in capsys.readouterr().err


def test_db_path_prints_location(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.setattr("hull_core.config.settings.default_config_dir", lambda: tmp_path)
    assert main(["db", "path"]) == 0
    assert "hull.db" in capsys.readouterr().out
