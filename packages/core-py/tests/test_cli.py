"""CLI: control-plane commands, help, error surface, token hashing."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from hull_core import __version__
from hull_core.auth.tokens import hash_token, verify_token
from hull_core.cli import main
from hull_core.server.run import ServerConfigError


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _stdin(monkeypatch: pytest.MonkeyPatch, text: str, *, tty: bool = False) -> None:
    monkeypatch.setattr(sys, "stdin", _Tty(text) if tty else io.StringIO(text))


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI (the one resolution point) at a throwaway config dir."""
    monkeypatch.setattr("hull_core.cli.default_config_dir", lambda: tmp_path)
    return tmp_path


# --- help and version -------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_lists_subcommands(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([flag])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for line in (
        "server              server lifecycle (start)",
        "status              print the config state, auth mode, bind and DB path",
        "doctor              check python, config dir, config and users file",
        "token               token utilities (hash)",
        "config              instance config utilities (init, path, show)",
        "db                  storage utilities (path)",
    ):
        assert line in out


def test_server_help_lists_start_and_its_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["server", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "start the HTTP MCP server, blocking (--host/--port)" in out
    assert "--host" in out
    assert "--port" in out


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_prints_the_installed_version(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([flag])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"hull {__version__}"


@pytest.mark.parametrize(
    ("argv", "choices"),
    [
        ([], "{server,status,doctor,token,config,db}"),
        (["server"], "{start}"),
        (["token"], "{hash}"),
        (["config"], "{init,path,show}"),
        (["db"], "{path}"),
    ],
)
def test_missing_subcommand_names_the_choices(
    argv: list[str], choices: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"the following arguments are required: {choices}" in err
    # The old message named the internal dest, e.g. "server_command".
    assert "_command" not in err


# --- token hash -------------------------------------------------------------------------


def test_token_hash_reads_piped_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _stdin(monkeypatch, "my-plain-token\n")
    assert main(["token", "hash"]) == 0
    assert verify_token("my-plain-token", capsys.readouterr().out.strip())


def test_token_hash_prompts_on_a_tty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _stdin(monkeypatch, "", tty=True)
    monkeypatch.setattr("hull_core.cli.getpass.getpass", lambda _prompt: "prompted-token")
    assert main(["token", "hash"]) == 0
    assert verify_token("prompted-token", capsys.readouterr().out.strip())


def test_token_hash_refuses_an_argv_token(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["token", "hash", "my-plain-token"]) == 2
    err = capsys.readouterr().err
    assert "refusing to read a token from the command line" in err
    assert "stdin" in err


def test_token_hash_rejects_an_empty_token(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _stdin(monkeypatch, "  \n")
    assert main(["token", "hash"]) == 1
    assert "empty token" in capsys.readouterr().err


# --- config and db ----------------------------------------------------------------------


def test_config_init_writes_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _home(tmp_path, monkeypatch)
    assert main(["config", "init"]) == 0
    assert (tmp_path / "config.toml").is_file()
    assert main(["config", "init"]) == 2  # refuses clobber
    assert main(["config", "init", "--force"]) == 0


def test_config_path_prints_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # noqa: ANN001
    _home(tmp_path, monkeypatch)
    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "config.toml")


def test_config_show_prints_template_when_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _home(tmp_path, monkeypatch)
    assert main(["config", "show"]) == 0
    assert 'auth = "no-auth"' in capsys.readouterr().out


def test_config_show_prefers_the_file_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text("[server]\nport = 9999\n", encoding="utf-8")
    _home(tmp_path, monkeypatch)
    assert main(["config", "show"]) == 0
    assert capsys.readouterr().out == "[server]\nport = 9999\n"


def test_db_path_prints_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # noqa: ANN001
    _home(tmp_path, monkeypatch)
    assert main(["db", "path"]) == 0
    assert "hull.db" in capsys.readouterr().out


# --- status -----------------------------------------------------------------------------


def test_status_reports_an_unconfigured_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _home(tmp_path, monkeypatch)
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "hull: not configured" in out
    assert f"path: {tmp_path / 'config.toml'} (not created yet)" in out


def test_status_prints_mode_bind_and_db_without_the_token_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token_hash = "scrypt$32768$8$1$aa$bb"
    (tmp_path / "config.toml").write_text(
        f'[server]\nauth = "token"\ntoken_hash = "{token_hash}"\nport = 8123\n', encoding="utf-8"
    )
    _home(tmp_path, monkeypatch)
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "hull: configured" in out
    assert "auth: token" in out
    assert "bind: 127.0.0.1:8123" in out
    assert f"db: {tmp_path / 'hull.db'}" in out
    assert token_hash not in out


def test_status_names_the_users_file_in_multi_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text('[server]\nauth = "multi"\n', encoding="utf-8")
    _home(tmp_path, monkeypatch)
    assert main(["status"]) == 0
    assert f"users: {tmp_path / 'users.toml'}" in capsys.readouterr().out


def test_status_fails_on_an_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text("[server]\nauth = 'nope'\n", encoding="utf-8")
    _home(tmp_path, monkeypatch)
    assert main(["status"]) == 1
    assert "auth must be one of" in capsys.readouterr().err


# --- doctor -----------------------------------------------------------------------------


def test_doctor_warns_on_a_fresh_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _home(tmp_path, monkeypatch)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[ok] python" in out
    assert "[ok] config dir writable" in out
    assert "[warn] config: not configured" in out


def test_doctor_warns_when_the_config_dir_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _home(tmp_path / "absent", monkeypatch)
    assert main(["doctor"]) == 0
    assert "[warn] config dir does not exist yet" in capsys.readouterr().out


def test_doctor_fails_when_the_users_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text('[server]\nauth = "multi"\n', encoding="utf-8")
    _home(tmp_path, monkeypatch)
    assert main(["doctor"]) == 1
    assert "[fail] users file: users file not found" in capsys.readouterr().out


def test_doctor_fails_on_an_invalid_users_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text('[server]\nauth = "multi"\n', encoding="utf-8")
    (tmp_path / "users.toml").write_text('[users.alice]\nnamespace = "alice"\n', encoding="utf-8")
    _home(tmp_path, monkeypatch)
    assert main(["doctor"]) == 1
    assert "token_hash missing or not a scrypt encoding" in capsys.readouterr().out


def test_doctor_passes_when_the_users_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text('[server]\nauth = "multi"\n', encoding="utf-8")
    alice_hash = hash_token("alice-token")
    (tmp_path / "users.toml").write_text(
        f'[users.alice]\ntoken_hash = "{alice_hash}"\nnamespace = "alice"\n', encoding="utf-8"
    )
    _home(tmp_path, monkeypatch)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[ok] config: valid (auth=multi)" in out
    assert f"[ok] users file: {tmp_path / 'users.toml'} (1 users)" in out


def test_doctor_fails_when_the_config_dir_is_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _home(tmp_path, monkeypatch)
    monkeypatch.setattr("hull_core.cli.os.access", lambda *_args: False)
    assert main(["doctor"]) == 1
    assert "[fail] config dir not writable" in capsys.readouterr().out


def test_doctor_fails_on_an_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.toml").write_text("[server\n", encoding="utf-8")
    _home(tmp_path, monkeypatch)
    assert main(["doctor"]) == 1
    assert "[fail] config:" in capsys.readouterr().out


# --- server start -----------------------------------------------------------------------


def test_server_start_forwards_bind_args_and_swallows_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_run(**kwargs):  # noqa: ANN003
        seen.update(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr("hull_core.cli.run_server", fake_run)
    assert main(["server", "start", "--host", "127.0.0.1", "--port", "9123"]) == 0
    assert seen == {"host": "127.0.0.1", "port": 9123}


def test_server_start_reports_config_error_as_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(**_kwargs):  # noqa: ANN003
        raise ServerConfigError("auth = 'no-auth' only permits loopback binds")

    monkeypatch.setattr("hull_core.cli.run_server", fake_run)
    assert main(["server", "start", "--host", "0.0.0.0"]) == 2
    assert "only permits loopback" in capsys.readouterr().err
