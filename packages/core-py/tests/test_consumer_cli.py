"""build_cli: dispatch shapes, built-in subcommands, extras, storage refusals."""

from __future__ import annotations

import argparse
import io
import json
import secrets
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hull_core import consumer_cli
from hull_core.auth.tokens import verify_token
from hull_core.consumer_cli import build_cli
from hull_core.storage import per_plugin_store
from hull_core.storage.per_plugin_store import PerPluginStore

SERVER = "better-telegram-mcp"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("CREDENTIAL_SECRET", raising=False)
    monkeypatch.delenv("MCP_STORAGE_BACKEND", raising=False)
    monkeypatch.setattr(per_plugin_store, "_tightened_dirs", set())
    return tmp_path


class _Serve:
    def __init__(self, rc: int | None = None) -> None:
        self.rc = rc
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> int | None:
        self.calls.append(argv)
        return self.rc


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _stdin(monkeypatch: pytest.MonkeyPatch, text: str = "", *, tty: bool = False) -> None:
    monkeypatch.setattr(sys, "stdin", _Tty(text) if tty else io.StringIO(text))


def _cli(**kwargs):  # noqa: ANN003, ANN202
    serve = kwargs.pop("serve", None) or _Serve()
    return build_cli(SERVER, serve=serve, **kwargs), serve


# --- dispatch ---------------------------------------------------------------------------


def test_bare_invocation_serves(home: Path) -> None:
    run, serve = _cli()
    assert run([]) == 0
    assert serve.calls == [[]]


def test_serve_return_code_propagates(home: Path) -> None:
    run, _ = _cli(serve=_Serve(rc=3))
    assert run([]) == 3
    assert run(["--http"]) == 3


def test_argv_defaults_to_sys_argv(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--http", "--port", "9000"])
    run, serve = _cli()
    assert run() == 0
    assert serve.calls == [["--http", "--port", "9000"]]


def test_flags_pass_through_to_serve(home: Path) -> None:
    run, serve = _cli()
    assert run(["--http", "status"]) == 0
    assert serve.calls == [["--http", "status"]]


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_is_answered_here(home: Path, capsys: pytest.CaptureFixture[str], flag: str) -> None:
    run, serve = _cli()
    assert run([flag]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: {SERVER} [-h] [--version] [<subcommand> ...]")
    assert "passed through to the MCP server" in out
    assert "subcommands: config, doctor, logout, status, token" in out
    assert serve.calls == []


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_when_set(home: Path, capsys: pytest.CaptureFixture[str], flag: str) -> None:
    run, serve = _cli(version="1.2.3")
    assert run([flag]) == 0
    assert capsys.readouterr().out.strip() == f"{SERVER} 1.2.3"
    assert serve.calls == []


def test_version_without_version_passes_through(home: Path) -> None:
    run, serve = _cli()
    assert run(["--version"]) == 0
    assert serve.calls == [["--version"]]


def test_unknown_subcommand(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, serve = _cli()
    assert run(["frobnicate"]) == 2
    err = capsys.readouterr().err
    assert f"{SERVER}: unknown subcommand 'frobnicate'" in err
    assert "usage:" in err
    assert serve.calls == []


def test_bad_subcommand_args_return_2(home: Path) -> None:
    run, _ = _cli()
    assert run(["config", "bogus"]) == 2
    assert run(["token"]) == 2


def test_subcommand_help_returns_0(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, _ = _cli()
    assert run(["config", "-h"]) == 0
    assert "status" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("server_name", "plugin"), [(SERVER, "telegram"), ("plain-mcp", "plain"), ("better-x", "x"), ("solo", "solo")]
)
def test_plugin_name_default_strips_affixes(
    home: Path, capsys: pytest.CaptureFixture[str], server_name: str, plugin: str
) -> None:
    run = build_cli(server_name, serve=_Serve())
    assert run(["status"]) == 0
    assert f"path: {home / f'.{plugin}' / 'config.json'} (not created yet)" in capsys.readouterr().out


# --- status / config status -------------------------------------------------------------


def test_status_not_configured(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, _ = _cli()
    assert run(["status"]) == 0
    out = capsys.readouterr().out
    assert f"{SERVER}: not configured" in out
    assert f"path: {home / '.telegram' / 'config.json'} (not created yet)" in out
    assert not (home / ".telegram").exists(), "status never creates the data dir"


def test_config_status_configured(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"api_hash": "SECRET-VALUE"})
    run, _ = _cli()
    assert run(["config", "status"]) == 0
    out = capsys.readouterr().out
    assert f"{SERVER}: configured" in out
    assert f"path: {home / '.telegram' / 'config.json'}\n" in out
    assert "SECRET-VALUE" not in out


def _encrypt(key: bytes, payload: dict) -> bytes:
    nonce = secrets.token_bytes(12)
    return nonce + AESGCM(key).encrypt(nonce, json.dumps(payload).encode(), None)


def test_status_marks_legacy_location(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    legacy = home / ".better-telegram-mcp"
    legacy.mkdir()
    key = secrets.token_bytes(32)
    (legacy / ".secret").write_bytes(key)
    (legacy / "config.json").write_bytes(_encrypt(key, {"a": 1}))
    run, _ = _cli()
    assert run(["status"]) == 0
    out = capsys.readouterr().out
    assert f"{SERVER}: configured" in out
    assert f"path: {legacy / 'config.json'} (legacy dir, migrated on next save)" in out


def test_status_corrupt(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"a": 1})
    (home / ".telegram" / "config.json").write_bytes(b"\x00" * 40)
    run, _ = _cli()
    assert run(["status"]) == 1
    assert "config is corrupt" in capsys.readouterr().err


def test_status_non_local_backend(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MCP_STORAGE_BACKEND", "memory")
    run, _ = _cli()
    assert run(["status"]) == 0
    out = capsys.readouterr().out
    assert "not configured" in out
    assert "backend: InMemoryBackend" in out


def test_status_bad_backend_is_one_line(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MCP_STORAGE_BACKEND", "cf-kv")
    run, _ = _cli()
    assert run(["status"]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"{SERVER}: ")
    assert "cf-kv" in err
    assert "Traceback" not in err


def test_status_foreign_secret_is_one_line(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"a": 1})
    secret = home / ".telegram" / ".secret"
    secret.write_bytes(b"ab" * 32)
    run, _ = _cli()
    assert run(["status"]) == 1
    err = capsys.readouterr().err
    assert str(secret) in err
    assert "not a hull machine key" in err
    assert secret.read_bytes() == b"ab" * 32


# --- config delete / logout ---------------------------------------------------------------


def test_config_delete_yes(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"a": 1})
    run, _ = _cli()
    assert run(["config", "delete", "--yes"]) == 0
    assert f"{SERVER}: config deleted" in capsys.readouterr().out
    assert PerPluginStore("telegram").load() is None


def test_config_delete_refuses_non_interactive(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    PerPluginStore("telegram").save({"a": 1})
    _stdin(monkeypatch)
    run, _ = _cli()
    assert run(["config", "delete"]) == 1
    assert "without --yes in non-interactive mode" in capsys.readouterr().err
    assert PerPluginStore("telegram").load() == {"a": 1}


@pytest.mark.parametrize("answer", ["y\n", "YES\n"])
def test_config_delete_interactive_confirm(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], answer: str
) -> None:
    PerPluginStore("telegram").save({"a": 1})
    _stdin(monkeypatch, answer, tty=True)
    run, _ = _cli()
    assert run(["config", "delete"]) == 0
    captured = capsys.readouterr()
    assert f"Delete stored config for {SERVER}? [y/N]" in captured.err
    assert PerPluginStore("telegram").load() is None


@pytest.mark.parametrize("answer", ["n\n", "\n", ""])
def test_config_delete_interactive_abort(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], answer: str
) -> None:
    PerPluginStore("telegram").save({"a": 1})
    _stdin(monkeypatch, answer, tty=True)
    run, _ = _cli()
    assert run(["config", "delete"]) == 1
    assert f"{SERVER}: aborted" in capsys.readouterr().err
    assert PerPluginStore("telegram").load() == {"a": 1}


def test_logout_clears_config_and_session(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"a": 1})
    PerPluginStore("telegram", sub_key="session").save({"session": "s"})
    run, _ = _cli()
    assert run(["logout", "--yes"]) == 0
    assert "stored credentials removed" in capsys.readouterr().out
    assert PerPluginStore("telegram").load() is None
    assert PerPluginStore("telegram", sub_key="session").load() is None


def test_logout_refuses_non_interactive(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    PerPluginStore("telegram").save({"a": 1})
    _stdin(monkeypatch)
    run, _ = _cli()
    assert run(["logout"]) == 1
    assert PerPluginStore("telegram").load() == {"a": 1}


# --- doctor -----------------------------------------------------------------------------


def test_doctor_healthy(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"a": 1})
    (home / ".telegram-mcp").mkdir()
    run, _ = _cli()
    assert run(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[ok] python" in out
    assert "[ok] credential backend: LocalFsBackend" in out
    assert f"[ok] data dir writable: {home / '.telegram'}" in out
    assert f"[ok] legacy data dir (read-only fallback): {home / '.telegram-mcp'}" in out
    assert "[ok] config: configured" in out


def test_doctor_fresh_machine_warns_without_creating(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, _ = _cli()
    assert run(["doctor"]) == 0
    out = capsys.readouterr().out
    assert f"[warn] data dir does not exist yet: {home / '.telegram'}" in out
    assert "[warn] config: not configured" in out
    assert not (home / ".telegram").exists()


def test_doctor_unwritable_dir_fails(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (home / ".telegram").mkdir()
    monkeypatch.setattr(consumer_cli.os, "access", lambda path, mode: False)
    run, _ = _cli()
    assert run(["doctor"]) == 1
    assert "[fail] data dir not writable" in capsys.readouterr().out


def test_doctor_corrupt_config_fails(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"a": 1})
    (home / ".telegram" / "config.json").write_bytes(b"\x00" * 40)
    run, _ = _cli()
    assert run(["doctor"]) == 1
    assert "[fail] config: corrupt" in capsys.readouterr().out


def test_doctor_old_python_fails(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 12, 9, "final", 0))
    run, _ = _cli()
    rc = run(["doctor"])
    monkeypatch.undo()
    assert rc == 1
    assert "[fail] python 3.12 (hull requires >= 3.13)" in capsys.readouterr().out


def test_doctor_bad_backend_fails(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MCP_STORAGE_BACKEND", "nope")
    run, _ = _cli()
    assert run(["doctor"]) == 1
    assert "[fail] credential backend:" in capsys.readouterr().out


def test_doctor_memory_backend_skips_disk(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MCP_STORAGE_BACKEND", "memory")
    run, _ = _cli()
    assert run(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "InMemoryBackend" in out
    assert "data dir" not in out


def test_doctor_foreign_secret_fails(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("telegram").save({"a": 1})
    secret = home / ".telegram" / ".secret"
    secret.write_bytes(b"")
    run, _ = _cli()
    assert run(["doctor"]) == 1
    out = capsys.readouterr().out
    assert f"[fail] machine key: {secret}" in out
    assert secret.read_bytes() == b""


# --- token hash -------------------------------------------------------------------------


def test_token_hash_from_stdin(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _stdin(monkeypatch, "my-token\n")
    run, _ = _cli()
    assert run(["token", "hash"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("scrypt$")
    assert "my-token" not in out
    assert verify_token("my-token", out)


def test_token_hash_from_prompt(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stdin(monkeypatch, tty=True)
    prompts: list[str] = []
    monkeypatch.setattr(consumer_cli.getpass, "getpass", lambda prompt: prompts.append(prompt) or "typed")
    run, _ = _cli()
    assert run(["token", "hash"]) == 0
    assert prompts == ["Token: "]
    assert verify_token("typed", capsys.readouterr().out.strip())


def test_token_hash_empty(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _stdin(monkeypatch, "  \n")
    run, _ = _cli()
    assert run(["token", "hash"]) == 1
    assert f"{SERVER}: empty token" in capsys.readouterr().err


def test_token_on_argv_refused(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, _ = _cli()
    assert run(["token", "hash", "leaky"]) == 2
    captured = capsys.readouterr()
    assert "refusing to read a token from the command line" in captured.err
    assert "leaky" not in captured.err + captured.out


# --- extras -----------------------------------------------------------------------------


def test_extra_bare_handler(home: Path) -> None:
    seen: list[argparse.Namespace] = []
    run, _ = _cli(extra={"login": lambda ns: seen.append(ns) or 5})
    assert run(["login"]) == 5
    assert seen[0].subcommand == "login"


def test_extra_with_configure(home: Path) -> None:
    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--phone")

    run, _ = _cli(extra={"login": (configure, lambda ns: 0 if ns.phone == "+1" else 9)})
    assert run(["login", "--phone", "+1"]) == 0
    assert run(["login", "--bogus"]) == 2


def test_extra_overrides_builtin_and_its_args(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run, _ = _cli(extra={"config": lambda ns: 7})
    assert run(["config"]) == 7, "the built-in positional is gone with the override"
    assert run(["config", "status"]) == 2
    assert run(["-h"]) == 0
    assert "subcommands: config, doctor, logout, status, token" in capsys.readouterr().out


def test_extra_value_error_keeps_traceback(home: Path) -> None:
    def broken(ns: argparse.Namespace) -> int:
        raise ValueError("bug in server code")

    run, _ = _cli(extra={"login": broken, "status": broken})
    with pytest.raises(ValueError, match="bug in server code"):
        run(["login"])
    with pytest.raises(ValueError, match="bug in server code"):
        run(["status"])


def test_extra_machine_key_error_is_one_line(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def broken(ns: argparse.Namespace) -> int:
        raise per_plugin_store.MachineKeyError("/x/.secret is not a hull machine key")

    run, _ = _cli(extra={"login": broken})
    assert run(["login"]) == 1
    assert capsys.readouterr().err.strip() == f"{SERVER}: /x/.secret is not a hull machine key"


def test_explicit_plugin_name(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    PerPluginStore("tg").save({"a": 1})
    run = build_cli(SERVER, plugin_name="tg", serve=_Serve())
    assert run(["status"]) == 0
    assert f"{home / '.tg' / 'config.json'}" in capsys.readouterr().out
