"""``config set/get``: in-place TOML edits that keep every comment."""

from __future__ import annotations

import pytest

from hull_core.cli import main
from hull_core.config import tomledit
from hull_core.config.settings import CONFIG_TEMPLATE, ConfigError

SAMPLE = """\
# host-owned config
[server]
host = "127.0.0.1"   # bind address
port = 8000
# auth = "token"

[models.embed]
model = "m1"
"""


def _write(tmp_path, text: str = SAMPLE):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_set_replaces_a_value_and_keeps_the_trailing_comment(tmp_path):
    path = _write(tmp_path)
    tomledit.set_value(path, "server.port", "8771")
    body = path.read_text(encoding="utf-8")
    assert "port = 8771" in body
    assert 'host = "127.0.0.1"   # bind address' in body
    assert '# auth = "token"' in body
    assert tomledit.get_value(path, "server.port") == 8771


def test_set_inserts_a_missing_key_inside_its_section_not_at_eof(tmp_path):
    path = _write(tmp_path)
    tomledit.set_value(path, "server.rpm", "120")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines.index("rpm = 120") < lines.index("[models.embed]")
    assert tomledit.get_value(path, "server.rpm") == 120


def test_set_quotes_strings_and_keeps_bools_typed(tmp_path):
    path = _write(tmp_path)
    tomledit.set_value(path, "server.auth", "token")
    tomledit.set_value(path, "server.host", "0.0.0.0")
    assert tomledit.get_value(path, "server.auth") == "token"
    assert tomledit.get_value(path, "server.host") == "0.0.0.0"


def test_set_refuses_a_section_that_is_not_there(tmp_path):
    path = _write(tmp_path)
    with pytest.raises(ConfigError, match=r"section \[nope\] not found"):
        tomledit.set_value(path, "nope.key", "1")


def test_set_refuses_a_key_that_is_not_dotted(tmp_path):
    path = _write(tmp_path)
    with pytest.raises(ConfigError, match="key must be dotted"):
        tomledit.set_value(path, "port", "1")


def test_set_refuses_a_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="no config at"):
        tomledit.set_value(tmp_path / "absent.toml", "server.port", "1")


def test_cli_set_then_get_roundtrip_masks_secrets(tmp_path, monkeypatch, capsys):
    _write(tmp_path)
    monkeypatch.setattr("hull_core.cli.default_config_dir", lambda: tmp_path)
    monkeypatch.setattr("hull_core.cli.CONFIG_TEMPLATE", CONFIG_TEMPLATE)

    assert main(["config", "set", "server.token_hash", "scrypt$abc"]) == 0
    out = capsys.readouterr().out
    assert "*** (value not echoed)" in out
    assert "scrypt$abc" not in out

    assert main(["config", "get", "server.token_hash"]) == 0
    assert capsys.readouterr().out.strip() == "*** (value not echoed)"


def test_cli_get_reports_an_unknown_key_on_stderr(tmp_path, monkeypatch, capsys):
    _write(tmp_path)
    monkeypatch.setattr("hull_core.cli.default_config_dir", lambda: tmp_path)
    assert main(["config", "get", "server.nope"]) == 2
    assert "key not found: server.nope" in capsys.readouterr().err
