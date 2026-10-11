"""legacy_config_file: read-only port of mcp-core's shared ``config.enc``.

Fixtures are produced by a test-local encryptor that re-implements mcp-core's
``mcp_core.storage.config_file`` writer (current salted format plus both
legacy fixed-salt derivations), never by importing mcp-core.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import types
from collections.abc import Mapping
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from platformdirs import user_config_dir

from hull_core.storage import legacy_config_file as lcf
from hull_core.storage import delete_legacy_config, read_legacy_config

MACHINE = "test-machine-guid"
USER = "test-user"


@pytest.fixture(autouse=True)
def identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lcf, "machine_id", lambda: MACHINE)
    monkeypatch.setattr(lcf, "username", lambda: USER)


# --- reference encryptor (mirrors mcp-core's writer) ---------------------------


def _key(salt: bytes, iterations: int, machine: str = MACHINE, user: str = USER) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", f"{machine}:{user}".encode(), salt, iterations, dklen=32)


def _store(servers: Mapping[str, object]) -> bytes:
    return json.dumps({"version": 1, "servers": servers}).encode("utf-8")


def _write_current(path: Path, servers: Mapping[str, object], *, salt: bytes | None = None) -> bytes:
    salt = salt or os.urandom(16)
    iv = os.urandom(12)
    path.write_bytes(salt + iv + AESGCM(_key(salt, 600_000)).encrypt(iv, _store(servers), None))
    return salt


def _write_legacy(path: Path, servers: Mapping[str, object], iterations: int) -> None:
    iv = os.urandom(12)
    path.write_bytes(iv + AESGCM(_key(b"mcp-relay-config", iterations)).encrypt(iv, _store(servers), None))


def _decrypt_current(path: Path) -> tuple[bytes, dict[str, object]]:
    data = path.read_bytes()
    salt = data[:16]
    plaintext = AESGCM(_key(salt, 600_000)).decrypt(data[16:28], data[28:], None)
    return salt, json.loads(plaintext)


SERVERS = {
    "better-telegram-mcp": {"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h"},
    "better-email-mcp": {"EMAIL_CREDENTIALS": "e"},
}


# --- read -------------------------------------------------------------------------


def test_reads_current_format(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    _write_current(path, SERVERS)
    assert read_legacy_config("better-telegram-mcp", path=path) == SERVERS["better-telegram-mcp"]
    assert read_legacy_config("better-email-mcp", path=path) == {"EMAIL_CREDENTIALS": "e"}


@pytest.mark.parametrize("iterations", [600_000, 100_000])
def test_reads_legacy_fixed_salt_formats(tmp_path: Path, iterations: int) -> None:
    path = tmp_path / "config.enc"
    _write_legacy(path, SERVERS, iterations)
    assert read_legacy_config("better-telegram-mcp", path=path) == SERVERS["better-telegram-mcp"]


def test_missing_file_is_none(tmp_path: Path) -> None:
    assert read_legacy_config("better-telegram-mcp", path=tmp_path / "absent.enc") is None


def test_unreadable_path_is_none(tmp_path: Path) -> None:
    # A directory where the file should be: read_bytes raises an OSError.
    assert read_legacy_config("better-telegram-mcp", path=tmp_path) is None


def test_missing_entry_is_none(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    _write_current(path, SERVERS)
    assert read_legacy_config("better-notion-mcp", path=path) is None


def test_non_dict_entry_is_none(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    _write_current(path, {"better-telegram-mcp": "not-a-dict"})
    assert read_legacy_config("better-telegram-mcp", path=path) is None


def test_tampered_blob_is_none(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    _write_current(path, SERVERS)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    assert read_legacy_config("better-telegram-mcp", path=path) is None


@pytest.mark.parametrize("blob", [b"", b"\x00" * 5, b"\x00" * 27, b"\x00" * 200])
def test_garbage_is_none(tmp_path: Path, blob: bytes) -> None:
    path = tmp_path / "config.enc"
    path.write_bytes(blob)
    assert read_legacy_config("better-telegram-mcp", path=path) is None


def test_other_machine_identity_is_none(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    salt = os.urandom(16)
    iv = os.urandom(12)
    key = _key(salt, 600_000, machine="other-machine")
    path.write_bytes(salt + iv + AESGCM(key).encrypt(iv, _store(SERVERS), None))
    assert read_legacy_config("better-telegram-mcp", path=path) is None


@pytest.mark.parametrize("payload", [b"[1, 2]", b'{"version": 1}', b'{"servers": []}', b"\xff not json"])
def test_invalid_store_shape_is_none(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "config.enc"
    salt = os.urandom(16)
    iv = os.urandom(12)
    path.write_bytes(salt + iv + AESGCM(_key(salt, 600_000)).encrypt(iv, payload, None))
    assert read_legacy_config("better-telegram-mcp", path=path) is None


def test_invalid_store_shape_in_legacy_format_is_none(tmp_path: Path) -> None:
    # Decrypts under the fixed legacy salt (600k) but is not a store: the
    # 100k derivation is still tried, then the file counts as unreadable.
    path = tmp_path / "config.enc"
    iv = os.urandom(12)
    path.write_bytes(iv + AESGCM(_key(b"mcp-relay-config", 600_000)).encrypt(iv, b'{"version": 1}', None))
    assert read_legacy_config("better-telegram-mcp", path=path) is None
    assert delete_legacy_config("better-telegram-mcp", path=path) is False


def test_read_never_writes(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    _write_legacy(path, SERVERS, 100_000)
    before = path.read_bytes()
    read_legacy_config("better-telegram-mcp", path=path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.enc"]


def test_default_path_matches_mcp_core(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = Path(user_config_dir("mcp", appauthor=False)) / "config.enc"
    assert lcf.default_legacy_config_path() == expected

    target = tmp_path / "config.enc"
    _write_current(target, SERVERS)
    monkeypatch.setattr(lcf, "default_legacy_config_path", lambda: target)
    assert read_legacy_config("better-email-mcp") == {"EMAIL_CREDENTIALS": "e"}
    assert delete_legacy_config("better-email-mcp") is True
    assert read_legacy_config("better-email-mcp") is None


# --- delete -----------------------------------------------------------------------


def test_delete_keeps_other_entries_in_current_format(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    salt = _write_current(path, SERVERS)

    assert delete_legacy_config("better-telegram-mcp", path=path) is True

    new_salt, store = _decrypt_current(path)
    assert new_salt == salt, "re-encrypted with the file's own salt"
    assert store == {"version": 1, "servers": {"better-email-mcp": {"EMAIL_CREDENTIALS": "e"}}}
    assert read_legacy_config("better-telegram-mcp", path=path) is None
    assert read_legacy_config("better-email-mcp", path=path) == {"EMAIL_CREDENTIALS": "e"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.enc"], "no tmp files left behind"


@pytest.mark.parametrize("iterations", [600_000, 100_000])
def test_delete_rewrites_legacy_file_in_current_format(tmp_path: Path, iterations: int) -> None:
    path = tmp_path / "config.enc"
    _write_legacy(path, SERVERS, iterations)

    assert delete_legacy_config("better-telegram-mcp", path=path) is True

    salt, store = _decrypt_current(path)
    assert len(salt) == 16
    assert salt != b"mcp-relay-config"
    assert store["servers"] == {"better-email-mcp": {"EMAIL_CREDENTIALS": "e"}}


def test_delete_last_entry_removes_file(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    _write_current(path, {"better-telegram-mcp": {"K": "v"}})
    assert delete_legacy_config("better-telegram-mcp", path=path) is True
    assert not path.exists()


def test_delete_absent_entry_returns_false_and_leaves_file(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    _write_current(path, SERVERS)
    before = path.read_bytes()
    assert delete_legacy_config("better-notion-mcp", path=path) is False
    assert path.read_bytes() == before


def test_delete_missing_file_returns_false(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    assert delete_legacy_config("better-telegram-mcp", path=path) is False
    assert not path.exists(), "never creates the file"


def test_delete_undecryptable_file_returns_false_and_leaves_it(tmp_path: Path) -> None:
    path = tmp_path / "config.enc"
    path.write_bytes(b"\x01" * 100)
    assert delete_legacy_config("better-telegram-mcp", path=path) is False
    assert path.read_bytes() == b"\x01" * 100


# --- machine identity (port of mcp_core.storage.machine_id) -------------------------


@pytest.fixture
def real_identity(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Undo the autouse stubs and the lru_cache so the real functions run."""
    monkeypatch.undo()
    lcf.machine_id.cache_clear()
    yield
    lcf.machine_id.cache_clear()


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_machine_id_linux_reads_etc_machine_id(
    real_identity: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "machine-id"
    source.write_text("abc123\n")
    real_open = open
    monkeypatch.setattr(lcf.sys, "platform", "linux")
    monkeypatch.setattr(
        "builtins.open", lambda file, *a, **k: real_open(source if file == "/etc/machine-id" else file, *a, **k)
    )
    assert lcf.machine_id() == "abc123"


def test_machine_id_darwin_reads_platform_uuid(real_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        calls.append(cmd)
        return _completed('  "IOPlatformUUID" = "UUID-1234"\n')

    monkeypatch.setattr(lcf.sys, "platform", "darwin")
    monkeypatch.setattr(lcf.subprocess, "run", fake_run)
    assert lcf.machine_id() == "UUID-1234"
    assert calls[0][0] == "ioreg"


def test_machine_id_win32_reads_machine_guid(real_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        calls.append(cmd)
        return _completed(
            "\r\nHKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography\r\n    MachineGuid    REG_SZ    guid-5678\r\n"
        )

    monkeypatch.setattr(lcf.sys, "platform", "win32")
    monkeypatch.setattr(lcf.subprocess, "run", fake_run)
    assert lcf.machine_id() == "guid-5678"
    assert calls[0][:2] == ["reg", "query"]


@pytest.mark.parametrize("plat", ["darwin", "win32"])
def test_machine_id_falls_back_when_command_has_no_match(
    real_identity: None, monkeypatch: pytest.MonkeyPatch, plat: str
) -> None:
    monkeypatch.setattr(lcf.sys, "platform", plat)
    monkeypatch.setattr(lcf.subprocess, "run", lambda cmd, **kwargs: _completed("nothing useful"))
    monkeypatch.setattr(lcf.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(lcf, "_first_mac", lambda: "aa:bb:cc:dd:ee:ff")
    assert lcf.machine_id() == "host-aa:bb:cc:dd:ee:ff"


def test_machine_id_falls_back_on_error(real_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(cmd, **kwargs):  # noqa: ANN001, ANN003
        raise OSError("no reg.exe")

    monkeypatch.setattr(lcf.sys, "platform", "win32")
    monkeypatch.setattr(lcf.subprocess, "run", broken)
    monkeypatch.setattr(lcf.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(lcf, "_first_mac", lambda: "unknown")
    assert lcf.machine_id() == "host-unknown"


def test_machine_id_unknown_platform_falls_back(real_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lcf.sys, "platform", "freebsd14")
    monkeypatch.setattr(lcf.socket, "gethostname", lambda: "bsd")
    monkeypatch.setattr(lcf, "_first_mac", lambda: "unknown")
    assert lcf.machine_id() == "bsd-unknown"


def test_machine_id_is_cached(real_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = {"n": 0}

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        counter["n"] += 1
        return _completed("MachineGuid    REG_SZ    g")

    monkeypatch.setattr(lcf.sys, "platform", "win32")
    monkeypatch.setattr(lcf.subprocess, "run", fake_run)
    assert lcf.machine_id() == lcf.machine_id() == "g"
    assert counter["n"] == 1


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (0x0011_2233_4455, "00:11:22:33:44:55"),
        (0x0100_0000_0001, "unknown"),  # multicast bit: random node id
        (0, "unknown"),
    ],
)
def test_first_mac(monkeypatch: pytest.MonkeyPatch, node: int, expected: str) -> None:
    monkeypatch.setattr(lcf.uuid, "getnode", lambda: node)
    assert lcf._first_mac() == expected


def test_first_mac_error_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> int:
        raise OSError("no nic")

    monkeypatch.setattr(lcf.uuid, "getnode", broken)
    assert lcf._first_mac() == "unknown"


def test_username_uses_getpass(real_identity: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lcf.getpass, "getuser", lambda: "alice")
    assert lcf.username() == "alice"


@pytest.mark.parametrize(
    ("env", "expected"),
    [({"USER": "bob", "USERNAME": "x"}, "bob"), ({"USERNAME": "carol"}, "carol"), ({}, "unknown")],
)
def test_username_env_fallback(
    real_identity: None, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: str
) -> None:
    def broken() -> str:
        raise OSError("no pwd entry")

    monkeypatch.setattr(lcf.getpass, "getuser", broken)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert lcf.username() == expected


def test_file_key_binds_machine_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    salt = b"s" * 16
    assert lcf._file_key(salt, 1000) == _key(salt, 1000)
    monkeypatch.setattr(lcf, "username", lambda: "someone-else")
    assert lcf._file_key(salt, 1000) != _key(salt, 1000)


def test_module_does_not_import_mcp_core() -> None:
    source = Path(lcf.__file__).read_text(encoding="utf-8")
    assert "import mcp_core" not in source
    assert "from mcp_core" not in source
    assert isinstance(lcf, types.ModuleType)
