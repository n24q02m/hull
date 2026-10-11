"""PerPluginStore: AES-GCM blobs, key derivation, legacy-dir reads, cross-format compatibility.

The cross-format tests write blobs with the OLD writers' algorithms
re-implemented inline here (mcp-core Python store, hull TS store), never by
importing those packages, so the formats are pinned independently of either.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hull_core.storage import backends, per_plugin_store
from hull_core.storage.backends import InMemoryBackend, LocalFsBackend
from hull_core.storage.per_plugin_store import (
    MachineKeyError,
    PerPluginStore,
    derive_legacy_multi_user_key,
    derive_multi_user_key,
)

MASTER = "test-master-secret"
PLUGIN = "telegram"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("CREDENTIAL_SECRET", raising=False)
    monkeypatch.delenv("MCP_STORAGE_BACKEND", raising=False)
    monkeypatch.setattr(per_plugin_store, "_tightened_dirs", set())
    return tmp_path


@pytest.fixture
def master(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("CREDENTIAL_SECRET", MASTER)
    return MASTER


# --- inline reference writers -----------------------------------------------


def _encrypt(key: bytes, payload: object) -> bytes:
    nonce = secrets.token_bytes(12)
    return nonce + AESGCM(key).encrypt(nonce, json.dumps(payload).encode("utf-8"), None)


def _write_single_user(directory: Path, payload: object) -> bytes:
    """Old mcp-core py AND hull TS single-user writer: raw 32-byte .secret + config.json."""
    directory.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    (directory / ".secret").write_bytes(key)
    (directory / "config.json").write_bytes(_encrypt(key, payload))
    return key


def _old_py_multi_user_key(key_plugin: str, sub: str) -> bytes:
    """Old mcp-core py multi-user key: PBKDF2-HMAC-SHA256 600k over CREDENTIAL_SECRET."""
    return hashlib.pbkdf2_hmac("sha256", MASTER.encode(), f"{key_plugin}:{sub}".encode(), 600_000, dklen=32)


def _ts_multi_user_key(plugin: str, sub: str) -> bytes:
    """hull TS multi-user key: scryptSync(secret, "<plugin>:<sub>", 32) with Node defaults."""
    return hashlib.scrypt(MASTER.encode(), salt=f"{plugin}:{sub}".encode(), n=16384, r=8, p=1, dklen=32)


def _write_multi_user(directory: Path, sub: str, key: bytes, payload: object) -> Path:
    path = directory / "subs" / sub / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encrypt(key, payload))
    return path


# --- key derivation ----------------------------------------------------------


def test_derivations_require_master(home: Path) -> None:
    with pytest.raises(RuntimeError, match="CREDENTIAL_SECRET env required"):
        derive_multi_user_key(PLUGIN, "u1")
    with pytest.raises(RuntimeError, match="CREDENTIAL_SECRET env required"):
        derive_legacy_multi_user_key(PLUGIN, "u1")


def test_derivations_match_reference_algorithms(home: Path, master: str) -> None:
    assert derive_multi_user_key(PLUGIN, "u1") == _ts_multi_user_key(PLUGIN, "u1")
    assert derive_legacy_multi_user_key(PLUGIN, "u1") == _old_py_multi_user_key(PLUGIN, "u1")
    assert derive_multi_user_key(PLUGIN, "u1") != derive_multi_user_key(PLUGIN, "u2")


# --- current layout round trips ---------------------------------------------


def test_single_user_round_trip_writes_current_layout(home: Path) -> None:
    store = PerPluginStore(PLUGIN)
    assert store.load() is None
    assert not store.has_stored_blob()
    store.save({"api_id": 1, "nested": {"k": "v"}})

    secret = home / ".telegram" / ".secret"
    blob_path = home / ".telegram" / "config.json"
    assert len(secret.read_bytes()) == 32
    assert store.cred_path == blob_path
    blob = blob_path.read_bytes()
    assert b"api_id" not in blob, "blob must be encrypted"
    # Format pin: [12 nonce][ciphertext][16 tag] under the raw .secret key.
    assert json.loads(AESGCM(secret.read_bytes()).decrypt(blob[:12], blob[12:], None)) == {
        "api_id": 1,
        "nested": {"k": "v"},
    }
    assert PerPluginStore(PLUGIN).load() == {"api_id": 1, "nested": {"k": "v"}}
    assert store.has_stored_blob()
    assert store.blob_locations() == [blob_path]


def test_machine_key_is_reused_across_saves(home: Path) -> None:
    PerPluginStore(PLUGIN).save({"a": 1})
    key = (home / ".telegram" / ".secret").read_bytes()
    PerPluginStore(PLUGIN).save({"a": 2})
    assert (home / ".telegram" / ".secret").read_bytes() == key
    assert PerPluginStore(PLUGIN).load() == {"a": 2}


def test_multi_user_requires_master_on_save(home: Path) -> None:
    with pytest.raises(RuntimeError, match="CREDENTIAL_SECRET"):
        PerPluginStore(PLUGIN, "u1").save({"x": 1})


def test_multi_user_round_trip_is_ts_compatible(home: Path, master: str) -> None:
    store = PerPluginStore(PLUGIN, "u1")
    assert store.cred_key == "telegram/subs/u1/config"
    store.save({"token": "t"})
    path = home / ".telegram" / "subs" / "u1" / "config.json"
    blob = path.read_bytes()
    # hull TS reads it with its own scrypt key.
    plaintext = AESGCM(_ts_multi_user_key(PLUGIN, "u1")).decrypt(blob[:12], blob[12:], None)
    assert json.loads(plaintext) == {"token": "t"}
    assert PerPluginStore(PLUGIN, "u1").load() == {"token": "t"}
    assert PerPluginStore(PLUGIN, "u2").load() is None
    assert not (home / ".telegram" / ".secret").exists(), "multi-user never touches the machine key"


def test_sub_key_selects_leaf(home: Path) -> None:
    store = PerPluginStore(PLUGIN, sub_key="session")
    assert store.cred_key == "telegram/session"
    store.save({"session": "s"})
    assert (home / ".telegram" / "session.json").exists()
    assert not (home / ".telegram" / "config.json").exists()
    assert PerPluginStore(PLUGIN).load() is None


@pytest.mark.parametrize(("plugin", "sub"), [("../x", None), ("ok", "../u"), ("ok", "")])
def test_invalid_names_rejected(home: Path, plugin: str, sub: str | None) -> None:
    with pytest.raises(ValueError):
        PerPluginStore(plugin, sub)


# --- cross-format: blobs written by older stores ----------------------------


def test_reads_ts_format_single_user_blob(home: Path) -> None:
    _write_single_user(home / ".telegram", {"from": "ts"})
    assert PerPluginStore(PLUGIN).load() == {"from": "ts"}


def test_reads_ts_format_multi_user_blob(home: Path, master: str) -> None:
    _write_multi_user(home / ".telegram", "u1", _ts_multi_user_key(PLUGIN, "u1"), {"from": "ts-multi"})
    assert PerPluginStore(PLUGIN, "u1").load() == {"from": "ts-multi"}


def test_reads_old_py_single_user_blob_and_migrates_on_save(home: Path) -> None:
    legacy = home / ".telegram-mcp"
    legacy_key = _write_single_user(legacy, {"from": "old-py"})
    store = PerPluginStore(PLUGIN)

    assert store.load() == {"from": "old-py"}
    assert store.has_stored_blob()
    assert store.blob_locations() == [legacy / "config.json"]

    store.save({"from": "old-py", "migrated": True})
    current = home / ".telegram"
    assert (current / "config.json").exists()
    new_key = (current / ".secret").read_bytes()
    assert new_key != legacy_key, "the current dir gets its own machine key"
    assert (legacy / ".secret").read_bytes() == legacy_key, "legacy dirs are never written"
    assert (legacy / "config.json").exists(), "legacy copy is kept until clear()"
    assert PerPluginStore(PLUGIN).load() == {"from": "old-py", "migrated": True}
    assert store.blob_locations() == [current / "config.json", legacy / "config.json"]


def test_reads_old_py_multi_user_blob_pbkdf2(home: Path, master: str) -> None:
    _write_multi_user(home / ".telegram-mcp", "u1", _old_py_multi_user_key(PLUGIN, "u1"), {"from": "old-py-multi"})
    store = PerPluginStore(PLUGIN, "u1")
    assert store.load() == {"from": "old-py-multi"}
    store.save({"v": 2})
    # The migrated copy uses the current (scrypt) derivation.
    blob = (home / ".telegram" / "subs" / "u1" / "config.json").read_bytes()
    assert json.loads(AESGCM(_ts_multi_user_key(PLUGIN, "u1")).decrypt(blob[:12], blob[12:], None)) == {"v": 2}


def test_reads_better_prefixed_legacy_dir_with_its_own_key_plugin(home: Path, master: str) -> None:
    legacy = home / ".better-telegram-mcp"
    _write_single_user(legacy, {"from": "better-single"})
    _write_multi_user(legacy, "u1", _old_py_multi_user_key("better-telegram", "u1"), {"from": "better-multi"})
    assert PerPluginStore(PLUGIN).load() == {"from": "better-single"}
    assert PerPluginStore(PLUGIN, "u1").load() == {"from": "better-multi"}


def test_legacy_precedence_newest_name_first(home: Path) -> None:
    _write_single_user(home / ".telegram-mcp", {"from": "mcp"})
    _write_single_user(home / ".better-telegram-mcp", {"from": "better"})
    assert PerPluginStore(PLUGIN).load() == {"from": "mcp"}
    _write_single_user(home / ".telegram", {"from": "current"})
    assert PerPluginStore(PLUGIN).load() == {"from": "current"}


def test_corrupt_current_blob_does_not_resurrect_legacy(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_single_user(home / ".telegram-mcp", {"from": "legacy"})
    PerPluginStore(PLUGIN).save({"from": "current"})
    (home / ".telegram" / "config.json").write_bytes(b"\x00" * 40)

    assert PerPluginStore(PLUGIN).load() is None
    err = capsys.readouterr().err
    assert "telegram/config" in err
    assert "corrupt" in err
    assert "legacy dir" not in err


# --- corrupt / undecryptable ------------------------------------------------


def test_short_blob_is_corrupt(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = PerPluginStore(PLUGIN, backend=InMemoryBackend())
    store.backend.put(store.cred_key, b"\x00" * 28)
    assert store.load() is None
    assert "corrupt" in capsys.readouterr().err


def test_changed_machine_key_is_corrupt(home: Path) -> None:
    PerPluginStore(PLUGIN).save({"a": 1})
    (home / ".telegram" / ".secret").write_bytes(secrets.token_bytes(32))
    assert PerPluginStore(PLUGIN).load() is None


def test_non_dict_payload_is_corrupt(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_single_user(home / ".telegram", ["not", "a", "dict"])
    assert PerPluginStore(PLUGIN).load() is None
    assert "corrupt" in capsys.readouterr().err


def test_non_json_plaintext_is_corrupt(home: Path) -> None:
    directory = home / ".telegram"
    directory.mkdir()
    key = secrets.token_bytes(32)
    (directory / ".secret").write_bytes(key)
    nonce = secrets.token_bytes(12)
    (directory / "config.json").write_bytes(nonce + AESGCM(key).encrypt(nonce, b"\xff not json", None))
    assert PerPluginStore(PLUGIN).load() is None


def test_legacy_blob_without_secret_is_corrupt(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    legacy = home / ".telegram-mcp"
    _write_single_user(legacy, {"a": 1})
    (legacy / ".secret").unlink()
    assert PerPluginStore(PLUGIN).load() is None
    err = capsys.readouterr().err
    assert f"legacy dir {legacy}" in err


def test_legacy_secret_of_wrong_length_is_corrupt(home: Path) -> None:
    legacy = home / ".telegram-mcp"
    _write_single_user(legacy, {"a": 1})
    (legacy / ".secret").write_bytes(b"short")
    assert PerPluginStore(PLUGIN).load() is None


def test_legacy_multi_user_with_wrong_master_is_corrupt(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDENTIAL_SECRET", MASTER)
    _write_multi_user(home / ".telegram-mcp", "u1", _old_py_multi_user_key(PLUGIN, "u1"), {"a": 1})
    monkeypatch.setenv("CREDENTIAL_SECRET", "different")
    assert PerPluginStore(PLUGIN, "u1").load() is None


# --- clear / locations -------------------------------------------------------


def test_clear_removes_every_location(home: Path) -> None:
    _write_single_user(home / ".telegram-mcp", {"a": 1})
    _write_single_user(home / ".better-telegram-mcp", {"a": 2})
    store = PerPluginStore(PLUGIN)
    store.save({"a": 3})
    assert len(store.blob_locations()) == 3
    store.clear()
    assert not store.has_stored_blob()
    assert store.blob_locations() == []
    assert store.load() is None
    assert (home / ".telegram" / ".secret").exists(), "clear removes blobs, not the machine key"


# --- non-default backends -----------------------------------------------------


def test_in_memory_backend_skips_legacy_reads(home: Path) -> None:
    _write_single_user(home / ".telegram-mcp", {"from": "legacy"})
    store = PerPluginStore(PLUGIN, backend=InMemoryBackend())
    assert store.load() is None
    assert not store.has_stored_blob()
    assert store.cred_path is None
    store.save({"a": 1})
    assert store.load() == {"a": 1}
    assert store.blob_locations() == []
    store.clear()
    assert store.load() is None
    assert (home / ".telegram-mcp" / "config.json").exists(), "clear on an injected backend leaves disk alone"


def test_custom_base_dir_skips_legacy_reads(home: Path, tmp_path: Path) -> None:
    _write_single_user(home / ".telegram-mcp", {"from": "legacy"})
    store = PerPluginStore(PLUGIN, backend=LocalFsBackend(base_dir=tmp_path / "custom"))
    assert store.load() is None
    store.save({"a": 1})
    assert store.cred_path == tmp_path / "custom" / "config.json"
    assert store.blob_locations() == [tmp_path / "custom" / "config.json"]


def test_backend_defaults_from_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_STORAGE_BACKEND", "memory")
    assert isinstance(PerPluginStore(PLUGIN).backend, InMemoryBackend)
    monkeypatch.setenv("MCP_STORAGE_BACKEND", "cf-kv")
    with pytest.raises(ValueError, match="cf-kv"):
        PerPluginStore(PLUGIN)


# --- permissions ---------------------------------------------------------------


def test_existing_data_dir_tightened_once_per_process(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (home / ".telegram").mkdir()
    monkeypatch.setattr(backends, "_POSIX", True)
    with patch("hull_core.storage.backends.os.chmod") as chmod:
        store = PerPluginStore(PLUGIN, backend=InMemoryBackend())
        store.save({"a": 1})
        store.load()
        store.load()
    tightened = [call for call in chmod.call_args_list if call.args[0] == home / ".telegram"]
    # Once by the read-path tighten, once by atomic_write_bytes creating .secret.
    assert len(tightened) == 2
    assert all(call.args[1] == 0o700 for call in tightened)


def test_missing_data_dir_not_created_by_read(home: Path) -> None:
    store = PerPluginStore(PLUGIN, backend=InMemoryBackend())
    store.backend.put(store.cred_key, b"\x00" * 40)
    with patch("hull_core.storage.per_plugin_store.tighten_dir") as tighten:
        assert store.load() is None
    tighten.assert_not_called()


# --- foreign .secret in the current dir ------------------------------------------
#
# telegram's data-dir migration can move ~/.better-telegram-mcp/.secret (64 hex
# chars, mcp-core credential_store's master secret) into ~/.telegram/. Hull must
# never decrypt with it, regenerate over it, or delete it.


def _foreign_secret(home: Path, content: bytes) -> Path:
    directory = home / ".telegram"
    directory.mkdir(exist_ok=True)
    secret = directory / ".secret"
    secret.write_bytes(content)
    return secret


FOREIGN_SECRETS = {
    "hex-text-64": secrets.token_hex(32).encode("ascii"),
    "hex-text-64-newline": secrets.token_hex(32).encode("ascii") + b"\n",
    "empty": b"",
    "short-31": secrets.token_bytes(31),
    "long-33": secrets.token_bytes(33),
}


@pytest.mark.parametrize("content", list(FOREIGN_SECRETS.values()), ids=list(FOREIGN_SECRETS))
def test_foreign_secret_before_first_save_fails_loudly(home: Path, content: bytes) -> None:
    secret = _foreign_secret(home, content)
    store = PerPluginStore(PLUGIN)
    assert store.load() is None, "no blob yet: load never touches the key"

    with pytest.raises(MachineKeyError) as excinfo:
        store.save({"api_id": 1})

    message = str(excinfo.value)
    assert str(secret) in message
    assert f"found {len(content)}" in message
    assert "will not use, overwrite or delete" in message
    if content.strip():
        assert content.strip().decode("latin-1") not in message, "key material never echoed"
    assert secret.read_bytes() == content, "foreign file left byte-identical"
    assert not (home / ".telegram" / "config.json").exists(), "nothing written under an unusable key"
    assert [p.name for p in (home / ".telegram").iterdir()] == [".secret"], "no tmp files left behind"


@pytest.mark.parametrize("content", list(FOREIGN_SECRETS.values()), ids=list(FOREIGN_SECRETS))
def test_foreign_secret_with_existing_blob_fails_loudly_on_load(home: Path, content: bytes) -> None:
    PerPluginStore(PLUGIN).save({"a": 1})
    blob = (home / ".telegram" / "config.json").read_bytes()
    secret = _foreign_secret(home, content)

    with pytest.raises(MachineKeyError, match="not a hull machine key"):
        PerPluginStore(PLUGIN).load()
    assert secret.read_bytes() == content
    assert (home / ".telegram" / "config.json").read_bytes() == blob, "blob untouched"


def test_foreign_secret_error_is_runtime_error_subclass() -> None:
    assert issubclass(MachineKeyError, RuntimeError)


def test_foreign_32_byte_secret_is_used_as_key(home: Path) -> None:
    # 32 raw bytes cannot be told apart from a hull key: documented as accepted.
    key = secrets.token_bytes(32)
    secret = _foreign_secret(home, key)
    PerPluginStore(PLUGIN).save({"a": 1})
    assert secret.read_bytes() == key
    blob = (home / ".telegram" / "config.json").read_bytes()
    assert json.loads(AESGCM(key).decrypt(blob[:12], blob[12:], None)) == {"a": 1}


def test_multi_user_ignores_foreign_secret(home: Path, master: str) -> None:
    secret = _foreign_secret(home, b"")
    store = PerPluginStore(PLUGIN, "u1")
    store.save({"a": 1})
    assert store.load() == {"a": 1}
    assert secret.read_bytes() == b""


# --- machine key creation (create-if-absent) ---------------------------------------


def test_create_machine_key_race_keeps_winner(home: Path) -> None:
    directory = home / ".telegram"
    directory.mkdir()
    secret = directory / ".secret"
    winner = secrets.token_bytes(32)
    real_link = per_plugin_store.os.link

    def racing_link(src, dst):  # noqa: ANN001
        Path(dst).write_bytes(winner)  # another process got there first
        real_link(src, dst)

    with patch.object(per_plugin_store.os, "link", racing_link):
        assert per_plugin_store._create_machine_key(secret) == winner
    assert secret.read_bytes() == winner
    assert [p.name for p in directory.iterdir()] == [".secret"]


def test_create_machine_key_race_with_foreign_winner_fails_loudly(home: Path) -> None:
    directory = home / ".telegram"
    directory.mkdir()
    secret = directory / ".secret"

    def racing_link(src, dst):  # noqa: ANN001
        Path(dst).write_bytes(b"deadbeef")
        raise FileExistsError(dst)

    with patch.object(per_plugin_store.os, "link", racing_link), pytest.raises(MachineKeyError):
        per_plugin_store._create_machine_key(secret)
    assert secret.read_bytes() == b"deadbeef"
    assert [p.name for p in directory.iterdir()] == [".secret"]


def test_create_machine_key_falls_back_to_exclusive_create(home: Path) -> None:
    directory = home / ".telegram"
    directory.mkdir()
    secret = directory / ".secret"

    def no_links(src, dst):  # noqa: ANN001
        raise OSError("hard links not supported")

    with patch.object(per_plugin_store.os, "link", no_links):
        key = per_plugin_store._create_machine_key(secret)
    assert len(key) == 32
    assert secret.read_bytes() == key
    assert [p.name for p in directory.iterdir()] == [".secret"]


def test_create_machine_key_exclusive_create_race_keeps_winner(home: Path) -> None:
    directory = home / ".telegram"
    directory.mkdir()
    secret = directory / ".secret"
    winner = secrets.token_bytes(32)

    def no_links(src, dst):  # noqa: ANN001
        Path(dst).write_bytes(winner)  # appears between link failure and O_EXCL
        raise OSError("hard links not supported")

    with patch.object(per_plugin_store.os, "link", no_links):
        assert per_plugin_store._create_machine_key(secret) == winner
    assert secret.read_bytes() == winner
    assert [p.name for p in directory.iterdir()] == [".secret"]


def test_create_machine_key_cleans_tmp_on_unexpected_error(home: Path) -> None:
    directory = home / ".telegram"
    directory.mkdir()

    def boom(src, dst):  # noqa: ANN001
        raise FileExistsError(dst)

    # FileExistsError but the target vanished before the re-read: the error
    # propagates and the temp file is still removed.
    with patch.object(per_plugin_store.os, "link", boom), pytest.raises(FileNotFoundError):
        per_plugin_store._create_machine_key(directory / ".secret")
    assert list(directory.iterdir()) == []
