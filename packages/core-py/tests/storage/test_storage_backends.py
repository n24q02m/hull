"""Credential blob backends: key mapping, path safety, atomic writes, env selection."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hull_core.storage import backends
from hull_core.storage.backends import (
    CredentialBackend,
    InMemoryBackend,
    LocalFsBackend,
    atomic_write_bytes,
    backend_from_env,
    tighten_dir,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


# --- InMemoryBackend ---------------------------------------------------------


def test_in_memory_round_trip_and_copy() -> None:
    backend = InMemoryBackend()
    blob = bytearray(b"abc")
    backend.put("p/config", blob)  # ty: ignore[invalid-argument-type]
    blob[0] = ord("z")
    assert backend.get("p/config") == b"abc", "put must store an independent bytes copy"
    backend.delete("p/config")
    assert backend.get("p/config") is None
    backend.delete("p/config")  # idempotent


def test_backends_satisfy_protocol() -> None:
    assert isinstance(InMemoryBackend(), CredentialBackend)
    assert isinstance(LocalFsBackend(), CredentialBackend)


# --- key -> path mapping -----------------------------------------------------


@pytest.mark.parametrize(
    ("key", "relative"),
    [
        ("tg/config", Path("config.json")),
        ("tg/session", Path("session.json")),
        ("tg/tokens/google", Path("tokens", "google.json")),
        ("tg/subs/u1/config", Path("subs", "u1", "config.json")),
        ("tg/subs/u1/session", Path("subs", "u1", "session.json")),
        ("tg/subs/u1/tokens/ms", Path("subs", "u1", "tokens", "ms.json")),
    ],
)
def test_key_mapping_default_layout(home: Path, key: str, relative: Path) -> None:
    assert LocalFsBackend().path_for(key) == home / ".tg" / relative


@pytest.mark.parametrize(
    "key",
    [
        "tg",
        "tg/",
        "tg/other",
        "tg/config/extra",
        "tg/tokens",
        "tg/tokens/a/b",
        "tg/subs/u1",
        "tg/subs/u1/other",
        "/config",
        "../config",
    ],
)
def test_invalid_keys_raise(home: Path, key: str) -> None:
    with pytest.raises(ValueError):
        LocalFsBackend().path_for(key)


@pytest.mark.parametrize("key", ["tg/subs/../config", "tg/subs/./config", "tg/tokens/..", "tg/tokens/a\\b"])
def test_traversal_components_rejected(home: Path, key: str) -> None:
    with pytest.raises(ValueError):
        LocalFsBackend().path_for(key)


def test_plugin_with_traversal_rejected_by_default_layout(home: Path) -> None:
    with pytest.raises(ValueError):
        LocalFsBackend().path_for("..x/config")


def test_fixed_and_callable_base_dir(tmp_path: Path) -> None:
    fixed = LocalFsBackend(base_dir=tmp_path / "fixed")
    assert not fixed.is_default_layout
    assert fixed.path_for("tg/config") == tmp_path / "fixed" / "config.json"

    calls: list[int] = []

    def base() -> Path:
        calls.append(1)
        return tmp_path / "dyn"

    dynamic = LocalFsBackend(base_dir=base)
    assert dynamic.path_for("tg/session") == tmp_path / "dyn" / "session.json"
    assert dynamic.path_for("tg/config") == tmp_path / "dyn" / "config.json"
    assert len(calls) == 2, "a callable base is resolved per operation"
    assert LocalFsBackend().is_default_layout


def test_path_escaping_base_via_symlinked_resolution_rejected(tmp_path: Path) -> None:
    backend = LocalFsBackend(base_dir=tmp_path / "base")
    real_resolve = Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self.name == "config.json":
            return tmp_path / "elsewhere" / "config.json"
        return real_resolve(self, strict=strict)

    with patch.object(Path, "resolve", fake_resolve), pytest.raises(ValueError, match="Path escapes base"):
        backend.path_for("tg/config")


def test_local_fs_round_trip(tmp_path: Path) -> None:
    backend = LocalFsBackend(base_dir=tmp_path / "b")
    assert backend.get("tg/subs/u1/config") is None
    backend.put("tg/subs/u1/config", b"\x00blob")
    assert backend.get("tg/subs/u1/config") == b"\x00blob"
    assert (tmp_path / "b" / "subs" / "u1" / "config.json").read_bytes() == b"\x00blob"
    backend.delete("tg/subs/u1/config")
    assert backend.get("tg/subs/u1/config") is None
    backend.delete("tg/subs/u1/config")  # missing is fine


# --- atomic write + permissions ---------------------------------------------


def test_atomic_write_creates_parents_and_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "config.json"
    atomic_write_bytes(target, b"one")
    atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert list(target.parent.iterdir()) == [target]


def test_atomic_write_cleans_tmp_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    with patch("hull_core.storage.backends.os.replace", side_effect=OSError("boom")), pytest.raises(OSError):
        atomic_write_bytes(target, b"data")
    assert not target.exists()
    assert not (tmp_path / "config.json.tmp").exists()


def test_atomic_write_posix_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_POSIX", True)
    opened: list[int] = []
    real_open = os.open

    def spy_open(path, flags, mode=0o777):  # noqa: ANN001
        opened.append(mode)
        return real_open(path, flags, mode)

    target = tmp_path / "d" / "config.json"
    with patch("hull_core.storage.backends.os.chmod") as chmod, patch("hull_core.storage.backends.os.open", spy_open):
        atomic_write_bytes(target, b"x")
    chmod.assert_called_once_with(target.parent, 0o700)
    assert opened == [0o600]
    assert target.read_bytes() == b"x"


def test_tighten_dir_noop_on_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_POSIX", False)
    with patch("hull_core.storage.backends.os.chmod") as chmod:
        tighten_dir(tmp_path)
    chmod.assert_not_called()


def test_tighten_dir_chmods_on_posix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_POSIX", True)
    with patch("hull_core.storage.backends.os.chmod") as chmod:
        tighten_dir(tmp_path)
    chmod.assert_called_once_with(tmp_path, 0o700)


# --- backend_from_env -------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "local", "LOCAL", " local-fs "])
def test_backend_from_env_local(value: str | None) -> None:
    env = {} if value is None else {"MCP_STORAGE_BACKEND": value}
    backend = backend_from_env(env)
    assert isinstance(backend, LocalFsBackend)
    assert backend.is_default_layout


@pytest.mark.parametrize("value", ["memory", "In-Memory"])
def test_backend_from_env_memory(value: str) -> None:
    assert isinstance(backend_from_env({"MCP_STORAGE_BACKEND": value}), InMemoryBackend)


def test_backend_from_env_cf_kv_is_removed() -> None:
    with pytest.raises(ValueError, match="cf-kv is no longer supported"):
        backend_from_env({"MCP_STORAGE_BACKEND": "cf-kv"})


def test_backend_from_env_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown MCP_STORAGE_BACKEND: 's3'"):
        backend_from_env({"MCP_STORAGE_BACKEND": "s3"})


def test_backend_from_env_reads_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_STORAGE_BACKEND", "memory")
    assert isinstance(backend_from_env(), InMemoryBackend)
