"""StringSessionStore persistence + the telethon-gated SaveOnChangeStringSession.

telethon is not a hull dependency, so a minimal fake ``telethon.sessions``
module is injected via ``sys.modules`` for the subclass tests, and the
module-level class cache is reset around every test.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import hull_core.storage as storage_pkg
from hull_core.storage import string_session_store
from hull_core.storage.backends import InMemoryBackend
from hull_core.storage.string_session_store import TELETHON_INSTALL_HINT, StringSessionStore


@pytest.fixture(autouse=True)
def reset_cls_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(string_session_store, "_save_on_change_cls", None)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from hull_core.storage import per_plugin_store

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("CREDENTIAL_SECRET", raising=False)
    monkeypatch.delenv("MCP_STORAGE_BACKEND", raising=False)
    monkeypatch.setattr(per_plugin_store, "_tightened_dirs", set())
    return tmp_path


class _FakeStringSession:
    """Shape of telethon.sessions.StringSession that the subclass relies on."""

    def __init__(self, string: str | None = None) -> None:
        self.init_arg = string
        self.state = string or ""

    def save(self) -> str:
        return f"serialized:{self.state}"


@pytest.fixture
def fake_telethon(monkeypatch: pytest.MonkeyPatch) -> type:
    telethon = types.ModuleType("telethon")
    sessions = types.ModuleType("telethon.sessions")
    sessions.StringSession = _FakeStringSession  # ty: ignore[unresolved-attribute]
    telethon.sessions = sessions  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "telethon", telethon)
    monkeypatch.setitem(sys.modules, "telethon.sessions", sessions)
    return _FakeStringSession


@pytest.fixture
def no_telethon(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes `import telethon.sessions` raise ImportError.
    monkeypatch.setitem(sys.modules, "telethon", None)
    monkeypatch.setitem(sys.modules, "telethon.sessions", None)


# --- StringSessionStore -------------------------------------------------------------


def test_round_trip_in_memory() -> None:
    store = StringSessionStore("telegram", backend=InMemoryBackend())
    assert store.load() is None
    store.save("1BVtsOK...")
    assert store.load() == "1BVtsOK..."
    assert store.store.cred_key == "telegram/session"
    store.clear()
    assert store.load() is None


def test_round_trip_on_disk_uses_session_leaf(home: Path) -> None:
    StringSessionStore("telegram").save("abc")
    assert (home / ".telegram" / "session.json").exists()
    assert not (home / ".telegram" / "config.json").exists()
    assert b"abc" not in (home / ".telegram" / "session.json").read_bytes()
    assert StringSessionStore("telegram").load() == "abc"


def test_multi_user_sub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDENTIAL_SECRET", "m")
    backend = InMemoryBackend()
    StringSessionStore("telegram", "u1", backend=backend).save("s1")
    assert StringSessionStore("telegram", "u1", backend=backend).load() == "s1"
    assert StringSessionStore("telegram", "u2", backend=backend).load() is None
    assert StringSessionStore("telegram", "u1", backend=backend).store.cred_key == "telegram/subs/u1/session"


def test_payload_without_session_key_is_none() -> None:
    store = StringSessionStore("telegram", backend=InMemoryBackend())
    store.store.save({"other": "x"})
    assert store.load() is None


def test_undecryptable_blob_is_none(capsys: pytest.CaptureFixture[str]) -> None:
    store = StringSessionStore("telegram", backend=InMemoryBackend())
    store.store.backend.put(store.store.cred_key, b"\x00" * 10)
    assert store.load() is None
    assert "corrupt" in capsys.readouterr().err


# --- SaveOnChangeStringSession --------------------------------------------------------


def test_save_on_change_pushes_to_sink(fake_telethon: type) -> None:
    cls = string_session_store.SaveOnChangeStringSession
    assert issubclass(cls, fake_telethon)
    assert cls.__module__ == "hull_core.storage.string_session_store"

    pushed: list[str] = []
    session = cls("initial", sink=pushed.append)
    assert session.init_arg == "initial"
    assert session.save() == "serialized:initial"
    assert pushed == ["serialized:initial"]


def test_save_on_change_without_session_string(fake_telethon: type) -> None:
    session = string_session_store.SaveOnChangeStringSession()
    assert session.init_arg is None
    assert session.save() == "serialized:", "no sink: save still returns the string"


def test_save_on_change_feeds_store(fake_telethon: type) -> None:
    store = StringSessionStore("telegram", backend=InMemoryBackend())
    session = string_session_store.SaveOnChangeStringSession(None, sink=store.save)
    session.state = "authorized"
    session.save()
    assert store.load() == "serialized:authorized"


def test_class_is_built_once(fake_telethon: type) -> None:
    first = string_session_store.SaveOnChangeStringSession
    assert string_session_store.SaveOnChangeStringSession is first


def test_storage_package_lazy_export(fake_telethon: type) -> None:
    assert storage_pkg.SaveOnChangeStringSession is string_session_store.SaveOnChangeStringSession


def test_missing_telethon_raises_install_hint(no_telethon: None) -> None:
    with pytest.raises(ImportError, match="requires telethon") as excinfo:
        string_session_store.SaveOnChangeStringSession  # noqa: B018
    assert str(excinfo.value) == TELETHON_INSTALL_HINT
    assert isinstance(excinfo.value.__cause__, ImportError)
    assert string_session_store._save_on_change_cls is None, "a failed build is not cached"


def test_missing_telethon_via_storage_package(no_telethon: None) -> None:
    with pytest.raises(ImportError, match="uv add telethon"):
        storage_pkg.SaveOnChangeStringSession  # noqa: B018


def test_import_of_modules_never_requires_telethon(no_telethon: None) -> None:
    # Both modules are already importable with telethon blocked.
    assert StringSessionStore is not None
    assert "StringSessionStore" in storage_pkg.__all__


@pytest.mark.parametrize("module", [string_session_store, storage_pkg])
def test_unknown_attribute_raises_attribute_error(module: types.ModuleType) -> None:
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        module.nope  # noqa: B018
