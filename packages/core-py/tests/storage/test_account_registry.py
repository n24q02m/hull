"""NamedAccountRegistry (Tier-3 per-plugin multi-account registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hull_core.storage.account_registry import NamedAccountRegistry, validate_account_alias
from hull_core.storage.backends import InMemoryBackend


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("N24Q02M_HOME", raising=False)
    monkeypatch.delenv("HULL_PROFILE", raising=False)
    return tmp_path


def test_validate_account_alias() -> None:
    assert validate_account_alias("work-1_a.b") == "work-1_a.b"
    for bad in ("", ".", "..", "a/b", "a..b"):
        with pytest.raises(ValueError, match="Invalid account alias"):
            validate_account_alias(bad)


def test_account_registry_crud_and_active_switching(home: Path) -> None:
    reg = NamedAccountRegistry("notion", backend=InMemoryBackend())
    assert reg.list_accounts() == []
    assert reg.active_account is None
    assert reg.get_account() is None
    assert reg.get_account("work") is None

    reg.put_account("work", {"token": "ntn-work"})
    assert reg.list_accounts() == ["work"]
    assert reg.active_account == "work"
    assert reg.get_account() == {"token": "ntn-work"}
    assert reg.get_account("work") == {"token": "ntn-work"}

    reg.put_account("personal", {"token": "ntn-personal"}, set_active=False)
    assert reg.list_accounts() == ["personal", "work"]
    assert reg.active_account == "work"
    assert reg.get_account("personal") == {"token": "ntn-personal"}

    reg.set_active("personal")
    assert reg.active_account == "personal"
    assert reg.get_account() == {"token": "ntn-personal"}

    with pytest.raises(KeyError, match="Unknown account alias"):
        reg.set_active("missing")

    with pytest.raises(ValueError, match="Account credentials must be a dict"):
        reg.put_account("bad", ["not-a-dict"])  # ty: ignore[invalid-argument-type]

    assert reg.remove_account("missing") is False
    assert reg.remove_account("personal") is True
    assert reg.active_account == "work"
    assert reg.list_accounts() == ["work"]

    assert reg.remove_account("work") is True
    assert reg.list_accounts() == []
    assert reg.active_account is None
    assert not reg.store.has_stored_blob()


def test_account_registry_recovers_when_stored_active_is_invalid() -> None:
    backend = InMemoryBackend()
    reg = NamedAccountRegistry("notion", backend=backend, profile="work")
    reg.store.save({"active": "ghost", "accounts": {"alpha": {"k": "v"}, "bad": "skip"}})
    assert reg.active_account == "alpha"
    assert reg.list_accounts() == ["alpha"]
    assert reg.get_account() == {"k": "v"}
