"""CredentialStore: schema, encryption at rest, CRUD, pools, read API."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hull_core.config.models import ModelCell
from hull_core.credentials.models import (
    AccountError,
    CredentialExistsError,
    CredentialNotFoundError,
    provider_from_base_url,
)
from hull_core.credentials.store import POOL_SCHEMA, CredentialStore, default_pool_db_path

SECRET = "sk-live-store-secret-0001"


def test_schema_is_additive_and_idempotent(db_path: Path, clock) -> None:
    # An older hull database with only `kv` keeps working; opening the pool
    # twice must not fail or drop anything.
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE kv (namespace TEXT, key TEXT, value TEXT)")
    conn.execute("INSERT INTO kv VALUES ('old', 'k', 'v')")
    conn.commit()
    conn.close()
    first = CredentialStore(db_path, clock=clock)
    first.add_credential("openrouter", SECRET, base_url="https://openrouter.ai/api/v1")
    first.close()
    second = CredentialStore(db_path, clock=clock)
    try:
        assert [row.api_key for row in second.credentials()] == [SECRET]
    finally:
        second.close()
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT value FROM kv").fetchone() == ("v",)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {"kv", "provider_accounts", "provider_credentials"} <= tables
    assert "CREATE TABLE IF NOT EXISTS provider_accounts" in POOL_SCHEMA


def test_secret_is_encrypted_at_rest(store: CredentialStore, db_path: Path) -> None:
    credential = store.add_credential("openrouter", SECRET, base_url="https://openrouter.ai/api/v1")
    assert credential.api_key == SECRET
    assert credential.key_hint == "…0001"
    raw = db_path.read_bytes()
    assert SECRET.encode() not in raw
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT secret FROM provider_credentials").fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0].startswith("v1:")
    assert store.get_credential(credential.id).api_key == SECRET


def test_add_refuses_duplicates_and_orders_by_priority(store: CredentialStore) -> None:
    first = store.add_credential("openrouter", "key-one-0001", base_url="https://openrouter.ai/api/v1")
    second = store.add_credential("openrouter", "key-two-0002", base_url="https://openrouter.ai/api/v1", label="backup")
    assert (first.priority, second.priority) == (0, 1)
    assert [item.id for item in store.pool_credentials("openrouter")] == [first.id, second.id]
    with pytest.raises(CredentialExistsError, match="already has this credential"):
        store.add_credential("openrouter", "key-one-0001", base_url="https://openrouter.ai/api/v2")
    store.set_priority(first.id, 9)
    assert [item.id for item in store.pool_credentials("openrouter")] == [second.id, first.id]
    store.remove_credential(second.id)
    with pytest.raises(CredentialNotFoundError):
        store.remove_credential(second.id)
    with pytest.raises(CredentialNotFoundError):
        store.set_priority(second.id, 1)


def test_health_reports_state_and_stays_key_free(store: CredentialStore, clock) -> None:
    credential = store.add_credential("openrouter", SECRET, base_url="https://openrouter.ai/api/v1")
    store.record_failure(credential.id, 500, now=clock.value)
    store.record_failure(credential.id, 503, now=clock.value)
    health = store.health("openrouter")[0]
    assert (health.disabled, health.last_status, health.failure_count) == (False, 503, 2)
    assert health.usable is True
    assert SECRET not in repr(health)
    store.disable(credential.id)
    assert store.health()[0].usable is False
    store.enable(credential.id)
    enabled = store.health()[0]
    assert (enabled.disabled, enabled.last_status, enabled.failure_count) == (False, None, 0)


def test_cooldown_window_expires(store: CredentialStore, clock) -> None:
    credential = store.add_credential("openrouter", SECRET, base_url="https://openrouter.ai/api/v1")
    store.record_failure(credential.id, 429, now=clock.value, cooldown_until=clock.value + 30)
    health = store.pool_health("openrouter")[0]
    assert health.usable is False
    assert health.cooldown_remaining == pytest.approx(30.0)
    assert store.lease_first("openrouter", None, now=clock.value, lease_seconds=60) is None
    clock.advance(31)
    leased = store.lease_first("openrouter", None, now=clock.value, lease_seconds=60)
    assert leased is not None and leased.id == credential.id


def test_diagnose_counts_every_blocking_state(store: CredentialStore, clock) -> None:
    first = store.add_credential("openrouter", "key-a-0001", base_url="https://openrouter.ai/api/v1")
    second = store.add_credential("openrouter", "key-b-0002", base_url="https://openrouter.ai/api/v1")
    store.disable(first.id)
    store.record_failure(second.id, 429, now=clock.value, cooldown_until=clock.value + 10)
    detail = store.diagnose("openrouter")
    assert detail == {
        "total": 2,
        "disabled": 1,
        "cooling_down": 1,
        "leased": 0,
        "retry_after": pytest.approx(10.0),
    }
    assert store.diagnose("missing")["total"] == 0


def test_reset_health_scopes_to_provider(store: CredentialStore, clock) -> None:
    first = store.add_credential("openrouter", "key-a-0001", base_url="https://openrouter.ai/api/v1")
    second = store.add_credential("openai", "key-b-0002", base_url="https://api.openai.com/v1")
    store.disable(first.id)
    store.disable(second.id)
    assert store.reset_health("openrouter") == 1
    assert store.health("openrouter")[0].disabled is False
    assert store.health("openai")[0].disabled is True
    # UPDATE rowcount is rows matched, so the second sweep reports both rows.
    assert store.reset_health() == 2
    assert store.health("openai")[0].disabled is False


def test_account_scoping_requires_matching_provider(store: CredentialStore) -> None:
    account = store.accounts().add(provider="openrouter", name="Work")
    with pytest.raises(AccountError, match="belongs to provider"):
        store.add_credential(
            "openai",
            SECRET,
            base_url="https://api.openai.com/v1",
            account_id=account.account_id,
        )
    with pytest.raises(AccountError, match="unknown account"):
        store.add_credential(
            "openrouter",
            SECRET,
            base_url="https://openrouter.ai/api/v1",
            account_id="missing",
        )
    store.add_credential(
        "openrouter",
        SECRET,
        base_url="https://openrouter.ai/api/v1",
        account_id=account.account_id,
    )
    assert store.pool_credentials("openrouter") == []  # the default pool stays separate
    assert len(store.pool_credentials("openrouter", account.account_id)) == 1
    assert len(store.credentials("openrouter")) == 1


def test_pool_for_cell_uses_provider_and_account(store: CredentialStore) -> None:
    account = store.accounts().add(provider="openrouter", name="Work")
    store.add_credential(
        "openrouter",
        SECRET,
        base_url="https://openrouter.ai/api/v1",
        account_id=account.account_id,
    )
    cell = ModelCell(
        task="chat",
        base_url="https://openrouter.ai/api/v1",
        api_key=SECRET,
        model="m",
        account=account.account_id,
    )
    pool = store.pool_for_cell(cell)
    assert [item.id for item in pool.credentials()] == [
        row.id for row in store.pool_health("openrouter", account.account_id)
    ]
    inferred = store.pool_for_cell(ModelCell(task="chat", base_url="https://api.openai.com/v1", api_key="", model="m"))
    assert inferred.provider == "openai"


def test_default_path_follows_config_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("hull_core.config.settings.default_config_dir", lambda: tmp_path)
    assert default_pool_db_path() == tmp_path / "hull.db"
    store = CredentialStore()
    try:
        assert store.path == tmp_path / "hull.db"
        store.add_credential("openrouter", SECRET, base_url="https://openrouter.ai/api/v1")
        assert (tmp_path / ".credential-key").is_file()
    finally:
        store.close()


def test_provider_inference_is_stable() -> None:
    assert provider_from_base_url("https://openrouter.ai/api/v1") == "openrouter"
    assert provider_from_base_url("https://api.openai.com/v1") == "openai"
    assert provider_from_base_url("https://gateway.example.com/v1") == "gateway-example-com"
    assert provider_from_base_url("http://127.0.0.1:9800") == "127-0-0-1"
    assert provider_from_base_url("not a url") == "default"


def test_credential_repr_and_masked_hide_the_key(store: CredentialStore) -> None:
    credential = store.add_credential("openrouter", SECRET, base_url="https://openrouter.ai/api/v1", label="primary")
    assert SECRET not in repr(credential)
    assert SECRET not in str(credential)
    assert SECRET not in credential.masked
    assert credential.masked.endswith("…0001")
