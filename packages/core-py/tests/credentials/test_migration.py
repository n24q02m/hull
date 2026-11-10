"""Seeding the pool from the pre-pool single-key config, idempotently."""

from __future__ import annotations

import pytest

from hull_core.config.models import ModelCell
from hull_core.credentials.models import AccountError
from hull_core.credentials.store import CredentialStore

SECRET = "sk-live-migrate-" + "0001"


def _cell(task: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1", **kwargs) -> ModelCell:
    return ModelCell(task=task, base_url=base_url, api_key=api_key, model="model-x", **kwargs)


def test_single_key_config_becomes_one_pool_entry(store: CredentialStore) -> None:
    cells = {task: _cell(task, SECRET) for task in ("embed", "rerank", "chat", "jev_score")}
    result = store.migrate_model_cells(cells)
    assert (result.provider, result.added, result.skipped) == ("openrouter", 1, 3)
    pool = store.pool("openrouter")
    assert len(pool.credentials()) == 1
    assert pool.credentials()[0].api_key == SECRET
    # Re-running on every boot is a no-op, not a duplicate.
    again = store.migrate_model_cells(cells)
    assert (again.added, again.skipped) == (0, 4)
    assert len(store.pool_credentials("openrouter")) == 1


def test_distinct_keys_become_ordered_entries(store: CredentialStore) -> None:
    cells = [
        _cell("chat", "key-a-0001"),
        _cell("chat", "key-b-0002", base_url="https://openrouter.ai/api/v2"),
        _cell("chat", "key-c-0003"),
    ]
    result = store.migrate_model_cells(cells)
    assert (result.added, result.skipped) == (3, 0)
    assert [item.api_key for item in store.pool_credentials("openrouter")] == [
        "key-a-0001",
        "key-b-0002",
        "key-c-0003",
    ]


def test_empty_config_is_a_no_op(store: CredentialStore) -> None:
    result = store.migrate_model_config({"embed": {"base_url": "https://openrouter.ai/api/v1"}}, env={})
    assert (result.provider, result.added, result.skipped) == ("openrouter", 0, 0)
    assert store.pool_credentials("openrouter") == []


def test_env_key_override_is_migrated(store: CredentialStore) -> None:
    result = store.migrate_model_config(
        {"chat": {"base_url": "https://openrouter.ai/api/v1", "api_key": "file-key-0001"}},
        env={"HULL_CHAT_API_KEY": "env-key-0002"},
    )
    assert result.added == 1
    assert [item.api_key for item in store.pool_credentials("openrouter")] == ["env-key-0002"]


def test_cells_spanning_providers_need_an_explicit_provider(store: CredentialStore) -> None:
    cells = [_cell("chat", "key-a-0001"), _cell("chat", "key-b-0002", base_url="https://api.openai.com/v1")]
    with pytest.raises(ValueError, match="span providers"):
        store.migrate_model_cells(cells)
    result = store.migrate_model_cells(cells, provider="gateway")
    assert (result.provider, result.added) == ("gateway", 2)


def test_migration_into_an_account_requires_the_account(store: CredentialStore) -> None:
    cell = _cell("chat", SECRET, account="work")
    with pytest.raises(AccountError, match="unknown account"):
        store.migrate_model_cells([cell])
    account = store.accounts().add(provider="openrouter", name="Work")
    cell = _cell("chat", SECRET, account=account.account_id)
    result = store.migrate_model_cells([cell])
    assert result.added == 1
    assert store.pool_credentials("openrouter") == []
    assert len(store.pool_credentials("openrouter", account.account_id)) == 1


def test_explicit_account_id_wins_over_the_cell(store: CredentialStore) -> None:
    account = store.accounts().add(provider="openrouter", name="Team")
    result = store.migrate_model_cells([_cell("chat", SECRET)], account_id=account.account_id)
    assert result.added == 1
    assert len(store.pool_credentials("openrouter", account.account_id)) == 1


def test_cells_without_keys_are_ignored_on_a_spanning_config(store: CredentialStore) -> None:
    cells = [
        _cell("chat", "key-a-0001"),
        ModelCell(task="embed", base_url="https://api.example.com/v1", api_key="", model="m"),
    ]
    # The keyless cell does not decide the provider: this is a single-provider
    # migration even though the empty cell's host differs.
    result = store.migrate_model_cells(cells)
    assert (result.provider, result.added) == ("openrouter", 1)
    assert store.pool_credentials("openrouter")[0].api_key == "key-a-0001"
