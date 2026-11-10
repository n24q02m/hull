"""Account registry: named accounts, labels, per-account pools, removal rules."""

from __future__ import annotations

import pytest

from helpers import ProviderFailure
from hull_core.credentials.models import AccountError
from hull_core.credentials.store import CredentialStore

BASE_URL = "https://openrouter.ai/api/v1"


def test_add_derives_a_stable_account_id(store: CredentialStore) -> None:
    registry = store.accounts()
    account = registry.add(provider="openrouter", name="Work Laptop", email="work@example.com")
    assert account.account_id == "openrouter:work-laptop"
    assert registry.get("openrouter:work-laptop") == account
    assert registry.get("Work Laptop") == account
    assert [item.account_id for item in registry.list_accounts()] == [account.account_id]


def test_explicit_account_id_and_labels_round_trip(store: CredentialStore) -> None:
    registry = store.accounts()
    account = registry.add(
        provider="openrouter",
        name="Team",
        account_id="team-eu",
        workspace="eu-west",
        labels={"tier": "pro", "owner": "ops"},
    )
    assert account.account_id == "team-eu"
    loaded = registry.get("team-eu")
    assert loaded.labels == {"tier": "pro", "owner": "ops"}
    assert loaded.workspace == "eu-west"


def test_duplicate_name_per_provider_is_refused(store: CredentialStore) -> None:
    registry = store.accounts()
    registry.add(provider="openrouter", name="Work")
    with pytest.raises(AccountError, match="already exists"):
        registry.add(provider="openrouter", name="Work")
    # The same name under another provider is fine.
    assert registry.add(provider="openai", name="Work").provider == "openai"


def test_name_lookup_is_ambiguous_across_providers(store: CredentialStore) -> None:
    registry = store.accounts()
    registry.add(provider="openrouter", name="Work")
    registry.add(provider="openai", name="Work")
    with pytest.raises(AccountError, match="ambiguous"):
        registry.get("Work")
    with pytest.raises(AccountError, match="unknown account"):
        registry.get("nobody")
    assert registry.maybe_get("nobody") is None


def test_update_is_partial_and_can_clear_fields(store: CredentialStore) -> None:
    registry = store.accounts()
    account = registry.add(provider="openrouter", name="Work", email="a@b.c", labels={"tier": "free"})
    updated = registry.update(account.account_id, name="Work EU", labels={"tier": "pro"})
    assert updated.name == "Work EU"
    assert updated.email == "a@b.c"  # untouched
    assert updated.labels == {"tier": "pro"}
    cleared = registry.update(account.account_id, email=None, labels=None)
    assert cleared.email is None
    assert cleared.labels == {}
    assert cleared.workspace is None


def test_account_credentials_are_isolated_from_the_default_pool(store: CredentialStore) -> None:
    registry = store.accounts()
    work = registry.add(provider="openrouter", name="Work")
    personal = registry.add(provider="openrouter", name="Personal")
    work_credential = store.add_credential(
        "openrouter", "sk-live-work-0001", base_url=BASE_URL, account_id=work.account_id
    )
    home_credential = store.add_credential(
        "openrouter", "sk-live-home-0002", base_url=BASE_URL, account_id=personal.account_id
    )
    assert [item.account_id for item in registry.credentials(work.account_id)] == [work.account_id]
    assert len(registry.pool(work.account_id).credentials()) == 1
    # A pool scoped to one account never sees another account's key.
    used = registry.pool(work.account_id).run(lambda credential: credential)
    assert used.id == work_credential.id
    assert used.id != home_credential.id
    assert used.account_id == work.account_id


def test_remove_requires_explicit_credential_drop(store: CredentialStore) -> None:
    registry = store.accounts()
    account = registry.add(provider="openrouter", name="Temp")
    store.add_credential("openrouter", "sk-live-temp-0001", base_url=BASE_URL, account_id=account.account_id)
    with pytest.raises(AccountError, match="still has 1 credential"):
        registry.remove(account.account_id)
    registry.remove(account.account_id, drop_credentials=True)
    assert registry.maybe_get(account.account_id) is None
    assert store.pool_credentials("openrouter", account.account_id) == []


def test_registry_pool_rotates_within_the_account(store: CredentialStore) -> None:
    registry = store.accounts()
    work = registry.add(provider="openrouter", name="Work")
    first = store.add_credential("openrouter", "sk-live-work-0001", base_url=BASE_URL, account_id=work.account_id)
    second = store.add_credential("openrouter", "sk-live-work-0002", base_url=BASE_URL, account_id=work.account_id)
    seen: list[int] = []

    def call(credential):
        seen.append(credential.id)
        if credential.id == first.id:
            raise ProviderFailure(401, "revoked")
        return "ok"

    assert registry.pool(work.account_id).run(call) == "ok"
    assert seen == [first.id, second.id]
